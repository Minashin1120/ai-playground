# =============================================================================
# True real-time STS sessions (OpenAI Realtime / Grok Voice / Gemini native-audio)
# -----------------------------------------------------------------------------
# Server-held WebSocket sessions.  The browser streams microphone PCM over HTTP,
# the server relays it to the provider WebSocket in real time, and the provider's
# audio + transcripts stream back to the browser over SSE.  Provider-side VAD
# (or Gemini's natural turn handling) drives automatic turn-taking, so the user
# can talk continuously and hear the model reply while speaking.
# =============================================================================
RT_SESSIONS = {}
RT_SESSIONS_LOCK = threading.Lock()
RT_MAX_SESSION_SECONDS = 15 * 60
RT_SESSION_TTL_SECONDS = 30 * 60
RT_AUDIO_POST_MAX = 1 << 20
RT_PCM_CAP = 512 * 1024 * 1024


def _rt_is_conversation_model(model_key):
    """True for STS models that support a persistent streaming conversation."""
    model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
    if not is_sts_model(model_key):
        return False
    meta = STS_MODELS.get(model_key, {})
    if meta.get("mode") == "transcription":
        return False
    # Browser-direct Gemini Live models already stream in real time.
    if model_key in ("gemini-3.1-flash-live-preview", "gemini-3.5-live-translate-preview"):
        return False
    # One-shot transcription session model.
    if model_key == "gpt-realtime-whisper":
        return False
    return True


def _rt_push_event(session, event):
    with session.pending_cond:
        session.pending.append(event)
        session.pending_cond.notify_all()


def _rt_purge_old_sessions():
    now = time.time()
    with RT_SESSIONS_LOCK:
        stale = [
            sid for sid, sess in list(RT_SESSIONS.items())
            if sess.status in ("closed", "error", "stopped")
            and (now - sess.started_at) > RT_SESSION_TTL_SECONDS
        ]
        for sid in stale:
            RT_SESSIONS.pop(sid, None)


def _rt_get_session(session_id):
    session = RT_SESSIONS.get(session_id or "")
    if not session or session.user_id != current_user.id:
        return None
    return session


class RtSession:
    """One persistent real-time speech-to-speech session for a single user."""

    def __init__(self, session_id, user_id, model_key, api_key, params):
        self.session_id = session_id
        self.user_id = user_id
        self.model_key = model_key
        self.provider = get_sts_provider(model_key)
        self.api_key = api_key
        self.params = params
        meta = STS_MODELS.get(model_key, {})
        self.rate_in = int(params.get("rate_in") or meta.get("rate_in", 24000))
        self.rate_out = int(params.get("rate_out") or meta.get("rate_out", 24000))
        self.loop = None                # asyncio event loop of the worker thread
        self.ws = None                  # provider WebSocket (owned by worker thread)
        self.audio_in = _queue.Queue()  # ("audio", bytes) / ("commit",)
        self.pending = []               # output events awaiting the SSE stream
        self.pending_cond = threading.Condition()
        self.cmd_queue = _queue.Queue()  # reserved for future steering commands
        self.stop_event = threading.Event()
        self.status = "connecting"       # connecting|ready|speaking|stopped|error|closed
        self.error = None
        self.started_at = time.time()
        self.thread = None
        self.assistant_audio = bytearray()   # accumulated output PCM (for saving)
        self.assistant_lock = threading.Lock()
        self.user_audio = bytearray()        # accumulated input PCM (for saving)
        self.user_lock = threading.Lock()
        self.user_transcript = ""
        self.assistant_transcript = ""
        self.assistant_thought = ""
        self.speech_active = False
        self.turn_count = 0
        self.saved = False


def _normalize_rt_params(provider, model_key, data):
    """Validate / normalize real-time session parameters (always returns a dict)."""
    data = data or {}
    meta = STS_MODELS.get(model_key, {})
    params = {
        "rate_in": int(data.get("rate_in") or meta.get("rate_in", 24000)),
        "rate_out": int(data.get("rate_out") or meta.get("rate_out", 24000)),
    }
    if provider == "openai":
        v = str(data.get("voice") or "alloy").lower()
        params["voice"] = v if v in OPENAI_STS_VOICES else "alloy"
        speed = clamp_float(data.get("speed"), 0.25, 1.5)
        if speed is not None:
            params["speed"] = speed
    elif provider == "xai":
        v = str(data.get("voice") or "Ara")
        params["voice"] = v if v in XAI_STS_VOICES else "Ara"
    elif provider == "google":
        v = str(data.get("voice") or "Kore")
        params["voice"] = v if v in GEMINI_STS_VOICES else "Kore"
        thinking = str(data.get("thinking_level") or "").strip()
        params["thinking_level"] = thinking or None
        params["include_thoughts"] = bool(data.get("include_thoughts"))
    return params


async def _rt_openai_xai_send_loop(session, ws):
    while not session.stop_event.is_set():
        try:
            item = session.audio_in.get_nowait()
        except _queue.Empty:
            item = None
        if item is None:
            await asyncio.sleep(0.02)
            continue
        kind = item[0]
        try:
            if kind == "audio":
                data = item[1]
                if not data:
                    continue
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode("ascii"),
                }))
            elif kind == "commit":
                # Finalize any trailing audio; with server VAD this is usually
                # automatic, but the explicit commit covers the push-to-talk end.
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                session.status = "speaking"
        except Exception as exc:
            logger.error(f"Realtime STS send error: {exc}")


async def _rt_openai_xai_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "session.updated":
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
            elif mtype == "input_audio_buffer.speech_started":
                session.speech_active = True
                _rt_push_event(session, {"type": "speech_started"})
            elif mtype == "input_audio_buffer.speech_stopped":
                session.speech_active = False
                _rt_push_event(session, {"type": "speech_stopped"})
            elif mtype == "input_audio_buffer.committed":
                _rt_push_event(session, {"type": "committed"})
            elif mtype == "response.created":
                _rt_push_event(session, {"type": "response_start"})
            elif mtype in ("response.output_audio.delta", "response.audio.delta"):
                delta = msg.get("delta")
                if delta:
                    try:
                        binary = base64.b64decode(delta)
                    except Exception:
                        binary = b""
                    if binary:
                        with session.assistant_lock:
                            if len(session.assistant_audio) + len(binary) <= RT_PCM_CAP:
                                session.assistant_audio += binary
                        _rt_push_event(session, {"type": "audio", "data": delta})
            elif mtype == "response.output_audio_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    session.assistant_transcript += delta
                    _rt_push_event(session, {"type": "transcript", "role": "assistant", "delta": delta})
            elif mtype == "conversation.item.input_audio_transcription.delta":
                delta = msg.get("delta")
                if delta:
                    session.user_transcript += delta
                    _rt_push_event(session, {"type": "transcript", "role": "user", "delta": delta})
            elif mtype == "conversation.item.input_audio_transcription.updated":
                # xAI emits a cumulative transcript here.
                text = str(msg.get("transcript") or "")
                if text:
                    session.user_transcript = text
                    _rt_push_event(session, {"type": "transcript", "role": "user", "delta": text, "cumulative": True})
            elif mtype == "conversation.item.input_audio_transcription.completed":
                # OpenAI emits the full committed transcript here.
                text = str(msg.get("transcript") or "")
                if text:
                    session.user_transcript = text
                    _rt_push_event(session, {"type": "transcript", "role": "user", "delta": text, "cumulative": True})
            elif mtype == "response.done":
                session.turn_count += 1
                _rt_push_event(session, {"type": "response_done"})
            elif mtype == "error":
                raise RuntimeError(str(msg.get("error") or "Provider error"))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        _rt_push_event(session, {"type": "error", "message": str(exc)})
        with session.pending_cond:
            session.pending_cond.notify_all()


async def _rt_openai_xai_session_async(session):
    model_key = session.model_key
    if session.provider == "xai":
        model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
        url = f"wss://{_XAI_API_HOST}/v1/realtime?model={model_key}"
        headers = {"Authorization": f"Bearer {session.api_key}"}
    else:
        if model_key == "gpt-realtime-translate":
            url = f"wss://api.openai.com/v1/realtime/translations?model={model_key}"
        else:
            url = f"wss://api.openai.com/v1/realtime?model={model_key}"
        headers = {
            "Authorization": f"Bearer {session.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        session.ws = ws
        voice = session.params.get("voice") or ("alloy" if session.provider == "openai" else "Ara")
        audio_cfg = {
            "input": {"format": {"type": "audio/pcm", "rate": session.rate_in}},
            "output": {"format": {"type": "audio/pcm", "rate": session.rate_out}},
        }
        if session.provider == "xai":
            # xAI is OpenAI-Realtime compatible but uses a top-level turn_detection.
            sess = {
                "voice": voice,
                "turn_detection": {"type": "server_vad"},
                "audio": audio_cfg,
            }
        else:
            sess = {
                "type": "realtime",
                "model": model_key,
                "output_modalities": ["audio"],
                "voice": voice,
                "audio": audio_cfg,
            }
            sess["audio"]["input"]["turn_detection"] = {"type": "server_vad"}
            speed = session.params.get("speed")
            if speed is not None:
                sess["speed"] = speed
        await ws.send(json.dumps({"type": "session.update", "session": sess}))

        # Wait for the session to be ready before streaming audio.
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("type") == "session.updated":
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
                break
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("error") or "Session setup failed"))

        recv_task = asyncio.ensure_future(_rt_openai_xai_receive_loop(session, ws))
        send_task = asyncio.ensure_future(_rt_openai_xai_send_loop(session, ws))
        while not session.stop_event.is_set():
            if time.time() - session.started_at > RT_MAX_SESSION_SECONDS:
                session.error = "最大セッション時間（15分）に達したため自動停止しました。"
                session.status = "stopped"
                session.stop_event.set()
                break
            if recv_task.done():
                break
            await asyncio.sleep(0.05)
        recv_task.cancel()
        send_task.cancel()
        try:
            await recv_task
        except Exception:
            pass
        try:
            await send_task
        except Exception:
            pass


async def _rt_gemini_send_loop(session, ws):
    while not session.stop_event.is_set():
        try:
            item = session.audio_in.get_nowait()
        except _queue.Empty:
            item = None
        if item is None:
            await asyncio.sleep(0.02)
            continue
        kind = item[0]
        if kind == "audio":
            data = item[1]
            if not data:
                continue
            try:
                await ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(data).decode("ascii"),
                            "mimeType": f"audio/pcm;rate={session.rate_in}",
                        }
                    }
                }))
            except Exception as exc:
                logger.error(f"Realtime STS Gemini send error: {exc}")


async def _rt_gemini_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("setupComplete") is not None:
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
            sc = msg.get("serverContent")
            if sc is not None:
                model_turn = sc.get("modelTurn")
                if model_turn:
                    for part in model_turn.get("parts") or []:
                        inline = part.get("inlineData") or {}
                        audio_b64 = inline.get("data")
                        if audio_b64:
                            try:
                                binary = base64.b64decode(audio_b64)
                            except Exception:
                                binary = b""
                            if binary:
                                with session.assistant_lock:
                                    if len(session.assistant_audio) + len(binary) <= RT_PCM_CAP:
                                        session.assistant_audio += binary
                                _rt_push_event(session, {"type": "audio", "data": audio_b64})
                        text = part.get("text")
                        if text:
                            if part.get("thought"):
                                session.assistant_thought += text
                                _rt_push_event(session, {"type": "transcript", "role": "thought", "delta": text})
                            else:
                                session.assistant_transcript += text
                                _rt_push_event(session, {"type": "transcript", "role": "assistant", "delta": text})
                out_tr = sc.get("outputTranscription") or {}
                if out_tr.get("text"):
                    text = out_tr["text"]
                    session.assistant_transcript += text
                    _rt_push_event(session, {"type": "transcript", "role": "assistant", "delta": text})
                in_tr = sc.get("inputTranscription") or {}
                if in_tr.get("text"):
                    text = in_tr["text"]
                    session.user_transcript += text
                    _rt_push_event(session, {"type": "transcript", "role": "user", "delta": text})
                if sc.get("interrupted"):
                    _rt_push_event(session, {"type": "interrupted"})
                if sc.get("turnComplete"):
                    session.turn_count += 1
                    _rt_push_event(session, {"type": "turn_complete"})
            if msg.get("error"):
                raise RuntimeError(str(msg.get("error")))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        _rt_push_event(session, {"type": "error", "message": str(exc)})
        with session.pending_cond:
            session.pending_cond.notify_all()


async def _rt_gemini_session_async(session):
    ws_url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={quote(session.api_key, safe='')}"
    )
    async with websockets.connect(ws_url, max_size=None) as ws:
        session.ws = ws
        setup = {
            "setup": {
                "model": f"models/{session.model_key}",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
        }
        voice = session.params.get("voice")
        if voice and voice in GEMINI_STS_VOICES:
            setup["setup"]["generationConfig"]["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            }
        thinking_level = session.params.get("thinking_level")
        if thinking_level:
            setup["setup"]["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": thinking_level,
                "includeThoughts": bool(session.params.get("include_thoughts")),
            }
        await ws.send(json.dumps(setup))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("setupComplete") is not None:
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
                break
            if msg.get("error"):
                raise RuntimeError(str(msg.get("error")))

        recv_task = asyncio.ensure_future(_rt_gemini_receive_loop(session, ws))
        send_task = asyncio.ensure_future(_rt_gemini_send_loop(session, ws))
        while not session.stop_event.is_set():
            if time.time() - session.started_at > RT_MAX_SESSION_SECONDS:
                session.error = "最大セッション時間（15分）に達したため自動停止しました。"
                session.status = "stopped"
                session.stop_event.set()
                break
            if recv_task.done():
                break
            await asyncio.sleep(0.05)
        recv_task.cancel()
        send_task.cancel()
        try:
            await recv_task
        except Exception:
            pass
        try:
            await send_task
        except Exception:
            pass


def _rt_worker(session):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session.loop = loop
    try:
        if session.provider == "google":
            loop.run_until_complete(_rt_gemini_session_async(session))
        else:
            loop.run_until_complete(_rt_openai_xai_session_async(session))
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        _rt_push_event(session, {"type": "error", "message": str(exc)})
        logger.exception("Realtime STS session error")
    finally:
        if session.status not in ("error", "stopped", "paused"):
            session.status = "closed"
        session.stop_event.set()
        with session.pending_cond:
            session.pending_cond.notify_all()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _rt_resolve_api_key(current_user_obj, model_key, provider):
    model_specific_key = _get_model_specific_api_key(current_user_obj, model_key)
    if provider == "google":
        runtime = _resolve_gemini_runtime(current_user_obj)
        return (model_specific_key or runtime.get("api_key")), None
    if provider == "openai":
        key = model_specific_key or decrypt_val(current_user_obj.openai_api_key)
        if not key and _admin_env_fallback_enabled(current_user_obj):
            key = os.getenv("OPENAI_API_KEY")
        return key, None
    if provider == "xai":
        key = model_specific_key or decrypt_val(current_user_obj.xai_api_key)
        if not key and _admin_env_fallback_enabled(current_user_obj):
            key = os.getenv("XAI_API_KEY")
        return key, None
    return None, None

async def _google_sts_live(
    pcm_bytes,
    model_key,
    gemini_api_key=None,
    gemini_backend="gemini_api",
    gemini_vertex_project=None,
    gemini_vertex_location=None,
    gemini_vertex_credentials_json=None,
    rate=16000,
    voice="Kore",
    thinking_level=None,
    include_thoughts=False,
):
    client = _get_gemini_client(
        api_key=gemini_api_key,
        backend=gemini_backend,
        vertex_project=gemini_vertex_project,
        vertex_location=gemini_vertex_location,
        vertex_credentials_json=gemini_vertex_credentials_json,
    )
    if not client:
        raise ValueError("Gemini client not configured")

    live_conf = {"response_modalities": ["AUDIO"]}
    if voice and voice in GEMINI_STS_VOICES:
        live_conf["speech_config"] = {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": voice}
            }
        }
    if thinking_level:
        live_conf["thinking_config"] = {
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts
        }

    async with client.aio.live.connect(
        model=model_key,
        config=live_conf,
    ) as session:
        # Send audio in small chunks to the Live API
        for chunk in _chunk_bytes(pcm_bytes, 4096):
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={rate}")
            )
        await session.send_realtime_input(audio_stream_end=True)
        
        total_audio_len = 0
        async for msg in session.receive():
            if total_audio_len > 10 * 1024 * 1024:
                break

            chunk_audio = bytearray()
            chunk_transcript = ""
            chunk_thought = ""
            chunk_input_transcript = ""
            turn_complete = False

            sc = getattr(msg, "server_content", None)
            if sc:
                model_turn = getattr(sc, "model_turn", None)
                if model_turn:
                    for part in model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            chunk_audio.extend(part.inline_data.data)
                        if part.text:
                            if getattr(part, "thought", False):
                                chunk_thought += part.text
                            else:
                                chunk_transcript += part.text

                if getattr(sc, "output_transcription", None) and sc.output_transcription.text:
                    chunk_transcript += sc.output_transcription.text
                if getattr(sc, "input_transcription", None) and sc.input_transcription.text:
                    chunk_input_transcript = sc.input_transcription.text
                
                if sc.turn_complete:
                    turn_complete = True
            elif msg.data:
                chunk_audio.extend(msg.data)

            if chunk_audio:
                total_audio_len += len(chunk_audio)
            
            if chunk_audio or chunk_transcript or chunk_thought or chunk_input_transcript or turn_complete:
                yield bytes(chunk_audio), chunk_transcript, chunk_input_transcript, chunk_thought, turn_complete
                if turn_complete:
                    break

