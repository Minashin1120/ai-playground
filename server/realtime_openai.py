"""OpenAI Realtime / xAI Voice / GPT-Live のセッションループ。

`server/realtime.py` から抽出（2026-11-12）。セッションの状態（`RtSession`）・
キュー・Redis ブリッジは `realtime.py`、Gemini Live は `realtime_gemini.py`、
xAI のストリーミング文字起こしは `realtime_xai_stt.py`。すべて `app.py` の
`_SERVER_PARTS` で1つの名前空間へ exec されるため、呼び出しは名前で解決する。
"""

async def _rt_openai_xai_send_loop(session, ws):
    while not session.stop_event.is_set():
        item = await _rt_next_input(session)
        if item is None:
            return
        kind = item[0]
        try:
            if kind == "audio":
                data = item[1]
                if not data or session.input_closed:
                    continue
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode("ascii"),
                }))
            elif kind == "commit":
                # Server VAD owns turn-taking: an explicit commit is rejected by
                # xAI and fails on an empty buffer at OpenAI.  Trailing silence
                # lets the VAD close a turn the user was still speaking.
                if session.speech_active and not session.input_closed:
                    silence = _rt_silence(session)
                    step = max(2, session.rate_in // 5) * 2
                    for offset in range(0, len(silence), step):
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(silence[offset:offset + step]).decode("ascii"),
                        }))
                session.input_closed = True
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
                if session.status == "connecting":
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
                _rt_append_assistant_audio(session, msg.get("delta"))
            elif mtype in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                _rt_append_assistant_text(session, msg.get("delta") or "")
            elif mtype == "conversation.item.input_audio_transcription.delta":
                # OpenAI: incremental text for one committed item.
                delta = msg.get("delta")
                if delta:
                    item_id = str(msg.get("item_id") or "current")
                    session.user_live[item_id] = session.user_live.get(item_id, "") + delta
                    _rt_refresh_user_transcript(session)
            elif mtype == "conversation.item.input_audio_transcription.updated":
                # xAI emits the cumulative transcript of the item here.
                text = str(msg.get("transcript") or "")
                if text:
                    session.user_live[str(msg.get("item_id") or "current")] = text
                    _rt_refresh_user_transcript(session)
            elif mtype == "conversation.item.input_audio_transcription.completed":
                text = str(msg.get("transcript") or "").strip()
                session.user_live.pop(str(msg.get("item_id") or "current"), None)
                if text:
                    session.user_turns.append(text)
                _rt_refresh_user_transcript(session)
            elif mtype == "conversation.item.input_audio_transcription.failed":
                logger.warning(f"Realtime STS input transcription failed: {_rt_error_message(msg.get('error'))}")
            elif mtype == "response.done":
                session.turn_count += 1
                session.assistant_turn_break = True
                response = msg.get("response") or {}
                if response.get("status") == "failed":
                    details = (response.get("status_details") or {}).get("error")
                    _rt_push_event(session, {"type": "notice", "message": _rt_error_message(details, "応答の生成に失敗しました")})
                _rt_push_event(session, {"type": "response_done"})
            elif mtype == "error":
                # Most provider errors are recoverable and the session stays
                # open; fatal ones close the socket and end this loop.
                message = _rt_error_message(msg.get("error"))
                logger.warning(f"Realtime STS provider error ({session.model_key}): {message}")
                _rt_push_event(session, {"type": "notice", "message": message})
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosedOK:
        session.stop_event.set()
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        _rt_push_event(session, {"type": "error", "message": str(exc)})
        with session.pending_cond:
            session.pending_cond.notify_all()


async def _rt_openai_translate_send_loop(session, ws):
    while not session.stop_event.is_set():
        item = await _rt_next_input(session)
        if item is None:
            return
        try:
            if item[0] == "audio":
                if item[1] and not session.input_closed:
                    await ws.send(json.dumps({
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(item[1]).decode("ascii"),
                    }))
            elif item[0] == "commit" and not session.input_closed:
                # Flush buffered audio; the server replies with session.closed.
                session.input_closed = True
                session.status = "speaking"
                await ws.send(json.dumps({"type": "session.close"}))
        except Exception as exc:
            logger.error(f"Realtime translation send error: {exc}")


async def _rt_openai_translate_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "session.output_audio.delta":
                _rt_append_assistant_audio(session, msg.get("delta"))
            elif mtype == "session.output_transcript.delta":
                _rt_append_assistant_text(session, msg.get("delta") or "")
            elif mtype == "session.input_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    session.user_live["source"] = session.user_live.get("source", "") + delta
                    _rt_refresh_user_transcript(session)
            elif mtype == "session.closed":
                session.turn_count += 1
                _rt_push_event(session, {"type": "response_done"})
                session.stop_event.set()
                return
            elif mtype == "error":
                message = _rt_error_message(msg.get("error"))
                logger.warning(f"Realtime translation provider error: {message}")
                _rt_push_event(session, {"type": "notice", "message": message})
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosedOK:
        session.stop_event.set()
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        _rt_push_event(session, {"type": "error", "message": str(exc)})


def _rt_openai_xai_session_config(session, model_key):
    """session.update payload for OpenAI Realtime GA / xAI / OpenAI translation."""
    if session.provider == "xai":
        # xAI is OpenAI-Realtime compatible but uses a top-level turn_detection.
        return {
            "voice": session.params.get("voice") or "ara",
            "turn_detection": {"type": "server_vad"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": session.rate_in},
                    # Required for conversation.item.input_audio_transcription.* events.
                    "transcription": {"model": "grok-transcribe"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": session.rate_out}},
            },
        }
    if model_key in OPENAI_TRANSLATE_MODELS:
        # Translation sessions accept only language / transcription / noise settings.
        return {
            "audio": {
                "input": {"transcription": {"model": OPENAI_TRANSLATE_TRANSCRIPTION_MODEL}},
                "output": {"language": session.params.get("target_lang") or "ja"},
            },
        }
    output = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": session.params.get("voice") or "alloy",
    }
    speed = session.params.get("speed")
    if speed is not None:
        output["speed"] = speed
    config = {
        "type": "realtime",
        "model": model_key,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": OPENAI_RT_INPUT_TRANSCRIPTION_MODEL},
                "turn_detection": {"type": "server_vad"},
            },
            "output": output,
        },
    }
    effort = session.params.get("reasoning_effort")
    if effort and model_key in OPENAI_RT_REASONING_MODELS:
        config["reasoning"] = {"effort": effort}
    return config


def _rt_openai_live_session_config(session):
    """session.start payload for GPT-Live (all fields are fixed at startup)."""
    return {
        "model": session.model_key,
        "audio": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "output": {"voice": session.params.get("voice") or "marin"},
        },
        # Reasoning and tool use are delegated to a Responses model.
        "delegation": {
            "type": "responses",
            "responses": {"model": OPENAI_LIVE_DELEGATION_MODEL},
        },
    }


async def _rt_openai_live_send_loop(session, ws):
    while not session.stop_event.is_set():
        item = await _rt_next_input(session)
        if item is None:
            return
        try:
            if item[0] == "audio":
                if item[1] and not session.input_closed:
                    await ws.send(json.dumps({
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(item[1]).decode("ascii"),
                    }))
            elif item[0] == "commit" and not session.input_closed:
                # Graceful close: pending speech drains, then session.closed.
                session.input_closed = True
                session.status = "speaking"
                await ws.send(json.dumps({"type": "session.close"}))
        except Exception as exc:
            logger.error(f"GPT-Live send error: {exc}")


def _rt_live_input_delta(session, delta):
    """Full-duplex input transcript: a new user turn starts after assistant speech."""
    if not delta:
        return
    if session.live_last_speaker == "assistant":
        text = session.user_live.pop("source", "").strip()
        if text:
            session.user_turns.append(text)
    session.live_last_speaker = "user"
    session.user_live["source"] = session.user_live.get("source", "") + delta
    _rt_refresh_user_transcript(session)


def _rt_live_output_delta(session, delta):
    if not delta:
        return
    if session.live_last_speaker == "user":
        session.assistant_turn_break = True
        session.turn_count += 1
        _rt_push_event(session, {"type": "response_start"})
    session.live_last_speaker = "assistant"
    _rt_append_assistant_text(session, delta)


async def _rt_openai_live_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "session.output_audio.delta":
                _rt_append_assistant_audio(session, msg.get("delta"))
            elif mtype == "session.output_transcript.delta":
                _rt_live_output_delta(session, msg.get("delta") or "")
            elif mtype == "session.input_transcript.delta":
                _rt_live_input_delta(session, msg.get("delta") or "")
            elif mtype == "session.closed":
                reason = str(msg.get("reason") or "")
                if reason in ("expired", "content", "connection_lost"):
                    _rt_push_event(session, {"type": "notice", "message": f"GPT-Live session closed: {reason}"})
                _rt_push_event(session, {"type": "response_done"})
                session.stop_event.set()
                return
            elif mtype == "error":
                message = _rt_error_message(msg.get("error"))
                logger.warning(f"GPT-Live provider error: {message}")
                _rt_push_event(session, {"type": "notice", "message": message})
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosedOK:
        session.stop_event.set()
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        _rt_push_event(session, {"type": "error", "message": str(exc)})


async def _rt_openai_live_session_async(session):
    headers = {"Authorization": f"Bearer {session.api_key}"}
    async with websockets.connect("wss://api.openai.com/v1/live/sessions",
                                  additional_headers=headers, max_size=None) as ws:
        session.ws = ws
        await ws.send(json.dumps({"type": "session.start", "session": _rt_openai_live_session_config(session)}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("type") == "session.started":
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
                break
            if msg.get("type") == "error":
                raise RuntimeError(_rt_error_message(msg.get("error"), "Session setup failed"))
            if msg.get("type") == "session.closed":
                raise RuntimeError(f"GPT-Live session closed: {msg.get('reason') or 'unknown'}")
        recv_task = asyncio.ensure_future(_rt_openai_live_receive_loop(session, ws))
        send_task = asyncio.ensure_future(_rt_openai_live_send_loop(session, ws))
        await _rt_run_until_stopped(session, recv_task, send_task)


async def _rt_openai_xai_session_async(session):
    model_key = session.model_key
    is_translate = session.provider == "openai" and model_key in OPENAI_TRANSLATE_MODELS
    if session.provider == "xai":
        model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
        url = f"wss://{_XAI_API_HOST}/v1/realtime?model={model_key}"
    elif is_translate:
        url = f"wss://api.openai.com/v1/realtime/translations?model={model_key}"
    else:
        url = f"wss://api.openai.com/v1/realtime?model={model_key}"
    # GA Realtime: no OpenAI-Beta header (it switches the socket to the beta
    # protocol, which rejects the GA session shape).
    headers = {"Authorization": f"Bearer {session.api_key}"}

    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        session.ws = ws
        await ws.send(json.dumps({
            "type": "session.update",
            "session": _rt_openai_xai_session_config(session, model_key),
        }))

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
                raise RuntimeError(_rt_error_message(msg.get("error"), "Session setup failed"))

        if is_translate:
            recv_task = asyncio.ensure_future(_rt_openai_translate_receive_loop(session, ws))
            send_task = asyncio.ensure_future(_rt_openai_translate_send_loop(session, ws))
        else:
            recv_task = asyncio.ensure_future(_rt_openai_xai_receive_loop(session, ws))
            send_task = asyncio.ensure_future(_rt_openai_xai_send_loop(session, ws))
        await _rt_run_until_stopped(session, recv_task, send_task)


async def _rt_run_until_stopped(session, recv_task, send_task):
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
    for task in (recv_task, send_task):
        try:
            await task
        except BaseException:
            pass
