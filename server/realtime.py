# =============================================================================
# True real-time STS sessions (OpenAI Realtime / Grok Voice / Gemini Live)
# -----------------------------------------------------------------------------
# Server-held WebSocket sessions.  The browser streams microphone PCM over HTTP,
# the server relays it to the provider WebSocket in real time, and the provider's
# audio + transcripts stream back to the browser over SSE.  Provider-side VAD
# (or Gemini's natural turn handling) drives automatic turn-taking, so the user
# can talk continuously and hear the model reply while speaking.
#
# gunicorn runs several worker processes, so the HTTP requests of one session
# (start / audio / stream / commit / save) can land on different workers.  The
# worker that handled /start owns the provider WebSocket; every other request
# talks to it through Redis:
#   rt:meta:<sid>  hash  user_id / status        (any worker: ownership check)
#   rt:in:<sid>    list  A<pcm> audio, C commit, X cancel, F<json> save
#   rt:ev:<sid>    list  JSON events for the SSE stream
#   rt:res:<sid>   list  JSON result of the save request
# =============================================================================
RT_SESSIONS = {}                 # sessions owned by this worker process
RT_SESSIONS_LOCK = threading.Lock()
RT_MAX_SESSION_SECONDS = 15 * 60
RT_SESSION_TTL_SECONDS = 30 * 60
RT_BRIDGE_TTL_SECONDS = RT_MAX_SESSION_SECONDS + RT_SESSION_TTL_SECONDS
RT_AUDIO_POST_MAX = 1 << 20
RT_PCM_CAP = 512 * 1024 * 1024
RT_SAVE_WAIT_SECONDS = 45
# Trailing silence appended on stop so provider server VAD closes the last turn
# (input_audio_buffer.commit is rejected while server VAD is enabled on xAI).
RT_STOP_SILENCE_SECONDS = 1.5

# OpenAI Realtime GA: user speech is transcribed only when input transcription
# is configured on the session.
OPENAI_RT_INPUT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
OPENAI_TRANSLATE_MODELS = {"gpt-realtime-translate"}
OPENAI_TRANSLATE_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
# Reasoning Realtime models accept session.reasoning.effort.
OPENAI_RT_REASONING_MODELS = {"gpt-realtime-2", "gpt-realtime-2.1", "gpt-realtime-2.1-mini"}
OPENAI_RT_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
# GPT-Live: full-duplex voice on /v1/live/sessions with Responses delegation.
OPENAI_LIVE_MODELS = {"gpt-live-1"}
OPENAI_LIVE_DELEGATION_MODEL = "gpt-5.6-luna"

GEMINI_LIVE_TRANSLATE_MODELS = {"gemini-3.5-live-translate-preview"}
GEMINI_LIVE_TRANSCRIBE_MODELS = {"gemini-3.5-transcribe-live"}
# Gemini Live models that reject thinkingConfig.thinkingLevel (2.5 uses a
# thinking budget, 3.8 Live has fixed latency, translate/transcribe do not think).
GEMINI_LIVE_NO_THINKING_LEVEL_MODELS = {
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.8-live",
} | GEMINI_LIVE_TRANSLATE_MODELS | GEMINI_LIVE_TRANSCRIBE_MODELS


def _rt_is_conversation_model(model_key):
    """True for STS models that support a persistent streaming conversation."""
    model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
    if not is_sts_model(model_key):
        return False
    # Gemini Live used to be browser-direct only.  It now also uses this
    # server-held session so native clients can use the same authenticated
    # provider connection without receiving an API key.
    if model_key in (
        "gemini-3.1-flash-live-preview",
        "gemini-3.8-live",
        "gemini-3.8-live-extended-thinking",
        "gemini-3.5-live-translate-preview",
        "gemini-3.5-transcribe-live",
    ):
        return True
    if model_key in XAI_LIVE_STT_MODELS:
        return True
    meta = STS_MODELS.get(model_key, {})
    if meta.get("mode") == "transcription":
        return False
    # One-shot transcription session models remain excluded.
    if model_key == "gpt-realtime-whisper":
        return False
    return True


# Streaming STT models served by xAI's wss://api.x.ai/v1/stt endpoint.
XAI_LIVE_STT_MODELS = {"grok-voice-transcribe-2.0"}
XAI_LIVE_STT_MAX_KEYTERMS = 100
XAI_LIVE_STT_KEYTERM_MAX_CHARS = 50


def _rt_is_live_transcription_session(session):
    return session.model_key in XAI_LIVE_STT_MODELS


def _rt_is_transcription_session(session):
    """Speech-to-text sessions: the transcript is saved as the assistant reply."""
    return _rt_is_live_transcription_session(session) or session.model_key in GEMINI_LIVE_TRANSCRIBE_MODELS


def _rt_drains_on_stop(session):
    """Sessions that flush buffered audio after the input ends and then close."""
    return (
        _rt_is_live_transcription_session(session)
        or session.model_key in OPENAI_TRANSLATE_MODELS
        or session.model_key in OPENAI_LIVE_MODELS
    )


def _rt_key(kind, session_id):
    return f"rt:{kind}:{session_id}"


def _rt_push_event(session, event):
    if getattr(session, "bridged", False):
        try:
            key = _rt_key("ev", session.session_id)
            pipe = redis_conn.pipeline()
            pipe.rpush(key, json.dumps(event, ensure_ascii=False))
            pipe.expire(key, RT_BRIDGE_TTL_SECONDS)
            pipe.execute()
        except Exception as exc:
            logger.error(f"Realtime STS event publish failed: {exc}")
        return
    with session.pending_cond:
        session.pending.append(event)
        session.pending_cond.notify_all()


def _rt_set_meta_status(session, status):
    session.status = status
    if not getattr(session, "bridged", False):
        return
    try:
        key = _rt_key("meta", session.session_id)
        pipe = redis_conn.pipeline()
        pipe.hset(key, "status", status)
        pipe.expire(key, RT_BRIDGE_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.error(f"Realtime STS status publish failed: {exc}")


def _rt_error_message(error, default="Provider error"):
    """Readable text from a provider error payload (dict, string or None)."""
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        return str(message or default)
    if error:
        return str(error)
    return default


def _rt_refresh_user_transcript(session):
    """Recompute the whole user transcript and send it as one cumulative event."""
    parts = [t.strip() for t in session.user_turns + list(session.user_live.values()) if t and t.strip()]
    text = "\n".join(parts)
    session.user_transcript = text
    shown = _rt_join_transcript(text, session.user_interim) if session.user_interim else text
    _rt_push_event(session, {"type": "transcript", "role": "user", "delta": shown, "cumulative": True})


def _rt_append_assistant_text(session, text):
    if not text:
        return
    if session.assistant_turn_break and session.assistant_transcript and not session.assistant_transcript.endswith("\n"):
        text = "\n" + text
    session.assistant_turn_break = False
    session.assistant_transcript += text
    _rt_push_event(session, {"type": "transcript", "role": "assistant", "delta": text})


def _rt_append_assistant_audio(session, audio_b64):
    if not audio_b64:
        return
    try:
        binary = base64.b64decode(audio_b64)
    except Exception:
        binary = b""
    if not binary:
        return
    with session.assistant_lock:
        if len(session.assistant_audio) + len(binary) <= RT_PCM_CAP:
            session.assistant_audio += binary
    _rt_push_event(session, {"type": "audio", "data": audio_b64})


def _rt_purge_old_sessions():
    now = time.time()
    with RT_SESSIONS_LOCK:
        stale = [
            sid for sid, sess in list(RT_SESSIONS.items())
            if (now - sess.started_at) > RT_BRIDGE_TTL_SECONDS
        ]
        for sid in stale:
            sess = RT_SESSIONS.pop(sid, None)
            if sess:
                sess.cancelled = True
                sess.stop_event.set()


def _rt_get_session(session_id):
    """Bridge metadata of a session owned by the current user (any worker)."""
    session_id = str(session_id or "")
    if not session_id.startswith("rt_"):
        return None
    try:
        raw = redis_conn.hgetall(_rt_key("meta", session_id)) or {}
    except Exception as exc:
        logger.error(f"Realtime STS session lookup failed: {exc}")
        return None
    meta = {
        (k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else str(v))
        for k, v in raw.items()
    }
    if not meta or meta.get("user_id") != str(current_user.id):
        return None
    meta["session_id"] = session_id
    return meta


def _rt_send_command(session_id, payload):
    key = _rt_key("in", session_id)
    pipe = redis_conn.pipeline()
    pipe.rpush(key, payload)
    pipe.expire(key, RT_BRIDGE_TTL_SECONDS)
    pipe.execute()


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
        self.pending = []               # output events (unbridged sessions / tests)
        self.pending_cond = threading.Condition()
        self.cmd_queue = _queue.Queue()  # reserved for future steering commands
        self.stop_event = threading.Event()
        self.status = "connecting"       # connecting|ready|speaking|stopped|error|closed
        self.error = None
        self.started_at = time.time()
        self.thread = None
        self.pump_thread = None
        self.bridged = False             # True: events/commands go through Redis
        self.e2ee = False
        self.cancelled = False
        self.assistant_audio = bytearray()   # accumulated output PCM (for saving)
        self.assistant_lock = threading.Lock()
        self.user_audio = bytearray()        # accumulated input PCM (for saving)
        self.user_lock = threading.Lock()
        self.user_transcript = ""
        self.user_turns = []                 # finalized user utterances
        self.user_live = {}                  # item/turn id -> in-progress text
        self.user_interim = ""               # Gemini speculative partial
        self.user_turn_index = 0
        self.assistant_transcript = ""
        self.assistant_turn_break = False
        self.live_last_speaker = None        # GPT-Live: "user" / "assistant"
        self.assistant_thought = ""
        self.speech_active = False
        self.input_closed = False
        self.turn_count = 0
        self.saved = False


def _normalize_rt_params(provider, model_key, data):
    """Validate / normalize real-time session parameters (always returns a dict)."""
    data = data or {}
    meta = STS_MODELS.get(model_key, {})
    params = {
        "rate_in": int(meta.get("rate_in", 24000)),
        "rate_out": int(meta.get("rate_out", 24000)),
    }
    if provider == "openai":
        # OpenAI Realtime accepts 24 kHz PCM only.
        params["rate_in"] = params["rate_out"] = 24000
        if model_key in OPENAI_LIVE_MODELS:
            # Voice is fixed at startup; GPT-Live has no speed setting.
            v = str(data.get("voice") or "marin").lower()
            params["voice"] = v if v in OPENAI_LIVE_VOICES else "marin"
        else:
            v = str(data.get("voice") or "alloy").lower()
            params["voice"] = v if v in OPENAI_STS_VOICES else "alloy"
            speed = clamp_float(data.get("speed"), 0.25, 1.5)
            if speed is not None:
                params["speed"] = speed
        if model_key in OPENAI_RT_REASONING_MODELS:
            effort = str(data.get("reasoning_effort") or "").strip().lower()
            if effort in OPENAI_RT_REASONING_EFFORTS:
                params["reasoning_effort"] = effort
        if model_key in OPENAI_TRANSLATE_MODELS:
            lang = str(data.get("target_lang") or "ja").strip().lower()
            # Translation output takes a base language code (zh-CN -> zh).
            lang = lang.split("-")[0].split("_")[0][:8]
            params["target_lang"] = lang if lang.isalpha() else "ja"
    elif provider == "xai":
        # xAI voice IDs are lowercase (case-insensitive on the API).
        v = str(data.get("voice") or "ara").strip().lower()
        params["voice"] = v if v in XAI_STS_VOICES else "ara"
        for field in ("rate_in", "rate_out"):
            try:
                rate = int(data.get(field) or 0)
            except (TypeError, ValueError):
                rate = 0
            if rate in XAI_PCM_RATES:
                params[field] = rate
    elif provider == "google":
        v = str(data.get("voice") or "Kore")
        params["voice"] = v if v in GEMINI_STS_VOICES else "Kore"
        thinking = str(data.get("thinking_level") or "").strip().lower()
        if model_key in GEMINI_LIVE_NO_THINKING_LEVEL_MODELS:
            params["thinking_level"] = None
        elif model_key == "gemini-3.8-live-extended-thinking":
            params["thinking_level"] = thinking if thinking in {"low", "medium", "high"} else "medium"
        else:
            params["thinking_level"] = thinking if thinking in {"minimal", "low", "medium", "high"} else None
        params["include_thoughts"] = bool(data.get("include_thoughts")) and params["thinking_level"] is not None
        params["target_lang"] = str(data.get("target_lang") or "ja").strip().lower()[:16] or "ja"
        mode = str(data.get("transcription_mode") or "VERBATIM").strip().upper()
        params["transcription_mode"] = mode if mode in {"SMART", "VERBATIM"} else "VERBATIM"
        vocabulary = data.get("custom_vocabulary")
        if isinstance(vocabulary, list):
            params["custom_vocabulary"] = [str(item).strip()[:120] for item in vocabulary if str(item).strip()][:1000]
        else:
            params["custom_vocabulary"] = []
    if model_key in XAI_LIVE_STT_MODELS:
        # 16 kHz PCM is the model's native rate; keyterms bias recognition.
        params["rate_in"] = params["rate_out"] = 16000
        vocabulary = data.get("custom_vocabulary")
        terms = []
        if isinstance(vocabulary, list):
            for item in vocabulary:
                term = str(item or "").strip()[:XAI_LIVE_STT_KEYTERM_MAX_CHARS]
                if term and term not in terms:
                    terms.append(term)
        params["keyterms"] = terms[:XAI_LIVE_STT_MAX_KEYTERMS]
    return params


def _rt_silence(session, seconds=RT_STOP_SILENCE_SECONDS):
    return b"\x00\x00" * int(session.rate_in * seconds)


async def _rt_next_input(session):
    """Next queued input item, or None once the session stops."""
    while not session.stop_event.is_set():
        try:
            return session.audio_in.get_nowait()
        except _queue.Empty:
            await asyncio.sleep(0.02)
    return None


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


async def _rt_gemini_send_loop(session, ws):
    while not session.stop_event.is_set():
        item = await _rt_next_input(session)
        if item is None:
            return
        try:
            if item[0] == "audio":
                data = item[1]
                if not data:
                    continue
                await ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(data).decode("ascii"),
                            "mimeType": f"audio/pcm;rate={session.rate_in}",
                        }
                    }
                }))
            elif item[0] == "commit" and not session.input_closed:
                # Microphone closed: flush cached audio so the last utterance
                # is answered / finalized without waiting for more input.
                session.input_closed = True
                await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        except Exception as exc:
            logger.error(f"Realtime STS Gemini send error: {exc}")


def _rt_gemini_close_user_turn(session):
    key = f"turn{session.user_turn_index}"
    text = session.user_live.pop(key, "")
    if text.strip():
        session.user_turns.append(text.strip())
    session.user_turn_index += 1


async def _rt_gemini_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            interaction_status = (
                msg.get("interactionStatus")
                or msg.get("interaction_status")
                or (msg.get("serverContent") or {}).get("interactionStatus")
                or (msg.get("serverContent") or {}).get("interaction_status")
            )
            if interaction_status:
                status_text = str(interaction_status).upper()
                session.status = "processing" if status_text == "IN_PROGRESS" else "ready"
                _rt_push_event(session, {
                    "type": "interaction_status",
                    "status": status_text,
                })
            if msg.get("setupComplete") is not None:
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
            sc = msg.get("serverContent")
            if sc is not None:
                model_turn = sc.get("modelTurn")
                if model_turn:
                    for part in model_turn.get("parts") or []:
                        inline = part.get("inlineData") or {}
                        _rt_append_assistant_audio(session, inline.get("data"))
                        text = part.get("text")
                        if text:
                            if part.get("thought"):
                                session.assistant_thought += text
                                _rt_push_event(session, {"type": "transcript", "role": "thought", "delta": text})
                            elif session.model_key not in GEMINI_LIVE_TRANSCRIBE_MODELS:
                                _rt_append_assistant_text(session, text)
                out_tr = sc.get("outputTranscription") or {}
                if out_tr.get("text"):
                    _rt_append_assistant_text(session, out_tr["text"])
                in_tr = sc.get("inputTranscription") or {}
                interim = sc.get("interimInputTranscription") or {}
                if in_tr.get("text"):
                    key = f"turn{session.user_turn_index}"
                    session.user_live[key] = session.user_live.get(key, "") + in_tr["text"]
                    session.user_interim = ""
                    _rt_refresh_user_transcript(session)
                elif "text" in interim:
                    session.user_interim = str(interim.get("text") or "")
                    _rt_refresh_user_transcript(session)
                if sc.get("interrupted"):
                    _rt_push_event(session, {"type": "interrupted"})
                if sc.get("turnComplete"):
                    session.turn_count += 1
                    session.assistant_turn_break = True
                    if session.model_key not in GEMINI_LIVE_TRANSCRIBE_MODELS:
                        _rt_gemini_close_user_turn(session)
                    _rt_push_event(session, {"type": "turn_complete"})
            if msg.get("goAway") is not None:
                logger.info(f"Realtime STS Gemini goAway: {msg.get('goAway')}")
            if msg.get("error"):
                raise RuntimeError(_rt_error_message(msg.get("error")))
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


def _rt_gemini_setup_message(session):
    """BidiGenerateContentSetup for the server-held Gemini Live session."""
    is_translate = session.model_key in GEMINI_LIVE_TRANSLATE_MODELS
    is_transcribe = session.model_key in GEMINI_LIVE_TRANSCRIBE_MODELS
    generation_config = {
        "responseModalities": ["TEXT"] if is_transcribe else ["AUDIO"],
    }
    setup = {
        "model": f"models/{session.model_key}",
        "generationConfig": generation_config,
    }
    if is_transcribe:
        transcription = {"mode": session.params.get("transcription_mode", "VERBATIM")}
        vocabulary = session.params.get("custom_vocabulary") or []
        if vocabulary:
            transcription["customVocabulary"] = vocabulary
        setup["inputAudioTranscription"] = transcription
    else:
        setup["inputAudioTranscription"] = {}
        setup["outputAudioTranscription"] = {}
    if is_translate:
        # translationConfig is a GenerationConfig field (not a setup field).
        generation_config["translationConfig"] = {
            "targetLanguageCode": session.params.get("target_lang", "ja"),
            "echoTargetLanguage": True,
        }
    voice = session.params.get("voice")
    if not is_translate and not is_transcribe and voice and voice in GEMINI_STS_VOICES:
        generation_config["speechConfig"] = {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
        }
    thinking_level = session.params.get("thinking_level")
    if thinking_level and session.model_key not in GEMINI_LIVE_NO_THINKING_LEVEL_MODELS:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
            "includeThoughts": bool(session.params.get("include_thoughts")),
        }
    return {"setup": setup}


async def _rt_gemini_session_async(session):
    ws_url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={quote(session.api_key, safe='')}"
    )
    async with websockets.connect(ws_url, max_size=None) as ws:
        session.ws = ws
        await ws.send(json.dumps(_rt_gemini_setup_message(session)))
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
                raise RuntimeError(_rt_error_message(msg.get("error")))

        recv_task = asyncio.ensure_future(_rt_gemini_receive_loop(session, ws))
        send_task = asyncio.ensure_future(_rt_gemini_send_loop(session, ws))
        await _rt_run_until_stopped(session, recv_task, send_task)


def _rt_join_transcript(prev, nxt):
    """Join finalized STT chunks; no space between CJK text (Japanese etc.)."""
    prev = prev or ""
    nxt = (nxt or "").strip()
    if not prev:
        return nxt
    if not nxt:
        return prev
    def _cjk(ch):
        return ord(ch) >= 0x3000
    if _cjk(prev[-1]) or _cjk(nxt[0]) or prev[-1].isspace():
        return prev + nxt
    return prev + " " + nxt


async def _rt_xai_stt_send_loop(session, ws):
    while not session.stop_event.is_set():
        try:
            item = session.audio_in.get_nowait()
        except _queue.Empty:
            item = None
        if item is None:
            await asyncio.sleep(0.02)
            continue
        try:
            if item[0] == "audio":
                if item[1]:
                    # Raw binary PCM frames (no base64) per xAI streaming STT.
                    await ws.send(item[1])
            elif item[0] == "commit":
                # End of audio: the server flushes and replies with transcript.done.
                await ws.send(json.dumps({"type": "audio.done"}))
                session.status = "speaking"
                return
        except Exception as exc:
            logger.error(f"Realtime STT send error: {exc}")


async def _rt_xai_stt_receive_loop(session, ws):
    finalized = ""
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "transcript.partial":
                text = str(msg.get("text") or "")
                if msg.get("is_final"):
                    finalized = _rt_join_transcript(finalized, text)
                    shown = finalized
                else:
                    shown = _rt_join_transcript(finalized, text)
                session.user_transcript = finalized
                _rt_push_event(session, {"type": "transcript", "role": "user", "delta": shown, "cumulative": True})
                if msg.get("speech_final"):
                    _rt_push_event(session, {"type": "speech_stopped"})
            elif mtype == "transcript.done":
                text = str(msg.get("text") or "").strip()
                if text:
                    finalized = text
                session.user_transcript = finalized
                _rt_push_event(session, {"type": "transcript", "role": "user", "delta": finalized, "cumulative": True})
                session.turn_count += 1
                _rt_push_event(session, {"type": "turn_complete"})
                session.stop_event.set()
                return
            elif mtype == "error":
                raise RuntimeError(str(msg.get("message") or msg.get("error") or "Provider error"))
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


async def _rt_xai_stt_session_async(session):
    query = [
        ("model", session.model_key),
        ("sample_rate", str(session.rate_in)),
        ("encoding", "pcm"),
        ("interim_results", "true"),
    ]
    query += [("keyterm", term) for term in session.params.get("keyterms", [])]
    url = f"wss://{_XAI_API_HOST}/v1/stt?{urlencode(query)}"
    headers = {"Authorization": f"Bearer {session.api_key}"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        session.ws = ws
        # Wait for transcript.created before streaming audio.
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("type") == "transcript.created":
                session.status = "ready"
                _rt_push_event(session, {"type": "status", "status": "ready"})
                break
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message") or "Session setup failed"))

        recv_task = asyncio.ensure_future(_rt_xai_stt_receive_loop(session, ws))
        send_task = asyncio.ensure_future(_rt_xai_stt_send_loop(session, ws))
        await _rt_run_until_stopped(session, recv_task, send_task)


def _rt_worker(session):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session.loop = loop
    try:
        if session.provider == "google":
            loop.run_until_complete(_rt_gemini_session_async(session))
        elif _rt_is_live_transcription_session(session):
            loop.run_until_complete(_rt_xai_stt_session_async(session))
        elif session.model_key in OPENAI_LIVE_MODELS:
            loop.run_until_complete(_rt_openai_live_session_async(session))
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
        _rt_set_meta_status(session, session.status)
        # Tell the SSE stream the provider session is over (clients then save).
        _rt_push_event(session, {"type": "final", "status": session.status})
        with session.pending_cond:
            session.pending_cond.notify_all()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _rt_save_session(session, thread_id):
    """Persist the finished session as a user/assistant message pair.

    Runs in the owning worker (it holds the audio) inside an app context.
    Returns (payload, http_status).
    """
    with session.assistant_lock:
        assistant_pcm = bytes(session.assistant_audio)
    with session.user_lock:
        user_pcm = bytes(session.user_audio)
    live_transcript = None
    if _rt_is_transcription_session(session):
        live_transcript = (session.user_transcript or "").strip()
        if not live_transcript:
            user_pcm = b""  # nothing recognized: do not keep the recording

    raw_user_text = (session.user_transcript or "").strip()
    user_text = raw_user_text or "音声メッセージ"
    assistant_text = (session.assistant_transcript or "").strip()
    assistant_thought = (session.assistant_thought or "").strip()
    if live_transcript is not None:
        # Live transcription: the transcript is the assistant output; the user
        # message keeps the recorded audio (same layout as Gemini Live transcribe).
        assistant_text = live_transcript
        user_text = "音声文字起こし" if live_transcript else ""

    # Nothing was captured — drop the empty session without saving a message.
    if (len(assistant_pcm) < 1024 and len(user_pcm) < 1024
            and not raw_user_text and not assistant_text.strip()):
        return {'status': 'empty'}, 200

    audio_url = None
    in_fname = None
    try:
        if len(assistant_pcm) >= 1024:
            wav_bytes = _pcm_to_wav_bytes(assistant_pcm, rate=session.rate_out)
            out_fname, _ = _save_user_audio(session.user_id, wav_bytes, ".wav", session.e2ee)
            audio_url = f"/files/{session.user_id}/{out_fname}"
        if len(user_pcm) >= 1024:
            u_wav = _pcm_to_wav_bytes(user_pcm, rate=session.rate_in)
            in_fname, _ = _save_user_audio(session.user_id, u_wav, ".wav", session.e2ee)
    except Exception:
        logger.exception("Realtime STS audio save error")

    t = resolve_thread_for_user(thread_id, session.user_id) if thread_id else None
    if not t:
        t = Thread(
            user_id=session.user_id,
            public_id=generate_thread_public_id(),
            is_temporary=True,
        )
        db.session.add(t)
        safe_db_commit()
    thread_db_id = t.id

    thought_tag = f"<thought>\n{assistant_thought}\n</thought>\n" if assistant_thought else ""
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n' if audio_url else ""
    assistant_content = thought_tag + (assistant_text + "\n" if assistant_text else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if session.e2ee else user_text
        a_content = encrypt_val(assistant_content) if session.e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(user_text, session.model_key)
        assistant_tokens_out = count_tokens_for_display(assistant_text, session.model_key)
        if assistant_thought:
            assistant_tokens_out += count_tokens_for_display(assistant_thought, session.model_key)

        parent_id = None
        last_msg = Message.query.filter_by(thread_id=thread_db_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_db_id,
            role='user',
            content=u_content,
            image_url=json.dumps([f"{session.user_id}/{in_fname}"]) if in_fname else None,
            is_encrypted=session.e2ee,
            parent_id=parent_id,
            model=session.model_key,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None),
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_db_id,
            role='assistant',
            content=a_content,
            model=session.model_key,
            is_encrypted=session.e2ee,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out),
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as exc:
        logger.exception("Realtime STS message save error")
        try:
            db.session.rollback()
        except Exception:
            pass
        return {'error': f'メッセージ保存に失敗しました: {exc}', 'audio_url': audio_url}, 500
    return {'status': 'ok', 'audio_url': audio_url, 'thread_id': str(thread_db_id)}, 200


def _rt_finish_and_save(session, thread_id):
    if _rt_drains_on_stop(session) and not session.stop_event.is_set():
        # Flush trailing audio and wait briefly for the final transcript /
        # translation (transcript.done or session.closed).
        session.audio_in.put(("commit",))
        session.stop_event.wait(timeout=8)
    session.stop_event.set()
    if session.thread:
        session.thread.join(timeout=6)
    with app.app_context():
        try:
            return _rt_save_session(session, thread_id)
        finally:
            try:
                db.session.remove()
            except Exception:
                pass


def _rt_cleanup_bridge(session_id, keep_result=False):
    keys = [_rt_key("meta", session_id), _rt_key("in", session_id), _rt_key("ev", session_id)]
    if not keep_result:
        keys.append(_rt_key("res", session_id))
    try:
        redis_conn.delete(*keys)
    except Exception as exc:
        logger.error(f"Realtime STS bridge cleanup failed: {exc}")


def _rt_input_pump(session):
    """Owner-side loop: moves Redis commands into the provider session."""
    in_key = _rt_key("in", session.session_id)
    idle_deadline = None
    try:
        while True:
            now = time.time()
            if session.stop_event.is_set():
                # Provider session ended on its own: keep the audio for a
                # while so the client can still save it.
                if idle_deadline is None:
                    idle_deadline = now + RT_SESSION_TTL_SECONDS
                elif now > idle_deadline:
                    break
            if now - session.started_at > RT_BRIDGE_TTL_SECONDS or session.cancelled:
                break
            try:
                item = redis_conn.blpop([in_key], timeout=1)
            except Exception as exc:
                logger.error(f"Realtime STS input pump error: {exc}")
                time.sleep(1)
                continue
            if not item:
                continue
            payload = item[1] or b""
            kind = payload[:1]
            if kind == b"A":
                data = payload[1:]
                if not data or session.stop_event.is_set():
                    continue
                with session.user_lock:
                    if len(session.user_audio) + len(data) <= RT_PCM_CAP:
                        session.user_audio += data
                session.audio_in.put(("audio", data))
            elif kind == b"C":
                session.audio_in.put(("commit",))
            elif kind == b"X":
                session.cancelled = True
                session.stop_event.set()
                break
            elif kind == b"F":
                try:
                    request_data = json.loads(payload[1:].decode("utf-8") or "{}")
                except Exception:
                    request_data = {}
                try:
                    result, status = _rt_finish_and_save(session, request_data.get("thread_id"))
                except Exception as exc:
                    logger.exception("Realtime STS save failed")
                    result, status = {'error': f'保存に失敗しました: {exc}'}, 500
                result["_status"] = status
                session.saved = True
                res_key = _rt_key("res", session.session_id)
                pipe = redis_conn.pipeline()
                pipe.rpush(res_key, json.dumps(result, ensure_ascii=False))
                pipe.expire(res_key, 120)
                pipe.execute()
                break
    finally:
        session.stop_event.set()
        if session.thread and session.thread is not threading.current_thread():
            session.thread.join(timeout=6)
        with RT_SESSIONS_LOCK:
            RT_SESSIONS.pop(session.session_id, None)
        _rt_cleanup_bridge(session.session_id, keep_result=session.saved)


def _rt_start_bridged_session(session, e2ee):
    """Register the session in Redis and start the provider + input threads."""
    session.bridged = True
    session.e2ee = bool(e2ee)
    meta_key = _rt_key("meta", session.session_id)
    pipe = redis_conn.pipeline()
    pipe.hset(meta_key, mapping={
        "user_id": str(session.user_id),
        "status": session.status,
        "model": session.model_key,
        "owner": str(os.getpid()),
    })
    pipe.expire(meta_key, RT_BRIDGE_TTL_SECONDS)
    pipe.execute()
    with RT_SESSIONS_LOCK:
        RT_SESSIONS[session.session_id] = session
    session.thread = threading.Thread(target=_rt_worker, args=(session,), daemon=True, name=f"rt-{session.session_id}")
    session.pump_thread = threading.Thread(target=_rt_input_pump, args=(session,), daemon=True, name=f"rt-in-{session.session_id}")
    session.thread.start()
    session.pump_thread.start()


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
