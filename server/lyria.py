# =============================================================================
# Lyria RealTime (lyria-realtime-exp) real-time music session manager
# -----------------------------------------------------------------------------
# Lyria RealTime is a WebSocket-based streaming music generation model. The
# browser cannot reach Google's BidiGenerateMusic WebSocket directly without
# exposing the user's raw Gemini API key, so the server keeps a persistent
# session (thread + asyncio loop) per active session, streams audio deltas to
# the client over SSE, and accepts steering commands over HTTP.
# =============================================================================
LYRIA_REALTIME_MODEL = "lyria-realtime-exp"
LYRIA_SESSIONS = {}
LYRIA_SESSIONS_LOCK = threading.Lock()
LYRIA_MAX_SESSION_SECONDS = 15 * 60  # auto-stop to bound cost
LYRIA_MAX_AUDIO_BYTES = 512 * 1024 * 1024  # cap accumulated PCM
LYRIA_PROMPT_MAX_CHARS = 4000
LYRIA_SESSION_TTL_SECONDS = 30 * 60  # closed sessions are purged after this

LYRIA_SCALES = {
    "C_MAJOR_A_MINOR": "C major / A minor",
    "D_FLAT_MAJOR_B_FLAT_MINOR": "D\u266d major / B\u266d minor",
    "D_MAJOR_B_MINOR": "D major / B minor",
    "E_FLAT_MAJOR_C_MINOR": "E\u266d major / C minor",
    "E_MAJOR_D_FLAT_MINOR": "E major / C\u266f/D\u266d minor",
    "F_MAJOR_D_MINOR": "F major / D minor",
    "G_FLAT_MAJOR_E_FLAT_MINOR": "G\u266d major / E\u266d minor",
    "G_MAJOR_E_MINOR": "G major / E minor",
    "A_FLAT_MAJOR_F_MINOR": "A\u266d major / F minor",
    "A_MAJOR_G_FLAT_MINOR": "A major / F\u266f/G\u266d minor",
    "B_FLAT_MAJOR_G_MINOR": "B\u266d major / G minor",
    "B_MAJOR_A_FLAT_MINOR": "B major / G\u266f/A\u266d minor",
}


class LyriaSession:
    """One persistent Lyria RealTime streaming session for a single user."""

    def __init__(self, session_id, user_id, api_key, prompts, config):
        self.session_id = session_id
        self.user_id = user_id
        self.api_key = api_key
        self.prompts = prompts          # [{"text", "weight"}]
        self.config = config            # normalized musicGenerationConfig (camelCase)
        self.loop = None                # asyncio event loop of the worker thread
        self.ws = None                  # websocket to Google (owned by worker thread)
        self.audio_buffer = bytearray()  # accumulated raw PCM (48kHz stereo s16le)
        self.audio_lock = threading.Lock()
        self.pending = []                # base64 deltas not yet consumed by the SSE stream
        self.pending_cond = threading.Condition()
        self.cmd_queue = _queue.Queue()  # steering commands from HTTP handlers
        self.stop_event = threading.Event()
        self.status = "connecting"       # connecting|streaming|paused|stopped|error|closed
        self.error = None
        self.filtered_prompt = None
        self.started_at = time.time()
        self.thread = None


def _normalize_lyria_config(raw):
    """Validate / normalize a Lyria RealTime music generation config (camelCase output)."""
    raw = raw or {}
    # The client must know the output format. Lyria RealTime emits raw 16-bit
    # PCM at 48kHz stereo; make it explicit so updates don't reset it.
    cfg = {
        "audioFormat": "pcm16",
        "sampleRateHz": 48000,
    }

    def clamp(name, lo, hi, cast=float):
        val = raw.get(name)
        if val is None or val == "":
            return None
        try:
            value = cast(val)
        except (TypeError, ValueError):
            return None
        return max(lo, min(hi, value))

    bpm = clamp("bpm", 60, 200, int)
    if bpm is not None:
        cfg["bpm"] = bpm
    guidance = clamp("guidance", 0.0, 6.0)
    if guidance is not None:
        cfg["guidance"] = guidance
    density = clamp("density", 0.0, 1.0)
    if density is not None:
        cfg["density"] = density
    brightness = clamp("brightness", 0.0, 1.0)
    if brightness is not None:
        cfg["brightness"] = brightness
    temperature = clamp("temperature", 0.0, 3.0)
    if temperature is not None:
        cfg["temperature"] = temperature
    top_k = clamp("top_k", 1, 1000, int)
    if top_k is not None:
        cfg["topK"] = top_k
    seed = raw.get("seed")
    if seed is not None and str(seed).strip():
        try:
            seed_val = int(str(seed).strip())
            if 0 <= seed_val <= 2147483647:
                cfg["seed"] = seed_val
        except (TypeError, ValueError):
            pass
    scale = str(raw.get("scale") or "").strip()
    if scale and scale != "SCALE_UNSPECIFIED" and scale in LYRIA_SCALES:
        cfg["scale"] = scale
    mode = str(raw.get("music_generation_mode") or "QUALITY").strip().upper()
    if mode not in ("QUALITY", "DIVERSITY", "VOCALIZATION"):
        mode = "QUALITY"
    cfg["musicGenerationMode"] = mode
    for src, dst in (
        ("mute_bass", "muteBass"),
        ("mute_drums", "muteDrums"),
        ("only_bass_and_drums", "onlyBassAndDrums"),
    ):
        val = raw.get(src)
        if val is not None:
            cfg[dst] = bool(val)
    return cfg


def _normalize_lyria_prompts(raw_list):
    prompts = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0 or weight > 100:
            weight = 1.0
        prompts.append({"text": text[:LYRIA_PROMPT_MAX_CHARS], "weight": weight})
    return prompts


def _lyria_pcm_to_wav_stereo(pcm_bytes, rate=48000):
    """Wrap raw 16-bit stereo PCM (interleaved L/R) into a WAV container."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def _lyria_send_control(session, action):
    if session.ws:
        await session.ws.send(json.dumps({"playbackControl": action}))


async def _lyria_receive_loop(session, ws):
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            server_content = msg.get("serverContent")
            if server_content and server_content.get("audioChunks"):
                for chunk in server_content["audioChunks"]:
                    data = chunk.get("data")
                    if not data:
                        continue
                    try:
                        binary = base64.b64decode(data)
                    except Exception:
                        continue
                    with session.audio_lock:
                        if len(session.audio_buffer) + len(binary) > LYRIA_MAX_AUDIO_BYTES:
                            continue
                        session.audio_buffer += binary
                    with session.pending_cond:
                        session.pending.append(data)
                        session.pending_cond.notify_all()
            if msg.get("filteredPrompt"):
                session.filtered_prompt = msg.get("filteredPrompt")
            if msg.get("error"):
                raise RuntimeError(str(msg.get("error")))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        session.stop_event.set()
        with session.pending_cond:
            session.pending_cond.notify_all()


async def _lyria_handle_command(session, ws, cmd):
    ctype = cmd.get("type")
    if ctype == "prompts":
        prompts = _normalize_lyria_prompts(cmd.get("weighted_prompts"))
        if not prompts:
            return
        await ws.send(json.dumps({"clientContent": {"weightedPrompts": prompts}}))
        session.prompts = prompts
    elif ctype == "config":
        cfg = _normalize_lyria_config(cmd.get("config"))
        await ws.send(json.dumps({"musicGenerationConfig": cfg}))
        session.config = cfg
        if cmd.get("reset_context"):
            await _lyria_send_control(session, "RESET_CONTEXT")
    elif ctype == "control":
        action = str(cmd.get("action") or "").upper()
        if action not in ("PLAY", "PAUSE", "STOP", "RESET_CONTEXT"):
            return
        await _lyria_send_control(session, action)
        if action == "PAUSE":
            session.status = "paused"
        elif action == "PLAY":
            session.status = "streaming"
        elif action == "STOP":
            session.status = "stopped"


async def _lyria_worker_async(session):
    ws_url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateMusic"
        f"?key={quote(session.api_key, safe='')}"
    )
    async with websockets.connect(ws_url, max_size=None) as ws:
        session.ws = ws
        await ws.send(json.dumps({"setup": {"model": f"models/{LYRIA_REALTIME_MODEL}"}}))
        # Wait for setup confirmation before sending any other message.
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            msg = json.loads(raw)
            if msg.get("setupComplete") is not None:
                break
            if msg.get("error"):
                raise RuntimeError(str(msg.get("error")))

        await ws.send(json.dumps({"clientContent": {"weightedPrompts": session.prompts}}))
        if session.config:
            await ws.send(json.dumps({"musicGenerationConfig": session.config}))
        await ws.send(json.dumps({"playbackControl": "PLAY"}))
        session.status = "streaming"

        recv_task = asyncio.ensure_future(_lyria_receive_loop(session, ws))
        while not session.stop_event.is_set():
            try:
                cmd = session.cmd_queue.get_nowait()
            except _queue.Empty:
                cmd = None
            if cmd is not None:
                try:
                    await _lyria_handle_command(session, ws, cmd)
                except Exception:
                    logger.exception("Lyria RealTime command error")
            if time.time() - session.started_at > LYRIA_MAX_SESSION_SECONDS:
                session.error = "最大セッション時間（15分）に達したため自動停止しました。"
                session.status = "stopped"
                session.stop_event.set()
                break
            if recv_task.done():
                break
            await asyncio.sleep(0.05)
        recv_task.cancel()
        try:
            await recv_task
        except Exception:
            pass
        # Best-effort graceful stop so the model finalizes the stream.
        try:
            await _lyria_send_control(session, "STOP")
        except Exception:
            pass


def _lyria_worker(session):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session.loop = loop
    try:
        loop.run_until_complete(_lyria_worker_async(session))
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        session.error = str(exc)
        session.status = "error"
        logger.exception("Lyria RealTime session error")
    finally:
        if session.status not in ("error", "stopped", "paused"):
            session.status = "closed"
        with session.pending_cond:
            session.pending_cond.notify_all()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _lyria_purge_old_sessions():
    """Remove closed sessions that are past the TTL (run at session start)."""
    now = time.time()
    with LYRIA_SESSIONS_LOCK:
        stale = [
            sid for sid, sess in list(LYRIA_SESSIONS.items())
            if sess.status in ("closed", "error", "stopped")
            and (now - sess.started_at) > LYRIA_SESSION_TTL_SECONDS
        ]
        for sid in stale:
            LYRIA_SESSIONS.pop(sid, None)


def _lyria_get_session(session_id):
    session = LYRIA_SESSIONS.get(session_id or "")
    if not session or session.user_id != current_user.id:
        return None
    return session

