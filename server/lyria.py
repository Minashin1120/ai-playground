# =============================================================================
# Lyria RealTime (lyria-realtime-exp) real-time music session manager
# -----------------------------------------------------------------------------
# Lyria RealTime is a WebSocket-based streaming music generation model. The
# browser cannot reach Google's BidiGenerateMusic WebSocket directly without
# exposing the user's raw Gemini API key, so the server keeps a persistent
# session (thread + asyncio loop) per active session, streams audio deltas to
# the client over SSE, and accepts steering commands over HTTP.
#
# Like the realtime speech sessions (server/realtime.py), the worker process
# that handled /start owns the Google WebSocket; the stream / command / save
# requests may reach another gunicorn worker and are relayed through Redis:
#   lyria:meta:<sid>  hash  user_id / status
#   lyria:in:<sid>    list  JSON commands ({"type": prompts|config|control|cancel|save})
#   lyria:ev:<sid>    list  JSON events for the SSE stream
#   lyria:res:<sid>   list  JSON result of the save request
# =============================================================================
LYRIA_REALTIME_MODEL = "lyria-realtime-exp"
LYRIA_SESSIONS = {}
LYRIA_SESSIONS_LOCK = threading.Lock()
LYRIA_MAX_SESSION_SECONDS = 15 * 60  # auto-stop to bound cost
LYRIA_MAX_AUDIO_BYTES = 512 * 1024 * 1024  # cap accumulated PCM
LYRIA_PROMPT_MAX_CHARS = 4000
LYRIA_SESSION_TTL_SECONDS = 30 * 60  # closed sessions are purged after this
LYRIA_BRIDGE_TTL_SECONDS = LYRIA_MAX_SESSION_SECONDS + LYRIA_SESSION_TTL_SECONDS
LYRIA_SAVE_WAIT_SECONDS = 45

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
        self.pump_thread = None
        self.bridged = False             # True: events/commands go through Redis
        self.e2ee = False
        self.cancelled = False
        self.saved = False


def _lyria_key(kind, session_id):
    return f"lyria:{kind}:{session_id}"


def _lyria_push_event(session, event):
    """Deliver one SSE event ({"audio"}, {"error"} or {"final"}) to the stream."""
    if session.bridged:
        try:
            key = _lyria_key("ev", session.session_id)
            pipe = redis_conn.pipeline()
            pipe.rpush(key, json.dumps(event))
            pipe.expire(key, LYRIA_BRIDGE_TTL_SECONDS)
            pipe.execute()
        except Exception as exc:
            logger.error(f"Lyria RealTime event publish failed: {exc}")
        return
    with session.pending_cond:
        session.pending.append(event)
        session.pending_cond.notify_all()


def _lyria_set_status(session, status):
    session.status = status
    if not session.bridged:
        return
    try:
        key = _lyria_key("meta", session.session_id)
        pipe = redis_conn.pipeline()
        pipe.hset(key, "status", status)
        pipe.expire(key, LYRIA_BRIDGE_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.error(f"Lyria RealTime status publish failed: {exc}")


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
                    _lyria_push_event(session, {"audio": data})
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
            _lyria_set_status(session, "paused")
        elif action == "PLAY":
            _lyria_set_status(session, "streaming")
        elif action == "STOP":
            _lyria_set_status(session, "stopped")


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
        _lyria_set_status(session, "streaming")

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
        session.stop_event.set()
        _lyria_set_status(session, session.status)
        if session.status == "error":
            _lyria_push_event(session, {"error": session.error or "Unknown error"})
        else:
            _lyria_push_event(session, {"final": True, "status": session.status})
        with session.pending_cond:
            session.pending_cond.notify_all()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _lyria_purge_old_sessions():
    """Stop sessions owned by this worker that outlived the bridge TTL."""
    now = time.time()
    with LYRIA_SESSIONS_LOCK:
        stale = [
            sid for sid, sess in list(LYRIA_SESSIONS.items())
            if (now - sess.started_at) > LYRIA_BRIDGE_TTL_SECONDS
        ]
        for sid in stale:
            sess = LYRIA_SESSIONS.pop(sid, None)
            if sess:
                sess.cancelled = True
                sess.stop_event.set()


def _lyria_get_session(session_id):
    """Bridge metadata of a Lyria session owned by the current user (any worker)."""
    session_id = str(session_id or "")
    if not session_id.startswith("lyria_"):
        return None
    try:
        raw = redis_conn.hgetall(_lyria_key("meta", session_id)) or {}
    except Exception as exc:
        logger.error(f"Lyria RealTime session lookup failed: {exc}")
        return None
    meta = {
        (k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else str(v))
        for k, v in raw.items()
    }
    if not meta or meta.get("user_id") != str(current_user.id):
        return None
    meta["session_id"] = session_id
    return meta


def _lyria_send_command(session_id, command):
    key = _lyria_key("in", session_id)
    pipe = redis_conn.pipeline()
    pipe.rpush(key, json.dumps(command))
    pipe.expire(key, LYRIA_BRIDGE_TTL_SECONDS)
    pipe.execute()


def _lyria_save_session(session, thread_id):
    """Store the recording and the prompt/answer messages (owner worker, app context)."""
    with session.audio_lock:
        pcm_bytes = bytes(session.audio_buffer)
    if len(pcm_bytes) < 1024:
        return {'error': 'オーディオデータがありません。再生を少し進めてから保存してください。'}, 400

    wav_bytes = _lyria_pcm_to_wav_stereo(pcm_bytes, rate=48000)
    try:
        fname, audio_url = _save_user_generated_bytes_verified(
            session.user_id,
            wav_bytes,
            lambda: f"lyria_realtime_{int(time.time())}_{os.urandom(4).hex()}.wav",
            session.e2ee,
        )
    except Exception as exc:
        logger.exception("Lyria RealTime save error")
        return {'error': f'保存に失敗しました: {exc}'}, 500

    try:
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

        prompt_lines = []
        for p in session.prompts:
            weight = float(p.get('weight', 1.0))
            prompt_lines.append(f"{p.get('text', '')} (weight: {weight})")
        prompt_text = "\n".join(prompt_lines) if prompt_lines else "Lyria RealTime 生成"
        audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
        assistant_content = f"**Lyria RealTime 生成**\n\n{audio_tag}"
        if session.filtered_prompt:
            assistant_content += f"\n\n*プロンプトが安全フィルターにより調整されました。*"

        u_content = encrypt_val(prompt_text) if session.e2ee else prompt_text
        a_content = encrypt_val(assistant_content) if session.e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(prompt_text, LYRIA_REALTIME_MODEL)
        assistant_tokens_out = count_tokens_for_display("Lyria RealTime 生成", LYRIA_REALTIME_MODEL)

        parent_id = None
        last_msg = Message.query.filter_by(thread_id=thread_db_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_db_id,
            role='user',
            content=u_content,
            is_encrypted=session.e2ee,
            parent_id=parent_id,
            model=LYRIA_REALTIME_MODEL,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None),
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_db_id,
            role='assistant',
            content=a_content,
            model=LYRIA_REALTIME_MODEL,
            is_encrypted=session.e2ee,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out),
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as exc:
        logger.exception("Lyria RealTime message save error")
        try:
            db.session.rollback()
        except Exception:
            pass
        return {'error': f'音声は保存されましたが、メッセージ保存に失敗しました: {exc}', 'audio_url': audio_url}, 500
    return {'status': 'ok', 'audio_url': audio_url, 'thread_id': str(thread_db_id)}, 200


def _lyria_finish_and_save(session, thread_id):
    session.stop_event.set()
    if session.thread:
        session.thread.join(timeout=5)
    with app.app_context():
        try:
            return _lyria_save_session(session, thread_id)
        finally:
            try:
                db.session.remove()
            except Exception:
                pass


def _lyria_input_pump(session):
    """Owner-side loop: moves Redis commands into the Lyria session."""
    in_key = _lyria_key("in", session.session_id)
    idle_deadline = None
    try:
        while True:
            now = time.time()
            if session.stop_event.is_set():
                # Session ended on its own: keep the audio for a while so the
                # client can still save it.
                if idle_deadline is None:
                    idle_deadline = now + LYRIA_SESSION_TTL_SECONDS
                elif now > idle_deadline:
                    break
            if now - session.started_at > LYRIA_BRIDGE_TTL_SECONDS or session.cancelled:
                break
            try:
                item = redis_conn.blpop([in_key], timeout=1)
            except Exception as exc:
                logger.error(f"Lyria RealTime input pump error: {exc}")
                time.sleep(1)
                continue
            if not item:
                continue
            try:
                cmd = json.loads(item[1])
            except Exception:
                continue
            ctype = cmd.get("type")
            if ctype in ("prompts", "config", "control"):
                if not session.stop_event.is_set():
                    session.cmd_queue.put(cmd)
            elif ctype == "cancel":
                session.cancelled = True
                session.stop_event.set()
                break
            elif ctype == "save":
                try:
                    result, status = _lyria_finish_and_save(session, cmd.get("thread_id"))
                except Exception as exc:
                    logger.exception("Lyria RealTime save failed")
                    result, status = {'error': f'保存に失敗しました: {exc}'}, 500
                result["_status"] = status
                session.saved = True
                res_key = _lyria_key("res", session.session_id)
                pipe = redis_conn.pipeline()
                pipe.rpush(res_key, json.dumps(result, ensure_ascii=False))
                pipe.expire(res_key, 120)
                pipe.execute()
                break
    finally:
        session.stop_event.set()
        if session.thread and session.thread is not threading.current_thread():
            session.thread.join(timeout=5)
        with LYRIA_SESSIONS_LOCK:
            LYRIA_SESSIONS.pop(session.session_id, None)
        keys = [_lyria_key("meta", session.session_id), _lyria_key("in", session.session_id),
                _lyria_key("ev", session.session_id)]
        if not session.saved:
            keys.append(_lyria_key("res", session.session_id))
        try:
            redis_conn.delete(*keys)
        except Exception as exc:
            logger.error(f"Lyria RealTime bridge cleanup failed: {exc}")


def _lyria_start_bridged_session(session, e2ee):
    """Register the session in Redis and start the Lyria + command threads."""
    session.bridged = True
    session.e2ee = bool(e2ee)
    meta_key = _lyria_key("meta", session.session_id)
    pipe = redis_conn.pipeline()
    pipe.hset(meta_key, mapping={
        "user_id": str(session.user_id),
        "status": session.status,
        "owner": str(os.getpid()),
    })
    pipe.expire(meta_key, LYRIA_BRIDGE_TTL_SECONDS)
    pipe.execute()
    with LYRIA_SESSIONS_LOCK:
        LYRIA_SESSIONS[session.session_id] = session
    session.thread = threading.Thread(target=_lyria_worker, args=(session,), daemon=True, name=f"lyria-{session.session_id}")
    session.pump_thread = threading.Thread(target=_lyria_input_pump, args=(session,), daemon=True, name=f"lyria-in-{session.session_id}")
    session.thread.start()
    session.pump_thread.start()

