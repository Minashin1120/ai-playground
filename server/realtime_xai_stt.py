"""xAI のストリーミング文字起こし（wss://api.x.ai/v1/stt）のセッションループ。

`server/realtime.py` から抽出（2026-11-12）。文字起こしのチャンク結合
（`_rt_join_transcript`）と送受信ループ・開始処理を持つ。`RtSession` と
`_rt_run_until_stopped` は `realtime.py` にある。
"""

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
