"""Gemini Live のセッションループ。

`server/realtime.py` から抽出（2026-11-12）。セッションの状態と開始処理は
`realtime.py`、OpenAI／xAI は `realtime_openai.py`、xAI のストリーミング
文字起こしは `realtime_xai_stt.py`。
"""

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
