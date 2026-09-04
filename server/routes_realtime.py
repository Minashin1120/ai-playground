# -----------------------------------------------------------------------------
# Lyria RealTime studio API
# -----------------------------------------------------------------------------
@app.route('/api/gemini/music/start', methods=['POST'])
@login_required
def gemini_music_start():
    _lyria_purge_old_sessions()
    data = request.get_json(silent=True) or {}
    gemini_runtime = _resolve_gemini_runtime(current_user)
    if gemini_runtime.get("backend") == "vertex_ai":
        return jsonify({'error': 'Lyria RealTimeはVertex AIでは利用できません。Gemini APIキーを使用してください。'}), 400
    model_specific_key = _get_model_specific_api_key(current_user, LYRIA_REALTIME_MODEL)
    key = model_specific_key or gemini_runtime.get("api_key")
    if not key:
        return jsonify({'error': 'Gemini API Key not configured'}), 400

    prompts = _normalize_lyria_prompts(data.get("weighted_prompts"))
    if not prompts:
        return jsonify({'error': 'プロンプトを入力してください'}), 400
    config = _normalize_lyria_config(data.get("config"))

    session_id = f"lyria_{int(time.time())}_{secrets.token_hex(4)}"
    session = LyriaSession(session_id, current_user.id, key, prompts, config)
    thread = threading.Thread(target=_lyria_worker, args=(session,), daemon=True, name=f"lyria-{session_id}")
    session.thread = thread
    with LYRIA_SESSIONS_LOCK:
        LYRIA_SESSIONS[session_id] = session
    thread.start()
    return jsonify({'session_id': session_id})


@app.route('/api/gemini/music/stream')
@login_required
def gemini_music_stream():
    session = _lyria_get_session(request.args.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    def generate():
        try:
            # Snapshot the audio accumulated so far so a reconnecting client can
            # reconstruct the full recording, then stream live deltas only.
            with session.audio_lock:
                snapshot_bytes = bytes(session.audio_buffer)
            with session.pending_cond:
                session.pending.clear()
            snapshot_b64 = base64.b64encode(snapshot_bytes).decode('ascii')
            yield f"data: {json.dumps({'snapshot': snapshot_b64, 'status': session.status})}\n\n"
            while True:
                with session.pending_cond:
                    while not session.pending and not session.stop_event.is_set() and session.status != "error":
                        session.pending_cond.wait(timeout=1.0)
                    pending = list(session.pending)
                    session.pending.clear()
                for delta in pending:
                    yield f"data: {json.dumps({'audio': delta})}\n\n"
                if session.status == "error":
                    yield f"data: {json.dumps({'error': session.error or 'Unknown error'})}\n\n"
                    break
                if session.stop_event.is_set() and not session.pending:
                    yield f"data: {json.dumps({'final': True, 'status': session.status})}\n\n"
                    break
        except GeneratorExit:
            pass
        except Exception:
            logger.exception("Lyria RealTime stream error")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/gemini/music/command', methods=['POST'])
@login_required
def gemini_music_command():
    data = request.get_json(silent=True) or {}
    session = _lyria_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.status in ("closed", "error"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    ctype = str(data.get('type') or '')
    if ctype not in ("prompts", "config", "control"):
        return jsonify({'error': 'Invalid command type'}), 400
    session.cmd_queue.put(data)
    return jsonify({'status': 'ok'})


@app.route('/api/gemini/music/cancel', methods=['POST'])
@login_required
def gemini_music_cancel():
    data = request.get_json(silent=True) or {}
    session = _lyria_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    session.stop_event.set()
    try:
        if session.loop and session.ws:
            asyncio.run_coroutine_threadsafe(_lyria_send_control(session, "STOP"), session.loop).result(timeout=3)
    except Exception:
        pass
    if session.thread:
        session.thread.join(timeout=3)
    with LYRIA_SESSIONS_LOCK:
        LYRIA_SESSIONS.pop(session.session_id, None)
    return jsonify({'status': 'ok'})


@app.route('/api/gemini/music/save', methods=['POST'])
@login_required
def gemini_music_save():
    data = request.get_json(silent=True) or {}
    session = _lyria_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    session.stop_event.set()
    try:
        if session.loop and session.ws:
            asyncio.run_coroutine_threadsafe(_lyria_send_control(session, "STOP"), session.loop).result(timeout=3)
    except Exception:
        pass
    if session.thread:
        session.thread.join(timeout=5)

    with session.audio_lock:
        pcm_bytes = bytes(session.audio_buffer)
    if len(pcm_bytes) < 1024:
        with LYRIA_SESSIONS_LOCK:
            LYRIA_SESSIONS.pop(session.session_id, None)
        return jsonify({'error': 'オーディオデータがありません。再生を少し進めてから保存してください。'}), 400

    wav_bytes = _lyria_pcm_to_wav_stereo(pcm_bytes, rate=48000)
    try:
        fname, audio_url = _save_user_generated_bytes_verified(
            current_user.id,
            wav_bytes,
            lambda: f"lyria_realtime_{int(time.time())}_{os.urandom(4).hex()}.wav",
            current_user.enable_e2ee,
        )
    except Exception as exc:
        logger.exception("Lyria RealTime save error")
        with LYRIA_SESSIONS_LOCK:
            LYRIA_SESSIONS.pop(session.session_id, None)
        return jsonify({'error': f'保存に失敗しました: {exc}'}), 500

    try:
        thread_id = data.get('thread_id')
        t = resolve_thread_for_user(thread_id, current_user.id) if thread_id else None
        if not t:
            t = Thread(
                user_id=current_user.id,
                public_id=generate_thread_public_id(),
                is_temporary=True,
            )
            db.session.add(t)
            safe_db_commit()
            thread_id = t.id
        else:
            thread_id = t.id

        prompt_lines = []
        for p in session.prompts:
            weight = float(p.get('weight', 1.0))
            prompt_lines.append(f"{p.get('text', '')} (weight: {weight})")
        prompt_text = "\n".join(prompt_lines) if prompt_lines else "Lyria RealTime 生成"
        audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
        assistant_content = f"**Lyria RealTime 生成**\n\n{audio_tag}"
        if session.filtered_prompt:
            assistant_content += f"\n\n*プロンプトが安全フィルターにより調整されました。*"

        u_content = encrypt_val(prompt_text) if current_user.enable_e2ee else prompt_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(prompt_text, LYRIA_REALTIME_MODEL)
        assistant_tokens_out = count_tokens_for_display("Lyria RealTime 生成", LYRIA_REALTIME_MODEL)

        parent_id = None
        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_id,
            role='user',
            content=u_content,
            is_encrypted=current_user.enable_e2ee,
            parent_id=parent_id,
            model=LYRIA_REALTIME_MODEL,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None),
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_id,
            role='assistant',
            content=a_content,
            model=LYRIA_REALTIME_MODEL,
            is_encrypted=current_user.enable_e2ee,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out),
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as exc:
        logger.exception("Lyria RealTime message save error")
        with LYRIA_SESSIONS_LOCK:
            LYRIA_SESSIONS.pop(session.session_id, None)
        return jsonify({'error': f'音声は保存されましたが、メッセージ保存に失敗しました: {exc}', 'audio_url': audio_url}), 500

    with LYRIA_SESSIONS_LOCK:
        LYRIA_SESSIONS.pop(session.session_id, None)
    return jsonify({'status': 'ok', 'audio_url': audio_url, 'thread_id': str(thread_id)})


# -----------------------------------------------------------------------------
# True real-time STS session API (OpenAI Realtime / Grok Voice / Gemini native-audio)
# -----------------------------------------------------------------------------
@app.route('/api/realtime/start', methods=['POST'])
@login_required
def realtime_start():
    _rt_purge_old_sessions()
    data = request.get_json(silent=True) or {}
    model_key = (data.get('model') or "").strip()
    if not _rt_is_conversation_model(model_key):
        return jsonify({'error': 'このモデルはリアルタイム会話セッションに対応していません'}), 400
    model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
    provider = get_sts_provider(model_key)

    key, _ = _rt_resolve_api_key(current_user, model_key, provider)
    if not key:
        labels = {"google": "Gemini", "openai": "OpenAI", "xai": "xAI"}
        return jsonify({'error': f'{labels.get(provider, "API")} API Key not configured'}), 400

    params = _normalize_rt_params(provider, model_key, data)
    session_id = f"rt_{int(time.time())}_{secrets.token_hex(4)}"
    session = RtSession(session_id, current_user.id, model_key, key, params)
    thread = threading.Thread(target=_rt_worker, args=(session,), daemon=True, name=f"rt-{session_id}")
    session.thread = thread
    with RT_SESSIONS_LOCK:
        RT_SESSIONS[session_id] = session
    thread.start()
    return jsonify({
        'session_id': session_id,
        'rate_in': session.rate_in,
        'rate_out': session.rate_out,
        'provider': provider,
    })


@app.route('/api/realtime/stream')
@login_required
def realtime_stream():
    session = _rt_get_session(request.args.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    def generate():
        try:
            with session.pending_cond:
                pending = list(session.pending)
                session.pending.clear()
            if pending:
                for ev in pending:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'status': session.status}, ensure_ascii=False)}\n\n"
            while True:
                with session.pending_cond:
                    while not session.pending and not session.stop_event.is_set() and session.status != "error":
                        session.pending_cond.wait(timeout=1.0)
                    events = list(session.pending)
                    session.pending.clear()
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if session.status == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': session.error or 'Unknown error'}, ensure_ascii=False)}\n\n"
                    break
                if session.stop_event.is_set() and not session.pending:
                    yield f"data: {json.dumps({'type': 'final', 'status': session.status}, ensure_ascii=False)}\n\n"
                    break
        except GeneratorExit:
            pass
        except Exception:
            logger.exception("Realtime STS stream error")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/realtime/audio', methods=['POST'])
@login_required
def realtime_audio():
    session = _rt_get_session(request.args.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.status in ("closed", "error", "stopped"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    data = request.get_data(cache=False)
    if not data or len(data) > RT_AUDIO_POST_MAX:
        return jsonify({'error': 'Invalid audio payload'}), 400
    with session.user_lock:
        if len(session.user_audio) + len(data) <= RT_PCM_CAP:
            session.user_audio += data
    session.audio_in.put(("audio", data))
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/commit', methods=['POST'])
@login_required
def realtime_commit():
    data = request.get_json(silent=True) or {}
    session = _rt_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.status in ("closed", "error", "stopped"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    session.audio_in.put(("commit",))
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/cancel', methods=['POST'])
@login_required
def realtime_cancel():
    data = request.get_json(silent=True) or {}
    session = _rt_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    session.stop_event.set()
    if session.thread:
        session.thread.join(timeout=3)
    with RT_SESSIONS_LOCK:
        RT_SESSIONS.pop(session.session_id, None)
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/save', methods=['POST'])
@login_required
def realtime_save():
    data = request.get_json(silent=True) or {}
    session = _rt_get_session(data.get('session_id'))
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    session.stop_event.set()
    if session.thread:
        session.thread.join(timeout=6)

    with session.assistant_lock:
        assistant_pcm = bytes(session.assistant_audio)
    with session.user_lock:
        user_pcm = bytes(session.user_audio)

    audio_url = None
    in_fname = None
    try:
        if len(assistant_pcm) >= 1024:
            wav_bytes = _pcm_to_wav_bytes(assistant_pcm, rate=session.rate_out)
            out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
            audio_url = f"/files/{current_user.id}/{out_fname}"
        if len(user_pcm) >= 1024:
            u_wav = _pcm_to_wav_bytes(user_pcm, rate=session.rate_in)
            in_fname, _ = _save_user_audio(current_user.id, u_wav, ".wav", current_user.enable_e2ee)
    except Exception as exc:
        logger.exception("Realtime STS audio save error")

    user_text = (session.user_transcript or "音声メッセージ").strip()
    assistant_text = (session.assistant_transcript or "").strip()
    assistant_thought = (session.assistant_thought or "").strip()

    # Nothing was captured — drop the empty session without saving a message.
    if (not assistant_pcm and not user_pcm
            and not user_text.strip() and not assistant_text.strip()):
        with RT_SESSIONS_LOCK:
            RT_SESSIONS.pop(session.session_id, None)
        return jsonify({'status': 'empty'})

    thread_id = data.get('thread_id')
    t = resolve_thread_for_user(thread_id, current_user.id) if thread_id else None
    if not t:
        t = Thread(
            user_id=current_user.id,
            public_id=generate_thread_public_id(),
            is_temporary=True,
        )
        db.session.add(t)
        safe_db_commit()
        thread_id = t.id
    else:
        thread_id = t.id

    thought_tag = f"<thought>\n{assistant_thought}\n</thought>\n" if assistant_thought else ""
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n' if audio_url else ""
    assistant_content = thought_tag + (assistant_text + "\n" if assistant_text else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(user_text, session.model_key)
        assistant_tokens_out = count_tokens_for_display(assistant_text, session.model_key)
        if assistant_thought:
            assistant_tokens_out += count_tokens_for_display(assistant_thought, session.model_key)

        parent_id = None
        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_id,
            role='user',
            content=u_content,
            image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
            is_encrypted=current_user.enable_e2ee,
            parent_id=parent_id,
            model=session.model_key,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None),
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_id,
            role='assistant',
            content=a_content,
            model=session.model_key,
            is_encrypted=current_user.enable_e2ee,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out),
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as exc:
        logger.exception("Realtime STS message save error")
        with RT_SESSIONS_LOCK:
            RT_SESSIONS.pop(session.session_id, None)
        return jsonify({'error': f'メッセージ保存に失敗しました: {exc}', 'audio_url': audio_url}), 500

    with RT_SESSIONS_LOCK:
        RT_SESSIONS.pop(session.session_id, None)
    return jsonify({'status': 'ok', 'audio_url': audio_url, 'thread_id': str(thread_id)})


@app.route('/api/gemini/session', methods=['POST'])
@login_required
def gemini_session():
    data = request.get_json(silent=True) or {}
    model_key = (data.get('model') or "gemini-3.1-flash-live-preview").strip()
    if model_key not in STS_MODELS or get_sts_provider(model_key) != 'google':
        return jsonify({'error': 'Invalid Gemini Live model'}), 400
    
    # Resolve API key and runtime
    gemini_runtime = _resolve_gemini_runtime(current_user)
    model_specific_key = _get_model_specific_api_key(current_user, model_key)
    key = model_specific_key or gemini_runtime.get("api_key")
    
    if not key:
        return jsonify({'error': 'Gemini API Key not configured'}), 400
        
    # Use v1alpha for token creation as seen in test_genai_token.py
    client = _get_gemini_client(
        api_key=key,
        backend=gemini_runtime.get("backend"),
        vertex_project=gemini_runtime.get("vertex_project"),
        vertex_location=gemini_runtime.get("vertex_location"),
        vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
        api_version='v1alpha'
    )
    
    if not client:
        return jsonify({'error': 'Gemini client not configured'}), 400

    # Thinking configuration and other setup
    thinking_level = data.get('thinking_level') or 'minimal'
    include_thoughts = data.get('include_thoughts') is True
    voice = (data.get('voice') or "Kore").strip()
    is_live_translate = (model_key == "gemini-3.5-live-translate-preview")
    is_live_transcribe = (model_key == "gemini-3.5-transcribe-live")

    if is_live_transcribe:
        # Live Transcription: TEXT output. The detailed transcription config
        # (language_codes / custom_vocabulary / mode) is sent by the client in
        # the WebSocket setup message; the installed SDK (v1.56.0) rejects those
        # fields inside the ephemeral token config, so pass an empty object here.
        generation_config = {
            'response_modalities': ['TEXT'],
            'input_audio_transcription': {},
        }
    else:
        generation_config = {
            'response_modalities': ['AUDIO'],
        }
        if not is_live_translate and voice and voice in GEMINI_STS_VOICES:
            generation_config['speech_config'] = {
                'voice_config': {
                    'prebuilt_voice_config': {'voice_name': voice}
                }
            }
        if not is_live_translate and thinking_level:
            generation_config['thinking_config'] = {
                'thinking_level': thinking_level,
                'include_thoughts': include_thoughts
            }
        if is_live_translate:
            # The installed SDK rejects `translation_config` inside the ephemeral
            # token config (no such field in LiveConnectConfig). The client sends
            # `translationConfig` in the WebSocket setup message instead, so we
            # intentionally omit it here to avoid a token-creation 400/422 error.
            pass

    config = {
        'live_connect_constraints': {
            'model': f'models/{model_key}',
            'config': generation_config
        }
    }
    
    try:
        # Note: auth_tokens.create is experimental in the SDK
        token = client.auth_tokens.create(config=config)
        return jsonify({
            'token': token.name,
            'url': 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained'
        })
    except Exception as e:
        logger.error(f"Failed to create Gemini session token: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini/save_sts', methods=['POST'])
@login_required
def save_sts_direct():
    data = request.get_json(silent=True) or {}
    thread_id = data.get('thread_id')
    model_key = data.get('model')
    user_text = data.get('user_text')
    assistant_text = data.get('assistant_text')
    assistant_thought = data.get('assistant_thought')
    audio_base64 = data.get('audio_base64') # Assistant audio
    user_audio_base64 = data.get('user_audio_base64') # User audio recorded by client
    
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    if model_key not in STS_MODELS or get_sts_provider(model_key) != 'google':
        return jsonify({'error': 'Invalid Gemini Live model'}), 400
    thread_db_id = t.id
    for text_value in (user_text, assistant_text, assistant_thought):
        if text_value is not None and len(str(text_value)) > 500_000:
            return jsonify({'error': 'Transcript is too large'}), 413

    # Save Assistant Audio (Gemini returns PCM 24kHz)
    audio_url = None
    if audio_base64:
        try:
            audio_data = _decode_base64_limited(audio_base64, _AUDIO_INPUT_MAX_BYTES)
            wav_bytes = _pcm_to_wav_bytes(audio_data, rate=24000)
            out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
            audio_url = f"/files/{current_user.id}/{out_fname}"
        except Exception as e:
            logger.error(f"Failed to save assistant audio: {e}")

    # Save User Audio (Client sends WebM/Opus)
    in_fname = None
    if user_audio_base64:
        try:
            user_audio_data = _decode_base64_limited(user_audio_base64, _AUDIO_INPUT_MAX_BYTES)
            in_fname, _ = _save_user_audio(current_user.id, user_audio_data, ".webm", current_user.enable_e2ee)
        except Exception as e:
            logger.error(f"Failed to save user audio: {e}")

    user_text = (user_text or "Voice message").strip()
    assistant_text_clean = (assistant_text or "").strip()
    assistant_thought_clean = (assistant_thought or "").strip()
    
    thought_tag = f"<thought>\n{assistant_thought_clean}\n</thought>\n" if assistant_thought_clean else ""
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n' if audio_url else ""
    assistant_content = thought_tag + (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(user_text, model_key)
        assistant_tokens_out = count_tokens_for_display(assistant_text_clean, model_key)
        if assistant_thought_clean:
            assistant_tokens_out += count_tokens_for_display(assistant_thought_clean, model_key)
        
        parent_id = None
        last_msg = Message.query.filter_by(thread_id=thread_db_id).order_by(Message.id.desc()).first()
        if last_msg: parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_db_id,
            role='user',
            content=u_content,
            image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
            is_encrypted=current_user.enable_e2ee,
            parent_id=parent_id,
            model=model_key,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None)
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_db_id,
            role='assistant',
            content=a_content,
            model=model_key,
            is_encrypted=current_user.enable_e2ee,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out)
        )
        db.session.add(assistant_msg)
        safe_db_commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Failed to save STS message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/robots.txt')
def robots_txt():
    # Allow landing and auth pages, but disallow private/API paths
    lines = [
        "User-agent: *",
        "Disallow: /files/",
        "Disallow: /api/",
        "Disallow: /thread/",
        "Disallow: /chat/",
        "Allow: /login",
        "Allow: /signup",
        "Allow: /landing",
        "Allow: /help",
        "Allow: /changelog",
        "Allow: /"
    ]
    return Response("\n".join(lines), mimetype="text/plain")

def _add_file_privacy_headers(resp):
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    resp.headers["Vary"] = "Cookie"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp

# File extensions whose contents can contain active scripts (HTML / SVG) when
# rendered inline.  They are now uploadable (the create_file tool can produce
# them), but serving them as their native MIME would let a browser execute any
# embedded script, so /files/ forces them to download instead of rendering.
_FILE_FORCE_DOWNLOAD_EXTS = {'.html', '.htm', '.xhtml', '.svg'}

def _add_thumb_cache_headers(resp, etag=None):
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "private, max-age=86400, stale-while-revalidate=604800"
    resp.headers["Vary"] = "Cookie"
    if etag:
        resp.headers["ETag"] = f'"{etag}"'
    return resp

def _unreadable_file_http_response(filename, is_thumb):
    """Distinguish a file whose encryption key is unavailable (exists but cannot
    be decrypted) from a genuinely-missing file.  409 + JSON lets the frontend
    show a clear warning instead of a broken thumbnail."""
    resp = jsonify({
        "error": "encryption_key_mismatch",
        "message": "このファイルは暗号キーが一致しないため閲覧できません",
        "unreadable": True,
        "filename": filename,
        "thumbnail": bool(is_thumb),
    })
    resp.status_code = 409
    resp.headers["Cache-Control"] = "no-store"
    return resp


