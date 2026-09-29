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

    session_id = f"lyria_{int(time.time())}_{secrets.token_hex(8)}"
    session = LyriaSession(session_id, current_user.id, key, prompts, config)
    try:
        # Later requests may reach another gunicorn worker; they talk to this
        # worker's session through Redis.
        _lyria_start_bridged_session(session, current_user.enable_e2ee)
    except Exception as exc:
        logger.error(f"Lyria RealTime session start failed: {exc}")
        with LYRIA_SESSIONS_LOCK:
            LYRIA_SESSIONS.pop(session_id, None)
        return jsonify({'error': 'Lyria RealTimeのセッションを開始できませんでした'}), 503
    return jsonify({'session_id': session_id})


@app.route('/api/gemini/music/stream')
@login_required
def gemini_music_stream():
    meta = _lyria_get_session(request.args.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    session_id = meta['session_id']
    ev_key = _lyria_key("ev", session_id)
    meta_key = _lyria_key("meta", session_id)

    def generate():
        try:
            # Status only: the server keeps the recording for saving, so a
            # reconnecting client just resumes the live deltas.
            yield f"data: {json.dumps({'snapshot': True, 'status': meta.get('status') or 'connecting'})}\n\n"
            deadline = time.time() + LYRIA_BRIDGE_TTL_SECONDS
            last_sent = time.time()
            while time.time() < deadline:
                item = redis_conn.blpop([ev_key], timeout=1)
                if not item:
                    if not redis_conn.exists(meta_key):
                        yield f"data: {json.dumps({'final': True, 'status': 'closed'})}\n\n"
                        break
                    if time.time() - last_sent > 15:
                        last_sent = time.time()
                        yield ": keep-alive\n\n"
                    continue
                raw = item[1]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                yield f"data: {raw}\n\n"
                last_sent = time.time()
                try:
                    ev = json.loads(raw)
                except Exception:
                    ev = {}
                if ev.get("final") or ev.get("error"):
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
    meta = _lyria_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    if meta.get('status') in ("closed", "error"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    ctype = str(data.get('type') or '')
    if ctype not in ("prompts", "config", "control"):
        return jsonify({'error': 'Invalid command type'}), 400
    command = {'type': ctype}
    for field in ('weighted_prompts', 'config', 'reset_context', 'action'):
        if field in data:
            command[field] = data.get(field)
    _lyria_send_command(meta['session_id'], command)
    return jsonify({'status': 'ok'})


@app.route('/api/gemini/music/cancel', methods=['POST'])
@login_required
def gemini_music_cancel():
    data = request.get_json(silent=True) or {}
    meta = _lyria_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    _lyria_send_command(meta['session_id'], {'type': 'cancel'})
    return jsonify({'status': 'ok'})


@app.route('/api/gemini/music/save', methods=['POST'])
@login_required
def gemini_music_save():
    data = request.get_json(silent=True) or {}
    meta = _lyria_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    session_id = meta['session_id']
    thread_id = data.get('thread_id')
    # The owning worker holds the audio: it stops the stream, stores the
    # recording and messages, and hands the result back through Redis.
    _lyria_send_command(session_id, {'type': 'save', 'thread_id': str(thread_id) if thread_id else None})
    item = redis_conn.blpop([_lyria_key("res", session_id)], timeout=LYRIA_SAVE_WAIT_SECONDS)
    if not item:
        return jsonify({'error': '保存がタイムアウトしました。しばらくしてからもう一度お試しください。'}), 504
    try:
        result = json.loads(item[1])
    except Exception:
        return jsonify({'error': '保存結果を読み取れませんでした'}), 500
    status = int(result.pop('_status', 200) or 200)
    redis_conn.delete(_lyria_key("res", session_id))
    return jsonify(result), status


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
    session_id = f"rt_{int(time.time())}_{secrets.token_hex(8)}"
    session = RtSession(session_id, current_user.id, model_key, key, params)
    try:
        # The other requests of this session may be served by another gunicorn
        # worker; they reach this worker's provider session through Redis.
        _rt_start_bridged_session(session, current_user.enable_e2ee)
    except Exception as exc:
        logger.error(f"Realtime STS session start failed: {exc}")
        with RT_SESSIONS_LOCK:
            RT_SESSIONS.pop(session_id, None)
        return jsonify({'error': 'リアルタイムセッションを開始できませんでした'}), 503
    return jsonify({
        'session_id': session_id,
        'rate_in': session.rate_in,
        'rate_out': session.rate_out,
        'provider': provider,
    })


@app.route('/api/realtime/stream')
@login_required
def realtime_stream():
    meta = _rt_get_session(request.args.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    session_id = meta['session_id']
    ev_key = _rt_key("ev", session_id)
    meta_key = _rt_key("meta", session_id)

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'status': meta.get('status') or 'connecting'}, ensure_ascii=False)}\n\n"
            deadline = time.time() + RT_BRIDGE_TTL_SECONDS
            last_sent = time.time()
            while time.time() < deadline:
                item = redis_conn.blpop([ev_key], timeout=1)
                if not item:
                    if not redis_conn.exists(meta_key):
                        # Session saved / cancelled / expired elsewhere.
                        yield f"data: {json.dumps({'type': 'final', 'status': 'closed'}, ensure_ascii=False)}\n\n"
                        break
                    if time.time() - last_sent > 15:
                        last_sent = time.time()
                        yield ": keep-alive\n\n"
                    continue
                raw = item[1]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                yield f"data: {raw}\n\n"
                last_sent = time.time()
                try:
                    ev_type = json.loads(raw).get("type")
                except Exception:
                    ev_type = None
                if ev_type == "final":
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
    meta = _rt_get_session(request.args.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    if meta.get('status') in ("closed", "error", "stopped"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    data = request.get_data(cache=False)
    if not data or len(data) > RT_AUDIO_POST_MAX:
        return jsonify({'error': 'Invalid audio payload'}), 400
    _rt_send_command(meta['session_id'], b"A" + data)
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/commit', methods=['POST'])
@login_required
def realtime_commit():
    data = request.get_json(silent=True) or {}
    meta = _rt_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    if meta.get('status') in ("closed", "error", "stopped"):
        return jsonify({'error': 'セッションは終了しています'}), 400
    _rt_send_command(meta['session_id'], b"C")
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/cancel', methods=['POST'])
@login_required
def realtime_cancel():
    data = request.get_json(silent=True) or {}
    meta = _rt_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    _rt_send_command(meta['session_id'], b"X")
    return jsonify({'status': 'ok'})


@app.route('/api/realtime/save', methods=['POST'])
@login_required
def realtime_save():
    data = request.get_json(silent=True) or {}
    meta = _rt_get_session(data.get('session_id'))
    if not meta:
        return jsonify({'error': 'Session not found'}), 404
    session_id = meta['session_id']
    thread_id = data.get('thread_id')
    request_payload = json.dumps({'thread_id': str(thread_id) if thread_id else None})
    # The owning worker holds the audio: it finishes the provider session,
    # stores the messages and hands the result back through Redis.
    _rt_send_command(session_id, b"F" + request_payload.encode("utf-8"))
    item = redis_conn.blpop([_rt_key("res", session_id)], timeout=RT_SAVE_WAIT_SECONDS)
    if not item:
        return jsonify({'error': '保存がタイムアウトしました。しばらくしてからもう一度お試しください。'}), 504
    try:
        result = json.loads(item[1])
    except Exception:
        return jsonify({'error': '保存結果を読み取れませんでした'}), 500
    status = int(result.pop('_status', 200) or 200)
    redis_conn.delete(_rt_key("res", session_id))
    return jsonify(result), status


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
    is_gemini_38_extended = (model_key == "gemini-3.8-live-extended-thinking")

    if is_live_transcribe:
        # Live Transcription: TEXT output. The transcription config
        # (language_codes / custom_vocabulary / mode) is sent by the client in
        # the WebSocket setup message; the installed SDK (v1.56.0) has no such
        # fields, so inputAudioTranscription stays unlocked (see below).
        generation_config = {
            'response_modalities': ['TEXT'],
        }
    else:
        generation_config = {
            'response_modalities': ['AUDIO'],
            'input_audio_transcription': {},
            'output_audio_transcription': {},
        }
        if not is_live_translate and voice and voice in GEMINI_STS_VOICES:
            generation_config['speech_config'] = {
                'voice_config': {
                    'prebuilt_voice_config': {'voice_name': voice}
                }
            }
        if is_gemini_38_extended:
            if thinking_level not in {'low', 'medium', 'high'}:
                thinking_level = 'medium'
            generation_config['thinking_config'] = {
                'thinking_level': thinking_level,
                'include_thoughts': include_thoughts
            }
        elif (
            model_key not in GEMINI_LIVE_NO_THINKING_LEVEL_MODELS
            and thinking_level in {'minimal', 'low', 'medium', 'high'}
        ):
            # Gemini 3.1 Flash Live. Gemini 3.8 Live, 2.5 native audio and
            # Live Translate do not accept thinking_level.
            generation_config['thinking_config'] = {
                'thinking_level': thinking_level,
                'include_thoughts': include_thoughts
            }

    config = {
        'live_connect_constraints': {
            'model': f'models/{model_key}',
            'config': generation_config
        }
    }
    if is_live_translate or is_live_transcribe:
        # Lock only the fields set above so the client can add
        # generationConfig.translationConfig (target language chosen by the
        # user) or the detailed inputAudioTranscription.  Without this the
        # whole setup is locked and those client fields are not applied.
        config['lock_additional_fields'] = []

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
