@app.route('/synthesize', methods=['POST'])
@login_required
def synthesize():
    data = request.get_json(silent=True) or {}
    text_content = data.get('text')
    voice_type = data.get('voice_type', 'neural') # studio, neural, standard
    language = data.get('language', 'ja-JP')
    
    if not isinstance(text_content, str) or not text_content.strip():
        return jsonify({'error': 'No text provided'}), 400
    if len(text_content) > 20_000:
        return jsonify({'error': 'Text is too large'}), 413
    
    try:
        g_key = decrypt_val(current_user.google_api_key)
        if not g_key and _admin_env_fallback_enabled(current_user):
            g_key = os.getenv('GOOGLE_API_KEY')
        if not g_key:
            return jsonify({'error': 'Google API Key not configured (Google Cloud API key required)'}), 400
        
        g_project = decrypt_val(current_user.google_cloud_project)
        if not g_project and _admin_env_fallback_enabled(current_user):
            g_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        opts = {"api_key": g_key}
        if g_project: opts["quota_project_id"] = g_project
        client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(**opts)
        )
        
        synthesis_input = texttospeech.SynthesisInput(text=text_content)
        
        # Voice selection
        if voice_type == 'studio':
            voice = pick_tts_voice(client, language, "studio")
        elif voice_type == 'neural':
            voice = pick_tts_voice(client, language, "neural")
        else:
            voice = pick_tts_voice(client, language, "standard")

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save audio file
        try:
            ok, used, limit = _check_storage_capacity(current_user, len(response.audio_content) if response.audio_content else 0)
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
        except Exception:
            return jsonify({'error': 'Unable to validate storage capacity'}), 500
        fname = f"tts_{int(time.time())}_{os.urandom(4).hex()}.mp3"
        _save_user_generated_bytes(
            current_user.id, response.audio_content, fname, current_user.enable_e2ee
        )
            
        return jsonify({'url': f"/files/{current_user.id}/{fname}", 'filename': f"{current_user.id}/{fname}"})
    except Exception as e:
        logger.error(f"TTS Synthesis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe():
    audio_content = None
    fname = None

    if request.files and 'file' in request.files:
        f = request.files['file']
        if not f or not f.filename:
            return jsonify({'error': 'No file'}), 400
        fname = secure_filename(f.filename)
        audio_content = f.read(_AUDIO_INPUT_MAX_BYTES + 1)
    else:
        data = request.get_json(silent=True) or {}
        filename = data.get('filename')
        if not filename: return jsonify({'error': 'No filename'}), 400

        rel_path = _resolve_user_upload_rel_path(filename, current_user.id)
        if not rel_path:
            return jsonify({'error': 'Unauthorized'}), 403
        info = _get_file_disk_info(rel_path)
        if not info or not info.get('exists'):
            return jsonify({'error': 'File not found'}), 404
        if int(info.get('size') or 0) > _AUDIO_INPUT_MAX_BYTES:
            return jsonify({'error': 'Audio file is too large'}), 413
        audio_content = _load_user_file_bytes(rel_path, info)
        fname = os.path.basename(rel_path)

    if not audio_content:
        return jsonify({'error': 'Empty audio'}), 400
    if len(audio_content) > _AUDIO_INPUT_MAX_BYTES:
        return jsonify({'error': 'Audio file is too large'}), 413

    try:
        transcribe_mode = _normalize_mic_transcribe_mode(getattr(current_user, 'mic_transcribe_mode', None))
        if transcribe_mode == "llm":
            req_data = request.form if request.form else (request.json or {})
            llm_model_key = (req_data.get('llm_model') or req_data.get('model') or "").strip()
            transcript = _transcribe_audio_with_llm(audio_content, fname, llm_model_key, current_user)
            return jsonify({'transcript': transcript, 'mode': 'llm'})

        allowed_models = {
            "gpt-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "gpt-4o-transcribe-diarize",
            "whisper-1"
        }
        model = (current_user.stt_model or "").strip()
        if model not in allowed_models:
            model = "gpt-4o-mini-transcribe"
        key = _get_model_specific_api_key(current_user, model) or decrypt_val(current_user.openai_api_key)
        if not key and _admin_env_fallback_enabled(current_user):
            key = os.getenv('OPENAI_API_KEY')
        if not key:
            return jsonify({'error': 'OpenAI API Key not configured'}), 400

        client = _get_openai_client(key, base_url=None)
        audio_file = BytesIO(audio_content)
        audio_file.name = fname

        kwargs = {"model": model, "file": audio_file}
        if model == "gpt-4o-transcribe-diarize":
            kwargs["response_format"] = "diarized_json"
            kwargs["chunking_strategy"] = "auto"

        transcription = client.audio.transcriptions.create(**kwargs)

        transcript = ""
        segments = None
        if isinstance(transcription, dict):
            transcript = transcription.get("text") or ""
            segments = transcription.get("segments")
        else:
            transcript = getattr(transcription, "text", "") or ""
            segments = getattr(transcription, "segments", None)

        if model == "gpt-4o-transcribe-diarize" and segments:
            lines = []
            for seg in segments:
                if isinstance(seg, dict):
                    speaker = seg.get("speaker") or "Speaker"
                    text = seg.get("text") or ""
                else:
                    speaker = getattr(seg, "speaker", None) or "Speaker"
                    text = getattr(seg, "text", "") or ""
                if text:
                    lines.append(f"{speaker}: {text}")
            if lines:
                transcript = "\n".join(lines)

        return jsonify({'transcript': transcript, 'mode': 'stt_api'})
    except ValueError as e:
        logger.warning(f"Transcription validation failed: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sts', methods=['POST'])
@login_required
def speech_to_speech():
    if not request.files or 'file' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'No file'}), 400

    model_key = (request.form.get('model') or "").strip()
    if not is_sts_model(model_key):
        return jsonify({'error': 'Invalid STS model'}), 400

    thread_id = request.form.get('thread_id')
    if not thread_id:
        return jsonify({'error': 'thread_id required'}), 400
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    thread_id = t.id

    audio_bytes = f.read(_AUDIO_INPUT_MAX_BYTES + 1)
    if not audio_bytes:
        return jsonify({'error': 'Empty audio'}), 400
    if len(audio_bytes) > _AUDIO_INPUT_MAX_BYTES:
        return jsonify({'error': 'Audio file is too large'}), 413

    provider = get_sts_provider(model_key)
    meta = STS_MODELS.get(model_key, {})
    rate_in = meta.get("rate_in", 24000)
    rate_out = meta.get("rate_out", 24000)
    sts_voice = (request.form.get('sts_voice') or "").strip()
    sts_speed_raw = request.form.get('sts_speed')
    sts_rate_in_raw = request.form.get('sts_rate_in')
    sts_rate_out_raw = request.form.get('sts_rate_out')
    sts_thinking_level = request.form.get('sts_thinking_level')
    sts_include_thoughts = request.form.get('sts_include_thoughts') == 'true'
    sts_speed = None

    if provider == "openai":
        v = sts_voice.lower() if sts_voice else "alloy"
        if v not in OPENAI_STS_VOICES:
            v = "alloy"
        sts_voice = v
        sts_speed = clamp_float(sts_speed_raw, 0.25, 1.5)
    elif provider == "xai":
        if sts_voice not in XAI_STS_VOICES:
            sts_voice = "Ara"
        try:
            ri = int(sts_rate_in_raw) if sts_rate_in_raw is not None and str(sts_rate_in_raw).strip() != "" else None
            ro = int(sts_rate_out_raw) if sts_rate_out_raw is not None and str(sts_rate_out_raw).strip() != "" else None
            if ri in XAI_PCM_RATES: rate_in = ri
            if ro in XAI_PCM_RATES: rate_out = ro
        except Exception:
            pass
    elif provider == "google":
        if sts_voice not in GEMINI_STS_VOICES:
            sts_voice = "Kore"

    model_specific_key = _get_model_specific_api_key(current_user, model_key)

    if provider == "google":
        gemini_runtime = _resolve_gemini_runtime(current_user)
        key = model_specific_key or gemini_runtime.get("api_key")
        if gemini_runtime.get("backend") == "vertex_ai":
            if not gemini_runtime.get("vertex_project"):
                return jsonify({'error': 'Vertex AI Project ID not configured'}), 400
        elif not key:
            return jsonify({'error': 'Gemini API Key not configured'}), 400

        def generate_sts_stream():
            assistant_audio = bytearray()
            assistant_text = ""
            assistant_thought = ""
            input_text = ""
            try:
                # Yield a processing status immediately
                yield json.dumps({'status': 'processing'}) + "\n"

                # Move conversion inside the stream for faster response start
                src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
                pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)

                # Use a small buffer for audio chunks to send to client
                audio_buffer = bytearray()
                
                # Consume the generator
                gen = _google_sts_live(
                    pcm_bytes,
                    model_key,
                    gemini_api_key=key,
                    gemini_backend=gemini_runtime.get("backend"),
                    gemini_vertex_project=gemini_runtime.get("vertex_project"),
                    gemini_vertex_location=gemini_runtime.get("vertex_location"),
                    gemini_vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
                    rate=rate_in,
                    voice=sts_voice,
                    thinking_level=sts_thinking_level,
                    include_thoughts=sts_include_thoughts
                )
                
                # Iterate over the async generator using an event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def run_gen():
                    nonlocal assistant_text, assistant_thought, input_text
                    async for audio_chunk, transcript_delta, input_delta, thought_delta, turn_complete in gen:
                        if audio_chunk:
                            assistant_audio.extend(audio_chunk)
                            audio_buffer.extend(audio_chunk)
                        if transcript_delta: assistant_text += transcript_delta
                        if input_delta: input_text += input_delta
                        if thought_delta: assistant_thought += thought_delta
                        
                        # Send chunks if we have enough audio or text updates
                        if len(audio_buffer) >= 1000 or transcript_delta or thought_delta or turn_complete:
                            payload = {}
                            if audio_buffer:
                                payload['audio_delta'] = base64.b64encode(audio_buffer).decode('utf-8')
                                audio_buffer.clear()
                            if transcript_delta: payload['transcript_delta'] = transcript_delta
                            if thought_delta: payload['thought_delta'] = thought_delta
                            if input_delta: payload['input_delta'] = input_delta
                            if turn_complete: payload['turn_complete'] = True
                            
                            if payload:
                                yield json.dumps(payload) + "\n"

                # Convert async generator to sync generator for Flask
                it = run_gen().__aiter__()
                while True:
                    try:
                        yield loop.run_until_complete(it.__anext__())
                    except StopAsyncIteration:
                        break
                
                # After stream ends, save to DB
                if assistant_audio:
                    wav_bytes = _pcm_to_wav_bytes(bytes(assistant_audio), rate=rate_out)
                    out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
                    audio_url = f"/files/{current_user.id}/{out_fname}"
                    
                    in_fname = None
                    try:
                        in_suffix = src_ext if src_ext.startswith('.') else f".{src_ext}"
                        in_fname, _ = _save_user_audio(current_user.id, audio_bytes, in_suffix, current_user.enable_e2ee)
                    except Exception: pass

                    user_text = (input_text or "Voice message").strip()
                    assistant_text_clean = (assistant_text or "").strip()
                    assistant_thought_clean = (assistant_thought or "").strip()
                    
                    thought_tag = f"<thought>\n{assistant_thought_clean}\n</thought>\n" if assistant_thought_clean else ""
                    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
                    assistant_content = thought_tag + (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

                    # DB Save Logic
                    try:
                        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
                        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
                        user_tokens_in = count_tokens_for_display(user_text, model_key)
                        assistant_tokens_out = count_tokens_for_display(assistant_text_clean, model_key)
                        if assistant_thought_clean:
                            assistant_tokens_out += count_tokens_for_display(assistant_thought_clean, model_key)
                        
                        parent_id = None
                        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
                        if last_msg: parent_id = last_msg.id

                        user_msg = Message(
                            thread_id=thread_id,
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
                            thread_id=thread_id,
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
                        
                        # Send final metadata
                        yield json.dumps({
                            'final': True,
                            'audio_url': audio_url,
                            'transcript': assistant_text_clean,
                            'thought': assistant_thought_clean,
                            'input_transcript': user_text
                        }) + "\n"
                    except Exception as e:
                        logger.error(f"STS stream message save failed: {e}")
            except Exception as e:
                logger.error(f"STS stream failed: {e}")
                yield json.dumps({'error': str(e)}) + "\n"
            finally:
                loop.close()

        resp = Response(stream_with_context(generate_sts_stream()), content_type='application/x-ndjson')
        resp.headers['X-Accel-Buffering'] = 'no'
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    if provider == "openai" and meta.get("mode") == "transcription":
        key = model_specific_key or decrypt_val(current_user.openai_api_key)
        if not key and _admin_env_fallback_enabled(current_user):
            key = os.getenv('OPENAI_API_KEY')
        if not key:
            return jsonify({'error': 'OpenAI API Key not configured'}), 400

        src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
        try:
            pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)
            transcript = asyncio.run(
                _openai_realtime_transcribe(pcm_bytes, key, model_key, rate=rate_in)
            ).strip()
        except Exception as e:
            logger.error(f"OpenAI realtime transcription failed: {e}")
            return jsonify({'error': str(e)}), 500
        if not transcript:
            return jsonify({'error': 'No transcript returned'}), 500

        try:
            ok, used, limit = _check_storage_capacity(current_user, len(audio_bytes))
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
        except Exception:
            return jsonify({'error': 'Unable to validate storage capacity'}), 500

        in_fname = None
        try:
            in_suffix = src_ext if src_ext.startswith('.') else f".{src_ext}"
            in_fname, _ = _save_user_audio(
                current_user.id, audio_bytes, in_suffix, current_user.enable_e2ee
            )
        except Exception as e:
            logger.error(f"Realtime transcription audio save failed: {e}")

        try:
            last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
            user_text = "音声文字起こし"
            user_msg = Message(
                thread_id=thread_id,
                role='user',
                content=encrypt_val(user_text) if current_user.enable_e2ee else user_text,
                image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
                is_encrypted=current_user.enable_e2ee,
                parent_id=last_msg.id if last_msg else None,
                model=model_key,
                tokens_in=count_tokens_for_display(user_text, model_key),
            )
            user_msg.tokens = sum_token_counts(user_msg.tokens_in, None)
            db.session.add(user_msg)
            safe_db_commit()

            assistant_msg = Message(
                thread_id=thread_id,
                role='assistant',
                content=encrypt_val(transcript) if current_user.enable_e2ee else transcript,
                model=model_key,
                is_encrypted=current_user.enable_e2ee,
                parent_id=user_msg.id,
                tokens_out=count_tokens_for_display(transcript, model_key),
            )
            assistant_msg.tokens = sum_token_counts(None, assistant_msg.tokens_out)
            db.session.add(assistant_msg)
            safe_db_commit()
        except Exception as e:
            logger.error(f"Realtime transcription message save failed: {e}")

        payload = {
            'final': True,
            'transcription_only': True,
            'transcript': transcript,
            'input_transcript': transcript,
        }
        resp = Response(
            json.dumps(payload, ensure_ascii=False) + "\n",
            content_type='application/x-ndjson',
        )
        resp.headers['X-Accel-Buffering'] = 'no'
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    # Original sync logic for OpenAI/xAI
    src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
    try:
        pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)
    except Exception as e:
        logger.error(f"STS audio conversion failed: {e}")
        return jsonify({'error': f'Audio conversion failed: {e}'}), 400

    assistant_audio = b""
    assistant_text = ""
    assistant_thought = ""
    input_text = ""
    try:
        if provider == "openai":
            key = model_specific_key or decrypt_val(current_user.openai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('OPENAI_API_KEY')
            if not key:
                return jsonify({'error': 'OpenAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _openai_sts_realtime(pcm_bytes, key, model_key, voice=sts_voice, speed=sts_speed, rate=rate_out)
            )
        elif provider == "xai":
            key = model_specific_key or decrypt_val(current_user.xai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('XAI_API_KEY')
            if not key:
                return jsonify({'error': 'xAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _xai_sts_realtime(pcm_bytes, key, model_key=model_key, voice=sts_voice, rate_in=rate_in, rate_out=rate_out)
            )
        else:
            return jsonify({'error': 'Unsupported provider'}), 400
    except Exception as e:
        logger.error(f"STS failed: {e}")
        return jsonify({'error': str(e)}), 500

    if not assistant_audio:
        return jsonify({'error': 'No audio response'}), 500

    wav_bytes = _pcm_to_wav_bytes(assistant_audio, rate=rate_out)
    try:
        incoming_size = len(wav_bytes) + (len(audio_bytes) if audio_bytes else 0)
        ok, used, limit = _check_storage_capacity(current_user, incoming_size)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
    except Exception:
        pass
    out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
    audio_url = f"/files/{current_user.id}/{out_fname}"

    in_fname = None
    try:
        in_suffix = src_ext if src_ext.startswith('.') else f".{src_ext}"
        in_fname, _ = _save_user_audio(current_user.id, audio_bytes, in_suffix, current_user.enable_e2ee)
    except Exception:
        in_fname = None

    parent_id = None
    try:
        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id
    except Exception:
        parent_id = None

    user_text = (input_text or "Voice message").strip()
    assistant_text_clean = (assistant_text or "").strip()
    assistant_thought_clean = (assistant_thought or "").strip()
    
    thought_tag = f"<thought>\n{assistant_thought_clean}\n</thought>\n" if assistant_thought_clean else ""
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
    assistant_content = thought_tag + (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(user_text, model_key)
        assistant_tokens_out = count_tokens_for_display(assistant_text_clean, model_key)
        # Add thought tokens to assistant tokens out if possible
        if assistant_thought_clean:
            assistant_tokens_out += count_tokens_for_display(assistant_thought_clean, model_key)
        
        user_msg = Message(
            thread_id=thread_id,
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
            thread_id=thread_id,
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
    except Exception as e:
        logger.error(f"STS message save failed: {e}")

    return jsonify({
        'audio_url': audio_url,
        'transcript': assistant_text_clean,
        'thought': assistant_thought_clean,
        'input_transcript': user_text,
        'filename': f"{current_user.id}/{out_fname}"
    })

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    ALLOWED_EXTENSIONS = _UPLOAD_ALLOWED_EXTENSIONS
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    if len(files) > int(app.config.get('ATTACHMENT_MAX_FILES') or 30):
        return jsonify({'error': 'Too many files'}), 400
    try:
        # Reclaim abandoned chunk sessions before they are included in the
        # storage-capacity check for a new upload.
        _cleanup_stale_chunk_uploads(current_user.id)
        if not _is_primary_admin_user(current_user):
            hard_limit = _get_user_storage_limit_bytes(current_user)
            if hard_limit:
                for f in files:
                    size = _get_filestorage_size(f)
                    if size is not None and size > hard_limit:
                        limit_mb = _bytes_to_mb_str(hard_limit)
                        return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        total_incoming = 0
        for f in files:
            size = _get_filestorage_size(f)
            if size is None:
                continue
            total_incoming += size
        if not _is_primary_admin_user(current_user):
            ok, used, limit = _check_storage_capacity(current_user, total_incoming)
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
    except Exception:
        return jsonify({'error': 'Unable to validate upload size'}), 400
    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass
    res = []
    cache_updated = False
    for f in files:
        if f.filename:
            orig_name, ext = _sanitize_upload_filename(f.filename)
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({'error': f'File type {ext} not allowed'}), 400
            
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            save_path = os.path.join(ud, fname)
            if current_user.enable_e2ee:
                with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                    ef.write(encrypt_bytes(f.read()))
            else:
                f.save(save_path)
            rel_path = f"{current_user.id}/{fname}"
            res.append(rel_path)
            try:
                disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
                size = None
                mtime = None
                try:
                    size = os.path.getsize(disk_path)
                except Exception:
                    size = None
                try:
                    mtime = int(os.path.getmtime(disk_path))
                except Exception:
                    mtime = None
                mime_guess = mimetypes.guess_type(fname)[0]
                mime = _normalize_media_mime(fname, mime_guess)
                _upsert_file_cache(
                    current_user.id,
                    rel_path,
                    "local",
                    size_bytes=size,
                    mtime=mtime,
                    mime_type=mime,
                    state="stored",
                    last_error=None
                )
                cache_updated = True
            except Exception:
                pass
    if cache_updated:
        try:
            safe_db_commit()
        except Exception:
            pass
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/upload/init', methods=['POST'])
@login_required
def upload_init():
    if not rate_limit(f"rl:upload_init:user:{current_user.id}", 30, 60):
        return jsonify({'error': 'Too many upload requests'}), 429
    data = request.get_json(silent=True) or {}
    filename, ext = _sanitize_upload_filename(data.get('filename') or '')
    try:
        total_size = int(data.get('size') or 0)
    except (TypeError, ValueError):
        total_size = 0
    if not filename or total_size <= 0:
        return jsonify({'error': 'Invalid upload'}), 400

    allowed = _UPLOAD_ALLOWED_EXTENSIONS
    if ext not in allowed:
        return jsonify({'error': f'File type {ext} not allowed'}), 400

    # This must run before the quota scan. Otherwise an abandoned partial upload
    # can keep a user over quota and prevent the very request that would clean it.
    active_uploads = _cleanup_stale_chunk_uploads(current_user.id)

    if not _is_primary_admin_user(current_user):
        hard_limit = _get_user_storage_limit_bytes(current_user)
        if hard_limit and total_size > hard_limit:
            limit_mb = _bytes_to_mb_str(hard_limit)
            return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        ok, used, limit = _check_storage_capacity(current_user, total_size)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413

    if active_uploads >= _CHUNK_UPLOAD_MAX_ACTIVE:
        return jsonify({'error': 'Too many active uploads'}), 429

    upload_id = f"up_{int(time.time())}_{os.urandom(4).hex()}"
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir:
        return jsonify({'error': 'Init failed'}), 500
    os.makedirs(session_dir, exist_ok=True)
    os.chmod(session_dir, 0o700)
    meta = {
        "filename": filename,
        "size": total_size,
        "received": 0,
        "created": int(time.time()),
        "updated": int(time.time()),
        "ext": ext,
        "chunk_size": _CHUNK_SIZE_BYTES,
        "total_chunks": (total_size + _CHUNK_SIZE_BYTES - 1) // _CHUNK_SIZE_BYTES,
        "state": "receiving"
    }
    if not _save_chunk_meta(os.path.join(session_dir, 'meta.json'), meta):
        return jsonify({'error': 'Init failed'}), 500
    return jsonify({'upload_id': upload_id, 'chunk_size': _CHUNK_SIZE_BYTES})

@app.route('/upload/chunk', methods=['POST'])
@login_required
def upload_chunk():
    if not rate_limit(f"rl:upload_chunk:user:{current_user.id}", 240, 60):
        return jsonify({'error': 'Too many upload requests'}), 429
    upload_id = (request.form.get('upload_id') or '').strip()
    index = request.form.get('index')
    total = request.form.get('total')
    f = request.files.get('chunk')
    if not _is_valid_chunk_upload_id(upload_id) or f is None:
        return jsonify({'error': 'Invalid chunk'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir or not os.path.isdir(session_dir):
        return jsonify({'error': 'Upload not found'}), 404
    meta_path = os.path.join(session_dir, 'meta.json')
    try:
        index = int(index) if index is not None else 0
        total = int(total) if total is not None else 0
    except Exception:
        return jsonify({'error': 'Invalid chunk index'}), 400
    part_path = os.path.join(session_dir, 'data.part')
    try:
        with _chunk_upload_lock(session_dir):
            meta = _load_chunk_meta(meta_path)
            if not meta:
                return jsonify({'error': 'Upload not found'}), 404
            if meta.get('state') != 'receiving':
                return jsonify({'error': 'Upload is not accepting chunks'}), 409
            declared_size = int(meta.get('size') or 0)
            received = int(meta.get('received') or 0)
            chunk_size = int(meta.get('chunk_size') or _CHUNK_SIZE_BYTES)
            expected_total = int(meta.get('total_chunks') or 0)
            if received >= declared_size:
                return jsonify({'error': 'Upload already complete'}), 409
            if chunk_size != _CHUNK_SIZE_BYTES or expected_total <= 0 or total != expected_total:
                return jsonify({'error': 'Invalid chunk count'}), 400
            expected_index = received // chunk_size
            if index != expected_index or index < 0 or index >= expected_total:
                return jsonify({'error': 'Invalid chunk order'}), 409
            current_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if current_size != received:
                return jsonify({'error': 'Upload state mismatch'}), 409
            remaining = declared_size - received
            expected_bytes = min(chunk_size, remaining)
            chunk_data = f.read(chunk_size + 1)
            if expected_bytes <= 0 or len(chunk_data) != expected_bytes:
                return jsonify({'error': 'Invalid chunk size'}), 400
            if not _is_primary_admin_user(current_user):
                limit = _get_user_storage_limit_bytes(current_user)
                used = _get_user_storage_usage_bytes(current_user.id)
                if limit and used + len(chunk_data) > limit:
                    return jsonify({'error': 'Storage limit exceeded'}), 413
            with open(part_path, 'ab') as out:
                out.write(chunk_data)
                out.flush()
            meta['received'] = received + len(chunk_data)
            meta['updated'] = int(time.time())
            if not _save_chunk_meta(meta_path, meta):
                with open(part_path, 'r+b') as out:
                    out.truncate(received)
                return jsonify({'error': 'Chunk state write failed'}), 500
    except Exception:
        return jsonify({'error': 'Chunk write failed'}), 500

    return jsonify({'received': meta['received'], 'total': meta.get('size', 0), 'index': index, 'chunks': total})

@app.route('/upload/complete', methods=['POST'])
@login_required
def upload_complete():
    data = request.get_json(silent=True) or {}
    upload_id = (data.get('upload_id') or '').strip()
    if not _is_valid_chunk_upload_id(upload_id):
        return jsonify({'error': 'Invalid upload'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir or not os.path.isdir(session_dir):
        return jsonify({'error': 'Upload not found'}), 404
    meta_path = os.path.join(session_dir, 'meta.json')
    part_path = os.path.join(session_dir, 'data.part')
    with _chunk_upload_lock(session_dir):
        meta = _load_chunk_meta(meta_path)
        if not meta:
            return jsonify({'error': 'Upload not found'}), 404
        if meta.get('state') != 'receiving':
            return jsonify({'error': 'Upload is already finalizing'}), 409
        if not os.path.exists(part_path):
            return jsonify({'error': 'Upload missing'}), 400
        declared_size = int(meta.get('size') or 0)
        if int(meta.get('received') or 0) != declared_size or os.path.getsize(part_path) != declared_size:
            return jsonify({'error': 'Upload incomplete'}), 400
        meta['state'] = 'finalizing'
        if not _save_chunk_meta(meta_path, meta):
            return jsonify({'error': 'Finalize state failed'}), 500

    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass

    ext = (meta.get('ext') or '').lower()
    if not ext.startswith('.'):
        # Fall back to the sanitized filename's extension for sessions created
        # before the meta["ext"] value started carrying the original extension.
        ext = os.path.splitext(str(meta.get('filename') or 'file'))[1].lower()
    fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
    fname = f"{fname_base}{ext}"
    save_path = os.path.join(ud, fname)
    res = []
    cache_updated = False
    try:
        if current_user.enable_e2ee:
            with open(part_path, 'rb') as rf:
                with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                    ef.write(encrypt_bytes(rf.read()))
        else:
            os.replace(part_path, save_path)
        res.append(f"{current_user.id}/{fname}")
        rel_path = f"{current_user.id}/{fname}"
        try:
            disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
            size = None
            mtime = None
            try:
                size = os.path.getsize(disk_path)
            except Exception:
                size = None
            try:
                mtime = int(os.path.getmtime(disk_path))
            except Exception:
                mtime = None
            mime_guess = mimetypes.guess_type(fname)[0]
            mime = _normalize_media_mime(fname, mime_guess)
            _upsert_file_cache(
                current_user.id,
                rel_path,
                "local",
                size_bytes=size,
                mtime=mtime,
                mime_type=mime,
                state="stored",
                last_error=None
            )
            cache_updated = True
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Chunk finalize failed: {e}")
        return jsonify({'error': 'Finalize failed'}), 500
    finally:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except Exception:
            pass
        try:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            lock_path = os.path.join(session_dir, '.lock')
            if os.path.exists(lock_path):
                os.remove(lock_path)
            os.rmdir(session_dir)
        except Exception:
            pass
    if cache_updated:
        try:
            safe_db_commit()
        except Exception:
            pass
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/api/storage', methods=['GET'])
@login_required
def get_storage_usage():
    limit = _get_user_storage_limit_bytes(current_user)
    used = _get_user_storage_usage_bytes(current_user.id)
    if limit is None:
        limit = 0
    return jsonify({
        'used_bytes': used,
        'limit_bytes': limit,
        'used_mb': _bytes_to_mb_str(used),
        'limit_mb': _bytes_to_mb_str(limit) if limit else 'unlimited',
        'is_unlimited': limit == 0
    })

@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(e):
    limit = getattr(request, 'max_content_length', None) or app.config.get('MAX_CONTENT_LENGTH')
    if not limit:
        return jsonify({'error': 'File too large. The server rejected the upload.'}), 413
    limit_mb = limit // (1024 * 1024)
    return jsonify({'error': f'File too large. Max {limit_mb}MB'}), 413

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        try:
            logger.error(f"db.create_all failed: {e}")
        except Exception:
            pass
    # Run column additions BEFORE any model query to avoid SQLAlchemy metadata cache staleness
    # This column is required by every authenticated User SELECT, so it must not depend on
    # RUN_SCHEMA_MIGRATIONS. Fail startup clearly instead of serving HTTP 500 to all sessions.
    ensure_user_liquid_glass_column()
    # Model output, reasoning, and Fernet ciphertext can exceed MySQL TEXT's 64 KiB
    # limit. This is correctness-critical and must run even when optional migrations
    # are disabled.
    ensure_message_payload_longtext_columns()
    try:
        try_alter("ALTER TABLE user ADD COLUMN last_gem_uuid VARCHAR(36)")
    except: pass
    try:
        try_alter("ALTER TABLE thread ADD COLUMN last_gem_uuid VARCHAR(36)")
    except: pass
    try:
        try_alter("ALTER TABLE gem ADD COLUMN uuid VARCHAR(36)")
    except: pass
    try:
        import uuid as _uuid_backfill
        for gem in Gem.query.filter(Gem.uuid.is_(None)).all():
            gem.uuid = str(_uuid_backfill.uuid4())
        safe_db_commit()
    except: pass
    try:
        ensure_thread_last_model_column()
    except Exception:
        pass
    try:
        ensure_thread_temporary_column()
    except Exception:
        pass
    try:
        ensure_thread_prompt_caching_columns()
    except Exception:
        pass
    try:
        ensure_import_signature_columns()
    except Exception:
        pass
    try:
        ensure_user_minashin_columns()
    except Exception:
        pass
    try:
        ensure_message_token_io_columns()
    except Exception:
        pass
    try:
        ensure_user_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_user_file_creation_columns()
    except Exception:
        pass
    try:
        ensure_user_mcp_enable_columns()
    except Exception:
        pass
    try:
        ensure_user_gemini_backend_columns()
    except Exception:
        pass
    try:
        ensure_user_deepseek_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_kimi_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_mistral_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_anthropic_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_admin_api_key_mode_column()
    except Exception:
        pass
    try:
        ensure_user_2fa_default_columns()
    except Exception:
        pass
    try:
        ensure_user_model_api_keys_column()
    except Exception:
        pass
    try:
        ensure_user_temp_chat_timeout_column()
    except Exception:
        pass
    try:
        ensure_user_compact_prompt_mode_column()
    except Exception:
        pass
    try:
        ensure_user_minimal_prompt_mode_column()
    except Exception:
        pass
    try:
        ensure_user_voice_studio_ui_column()
    except Exception:
        pass
    try:
        ensure_gem_fixed_prompts_column()
    except Exception:
        pass
    try:
        ensure_gem_default_model_column()
    except Exception:
        pass
    try:
        ensure_user_stt_settings_columns()
    except Exception:
        pass
    try:
        ensure_chat_latency_trace_columns()
    except Exception:
        pass
    try:
        ensure_user_debug_settings_columns()
    except Exception:
        pass
    try:
        ensure_bot_evidence_columns()
    except Exception:
        pass
    try:
        ensure_user_cache_settings_columns()
    except Exception:
        pass
    try:
        ensure_user_default_model_columns()
    except Exception:
        pass
    try:
        ensure_user_vision_model_columns()
    except Exception:
        pass
    try:
        ensure_user_google_columns()
    except Exception:
        pass
    try:
        cleanup_user_temp_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_performance_indexes()
    except Exception:
        pass
    try:
        ensure_app_setting("bot_detection_global_enabled", "1")
    except Exception:
        pass
    try:
        # MCP外部連携のプリセットサーバー行を用意する（無ければ作成）
        from mcp_service.registry import get_or_create_presets
        get_or_create_presets()
    except Exception as _mcp_preset_err:
        try:
            log_force(f"MCP preset seeding failed: {_mcp_preset_err}")
        except Exception:
            pass
    try:
        admin_user = None
        primary_admin = _get_primary_admin_username()
        if primary_admin:
            admin_user = User.query.filter_by(username=primary_admin).first()
        if admin_user and not getattr(admin_user, "is_admin", False):
            admin_user.is_admin = True
            safe_db_commit()
    except Exception:
        pass
    if RUN_SCHEMA_MIGRATIONS:
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN thought_signature TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_in INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_out INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_thought INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_e2ee BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN is_encrypted BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_client_debug_log BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_done_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_total_latency_ms INTEGER")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN xai_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_backend VARCHAR(24) DEFAULT 'gemini_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_location VARCHAR(64) DEFAULT 'global'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_credentials_json TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN deepseek_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN kimi_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN mistral_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN admin_api_key_mode VARCHAR(24) DEFAULT 'env_fallback'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN model_api_keys TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_2fa_enabled BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN totp_secret VARCHAR(255)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN webauthn_credentials TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN passkey_only_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN skip_2fa_on_google_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_2fa_method VARCHAR(16) DEFAULT 'totp'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN mic_transcribe_mode VARCHAR(16) DEFAULT 'stt_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN llm_transcribe_prompt TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enter_to_send BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_sw_cache BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN clear_cache_on_version_update BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN theme_color VARCHAR(16)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_search_on_links BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN compact_prompt_mode BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN minimal_prompt_mode BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_last_chat_settings BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN voice_studio_ui BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter(f"ALTER TABLE user ADD COLUMN temp_chat_timeout_seconds INTEGER DEFAULT {_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS}")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_mcp BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_maps BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_file_creation BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN system_prompt_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_global_system_prompt BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_auto_system_prompt_notices BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_system_prompt_notices_config TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_default TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_use_custom_default BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_mcp BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_maps BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_file_creation BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_cloud_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_hash TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_expires_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_detection_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_bot_banned BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_banned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_ban_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unbanned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unban_notice BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_block_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN admin_reply TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN replied_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN last_seen_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN ip_address VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_bookmarked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN bookmarked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN public_id VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN last_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_temporary BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN enable_prompt_caching BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN prompt_cache_provider VARCHAR(32)")
        except: pass

@app.route('/api/metrics/first_token', methods=['POST'])
@login_required
def first_token_metric():
    if not rate_limit(f"rl:first_token_metric:user:{current_user.id}", 240, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}

        try:
            latency_seconds = float(d.get('latency_seconds'))
        except Exception:
            return jsonify({'error': 'latency_seconds is required'}), 400
        if latency_seconds != latency_seconds or latency_seconds < 0 or latency_seconds > 600:
            return jsonify({'error': 'latency_seconds out of range'}), 400

        latency_ms = _coerce_int_or_none(d.get('latency_ms'))
        if latency_ms is None:
            latency_ms = int(round(latency_seconds * 1000))
        if latency_ms < 0:
            latency_ms = 0

        thread_public_id = str(d.get('thread_id') or '').strip()[:64] or None
        job_id = str(d.get('job_id') or '').strip()[:64] or None
        model = str(d.get('model') or '').strip()[:80] or None
        first_event_type = str(d.get('first_event_type') or '').strip()[:32] or None

        client_sent_at = None
        client_sent_at_ms = _coerce_int_or_none(d.get('client_sent_at_ms'))
        if client_sent_at_ms is not None and 946684800000 <= client_sent_at_ms <= 4102444800000:
            client_sent_at = datetime.utcfromtimestamp(client_sent_at_ms / 1000.0)

        is_total = bool(d.get('is_total'))
        client_done_at_ms = _coerce_int_or_none(d.get('client_done_at_ms'))

        if is_total:
            log_force(f"RECEIVED-TOTAL-REPORT: job={job_id} model={model} latency_ms={latency_ms}")
            # If it's a total latency report, we primarily update the trace
            _upsert_chat_latency_trace(
                job_id=job_id,
                user_id=current_user.id,
                thread_public_id=thread_public_id,
                model=model,
                client_sent_at_ms=client_sent_at_ms,
                client_done_at_ms=client_done_at_ms,
                client_total_latency_ms=latency_ms
            )
            log_force(
                "TOTAL-LATENCY-METRIC: "
                f"user={current_user.id} "
                f"model={model or '-'} "
                f"thread={thread_public_id or '-'} "
                f"job={job_id or '-'} "
                f"total_seconds={latency_seconds:.3f} "
                f"client_done_at={client_done_at_ms or '-'}"
            )
            return jsonify({'status': 'ok', 'type': 'total'})

        row = FirstTokenLatencyMetric(
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            job_id=job_id,
            model=model,
            first_event_type=first_event_type,
            latency_seconds=round(latency_seconds, 6),
            latency_ms=latency_ms,
            client_sent_at=client_sent_at,
            ip_address=get_client_ip(),
            user_agent=get_request_user_agent()
        )
        db.session.add(row)
        safe_db_commit()
        trace = _upsert_chat_latency_trace(
            job_id=job_id,
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            model=model,
            client_sent_at_ms=client_sent_at_ms,
            client_first_event_type=first_event_type,
            client_first_latency_ms=latency_ms
        )

        window_start = datetime.utcnow() - timedelta(hours=24)
        stats = db.session.query(
            func.count(FirstTokenLatencyMetric.id),
            func.avg(FirstTokenLatencyMetric.latency_seconds),
            func.min(FirstTokenLatencyMetric.latency_seconds),
            func.max(FirstTokenLatencyMetric.latency_seconds)
        ).filter(
            FirstTokenLatencyMetric.user_id == current_user.id,
            FirstTokenLatencyMetric.created_at >= window_start
        ).first()
        stats_evt = None
        if first_event_type:
            stats_evt = db.session.query(
                func.count(FirstTokenLatencyMetric.id),
                func.avg(FirstTokenLatencyMetric.latency_seconds),
                func.min(FirstTokenLatencyMetric.latency_seconds),
                func.max(FirstTokenLatencyMetric.latency_seconds)
            ).filter(
                FirstTokenLatencyMetric.user_id == current_user.id,
                FirstTokenLatencyMetric.first_event_type == first_event_type,
                FirstTokenLatencyMetric.created_at >= window_start
            ).first()

        cnt = int((stats[0] or 0)) if stats else 0
        avg_s = float(stats[1]) if stats and stats[1] is not None else latency_seconds
        min_s = float(stats[2]) if stats and stats[2] is not None else latency_seconds
        max_s = float(stats[3]) if stats and stats[3] is not None else latency_seconds
        evt_cnt = int((stats_evt[0] or 0)) if stats_evt else 0
        evt_avg_s = float(stats_evt[1]) if stats_evt and stats_evt[1] is not None else latency_seconds
        evt_min_s = float(stats_evt[2]) if stats_evt and stats_evt[2] is not None else latency_seconds
        evt_max_s = float(stats_evt[3]) if stats_evt and stats_evt[3] is not None else latency_seconds
        phase_parts = []
        if trace:
            phase_candidates = {
                "client_to_route_ms": _trace_delta_ms(trace, "client_sent_at", "route_received_at"),
                "route_to_dispatch_ms": _trace_delta_ms(trace, "route_received_at", "route_dispatch_at"),
                "dispatch_to_worker_ms": _trace_delta_ms(trace, "route_dispatch_at", "worker_started_at"),
                "worker_to_provider_req_ms": _trace_delta_ms(trace, "worker_started_at", "provider_request_started_at"),
                "provider_req_to_first_chunk_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_chunk_at"),
                "provider_req_to_first_content_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_content_at"),
                "provider_content_to_client_ms": _trace_delta_ms(trace, "provider_first_content_at", "stream_first_content_to_client_at"),
                "route_to_client_content_ms": _trace_delta_ms(trace, "route_received_at", "stream_first_content_to_client_at"),
            }
            for key, val in phase_candidates.items():
                if val is not None:
                    phase_parts.append(f"{key}={val}")
        log_force(
            "FIRST-TOKEN-METRIC: "
            f"user={current_user.id} "
            f"model={model or '-'} "
            f"thread={thread_public_id or '-'} "
            f"job={job_id or '-'} "
            f"event={first_event_type or '-'} "
            f"seconds={latency_seconds:.3f} "
            f"window24h(count={cnt},avg={avg_s:.3f},min={min_s:.3f},max={max_s:.3f}) "
            f"event24h(count={evt_cnt},avg={evt_avg_s:.3f},min={evt_min_s:.3f},max={evt_max_s:.3f}) "
            f"path={getattr(trace, 'execution_path', '-') or '-'} "
            f"phases({','.join(phase_parts)})"
        )

        return jsonify({
            'status': 'ok',
            'latency_seconds': round(latency_seconds, 3),
            'window24h': {
                'count': cnt,
                'avg_seconds': round(avg_s, 3),
                'min_seconds': round(min_s, 3),
                'max_seconds': round(max_s, 3),
            },
            'event24h': {
                'event': first_event_type or None,
                'count': evt_cnt,
                'avg_seconds': round(evt_avg_s, 3),
                'min_seconds': round(evt_min_s, 3),
                'max_seconds': round(evt_max_s, 3),
            },
            'execution_path': getattr(trace, 'execution_path', None) if trace else None
        })
    except Exception as e:
        db.session.rollback()
        log_force(f"FIRST-TOKEN-METRIC-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 500

@app.route('/api/client_log', methods=['POST'])
@login_required
def client_log():
    if not getattr(current_user, 'enable_client_debug_log', False):
        return jsonify({'status': 'ignored', 'reason': 'disabled'}), 200
    if not rate_limit(f"rl:client_log:user:{current_user.id}", 60, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}
        level = str(d.get('level') or 'info').upper()
        if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR'}:
            level = 'INFO'
        msg = str(d.get('message') or '')
        if not msg:
            return jsonify({'status': 'ignored', 'reason': 'empty'}), 200
        msg = re.sub(r'[\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', ' ', msg)[:8192]
        log_force(f"CLIENT-DEBUG [LEGACY {level}]: {msg}")
        return jsonify({'status': 'ok'})
    except Exception:
        return jsonify({'status': 'error'}), 500

