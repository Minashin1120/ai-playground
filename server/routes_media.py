# region: TTS synthesis
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

# region: Speech-to-text transcription
def _xai_stt_transcript(audio_content, fname, model, key):
    """Batch transcription via xAI Grok Voice Transcribe (POST /v1/stt).

    Returns (text, None) on success or (None, (message, http_status)).
    """
    mime = mimetypes.guess_type(fname or "")[0] or "application/octet-stream"
    if _audio_ffmpeg_demuxer(audio_content) in ("matroska", None):
        # Browser recordings are WebM/Opus; xAI's container list covers
        # Matroska only with MP3/AAC/FLAC audio, so send 16 kHz WAV instead.
        try:
            pcm = _convert_audio_to_pcm(audio_content, rate=16000, max_seconds=None, timeout=120)
        except ValueError as exc:
            return None, (str(exc), 400)
        audio_content = _pcm_to_wav_bytes(pcm, rate=16000)
        fname = os.path.splitext(fname or "audio")[0] + ".wav"
        mime = "audio/wav"
    # Option fields must precede `file` in the multipart body (xAI docs).
    resp = requests.post(
        f"https://{_XAI_API_HOST}/v1/stt",
        headers={"Authorization": f"Bearer {key}"},
        data=[("model", model)],
        files=[("file", (fname or "audio", audio_content, mime))],
        timeout=600,
    )
    if resp.status_code != 200:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("error") or body.get("message") or ""
            if isinstance(detail, dict):
                detail = detail.get("message") or ""
        except Exception:
            detail = (resp.text or "")[:300]
        logger.error(f"xAI STT failed: status={resp.status_code} {detail}")
        status = resp.status_code if resp.status_code in (400, 401, 413, 429) else 502
        return None, (f"xAI STT error {resp.status_code}: {detail}".strip(), status)
    result = resp.json() or {}
    return result.get("text") or "", None

def _transcribe_with_xai_stt(audio_content, fname, model, user):
    key = _get_model_specific_api_key(user, model) or decrypt_val(user.xai_api_key)
    if not key and _admin_env_fallback_enabled(user):
        key = os.getenv('XAI_API_KEY')
    if not key:
        return jsonify({'error': 'xAI API Key not configured'}), 400
    text, err = _xai_stt_transcript(audio_content, fname, model, key)
    if err:
        return jsonify({'error': err[0]}), err[1]
    return jsonify({'transcript': text, 'mode': 'stt_api'})

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

        model = (current_user.stt_model or "").strip()
        if model not in VALID_STT_MODELS:
            model = "gpt-4o-mini-transcribe"
        if model in XAI_STT_MODELS:
            return _transcribe_with_xai_stt(audio_content, fname, model, current_user)
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

# region: Speech-to-speech
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
    sts_reasoning_effort = None

    if provider == "openai":
        v = sts_voice.lower() if sts_voice else "alloy"
        if v not in OPENAI_STS_VOICES:
            v = "alloy"
        sts_voice = v
        sts_speed = clamp_float(sts_speed_raw, 0.25, 1.5)
        effort = str(request.form.get('sts_reasoning_effort') or "").strip().lower()
        if model_key in OPENAI_RT_REASONING_MODELS and effort in OPENAI_RT_REASONING_EFFORTS:
            sts_reasoning_effort = effort
    elif provider == "xai":
        sts_voice = sts_voice.lower()
        if sts_voice not in XAI_STS_VOICES:
            sts_voice = "ara"
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

    if meta.get("mode") == "transcription" and (provider == "openai" or model_key in XAI_STT_FILE_STS_MODELS):
        src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
        if provider == "openai":
            key = model_specific_key or decrypt_val(current_user.openai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('OPENAI_API_KEY')
            if not key:
                return jsonify({'error': 'OpenAI API Key not configured'}), 400
            try:
                pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)
                transcript = asyncio.run(
                    _openai_realtime_transcribe(pcm_bytes, key, model_key, rate=rate_in)
                ).strip()
            except Exception as e:
                logger.error(f"OpenAI realtime transcription failed: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            key = model_specific_key or decrypt_val(current_user.xai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('XAI_API_KEY')
            if not key:
                return jsonify({'error': 'xAI API Key not configured'}), 400
            try:
                transcript, stt_err = _xai_stt_transcript(
                    audio_bytes, secure_filename(f.filename), XAI_STT_FILE_STS_MODELS[model_key], key
                )
            except Exception as e:
                logger.error(f"xAI file transcription failed: {e}")
                return jsonify({'error': str(e)}), 500
            if stt_err:
                return jsonify({'error': stt_err[0]}), stt_err[1]
            transcript = transcript.strip()
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
                _openai_sts_realtime(pcm_bytes, key, model_key, voice=sts_voice, speed=sts_speed, rate=rate_out,
                                     reasoning_effort=sts_reasoning_effort)
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
