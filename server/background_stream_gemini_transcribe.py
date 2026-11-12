"""Gemini 3.5 Transcribe branch of `background_chat_task`.

Extracted from `server/background_stream_gemini.py`.  The branch used to run
inline in `background_chat_task`, so it shared that function's locals.  It now
receives a `locals()` snapshot as ``local`` and returns the updated answer text.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to the caller when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_gemini_transcribe(local):
    """Transcribe an attached audio file and return the updated answer text."""

    _mark_provider_request_started = local.get("_mark_provider_request_started")
    full_res = local.get("full_res")
    g_client = local.get("g_client")
    gemini_backend_mode = local.get("gemini_backend_mode")
    gemini_files_api_enabled = gemini_backend_mode != "vertex_ai"
    job_id = local.get("job_id")
    key = local.get("key")
    loaded_files = local.get("loaded_files")
    options = local.get("options")
    pub = local.get("pub")

    try:
        _mark_provider_request_started()
        if gemini_backend_mode == "vertex_ai":
            pub("error", "Gemini 3.5 Transcribe: 現在は Gemini API モード（Vertex AI 以外）でのみ利用できます。")
            return BACKGROUND_STREAM_EARLY_RETURN
        audio_fi = next(
            (
                fi for fi in loaded_files
                if fi.get('bytes') and str(fi.get('mime', '')).startswith('audio/')
            ),
            None
        )
        if not audio_fi:
            pub("error", "Gemini 3.5 Transcribe: 文字起こしには音声ファイル（MP3/WAV/M4A/OGG/FLAC等）を添付してください。")
            return BACKGROUND_STREAM_EARLY_RETURN
        audio_data = audio_fi.get('bytes')
        audio_mime = audio_fi.get('mime') or "audio/mpeg"
        audio_name = audio_fi.get('name') or "audio.mp3"

        # Gemini 3.5 Transcribe accepts WAV/MP3/AIFF/AAC/OGG/FLAC/M4A/Opus/WebM
        # directly (up to 1 hour), so the file is uploaded as is. Only the
        # MIME type is normalized to the documented names.
        audio_mime = _gemini_transcribe_mime(audio_mime, audio_name)

        # Upload via Gemini Files API (documented path for transcribe input)
        file_uri = None
        try:
            if not gemini_files_api_enabled:
                pub("error", "Gemini 3.5 Transcribe: Vertex AI モードではファイルアップロードを利用できません。Gemini API モードへ切り替えてください。")
                return BACKGROUND_STREAM_EARLY_RETURN
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_name)[1] or ".wav") as tmp:
                tmp.write(audio_data)
                tmp.flush()
                up = g_client.files.upload(file=tmp.name, config={"mimeType": audio_mime})
            up_name = getattr(up, "name", None) or (up.get("name") if isinstance(up, dict) else None)
            deadline = time.time() + 120
            while time.time() < deadline:
                if isinstance(up, dict):
                    st = up.get("state")
                else:
                    st = getattr(up, "state", None)
                if isinstance(st, dict):
                    st = st.get("name") or st.get("state")
                else:
                    st = getattr(st, "name", None) or st
                if not st or st == "ACTIVE" or st == "FAILED":
                    break
                time.sleep(2)
                try:
                    if up_name:
                        up = g_client.files.get(name=up_name)
                except Exception:
                    break
            if str(st or "").upper() == "FAILED":
                pub("error", "Gemini 3.5 Transcribe: アップロードした音声を処理できませんでした（対応形式か、ファイルが壊れていないか確認してください）。")
                return BACKGROUND_STREAM_EARLY_RETURN
            if isinstance(up, dict):
                file_uri = up.get("uri") or up.get("file_uri") or up.get("fileUri") or up.get("name")
            else:
                file_uri = (
                    getattr(up, "uri", None)
                    or getattr(up, "file_uri", None)
                    or getattr(up, "fileUri", None)
                    or getattr(up, "name", None)
                )
        except Exception as up_e:
            logger.exception("Gemini Transcribe file upload failed")
            pub("error", f"Gemini 3.5 Transcribe: 音声のアップロードに失敗しました: {str(up_e)}")
            return BACKGROUND_STREAM_EARLY_RETURN
        if not file_uri:
            pub("error", "Gemini 3.5 Transcribe: 音声のアップロードに失敗しました（ファイルURIを取得できません）。")
            return BACKGROUND_STREAM_EARLY_RETURN

        transcription_config = {}
        try:
            lang_codes = options.get('transcription_language_codes') or []
            if isinstance(lang_codes, (list, tuple)) and lang_codes:
                transcription_config['language_codes'] = [str(x) for x in lang_codes][:20]
            else:
                transcription_config['language_codes'] = []
            custom_vocab = options.get('transcription_custom_vocabulary') or []
            wants_word_info = bool(options.get('transcription_diarization') or options.get('transcription_word_timestamps'))
            t_mode_opt = str(options.get('transcription_mode') or 'verbatim').lower()
            if isinstance(custom_vocab, (list, tuple)) and custom_vocab:
                if wants_word_info and t_mode_opt != "smart":
                    # The API rejects custom_vocabulary together with
                    # diarization or word timestamps.
                    pub("status", "カスタム語彙は話者分離・単語タイムスタンプと併用できないため、今回は使用しません。")
                else:
                    transcription_config['custom_vocabulary'] = [str(x) for x in custom_vocab][:1000]
            t_mode = t_mode_opt
            if t_mode == "smart":
                transcription_config['mode'] = {"type": "smart"}
            else:
                verbatim_mode = {"type": "verbatim"}
                if options.get('transcription_diarization'):
                    verbatim_mode['diarization_mode'] = "speaker"
                if options.get('transcription_word_timestamps'):
                    verbatim_mode['timestamp_granularities'] = ["word"]
                transcription_config['mode'] = verbatim_mode
        except Exception as cfg_e:
            logger.warning(f"Gemini Transcribe config fallback: {cfg_e}")
            transcription_config = {"language_codes": []}

        pub("status", "文字起こしを実行中です。音声の長さによって時間がかかります...")
        transcript = _gemini_transcribe_rest(
            api_key=key,
            file_uri=file_uri,
            mime_type=audio_mime,
            transcription_config=transcription_config,
        )
        if not transcript or not str(transcript).strip():
            pub("error", "Gemini 3.5 Transcribe: 文字起こし結果が空でした。音声が無声または対応外形式の可能性があります。")
            return BACKGROUND_STREAM_EARLY_RETURN
        transcript = str(transcript).strip()
        full_res += transcript
        pub("content", transcript)
        log_force(f"Gemini Transcribe completed for {job_id} chars={len(transcript)}")
    except Exception as e:
        logger.exception("Gemini Transcribe Error")
        pub("error", f"Gemini 3.5 Transcribe Error: {str(e)}")

    return full_res
