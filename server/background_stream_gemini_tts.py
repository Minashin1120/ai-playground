"""Gemini TTS branch of `background_chat_task`.

Extracted from `server/background_stream_gemini.py`.  The branch used to run
inline in `background_chat_task`, so it shared that function's locals.  It now
receives a `locals()` snapshot as ``local`` and returns the updated answer text.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_gemini_tts(local):
    """Generate speech and return the updated answer text."""

    _mark_provider_request_started = local.get("_mark_provider_request_started")
    final_message_text = local.get("final_message_text")
    full_res = local.get("full_res")
    g_client = local.get("g_client")
    gemini_backend_mode = local.get("gemini_backend_mode")
    generated_images = local.get("generated_images")
    key = local.get("key")
    model_key = local.get("model_key")
    options = local.get("options")
    pub = local.get("pub")
    re = local.get("re")
    user_config = local.get("user_config")
    user_id = local.get("user_id")

    try:
        voice_name = (options.get('tts_voice') or "Kore").strip()
        if voice_name not in GEMINI_TTS_VOICES:
            voice_name = "Kore"
        tts_lang = (options.get('tts_language') or "").strip() or None
        audio_bytes = b""
        audio_mime = ""
        cand0 = None
        _mark_provider_request_started()
        if is_gemini_interactions_tts_model_key(model_key) and gemini_backend_mode != "vertex_ai":
            # Gemini 3.8 TTS: verbatim text, delivery style in
            # speech_metadata, custom voice IDs via Interactions.
            # (Vertex AI keeps generateContent with prebuilt voices.)
            audio_bytes, audio_mime = _gemini_tts_interactions_rest(
                key, model_key, final_message_text,
                _gemini_tts_voice(voice_name, options.get('tts_voice_custom')),
                style=options.get('tts_style'),
            )
        else:
            tts_resp = g_client.models.generate_content(
                model=model_key,
                contents=final_message_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        ),
                        language_code=tts_lang
                    ),
                ),
            )
            cand0 = tts_resp.candidates[0] if tts_resp.candidates else None
            parts0 = getattr(getattr(cand0, "content", None), "parts", None) or []
            # Long text can come back as several audio parts; join them all.
            for p0 in parts0:
                inline = getattr(p0, 'inline_data', None)
                if not inline or not inline.data:
                    continue
                data = inline.data
                if isinstance(data, str):
                    data = base64.b64decode(data)
                audio_bytes += bytes(data)
                audio_mime = audio_mime or str(getattr(inline, "mime_type", "") or "")

        if not audio_bytes:
            finish = getattr(cand0, "finish_reason", None) if cand0 else None
            pub("error", f"Gemini TTS Error: No audio data returned.{f' (finish_reason: {finish})' if finish else ''}")
        else:
            if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
                # Already a WAV file (newer TTS responses).
                wav_bytes = audio_bytes
            else:
                # Headerless 16-bit PCM (audio/L16;codec=pcm;rate=24000).
                rate_match = re.search(r"rate=(\d+)", audio_mime)
                pcm_rate = int(rate_match.group(1)) if rate_match else 24000
                buf = BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(pcm_rate)
                    wf.writeframes(audio_bytes)
                wav_bytes = buf.getvalue()

            speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.wav"
            _save_user_generated_bytes(
                user_id, wav_bytes, speech_file_name, user_config.get('enable_e2ee')
            )

            audio_url = f"/files/{user_id}/{speech_file_name}"
            audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
            full_res += audio_tag
            pub("content", audio_tag)
            generated_images.append(f"{user_id}/{speech_file_name}")
    except Exception as e:
        logger.exception("Gemini TTS Error")
        pub("error", f"Gemini TTS Error: {str(e)}")

    return full_res
