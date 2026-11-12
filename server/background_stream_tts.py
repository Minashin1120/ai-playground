"""background_chat_task TTS branch (Google Cloud TTS, xAI TTS, OpenAI TTS).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot
as ``local`` and returns the updated answer text (`full_res`), because the
branch appends the generated `<audio>` tag to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_tts(local):
    """Generate speech for a TTS model and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    model_api_key_override = local.get('model_api_key_override')
    model_key = local.get('model_key')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    user = local.get('user')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    try:
        pub("content", "**Processing Audio Generation...**\n")

        speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.mp3"
        audio_content = None

        if 'google-tts' in model_key:
            # Google Cloud TTS (requires Google Cloud API key, not Gemini API key)
            g_key = model_api_key_override or decrypt_val(user.google_api_key)
            if not g_key and _admin_env_fallback_enabled(user):
                g_key = os.getenv('GOOGLE_API_KEY')
            if not g_key:
                raise RuntimeError("Google API Key is not configured for Google TTS.")
            g_project = decrypt_val(user.google_cloud_project)
            if not g_project and _admin_env_fallback_enabled(user):
                g_project = os.getenv('GOOGLE_CLOUD_PROJECT')
            opts = {"api_key": g_key}
            if g_project: opts["quota_project_id"] = g_project
            client_tts = texttospeech.TextToSpeechClient(
                client_options=ClientOptions(**opts)
            )
            synthesis_input = texttospeech.SynthesisInput(text=final_message_text)
            tts_lang = (options.get('tts_language') or "ja-JP").strip() or "ja-JP"
            tts_voice_custom = (options.get('tts_voice_custom') or "").strip()

            # Selection logic
            if tts_voice_custom:
                voice = texttospeech.VoiceSelectionParams(language_code=tts_lang, name=tts_voice_custom)
            elif 'studio' in model_key:
                voice = pick_tts_voice(client_tts, tts_lang, "studio")
            else:
                voice = pick_tts_voice(client_tts, tts_lang, "neural")

            speed_val = clamp_float(options.get('tts_speed'), 0.25, 2.0)
            audio_kwargs = {"audio_encoding": texttospeech.AudioEncoding.MP3}
            if speed_val is not None:
                audio_kwargs["speaking_rate"] = speed_val
            audio_config = texttospeech.AudioConfig(**audio_kwargs)
            response_tts = client_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            audio_content = response_tts.audio_content

        elif 'grok-tts' in model_key or 'xai-tts' in model_key:
            # xAI standalone TTS (単独モデル化)
            xai_key = model_api_key_override or decrypt_val(user.xai_api_key)
            if not xai_key and _admin_env_fallback_enabled(user):
                xai_key = os.getenv('XAI_API_KEY')
            if not xai_key:
                raise RuntimeError("xAI API Key is not configured for Grok TTS.")

            raw_voice = (options.get('tts_voice') or "eve").strip()
            custom_voice = (options.get('tts_voice_custom') or "").strip()
            # xAI voice IDs are case-insensitive; the docs use lowercase.
            # A custom voice ID (Custom Voices API) overrides the preset.
            # (A leftover Google voice name such as "ja-JP-Neural2-B" is ignored.)
            if (custom_voice and re.fullmatch(r"[A-Za-z0-9_\-]{1,128}", custom_voice)
                    and not re.match(r"[a-z]{2,3}-[A-Za-z]{2,4}-", custom_voice)):
                xai_voice = custom_voice
            else:
                xai_voice = raw_voice.lower()
                if xai_voice not in ("eve", "ara", "rex", "sal", "leo"):
                    xai_voice = "eve"

            tts_lang = (options.get('tts_language') or "ja").strip() or "ja"
            speed_val = clamp_float(options.get('tts_speed'), 0.7, 1.5)

            payload = {
                "text": final_message_text,
                "voice_id": xai_voice,
                "language": tts_lang,
            }
            if speed_val is not None:
                payload["speed"] = speed_val

            headers = {
                "Authorization": f"Bearer {xai_key}",
                "Content-Type": "application/json"
            }
            tts_url = f"https://{_XAI_API_HOST}/v1/tts"
            resp = requests.post(tts_url, headers=headers, json=payload, timeout=180, stream=True)
            if resp.status_code >= 400:
                detail = ""
                try:
                    err_body = resp.json()
                    detail = err_body.get("error") or err_body.get("message") or ""
                    if isinstance(detail, dict):
                        detail = detail.get("message") or ""
                except Exception:
                    detail = (resp.text or "")[:300]
                raise RuntimeError(f"xAI TTS error {resp.status_code}: {detail}".strip())
            audio_buf = bytearray()
            for audio_chunk in resp.iter_content(chunk_size=64 * 1024):
                if not audio_chunk:
                    continue
                audio_buf.extend(audio_chunk)
                if len(audio_buf) > _AUDIO_INPUT_MAX_BYTES:
                    raise ValueError("Generated audio is too large")
            audio_content = bytes(audio_buf)
            if "json" in (resp.headers.get("Content-Type") or "").lower():
                # JSON envelope (timestamps mode): base64 audio inside.
                audio_content = base64.b64decode(json.loads(audio_content.decode("utf-8")).get("audio") or "")

        else:
            # OpenAI TTS (default for generic tts models)
            tts_voice = (options.get('tts_voice') or "alloy").strip().lower() or "alloy"
            speed_val = clamp_float(options.get('tts_speed'), 0.25, 4.0)
            tts_kwargs = {
                "model": model_key,
                "voice": tts_voice,
                "input": final_message_text
            }
            if speed_val is not None:
                tts_kwargs["speed"] = speed_val

            _mark_provider_request_started()
            with o_client.audio.speech.with_streaming_response.create(**tts_kwargs) as response:
                audio_buf = bytearray()
                for audio_chunk in response.iter_bytes(chunk_size=64 * 1024):
                    audio_buf.extend(audio_chunk)
                    if len(audio_buf) > _AUDIO_INPUT_MAX_BYTES:
                        raise ValueError("Generated audio is too large")
                audio_content = bytes(audio_buf)

        _save_user_generated_bytes(
            user_id, audio_content, speech_file_name, user_config.get('enable_e2ee')
        )

        audio_url = f"/files/{user_id}/{speech_file_name}"
        audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'

        full_res += audio_tag
        pub("content", audio_tag)
        generated_images.append(f"{user_id}/{speech_file_name}")

    except Exception as e:
        pub("error", f"TTS Error: {str(e)}")
    return full_res
