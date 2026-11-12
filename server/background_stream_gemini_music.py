"""background_chat_task Gemini music branch (Lyria 3 / realtime hint).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

def background_stream_gemini_music(local):
    """Generate music with Lyria and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    g_client = local.get('g_client')
    generated_images = local.get('generated_images')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    options = local.get('options')
    pub = local.get('pub')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Gemini Music Branch")
    try:
        if model_key == "lyria-realtime-exp":
            pub("error", "Lyria RealTimeはWebSocketによるリアルタイム生成専用の実験モデルです。プロンプトを入力して送信すると、Lyria RealTime Studioが開きます。")
        else:
            pub("content", "**Generating Music (Lyria)...**\n")
            music_prompt = final_message_text
            if options.get('music_instrumental'):
                music_prompt = f"{music_prompt} Instrumental only, no vocals."
            music_parts = [types.Part(text=music_prompt)]
            for fi in loaded_files:
                if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                    music_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                    if len(music_parts) >= 11:
                        break
            music_cfg = types.GenerateContentConfig(response_modalities=["AUDIO", "TEXT"])
            _mark_provider_request_started()
            m_resp = g_client.models.generate_content(
                model=model_key,
                contents=music_parts,
                config=music_cfg,
            )
            lyrics = []
            audio_data = None
            audio_mime = "audio/mpeg"
            cand0 = m_resp.candidates[0] if m_resp.candidates else None
            for part in (getattr(getattr(cand0, "content", None), "parts", None) or []):
                if part.text:
                    lyrics.append(part.text)
                elif getattr(part, 'inline_data', None) and part.inline_data:
                    audio_data = part.inline_data.data
                    audio_mime = part.inline_data.mime_type or "audio/mpeg"
            if lyrics:
                lyrics_text = "\n".join(lyrics)
                pub("content", lyrics_text + "\n")
                full_res += lyrics_text + "\n"
            if audio_data:
                if isinstance(audio_data, str):
                    audio_data = base64.b64decode(audio_data)
                ext = ".wav" if "wav" in str(audio_mime).lower() else ".mp3"
                fn2 = f"music_{int(time.time())}_{os.urandom(4).hex()}{ext}"
                _save_user_generated_bytes(user_id, bytes(audio_data), fn2, user_config.get('enable_e2ee'))
                audio_tag = f'\n<audio controls src="/files/{user_id}/{fn2}" class="w-full mt-2"></audio>\n'
                pub("content", audio_tag)
                # Save the player as streamed (like TTS) so a reloaded answer keeps it.
                full_res += audio_tag
                generated_images.append(f"{user_id}/{fn2}")
            if not audio_data and not lyrics:
                pub("error", "Lyria returned no output.")
    except Exception as e:
        logger.exception("Gemini Music Gen Error")
        pub("error", f"Gemini Music Gen Error: {str(e)}")
    return full_res
