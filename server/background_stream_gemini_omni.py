"""background_chat_task Gemini Omni 1.1 Flash video branch (Interactions API).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

def background_stream_gemini_omni(local):
    """Generate a video with Gemini Omni and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    g_client = local.get('g_client')
    gemini_backend_mode = local.get('gemini_backend_mode')
    generated_images = local.get('generated_images')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    options = local.get('options')
    pub = local.get('pub')
    re = local.get('re')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Gemini Omni 1.1 Flash (Interactions API)")
    try:
        if gemini_backend_mode != "gemini_api":
            pub("error", "Gemini Omni 1.1 Flash は Gemini API バックエンドでのみ利用できます（Vertex AI バックエンドでは利用不可）。")
        else:
            pub("content", "**Generating Video (Gemini Omni 1.1 Flash)...**\n")
            aspect_ratio = str(options.get('gemini_video_aspect') or "16:9")
            if aspect_ratio not in ("16:9", "9:16"):
                aspect_ratio = "16:9"
            resolution = str(options.get('gemini_video_resolution') or "720p")
            if resolution == "4K":
                resolution = "4k"
            elif resolution not in ("360p", "720p", "1080p", "4k"):
                resolution = "720p"

            # Build Interactions API input parts: attached images + text prompt.
            # Omni 1.1 accepts text / image / video / audio inputs via the Interactions API.
            interaction_input = []
            for fi in loaded_files:
                if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                    interaction_input.append({
                        "type": "image",
                        "data": base64.b64encode(fi['bytes']).decode('ascii'),
                        "mime_type": str(fi.get('mime') or 'image/jpeg'),
                    })
            interaction_input.append({"type": "text", "text": final_message_text})

            # Synchronous unary generation (background=false / store=false / stream=false).
            # Use URI delivery so videos larger than 4MB are not clipped by payload limits.
            _mark_provider_request_started()
            interaction = g_client.interactions.create(
                model=model_key,
                input=interaction_input,
                response_format={
                    "type": "video",
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "delivery": "uri",
                },
                background=False,
                store=False,
                stream=False,
                timeout=600.0,
            )
            if getattr(interaction, "status", None) == "failed":
                raise RuntimeError("Omni 1.1 video generation failed.")
            video_data = None
            video_uri = None
            for out in (getattr(interaction, "outputs", None) or []):
                if getattr(out, "type", None) == "video":
                    video_data = getattr(out, "data", None)
                    video_uri = getattr(out, "uri", None)
                    break
            if video_data:
                video_bytes = base64.b64decode(video_data)
            elif video_uri:
                file_id_match = re.search(r'files/([A-Za-z0-9_-]+)', video_uri)
                if file_id_match:
                    for _i in range(15):
                        finfo = g_client.files.get(name=f"files/{file_id_match.group(1)}")
                        if str(getattr(getattr(finfo, "state", None), "name", "")) == "ACTIVE":
                            break
                        time.sleep(2)
                _mark_provider_request_started()
                video_bytes = g_client.files.download(file=video_uri)
            else:
                raise RuntimeError("No video output returned from Omni 1.1.")
            if not video_bytes:
                raise RuntimeError("Video download failed.")
            fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
            _save_user_generated_bytes(user_id, video_bytes, fn2, user_config.get('enable_e2ee'))
            video_tag = f'\n<video controls src="/files/{user_id}/{fn2}" class="w-full max-w-2xl rounded-lg"></video>\n'
            pub("content", video_tag)
            full_res += f"Generated Video for: {final_message_text}\n"
            generated_images.append(f"{user_id}/{fn2}")
    except Exception as e:
        logger.exception("Gemini Video Gen Error")
        pub("error", f"Gemini Video Gen Error: {str(e)}")
    return full_res
