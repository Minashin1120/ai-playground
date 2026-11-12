"""background_chat_task Gemini video branch (Veo 3.1 / Omni Flash polling).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

def background_stream_gemini_video(local):
    """Generate a video with Gemini and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    check_stop = local.get('check_stop')
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

    log_force("Routing: Gemini Video Branch")
    try:
        pub("content", "**Generating Video (Gemini)...**\n")
        try:
            video_duration = int(options.get('gemini_video_duration') or 8)
        except Exception:
            video_duration = 8
        if video_duration < 1:
            video_duration = 1
        if video_duration > 8:
            video_duration = 8
        aspect_ratio = str(options.get('gemini_video_aspect') or "16:9")
        if aspect_ratio not in ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"):
            aspect_ratio = "16:9"
        resolution = str(options.get('gemini_video_resolution') or "720p")
        if resolution not in ("480p", "720p", "1080p", "4K"):
            resolution = "720p"
        if resolution == "4K" and model_key != "veo-3.1-generate-preview":
            resolution = "1080p"

        # Image-to-video support: use the first attached image if present
        src_image = None
        for fi in loaded_files:
            if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                src_image = types.Image(imageBytes=fi['bytes'], mimeType=fi['mime'])
                break
        video_source = types.GenerateVideosSource(
            prompt=final_message_text,
            image=src_image,
        )

        _mark_provider_request_started()
        op = g_client.models.generate_videos(
            model=model_key,
            source=video_source,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=video_duration,
                number_of_videos=1,
            ),
        )
        pub("content", "生成中です。数分かかる場合があります...\n")
        max_polls = 120
        done_op = None
        for i in range(max_polls):
            if check_stop():
                break
            time.sleep(5)
            op = g_client.operations.get(op)
            if op.done:
                done_op = op
                break
        if done_op is None:
            raise RuntimeError("Video generation timed out.")
        if getattr(done_op, "error", None):
            raise RuntimeError(str(done_op.error))
        gen_videos = (done_op.result.generated_videos or []) if done_op.result else []
        if not gen_videos:
            raise RuntimeError("No video output returned.")
        video_uri = gen_videos[0].video.uri if gen_videos[0].video else None
        if not video_uri:
            raise RuntimeError("No video URI in response.")
        _mark_provider_request_started()
        video_bytes = _download_public_https_bytes(video_uri, 256 * 1024 * 1024, timeout=180.0)
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
