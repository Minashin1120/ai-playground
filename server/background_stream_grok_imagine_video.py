"""background_chat_task Grok Imagine video branch (generation and polling).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

def background_stream_grok_imagine_video(local):
    """Generate a video with Grok Imagine and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    key = local.get('key')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    options = local.get('options')
    pub = local.get('pub')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Grok Video Branch")
    try:
        pub("content", "**Generating Video (Grok)...**\n")

        # Prepare params
        duration = None
        try:
            duration = int(options.get('grok_video_duration') or 5)
        except: duration = 5

        aspect_ratio = options.get('grok_video_aspect') or "16:9"
        resolution = options.get('grok_video_resolution') or "720p"
        if resolution not in ("480p", "720p", "1080p"):
            resolution = "720p"
        if resolution == "1080p" and model_key not in ("grok-imagine-video-1.5", "grok-imagine-video-1.5-lite"):
            resolution = "720p"

        api_key = key # Decrypted XAI API Key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Determine endpoint and payload
        endpoint = f"https://{_XAI_API_HOST}/v1/videos/generations"
        payload = {
            "model": model_key,
            "prompt": final_message_text,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution
        }

        # Check for image or video inputs
        img_urls = []
        vid_urls = []
        for fi in loaded_files:
            if fi.get('bytes'):
                # For simplicity, if we have local bytes, we might need to upload them to a public URL 
                # or use data URIs if supported. The docs say:
                # "Note: The input video URL must be a direct, publicly accessible link to the video file."
                # This is a limitation for local files.
                # However, for Image-to-Video, the docs show:
                # image: { url: '<url of the image>' }
                # But also curl example shows "image": {"url": "<url of the image>"}
                # Wait, can we use base64? 
                # image-gen docs showed base64 support for Image.
                # Let's try base64 for image-to-video.
                mime = fi.get('mime', 'image/png')
                if mime.startswith('image/'):
                    b64 = base64.b64encode(fi['bytes']).decode('utf-8')
                    payload["image"] = {"url": f"data:{mime};base64,{b64}"}
                    image_info = _probe_image(fi['bytes'])
                    if image_info:
                        inferred = _closest_aspect_ratio(image_info["width"], image_info["height"], {"16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3"})
                        if inferred:
                            payload["aspect_ratio"] = inferred
                elif mime.startswith('video/'):
                    # Video edit requires a public URL. Local files won't work easily here.
                    # But we'll try to provide it if we had a public URL.
                    pass

        # Send request
        _mark_provider_request_started()
        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=60.0)
        if resp.status_code != 200:
            raise RuntimeError(f"xAI API Error: {resp.status_code} - {resp.text}")

        data = resp.json()
        request_id = data.get("request_id")
        if not request_id:
            raise RuntimeError(f"No request_id returned: {data}")

        pub("content", f"Request ID: `{request_id}`. Polling for result...\n")

        # Polling
        poll_url = f"https://{_XAI_API_HOST}/v1/videos/{request_id}"
        max_polls = 300 # 10 minutes if 2s interval
        video_url = None
        for i in range(max_polls):
            if check_stop(): break
            time.sleep(2)
            _mark_provider_request_started()
            p_resp = httpx.get(poll_url, headers=headers, timeout=30.0)
            if p_resp.status_code == 200:
                p_data = p_resp.json()
                status = p_data.get("status")
                # xAI Video API might return URL nested inside "video" object
                video_url = p_data.get("url")
                if not video_url and isinstance(p_data.get("video"), dict):
                    video_url = p_data["video"].get("url")

                if status == "completed" or video_url:
                    break
                elif status == "failed":
                    raise RuntimeError(f"Video generation failed: {p_data.get('error')}")
                else:
                    if i % 5 == 0: # Log every 10s
                        log_force(f"Polling video {request_id}: status={status}, has_url={bool(video_url)}")
            elif p_resp.status_code != 200:
                log_force(f"Polling error {p_resp.status_code}: {p_resp.text}")

        if video_url:
            # Download and save the video locally
            _mark_provider_request_started()
            try:
                video_bytes = _download_public_https_bytes(video_url, 128 * 1024 * 1024, timeout=60.0)
            except Exception as download_error:
                video_bytes = None
                log_force(f"Grok video download rejected: {download_error}")
            if video_bytes:
                fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
                _save_user_generated_bytes(
                    user_id, video_bytes, fn2, user_config.get('enable_e2ee')
                )

                generated_images.append(f"{user_id}/{fn2}")
                vid_tag = f'\n<video controls playsinline preload="metadata" src="/files/{user_id}/{fn2}" class="w-full mt-2"></video>\n'
                pub("content", vid_tag)
                full_res += f"Generated Video for: {final_message_text}\n"
            else:
                pub("error", "Failed to download generated video safely.")
        else:
            pub("error", "Video generation timed out or was canceled.")

    except Exception as e:
        logger.exception("Grok Imagine Video Error")
        pub("error", f"Grok Imagine Video Error: {str(e)}")
    return full_res
