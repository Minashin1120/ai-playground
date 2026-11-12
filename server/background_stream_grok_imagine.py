"""background_chat_task Grok Imagine image branch (generation and multi-image edits).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

def background_stream_grok_imagine(local):
    """Generate or edit images with Grok Imagine and return the updated answer text."""

    _build_non_llm_image_context = local.get('_build_non_llm_image_context')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    key = local.get('key')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Grok Imagine Branch")
    try:
        pub("status", "画像を生成中...")

        aspect_ratio = options.get('grok_image_aspect') or "1:1"
        resolution = str(options.get('grok_image_resolution') or "1k").lower().strip()
        if resolution not in ("1k", "2k"):
            resolution = "1k"
        quality = str(options.get('grok_image_quality') or "medium").lower().strip()
        if quality not in ("low", "medium"):
            quality = "medium"
        image_count = 1
        try:
            image_count = max(1, min(10, int(options.get('grok_image_count') or 1)))
        except (TypeError, ValueError):
            pass
        img_response_format = str(options.get('grok_image_format') or "b64_json").lower().strip()
        if img_response_format not in ("url", "b64_json"):
            img_response_format = "b64_json"
        grok_supports_resolution = model_key in (
            "grok-imagine-image-2.0",
            "grok-imagine-image-quality",
        )
        grok_supports_quality = model_key == "grok-imagine-image-2.0"
        grok_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)

        img_kwargs = {
            "model": model_key,
            "prompt": grok_prompt,
            "n": image_count,
            "response_format": img_response_format
        }
        # aspect_ratio / resolution / quality are xAI-specific; pass via extra_body
        eb = {}
        if aspect_ratio:
            eb["aspect_ratio"] = aspect_ratio
        if grok_supports_resolution:
            eb["resolution"] = resolution
        if grok_supports_quality:
            eb["quality"] = quality

        img_inputs = []
        for fi in loaded_files:
            if not fi.get('bytes') or not fi.get('mime', '').startswith('image/'):
                continue
            img_bytes = fi['bytes']
            img_mime = fi['mime']
            # xAI supports jpg/jpeg or png.
            if img_mime not in ('image/png', 'image/jpeg'):
                converted_png = _convert_image_to_png(img_bytes)
                if converted_png:
                    img_bytes = converted_png
                    img_mime = 'image/png'
            img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"input_{len(img_inputs)}")
            img_inputs.append((img_name, img_bytes, img_mime))
        for hp in history_image_parts:
            ref = hp.get("ref")
            if any(existing[0] == os.path.basename(ref or "") for existing in img_inputs):
                continue
            img_inputs.append((hp['name'], hp['bytes'], hp['mime']))

        img_data_b64 = None
        if img_inputs:
            # xAI image edits expect JSON (not multipart), so send base64.
            # Multi-image editing accepts up to 3 references via `images`.
            image_payloads = []
            for img_entry in img_inputs[:3]:
                img_bytes = img_entry[1]
                img_mime = img_entry[2] if len(img_entry) > 2 else "image/png"
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_payloads.append({
                    "url": f"data:{img_mime};base64,{img_b64}",
                    "type": "image_url",
                })
            endpoint = f"https://{_XAI_API_HOST}/v1/images/edits"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "model": model_key,
                "prompt": grok_prompt,
                "n": image_count,
                "response_format": img_response_format
            }
            if len(image_payloads) == 1:
                payload["image"] = image_payloads[0]
            else:
                payload["images"] = image_payloads
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            if grok_supports_resolution:
                payload["resolution"] = resolution
            if grok_supports_quality:
                payload["quality"] = quality
            _mark_provider_request_started()
            resp = httpx.post(endpoint, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 400:
                try:
                    log_force(f"Grok Imagine edit error {resp.status_code}: {resp.text}")
                except Exception:
                    pass
            resp.raise_for_status()
            resp_json = resp.json()
            image_results = []
            if isinstance(resp_json, dict):
                image_results = [item or {} for item in (resp_json.get("data") or [])]
                if image_results:
                    img_data_b64 = image_results[0].get("b64_json")
                if not img_data_b64 and resp_json.get("image"):
                    img_data_b64 = resp_json.get("image")
        else:
            _mark_provider_request_started()
            resp = o_client.images.generate(**img_kwargs, extra_body=eb)
            image_results = []
            for item in (getattr(resp, "data", None) or []):
                image_results.append({
                    "b64_json": getattr(item, "b64_json", None),
                    "url": getattr(item, "url", None),
                })
            if image_results:
                img_data_b64 = image_results[0].get("b64_json")

        if img_data_b64 and not image_results:
            image_results = [{"b64_json": img_data_b64}]
        saved_any = False
        for image_index, image_result in enumerate(image_results[:image_count]):
            image_b64 = image_result.get("b64_json")
            image_url = image_result.get("url")
            if image_b64:
                img_bytes = _decode_base64_limited(image_b64, 50 * 1024 * 1024)
            elif image_url:
                img_bytes = _download_public_https_bytes(image_url, 50 * 1024 * 1024, timeout=120.0)
            else:
                continue
            ext = "png"
            fn2 = f"gen_grok_{int(time.time())}_{len(generated_images)}_{image_index}.{ext}"
            _save_user_generated_bytes(
                user_id, img_bytes, fn2, user_config.get('enable_e2ee')
            )

            generated_images.append(f"{user_id}/{fn2}")
            pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
            full_res += f"Generated Image {image_index + 1} for: {final_message_text}\n"
            saved_any = True
        if not saved_any:
            pub("error", "Grok Image Gen Error: No data returned.")
    except Exception as e:
        logger.exception("Grok Imagine Error")
        err_body = ""
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            err_body = e.response.text
        elif hasattr(e, 'body'): # OpenAI SDK errors
            err_body = str(e.body)

        err_msg = str(e)
        if "content moderation" in err_msg.lower() or "content moderation" in err_body.lower():
            err_msg = "不適切な内容が含まれている可能性があるため、xAIの安全フィルタにより画像生成が拒否されました。プロンプトをより一般的な表現に変更して、再度お試しください。"
        elif err_body:
            err_msg = f"{err_msg} - {err_body}"

        pub("error", f"Grok Imagine Error: {err_msg}")
    return full_res
