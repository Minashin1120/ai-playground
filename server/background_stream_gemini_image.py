"""Gemini image generation branch of `background_chat_task`.

Extracted from `server/background_stream_gemini.py`.  The branch used to run
inline in `background_chat_task`, so it shared that function's locals.  It now
receives a `locals()` snapshot as ``local`` and returns the updated answer text.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to the caller when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_gemini_image(local):
    """Generate images and return the updated answer text."""

    _build_non_llm_image_context = local.get("_build_non_llm_image_context")
    _mark_provider_request_started = local.get("_mark_provider_request_started")
    _persist_batch_failure = local.get("_persist_batch_failure")
    _submit_gemini_batch = local.get("_submit_gemini_batch")
    batch_mode = local.get("batch_mode")
    final_message_text = local.get("final_message_text")
    full_res = local.get("full_res")
    g_client = local.get("g_client")
    gemini_backend_mode = local.get("gemini_backend_mode")
    generated_images = local.get("generated_images")
    job_id = local.get("job_id")
    loaded_files = local.get("loaded_files")
    model_key = local.get("model_key")
    options = local.get("options")
    pub = local.get("pub")
    thread_id = local.get("thread_id")
    user_config = local.get("user_config")
    user_id = local.get("user_id")

    try:
        def _collect_gemini_image_output_parts(resp_obj, keep_only_last_image=False):
            text_chunks = []
            image_parts = []
            seen_part_ids = set()

            def _append_parts(parts_seq):
                for _part in parts_seq or []:
                    # Gemini 3 image models can return intermediate thought
                    # images/text. They are reasoning artifacts, not user-facing
                    # output, and must not be saved as the final generated image.
                    if bool(getattr(_part, "thought", False)):
                        continue
                    part_id = id(_part)
                    if part_id in seen_part_ids:
                        continue
                    seen_part_ids.add(part_id)
                    if hasattr(_part, 'text') and _part.text:
                        txt = str(_part.text)
                        if txt.strip():
                            text_chunks.append(txt)
                    if hasattr(_part, 'inline_data') and _part.inline_data:
                        image_parts.append(_part)

            _append_parts(getattr(resp_obj, 'parts', None) or [])
            for cand in getattr(resp_obj, 'candidates', None) or []:
                _append_parts(getattr(getattr(cand, 'content', None), 'parts', None) or [])

            if keep_only_last_image and len(image_parts) > 1:
                image_parts = [image_parts[-1]]
            return text_chunks, image_parts

        def _save_gemini_image_part(part_obj):
            mime = getattr(part_obj.inline_data, "mime_type", None) or "image/png"
            ext_map = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/webp": "webp"
            }
            ext = ext_map.get(mime, "png")
            fn2 = f"gen_{int(time.time())}_{len(generated_images)}.{ext}"
            img_data = part_obj.inline_data.data
            if isinstance(img_data, str):
                img_data = _decode_base64_limited(img_data, 50 * 1024 * 1024)
            _save_user_generated_bytes(
                user_id, img_data, fn2, user_config.get('enable_e2ee')
            )
            generated_images.append(f"{user_id}/{fn2}")
            return fn2

        # [FIX] Apply System Prompt to Image Prompts if available
        img_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
        if options.get('system_prompt'):
            img_prompt = f"{options.get('system_prompt')}\n\n{img_prompt}"

        mk_lower = str(model_key or "").lower()
        if mk_lower == "gemini-nano-banana-2.1":
            img_model = "gemini-nano-banana-2.1"
        elif "gemini-3.1-flash-lite-image" in mk_lower:
            img_model = "gemini-3.1-flash-lite-image"
        elif "gemini-3.1-flash-image" in mk_lower:
            img_model = "gemini-3.1-flash-image"
        elif "2.5" in mk_lower:
            img_model = "gemini-2.5-flash-image"
        else:
            img_model = "gemini-3-pro-image"
        aspect_allowed = {
            "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
            "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
            "auto"
        }
        size_allowed = {"1K", "2K", "4K"}
        aspect_val = options.get('gemini_image_aspect')
        if aspect_val:
            aspect_val = str(aspect_val).strip()
            if aspect_val not in aspect_allowed or aspect_val == "auto":
                aspect_val = None
        size_val = options.get('gemini_image_size')
        if size_val:
            size_val = str(size_val).strip().upper()
            if size_val not in size_allowed:
                size_val = None
        image_cfg_kwargs = {}
        if aspect_val:
            image_cfg_kwargs["aspect_ratio"] = aspect_val
        if img_model == "gemini-3.1-flash-lite-image":
            # Nano Banana 2 Lite supports 1K output only.
            image_cfg_kwargs["image_size"] = "1K"
        elif size_val and (
            "gemini-3-pro-image" in img_model
            or img_model in ("gemini-3.1-flash-image", "gemini-nano-banana-2.1")
        ):
            image_cfg_kwargs["image_size"] = size_val
        config_kwargs = {
            "temperature": 0.7,
            "candidate_count": 1,
            "response_modalities": ["TEXT", "IMAGE"],
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
            ]
        }
        # Flex is offered for Gemini 3 Pro Image and 2.5 Flash Image. It is a
        # synchronous, paid-only tier and is meaningless with Batch.
        flex_image_active = bool(
            options.get('gemini_flex')
            and not options.get('gemini_free_tier')
            and not batch_mode
            and img_model in ("gemini-3-pro-image", "gemini-2.5-flash-image")
        )
        if flex_image_active:
            config_kwargs["service_tier"] = types.ServiceTier.FLEX
            config_kwargs["http_options"] = types.HttpOptions(timeout=_GEMINI_FLEX_TIMEOUT_MS)
            log_force(f"Gemini Flex tier enabled for image job={job_id} model={img_model}")

        def _gemini_image_generate(contents, cfg):
            """Generate one image, retrying Flex shedable errors with backoff."""
            attempt = 0
            while True:
                try:
                    return g_client.models.generate_content(
                        model=img_model,
                        contents=contents,
                        config=types.GenerateContentConfig(**cfg),
                    )
                except Exception as exc:
                    text = str(exc)
                    retryable = flex_image_active and (
                        "503" in text or "429" in text
                        or "UNAVAILABLE" in text or "RESOURCE_EXHAUSTED" in text
                        or "504" in text or "DEADLINE_EXCEEDED" in text
                    )
                    if not retryable or attempt >= _GEMINI_FLEX_RETRIES:
                        raise
                    attempt += 1
                    delay = min(30, 5 * (2 ** (attempt - 1)))
                    log_force(
                        f"Gemini Flex image busy, retry {attempt}/{_GEMINI_FLEX_RETRIES} "
                        f"in {delay}s for {job_id}: {text[:200]}"
                    )
                    time.sleep(delay)
        if img_model in ("gemini-nano-banana-2.1", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"):
            default_level = (
                "medium" if img_model == "gemini-nano-banana-2.1"
                else "minimal" if img_model == "gemini-3.1-flash-lite-image"
                else "high"
            )
            raw_lvl = str(options.get('thinking_level') or default_level).lower()
            if img_model == "gemini-nano-banana-2.1":
                nano_banana2_lvl = raw_lvl if raw_lvl in ("minimal", "medium", "high") else default_level
            elif raw_lvl in ("low", "minimal"):
                nano_banana2_lvl = "minimal"
            elif raw_lvl in ("medium", "high"):
                nano_banana2_lvl = "high"
            else:
                nano_banana2_lvl = default_level
            # Gemini 3.1 Flash image models support minimal/high; Nano Banana 2.1 also supports medium.
            # The UI checkbox controls thought output visibility; internal thinking remains model-driven.
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=bool(options.get('enable_thinking')),
                thinking_level=nano_banana2_lvl
            )
            # Google Search grounding is supported by Nano Banana 2,
            # but explicitly unsupported by Nano Banana 2 Lite.
            if img_model != "gemini-3.1-flash-lite-image" and options.get('enable_search'):
                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        if image_cfg_kwargs:
            config_kwargs["image_config"] = types.ImageConfig(**image_cfg_kwargs)

        _mark_provider_request_started()
        gemini_image_parts = []
        history_image_refs_included = set()
        def _upload_nano_banana_input_file(fi, mime, label):
            data = fi.get('bytes')
            if not data:
                return None
            file_name = os.path.basename(fi.get('send_name') or fi.get('name') or fi.get('path') or label)
            suffix = os.path.splitext(file_name)[1] or ('.mp4' if mime.startswith('video/') else '.pdf')
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(data)
                tmp.flush()
                uploaded = g_client.files.upload(
                    file=tmp.name,
                    config={"mime_type": mime, "display_name": file_name[:128]},
                )
            uploaded_name = getattr(uploaded, 'name', None)
            state_obj = getattr(uploaded, 'state', None)
            state = getattr(state_obj, 'name', None) or state_obj
            deadline = time.time() + 120
            while state == 'PROCESSING' and uploaded_name and time.time() < deadline:
                time.sleep(2)
                uploaded = g_client.files.get(name=uploaded_name)
                state_obj = getattr(uploaded, 'state', None)
                state = getattr(state_obj, 'name', None) or state_obj
            if state and state != 'ACTIVE':
                raise RuntimeError(f"Gemini {label} processing failed: {state}")
            uri = getattr(uploaded, 'uri', None) or uploaded_name
            if not uri:
                raise RuntimeError(f"Gemini {label} upload returned no URI")
            return types.Part.from_uri(uri, mime_type=mime)

        image_reference_count = 0
        for fi in loaded_files:
            mime = str(fi.get('mime') or '')
            if fi.get('bytes') and mime.startswith('image/'):
                if img_model in ("gemini-3.1-flash-lite-image", "gemini-nano-banana-2.1") and image_reference_count >= 14:
                    continue
                gemini_image_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=mime))
                image_reference_count += 1
            elif img_model == "gemini-nano-banana-2.1" and fi.get('bytes') and mime.startswith('video/'):
                if gemini_backend_mode != "gemini_api":
                    raise RuntimeError("動画を使った画像生成にはGemini APIバックエンドが必要です。")
                pub("status", "動画をGeminiへアップロード中...")
                video_part = _upload_nano_banana_input_file(fi, mime, "video")
                if video_part:
                    gemini_image_parts.append(video_part)
            elif img_model == "gemini-nano-banana-2.1" and fi.get('bytes') and (mime == 'application/pdf' or fi.get('is_pdf')):
                if len(fi['bytes']) <= 20 * 1024 * 1024:
                    gemini_image_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=mime or 'application/pdf'))
                else:
                    if gemini_backend_mode != "gemini_api":
                        raise RuntimeError("大容量PDFを使うにはGemini APIバックエンドが必要です。")
                    pub("status", "PDFをGeminiへアップロード中...")
                    pdf_part = _upload_nano_banana_input_file(fi, mime or 'application/pdf', "PDF")
                    if pdf_part:
                        gemini_image_parts.append(pdf_part)
        for hp in history_image_parts:
            if img_model in ("gemini-3.1-flash-lite-image", "gemini-nano-banana-2.1") and image_reference_count >= 14:
                break
            ref = hp.get("ref")
            if ref and ref in history_image_refs_included:
                continue
            gemini_image_parts.append(types.Part.from_bytes(data=hp['bytes'], mime_type=hp['mime']))
            image_reference_count += 1
            if ref:
                history_image_refs_included.add(ref)

        if batch_mode:
            _mark_provider_request_started()
            try:
                _submit_gemini_batch(
                    img_model,
                    [*gemini_image_parts, types.Part(text=img_prompt)],
                    config_kwargs,
                )
            except Exception as batch_error:
                logger.exception('Gemini Image Batch submission failed')
                _persist_batch_failure(str(batch_error))
            return BACKGROUND_STREAM_EARLY_RETURN

        resp = _gemini_image_generate(
            [
                *gemini_image_parts,
                types.Part(text=img_prompt)
            ],
            config_kwargs
        )

        text_outputs, image_outputs = _collect_gemini_image_output_parts(
            resp,
            keep_only_last_image=(img_model in ("gemini-nano-banana-2.1", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"))
        )

        if not image_outputs and img_model in ("gemini-nano-banana-2.1", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"):
            log_force(
                f"Nano Banana returned text-only output; retrying with image-only mode. "
                f"thread={thread_id} job={job_id}"
            )
            retry_cfg_kwargs = dict(config_kwargs)
            retry_cfg_kwargs.pop("tools", None)
            retry_cfg_kwargs["response_modalities"] = ["IMAGE"]
            retry_prompt = (
                f"{img_prompt}\n\n"
                "Return an image for this request. Do not answer with text only."
            )
            retry_resp = _gemini_image_generate(
                [
                    *gemini_image_parts,
                    types.Part(text=retry_prompt)
                ],
                retry_cfg_kwargs
            )
            retry_text_outputs, retry_image_outputs = _collect_gemini_image_output_parts(
                retry_resp,
                keep_only_last_image=True
            )
            if retry_image_outputs:
                text_outputs, image_outputs = retry_text_outputs, retry_image_outputs

        for txt in text_outputs:
            pub("content", txt)
            full_res += txt + ("\n" if not txt.endswith("\n") else "")

        for part in image_outputs:
            fn2 = _save_gemini_image_part(part)
            pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
            full_res += f"Generated Image for: {final_message_text}\n"

        if not image_outputs and not text_outputs:
            pub("error", "No image output returned.")
    except Exception as e:
        logger.exception("Gemini Image Gen Error")
        pub("error", f"Gemini Image Gen Error: {str(e)}")

    return full_res
