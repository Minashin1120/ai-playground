"""GPT Image branch of `background_chat_task` (Images API / Responses API image tool).

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns the updated answer text.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_gpt_image(local):
    """Run the GPT Image branch and return the updated answer text."""

    _build_non_llm_image_context = local.get('_build_non_llm_image_context')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    job_id = local.get('job_id')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    # --- 3. GPT Image Branch ---
    log_force("Routing: GPT Image Branch")
    try:
        pub("status", "画像生成の準備中...")
        # GPT Image models always return base64; response_format is not supported for them.
        # Use a dedicated timeout/retry so image generation can be slower without timing out.
        img_client = o_client.with_options(
            timeout=_OPENAI_IMAGE_TIMEOUT_SECONDS,
            max_retries=_OPENAI_IMAGE_MAX_RETRIES
        )
        def _pick_image_opt(val, allowed):
            if val is None:
                return None
            v = str(val).strip()
            return v if v in allowed else None
        size_opt = _pick_image_opt(options.get('image_size'), {"auto", "1024x1024", "1536x1024", "1024x1536"}) or _OPENAI_IMAGE_DEFAULT_SIZE
        model_key_lower = model_key.lower()
        gpt_image_26_models = {"gpt-image-2.6-sunburst", "gpt-image-2.6-flare"}
        gpt_image_high_quality_models = {"gpt-image-2.5-sunburst", "gpt-image-2.5-flare"} | gpt_image_26_models
        is_gpt_image_26 = model_key_lower in gpt_image_26_models
        quality_values = {"auto", "low", "medium", "high"}
        if model_key_lower in gpt_image_high_quality_models:
            quality_values.update({"xhigh", "max"})
        quality_opt = _pick_image_opt(options.get('image_quality'), quality_values) or _OPENAI_IMAGE_DEFAULT_QUALITY
        format_values = {"png", "jpeg", "webp"}
        if is_gpt_image_26:
            # Animated GIF output is a GPT Image 2.6 feature. SVG is not offered:
            # /files/ forces .svg to download, so it cannot be shown inline.
            format_values.add("gif")
        format_opt = _pick_image_opt(options.get('image_format'), format_values) or _OPENAI_IMAGE_OUTPUT_FORMAT
        comp_opt = None
        try:
            comp_opt = int(options.get('image_compression')) if options.get('image_compression') is not None else None
        except Exception:
            comp_opt = None
        if comp_opt is not None and (comp_opt < 0 or comp_opt > 100):
            comp_opt = None
        complexity_opt = None
        if is_gpt_image_26:
            complexity_opt = _pick_image_opt(options.get('image_complexity'), {"low", "medium", "high"})
        def _pick_int_opt(val, minimum, maximum):
            try:
                number = int(val)
            except Exception:
                return None
            return number if minimum <= number <= maximum else None
        frame_count_opt = _pick_int_opt(options.get('image_frame_count'), 2, 16) if is_gpt_image_26 and format_opt == "gif" else None
        frame_duration_opt = _pick_int_opt(options.get('image_frame_duration_ms'), 20, 2000) if is_gpt_image_26 and format_opt == "gif" else None

        pub("status", "プロンプトとコンテキストを構成中...")
        img_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
        if options.get('system_prompt'):
            img_prompt = f"{options.get('system_prompt')}\n\n{img_prompt}"
        img_kwargs = {"model": model_key, "prompt": img_prompt}
        if size_opt:
            img_kwargs["size"] = size_opt
        if quality_opt:
            img_kwargs["quality"] = quality_opt
        if format_opt:
            img_kwargs["output_format"] = format_opt
            if format_opt in {"jpeg", "webp"}:
                img_kwargs["output_compression"] = comp_opt if comp_opt is not None else _OPENAI_IMAGE_OUTPUT_COMPRESSION

        pub("status", "入力画像を処理中...")
        img_inputs = []
        for fi in loaded_files:
            if not fi.get('bytes') or not fi.get('mime', '').startswith('image/'):
                continue
            img_bytes = fi['bytes']
            img_mime = fi['mime']
            if img_mime not in ('image/png', 'image/jpeg', 'image/webp'):
                converted_png = _convert_image_to_png(img_bytes)
                if converted_png:
                    img_bytes = converted_png
                    img_mime = 'image/png'
            img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"input_{len(img_inputs)}")
            img_inputs.append((img_name, img_bytes, img_mime))
        existing_input_names = {item[0] for item in img_inputs}
        for hp in history_image_parts:
            if hp['name'] in existing_input_names:
                continue
            img_inputs.append((hp['name'], hp['bytes'], hp['mime']))
            existing_input_names.add(hp['name'])
        mask_file = None
        mask_name = options.get('image_mask')
        if mask_name:
            pub("status", "マスク画像を処理中...")
            if not img_inputs:
                raise RuntimeError("Mask requires at least one input image.")
            norm = os.path.normpath(mask_name)
            if norm.startswith("..") or os.path.isabs(norm) or not norm.startswith(f"{user_id}/"):
                raise RuntimeError("Invalid mask path.")
            mp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
            me = mp + '.enc'
            mbytes = None
            if os.path.exists(mp):
                with open(mp, 'rb') as f: mbytes = f.read()
            elif os.path.exists(me):
                with open(me, 'rb') as f: mbytes = decrypt_bytes(f.read())
            if not mbytes:
                raise RuntimeError("Mask file not found.")
            try:
                base_info = _probe_image(img_inputs[0][1])
                mask_info = _probe_image(mbytes)
                if not base_info or not mask_info:
                    raise ValueError("Unreadable mask or input image.")
                if (base_info["width"], base_info["height"]) != (mask_info["width"], mask_info["height"]):
                    raise RuntimeError("Mask must match input image size.")
                if not (mask_info["format"] == "PNG" and mask_info["rgba8"] and _validate_image_structure(mbytes, mask_info)):
                    mbytes = _convert_image_to_png(mbytes, rgba=True)
                    if not mbytes:
                        raise ValueError("Mask conversion failed.")
                if len(mbytes) > 4 * 1024 * 1024:
                    raise RuntimeError("Mask must be less than 4MB.")
                mask_file = ("mask.png", mbytes, "image/png")
            except RuntimeError:
                raise
            except Exception:
                raise RuntimeError("Failed to process mask file.")

        if img_inputs:
            pub("status", "OpenAI API (Edit) を呼び出し中...")
        else:
            pub("status", "OpenAI API (Generations) を呼び出し中... (これには時間がかかる場合があります)")

        _mark_provider_request_started()

        image_data_b64 = None
        if is_gpt_image_26:
            # GPT Image 2.6 is served by the Images API (generations/edits).
            if mask_file and format_opt == "gif":
                raise RuntimeError("GIF output does not support masks.")
            extra_body = {}
            if complexity_opt:
                extra_body["complexity"] = complexity_opt
            if frame_count_opt is not None:
                extra_body["frame_count"] = frame_count_opt
            if frame_duration_opt is not None:
                extra_body["frame_duration_ms"] = frame_duration_opt
            img_kwargs = {"model": model_key, "prompt": img_prompt}
            if size_opt:
                img_kwargs["size"] = size_opt
            if quality_opt:
                img_kwargs["quality"] = quality_opt
            if format_opt:
                img_kwargs["output_format"] = format_opt
                if comp_opt is not None and format_opt in {"jpeg", "webp"}:
                    img_kwargs["output_compression"] = comp_opt
            log_force(f"GPT Image 2.6 request: model={model_key} action={'edit' if img_inputs else 'generate'} format={format_opt}")
            if img_inputs:
                edit_kwargs = dict(img_kwargs)
                edit_kwargs["image"] = [(name, bits, mime) for name, bits, mime in img_inputs]
                if mask_file:
                    edit_kwargs["mask"] = mask_file
                resp = img_client.images.edit(**edit_kwargs, extra_body=extra_body or None)
            else:
                resp = img_client.images.generate(**img_kwargs, extra_body=extra_body or None)
            image_results = list(getattr(resp, "data", None) or [])
            if image_results:
                image_data_b64 = getattr(image_results[0], "b64_json", None)
        else:
            # Build tools and input for Responses API
            tools = [
                {
                    "type": "image_generation",
                    "model": model_key,
                    "size": size_opt,
                    "quality": quality_opt,
                    "output_format": format_opt,
                }
            ]
            # PNG is lossless; the API rejects output_compression below 100 for it.
            if comp_opt is not None and format_opt in {"jpeg", "webp"}:
                tools[0]["output_compression"] = comp_opt

            input_content = [{"type": "input_text", "text": img_prompt}]
            for name, bits, mime in img_inputs:
                b64 = base64.b64encode(bits).decode()
                input_content.append({
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}"
                })

            if mask_file:
                mask_b64 = base64.b64encode(mask_file[1]).decode()
                tools[0]["input_image_mask"] = {"image_url": f"data:image/png;base64,{mask_b64}"}
                tools[0]["action"] = "edit"
            elif img_inputs:
                tools[0]["action"] = "edit"
            else:
                tools[0]["action"] = "generate"

            # Use gpt-4o-mini as the driver for image generation tool
            # background=True allows cancellation via the API
            resp_obj = img_client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": input_content}],
                tools=tools,
                background=True
            )

            # Polling loop to check for completion and cancellation
            while resp_obj.status in {"queued", "in_progress"}:
                if check_stop():
                    try:
                        img_client.responses.cancel(resp_obj.id)
                        log_force(f"GPT Image Gen Job {job_id} cancelled via Responses API.")
                    except Exception as ce:
                        log_force(f"Failed to cancel GPT Image Gen: {ce}")
                    raise RuntimeError("Generation stopped by user.")
                time.sleep(2)
                resp_obj = img_client.responses.retrieve(resp_obj.id)

            if resp_obj.status == "failed":
                err_msg = "Unknown error"
                if hasattr(resp_obj, "error") and resp_obj.error:
                    err_msg = resp_obj.error.message
                raise RuntimeError(f"OpenAI Responses API failed: {err_msg}")

            if resp_obj.status == "cancelled":
                raise RuntimeError("Generation was cancelled.")

            # Extract the generated image from the tool output
            for out_item in (resp_obj.output or []):
                # Some versions of the SDK might use dict, others objects
                if isinstance(out_item, dict):
                    if out_item.get("type") == "image_generation_call":
                        image_data_b64 = out_item.get("result")
                        break
                else:
                    if getattr(out_item, "type", None) == "image_generation_call":
                        image_data_b64 = getattr(out_item, "result", None)
                        break

        if not image_data_b64:
            raise RuntimeError("No image data found in the response.")

        img_bytes = _decode_base64_limited(image_data_b64, 50 * 1024 * 1024)
        ext = {"jpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}.get(format_opt, "png")
        fn2 = f"gen_gpt_{int(time.time())}_{len(generated_images)}.{ext}"
        pub("status", "画像を保存して暗号化を適用中...")
        _save_user_generated_bytes(
            user_id, img_bytes, fn2, user_config.get('enable_e2ee')
        )
        generated_images.append(f"{user_id}/{fn2}")
        pub("status", "完了")
        pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
        full_res += f"Generated Image for: {final_message_text}\n"
    except APITimeoutError:
        pub("error", "GPT Image Gen Timeout: Upstream is slow. Please retry.")
    except (APIConnectionError, RateLimitError) as e:
        pub("error", f"GPT Image Gen Error: {str(e)}")
    except APIError as e:
        pub("error", f"GPT Image Gen Error: {str(e)}")
    except Exception as e:
        pub("error", f"GPT Image Gen Error: {str(e)}")
