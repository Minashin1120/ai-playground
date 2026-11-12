"""background_stream_ideogram branch of `background_chat_task` (provider dispatch chain).

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns full_res.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_ideogram(local):
    """Generate or edit Ideogram images and return the updated answer text."""

    _load_history_image_parts = local.get('_load_history_image_parts')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    key = local.get('key')
    loaded_files = local.get('loaded_files')
    model_key_l = local.get('model_key_l')
    msg = local.get('msg')
    options = local.get('options')
    pub = local.get('pub')
    thread_id = local.get('thread_id')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Ideogram Branch")
    # These were locals of `background_chat_task`; the history loader owns them now.
    MAX_CONTEXT_MESSAGES, MAX_CONTEXT_TOKENS = background_context_limits()
    try:
        pub("status", "画像生成の準備中...")
        # Edit the last model output in the selected branch; new uploads
        # are references. Text context limits must not change the target.
        image_history = None
        remove_background = options.get("ideogram_operation") == "remove_background"
        remove_object = options.get("ideogram_operation") == "remove_object"
        if (ideogram_supports_edit(model_key_l) or remove_background or remove_object) and (MAX_CONTEXT_MESSAGES or MAX_CONTEXT_TOKENS):
            image_history = list(reversed([
                {"role": node.role, "image_url": node.image_url}
                for node in _iter_chat_history_ancestors(thread_id, msg.parent_id, user_id)
                if node.image_url
            ]))
        history_image_parts = _load_history_image_parts(
            include_roles={"assistant"}, newest_first=True,
            source_messages=image_history, reverse_within_message=True
        ) if (ideogram_supports_edit(model_key_l) or remove_background or remove_object) else []

        def _ideogram_source_entry(raw_bytes, raw_mime, raw_name):
            entry_bytes = raw_bytes
            entry_mime = raw_mime
            if entry_mime not in IDEOGRAM_SOURCE_MIMES:
                converted_png = _convert_image_to_png(entry_bytes)
                if not converted_png:
                    raise RuntimeError("入力画像を Ideogram が受け付ける形式（PNG / JPEG / WEBP）へ変換できませんでした。")
                entry_bytes = converted_png
                entry_mime = "image/png"
            if len(entry_bytes) > IDEOGRAM_MAX_IMAGE_BYTES:
                raise RuntimeError("Ideogram に送れる入力画像は1枚あたり25MBまでです。")
            return (os.path.basename(raw_name or "input.png"), entry_bytes, entry_mime)

        current_sources = []
        for fi in loaded_files:
            if not fi.get('bytes') or not str(fi.get('mime', '')).startswith('image/'):
                continue
            current_sources.append(_ideogram_source_entry(
                fi['bytes'], fi['mime'], fi.get('send_name') or fi.get('name') or f"input_{len(current_sources)}.png"
            ))
        generated_sources = [
            _ideogram_source_entry(hp['bytes'], hp['mime'], hp['name'])
            for hp in history_image_parts[:1]
        ]
        ideogram_sources = (current_sources or generated_sources[:1]) if (remove_background or remove_object) else ideogram_select_sources(current_sources, generated_sources)
        if not ideogram_sources and (ideogram_supports_edit(model_key_l) or remove_background or remove_object):
            fallback_images = _load_history_image_parts(
                include_roles={"user"}, newest_first=True, source_messages=image_history
            )
            if fallback_images:
                hp = fallback_images[0]
                ideogram_sources = [_ideogram_source_entry(hp['bytes'], hp['mime'], hp['name'])]

        if remove_object:
            mask_name = options.get("image_mask")
            if not mask_name:
                raise RuntimeError("Object Remover にはマスク画像が必要です。")
            if not ideogram_sources:
                raise RuntimeError("Object Remover には入力画像が必要です。")
            norm = os.path.normpath(str(mask_name))
            if norm.startswith("..") or os.path.isabs(norm) or not norm.startswith(f"{user_id}/"):
                raise RuntimeError("Invalid mask path.")
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], norm)
            encrypted_mask_path = mask_path + '.enc'
            mask_bytes = None
            if os.path.exists(mask_path):
                with open(mask_path, 'rb') as f:
                    mask_bytes = f.read()
            elif os.path.exists(encrypted_mask_path):
                with open(encrypted_mask_path, 'rb') as f:
                    mask_bytes = decrypt_bytes(f.read())
            if not mask_bytes:
                raise RuntimeError("Mask file not found.")
            try:
                source_info = _probe_image(ideogram_sources[0][1])
                mask_info = _probe_image(mask_bytes)
                if not source_info or not mask_info:
                    raise ValueError("Unreadable mask or input image.")
                if (source_info['width'], source_info['height']) != (mask_info['width'], mask_info['height']):
                    raise RuntimeError("マスク画像は入力画像と同じサイズにしてください。")
                mask_mime = mimetypes.guess_type(str(mask_name))[0]
                if mask_mime not in IDEOGRAM_SOURCE_MIMES:
                    mask_bytes = _convert_image_to_png(mask_bytes)
                    mask_mime = 'image/png' if mask_bytes else None
                if not mask_bytes or mask_mime not in IDEOGRAM_SOURCE_MIMES:
                    raise ValueError("Unsupported mask image.")
                if len(mask_bytes) > IDEOGRAM_MAX_IMAGE_BYTES:
                    raise RuntimeError("マスク画像は25MBまでです。")
                options['_ideogram_mask_image'] = (os.path.basename(str(mask_name)), mask_bytes, mask_mime)
            except RuntimeError:
                raise
            except Exception:
                raise RuntimeError("Failed to process mask file.")

        if remove_background:
            pub("status", "Ideogram API (Remove background) を呼び出し中...")
        elif remove_object:
            pub("status", "Ideogram API (Object Remover) を呼び出し中...")
        elif ideogram_sources:
            pub("status", "Ideogram API (Precise Edit) を呼び出し中...")
        else:
            pub("status", "Ideogram API を呼び出し中...")
        _mark_provider_request_started()
        ideogram_items, _ideogram_edit_mode = ideogram_request(
            key, model_key_l, final_message_text, options, ideogram_sources
        )

        saved_any = False
        blocked_count = 0
        for image_index, item in enumerate(ideogram_items):
            image_url = item.get("url")
            if item.get("is_image_safe") is False or not image_url:
                blocked_count += 1
                continue
            img_bytes = _download_public_https_bytes(image_url, IDEOGRAM_MAX_OUTPUT_BYTES, timeout=120.0)
            probed = _probe_image(img_bytes) or {}
            ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(probed.get("format"), "png")
            fn2 = f"gen_ideogram_{int(time.time())}_{len(generated_images)}_{image_index}.{ext}"
            _save_user_generated_bytes(
                user_id, img_bytes, fn2, user_config.get('enable_e2ee')
            )
            generated_images.append(f"{user_id}/{fn2}")
            pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
            full_res += f"Generated Image {image_index + 1} for: {final_message_text}\n"
            saved_any = True
        if not saved_any:
            if blocked_count:
                pub("error", "Ideogram Error: 安全性チェックにより画像が返されませんでした。プロンプトや入力画像の内容を変更して再度お試しください。")
            else:
                pub("error", "Ideogram Error: No data returned.")
    except Exception as e:
        logger.exception("Ideogram Error")
        pub("error", f"Ideogram Error: {e}")
    return full_res
