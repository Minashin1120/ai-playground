"""background_chat_task Mistral OCR 4 document branch (PDF/image/DOCX/PPTX upload, OCR markdown assembly).

Extracted from `server/background.py` as part of the provider dispatch chain.
The branch used to run inline, so it shared `background_chat_task`'s locals.
It now receives a `locals()` snapshot as ``local`` and returns the updated
answer text (`full_res`), because the branch appends to it.
Early exits that used to `return` from `background_chat_task` now return
`BACKGROUND_STREAM_EARLY_RETURN`, which the caller turns back into `return`.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_mistral_ocr(local):
    """Run the Mistral OCR branch and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    generated_images = local.get('generated_images')
    key = local.get('key')
    loaded_files = local.get('loaded_files')
    options = local.get('options')
    pub = local.get('pub')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Mistral OCR Branch")
    uploaded_file_ids = []
    try:
        pub("status", "OCR対象の文書を準備中...")
        table_format = str(options.get("ocr_table_format") or "").strip().lower()
        if table_format not in ("markdown", "html"):
            table_format = None
        extract_header = bool(options.get("ocr_extract_header"))
        extract_footer = bool(options.get("ocr_extract_footer"))
        include_blocks = bool(options.get("ocr_include_blocks"))
        include_image_base64 = options.get("ocr_include_image_base64")
        if include_image_base64 is None:
            include_image_base64 = True
        else:
            include_image_base64 = bool(include_image_base64)
        pages_opt = str(options.get("ocr_pages") or "").strip() or None
        extra = {
            "include_image_base64": include_image_base64,
            "include_blocks": include_blocks,
        }
        if table_format:
            extra["table_format"] = table_format
        if extract_header:
            extra["extract_header"] = True
        if extract_footer:
            extra["extract_footer"] = True
        if pages_opt:
            extra["pages"] = pages_opt

        jobs = []
        for fi in loaded_files:
            data = fi.get("bytes")
            if not data:
                continue
            path = str(fi.get("path") or fi.get("name") or "")
            ext = os.path.splitext(path)[1].lower()
            mime = str(fi.get("mime") or "")
            label = fi.get("send_name") or os.path.basename(path) or "document"
            if ext in MISTRAL_OCR_IMAGE_EXTS or mime.startswith("image/"):
                jobs.append({
                    "label": label,
                    "document": {
                        "type": "image_url",
                        "image_url": _mistral_data_uri(mime or "image/jpeg", data),
                    },
                })
            elif (
                ext in MISTRAL_OCR_DOC_EXTS
                or fi.get("is_pdf")
                or fi.get("is_docx")
                or fi.get("is_pptx")
            ):
                file_id = _mistral_upload_ocr_file(key, label, data, mime)
                uploaded_file_ids.append(file_id)
                jobs.append({
                    "label": label,
                    "document": {"type": "file", "file_id": file_id},
                })
            else:
                pub("error", f"Mistral OCR は {label} の形式に対応していません。PDF / 画像 / DOCX / PPTX を添付してください。")
                return BACKGROUND_STREAM_EARLY_RETURN

        for url in _extract_mistral_ocr_urls(final_message_text):
            kind = _mistral_guess_url_kind(url)
            if kind == "image":
                jobs.append({
                    "label": url,
                    "document": {"type": "image_url", "image_url": url},
                })
            else:
                jobs.append({
                    "label": url,
                    "document": {"type": "document_url", "document_url": url},
                })

        if not jobs:
            pub("error", "Mistral OCR は文書専用です。PDF・画像・DOCX・PPTX を添付するか、公開URLを入力してください。会話履歴は送信されません。")
            return BACKGROUND_STREAM_EARLY_RETURN

        assembled = []
        pages_total = 0
        for idx, job in enumerate(jobs, start=1):
            pub("status", f"OCR処理中です（{idx}/{len(jobs)}）...")
            _mark_provider_request_started()
            ocr_json = _mistral_ocr_process_document(key, job["document"], extra)
            usage = (ocr_json or {}).get("usage_info") or {}
            try:
                pages_total += int(usage.get("pages_processed") or 0)
            except Exception:
                pass
            image_url_by_id = {}
            if include_image_base64:
                for page in (ocr_json or {}).get("pages") or []:
                    for img in (page or {}).get("images") or []:
                        if not isinstance(img, dict):
                            continue
                        img_id = str(img.get("id") or img.get("image_id") or "").strip()
                        raw_b64 = img.get("image_base64") or img.get("imageBase64")
                        img_bytes, img_mime = _decode_mistral_image_base64(raw_b64)
                        if not img_bytes:
                            continue
                        ext = _mistral_ext_from_mime(img_mime, "jpg")
                        fn2 = f"ocr_{int(time.time())}_{os.urandom(3).hex()}.{ext}"
                        _save_user_generated_bytes(
                            user_id, img_bytes, fn2, user_config.get("enable_e2ee")
                        )
                        rel = f"{user_id}/{fn2}"
                        generated_images.append(rel)
                        local_url = f"/files/{rel}"
                        if img_id:
                            image_url_by_id[img_id] = local_url
            body = _build_mistral_ocr_markdown(
                ocr_json, image_url_by_id, include_blocks=include_blocks
            )
            heading = f"## {job['label']}" if len(jobs) > 1 else ""
            chunk = f"{heading}\n\n{body}".strip()
            assembled.append(chunk)
            pub("content", (chunk if not full_res else "\n\n" + chunk))
            full_res += (("\n\n" if full_res else "") + chunk)

        if pages_total:
            footer = f"\n\n— 合計処理ページ数: {pages_total}"
            full_res += footer
            pub("content", footer)
    except Exception as e:
        logger.exception("Mistral OCR Error")
        pub("error", f"Mistral OCR Error: {str(e)}")
        return BACKGROUND_STREAM_EARLY_RETURN
    finally:
        for file_id in uploaded_file_ids:
            _mistral_delete_file(key, file_id)
    return full_res
