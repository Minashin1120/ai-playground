"""Attachment loading for `background_chat_task`.

Extracted from `server/background.py` so the worker entry point keeps only the
job setup and provider dispatch.  Loads the current turn's attachments from the
user's upload directory, validates sizes and paths, extracts PDF/DOCX/XLSX/text
content, and records the file cache.  Module-level helpers are resolved through
the `app` namespace because `app.py` execs every server part into one module.
"""


def background_load_attachments(img_list, options, user_id, is_llm_model, is_mistral_ocr):
    """Return `(loaded_files, file_errors)` for this turn."""
    loaded_files = []
    file_errors = []
    cache_updated = False
    total_loaded_bytes = 0
    attachment_name_map = {}
    raw_attachment_name_map = options.get("attachment_name_map") or {}
    if isinstance(raw_attachment_name_map, dict):
        for raw_path, raw_name in raw_attachment_name_map.items():
            norm_path = _normalize_upload_ref(raw_path)
            if not norm_path or not norm_path.startswith(f"{user_id}/"):
                continue
            norm_name = _normalize_display_name_for_path(norm_path, raw_name)
            if norm_name:
                attachment_name_map[norm_path] = norm_name
    label_name_map = _get_user_file_label_map(user_id) if img_list else {}

    def _resolve_send_name(rel_path, mime):
        norm_rel = _normalize_upload_ref(rel_path)
        base_name = os.path.basename(norm_rel or "") or "file"
        explicit = attachment_name_map.get(norm_rel) if norm_rel else None
        if not explicit and norm_rel:
            explicit = label_name_map.get(norm_rel)
        if explicit and norm_rel:
            fixed = _normalize_display_name_for_path(norm_rel, explicit)
            if fixed:
                return fixed
        if norm_rel:
            fixed = _normalize_display_name_for_path(norm_rel, base_name)
            if fixed:
                return fixed
        return _sanitize_file_display_name(base_name) or "file"

    if img_list:
        try:
            max_single_mb = int(os.getenv("ATTACHMENT_MAX_MB", str(_upload_max_mb)) or _upload_max_mb)
        except Exception:
            max_single_mb = _upload_max_mb
        max_single_bytes = max_single_mb * 1024 * 1024 if max_single_mb else 0
        max_total_bytes = 0
        try:
            max_total_mb = os.getenv("ATTACHMENT_TOTAL_MAX_MB")
            if max_total_mb and str(max_total_mb).strip():
                max_total_bytes = int(max_total_mb) * 1024 * 1024
        except Exception:
            max_total_bytes = 0

        for fn in img_list:
            clean_fn = _normalize_upload_ref(fn)
            if not clean_fn:
                file_errors.append({"name": str(fn)[:80], "reason": "無効な参照"})
                continue
            if not clean_fn.startswith(f"{user_id}/"):
                file_errors.append({"name": clean_fn, "reason": "権限外のパス"})
                continue
            info = _get_file_disk_info(clean_fn)
            if not info.get("exists"):
                file_errors.append({"name": clean_fn, "reason": "見つかりません"})
                continue
            if max_single_bytes and info.get("size") and info["size"] > max_single_bytes:
                size_mb = info["size"] // (1024 * 1024)
                file_errors.append({"name": clean_fn, "reason": f"サイズ超過({size_mb}MB)"})
                continue
            data = _load_user_file_bytes(clean_fn, info)
            if data is None:
                file_errors.append({"name": clean_fn, "reason": "読み込み失敗"})
                continue
            if len(data) == 0:
                file_errors.append({"name": clean_fn, "reason": "空ファイル"})
                continue
            if max_total_bytes:
                total_loaded_bytes += len(data)
                if total_loaded_bytes > max_total_bytes:
                    file_errors.append({"name": clean_fn, "reason": "合計サイズ超過"})
                    break

            is_pdf = clean_fn.lower().endswith('.pdf')
            is_docx = clean_fn.lower().endswith('.docx')
            is_pptx = clean_fn.lower().endswith('.pptx')
            is_xlsx = clean_fn.lower().endswith(('.xlsx', '.xlsm'))
            mime_guess = mimetypes.guess_type(clean_fn)[0]
            mime = _normalize_media_mime(clean_fn, mime_guess)
            clean_ext = os.path.splitext(clean_fn)[1].lower()
            is_text = (mime or '').startswith('text/') or clean_ext in _TEXT_LIKE_UPLOAD_EXTS
            if is_pdf:
                mime = 'application/pdf'
            elif is_docx:
                mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif is_pptx:
                mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            elif is_text:
                mime = 'text/plain'
            send_name = _resolve_send_name(clean_fn, mime) if is_llm_model else None

            if is_pdf:
                extracted = None if is_mistral_ocr else _extract_text_from_pdf(data)
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': extracted if extracted else None,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': True,
                    'is_docx': False,
                    'is_pptx': False,
                    'is_text': False,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            elif is_docx:
                # Render paragraphs with [N] numbers so the model can
                # reference them with edit_file's paragraph_edits.
                extracted = None if is_mistral_ocr else _extract_docx_as_numbered(data)
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': extracted if extracted else None,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': False,
                    'is_docx': True,
                    'is_pptx': False,
                    'is_text': False,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            elif is_pptx:
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': None,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': False,
                    'is_docx': False,
                    'is_pptx': True,
                    'is_text': False,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            elif is_text:
                extracted = _decode_text_bytes_for_prompt(data)
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': extracted if extracted else None,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': False,
                    'is_text': True,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            elif is_xlsx:
                # Spreadsheets are binary, so the model cannot read them
                # as a native file part.  Render the cells as TSV text
                # (with a column-letter header row) so the model can see
                # the content and derive cell addresses for edit_file.
                xlsx_text = _extract_xlsx_as_tsv(data, include_column_headers=True)
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': xlsx_text,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': False,
                    'is_xlsx': True,
                    'is_text': False,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            else:
                loaded_files.append({
                    'name': clean_fn,
                    'path': clean_fn,
                    'text': None,
                    'bytes': data,
                    'mime': mime,
                    'is_pdf': False,
                    'is_text': False,
                    'send_name': send_name,
                    'size': len(data),
                    'mtime': info.get("mtime")
                })
            try:
                _upsert_file_cache(
                    user_id,
                    clean_fn,
                    "local",
                    size_bytes=len(data),
                    mtime=info.get("mtime"),
                    mime_type=mime,
                    state="loaded",
                    last_error=None
                )
                cache_updated = True
            except Exception:
                pass

        if cache_updated:
            try:
                safe_db_commit()
            except Exception:
                pass

    return loaded_files, file_errors
