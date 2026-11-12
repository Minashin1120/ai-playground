"""OpenAI Search API branch of `background_chat_task` (Chat Completions with web search).

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns the updated answer text.

Early exits that used to `return` from `background_chat_task` now return
`BACKGROUND_STREAM_EARLY_RETURN`, which the caller turns back into `return`.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_openai_search(local):
    """Run the OpenAI Search branch and return the updated answer text."""

    _load_message_history_images = local.get('_load_message_history_images')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    _openai_cache_fresh = local.get('_openai_cache_fresh')
    _openai_system_prompt = local.get('_openai_system_prompt')
    _openai_upload_with_retry = local.get('_openai_upload_with_retry')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    history = local.get('history')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    user_id = local.get('user_id')

    # --- 3.5 OpenAI Search API (Chat Completions) ---
    log_force("Routing: OpenAI Search API Branch (Chat Completions)")
    try:
        if any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files):
            pub("error", "gpt-5-search-api does not support image inputs. Please remove images and retry.")
            return BACKGROUND_STREAM_EARLY_RETURN
        if check_stop():
            return BACKGROUND_STREAM_EARLY_RETURN
        pub("search_status", "searching")
        client = o_client
        sys_prompt = _openai_system_prompt(options.get('system_prompt'), True)
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        history_img_seen = set()
        history_img_bytes = 0

        for m in history:
            if m.get('image_url'):
                try:
                    content_parts = [{"type": "text", "text": m['content']}]
                    msg_images, history_img_bytes = _load_message_history_images(
                        m.get('image_url'),
                        seen=history_img_seen,
                        total_bytes=history_img_bytes,
                        include_only_images=True
                    )
                    for msg_img in msg_images:
                        b64 = base64.b64encode(msg_img['bytes']).decode('utf-8')
                        content_parts.append({"type": "image_url", "image_url": {"url": f"data:{msg_img['mime']};base64,{b64}"}})
                    messages.append({"role": m['role'], "content": content_parts})
                except Exception as e:
                    log_force(f"Error processing history image in search branch: {e}")
                    messages.append({"role": m['role'], "content": m['content']})
            else:
                messages.append({"role": m['role'], "content": m['content']})

        user_text = final_message_text
        file_parts = []
        file_attach_errors = []
        file_inline_limit = 20 * 1024 * 1024  # 20MiB inline limit for file inputs
        for fi in loaded_files:
            if (fi.get('is_pdf') or fi.get('is_docx') or fi.get('is_text')) and fi.get('bytes'):
                attached = False
                try:
                    f_bytes = fi['bytes']
                    f_name = os.path.basename(fi.get('send_name') or fi.get('name') or ('document.pdf' if fi.get('is_pdf') else 'document.docx' if fi.get('is_docx') else 'document.txt'))
                    if len(f_bytes) <= file_inline_limit:
                        b64 = base64.b64encode(f_bytes).decode('utf-8')
                        file_parts.append({"type": "file", "file": {"file_data": b64, "filename": f_name}})
                    else:
                        rel_path = fi.get('path') or fi.get('name') or f_name
                        cache = _get_file_cache(user_id, rel_path, "openai")
                        file_id = None
                        if _openai_cache_fresh(cache, fi.get('size'), fi.get('mtime'), fi.get('mime')):
                            file_id = cache.file_id
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "openai",
                                state="ACTIVE",
                                last_error=None,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        if not file_id:
                            suffix = os.path.splitext(f_name)[1] or ('.pdf' if fi.get('is_pdf') else '.docx' if fi.get('is_docx') else '.txt')
                            file_id, up_err = _openai_upload_with_retry(
                                client,
                                f_bytes,
                                suffix,
                                rel_path,
                                mime=fi.get('mime'),
                                size=fi.get('size'),
                                mtime=fi.get('mtime')
                            )
                            if not file_id:
                                raise RuntimeError(up_err or "file upload failed")
                        file_parts.append({"type": "file", "file": {"file_id": file_id, "filename": f_name}})
                    attached = True
                except Exception as e:
                    log_force(f"OpenAI Search file attach failed: {e}")
                    file_attach_errors.append(f"{f_name}({str(e)[:120]})")
                if attached:
                    continue
            if fi.get('text'):
                user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
        if file_attach_errors:
            parts = file_attach_errors[:5]
            if len(file_attach_errors) > 5:
                parts.append(f"...他{len(file_attach_errors)-5}件")
            pub("error", "ファイル添付に失敗しました: " + " / ".join(parts))
            return BACKGROUND_STREAM_EARLY_RETURN
        if file_parts:
            user_parts = [{"type": "text", "text": user_text}]
            user_parts.extend(file_parts)
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": user_text})

        _mark_provider_request_started()
        resp = client.chat.completions.create(
            model=model_key,
            messages=messages,
            web_search_options={"search_context_size": "medium"}
        )
        if not resp or not getattr(resp, "choices", None):
            pub("error", "Search API Error: Empty response.")
            return BACKGROUND_STREAM_EARLY_RETURN

        msg = resp.choices[0].message
        text_parts = []
        citations = []
        seen_urls = set()

        def _add_citation(title, url):
            if not url or url in seen_urls:
                return
            seen_urls.add(url)
            citations.append((title or url, url))

        def _handle_annotations(ann_list):
            for ann in ann_list or []:
                if isinstance(ann, dict):
                    a_type = ann.get("type")
                    a_url = ann.get("url") or ann.get("source") or ann.get("link")
                    a_title = ann.get("title") or a_url
                else:
                    a_type = getattr(ann, "type", None)
                    a_url = getattr(ann, "url", None)
                    a_title = getattr(ann, "title", None) or a_url
                if a_type and "citation" in str(a_type).lower() and a_url:
                    _add_citation(a_title, a_url)

        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    p_type = part.get("type")
                    p_text = part.get("text")
                    p_anns = part.get("annotations")
                else:
                    p_type = getattr(part, "type", None)
                    p_text = getattr(part, "text", None)
                    p_anns = getattr(part, "annotations", None)
                if p_type in (None, "text", "output_text") and p_text:
                    text_parts.append(p_text)
                if p_anns:
                    _handle_annotations(p_anns)
        elif isinstance(content, str):
            if content:
                text_parts.append(content)
        elif content is not None:
            text_parts.append(str(content))

        _handle_annotations(getattr(msg, "annotations", None))

        final_text = "".join(text_parts).strip()
        if final_text:
            full_res += final_text
            pub("content", final_text)

        if citations:
            citations_text = "\n\n**Sources:**\n"
            for title, url in citations:
                citations_text += f"- [{title}]({url})\n"
            full_res += citations_text
            pub("content", citations_text)
    except Exception as e:
        pub("error", f"Search API Error: {str(e)}")
    finally:
        pub("search_status", "done")
