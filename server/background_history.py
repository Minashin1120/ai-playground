"""Chat history reconstruction for `background_chat_task`.

Extracted from `server/background.py` so the worker entry point keeps only the
job setup and provider dispatch.  The history list is built here, and the
image/document context helpers take it as their first argument instead of
closing over it.  Module-level helpers are resolved through the `app`
namespace because `app.py` execs every server part into one module.
"""


def background_load_history(model_key, msg, thread_id, user_id):
    """Rebuild the ancestor history and persist token counts.

    Traverses UP the tree (parent_id).  The current message (msg) is the
    user's new prompt, so only its ancestors are loaded.
    """
    history_rev = []
    total_history_tokens = 0
    try:
        # 0 means unlimited.
        MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "0") or "0")
    except Exception:
        MAX_CONTEXT_TOKENS = 0
    if MAX_CONTEXT_TOKENS < 0:
        MAX_CONTEXT_TOKENS = 0
    try:
        # 0 means unlimited.
        MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "0") or "0")
    except Exception:
        MAX_CONTEXT_MESSAGES = 0
    if MAX_CONTEXT_MESSAGES < 0:
        MAX_CONTEXT_MESSAGES = 0
    history_count = 0

    messages_to_update = False
    for current_node in _iter_chat_history_ancestors(
        thread_id, msg.parent_id, user_id, MAX_CONTEXT_MESSAGES
    ):
        if MAX_CONTEXT_MESSAGES and history_count >= MAX_CONTEXT_MESSAGES:
            break
        raw_cnt = current_node.content or ""
        cached_tokens = None
        try:
            if current_node.role == 'user' and current_node.tokens_in and current_node.tokens_in > 0:
                cached_tokens = int(current_node.tokens_in)
            elif current_node.role == 'assistant' and current_node.tokens_out and current_node.tokens_out > 0:
                cached_tokens = int(current_node.tokens_out)
        except Exception:
            cached_tokens = None
        if cached_tokens is not None:
            t_len = cached_tokens
        else:
            token_src = decrypt_val(raw_cnt) if current_node.is_encrypted else raw_cnt
            token_model = current_node.model or model_key
            t_len = max(1, count_tokens(token_src or "", token_model))
            # Mark for single commit at the end of reconstruction
            try:
                if current_node.role == 'user':
                    current_node.tokens_in = t_len
                else:
                    current_node.tokens_out = t_len
                messages_to_update = True
            except: pass

        if (not MAX_CONTEXT_TOKENS) or (total_history_tokens + t_len <= MAX_CONTEXT_TOKENS):
            cnt = decrypt_val(raw_cnt) if current_node.is_encrypted else raw_cnt
            deepseek_tool_context = None
            if current_node.role == "assistant" and current_node.thought_data:
                try:
                    raw_thought = (
                        decrypt_val(current_node.thought_data)
                        if current_node.is_encrypted
                        else current_node.thought_data
                    )
                    parsed_thought = json.loads(raw_thought)
                    if isinstance(parsed_thought, dict):
                        candidate_context = parsed_thought.get("deepseek_tool_context")
                        if isinstance(candidate_context, list):
                            deepseek_tool_context = candidate_context
                except Exception:
                    deepseek_tool_context = None
            if current_node.role == 'user':
                raw_context = getattr(current_node, 'thought_data', None)
                if raw_context and current_node.is_encrypted:
                    raw_context = decrypt_val(raw_context)
                saved_text = _saved_prompt_text(raw_context, cnt, getattr(current_node, 'quote_text', None))
                cnt = saved_text if isinstance(saved_text, str) else _append_turn_time_notice(cnt, getattr(current_node, 'prompt_options', None))
            history_rev.append({
                'role': current_node.role,
                'content': cnt,
                'image_url': current_node.image_url,
                'signature': current_node.thought_signature,
                'deepseek_tool_context': deepseek_tool_context,
            })
            total_history_tokens += t_len
            history_count += 1
        else:
            break

    # Commit any token count updates in a single batch
    if messages_to_update:
        try:
            safe_db_commit()
        except Exception:
            pass

    return list(reversed(history_rev))


def background_history_image_parts(history, include_roles=None, newest_first=False, include_only_images=True, source_messages=None, reverse_within_message=False):
    parts = []
    seen = set()
    total_bytes = 0
    image_history = history if source_messages is None else source_messages
    src_messages = list(reversed(image_history)) if newest_first else image_history
    for m in src_messages:
        role = str(m.get('role') or '').strip().lower()
        if include_roles and role not in include_roles:
            continue
        raw_urls = m.get('image_url')
        if not raw_urls:
            continue
        try:
            ref_list = json.loads(raw_urls)
        except Exception:
            ref_list = raw_urls
        if not isinstance(ref_list, list):
            ref_list = [ref_list]
        if reverse_within_message:
            ref_list = list(reversed(ref_list))
        for ref in ref_list:
            if _HISTORY_IMAGE_MAX_ITEMS and len(parts) >= _HISTORY_IMAGE_MAX_ITEMS:
                return parts
            norm_h = _normalize_upload_ref(ref)
            if not norm_h or norm_h in seen:
                continue
            info_h = _get_file_disk_info(norm_h)
            if not info_h.get("exists"):
                continue
            est_size = info_h.get("size") or 0
            if _HISTORY_IMAGE_MAX_BYTES and est_size and (total_bytes + est_size > _HISTORY_IMAGE_MAX_BYTES):
                continue
            data_h = _load_user_file_bytes(norm_h, info_h)
            if not data_h:
                continue
            if _HISTORY_IMAGE_MAX_BYTES and (total_bytes + len(data_h) > _HISTORY_IMAGE_MAX_BYTES):
                continue
            mime_h = _normalize_media_mime(norm_h, mimetypes.guess_type(norm_h)[0] or 'application/octet-stream')
            if include_only_images and not str(mime_h).startswith('image/'):
                continue
            parts.append({
                "role": role,
                "ref": norm_h,
                "bytes": data_h,
                "mime": mime_h,
                "name": os.path.basename(norm_h),
                "content": m.get('content') or ""
            })
            seen.add(norm_h)
            total_bytes += len(data_h)
    return parts


def background_message_history_images(raw_urls, seen=None, total_bytes=0, include_only_images=True):
    items = []
    if not raw_urls:
        return items, total_bytes
    if seen is None:
        seen = set()
    try:
        ref_list = json.loads(raw_urls)
    except Exception:
        ref_list = raw_urls
    if not isinstance(ref_list, list):
        ref_list = [ref_list]
    for ref in ref_list:
        if _HISTORY_IMAGE_MAX_ITEMS and len(seen) >= _HISTORY_IMAGE_MAX_ITEMS:
            break
        norm_h = _normalize_upload_ref(ref)
        if not norm_h or norm_h in seen:
            continue
        info_h = _get_file_disk_info(norm_h)
        if not info_h.get("exists"):
            continue
        est_size = info_h.get("size") or 0
        if _HISTORY_IMAGE_MAX_BYTES and est_size and (total_bytes + est_size > _HISTORY_IMAGE_MAX_BYTES):
            continue
        data_h = _load_user_file_bytes(norm_h, info_h)
        if not data_h:
            continue
        if _HISTORY_IMAGE_MAX_BYTES and (total_bytes + len(data_h) > _HISTORY_IMAGE_MAX_BYTES):
            continue
        mime_h = _normalize_media_mime(norm_h, mimetypes.guess_type(norm_h)[0] or 'application/octet-stream')
        if include_only_images and not str(mime_h).startswith('image/'):
            continue
        items.append({
            "ref": norm_h,
            "bytes": data_h,
            "mime": mime_h,
            "name": os.path.basename(norm_h)
        })
        seen.add(norm_h)
        total_bytes += len(data_h)
    return items, total_bytes


def background_non_llm_image_context(history, current_text, include_assistant_images=True):
    max_context_chars = 12000
    text_lines = []
    for m in history:
        role = 'User' if m.get('role') == 'user' else 'Assistant'
        msg_text = (m.get('content') or '').strip()
        image_count = 0
        try:
            raw_urls = m.get('image_url')
            if raw_urls:
                parsed_urls = json.loads(raw_urls)
                if isinstance(parsed_urls, list):
                    image_count = len(parsed_urls)
                elif parsed_urls:
                    image_count = 1
        except Exception:
            image_count = 1 if m.get('image_url') else 0
        if msg_text:
            text_lines.append(f"{role}: {msg_text}")
        elif image_count:
            text_lines.append(f"{role}: [attached {image_count} image(s)]")
    history_images = background_history_image_parts(
        history,
        include_roles={"user", "assistant"} if include_assistant_images else {"user"},
        newest_first=True,
        include_only_images=True
    )
    if not text_lines and not history_images:
        return current_text, history_images
    if text_lines:
        combined_text = "\n".join(text_lines)
        if len(combined_text) > max_context_chars:
            combined_text = combined_text[-max_context_chars:]
            text_lines = ["[earlier context trimmed]", combined_text]
    prompt_sections = [
        "Conversation context for this image follow-up:",
        "\n".join(text_lines) if text_lines else "(no prior text context)",
        "Current user request:",
        current_text
    ]
    return "\n\n".join([section for section in prompt_sections if section]), history_images


def background_append_history_document_text(history, img_list):
    """Give follow-up turns the text of documents from earlier turns.

    Only images from earlier messages are re-sent to the model, so a
    DOCX/PDF/text file attached or generated earlier in the thread
    could not be read afterwards. Append its extracted text to that
    history message (newest first, within a size budget).
    """
    max_total_chars = 120000
    max_file_chars = 60000
    max_file_bytes = 50 * 1024 * 1024
    used_chars = 0
    seen_refs = set()
    for fn in img_list:
        norm_cur = _normalize_upload_ref(fn)
        if norm_cur:
            seen_refs.add(norm_cur)
    for m in reversed(history):
        raw_urls = m.get('image_url')
        if not raw_urls:
            continue
        try:
            ref_list = json.loads(raw_urls)
        except Exception:
            ref_list = raw_urls
        if not isinstance(ref_list, list):
            ref_list = [ref_list]
        blocks = []
        for ref in ref_list:
            if used_chars >= max_total_chars:
                break
            norm_h = _normalize_upload_ref(ref)
            if not norm_h or norm_h in seen_refs:
                continue
            ext_h = os.path.splitext(norm_h)[1].lower()
            if ext_h not in ('.docx', '.pdf') and ext_h not in _TEXT_LIKE_UPLOAD_EXTS:
                continue
            seen_refs.add(norm_h)
            try:
                info_h = _get_file_disk_info(norm_h)
                if not info_h.get("exists"):
                    continue
                if (info_h.get("size") or 0) > max_file_bytes:
                    continue
                data_h = _load_user_file_bytes(norm_h, info_h)
                if not data_h:
                    continue
                if ext_h == '.docx':
                    text_h = _extract_docx_as_numbered(data_h)
                elif ext_h == '.pdf':
                    text_h = _extract_text_from_pdf(data_h)
                else:
                    text_h = data_h.decode('utf-8', errors='replace')
            except Exception:
                continue
            text_h = (text_h or '').strip()
            if not text_h:
                continue
            room = min(max_file_chars, max_total_chars - used_chars)
            truncated = len(text_h) > room
            text_h = text_h[:room]
            used_chars += len(text_h)
            name_h = os.path.basename(norm_h)
            blocks.append(
                f"--- Attached file from earlier in this conversation: {name_h} ---\n"
                f"{text_h}\n"
                + ("(truncated)\n" if truncated else "")
                + f"--- End of {name_h} ---"
            )
        if blocks:
            base_cnt = m.get('content') or ""
            m['content'] = (base_cnt + "\n\n" if base_cnt else "") + "\n\n".join(blocks)
