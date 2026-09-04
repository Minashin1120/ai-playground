# -----------------------------------------------------------
# API Routes
# -----------------------------------------------------------

def _is_browser_fast_mode_model(model_key):
    model_l = str(model_key or '').strip().lower()
    return (
        model_key in ALL_VALID_MODEL_IDS
        and model_l.startswith('gemini-')
        and not any(marker in model_l for marker in ('image', 'native-audio', 'tts', 'live'))
    )

def _get_browser_fast_mode_user_key(user, model_key):
    """Return only a key owned by this user; never disclose admin/env fallback keys."""
    model_key_value = _get_model_specific_api_key(user, model_key)
    if model_key_value:
        return model_key_value, 'model_specific'
    common_value = decrypt_val(getattr(user, 'gemini_api_key', None)) if user else None
    if common_value and str(common_value).strip():
        return str(common_value).strip(), 'gemini_common'
    return None, None

def _browser_fast_mode_history(thread, parent_message):
    if not thread or not parent_message:
        return []
    all_messages = Message.query.filter_by(thread_id=thread.id).all()
    message_map = {message.id: message for message in all_messages}
    current = message_map.get(parent_message.id)
    history_rev = []
    total_chars = 0
    selected_image_count = 0
    selected_image_bytes = 0
    while current and len(history_rev) < 200:
        raw_content = current.content or ''
        content = decrypt_val(raw_content) if current.is_encrypted else raw_content
        content = str(content or '')
        if total_chars + len(content) > 1_000_000:
            break
        images = []
        if current.image_url:
            for raw_ref in _iter_message_attachment_refs(current.image_url):
                if selected_image_count >= BROWSER_FAST_HISTORY_IMAGE_MAX_ITEMS:
                    break
                normalized = _normalize_attachment_list([raw_ref], thread.user_id)
                if len(normalized) != 1:
                    continue
                ref = normalized[0]
                if os.path.splitext(ref)[1].lower() not in _IMAGE_THUMB_EXTS:
                    continue
                info = _get_file_disk_info(ref)
                size = int(info.get('size') or 0) if info.get('exists') else 0
                if size <= 0 or selected_image_bytes + size > BROWSER_FAST_HISTORY_IMAGE_MAX_BYTES:
                    continue
                images.append({
                    'path': ref,
                    'mime_type': _normalize_media_mime(ref, mimetypes.guess_type(ref)[0] or 'application/octet-stream'),
                })
                selected_image_count += 1
                selected_image_bytes += size
        signatures = []
        if current.role == 'assistant' and current.thought_signature:
            try:
                parsed_signatures = json.loads(current.thought_signature)
                if isinstance(parsed_signatures, list):
                    signatures = [str(value) for value in parsed_signatures if value][:16]
                elif isinstance(parsed_signatures, str) and parsed_signatures:
                    signatures = [parsed_signatures]
            except Exception:
                signatures = [str(current.thought_signature)]
        if current.role in ('user', 'assistant') and (content or images or signatures):
            history_rev.append({
                'role': 'model' if current.role == 'assistant' else 'user',
                'text': content,
                'images': images,
                'thought_signatures': signatures,
            })
            total_chars += len(content)
        current = message_map.get(current.parent_id) if current.parent_id else None
    return list(reversed(history_rev))

@app.route('/api/browser_fast_mode/bootstrap', methods=['POST'])
@login_required
def bootstrap_browser_fast_mode():
    """Return the selected user's own Gemini key and the selected branch context."""
    if not rate_limit(f"rl:browser_fast_bootstrap:user:{current_user.id}", 60, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request'}), 400
    model_key = str(data.get('model') or '').strip()
    if not _is_browser_fast_mode_model(model_key):
        return jsonify({'error': 'Browser fast mode supports Gemini text models only'}), 400
    api_key, key_source = _get_browser_fast_mode_user_key(current_user, model_key)
    if not api_key:
        return jsonify({
            'error': '選択中モデルのモデル別APIキーまたは共通Gemini APIキーを設定してください',
            'code': 'user_api_key_missing',
        }), 400

    thread = None
    parent_message = None
    thread_ref = data.get('thread_id')
    if thread_ref:
        thread = resolve_thread_for_user(thread_ref, current_user.id)
        if not thread:
            return jsonify({'error': 'Invalid thread'}), 403
        parent_raw = data.get('parent_id')
        if parent_raw is not None and str(parent_raw).strip():
            try:
                parent_id = int(parent_raw)
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid parent message'}), 400
            parent_message = Message.query.filter_by(id=parent_id, thread_id=thread.id).first()
            if not parent_message:
                return jsonify({'error': 'Invalid parent message'}), 400
        else:
            parent_message = Message.query.filter_by(thread_id=thread.id).order_by(Message.id.desc()).first()

    response = jsonify({
        'status': 'ok',
        'api_key': api_key,
        'key_source': key_source,
        'model': model_key,
        'thread_id': thread.public_id if thread else None,
        'parent_id': parent_message.id if parent_message else None,
        'history': _browser_fast_mode_history(thread, parent_message),
    })
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/api/browser_fast_mode/save', methods=['POST'])
@login_required
def save_browser_fast_mode_chat():
    """Persist a completed browser-direct Gemini turn as one atomic DB transaction."""
    if not rate_limit(f"rl:browser_fast_save:user:{current_user.id}", 30, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request'}), 400
    gate_resp = _bot_turnstile_gate(data.get('turnstile_token'))
    if gate_resp is not None:
        return gate_resp
    client_request_id = str(data.get('client_request_id') or '').strip()
    if client_request_id and not re.fullmatch(r'[A-Za-z0-9_-]{8,64}', client_request_id):
        return jsonify({'error': 'Invalid client request ID'}), 400

    user_text = data.get('message')
    assistant_text = data.get('assistant_content')
    thought_text = data.get('thought_content') or ''
    model_key = str(data.get('model') or '').strip()
    if not isinstance(user_text, str) or len(user_text) > 500_000:
        return jsonify({'error': 'Invalid or oversized message'}), 400
    # Allow image-only sends (no prompt text) when images are attached.
    fast_refs = data.get('image_urls') or []
    if not isinstance(fast_refs, list):
        fast_refs = [fast_refs]
    if not user_text.strip() and not bool([r for r in fast_refs if r]):
        return jsonify({'error': 'Invalid or oversized message'}), 400
    if not isinstance(assistant_text, str) or not assistant_text.strip() or len(assistant_text) > 2_000_000:
        return jsonify({'error': 'Invalid or oversized assistant content'}), 400
    if not isinstance(thought_text, str) or len(thought_text) > 2_000_000:
        return jsonify({'error': 'Invalid or oversized thought content'}), 400
    if not _is_browser_fast_mode_model(model_key):
        return jsonify({'error': 'Browser fast mode supports Gemini text models only'}), 400

    raw_signatures = data.get('thought_signatures') or []
    if not isinstance(raw_signatures, list) or len(raw_signatures) > 16:
        return jsonify({'error': 'Invalid thought signatures'}), 400
    thought_signatures = []
    signature_bytes = 0
    for raw_signature in raw_signatures:
        signature = str(raw_signature or '').strip()
        if not signature or len(signature) > 100_000:
            return jsonify({'error': 'Invalid thought signature'}), 400
        try:
            decoded_signature = base64.b64decode(signature, validate=True)
        except Exception:
            return jsonify({'error': 'Invalid thought signature'}), 400
        signature_bytes += len(decoded_signature)
        if signature_bytes > 256_000:
            return jsonify({'error': 'Thought signatures are too large'}), 400
        thought_signatures.append(signature)

    raw_refs = data.get('image_urls') or []
    if not isinstance(raw_refs, list):
        raw_refs = [raw_refs]
    refs = _normalize_attachment_list(raw_refs, current_user.id)
    if len(refs) != len(raw_refs):
        return jsonify({'error': 'One or more browser fast mode image references are invalid'}), 400
    if len(refs) > min(4, int(app.config.get('ATTACHMENT_MAX_FILES') or 30)):
        return jsonify({'error': 'Browser fast mode accepts at most 4 images'}), 400
    total_image_bytes = 0
    for ref in refs:
        ext = os.path.splitext(ref)[1].lower()
        info = _get_file_disk_info(ref)
        if ext not in _IMAGE_THUMB_EXTS or not info.get('exists'):
            return jsonify({'error': 'A saved browser fast mode image is missing or invalid'}), 400
        total_image_bytes += int(info.get('size') or 0)
        if total_image_bytes > _LOW_LATENCY_IMAGE_MAX_BYTES:
            return jsonify({'error': 'Browser fast mode images exceed the save limit'}), 400

    is_enc = bool(getattr(current_user, 'enable_e2ee', False))
    user_content = encrypt_val(user_text) if is_enc else user_text
    thought_payload = json.dumps({'text': thought_text}, ensure_ascii=False) if thought_text else None
    assistant_content = assistant_text
    if is_enc:
        assistant_content = encrypt_val(assistant_content)
        if thought_payload:
            thought_payload = encrypt_val(thought_payload)

    user_tokens = count_tokens(user_text, model_key)
    thought_tokens = count_tokens(thought_text, model_key) if thought_text else 0
    assistant_tokens = count_tokens_for_display(assistant_text, model_key, thought_text)
    thread_ref = data.get('thread_id')
    parent_raw = data.get('parent_id')
    if not thread_ref and parent_raw is not None and str(parent_raw).strip():
        return jsonify({'error': 'A parent message requires an existing thread'}), 400
    try:
        created_thread = not bool(thread_ref)
        if thread_ref:
            thread = resolve_thread_for_user(thread_ref, current_user.id)
            if not thread:
                return jsonify({'error': 'Invalid thread'}), 403
        else:
            thread = Thread(
                user_id=current_user.id,
                public_id=generate_thread_public_id(),
                title=_normalize_thread_title(user_text.strip()[:160] or 'New Chat'),
                is_temporary=bool(data.get('temporary_chat')),
                include_global_instruction=False,
                last_model=model_key,
                updated_at=datetime.utcnow(),
            )
            db.session.add(thread)
            db.session.flush()
        parent_message = None
        if parent_raw is not None and str(parent_raw).strip():
            try:
                parent_id = int(parent_raw)
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid parent message'}), 400
            parent_message = Message.query.filter_by(id=parent_id, thread_id=thread.id).first()
            if not parent_message:
                return jsonify({'error': 'Invalid parent message'}), 400
        elif not created_thread:
            parent_message = Message.query.filter_by(thread_id=thread.id).order_by(Message.id.desc()).first()
        submission_claimed, existing_submission = _claim_chat_submission(current_user.id, client_request_id)
        if not submission_claimed:
            if existing_submission and existing_submission.get("state") == "accepted":
                cached_response = existing_submission.get("response")
                if isinstance(cached_response, dict):
                    return jsonify(cached_response)
            return jsonify({
                "error": "This browser-fast turn is still being saved",
                "code": "submission_in_progress",
            }), 425
        user_msg = Message(
            thread_id=thread.id,
            role='user',
            content=user_content,
            model=model_key,
            image_url=json.dumps(refs) if refs else None,
            is_encrypted=is_enc,
            parent_id=parent_message.id if parent_message else None,
            tokens_in=user_tokens,
            tokens=sum_token_counts(user_tokens, None),
        )
        db.session.add(user_msg)
        db.session.flush()
        assistant_msg = Message(
            thread_id=thread.id,
            role='assistant',
            content=assistant_content,
            model=model_key,
            thought_data=thought_payload,
            thought_signature=json.dumps(thought_signatures) if thought_signatures else None,
            is_encrypted=is_enc,
            parent_id=user_msg.id,
            tokens_out=assistant_tokens,
            tokens=sum_token_counts(None, assistant_tokens),
            tokens_thought=thought_tokens,
        )
        db.session.add(assistant_msg)
        thread.updated_at = datetime.utcnow()
        thread.last_model = model_key
        current_user.last_model = model_key
        safe_db_commit()
        if thread.is_temporary:
            try:
                _mark_temp_chat_presence(
                    thread,
                    current_user.id,
                    timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user),
                )
                _track_temp_chat_uploaded_refs(thread, current_user.id, refs)
            except Exception as exc:
                logger.warning("Browser fast mode temporary tracking failed for thread %s: %s", thread.id, exc)
        response_payload = {
            'status': 'ok',
            'thread_id': thread.public_id,
            'user_message_id': user_msg.id,
            'assistant_message_id': assistant_msg.id,
            'created_thread': created_thread,
        }
        _store_idempotent_submission(
            current_user.id,
            client_request_id,
            {"response": response_payload},
        )
        return jsonify(response_payload)
    except Exception as exc:
        db.session.rollback()
        _release_chat_submission(current_user.id, client_request_id)
        logger.error("Browser fast mode save failed for user %s: %s", current_user.id, exc)
        return jsonify({'error': 'Failed to save browser fast mode chat'}), 500


@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request'}), 400
    gate_resp = _bot_turnstile_gate(data.get('turnstile_token'))
    if gate_resp is not None:
        return gate_resp
    raw_message = data.get('message')
    if not isinstance(raw_message, str) or len(raw_message) > 500_000:
        return jsonify({'error': 'Invalid or oversized message'}), 400
    # Allow image/file-only sends (no prompt text) as long as attachments are present.
    raw_img_hint = data.get('image_urls') or []
    if not isinstance(raw_img_hint, list):
        raw_img_hint = [raw_img_hint]
    has_attachment_hint = bool([u for u in raw_img_hint if u]) or bool(data.get('image_items')) or bool(data.get('uploaded_image_urls'))
    if not raw_message.strip() and not has_attachment_hint:
        return jsonify({'error': 'Invalid or oversized message'}), 400
    client_request_id = str(data.get('client_request_id') or '').strip()
    if client_request_id and not re.fullmatch(r'[A-Za-z0-9_-]{8,64}', client_request_id):
        return jsonify({'error': 'Invalid client request ID'}), 400
    model_key = str(data.get('model') or '').strip()
    if model_key not in ALL_VALID_MODEL_IDS:
        return jsonify({'error': 'Invalid model'}), 400
    coding_mode = data.get('coding_mode') is True
    coding_target = data.get('coding_target')
    coding_candidates = []
    if coding_mode:
        if any(marker in model_key.lower() for marker in ("image", "video", "tts", "audio", "native-audio")):
            return jsonify({'error': 'Coding Mode requires a text generation model'}), 400
        if not isinstance(coding_target, dict):
            return jsonify({'error': 'Coding Mode target is required'}), 400
        raw_candidates = data.get('coding_candidates')
        if not isinstance(raw_candidates, list) or not raw_candidates or len(raw_candidates) > 30:
            return jsonify({'error': 'Coding Mode candidates are required'}), 400
        prompt_blocks = extract_markdown_code_blocks(raw_message)
        candidate_ids = set()
        total_candidate_chars = 0
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                return jsonify({'error': 'Invalid Coding Mode candidate'}), 400
            candidate_id = str(raw_candidate.get('id') or '')[:100]
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', candidate_id) or candidate_id in candidate_ids:
                return jsonify({'error': 'Invalid Coding Mode candidate ID'}), 400
            candidate_ids.add(candidate_id)
            source = 'prompt' if raw_candidate.get('source') == 'prompt' else 'history'
            if source == 'prompt':
                try:
                    prompt_index = int(raw_candidate.get('prompt_index'))
                except (TypeError, ValueError):
                    return jsonify({'error': 'Invalid prompt code block index'}), 400
                if prompt_index < 0 or prompt_index >= len(prompt_blocks):
                    return jsonify({'error': 'Prompt code block was not found'}), 400
                code = prompt_blocks[prompt_index]['code']
                language = prompt_blocks[prompt_index]['language']
            else:
                code = raw_candidate.get('code')
                language = raw_candidate.get('language') or 'text'
            if not isinstance(code, str) or not code.strip() or len(code) > 300_000:
                return jsonify({'error': 'Invalid or oversized Coding Mode target'}), 400
            total_candidate_chars += len(code)
            if total_candidate_chars > 300_000:
                return jsonify({'error': 'Coding Mode candidates are too large'}), 400
            coding_candidates.append({
                'id': candidate_id,
                'code': code,
                'language': re.sub(r'[^A-Za-z0-9_+.#-]', '', str(language))[:40] or 'text',
                'source': source,
                'explicit': raw_candidate.get('explicit') is True,
            })
        default_target_id = str(coding_target.get('id') or '')[:100]
        if default_target_id not in candidate_ids:
            default_target_id = coding_candidates[-1]['id']
        default_candidate = next(item for item in coding_candidates if item['id'] == default_target_id)
        coding_target = {
            **default_candidate,
            'default_target_id': default_target_id,
            'explicit': (
                coding_target.get('explicit') is True
                and len(coding_candidates) == 1
                and default_target_id == coding_candidates[0]['id']
            ),
        }
    resolved_auth = _resolve_chat_model_auth(current_user, model_key)
    if resolved_auth.get("error_code"):
        return jsonify({
            "error": resolved_auth.get("error") or "APIキーが設定されていません。",
            "code": resolved_auth["error_code"],
            "model": model_key,
            "provider": resolved_auth.get("provider"),
        }), 400
    for bounded_key in ('quote_text', 'system_prompt', 'marker_system_prompt', 'thread_custom_instruction'):
        bounded_value = data.get(bounded_key)
        if bounded_value is not None and len(str(bounded_value)) > 100_000:
            return jsonify({'error': f'{bounded_key} is too large'}), 400
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}_{secrets.token_hex(8)}"
    _latency_mark_once(job_id, "route_received_ms")

    temporary_requested = _coerce_bool_or_none(data.get('temporary_chat'))
    thread_ref = data.get('thread_id')
    thread_was_created = False
    if thread_ref:
        t = resolve_thread_for_user(thread_ref, current_user.id)
        if not t:
            return jsonify({'error': 'Invalid thread'}), 403
        if temporary_requested is not None:
            t.is_temporary = bool(temporary_requested)
    else:
        # Avoid a separate round trip on first send; create the thread in the same transaction.
        t = Thread(
            user_id=current_user.id,
            public_id=generate_thread_public_id(),
            is_temporary=bool(temporary_requested)
        )
        db.session.add(t)
        thread_was_created = True
    thread_id = t.id
    thread_stream_id = t.public_id if t and t.public_id else None

    # Prompt-cache provider lock: reject before saving the user message
    enable_pc_request = bool(data.get('enable_prompt_caching'))
    provider_for_pc = get_model_api_provider(data.get('model'))
    locked_provider = getattr(t, 'prompt_cache_provider', None)
    if enable_pc_request and locked_provider and provider_for_pc and locked_provider != provider_for_pc:
        locked_label = _PROVIDER_LABELS.get(locked_provider, locked_provider)
        next_label = _PROVIDER_LABELS.get(provider_for_pc, provider_for_pc)
        return jsonify({
            'error': f'PromptCache有効中は他API（{next_label}）のモデルに変更できません。ロック中: {locked_label}'
        }), 400
    submission_claimed, existing_submission = _claim_chat_submission(current_user.id, client_request_id)
    if not submission_claimed:
        if existing_submission and existing_submission.get("state") == "accepted":
            return jsonify({
                "error": "This prompt was already accepted",
                "code": "request_already_accepted",
                "job_id": existing_submission.get("job_id"),
                "thread_id": existing_submission.get("thread_id"),
                "message_id": existing_submission.get("message_id"),
                "model": existing_submission.get("model") or model_key,
            }), 409
        return jsonify({
            "error": "This prompt is still being accepted",
            "code": "submission_in_progress",
        }), 425
    
    user_msg = None
    attachment_name_map = {}
    try:
        raw_msg_content = raw_message
        msg_content = raw_msg_content
        if user_config['enable_e2ee']: msg_content = encrypt_val(msg_content)
        raw_image_urls = data.get('image_urls') or []
        if not isinstance(raw_image_urls, list):
            raw_image_urls = [raw_image_urls]
        norm_image_urls = _normalize_attachment_list(raw_image_urls, current_user.id)
        raw_image_items = data.get('image_items') or []
        if not isinstance(raw_image_items, list):
            raw_image_items = [raw_image_items]
        uploaded_image_refs = []
        for item in raw_image_items:
            if not isinstance(item, dict):
                continue
            ref = item.get('path') or item.get('filepath') or item.get('url') or item.get('file')
            norm_ref = _normalize_upload_ref(ref)
            if norm_ref and norm_ref.startswith(f"{current_user.id}/"):
                raw_name = item.get('name') or item.get('filename') or item.get('display_name')
                norm_name = _normalize_display_name_for_path(norm_ref, raw_name)
                if norm_name:
                    attachment_name_map[norm_ref] = norm_name
            source = _normalize_attachment_source(item.get('source'))
            if source == "upload" and ref:
                uploaded_image_refs.append(ref)
        explicit_uploaded_refs = data.get('uploaded_image_urls') or []
        if isinstance(explicit_uploaded_refs, list):
            uploaded_image_refs.extend(explicit_uploaded_refs)
        elif explicit_uploaded_refs:
            uploaded_image_refs.append(explicit_uploaded_refs)
        max_files = int(app.config.get('ATTACHMENT_MAX_FILES') or 30)
        if len(norm_image_urls) > max_files:
            _release_chat_submission(current_user.id, client_request_id)
            return jsonify({'error': f'Too many attachments. Max {max_files} files per message.'}), 400
        
        parent_id = data.get('parent_id', None)
        parent_explicit = data.get('parent_id_explicit', False)
        if isinstance(parent_explicit, str):
            parent_explicit = parent_explicit.strip().lower() in ("1", "true", "yes", "on")
        else:
            parent_explicit = bool(parent_explicit)

        if parent_id is not None and parent_id != "":
            try:
                if isinstance(parent_id, str) and parent_id.strip().lower() in ("null", "none", "root"):
                    parent_id = None
                else:
                    parent_id_int = int(parent_id)
                    if parent_id_int <= 0:
                        parent_id = None
                    else:
                        pm = Message.query.get(parent_id_int)
                        if not pm or pm.thread_id != thread_id or pm.thread.user_id != current_user.id:
                            parent_id = None
                        else:
                            parent_id = pm.id
            except Exception:
                parent_id = None

        if parent_id is None and not parent_explicit and not thread_was_created and t.id is not None:
            # Default to the last message in the thread
            last_msg = Message.query.filter_by(thread_id=t.id).order_by(Message.id.desc()).first()
            if last_msg:
                parent_id = last_msg.id

        # Calculate user tokens on send to avoid worker re-counting.
        user_tokens_in = None
        if user_tokens_in is None:
            user_tokens_in = count_tokens(raw_msg_content or "", data.get('model'))
        gem_uuid_val = data.get('gem_uuid')
        gem_name_val = None
        if gem_uuid_val:
            gem = Gem.query.filter_by(uuid=gem_uuid_val, user_id=current_user.id).first()
            if gem:
                gem_name_val = gem.name
        user_msg = Message(
            thread=t,
            role='user',
            content=msg_content,
            model=model_key,
            image_url=json.dumps(norm_image_urls) if norm_image_urls else None,
            quote_text=data.get('quote_text'),
            is_encrypted=user_config['enable_e2ee'],
            parent_id=parent_id,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None),
            gem_uuid=gem_uuid_val,
            gem_name=gem_name_val
        )
        db.session.add(user_msg)
        if current_user.use_last_chat_settings:
            current_user.last_model = data.get('model')
            current_user.last_enable_search = bool(data.get('enable_search'))
            current_user.last_enable_url_context = bool(data.get('enable_url_context'))
            current_user.last_enable_maps = bool(data.get('enable_maps'))
            current_user.last_enable_python = bool(data.get('enable_python'))
            current_user.last_enable_file_creation = bool(data.get('enable_file_creation'))
            current_user.last_enable_thinking = bool(data.get('enable_thinking'))
            current_user.last_thinking_level = (data.get('thinking_level') or current_user.last_thinking_level or "high")
            tb = data.get('thinking_budget')
            try:
                if tb is not None and str(tb).strip() != "":
                    current_user.last_thinking_budget = int(tb)
            except Exception:
                pass
            current_user.last_reasoning_effort = (data.get('reasoning_effort') or current_user.last_reasoning_effort or "medium")
            current_user.last_enable_system_prompt = bool(data.get('enable_system_prompt'))
            current_user.last_safety_setting = (data.get('safety_setting') or current_user.last_safety_setting or "default")
        t.last_gem_uuid = data.get('gem_uuid')
        safe_db_commit()
        thread_id = t.id
        thread_stream_id = t.public_id if t and t.public_id else str(thread_id)
        if bool(getattr(t, "is_temporary", False)):
            _mark_temp_chat_presence(
                t,
                current_user.id,
                timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
            )
            _track_temp_chat_uploaded_refs(t, current_user.id, uploaded_image_refs)
        elif temporary_requested is False:
            _clear_temp_chat_tracking_for_thread(t)
    except Exception as e:
        logger.error(f"Failed to save user msg: {e}")
        _release_chat_submission(current_user.id, client_request_id)
        return jsonify({'error': 'Failed to save message'}), 500

    quote_text = data.get('quote_text')
    if quote_text:
        try:
            redis_conn.setex(f"quote:{job_id}", 600, quote_text)
        except: pass

    sys_prompt = data.get('system_prompt')
    if sys_prompt:
        try:
            redis_conn.setex(f"sys:{job_id}", 600, sys_prompt)
        except: pass

    options = {
        'system_prompt': None,
        'enable_search': data.get('enable_search'),
        'enable_url_context': data.get('enable_url_context'),
        'disable_auto_search': data.get('disable_auto_search'),
        'enable_python': data.get('enable_python'),
        'enable_file_creation': data.get('enable_file_creation'),
        'enable_thinking': data.get('enable_thinking'),
        'thinking_level': data.get('thinking_level'),
        'thinking_budget': data.get('thinking_budget'),
        'reasoning_effort': data.get('reasoning_effort'),
        'enable_system_prompt': data.get('enable_system_prompt'),
        'marker_system_prompt': data.get('marker_system_prompt'),
        'safety_setting': data.get('safety_setting'),
        'tts_voice': data.get('tts_voice'),
        'tts_voice_custom': data.get('tts_voice_custom'),
        'tts_language': data.get('tts_language'),
        'tts_speed': data.get('tts_speed'),
        'image_size': data.get('image_size'),
        'image_quality': data.get('image_quality'),
        'image_format': data.get('image_format'),
        'image_compression': data.get('image_compression'),
        'image_mask': data.get('image_mask'),
        'gemini_image_aspect': data.get('gemini_image_aspect'),
        'gemini_image_size': data.get('gemini_image_size'),
        'grok_image_aspect': data.get('grok_image_aspect'),
        'grok_image_resolution': data.get('grok_image_resolution'),
        'grok_image_quality': data.get('grok_image_quality'),
        'grok_image_format': data.get('grok_image_format'),
        'grok_image_count': data.get('grok_image_count'),
        'xai_temperature': data.get('xai_temperature'),
        'xai_top_p': data.get('xai_top_p'),
        'xai_max_completion_tokens': data.get('xai_max_completion_tokens'),
        'xai_seed': data.get('xai_seed'),
        'xai_presence_penalty': data.get('xai_presence_penalty'),
        'xai_frequency_penalty': data.get('xai_frequency_penalty'),
        'xai_stop': data.get('xai_stop'),
        'xai_response_format': data.get('xai_response_format'),
        'xai_tool_choice': data.get('xai_tool_choice'),
        'xai_parallel_tool_calls': data.get('xai_parallel_tool_calls'),
        'xai_logprobs': data.get('xai_logprobs'),
        'xai_top_logprobs': data.get('xai_top_logprobs'),
        'grok_video_duration': data.get('grok_video_duration'),
            'grok_video_aspect': data.get('grok_video_aspect'),
            'grok_video_resolution': data.get('grok_video_resolution'),
            'ocr_table_format': data.get('ocr_table_format'),
            'ocr_extract_header': data.get('ocr_extract_header'),
            'ocr_extract_footer': data.get('ocr_extract_footer'),
            'ocr_include_blocks': data.get('ocr_include_blocks'),
            'ocr_include_image_base64': data.get('ocr_include_image_base64'),
            'ocr_pages': data.get('ocr_pages'),
            'transcription_language_codes': data.get('transcription_language_codes'),
            'transcription_custom_vocabulary': data.get('transcription_custom_vocabulary'),
            'transcription_mode': data.get('transcription_mode'),
            'transcription_diarization': data.get('transcription_diarization'),
            'transcription_word_timestamps': data.get('transcription_word_timestamps'),
            'attachment_name_map': attachment_name_map,
            'image_vision_model': data.get('image_vision_model'),
            'gem_uuid': data.get('gem_uuid'),
            'enable_prompt_caching': enable_pc_request,
            'prompt_cache_key': None,
            'coding_mode': coding_mode,
            'coding_target': coding_target if coding_mode else None,
            'coding_candidates': coding_candidates if coding_mode else [],
        }
    # Persist prompt-cache flags on the thread (provider locked while enabled)
    try:
        if enable_pc_request:
            t.enable_prompt_caching = True
            if provider_for_pc:
                t.prompt_cache_provider = provider_for_pc
            cache_id = t.public_id or str(t.id)
            options['prompt_cache_key'] = f"thread-{cache_id}"
        else:
            t.enable_prompt_caching = False
            t.prompt_cache_provider = None
        db.session.add(t)
        safe_db_commit()
    except Exception as e:
        logger.warning(f"prompt caching thread flags update failed: {e}")
    if 'thread_custom_instruction' in data:
        options['thread_custom_instruction'] = data.get('thread_custom_instruction')

    model_key = str(data.get('model') or '').strip()
    model_key_l = model_key.lower()
    no_attachments = not bool(norm_image_urls)
    low_latency_image_attachments = _is_low_latency_image_attachment_set(norm_image_urls)
    model_looks_heavy = any(x in model_key_l for x in ("image", "video", "tts", "audio", "native-audio"))

    # Every generation runs outside gunicorn so a web-service restart cannot
    # terminate an in-process daemon thread. Small text/image requests retain
    # the low-latency fast queue; only the unsafe direct execution is removed.
    fast_queue_eligible = bool(
        not model_looks_heavy
        and (no_attachments or low_latency_image_attachments)
    )
    queue_name = _CHAT_FAST_QUEUE_NAME if fast_queue_eligible else _CHAT_HEAVY_QUEUE_NAME
    execution_path = "queued_fast" if fast_queue_eligible else "queued_heavy"
    try:
        redis_conn.hset(
            _latency_trace_key(job_id),
            mapping={
                "execution_path": execution_path,
                "queue_name": queue_name[:64],
                "model": model_key[:80],
                "thread_public_id": (thread_stream_id or "")[:64],
                "user_id": str(current_user.id),
            }
        )
        redis_conn.expire(_latency_trace_key(job_id), _LATENCY_TRACE_TTL_SECONDS)
    except Exception:
        pass

    enqueue_queue = chat_fast_queue if queue_name == _CHAT_FAST_QUEUE_NAME else chat_heavy_queue
    enqueue_queue.enqueue(
        background_chat_task,
        job_id,
        thread_id,
        data.get('model'),
        user_msg.id,
        options,
        current_user.id,
        user_config,
        job_timeout=600,
        at_front=(queue_name == _CHAT_FAST_QUEUE_NAME)
    )
    _latency_mark_once(job_id, "route_dispatch_ms")
    try:
        redis_conn.setex(
            f"pending_job:{current_user.id}:{thread_id}",
            600,
            json.dumps({
                "job_id": job_id,
                "message_id": user_msg.id,
                "created_at": int(time.time()),
                "model": data.get('model')
            })
        )
    except Exception:
        pass
    try:
        if queue_name == _CHAT_FAST_QUEUE_NAME:
            redis_conn.setex(f"stream_acc:{job_id}:status", 600, "高速キューに投入しました。優先ワーカー待機中です...")
        else:
            redis_conn.setex(f"stream_acc:{job_id}:status", 600, "通常キューに投入しました。ワーカー待機中です...")
    except Exception:
        pass

    # Publish the accepted job before opening the stream. A retry after an
    # ambiguous disconnect can resume this exact job instead of creating a
    # second user message.
    _complete_chat_submission(
        current_user.id,
        client_request_id,
        job_id,
        thread_stream_id,
        user_msg.id,
        model_key,
    )

    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        _latency_mark_once(job_id, "route_stream_open_ms")
        if thread_stream_id:
            yield json.dumps({"type": "thread_id", "content": thread_stream_id}) + "\n"
        yield json.dumps({"type": "job_id", "content": job_id}) + "\n"
        try:
            cached_status = redis_conn.get(f"stream_acc:{job_id}:status")
            if cached_status:
                _latency_mark_once(job_id, "stream_first_status_to_client_ms")
                yield json.dumps({"type": "status", "content": cached_status.decode("utf-8", "ignore")}) + "\n"
            cached_coding_diffs = redis_conn.lrange(f"stream_acc:{job_id}:coding_diff", 0, -1)
            for raw_diff in cached_coding_diffs:
                try:
                    yield json.dumps({
                        "type": "coding_diff",
                        "content": json.loads(raw_diff),
                    }, ensure_ascii=False) + "\n"
                except Exception:
                    continue
            cached_mcp = redis_conn.lrange(f"stream_acc:{job_id}:mcp", 0, -1)
            for raw_mcp in cached_mcp:
                try:
                    entry = json.loads(raw_mcp)
                    if isinstance(entry, dict):
                        yield json.dumps(entry, ensure_ascii=False) + "\n"
                except Exception:
                    continue
        except Exception:
            pass
        try:
            refresh_count = 0
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                refresh_count += 1
                if refresh_count % 20 == 0:
                    try:
                        redis_conn.expire(f"pending_job:{current_user.id}:{thread_id}", 600)
                    except Exception:
                        pass
                if message['type'] == 'message':
                    _latency_mark_once(job_id, "stream_first_pubsub_ms")
                    evt = json.loads(message['data'])
                    evt_type = str(evt.get('type') or '')
                    if evt_type == "status":
                        _latency_mark_once(job_id, "stream_first_status_to_client_ms")
                    elif evt_type == "thought":
                        _latency_mark_once(job_id, "stream_first_thought_to_client_ms")
                    elif evt_type == "content":
                        _latency_mark_once(job_id, "stream_first_content_to_client_ms")
                    yield json.dumps(evt) + "\n"
                    if evt_type in ['done', 'error']:
                        _latency_mark_once(job_id, "stream_done_ms")
                        break
        finally:
            _latency_mark_once(job_id, "stream_done_ms")
            pubsub.unsubscribe()
            _upsert_chat_latency_trace(
                job_id=job_id,
                user_id=current_user.id,
                thread_public_id=thread_stream_id,
                model=model_key,
                execution_path=execution_path
            )
    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/token_estimate', methods=['POST'])
@login_required
def estimate_prompt_tokens_api():
    data = request.get_json(silent=True) or {}
    model_key = str(data.get('model') or '')
    message_text = data.get('message')
    quote_text = data.get('quote_text')
    raw_image_urls = data.get('image_urls') or []
    if not isinstance(raw_image_urls, list):
        raw_image_urls = [raw_image_urls]

    if message_text is None:
        message_text = ''
    else:
        message_text = str(message_text)
    if quote_text is None:
        quote_text = ''
    else:
        quote_text = str(quote_text)

    norm_image_urls = _normalize_attachment_list(raw_image_urls, current_user.id)
    max_files = int(app.config.get('ATTACHMENT_MAX_FILES') or 30)
    if len(norm_image_urls) > max_files:
        norm_image_urls = norm_image_urls[:max_files]
    if quote_text:
        message_for_count = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"
    else:
        message_for_count = message_text

    if not should_count_tokens_for_display(model_key):
        return jsonify({
            'countable': False,
            'tokens_total': None,
            'tokens_prompt': None,
            'tokens_files': None,
            'files_total': len(norm_image_urls),
            'files_counted': 0,
            'files_non_text': 0,
            'files_missing': 0,
            'files_error': 0
        })

    prompt_tokens = count_tokens(message_for_count or "", model_key)
    file_tokens = 0
    files_counted = 0
    files_non_text = 0
    files_missing = 0
    files_error = 0

    for rel_path in norm_image_urls:
        est = _estimate_attachment_prompt_tokens(rel_path, model_key=model_key)
        tok = int(est.get("tokens") or 0)
        file_tokens += tok
        reason = est.get("reason")
        if est.get("countable"):
            files_counted += 1
        elif reason == "missing":
            files_missing += 1
        elif reason in ("non_text", "no_text"):
            files_non_text += 1
        else:
            files_error += 1

    return jsonify({
        'countable': True,
        'tokens_total': prompt_tokens + file_tokens,
        'tokens_prompt': prompt_tokens,
        'tokens_files': file_tokens,
        'files_total': len(norm_image_urls),
        'files_counted': files_counted,
        'files_non_text': files_non_text,
        'files_missing': files_missing,
        'files_error': files_error
    })


@app.route('/chat_stream_resume', methods=['POST'])
@login_required
def chat_stream_resume():
    data = request.get_json(silent=True) or {}
    gate_resp = _bot_turnstile_gate(data.get('turnstile_token'))
    if gate_resp is not None:
        return gate_resp
    job_id = str(data.get('job_id') or '')
    thread_id = data.get('thread_id')
    if not _is_valid_job_id(job_id) or not thread_id:
        return jsonify({'error': 'job_id and thread_id required'}), 400
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    pending_raw = None
    try:
        pending_raw = redis_conn.get(f"pending_job:{current_user.id}:{t.id}")
    except Exception:
        pending_raw = None
    if not pending_raw:
        # 接続エラー後のリロードで一時的にセッション読込不能になるのを防ぐため、古いpendingをクリア
        try:
            redis_conn.delete(f"pending_job:{current_user.id}:{t.id}")
        except Exception:
            pass
        return jsonify({'error': 'no pending job'}), 404
    pending_job = None
    try:
        pending_job = json.loads(pending_raw)
    except Exception:
        try:
            pending_job = {"job_id": pending_raw.decode("utf-8", "ignore")}
        except Exception:
            pending_job = None
    if pending_job and pending_job.get('job_id') and pending_job.get('job_id') != job_id:
        return jsonify({'error': 'job mismatch'}), 404

    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        yield json.dumps({"type": "job_id", "content": job_id}) + "\n"
        try:
            cached_status = redis_conn.get(f"stream_acc:{job_id}:status")
            if cached_status:
                yield json.dumps({"type": "status", "content": cached_status.decode("utf-8", "ignore")}) + "\n"
            cached_thought = redis_conn.get(f"stream_acc:{job_id}:thought")
            if cached_thought:
                yield json.dumps({"type": "thought", "content": cached_thought.decode("utf-8", "ignore")}) + "\n"
            cached_content = redis_conn.get(f"stream_acc:{job_id}:content")
            if cached_content:
                yield json.dumps({"type": "content", "content": cached_content.decode("utf-8", "ignore")}) + "\n"
            cached_coding_diffs = redis_conn.lrange(f"stream_acc:{job_id}:coding_diff", 0, -1)
            for raw_diff in cached_coding_diffs:
                try:
                    yield json.dumps({
                        "type": "coding_diff",
                        "content": json.loads(raw_diff),
                    }, ensure_ascii=False) + "\n"
                except Exception:
                    continue
            cached_search = redis_conn.get(f"stream_acc:{job_id}:search")
            if cached_search:
                yield json.dumps({"type": "search_status", "content": cached_search.decode("utf-8", "ignore")}) + "\n"
            cached_py = redis_conn.hgetall(f"stream_acc:{job_id}:python")
            if cached_py:
                for _, raw in cached_py.items():
                    try:
                        py = json.loads(raw)
                        yield json.dumps({"type": "python", "content": py}) + "\n"
                    except Exception:
                        continue
            cached_mcp = redis_conn.lrange(f"stream_acc:{job_id}:mcp", 0, -1)
            for raw_mcp in cached_mcp:
                try:
                    entry = json.loads(raw_mcp)
                    if isinstance(entry, dict):
                        yield json.dumps(entry, ensure_ascii=False) + "\n"
                except Exception:
                    continue
            cached_final = redis_conn.get(f"stream_acc:{job_id}:final")
            if cached_final:
                final_type = cached_final.decode("utf-8", "ignore").strip().lower()
                if final_type == "error":
                    cached_error = redis_conn.get(f"stream_acc:{job_id}:error")
                    if cached_error:
                        yield cached_error.decode("utf-8", "ignore") + "\n"
                    else:
                        yield json.dumps({"type": "error", "content": "The job has ended with an error. Please reload."}) + "\n"
                else:
                    yield json.dumps({"type": "done", "content": "OK"}) + "\n"
                return
        except Exception:
            pass
        try:
            refresh_count = 0
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                refresh_count += 1
                if refresh_count % 20 == 0:
                    try:
                        redis_conn.expire(f"pending_job:{current_user.id}:{t.id}", 600)
                    except Exception:
                        pass
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    yield json.dumps(data) + "\n"
                    if data.get('type') in ['done', 'error']: break
        finally:
            pubsub.unsubscribe()
    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route('/api/stop_chat', methods=['POST'])
@login_required
def stop_chat():
    data = request.get_json(silent=True) or {}
    requested_job_id = str(data.get('job_id') or '') or None
    if requested_job_id and not _is_valid_job_id(requested_job_id):
        return jsonify({'error': 'invalid job_id'}), 400
    thread_ref = data.get('thread_id')
    t = resolve_thread_for_user(thread_ref, current_user.id) if thread_ref else None
    if not t:
        return jsonify({'error': 'valid thread_id required'}), 400
    pending_job_id = _pending_job_id_for_thread(current_user.id, t.id)
    if requested_job_id and pending_job_id != requested_job_id:
        return jsonify({'error': 'job mismatch'}), 404
    job_id = requested_job_id or pending_job_id
    stop_source = 'job_id' if requested_job_id else 'thread_id'
    
    log_force(f"STREAM-STOP-SIGNAL: Received stop request for job_id={job_id} via {stop_source}")
    if job_id:
        redis_conn.set(f"stop_job:{job_id}", "1", ex=300)
        return jsonify({'status': 'stopped', 'job_id': job_id, 'source': stop_source})
    return jsonify({'error': 'no job_id', 'detail': 'pending job not found'}), 400

@app.route('/api/temporary_chat/heartbeat', methods=['POST'])
@login_required
def temporary_chat_heartbeat():
    data = request.json or {}
    thread_ref = data.get('thread_id')
    t = resolve_thread_for_user(thread_ref, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 404
    active = _coerce_bool_or_none(data.get('active'))
    if active is None:
        active = True
    if active and bool(getattr(t, "is_temporary", False)):
        _mark_temp_chat_presence(
            t,
            current_user.id,
            timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
        )
    temp_meta = _get_temp_chat_runtime_meta(t, user=current_user)
    return jsonify({
        'status': 'ok',
        'thread_id': t.public_id or t.id,
        'is_temporary': bool(getattr(t, "is_temporary", False)),
        'timeout_seconds': temp_meta.get('timeout_seconds'),
        'temp_chat_expires_at': temp_meta.get('temp_chat_expires_at'),
        'temp_chat_remaining_seconds': temp_meta.get('temp_chat_remaining_seconds')
    })

@app.route('/api/generate_title', methods=['POST'])
@login_required
def generate_title_api():
    """Auto-generate chat title with multi-model fallback, prioritizing requested model"""
    try:
        data = request.get_json(silent=True) or {}
        thread_id = data.get('thread_id')
        requested_model = str(data.get('model_id') or '').strip() or None
        if requested_model and requested_model not in ALL_VALID_MODEL_IDS:
            return jsonify({'error': 'Invalid model'}), 400
        
        thread = resolve_thread_for_user(thread_id, current_user.id)
        if not thread:
            return jsonify({'error': 'Unauthorized'}), 403

        first_msg = Message.query.filter_by(thread_id=thread.id, role='user').order_by(Message.timestamp).first()
        if not first_msg: return jsonify({'status': 'skipped'})
        
        content = decrypt_val(first_msg.content) if first_msg.is_encrypted else first_msg.content
        title = "New Chat"

        # Determine target provider for requested_model
        primary_provider = None
        if requested_model:
            rml = requested_model.lower()
            if is_mistral_ocr_model_key(rml):
                primary_provider = None
            elif "gemini" in rml: primary_provider = "gemini"
            elif "grok" in rml: primary_provider = "xai"
            elif "deepseek" in rml: primary_provider = "deepseek"
            else: primary_provider = "openai"

        # Preparation
        o_key = (
            _get_model_specific_api_key(current_user, "gpt-4o-mini")
            or decrypt_val(current_user.openai_api_key)
            or (os.getenv('OPENAI_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )
        gemini_runtime = _resolve_gemini_runtime(current_user)
        g_key = _get_model_specific_api_key(current_user, "gemini-2.0-flash-lite-preview") or gemini_runtime.get("api_key")
        x_key = (
            _get_model_specific_api_key(current_user, "grok-beta")
            or decrypt_val(current_user.xai_api_key)
            or (os.getenv('XAI_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )
        d_key = (
            _get_model_specific_api_key(current_user, requested_model or "deepseek-v4-flash-0731")
            or _get_model_specific_api_key(current_user, "deepseek-v4-flash-0731")
            or decrypt_val(current_user.deepseek_api_key)
            or (os.getenv('DEEPSEEK_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )

        # 1. Try Primary requested provider/model
        if primary_provider == "openai" and o_key:
            try:
                client = _get_openai_client(o_key, base_url=None)
                resp = client.chat.completions.create(
                    model=requested_model,
                    messages=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output only the title text."},
                        {"role": "user", "content": content[:500]}
                    ],
                    max_tokens=20
                )
                if resp.choices[0].message.content:
                    title = resp.choices[0].message.content.strip().replace('"', '')
            except: pass
        elif primary_provider == "gemini":
            try:
                backend = gemini_runtime.get("backend")
                if (backend == "gemini_api" and bool(g_key)) or (backend == "vertex_ai"):
                    g_client = _get_gemini_client(api_key=g_key, backend=backend, vertex_project=gemini_runtime.get("vertex_project"), vertex_location=gemini_runtime.get("vertex_location"), vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"))
                    if g_client:
                        resp = g_client.models.generate_content(
                            model=requested_model,
                            contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. Output only the title text.\n\nChat: {content[:500]}")]
                        )
                        if resp.text:
                            title = resp.text.strip().replace('"', '')
            except: pass
        elif primary_provider == "xai" and x_key and XAI_SDK_AVAILABLE:
            try:
                x_client = XAIClient(api_key=x_key)
                chat = x_client.chat.create(model=requested_model)
                chat.append(x_system("Generate a short, descriptive title (max 6 words) for this chat conversation. Output only the title text."))
                chat.append(x_user(content[:500]))
                resp = chat.sample()
                if resp and resp.message and resp.message.content:
                    title = resp.message.content.strip().replace('"', '')
            except: pass
        elif primary_provider == "deepseek" and d_key:
            try:
                client = _get_openai_client(d_key, base_url="https://api.deepseek.com")
                resp = client.chat.completions.create(
                    model=_deepseek_api_model_id(requested_model),
                    messages=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output only the title text."},
                        {"role": "user", "content": content[:500]}
                    ],
                    max_tokens=20,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    title = str(resp.choices[0].message.content).strip().replace('"', '')
            except: pass

        # 2. Fallbacks if still "New Chat"
        if title == "New Chat":
            # Try OpenAI (gpt-4o-mini)
            if o_key:
                try:
                    client = _get_openai_client(o_key, base_url=None)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output JSON: {\"title\": \"...\"}"},
                            {"role": "user", "content": content[:500]}
                        ],
                        response_format={"type": "json_object"}
                    )
                    title = json.loads(resp.choices[0].message.content).get('title', 'New Chat')
                except: pass
            
            # Try Gemini (flash)
            if title == "New Chat":
                try:
                    backend = gemini_runtime.get("backend")
                    if (backend == "gemini_api" and bool(g_key)) or (backend == "vertex_ai"):
                        g_client = _get_gemini_client(api_key=g_key, backend=backend, vertex_project=gemini_runtime.get("vertex_project"), vertex_location=gemini_runtime.get("vertex_location"), vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"))
                        if g_client:
                            resp = g_client.models.generate_content(
                                model="gemini-2.0-flash-lite-preview",
                                contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. JSON: {{'title': '...'}}\n\nChat: {content[:500]}")],
                                config=types.GenerateContentConfig(response_mime_type="application/json")
                            )
                            title = json.loads(resp.text).get('title', 'New Chat')
                except: pass

            # Try xAI (grok-beta)
            if title == "New Chat" and x_key and XAI_SDK_AVAILABLE:
                try:
                    x_client = XAIClient(api_key=x_key)
                    chat = x_client.chat.create(model="grok-beta")
                    chat.append(x_system("Generate a short, descriptive title (max 6 words) for this chat conversation. Output only the title text."))
                    chat.append(x_user(content[:500]))
                    resp = chat.sample()
                    if resp and resp.message and resp.message.content:
                        title = resp.message.content.strip()
                except: pass
            
        # 5. Final fallback if still "New Chat" or empty
        if title == "New Chat" or not title.strip():
            # Use content snippet as fallback title
            snippet = content[:50].strip().replace('\n', ' ')
            if snippet:
                title = snippet + ('...' if len(content) > 50 else '')
            else:
                title = "New Chat"

        title = _normalize_thread_title(title)
        thread.title = title
        safe_db_commit()
        return jsonify({'status': 'ok', 'title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

