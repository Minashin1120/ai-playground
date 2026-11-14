# -----------------------------------------------------------
# Browser fast mode routes
# ブラウザから Gemini へ直接送信するモードの bootstrap と保存。
# server/routes_chat.py から分割。
# -----------------------------------------------------------
# region: browser fast mode helpers
def _is_browser_fast_mode_model(model_key):
    model_l = str(model_key or '').strip().lower()
    return (
        model_key in ALL_VALID_MODEL_IDS
        and model_l.startswith('gemini-')
        and not any(marker in model_l for marker in ('image', 'nano', 'native-audio', 'tts', 'live'))
    )

# region: browser fast mode routes
def _get_browser_fast_mode_user_key(user, model_key, use_free=False):
    """Return only a key owned by this user; never disclose admin/env fallback keys."""
    if _coerce_gemini_free_tier(use_free):
        free_value = decrypt_val(getattr(user, 'gemini_free_api_key', None)) if user else None
        if free_value and str(free_value).strip():
            return str(free_value).strip(), 'gemini_free'
        return None, None
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
    history_rev = []
    total_chars = 0
    selected_image_count = 0
    selected_image_bytes = 0
    for current in _iter_chat_history_ancestors(thread.id, parent_message.id, thread.user_id):
        if len(history_rev) >= 200:
            break
        raw_content = current.content or ''
        content = decrypt_val(raw_content) if current.is_encrypted else raw_content
        content = str(content or '')
        if current.role == 'user':
            raw_context = getattr(current, 'thought_data', None)
            if raw_context and current.is_encrypted:
                raw_context = decrypt_val(raw_context)
            saved_text = _saved_prompt_text(raw_context, content, getattr(current, 'quote_text', None))
            content = saved_text if isinstance(saved_text, str) else _append_turn_time_notice(content, getattr(current, 'prompt_options', None))
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
    use_free = _coerce_gemini_free_tier(data.get('gemini_free_tier'))
    api_key, key_source = _get_browser_fast_mode_user_key(current_user, model_key, use_free=use_free)
    try:
        logger.info(
            "gemini_api_tier user=%s model=%s tier=%s backend=%s",
            current_user.id,
            model_key,
            "free" if use_free else "paid",
            "browser_fast",
        )
    except Exception:
        pass
    if not api_key:
        if use_free:
            return jsonify({
                'error': 'Gemini 無料枠のAPIキーが設定されていません。',
                'code': 'user_api_key_missing',
            }), 400
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

    response_payload = {
        'status': 'ok',
        'api_key': api_key,
        'key_source': key_source,
        'model': model_key,
        'thread_id': thread.public_id if thread else None,
        'parent_id': parent_message.id if parent_message else None,
        'history': _browser_fast_mode_history(thread, parent_message),
    }
    if data.get('math_steps') is True:
        response_payload['math_steps_system_prompt'] = build_math_steps_system_prompt(
            normalize_math_steps_config(getattr(current_user, 'math_steps_config', None))
        )
    response = jsonify(response_payload)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    return response

# プロンプトバーの設定と送信時のマスク参照を、プロンプトごとに Message.prompt_options へ記録するための許可リスト。
# 歯車ボタンから開く項目（SysPrompt・Compress・一時チャットの設定）と、
# 自由記述の項目（TTSスタイル・ネガティブプロンプト・stop文字列など）は記録しない。
_PROMPT_OPTION_KEYS = (
    'turn_time_notice',
    'canvas_mode', 'coding_mode', 'browser_fast_mode', 'batch_mode', 'math_steps',
    'enable_search', 'gemini_free_tier', 'gemini_flex', 'enable_url_context', 'enable_maps', 'enable_python',
    'enable_file_creation', 'enable_mcp', 'enable_skills', 'enable_thinking', 'thinking_level',
    'thinking_budget', 'reasoning_effort', 'safety_setting', 'enable_prompt_caching',
    'image_vision_model', 'image_mask',
    'tts_voice', 'tts_voice_custom', 'tts_language', 'tts_speed',
    'image_size', 'image_quality', 'image_complexity', 'image_format', 'image_compression',
    'image_frame_count', 'image_frame_duration_ms',
    'gemini_image_aspect', 'gemini_image_size',
    'grok_image_aspect', 'grok_image_resolution', 'grok_image_quality',
    'grok_image_format', 'grok_image_count',
    'ideogram_operation', 'ideogram_aspect', 'ideogram_resolution', 'ideogram_quality', 'ideogram_speed',
    'ideogram_magic_prompt', 'ideogram_style_type', 'ideogram_count', 'ideogram_seed',
    'xai_temperature', 'xai_top_p', 'xai_max_completion_tokens', 'xai_seed',
    'xai_presence_penalty', 'xai_frequency_penalty', 'xai_response_format',
    'xai_tool_choice', 'xai_parallel_tool_calls', 'xai_logprobs', 'xai_top_logprobs',
    'grok_video_duration', 'grok_video_aspect', 'grok_video_resolution',
    'gemini_video_duration', 'gemini_video_aspect', 'gemini_video_resolution',
    'music_instrumental',
    'ocr_table_format', 'ocr_extract_header', 'ocr_extract_footer',
    'ocr_include_blocks', 'ocr_include_image_base64', 'ocr_pages',
)


def _build_prompt_options_record(source, overrides=None):
    """Return the prompt-bar settings and mask reference of one prompt as JSON (None when empty)."""
    if not isinstance(source, dict):
        source = {}
    record = {}
    for key in _PROMPT_OPTION_KEYS:
        value = source.get(key)
        if isinstance(value, str):
            value = value.strip()[:512 if key == 'image_mask' else 80]
            if not value:
                continue
        elif value is None or not isinstance(value, (bool, int, float)):
            continue
        record[key] = value
    if overrides:
        record.update(overrides)
    if not record:
        return None
    try:
        return json.dumps(record, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return None


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
    if len(refs) > 4:
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
    thought_record = {'text': thought_text} if thought_text else {}
    pricing_input = _cost_input_tokens(data.get('pricing_usage'))
    if pricing_input is not None:
        thought_record['pricing_input_tokens'] = pricing_input
    pricing_call = None
    if 'pricing_usage' in data:
        pricing_call = ModelUsageCall(model_key, 'google')
        pricing_call.update_usage(data.get('pricing_usage') or {})
        pricing_call.finished = True
        thought_record['pricing_calls'] = [pricing_call.snapshot()]
    thought_payload = json.dumps(thought_record, ensure_ascii=False) if thought_record else None
    assistant_content = assistant_text
    if is_enc:
        assistant_content = encrypt_val(assistant_content)
        if thought_payload:
            thought_payload = encrypt_val(thought_payload)

    user_tokens = count_tokens(user_text, model_key)
    thought_tokens = count_tokens(thought_text, model_key) if thought_text else 0
    assistant_tokens = count_tokens_for_display(assistant_text, model_key, thought_text)
    if pricing_call is not None:
        if pricing_call.record['output_tokens'] is not None:
            assistant_tokens = pricing_call.record['output_tokens']
        if pricing_call.record['thought_tokens'] is not None:
            thought_tokens = pricing_call.record['thought_tokens']
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
        math_steps_requested = data.get('math_steps') is True
        if not math_steps_requested:
            raw_prompt_options = data.get('prompt_options')
            math_steps_requested = (
                isinstance(raw_prompt_options, dict)
                and raw_prompt_options.get('math_steps') is True
            )
        user_msg = Message(
            thread_id=thread.id,
            role='user',
            content=user_content,
            model=model_key,
            image_url=json.dumps(refs) if refs else None,
            is_encrypted=is_enc,
            prompt_options=_build_prompt_options_record(
                data.get('prompt_options'),
                {
                    'browser_fast_mode': True,
                    'batch_mode': False,
                    'coding_mode': False,
                    'math_steps': math_steps_requested,
                },
            ),
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
