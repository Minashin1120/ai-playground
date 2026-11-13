# -----------------------------------------------------------
# API Routes: chat stream
# -----------------------------------------------------------
# /chat_stream と SSE の購読・再生・引用元の解決。server/routes_chat.py から分割。
from contextlib import contextmanager


# region: Stream subscription, SSE, and replay helpers
@contextmanager
def _chat_stream_subscription(job_id):
    """Own the subscription even if the client leaves during the initial replay."""
    pubsub = redis_conn.pubsub()
    started = time.monotonic()
    phase = 'done'
    op = getattr(g, 'diagnostic_op', None)
    _record_server_diagnostic(current_user.id, 'server.stream.first', op=op,
                              job=job_id, phase='start', reason_code=120)
    try:
        pubsub.subscribe(f"ai_chat:channel:{job_id}")
        yield pubsub, started
    except GeneratorExit:
        phase = 'cancelled'
        raise
    except Exception:
        phase = 'error'
        raise
    finally:
        try:
            pubsub.close()
        finally:
            _record_server_diagnostic(current_user.id,
                'server.stream.error' if phase == 'error' else 'server.stream.end',
                op=op, job=job_id, phase=phase, reason_code=120,
                ms=round((time.monotonic() - started) * 1000))


def _chat_stream_messages(pubsub, started, timeout=600, heartbeat=15):
    """Poll within the deadline; blank-line heartbeats expose disconnected readers."""
    deadline = started + timeout
    next_heartbeat = time.monotonic() + heartbeat
    while True:
        now = time.monotonic()
        if now >= deadline:
            _record_server_diagnostic(current_user.id, 'server.stream.error',
                op=getattr(g, 'diagnostic_op', None), reason_code=123,
                ms=round((now - started) * 1000))
            return
        message = pubsub.get_message(timeout=min(1.0, deadline - now))
        if message is not None and message['type'] == 'message':
            yield message
            next_heartbeat = time.monotonic() + heartbeat
        elif time.monotonic() >= next_heartbeat:
            yield None
            next_heartbeat = time.monotonic() + heartbeat

def _resolve_quote_source(thread_id, parent_id, quote_message_id):
    """Return {"role", "number"} for the message a quote was taken from, or None.

    `number` is the 1-based position on the path from the thread root to the new
    message's parent (what the model sees as history); None when the quoted
    message is in the thread but on another branch.
    """
    try:
        quoted_id = int(quote_message_id)
    except (TypeError, ValueError):
        return None
    try:
        rows = db.session.query(Message.id, Message.parent_id, Message.role).filter(
            Message.thread_id == thread_id
        ).all()
    except Exception:
        return None
    by_id = {row.id: row for row in rows}
    quoted = by_id.get(quoted_id)
    if quoted is None or quoted.role not in ('user', 'assistant'):
        return None
    chain = []
    seen = set()
    cursor = parent_id
    while cursor is not None and cursor in by_id and cursor not in seen:
        seen.add(cursor)
        chain.append(cursor)
        cursor = by_id[cursor].parent_id
    chain.reverse()
    number = chain.index(quoted_id) + 1 if quoted_id in seen else None
    return {'role': quoted.role, 'number': number}

def _replay_finished_stream_events(job_id):
    """Replays what a job already published when it ended before the stream subscribed.

    The worker publishes to a Redis channel that keeps nothing for late subscribers, so a job that
    finishes right after it is queued (for example a failed attachment check) would leave the client
    waiting for an event that never arrives. Returns True once the job's final event was replayed.
    """
    final_raw = redis_conn.get(f"stream_acc:{job_id}:final")
    if not final_raw:
        return False
    cached_thought = redis_conn.get(f"stream_acc:{job_id}:thought")
    if cached_thought:
        yield json.dumps({"type": "thought", "content": cached_thought.decode("utf-8", "ignore")}) + "\n"
    cached_content = redis_conn.get(f"stream_acc:{job_id}:content")
    if cached_content:
        yield json.dumps({"type": "content", "content": cached_content.decode("utf-8", "ignore")}) + "\n"
    cached_search = redis_conn.get(f"stream_acc:{job_id}:search")
    if cached_search:
        yield json.dumps({"type": "search_status", "content": cached_search.decode("utf-8", "ignore")}) + "\n"
    for _, raw in (redis_conn.hgetall(f"stream_acc:{job_id}:python") or {}).items():
        try:
            yield json.dumps({"type": "python", "content": json.loads(raw)}) + "\n"
        except Exception:
            continue
    if final_raw.decode("utf-8", "ignore").strip().lower() == "error":
        cached_error = redis_conn.get(f"stream_acc:{job_id}:error")
        if cached_error:
            yield cached_error.decode("utf-8", "ignore") + "\n"
        else:
            yield json.dumps({"type": "error", "content": "The job has ended with an error. Please reload."}) + "\n"
    else:
        yield json.dumps({"type": "done", "content": "OK"}) + "\n"
    return True


# region: /chat_stream
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
    batch_mode = bool(data.get('batch_mode'))
    if batch_mode and not _is_batch_model(model_key):
        return jsonify({
            'error': 'Batch APIは対応するGemini／OpenAI／xAI／Z.AIのテキストモデル（画像入力を含む）で利用できます。動画・音声・埋め込み・検索専用モデル等では利用できません。'
        }), 400
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
    resolved_auth = _resolve_chat_model_auth(current_user, model_key, gemini_free_tier=data.get('gemini_free_tier'))
    if resolved_auth.get("error_code"):
        return jsonify({
            "error": resolved_auth.get("error") or "APIキーが設定されていません。",
            "code": resolved_auth["error_code"],
            "model": model_key,
            "provider": resolved_auth.get("provider"),
        }), 400
    if batch_mode:
        gemini_runtime = resolved_auth.get('gemini_runtime') or {}
        if resolved_auth.get('provider') == 'gemini' and gemini_runtime.get('backend') == 'vertex_ai':
            return jsonify({'error': 'Batch APIは現在 Gemini API モードでのみ利用できます。Vertex AI モードを切り替えてください。'}), 400
        if coding_mode:
            return jsonify({'error': 'Batch APIではCoding Modeを利用できません。'}), 400
    for bounded_key in ('quote_text', 'system_prompt', 'marker_system_prompt', 'thread_custom_instruction'):
        bounded_value = data.get(bounded_key)
        if bounded_value is not None and len(str(bounded_value)) > 100_000:
            return jsonify({'error': f'{bounded_key} is too large'}), 400
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}_{secrets.token_hex(8)}"
    _record_server_diagnostic(current_user.id, 'server.request.start', op=getattr(g, 'diagnostic_op', None), job=job_id, route='/chat_stream')
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
        skill_uuid_val = data.get('skill_uuid') or data.get('gem_uuid')
        skill_name_val = None
        if skill_uuid_val:
            skill = Skill.query.filter_by(uuid=skill_uuid_val, user_id=current_user.id).first()
            if skill:
                skill_name_val = skill.name
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
            skill_uuid=skill_uuid_val,
            skill_name=skill_name_val,
            prompt_options=_build_prompt_options_record(
                data,
                {'browser_fast_mode': False, 'batch_mode': batch_mode, 'coding_mode': coding_mode},
            ),
        )
        db.session.add(user_msg)
        assistant_batch_msg = None
        if batch_mode:
            db.session.flush()
            assistant_batch_msg = Message(
                thread=t,
                role='assistant',
                content='',
                model=model_key,
                is_encrypted=user_config['enable_e2ee'],
                parent_id=user_msg.id,
                tokens=0,
                tokens_out=0,
                tokens_thought=0,
                skill_uuid=skill_uuid_val,
                skill_name=skill_name_val,
            )
            db.session.add(assistant_batch_msg)
            db.session.flush()
            db.session.add(GeminiBatchJob(
                job_id=job_id,
                user_id=current_user.id,
                thread_id=t.id,
                user_message_id=user_msg.id,
                assistant_message_id=assistant_batch_msg.id,
                model=model_key,
                gemini_free_tier=_coerce_gemini_free_tier(data.get('gemini_free_tier')),
                provider=resolved_auth.get('provider') or get_model_api_provider(model_key) or 'unknown',
                state='JOB_STATE_QUEUED',
                status_text='Batch APIへ送信する準備中です',
            ))
        if current_user.use_last_chat_settings:
            current_user.last_model = data.get('model')
            current_user.last_enable_search = bool(data.get('enable_search'))
            current_user.last_gemini_free_tier = _coerce_gemini_free_tier(data.get('gemini_free_tier'))
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
            current_user.last_enable_mcp = bool(data.get('enable_mcp', True))
            current_user.last_safety_setting = (data.get('safety_setting') or current_user.last_safety_setting or "default")
        t.last_model = model_key
        t.last_skill_uuid = skill_uuid_val
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
        quote_source = _resolve_quote_source(thread_id, parent_id, data.get('quote_message_id'))
        if quote_source:
            try:
                redis_conn.setex(f"quote_source:{job_id}", 600, json.dumps(quote_source))
            except: pass

    sys_prompt = data.get('system_prompt')
    if sys_prompt:
        try:
            redis_conn.setex(f"sys:{job_id}", 600, sys_prompt)
        except: pass

    enable_skills_requested = _coerce_bool_or_none(data.get('enable_skills'))
    if enable_skills_requested is None:
        # Older clients do not send the switch; keep the previous always-on behavior.
        enable_skills_requested = True
    options = {
        'system_prompt': None,
        'enable_search': data.get('enable_search'),
        'gemini_free_tier': _coerce_gemini_free_tier(data.get('gemini_free_tier')),
        'enable_url_context': data.get('enable_url_context'),
        'enable_maps': data.get('enable_maps'),
        'disable_auto_search': data.get('disable_auto_search'),
        'enable_python': data.get('enable_python'),
        'enable_file_creation': data.get('enable_file_creation'),
        'enable_thinking': data.get('enable_thinking'),
        'thinking_level': data.get('thinking_level'),
        'thinking_budget': data.get('thinking_budget'),
        'reasoning_effort': data.get('reasoning_effort'),
        'enable_system_prompt': data.get('enable_system_prompt'),
        'enable_mcp': data.get('enable_mcp'),
        'enable_skills': enable_skills_requested,
        'marker_system_prompt': data.get('marker_system_prompt'),
        'safety_setting': data.get('safety_setting'),
        'tts_voice': data.get('tts_voice'),
        'tts_voice_custom': data.get('tts_voice_custom'),
        'tts_language': data.get('tts_language'),
        'tts_style': data.get('tts_style'),
        'tts_speed': data.get('tts_speed'),
        'image_size': data.get('image_size'),
        'image_quality': data.get('image_quality'),
        'image_complexity': data.get('image_complexity'),
        'image_format': data.get('image_format'),
        'image_compression': data.get('image_compression'),
        'image_frame_count': data.get('image_frame_count'),
        'image_frame_duration_ms': data.get('image_frame_duration_ms'),
        'image_mask': data.get('image_mask'),
        'gemini_image_aspect': data.get('gemini_image_aspect'),
        'gemini_image_size': data.get('gemini_image_size'),
        'grok_image_aspect': data.get('grok_image_aspect'),
        'grok_image_resolution': data.get('grok_image_resolution'),
        'grok_image_quality': data.get('grok_image_quality'),
        'grok_image_format': data.get('grok_image_format'),
        'grok_image_count': data.get('grok_image_count'),
        'ideogram_operation': data.get('ideogram_operation'),
        'ideogram_aspect': data.get('ideogram_aspect'),
        'ideogram_resolution': data.get('ideogram_resolution'),
        'ideogram_quality': data.get('ideogram_quality'),
        'ideogram_speed': data.get('ideogram_speed'),
        'ideogram_magic_prompt': data.get('ideogram_magic_prompt'),
        'ideogram_style_type': data.get('ideogram_style_type'),
        'ideogram_negative_prompt': data.get('ideogram_negative_prompt'),
        'ideogram_count': data.get('ideogram_count'),
        'ideogram_seed': data.get('ideogram_seed'),
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
        'gemini_video_duration': data.get('gemini_video_duration'),
        'gemini_video_aspect': data.get('gemini_video_aspect'),
        'gemini_video_resolution': data.get('gemini_video_resolution'),
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
        'skill_uuid': skill_uuid_val,
        'gem_uuid': skill_uuid_val,
        'enable_prompt_caching': enable_pc_request,
        'prompt_cache_key': None,
        'coding_mode': coding_mode,
        'coding_target': coding_target if coding_mode else None,
        'coding_candidates': coding_candidates if coding_mode else [],
        'batch_mode': batch_mode,
    }
    # Persist prompt-cache flags on the thread (provider locked while enabled)
    try:
        if enable_pc_request:
            t.enable_prompt_caching = True
            if provider_for_pc:
                t.prompt_cache_provider = provider_for_pc
            # Z.AI GLM uses automatic context caching and does not accept an app-supplied cache key.
            if provider_for_pc != 'zai':
                options['prompt_cache_key'] = _thread_prompt_cache_key(t)
        elif getattr(t, 'enable_prompt_caching', False) or getattr(t, 'prompt_cache_provider', None):
            _forget_prompt_cache(t)
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
    if not batch_mode:
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

    if batch_mode:
        def generate_batch_ack():
            if thread_stream_id:
                yield json.dumps({'type': 'thread_id', 'content': thread_stream_id}) + '\n'
            yield json.dumps({'type': 'job_id', 'content': job_id}) + '\n'
            yield json.dumps({'type': 'status', 'content': 'Batch APIへ送信する準備中です'}) + '\n'
            yield json.dumps({'type': 'done', 'content': 'BATCH_ACCEPTED'}) + '\n'
        resp = Response(stream_with_context(generate_batch_ack()), mimetype='application/x-ndjson')
        resp.headers['Cache-Control'] = 'no-cache, no-transform'
        resp.headers['X-Accel-Buffering'] = 'no'
        return resp

    def generate():
        with _chat_stream_subscription(job_id) as (pubsub, start_time):
            _latency_mark_once(job_id, "route_stream_open_ms")
            if thread_stream_id:
                yield json.dumps({"type": "thread_id", "content": thread_stream_id}) + "\n"
            yield json.dumps({"type": "job_id", "content": job_id}) + "\n"
            replayed_final = False
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
                # The job may have ended before this stream subscribed; its events are not on the channel any more.
                replayed_final = bool((yield from _replay_finished_stream_events(job_id)))
            except Exception:
                pass
            try:
                refresh_count = 0
                for message in (() if replayed_final else _chat_stream_messages(pubsub, start_time)):
                    if message is None:
                        yield "\n"
                        continue
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
