# --- Background Tasks ---

def _iter_chat_history_ancestors(thread_id, parent_id, user_id, max_messages=0):
    """Read branch links cheaply, then load only selected ancestors in batches."""
    if not parent_id:
        return
    links = dict(
        db.session.query(Message.id, Message.parent_id)
        .join(Thread, Thread.id == Message.thread_id)
        .filter(Message.thread_id == thread_id, Thread.user_id == user_id)
        .all()
    )
    ancestor_ids = []
    seen = set()
    while parent_id in links and parent_id not in seen:
        seen.add(parent_id)
        ancestor_ids.append(parent_id)
        if max_messages > 0 and len(ancestor_ids) >= max_messages:
            break
        parent_id = links[parent_id]
    del links, seen
    for offset in range(0, len(ancestor_ids), 64):
        batch_ids = ancestor_ids[offset:offset + 64]
        batch = {
            row.id: row for row in Message.query.filter(
                Message.thread_id == thread_id, Message.id.in_(batch_ids)
            ).all()
        }
        for message_id in batch_ids:
            row = batch.get(message_id)
            if row is None:
                return  # An ancestor was deleted while loading the branch.
            yield row
        del batch


_GEMINI_BUILTIN_TOOL_FIELDS = (
    "google_search",
    "google_search_retrieval",
    "url_context",
    "google_maps",
    "code_execution",
    "file_search",
    "enterprise_web_search",
)


def _gemini_combines_builtin_and_function_tools(tools):
    """Geminiのツール一覧が、組み込みツールと関数ツールの両方を含むか。"""
    has_builtin = False
    has_function = False
    for tool in tools or []:
        if callable(tool):
            has_function = True
            continue
        if isinstance(tool, dict):
            fields = {key: value for key, value in tool.items() if value is not None}
        else:
            fields = {
                name: getattr(tool, name, None)
                for name in _GEMINI_BUILTIN_TOOL_FIELDS + ("function_declarations",)
            }
        if fields.get("function_declarations") or fields.get("functionDeclarations"):
            has_function = True
        if any(fields.get(name) is not None for name in _GEMINI_BUILTIN_TOOL_FIELDS):
            has_builtin = True
    return has_builtin and has_function


def background_chat_task(job_id, thread_id, model_key, message_id, options, user_id, user_config):
    import re

    with app.app_context():
        g.diagnostic_user_id = user_id
        usage_ledger = ModelUsageLedger()
        _record_server_diagnostic(user_id, 'server.worker.start', job=job_id)
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        _latency_mark_once(job_id, "worker_started_ms")
        batch_mode = bool(options.get('batch_mode'))
        if batch_mode:
            # Batch API requests are declarative. Disable interactive tools and
            # tool-calling options before building the GenerateContentRequest.
            for _batch_option in (
                'enable_search', 'enable_url_context', 'enable_maps',
                'enable_python', 'enable_file_creation', 'enable_mcp',
            ):
                options[_batch_option] = False
        coding_stream_buffer = ""
        coding_stream_target_id = None
        coding_stream_code = None
        coding_stream_language = "text"
        coding_stream_edit_index = 0
        coding_candidate_map = {
            str(item.get("id")): item
            for item in (options.get("coding_candidates") or [])
            if isinstance(item, dict) and item.get("id")
        }

        def _emit_coding_diff(content):
            event_payload = {"type": "coding_diff", "content": content}
            r.publish(channel, json.dumps(event_payload, ensure_ascii=False))
            try:
                cache_key = f"stream_acc:{job_id}:coding_diff"
                r.rpush(cache_key, json.dumps(content, ensure_ascii=False))
                r.ltrim(cache_key, -50, -1)
                r.expire(cache_key, 600)
            except Exception:
                pass

        def _process_coding_stream_line(raw_line):
            nonlocal coding_stream_target_id
            nonlocal coding_stream_code
            nonlocal coding_stream_language
            nonlocal coding_stream_edit_index
            line = str(raw_line or "").strip()
            if not line or line.startswith("```"):
                return
            try:
                event = json.loads(line)
            except Exception:
                return
            if not isinstance(event, dict):
                return
            event_type = str(event.get("type") or "").strip().lower()
            if event_type == "target":
                target_id = str(event.get("target_id") or "").strip()
                candidate = coding_candidate_map.get(target_id)
                if candidate:
                    coding_stream_target_id = target_id
                    coding_stream_code = str(candidate.get("code") or "")
                    coding_stream_language = str(candidate.get("language") or "text")
                return
            if event_type != "edit":
                return
            search = event.get("search")
            replace = event.get("replace")
            if not isinstance(search, str) or not search or not isinstance(replace, str):
                return
            if coding_stream_code is None:
                matching = [
                    candidate for candidate in coding_candidate_map.values()
                    if str(candidate.get("code") or "").count(search) == 1
                ]
                if len(matching) != 1:
                    return
                candidate = matching[0]
                coding_stream_target_id = str(candidate.get("id") or "")
                coding_stream_code = str(candidate.get("code") or "")
                coding_stream_language = str(candidate.get("language") or "text")
            if coding_stream_code.count(search) != 1:
                return
            before_code = coding_stream_code
            after_code = before_code.replace(search, replace, 1)
            if len(after_code) > 500_000:
                return
            coding_stream_code = after_code
            coding_stream_edit_index += 1
            _latency_mark_once(job_id, "provider_first_content_ms")
            _emit_coding_diff({
                "target_id": coding_stream_target_id,
                "language": coding_stream_language,
                "edit_index": coding_stream_edit_index,
                "diff": build_coding_mode_unified_diff(
                    before_code,
                    after_code,
                    coding_stream_language,
                ),
            })

        def _consume_coding_stream_chunk(chunk, flush=False):
            nonlocal coding_stream_buffer
            coding_stream_buffer += str(chunk or "")
            while "\n" in coding_stream_buffer:
                line, coding_stream_buffer = coding_stream_buffer.split("\n", 1)
                _process_coding_stream_line(line)
            if flush and coding_stream_buffer.strip():
                _process_coding_stream_line(coding_stream_buffer)
                coding_stream_buffer = ""

        def _append_limited(key, chunk, limit=1_000_000):
            try:
                if chunk is None:
                    return
                if not isinstance(chunk, str):
                    chunk = str(chunk)
                r.append(key, chunk)
                size = r.strlen(key)
                if size and size > limit:
                    curr = r.get(key) or b""
                    if len(curr) > limit:
                        r.set(key, curr[-limit:])
                r.expire(key, 600)
            except Exception:
                pass

        # Persist stream errors as assistant messages so they remain visible after reload.
        stream_error_persisted = False

        def _persist_stream_error(error_text):
            """Save an assistant bubble for stream failures so reloads still show the error."""
            nonlocal stream_error_persisted
            if stream_error_persisted:
                return
            stream_error_persisted = True
            try:
                parent = Message.query.get(message_id)
                if not parent or parent.thread_id != thread_id:
                    return
                if getattr(parent, "thread", None) is not None and parent.thread.user_id != user_id:
                    return

                partial = ""
                try:
                    cached = r.get(f"stream_acc:{job_id}:content")
                    if cached:
                        partial = cached.decode("utf-8", "ignore")
                except Exception:
                    partial = ""
                thought_text = ""
                try:
                    cached_thought = r.get(f"stream_acc:{job_id}:thought")
                    if cached_thought:
                        thought_text = cached_thought.decode("utf-8", "ignore")
                except Exception:
                    thought_text = ""

                final_content = format_chat_error_content(error_text, partial)
                is_enc = bool(user_config.get("enable_e2ee", False))
                content_to_store = encrypt_val(final_content) if is_enc else final_content
                thought_metadata = {"text": thought_text} if thought_text and str(thought_text).strip() else {}
                if usage_ledger.calls:
                    thought_metadata['pricing_calls'] = usage_ledger.snapshot()
                if search_used:
                    thought_metadata["search_used"] = True
                thought_payload = json.dumps(thought_metadata, ensure_ascii=False) if thought_metadata else None
                thought_data = encrypt_val(thought_payload) if is_enc and thought_payload else thought_payload

                skill_uuid_val = options.get("skill_uuid") or options.get("gem_uuid")
                skill_name_val = None
                if skill_uuid_val:
                    skill = Skill.query.filter_by(uuid=skill_uuid_val).first()
                    if skill:
                        skill_name_val = skill.name
                if not skill_uuid_val and _used_skills:
                    skill_uuid_val = _used_skills[0].get("uuid")
                    skill_name_val = _used_skills[0].get("name")

                msg_entry = Message(
                    thread_id=thread_id,
                    role="assistant",
                    content=content_to_store,
                    model=model_key,
                    job_id=job_id,
                    thought_data=thought_data,
                    tokens_out=usage_ledger.token_totals()['output_tokens'] or 0,
                    tokens=usage_ledger.token_totals()['output_tokens'] or 0,
                    tokens_thought=usage_ledger.token_totals()['thought_tokens'] or 0,
                    is_encrypted=is_enc,
                    parent_id=message_id,
                    skill_uuid=skill_uuid_val,
                    skill_name=skill_name_val,
                )
                db.session.add(msg_entry)
                th = Thread.query.get(thread_id)
                if th:
                    th.updated_at = datetime.utcnow()
                    th.last_model = model_key
                    # A model-invoked use_skill is shown on the message only.  The
                    # thread's last_skill_uuid is the user's explicit selection.
                safe_db_commit()
            except Exception as exc:
                logger.exception(
                    "Failed to persist stream error message for job %s: %s", job_id, exc
                )
                try:
                    db.session.rollback()
                except Exception:
                    pass

        _diagnostic_first_types = set()
        search_used = False
        def pub(dt, d, **metadata):
            nonlocal search_used
            if dt == "search_status" and d in ("searching", "done"):
                search_used = True
            if dt in ('content', 'thought', 'status', 'done', 'error') and dt not in _diagnostic_first_types:
                _diagnostic_first_types.add(dt)
                _record_server_diagnostic(user_id, 'server.stream.error' if dt == 'error' else 'server.stream.end' if dt == 'done' else 'server.stream.first', job=job_id, type=dt)
            if options.get("coding_mode") and dt == "content" and not metadata.get("coding_final"):
                _consume_coding_stream_chunk(d)
                return
            if dt == "status":
                _latency_mark_once(job_id, "provider_first_status_ms")
            elif dt == "thought":
                _latency_mark_once(job_id, "provider_first_thought_ms")
            elif dt == "content":
                _latency_mark_once(job_id, "provider_first_content_ms")
            elif dt in ("done", "error"):
                _latency_mark_once(job_id, "worker_done_ms")
            # Persist before publish so a client reload right after the error event
            # already finds the saved assistant bubble.
            if dt == "error":
                d = readable_provider_error(d)
                _persist_stream_error(d)
            event_payload = {"type": dt, "content": d}
            event_payload.update(metadata)
            r.publish(channel, json.dumps(event_payload))
            try:
                if dt == "content":
                    _append_limited(f"stream_acc:{job_id}:content", d)
                elif dt == "thought":
                    _append_limited(f"stream_acc:{job_id}:thought", d)
                elif dt == "status":
                    r.setex(f"stream_acc:{job_id}:status", 600, d)
                elif dt == "python":
                    py = d if isinstance(d, dict) else {}
                    py_id = py.get("id") or "default"
                    r.hset(f"stream_acc:{job_id}:python", py_id, json.dumps(py))
                    r.expire(f"stream_acc:{job_id}:python", 600)
                elif dt == "search_status":
                    r.setex(f"stream_acc:{job_id}:search", 600, d)
                elif dt in ("mcp", "mcp_decision_request", "mcp_decision_resolved"):
                    # MCP実行カード・確認ダイアログ用イベントのリプレイ保存（発生順）
                    try:
                        entry = {"type": dt, "content": d}
                        entry.update(metadata)
                        r.rpush(f"stream_acc:{job_id}:mcp", json.dumps(entry, ensure_ascii=False))
                        r.ltrim(f"stream_acc:{job_id}:mcp", -200, -1)
                        r.expire(f"stream_acc:{job_id}:mcp", 600)
                    except Exception:
                        pass
                elif dt in ["error", "done"]:
                    r.setex(f"stream_acc:{job_id}:final", 600, dt)
                    if dt == "error":
                        r.setex(f"stream_acc:{job_id}:error", 600, json.dumps(event_payload))
            except Exception:
                pass
        
        def check_stop():
            try:
                res = r.get(f"stop_job:{job_id}")
                if res:
                    # Clear it immediately to avoid double processing if needed
                    # but actually we want all loops to see it if multiple.
                    # r.delete(f"stop_job:{job_id}") 
                    log_force(f"STREAM-STOP-DETECTED: Job {job_id} stop flag found in Redis.")
                    return True
            except Exception as e:
                log_force(f"STREAM-STOP-ERROR: Failed to check stop flag: {e}")
            return False

        def _refresh_pending_job():
            try:
                pending_data = json.dumps({
                    "job_id": job_id,
                    "message_id": message_id,
                    "created_at": int(time.time()),
                    "model": model_key,
                })
                r.setex(f"pending_job:{user_id}:{thread_id}", 600, pending_data)
            except Exception:
                pass

        def _mark_provider_request_started():
            _latency_mark_once(job_id, "provider_request_started_ms")

        def _submit_gemini_batch(provider_model, request_contents, config_kwargs):
            """Submit one inline GenerateContentRequest and persist its provider name."""
            row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=user_id).first()
            if not row:
                raise RuntimeError('Gemini Batchジョブの記録が見つかりません')
            # Keep the visible state at "preparing" until the provider returns
            # a real Batch job name. This prevents a stalled SDK call from
            # leaving a permanent "sending" bubble in the conversation.
            row.state = 'JOB_STATE_QUEUED'
            row.status_text = 'Batch APIへ送信する準備中です'
            safe_db_commit()
            request = {'contents': _batch_json_value(request_contents)}
            if config_kwargs:
                request['config'] = _batch_json_value(types.GenerateContentConfig(**config_kwargs))
            request_size = len(json.dumps(request, ensure_ascii=False).encode('utf-8'))
            if request_size > 20 * 1024 * 1024:
                raise ValueError('Batch APIのインライン入力上限（20MiB）を超えています')
            provider_job = g_client.batches.create(
                model=provider_model,
                src=[request],
                config={'display_name': f'ai-chat-{job_id[-24:]}'},
            )
            provider_name = (
                getattr(provider_job, 'name', None)
                or (provider_job.get('name') if isinstance(provider_job, dict) else None)
            )
            if not provider_name:
                raise RuntimeError('Gemini Batch APIからジョブ名が返されませんでした')
            row.provider_job_name = str(provider_name)
            row.state = _normalize_batch_state(
                getattr(getattr(provider_job, 'state', None), 'name', None)
                or getattr(provider_job, 'state', None)
                or (provider_job.get('state') if isinstance(provider_job, dict) else None)
                or 'JOB_STATE_PENDING'
            )
            row.status_text = _batch_state_label(row.state)
            db.session.add(row)
            safe_db_commit()
            pub('status', row.status_text)
            pub('done', 'BATCH_ACCEPTED')
            return True

        def _submit_openai_batch(client, request_kwargs):
            """Upload one Responses API request as JSONL and create an OpenAI Batch."""
            row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=user_id).first()
            if not row:
                raise RuntimeError('Batchジョブの記録が見つかりません')
            row.state = 'JOB_STATE_QUEUED'
            row.status_text = 'Batch APIへ送信する準備中です'
            safe_db_commit()

            request_body = dict(request_kwargs or {})
            request_body.pop('stream', None)
            request_body['store'] = False
            batch_line = {
                'custom_id': job_id,
                'method': 'POST',
                'url': '/v1/responses',
                'body': request_body,
            }
            jsonl_bytes = (json.dumps(batch_line, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
            if len(jsonl_bytes) > 200 * 1024 * 1024:
                raise ValueError('Batch APIの入力ファイル上限（200MB）を超えています')
            batch_file = BytesIO(jsonl_bytes)
            batch_file.name = f'{job_id}.jsonl'
            uploaded = client.files.create(file=batch_file, purpose='batch')
            input_file_id = _batch_field(uploaded, 'id')
            if not input_file_id:
                raise RuntimeError('OpenAI Files APIから入力ファイルIDが返されませんでした')
            provider_batch = client.batches.create(
                input_file_id=input_file_id,
                endpoint='/v1/responses',
                completion_window='24h',
            )
            provider_batch_id = _batch_field(provider_batch, 'id')
            if not provider_batch_id:
                raise RuntimeError('OpenAI Batch APIからジョブIDが返されませんでした')
            row.provider_job_name = str(provider_batch_id)
            row.provider = 'openai'
            row.state = _openai_batch_state_label(_batch_field(provider_batch, 'status'))
            row.status_text = _batch_state_label(row.state)
            db.session.add(row)
            safe_db_commit()
            pub('status', row.status_text)
            pub('done', 'BATCH_ACCEPTED')
            return True

        def _submit_xai_batch(api_key, request_kwargs):
            """Create an xAI Batch and add one Responses API request inline."""
            row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=user_id).first()
            if not row:
                raise RuntimeError('xAI Batchジョブの記録が見つかりません')
            if not api_key:
                raise RuntimeError('xAI APIキーを確認してください')
            row.state = 'JOB_STATE_QUEUED'
            row.status_text = 'Batch APIへ送信する準備中です'
            safe_db_commit()

            request_body = dict(request_kwargs or {})
            request_body.pop('stream', None)
            request_body.pop('store', None)
            batch_request = {
                'batch_request_id': job_id,
                'batch_request': {'responses': request_body},
            }
            request_size = len(json.dumps(batch_request, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            if request_size > 25 * 1024 * 1024:
                raise ValueError('xAI Batch APIの1リクエスト上限（25MB）を超えています')
            headers = {
                'Authorization': f'Bearer {str(api_key)}',
                'Content-Type': 'application/json',
            }
            base_url = f'https://{_XAI_API_HOST}/v1/batches'
            create_response = requests.post(
                base_url,
                headers=headers,
                json={'name': f'ai-chat-{job_id[-24:]}'},
                timeout=60,
            )
            create_payload = create_response.json() if create_response.content else {}
            if not create_response.ok:
                detail = _batch_field(create_payload, 'error') or create_payload
                if isinstance(detail, dict):
                    detail = _batch_field(detail, 'message') or detail
                raise RuntimeError(detail or f'xAI Batch API HTTP {create_response.status_code}')
            provider_batch_id = _batch_field(create_payload, 'batch_id')
            if not provider_batch_id:
                raise RuntimeError('xAI Batch APIからバッチIDが返されませんでした')

            add_response = requests.post(
                f'{base_url}/{quote(str(provider_batch_id), safe="")}/requests',
                headers=headers,
                json={'batch_requests': [batch_request]},
                timeout=60,
            )
            add_payload = add_response.json() if add_response.content else {}
            if not add_response.ok:
                detail = _batch_field(add_payload, 'error') or add_payload
                if isinstance(detail, dict):
                    detail = _batch_field(detail, 'message') or detail
                raise RuntimeError(detail or f'xAI Batchリクエスト HTTP {add_response.status_code}')

            row.provider = 'xai'
            row.provider_job_name = str(provider_batch_id)
            row.state = 'JOB_STATE_PENDING'
            row.status_text = _batch_state_label(row.state)
            db.session.add(row)
            safe_db_commit()
            pub('status', row.status_text)
            pub('done', 'BATCH_ACCEPTED')
            return True

        def _submit_zai_batch(client, request_kwargs):
            """Upload one Chat Completions request and create a Z.AI Batch job."""
            row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=user_id).first()
            if not row:
                raise RuntimeError('Batchジョブの記録が見つかりません')
            row.state = 'JOB_STATE_QUEUED'
            row.status_text = 'Batch APIへ送信する準備中です'
            safe_db_commit()

            request_body = dict(request_kwargs or {})
            request_body.pop('stream', None)
            request_body.pop('stream_options', None)
            batch_line = {
                'custom_id': job_id,
                'method': 'POST',
                'url': '/v4/chat/completions',
                'body': request_body,
            }
            jsonl_bytes = (json.dumps(batch_line, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
            if len(jsonl_bytes) > 200 * 1024 * 1024:
                raise ValueError('Batch APIの入力ファイル上限（200MB）を超えています')
            batch_file = BytesIO(jsonl_bytes)
            batch_file.name = f'{job_id}.jsonl'
            uploaded = client.files.create(file=batch_file, purpose='batch')
            input_file_id = _batch_field(uploaded, 'id')
            if not input_file_id:
                raise RuntimeError('Z.AI Files APIから入力ファイルIDが返されませんでした')
            provider_batch = client.batches.create(
                input_file_id=input_file_id,
                endpoint='/v4/chat/completions',
                completion_window='24h',
            )
            provider_batch_id = _batch_field(provider_batch, 'id')
            if not provider_batch_id:
                raise RuntimeError('Z.AI Batch APIからジョブIDが返されませんでした')
            row.provider = 'zai'
            row.provider_job_name = str(provider_batch_id)
            row.state = _openai_batch_state_label(_batch_field(provider_batch, 'status'))
            row.status_text = _batch_state_label(row.state)
            db.session.add(row)
            safe_db_commit()
            pub('status', row.status_text)
            pub('done', 'BATCH_ACCEPTED')
            return True

        def _persist_batch_failure(error_text):
            row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=user_id).first()
            if row:
                row.state = 'JOB_STATE_FAILED'
                row.error = str(error_text or 'Batch APIへの送信に失敗しました')[:4000]
                row.status_text = _batch_state_label(row.state)
                row.completed_at = datetime.utcnow()
            assistant = Message.query.filter_by(id=getattr(row, 'assistant_message_id', None)).first() if row else None
            if assistant:
                content = f'**{row.status_text}**\n\n{row.error}'
                assistant.content = encrypt_val(content) if assistant.is_encrypted else content
            safe_db_commit()
            pub('done', 'BATCH_FAILED')
        
        usage_context_token = _MODEL_USAGE_CONTEXT.set(usage_ledger)
        try:
            log_force(f"Task Start: model={model_key}, user={user_id}")
            pub("status", "ワーカーがジョブを受信しました。入力を処理中です...")
            user = User.query.get(user_id)
            msg = Message.query.get(message_id)
            if not msg or msg.thread_id != thread_id or msg.thread.user_id != user_id:
                pub("error", "Invalid message")
                return
            message_text = decrypt_val(msg.content) if msg.is_encrypted else msg.content
            img_list = []
            if msg.image_url:
                try:
                    img_list = json.loads(msg.image_url)
                    if not isinstance(img_list, list):
                        img_list = [img_list]
                except: pass
            def _is_non_llm_model(m):
                mk = str(m or "").lower().strip()
                if not mk:
                    return False
                if is_mistral_ocr_model_key(mk):
                    return True
                if "gpt-image" in mk:
                    return True
                if is_ideogram_model_key(mk):
                    return True
                if mk in (
                    "grok-imagine-image-2.0",
                    "grok-imagine-image",
                    "grok-imagine-image-pro",
                    "grok-imagine-image-quality",
                    "grok-imagine-video-1.5-lite",
                    "grok-imagine-video-1.5",
                    "grok-imagine-video",
                ):
                    return True
                if "tts" in mk:
                    return True
                if is_gemini_image_model_key(mk):
                    return True
                if "gemini" in mk and "native-audio" in mk:
                    return True
                if is_gemini_video_model_key(mk) or is_gemini_music_model_key(mk) or is_gemini_embedding_model_key(mk):
                    return True
                return False

            # System Prompt Construction
            base_sys_prompt = options.get('system_prompt')
            if not base_sys_prompt:
                try:
                    sv = r.get(f"sys:{job_id}")
                    if sv: base_sys_prompt = sv.decode('utf-8')
                except: pass
                finally:
                    try: r.delete(f"sys:{job_id}")
                    except: pass
            
            forced_prompt = base_sys_prompt or ""
            global_prompt = None
            user_prompt = None
            use_time_notice = False
            apply_global_prompt = True
            apply_auto_prompt_notices = get_user_auto_system_prompt_notices_enabled(user)
            auto_notice_config = get_user_auto_system_prompt_notices_config(user)
            def _auto_notice_enabled(notice_key):
                return bool(
                    apply_auto_prompt_notices
                    and get_user_auto_system_prompt_notice_enabled(user, notice_key, auto_notice_config)
                )
            def _auto_notice_text(notice_key):
                return get_user_auto_system_prompt_notice_text(user, notice_key, auto_notice_config)
            # Skill catalog / use_skill tool state.  Filled after the model type
            # and attachment checks are known; defined here so every later
            # branch (including stream-error persistence) can reference it.
            _skills_for_tools = []
            _used_skills = []
            _skill_active_uuid = options.get('skill_uuid') or options.get('gem_uuid')
            _skills_tool_enabled = False
            _skills_enabled = options.get('enable_skills', True)
            if isinstance(_skills_enabled, str):
                _skills_enabled = _skills_enabled.strip().lower() not in ('', '0', 'false', 'no', 'off')
            elif isinstance(_skills_enabled, (int, float)):
                _skills_enabled = bool(_skills_enabled)
            else:
                _skills_enabled = _skills_enabled is not False

            def _record_skill_meta(meta):
                if meta.get("ok"):
                    if not any(item.get("uuid") == meta.get("uuid") for item in _used_skills):
                        _used_skills.append({"uuid": meta.get("uuid"), "name": meta.get("name")})
                    log_force(f"Skill tool used: {meta.get('name')}")
                else:
                    log_force(f"Skill tool not found: {str(meta.get('name') or '')[:100]}")

            def _execute_skill_tool(skill_name):
                text, meta = execute_skill_tool(_skills_for_tools, skill_name)
                _record_skill_meta(meta)
                return text

            def _build_attachment_name_block(names):
                if not (is_llm_model and _auto_notice_enabled("attachment_names")):
                    return ""
                template_text = _auto_notice_text("attachment_names")
                return _render_attachment_names_notice(template_text, names)
            # Fetch thread to check instructions
            th = Thread.query.get(thread_id)
            include_global = th.include_global_instruction if th and th.include_global_instruction is not None else True

            try:
                if getattr(user, "apply_global_system_prompt", None) is False:
                    apply_global_prompt = False
            except Exception:
                apply_global_prompt = True
            
            if apply_global_prompt and include_global:
                global_enabled = get_bool_app_setting("global_system_prompt_enabled", True)
                global_value = get_app_setting("global_system_prompt", "") or ""
                if global_enabled:
                    if global_value.strip():
                        global_prompt = global_value
                    else:
                        use_time_notice = True
            
            if options.get('enable_system_prompt') and include_global:
                if user.system_prompt and (user.system_prompt_enabled is None or user.system_prompt_enabled):
                    sp = user.system_prompt
                    if user.enable_e2ee: sp = decrypt_val(sp)
                    user_prompt = sp

            # Thread specific prompt
            local_sys_prompt = None
            if 'thread_custom_instruction' in options:
                raw_local_sys_prompt = options.get('thread_custom_instruction')
            else:
                raw_local_sys_prompt = th.custom_instruction if th else None
            if raw_local_sys_prompt and str(raw_local_sys_prompt).strip():
                local_sys_prompt = str(raw_local_sys_prompt).strip()
            
            combined_prompt = ""
            for part in [forced_prompt, global_prompt, user_prompt]:
                if part and str(part).strip():
                    if combined_prompt:
                        combined_prompt = f"{combined_prompt}\n\n{part}"
                    else:
                        combined_prompt = str(part).strip()
            if local_sys_prompt:
                if combined_prompt:
                    combined_prompt = f"{combined_prompt}\n\n[Chat Specific Instructions]:\n{local_sys_prompt}"
                else:
                    combined_prompt = local_sys_prompt
            if options.get("coding_mode"):
                if combined_prompt:
                    combined_prompt = f"{combined_prompt}\n\n{CODING_MODE_SYSTEM_PROMPT}"
                else:
                    combined_prompt = CODING_MODE_SYSTEM_PROMPT
            
            options['system_prompt'] = combined_prompt

            if _auto_notice_enabled("python") and options.get('enable_python'):
                python_notice = _auto_notice_text("python")
                curr_p = options.get('system_prompt')
                if curr_p and str(curr_p).strip():
                    if python_notice.lower() not in str(curr_p).lower():
                        options['system_prompt'] = f"{python_notice}\n\n{curr_p}"
                else:
                    options['system_prompt'] = python_notice
            if _auto_notice_enabled("marker"):
                marker_prompt = options.get('marker_system_prompt')
                if marker_prompt and str(marker_prompt).strip():
                    marker_notice = _auto_notice_text("marker")
                    curr_p = options.get('system_prompt') or ""
                    if curr_p.strip():
                        if str(marker_notice).strip() not in str(curr_p):
                            options['system_prompt'] = f"{curr_p}\n\n{marker_notice}"
                    else:
                        options['system_prompt'] = marker_notice
            # Media-only models retain their existing prompt contract.
            if use_time_notice and _is_non_llm_model(model_key.lower()):
                options['system_prompt'] = f"{build_global_system_prompt()}\n\n{options.get('system_prompt') or ''}".strip()
            else:
                prompt_record = _prompt_context_record(getattr(msg, 'prompt_options', None))
                if use_time_notice and 'turn_time_notice' not in prompt_record:
                    prompt_record['turn_time_notice'] = build_global_system_prompt()
                    msg.prompt_options = json.dumps(prompt_record, ensure_ascii=False)
                    safe_db_commit()
                message_text = _append_turn_time_notice(message_text, prompt_record)

            if _auto_notice_enabled("mathjax"):
                mathjax_notice = _auto_notice_text("mathjax")
                curr_p = options.get('system_prompt') or ""
                if "MathJax" not in curr_p:
                    if curr_p.strip():
                        options['system_prompt'] = f"{curr_p}\n\n{mathjax_notice}"
                    else:
                        options['system_prompt'] = mathjax_notice

            quote_text = None
            try:
                qv = r.get(f"quote:{job_id}")
                if qv: quote_text = qv.decode('utf-8')
            except: pass
            finally:
                try: r.delete(f"quote:{job_id}")
                except: pass
            quote_source = None
            try:
                qsv = r.get(f"quote_source:{job_id}")
                if qsv: quote_source = json.loads(qsv.decode('utf-8'))
            except Exception:
                quote_source = None
            finally:
                try: r.delete(f"quote_source:{job_id}")
                except: pass

            history = background_load_history(model_key, msg, thread_id, user_id)

            def _load_history_image_parts(include_roles=None, newest_first=False, include_only_images=True, source_messages=None, reverse_within_message=False):
                return background_history_image_parts(
                    history,
                    include_roles=include_roles,
                    newest_first=newest_first,
                    include_only_images=include_only_images,
                    source_messages=source_messages,
                    reverse_within_message=reverse_within_message,
                )

            def _load_message_history_images(raw_urls, seen=None, total_bytes=0, include_only_images=True):
                return background_message_history_images(
                    raw_urls,
                    seen=seen,
                    total_bytes=total_bytes,
                    include_only_images=include_only_images,
                )

            def _build_non_llm_image_context(current_text, include_assistant_images=True):
                return background_non_llm_image_context(
                    history,
                    current_text,
                    include_assistant_images=include_assistant_images,
                )

            model_key = model_key.strip()
            model_key_l = model_key.lower()
            is_openai_search_model = model_key_l in ("gpt-5-search-api", "gpt-4o-search-preview", "gpt-4o-mini-search-preview")
            is_openai_batch_model = _is_openai_batch_model(model_key)
            is_gem = is_gemini_model_key(model_key_l)
            is_claude = is_anthropic_model_key(model_key_l)
            is_deepseek = is_deepseek_model_key(model_key_l)
            is_zai = model_key_l.startswith("glm-")
            is_kimi = 'kimi' in model_key_l
            is_grok = 'grok' in model_key_l and 'gpt' not in model_key_l
            is_mistral_ocr = is_mistral_ocr_model_key(model_key_l)
            is_ideogram = is_ideogram_model_key(model_key_l)
            gemini_backend_mode = "gemini_api"
            is_llm_model = not _is_non_llm_model(model_key_l)

            def _append_history_document_text():
                background_append_history_document_text(history, img_list)

            if is_llm_model and not is_mistral_ocr:
                try:
                    _append_history_document_text()
                except Exception as _hist_doc_exc:
                    log_force(f"History document text failed: {_hist_doc_exc}")
            grok_reasoning_supported = ("grok-4.3" in model_key_l) or ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l) or ("grok-build" in model_key_l) or ("grok-3-mini" in model_key_l) or ("reasoning" in model_key_l and "non-reasoning" not in model_key_l) or ("multi-agent" in model_key_l)
            grok_reasoning_effort_supported = ("grok-4.3" in model_key_l) or ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l) or ("grok-build" in model_key_l) or ("grok-3-mini" in model_key_l) or ("grok-4.20-0309-reasoning" in model_key_l) or ("multi-agent" in model_key_l)
            req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
            reasoning_requested = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
            if is_deepseek and req_reasoning_effort == "none":
                reasoning_requested = False
            pub("status", "モデルへの入力を準備中です...")

            def _is_gemini_text_model(m):
                if "gemini" not in m:
                    return False
                if any(x in m for x in ("image", "nano", "tts", "native-audio")):
                    return False
                return True

            supports_audio_inputs = _is_gemini_text_model(model_key_l)
            supports_video_inputs = supports_audio_inputs
            supports_pdf_inputs = supports_audio_inputs
            supports_docx_inputs = supports_audio_inputs
            supports_text_file_inputs = supports_audio_inputs

            def _openai_cache_fresh(cache, size, mtime, mime):
                if not cache or not cache.file_id:
                    return False
                if size is not None and cache.size_bytes is not None and cache.size_bytes != size:
                    return False
                if mtime is not None and cache.mtime is not None and cache.mtime != mtime:
                    return False
                if mime and cache.mime_type and cache.mime_type != mime:
                    return False
                ttl_hours = 24
                try:
                    ttl_val = os.getenv("OPENAI_FILE_CACHE_TTL_HOURS")
                    if ttl_val and str(ttl_val).strip():
                        ttl_hours = int(ttl_val)
                except Exception:
                    ttl_hours = 24
                try:
                    if ttl_hours > 0 and cache.updated_at:
                        age = (datetime.utcnow() - cache.updated_at).total_seconds()
                        if age > ttl_hours * 3600:
                            return False
                except Exception:
                    pass
                return True

            def _openai_upload_with_retry(client, data, suffix, rel_path, mime=None, size=None, mtime=None, upload_name=None):
                max_attempts = 2
                try:
                    max_attempts = int(os.getenv("OPENAI_FILE_UPLOAD_RETRIES", "2") or "2")
                except Exception:
                    max_attempts = 2
                last_err = None
                for attempt in range(max_attempts):
                    try:
                        _upsert_file_cache(
                            user_id,
                            rel_path,
                            "openai",
                            state="UPLOADING",
                            last_error=None,
                            retries=attempt + 1
                        )
                        safe_db_commit()
                        with tempfile.NamedTemporaryFile(suffix=suffix or '.bin') as tmp:
                            tmp.write(data)
                            tmp.flush()
                            tmp.seek(0)
                            if upload_name:
                                up = client.files.create(
                                    file=(upload_name, tmp, mime or "application/octet-stream"),
                                    purpose="user_data",
                                )
                            else:
                                up = client.files.create(file=tmp, purpose="user_data")
                        file_id = getattr(up, "id", None) or (up.get("id") if isinstance(up, dict) else None)
                        if not file_id:
                            last_err = "file_id missing"
                            time.sleep(1)
                            continue
                        _upsert_file_cache(
                            user_id,
                            rel_path,
                            "openai",
                            file_id=file_id,
                            file_uri=None,
                            state="ACTIVE",
                            last_error=None,
                            size_bytes=size if size is not None else (len(data) if data is not None else None),
                            mtime=mtime,
                            mime_type=mime,
                            last_checked_at=datetime.utcnow()
                        )
                        safe_db_commit()
                        return file_id, None
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(1)
                        continue
                _upsert_file_cache(
                    user_id,
                    rel_path,
                    "openai",
                    state="FAILED",
                    last_error=last_err
                )
                safe_db_commit()
                return None, last_err

            def _grok_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                is_grok_45 = "grok-4.5" in model_key_l
                is_grok_46 = "grok-4.6" in model_key_l
                if is_grok_45 or is_grok_46:
                    # Grok 4.5 / 4.6: reasoning cannot be disabled (defaults to high).
                    # "xhigh" is only supported by grok-4.6; grok-4.5 treats it as "high".
                    if raw in ("low", "medium", "high"):
                        return raw
                    if raw == "xhigh":
                        return "xhigh" if is_grok_46 else "high"
                    return None
                if "grok-4.20-0309-reasoning" in model_key_l:
                    return raw if raw in ("low", "medium", "high") else "high"
                if raw in ("none", "low", "medium", "high", "xhigh"):
                    return "xhigh" if raw == "xhigh" and is_grok_46 else ("high" if raw == "xhigh" else raw)
                lvl = (options.get('thinking_level') or "low").lower()
                return "high" if lvl == "high" else "low"

            def _deepseek_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if model_key_l in {"deepseek-v4.1-flash", "deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}:
                    if raw in ("low", "high", "max"):
                        return raw
                    # DeepSeek maps compatibility values medium/xhigh to high.
                    return "high"
                if raw in ("max", "xhigh"):
                    return "max"
                return "high"

            def _kimi_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if raw in ("low", "high", "max"):
                    return raw
                return "max"

            def _grok_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                if not _auto_notice_enabled("grok_search"):
                    return base_prompt
                notice = _auto_notice_text("grok_search")
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice

            def _openai_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                if not _auto_notice_enabled("openai_search"):
                    return base_prompt
                notice = _auto_notice_text("openai_search")
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice
            
            def get_k(db_val, env_key):
                k = decrypt_val(db_val)
                if k and str(k).strip():
                    return k
                if _admin_env_fallback_enabled(user):
                    return os.getenv(env_key)
                return None

            resolved_auth = _resolve_chat_model_auth(user, model_key)
            gemini_runtime = resolved_auth.get("gemini_runtime") or _resolve_gemini_runtime(user)
            model_api_key_override = _get_model_specific_api_key(user, model_key)

            api_keys = {
                'openai': get_k(user.openai_api_key, 'OPENAI_API_KEY'),
                'gemini': gemini_runtime.get('api_key'),
                'anthropic': get_k(user.anthropic_api_key, 'ANTHROPIC_API_KEY'),
                'xai': get_k(user.xai_api_key, 'XAI_API_KEY'),
                'deepseek': get_k(user.deepseek_api_key, 'DEEPSEEK_API_KEY'),
                'zai': get_k(user.zai_api_key, 'ZAI_API_KEY'),
                'kimi': get_k(user.kimi_api_key, 'MOONSHOT_API_KEY'),
                'mistral': get_k(user.mistral_api_key, 'MISTRAL_API_KEY'),
                'ideogram': get_k(user.ideogram_api_key, 'IDEOGRAM_API_KEY'),
            }

            key = resolved_auth.get("api_key")
            if resolved_auth.get("error_code"):
                pub(
                    "error",
                    resolved_auth.get("error") or "APIキーが設定されていません。",
                    code=resolved_auth["error_code"],
                    model=model_key,
                    provider=resolved_auth.get("provider"),
                )
                return

            g_client = None; o_client = None; x_client = None; c_client = None
            gemini_backend_mode = _normalize_gemini_backend(gemini_runtime.get("backend")) if is_gem else "gemini_api"
            if is_gem:
                try:
                    g_client = _get_gemini_client(
                        api_key=key,
                        backend=gemini_backend_mode,
                        vertex_project=gemini_runtime.get("vertex_project"),
                        vertex_location=gemini_runtime.get("vertex_location"),
                        vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
                    )
                except Exception as e:
                    pub("error", _format_gemini_runtime_error(e, gemini_backend_mode))
                    return
                if not g_client:
                    if gemini_backend_mode == "vertex_ai":
                        pub("error", _gemini_vertex_auth_error_message())
                    else:
                        pub("error", "Gemini client initialization failed. Gemini設定を確認してください。")
                    return
            elif is_claude:
                if not ANTHROPIC_AVAILABLE:
                    pub("error", "Anthropic SDK is not installed on the server.")
                    return
                c_client = Anthropic(api_key=key, http_client=httpx.Client(
                    timeout=httpx.Timeout(600, connect=5),
                    event_hooks={'request': [_observe_usage_request], 'response': [_observe_usage_response]},
                ))
            elif is_grok:
                x_client = _get_xai_client(key)
                o_client = _get_openai_client(key, base_url=f"https://{_XAI_API_HOST}/v1")
            elif is_deepseek:
                o_client = _get_openai_client(key, base_url="https://api.deepseek.com")
            elif is_zai:
                o_client = _get_openai_client(key, base_url="https://api.z.ai/api/paas/v4")
            elif is_kimi:
                o_client = _get_openai_client(key, base_url="https://api.moonshot.ai/v1")
            elif is_mistral_ocr:
                o_client = None
            elif is_ideogram:
                o_client = None
            else: o_client = _get_openai_client(key, base_url=None)

            loaded_files, file_errors = background_load_attachments(
                img_list, options, user_id, is_llm_model, is_mistral_ocr
            )

            if file_errors:
                parts = []
                for e in file_errors[:5]:
                    nm = e.get("name") or "file"
                    rs = e.get("reason") or "error"
                    parts.append(f"{nm}({rs})")
                if len(file_errors) > 5:
                    parts.append(f"...他{len(file_errors) - 5}件")
                pub("error", "添付ファイルの検証に失敗しました: " + " / ".join(parts))
                return

            if img_list and not loaded_files:
                pub("error", "添付ファイルを読み込めませんでした。再アップロードしてから再送してください。")
                return

            has_audio = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('audio/') for fi in loaded_files)
            has_video = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('video/') for fi in loaded_files)
            has_code_exec_unsupported_doc = any(
                fi.get('bytes') and (
                    fi.get('is_pdf')
                    or fi.get('is_docx')
                    or str(fi.get('mime', '')).lower() == 'application/pdf'
                    or str(fi.get('mime', '')).lower() == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
                for fi in loaded_files
            )
            gemini_local_python = False
            if is_gem and (has_audio or has_video or has_code_exec_unsupported_doc) and options.get('enable_python'):
                # Gemini code_execution does not accept audio/video/PDF/DOCX inputs; fall back to local exec.
                gemini_local_python = True
                log_force("Gemini: local python mode for audio/video/document inputs")
                if _auto_notice_enabled("gemini_local_python"):
                    local_py_notice = _auto_notice_text("gemini_local_python")
                    curr_p = options.get('system_prompt') or ""
                    if local_py_notice not in str(curr_p):
                        options['system_prompt'] = f"{local_py_notice}\n\n{curr_p}" if str(curr_p).strip() else local_py_notice
            # Skill catalog: name + short description for tool-capable routes,
            # with the full instructions embedded where tools cannot run. The
            # prompt-bar Skills switch (enable_skills) gates the whole feature.
            if is_llm_model and _skills_enabled and _auto_notice_enabled("skills"):
                try:
                    _skills_for_tools = (
                        Skill.query
                        .filter_by(user_id=user_id)
                        .order_by(Skill.name.asc())
                        .all()
                    )
                    if _skills_for_tools:
                        _skills_tool_enabled = skill_tool_supported(
                            model_key,
                            batch_mode=batch_mode,
                            gemini_local_python=gemini_local_python,
                        )
                        _skill_notice = render_skill_notice(
                            _auto_notice_text("skills"),
                            _skills_for_tools,
                            include_instructions=not _skills_tool_enabled,
                            active_uuid=_skill_active_uuid,
                        )
                        if _skill_notice:
                            _curr_p = options.get('system_prompt') or ""
                            if "Available skills:" not in str(_curr_p):
                                options['system_prompt'] = (
                                    f"{_skill_notice}\n\n{_curr_p}".strip()
                                    if str(_curr_p).strip() else _skill_notice
                                )
                        log_force(
                            f"Skill catalog injected: count={len(_skills_for_tools)} "
                            f"tool={_skills_tool_enabled} enabled={_skills_enabled}"
                        )
                except Exception as _skill_exc:
                    log_force(f"Skill prompt inject failed: {_skill_exc}")
            elif is_llm_model and not _skills_enabled:
                log_force("Skill catalog skipped: disabled by prompt bar")

            if (has_audio and not supports_audio_inputs) or (has_video and not supports_video_inputs):
                pub("error", "This model does not support audio/video inputs. Please remove them and retry.")
                return

            full_res, thought_accumulated, generated_images = "", "", []
            deepseek_tool_context = []
            agentic_image_digests = set()
            agentic_saved_urls = []
            agentic_consumed_urls = []
            sandbox_text_buffer = [""]
            agentic_filename_url_map = {}
            # Sandbox images earlier answers in this thread produced, loaded the
            # first time this answer cites a bare file name (None = not loaded).
            prior_sandbox_names = None
            pending_sandbox_filenames = []
            signature_parts = []

            original_quote_text = quote_text
            if quote_text and isinstance(quote_source, dict) and is_llm_model and _auto_notice_enabled("quote_source"):
                quote_source_notice = _render_quote_source_notice(
                    _auto_notice_text("quote_source"), quote_source.get("role"), quote_source.get("number")
                )
                if quote_source_notice:
                    quote_text = f"{quote_source_notice}\n\n{quote_text}"
            if options.get("coding_mode"):
                coding_target = options.get("coding_target") or {}
                coding_candidates = options.get("coding_candidates") or []
                target_code = str(coding_target.get("code") or "")
                target_language = str(coding_target.get("language") or "text")
                explicitly_selected = (
                    coding_target.get("explicit") is True
                    and len(coding_candidates) == 1
                    and str(coding_candidates[0].get("id") or "") == str(coding_target.get("id") or "")
                )
                if explicitly_selected:
                    coding_context = (
                        f"[Coding Mode Target]\nYou must edit candidate {coding_target.get('id')}.\n"
                        f"Language: {target_language}\nCharacter count: {len(target_code)}\n"
                        f"--- BEGIN TARGET CODE ---\n{target_code}\n--- END TARGET CODE ---"
                    )
                else:
                    candidate_lines = []
                    for candidate in coding_candidates:
                        candidate_code = str(candidate.get("code") or "")
                        preview = next(
                            (line.strip() for line in candidate_code.splitlines() if line.strip()),
                            "",
                        )[:160]
                        candidate_lines.append(
                            f"- id={candidate.get('id')}; source={candidate.get('source')}; "
                            f"language={candidate.get('language')}; chars={len(candidate_code)}; "
                            f"first_line={preview!r}"
                        )
                    coding_context = (
                        "[Coding Mode Candidates]\n"
                        "Choose the candidate that best matches User Message and return its id as target_id. "
                        "Prompt candidates correspond to Markdown code blocks in User Message; history candidates "
                        "correspond to code blocks in the conversation in chronological order.\n"
                        + "\n".join(candidate_lines)
                    )
                quote_text = f"{quote_text}\n\n{coding_context}" if quote_text else coding_context
            final_message_text = message_text
            if quote_text:
                final_message_text = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"

            if is_llm_model:
                image_names = [os.path.basename(str(fi.get('send_name') or fi.get('name') or 'image'))
                               for fi in loaded_files if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')]
                attachment_notice = _build_attachment_name_block(image_names)
                if attachment_notice:
                    final_message_text += f"\n\n{attachment_notice}"
                prompt_context = json.dumps({'prompt_text': final_message_text,
                    'prompt_source': decrypt_val(msg.content) if msg.is_encrypted else msg.content,
                    'prompt_quote': msg.quote_text or ''}, ensure_ascii=False)
                msg.thought_data = encrypt_val(prompt_context) if msg.is_encrypted else prompt_context
                safe_db_commit()

            auto_enable_search = options.get('enable_search')
            auto_enable_url_context = options.get('enable_url_context')
            is_gemini_3 = "gemini-3" in model_key or "gemini-3.1" in model_key or "gemini-4" in model_key
            auto_enable_maps = bool(options.get('enable_maps')) and is_gemini_3
            grok_enable_search = auto_enable_search
            user_auto_search = True
            try:
                user_auto_search = bool(getattr(user, "auto_search_on_links", True))
            except Exception:
                user_auto_search = True
            disable_auto = bool(options.get('disable_auto_search'))
            if batch_mode:
                # Batch requests cannot pause for interactive tool execution or
                # follow-up calls, including automatic URL/search helpers.
                auto_enable_search = False
                auto_enable_url_context = False
                auto_enable_maps = False
                grok_enable_search = False
                disable_auto = True
            # Gemini 3.8 Flash Cyber is a Vertex AI cybersecurity model. Its
            # model contract excludes grounding, code execution, URL context,
            # and function calling, so strip these options even if an older
            # client sends them.
            is_gemini_cyber = model_key_l == "gemini-3.8-flash-cyber"
            if is_gemini_cyber:
                auto_enable_search = False
                auto_enable_url_context = False
                auto_enable_maps = False
                options['enable_python'] = False
                options['enable_file_creation'] = False
                options['enable_mcp'] = False
            if is_mistral_ocr:
                grok_enable_search = False
                auto_enable_search = False
                auto_enable_url_context = False
                auto_enable_maps = False
            if is_grok and not is_mistral_ocr and not grok_enable_search and user_auto_search and not disable_auto:
                try:
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        grok_enable_search = True
                        auto_enable_search = True
                        log_force("Auto-enabled Grok search for URL/X post access")
                except Exception:
                    pass
            if not is_mistral_ocr and not is_grok and not auto_enable_search and user_auto_search and not disable_auto:
                try:
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        auto_enable_search = True
                        log_force("Auto-enabled Web search for URL/X post access")
                except Exception:
                    pass
            if is_gem and not is_mistral_ocr and not auto_enable_url_context and user_auto_search and not disable_auto:
                try:
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text):
                        auto_enable_url_context = True
                        log_force("Auto-enabled URL context for Gemini URL access")
                except Exception:
                    pass

            _ensure_mcp_env = background_mcp_env_builder(
                user_id, job_id, pub, check_stop, options,
                _auto_notice_enabled, _auto_notice_text,
            )

            if is_llm_model and not gemini_local_python:
                try:
                    _ensure_mcp_env()
                except Exception as _mcp_preload_exc:
                    log_force(f"MCP preload failed: {_mcp_preload_exc}")


            # region: mistral-ocr
            # --- 0. Mistral OCR 4 (document-only, no chat history) ---
            if is_mistral_ocr:
                _mistral_stream = background_stream_mistral_ocr(locals())
                if _mistral_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                full_res = _mistral_stream

            # region: gemini-omni
            # --- 1A-0. GEMINI OMNI 1.1 FLASH (Interactions API) ---
            elif model_key_l == "gemini-omni-1.1-flash":
                full_res = background_stream_gemini_omni(locals())

            # region: gemini-video
            # --- 1A. GEMINI VIDEO GENERATION (Veo 3.1 / Omni Flash) ---
            elif is_gemini_video_model_key(model_key_l):
                full_res = background_stream_gemini_video(locals())

            # region: gemini-music
            # --- 1B. GEMINI MUSIC GENERATION (Lyria 3) ---
            elif is_gemini_music_model_key(model_key_l):
                full_res = background_stream_gemini_music(locals())

            # region: gemini-embedding
            # --- 1C. GEMINI EMBEDDING ---
            elif is_gemini_embedding_model_key(model_key_l):
                log_force("Routing: Gemini Embedding Branch")
                full_res = background_stream_gemini_embedding(locals())

            # --- 1. GEMINI & GEMINI IMAGE ---
            elif is_gem:
                _gemini_stream = background_stream_gemini(locals())
                if _gemini_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                full_res, m, thought_accumulated, final_usage_metadata = _gemini_stream

            # region: ideogram
            # --- 1.4 Ideogram Image Generation / Precise Edit ---
            elif is_ideogram:
                full_res = background_stream_ideogram(locals())

            # --- 1.5 Grok Imagine Image Generation ---
            # region: grok-imagine
            elif model_key in (
                "grok-imagine-image-2.0",
                "grok-imagine-image",
                "grok-imagine-image-pro",
                "grok-imagine-image-quality",
            ):
                full_res = background_stream_grok_imagine(locals())

            # region: grok-imagine-video
            # --- 1.6 Grok Imagine Video Generation ---
            elif model_key in ("grok-imagine-video", "grok-imagine-video-1.5", "grok-imagine-video-1.5-lite"):
                full_res = background_stream_grok_imagine_video(locals())

            # region: claude
            # --- 1.5 Anthropic Claude ---
            elif is_claude:
                full_res, thought_accumulated = background_stream_claude(locals())

            # region: grok-native
            # --- 2. xAI Grok (Native SDK) ---
            elif is_grok and x_client and not batch_mode and not options.get('enable_python') and _ensure_mcp_env() is None:
                full_res, thought_accumulated = background_stream_grok_native(locals())

            # region: tts
            # --- 2.5 TTS Branch ---
            elif 'tts' in model_key:
                log_force("Routing: TTS Branch")
                full_res = background_stream_tts(locals())

            # region: gpt-image
            # --- 3. GPT Image Branch ---
            elif 'gpt-image' in model_key:
                full_res = background_stream_gpt_image(locals())

            # region: openai-search
            # --- 3.5 OpenAI Search API (Chat Completions) ---
            elif is_openai_search_model:
                _search_stream = background_stream_openai_search(locals())
                if _search_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                full_res = _search_stream

            elif is_deepseek:
                _deepseek_stream = background_stream_deepseek(locals())
                if _deepseek_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                (
                    full_res, m, _branch_search_used, thought_accumulated,
                    usage, final_openai_usage,
                ) = _deepseek_stream
                search_used = search_used or _branch_search_used

            elif is_kimi or is_zai:
                _kimi_stream = background_stream_kimi_zai(locals())
                if _kimi_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                (
                    full_res, m, _branch_search_used, thought_accumulated,
                    usage, final_openai_usage,
                ) = _kimi_stream
                search_used = search_used or _branch_search_used

            # --- 4. OpenAI Responses API (or Grok Fallback) ---
            else:
                _openai_stream = background_stream_openai(locals())
                if _openai_stream is BACKGROUND_STREAM_EARLY_RETURN:
                    return
                (
                    full_res, m, _branch_search_used, thought_accumulated,
                    usage, final_openai_usage,
                ) = _openai_stream
                search_used = search_used or _branch_search_used

            if background_stream_tail(locals()) is BACKGROUND_STREAM_EARLY_RETURN:
                return

        except Exception as e:
            logger.exception("Worker Error")
            log_force(f"Worker Exception: {e}")
            err_msg = str(e)
            try:
                if is_gem:
                    err_msg = _format_gemini_runtime_error(e, gemini_backend_mode)
            except Exception:
                pass
            if batch_mode:
                try:
                    _persist_batch_failure(err_msg)
                except Exception:
                    logger.exception("Failed to persist Gemini Batch error")
            else:
                pub("error", err_msg)
        finally:
            _MODEL_USAGE_CONTEXT.reset(usage_context_token)
            if locals().get('c_client') is not None:
                try:
                    c_client.close()
                except Exception:
                    log_force('Provider client close failed')
            _record_server_diagnostic(user_id, 'server.worker.end', job=job_id)
            _latency_mark_once(job_id, "worker_done_ms")
            r.delete(f"stop_job:{job_id}")
            try:
                r.delete(f"pending_job:{user_id}:{thread_id}")
            except Exception:
                pass
            try:
                r.delete(f"stream_acc:{job_id}:content")
                r.delete(f"stream_acc:{job_id}:thought")
                r.delete(f"stream_acc:{job_id}:search")
                r.delete(f"stream_acc:{job_id}:status")
                r.delete(f"stream_acc:{job_id}:final")
                r.delete(f"stream_acc:{job_id}:python")
                r.delete(f"stream_acc:{job_id}:coding_diff")
                r.delete(f"stream_acc:{job_id}:mcp")
                r.delete(f"mcp_decision:{job_id}")
            except Exception:
                pass
