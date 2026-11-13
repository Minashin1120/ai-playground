# -----------------------------------------------------------
# API Routes
# -----------------------------------------------------------
# ストリーム配信（/chat_stream と SSE 購読・再生の補助）は server/routes_stream.py にある。


# region: Thread prompt cache helpers
def _thread_prompt_cache_key(thread):
    """Provider cache key for this thread. A nonce is added after the user clears the cache."""
    cache_id = thread.public_id or str(thread.id)
    nonce = (getattr(thread, "prompt_cache_nonce", None) or "").strip()
    if nonce:
        return f"thread-{cache_id}-{nonce}"
    return f"thread-{cache_id}"


def _forget_prompt_cache(thread):
    """Drop the saved PromptCache lock and abandon the provider cache key."""
    thread.enable_prompt_caching = False
    thread.prompt_cache_provider = None
    thread.prompt_cache_nonce = secrets.token_hex(8)


# region: /api/token_estimate
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

    provider_estimate = _provider_draft_estimate(current_user, model_key, message_for_count, norm_image_urls,
                                                message_text=message_text, quote_text=quote_text,
                                                include_details=data.get('include_details') is True,
                                                gemini_free_tier=data.get('gemini_free_tier'))
    if provider_estimate is not None:
        return jsonify(provider_estimate)

    prompt_tokens = count_tokens(message_for_count or "", model_key)
    file_tokens = 0
    files_counted = 0
    files_non_text = 0
    files_missing = 0
    files_error = 0
    detail_files = []

    for rel_path in norm_image_urls:
        est = _estimate_attachment_prompt_tokens(rel_path, model_key=model_key)
        tok = int(est.get("tokens") or 0)
        file_tokens += tok
        reason = est.get("reason")
        if data.get('include_details') is True:
            detail_files.append({'reference': rel_path, 'name': os.path.basename(rel_path),
                'tokens': tok if est.get('countable') else None, 'method': 'approximate', 'reason': reason})
        if est.get("countable"):
            files_counted += 1
        elif reason == "missing":
            files_missing += 1
        elif reason in ("non_text", "no_text"):
            files_non_text += 1
        else:
            files_error += 1

    result = {
        'countable': True,
        'estimate_method': 'approximate',
        'estimate_scope': 'draft',
        'tokens_total': prompt_tokens + file_tokens,
        'tokens_prompt': prompt_tokens,
        'tokens_files': file_tokens,
        'files_total': len(norm_image_urls),
        'files_counted': files_counted,
        'files_non_text': files_non_text,
        'files_missing': files_missing,
        'files_error': files_error
    }
    if data.get('include_details') is True:
        message_tokens = count_tokens(message_text, model_key)
        quote_tokens = max(0, prompt_tokens - message_tokens) if quote_text else 0
        result['breakdown'] = _draft_breakdown_result(result, message_tokens, quote_tokens, detail_files)
    return jsonify(result)


# region: /chat_stream_resume
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
        with _chat_stream_subscription(job_id) as (pubsub, start_time):
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
            refresh_count = 0
            for message in _chat_stream_messages(pubsub, start_time):
                if message is None:
                    yield "\n"
                    continue
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
    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

# region: /api/stop_chat
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

# region: /api/temporary_chat/heartbeat
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

# region: /api/generate_title
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
            _get_model_specific_api_key(current_user, requested_model or "deepseek-v4.1-flash")
            or _get_model_specific_api_key(current_user, "deepseek-v4.1-flash")
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
