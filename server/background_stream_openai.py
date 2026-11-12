"""background_stream_openai branch of `background_chat_task`.

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` (the enclosing function's nested helpers such as ``pub`` and
``check_stop`` must be passed through it) and returns the values the shared
stream tail still needs.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to background_chat_task when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_openai(local):
    """Run the branch and report the values the stream tail needs."""
    # region: setup and input preparation
    _ensure_mcp_env = local.get('_ensure_mcp_env')
    _execute_skill_tool = local.get('_execute_skill_tool')
    _grok_reasoning_effort = local.get('_grok_reasoning_effort')
    _grok_system_prompt = local.get('_grok_system_prompt')
    _load_message_history_images = local.get('_load_message_history_images')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    _openai_cache_fresh = local.get('_openai_cache_fresh')
    _openai_system_prompt = local.get('_openai_system_prompt')
    _openai_upload_with_retry = local.get('_openai_upload_with_retry')
    _persist_batch_failure = local.get('_persist_batch_failure')
    _skills_tool_enabled = local.get('_skills_tool_enabled')
    _submit_openai_batch = local.get('_submit_openai_batch')
    _submit_xai_batch = local.get('_submit_xai_batch')
    agentic_consumed_urls = local.get('agentic_consumed_urls')
    agentic_filename_url_map = local.get('agentic_filename_url_map')
    agentic_image_digests = local.get('agentic_image_digests')
    agentic_saved_urls = local.get('agentic_saved_urls')
    api_keys = local.get('api_keys')
    auto_enable_search = local.get('auto_enable_search')
    batch_mode = local.get('batch_mode')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    final_openai_usage = local.get('final_openai_usage')
    full_res = local.get('full_res') or ""
    generated_images = local.get('generated_images')
    grok_enable_search = local.get('grok_enable_search')
    grok_reasoning_effort_supported = local.get('grok_reasoning_effort_supported')
    grok_reasoning_supported = local.get('grok_reasoning_supported')
    history = local.get('history')
    is_grok = local.get('is_grok')
    is_openai_batch_model = local.get('is_openai_batch_model')
    is_xai_batch_model = local.get('is_xai_batch_model')
    job_id = local.get('job_id')
    key = local.get('key')
    loaded_files = local.get('loaded_files')
    m = local.get('m')
    model_key = local.get('model_key')
    model_key_l = local.get('model_key_l')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    search_used = local.get('search_used')
    thought_accumulated = local.get('thought_accumulated') or ""
    thread_id = local.get('thread_id')
    usage = local.get('usage')
    user_config = local.get('user_config')
    user_id = local.get('user_id')

    log_force("Routing: Responses API Branch")
    client = o_client
    input_data = []
    sys_prompt = _grok_system_prompt(options.get('system_prompt'), grok_enable_search) if is_grok else _openai_system_prompt(options.get('system_prompt'), auto_enable_search)
    if sys_prompt: input_data.append({"role": "system", "content": sys_prompt})

    history_img_seen = set()
    history_img_bytes = 0
    text_type = "input_text"
    image_type = "input_image"
    # Thinking with images: the model crops, zooms, and rotates with the
    # hosted code interpreter. input_image is vision-only; the container
    # receives files named in container.file_ids (max 16, under 50MB).
    openai_ci_file_ids = []
    openai_ci_keys = set()
    response_python_images = []
    openai_ci_status_sent = False

    def _queue_openai_code_interpreter_image(data, mime, cache_key):
        nonlocal openai_ci_status_sent
        if is_grok or not options.get('enable_python'):
            return
        key = str(cache_key or "")
        if not key or key in openai_ci_keys or len(openai_ci_file_ids) >= 16:
            return
        if not data or len(data) > 50 * 1024 * 1024:
            log_force("OpenAI code interpreter image skipped: size")
            return
        openai_ci_keys.add(key)
        try:
            prepared, ext = _prepare_agentic_image_bytes(data, mime)
        except Exception:
            log_force("OpenAI code interpreter image skipped: unsupported")
            return
        if ext not in ("png", "jpg", "gif"):
            converted = _convert_image_to_png(prepared)
            if not converted:
                log_force("OpenAI code interpreter image skipped: convert")
                return
            prepared, ext = converted, "png"
        if not openai_ci_status_sent:
            openai_ci_status_sent = True
            pub("status", "画像をコード実行環境へ送っています...")
        filename = f"input_{len(openai_ci_file_ids) + 1}.{ext}"
        image_mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
        rel = "openai-ci:" + hashlib.sha256(prepared).hexdigest()
        cache = _get_file_cache(user_id, rel, "openai")
        file_id = cache.file_id if _openai_cache_fresh(cache, len(prepared), None, image_mime) else None
        if not file_id:
            file_id, up_err = _openai_upload_with_retry(
                client, prepared, f".{ext}", rel,
                mime=image_mime, size=len(prepared), upload_name=filename,
            )
            if not file_id:
                log_force(f"OpenAI code interpreter image upload failed: {str(up_err or '')[:120]}")
                return
        if file_id not in openai_ci_file_ids:
            openai_ci_file_ids.append(file_id)

    def _remember_python_image(data, mime):
        if data and len(response_python_images) < 8:
            response_python_images.append({"bytes": data, "mime": mime})

    for m in history:
        if m.get('image_url'):
            try:
                content_parts = [{"type": text_type, "text": m['content']}]
                msg_images, history_img_bytes = _load_message_history_images(
                    m.get('image_url'),
                    seen=history_img_seen,
                    total_bytes=history_img_bytes,
                    include_only_images=True
                )
                for msg_img in msg_images:
                    b64 = base64.b64encode(msg_img['bytes']).decode('utf-8')
                    content_parts.append({"type": image_type, "image_url": f"data:{msg_img['mime']};base64,{b64}"})
                input_data.append({"role": m['role'], "content": content_parts})
            except Exception as e:
                log_force(f"Error processing history image: {e}")
                input_data.append({"role": m['role'], "content": m['content']})
        else:
            input_data.append({"role": m['role'], "content": m['content']})

    curr_content = []
    if final_message_text and str(final_message_text).strip():
        curr_content.append({"type": text_type, "text": final_message_text})
    file_inline_limit = 20 * 1024 * 1024  # 20MiB inline limit for file inputs
    file_attach_errors = []
    current_image_names = []

    for fi in loaded_files:
        if (fi.get('is_pdf') or fi.get('is_docx') or fi.get('is_text')) and fi.get('bytes') and not is_grok:
            attached = False
            try:
                f_bytes = fi['bytes']
                f_name = os.path.basename(fi.get('send_name') or fi.get('name') or ('document.pdf' if fi.get('is_pdf') else 'document.docx' if fi.get('is_docx') else 'document.txt'))
                if len(f_bytes) <= file_inline_limit:
                    b64 = base64.b64encode(f_bytes).decode('utf-8')
                    curr_content.append({"type": "input_file", "file_data": b64, "filename": f_name})
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
                    curr_content.append({"type": "input_file", "file_id": file_id, "filename": f_name})
                attached = True
            except Exception as e:
                log_force(f"OpenAI file attach failed: {e}")
                file_attach_errors.append(f"{f_name}({str(e)[:120]})")
            if attached:
                continue
        if fi.get('text'):
            for part in reversed(curr_content):
                if part.get('type') == text_type:
                    part['text'] += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
                    break
        elif fi.get('bytes') and fi['mime'].startswith('image/'):
            img_bytes = fi['bytes']
            img_mime = fi['mime']
            if is_grok and img_mime not in ('image/jpeg', 'image/png'):
                converted_png = _convert_image_to_png(img_bytes)
                if converted_png:
                    img_bytes = converted_png
                    img_mime = 'image/png'
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            curr_content.append({"type": image_type, "image_url": f"data:{img_mime};base64,{b64}"})
            _remember_python_image(img_bytes, img_mime)
            img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
            _queue_openai_code_interpreter_image(
                img_bytes, img_mime, fi.get('path') or fi.get('name') or img_label,
            )
            current_image_names.append(os.path.basename(str(img_label)))
    if file_attach_errors:
        parts = file_attach_errors[:5]
        if len(file_attach_errors) > 5:
            parts.append(f"...他{len(file_attach_errors)-5}件")
        pub("error", "ファイル添付に失敗しました: " + " / ".join(parts))
        return BACKGROUND_STREAM_EARLY_RETURN

    input_data.append({"role": "user", "content": curr_content})

    # region: Responses API request building
    # OpenAI/xAI Responses API
    has_image_inputs = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files)
    # xAI docs: image understanding requests should avoid server-side storage.
    store_flag = False if (is_grok and has_image_inputs) else True
    kwargs = {
        "model": model_key,
        "input": input_data,
        "stream": True,
        "store": store_flag,
    }
    if options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
        kwargs["prompt_cache_key"] = options.get('prompt_cache_key')
        log_force(f"Responses API Prompt Caching key={options.get('prompt_cache_key')}")

    if is_grok and grok_enable_search:
        kwargs['tools'] = [{"type": "web_search"}, {"type": "x_search"}]
        kwargs.setdefault("include", [])
        if "inline_citations" not in kwargs["include"]:
            kwargs["include"].append("inline_citations")
        log_force("Enabled Web + X Search Tools (Responses API)")
    elif auto_enable_search:
        kwargs['tools'] = [{"type": "web_search"}]
        kwargs.setdefault("include", [])
        if "web_search_call.action.sources" not in kwargs["include"]:
            kwargs["include"].append("web_search_call.action.sources")
        log_force("Enabled Web Search Tool (Responses API)")

    if options.get('enable_python'):
        if 'tools' not in kwargs: kwargs['tools'] = []
        if is_grok:
            kwargs['tools'].append({
                "type": "function",
                "name": "execute_python",
                "description": "Execute Python code for calculations or data analysis. Isolated environment, no internet access. Attached images are /work/input_1.<ext>, input_2.<ext>, and so on.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to run."}
                    },
                    "required": ["code"]
                }
            })
        else:
            # Official python tool. The model uses it inside its reasoning
            # to crop, zoom, and rotate images (thinking with images).
            container = {"type": "auto"}
            if openai_ci_file_ids:
                container["file_ids"] = openai_ci_file_ids
            kwargs['tools'].append({"type": "code_interpreter", "container": container})
            log_force(f"OpenAI code interpreter images={len(openai_ci_file_ids)}")
        kwargs['tools'].append(_build_list_branch_images_tool_schema(flat=True))
        kwargs['tools'].append(_build_import_branch_images_tool_schema(flat=True))

    if options.get('enable_file_creation'):
        if 'tools' not in kwargs: kwargs['tools'] = []
        kwargs['tools'].append(_build_create_file_tool_schema())
        kwargs['tools'].append(_build_edit_file_tool_schema())
    if _skills_tool_enabled:
        if 'tools' not in kwargs: kwargs['tools'] = []
        kwargs['tools'].append(build_skill_tool_openai_responses())

    # MCP外部ツール（Responses API の function tool として追加）
    resp_mcp_env = None
    try:
        resp_mcp_env = _ensure_mcp_env()
    except Exception:
        resp_mcp_env = None
    if resp_mcp_env is not None:
        try:
            if 'tools' not in kwargs: kwargs['tools'] = []
            kwargs['tools'].extend(resp_mcp_env.serialize_openai())
        except Exception as _mcp_e:
            log_force(f"Responses MCP tool attach failed: {_mcp_e}")

    if is_grok and options.get('enable_thinking') and not grok_reasoning_supported:
        pub("thought", "APIの仕様により表示されません")
    is_reasoning_model = (not is_grok) and any(x in model_key.lower() for x in ['o1', 'o3', 'gpt-6', 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'reasoning'])
    req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
    enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")

    def _normalize_reasoning_effort(model_key_l, effort):
        if not effort:
            return effort
        effort = effort.lower().strip()
        # Smaller GPT-5 tiers do not accept "none"; use minimal instead.
        if any(x in model_key_l for x in ("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5-mini", "gpt-5.5-nano")) and effort == "none":
            return "minimal"
        # Astra and GPT-6.1 Sol reject none and minimal; the API default is medium.
        if effort == "none" and any(x in model_key_l for x in ("gpt-6-astra", "gpt-6.1-sol")):
            return "medium"
        return effort
    if is_grok and enable_reasoning and grok_reasoning_effort_supported:
        grok_effort = _grok_reasoning_effort()
        if grok_effort:
            kwargs['reasoning'] = {"effort": grok_effort}
            log_force(f"Grok reasoning config: {kwargs['reasoning']}")
    elif is_grok and enable_reasoning and grok_reasoning_supported:
        log_force("Grok reasoning_effort not supported for this model; skipping reasoning param")
    elif is_reasoning_model and (enable_reasoning or (model_key_l == "gpt-6.1-terra" and req_reasoning_effort == "none")):
        effort = req_reasoning_effort
        if not effort:
            lvl = (options.get('thinking_level') or "medium").lower()
            effort = "low" if lvl == "low" else "high" if lvl == "high" else "medium"
        effort = _normalize_reasoning_effort(model_key_l, effort)
        kwargs['reasoning'] = {"effort": effort}
        kwargs['reasoning']["summary"] = "auto"
        log_force(f"Reasoning config: {kwargs['reasoning']}")

    log_force(f"Responses API Params: {kwargs.keys()}")
    if batch_mode and is_xai_batch_model(model_key):
        _mark_provider_request_started()
        try:
            _submit_xai_batch(api_keys.get('xai') or key, kwargs)
        except Exception as batch_error:
            logger.exception('xAI Batch submission failed')
            _persist_batch_failure(str(batch_error))
        return BACKGROUND_STREAM_EARLY_RETURN
    if batch_mode and is_openai_batch_model:
        _mark_provider_request_started()
        try:
            _submit_openai_batch(client, kwargs)
        except Exception as batch_error:
            logger.exception('OpenAI Batch submission failed')
            _persist_batch_failure(str(batch_error))
        return BACKGROUND_STREAM_EARLY_RETURN
    pub("status", "APIへの送信を開始しています...")
    _mark_provider_request_started()
    stream = client.responses.create(**kwargs)
    search_reported = False
    saw_reasoning_summary_delta = False
    response_id = None
    pending_response_mcp_calls = []
    collected_sources = []
    seen_source_urls = set()
    sources_emitted = False
    final_openai_usage = None

    # region: stream helpers and MCP handling
    def _queue_response_mcp_call(call_id, call_name, call_args):
        """元ストリームを読み切った後に実行するMCP function callを積む。"""
        if not str(call_name or "").startswith("mcp__"):
            return
        key = (str(call_id or ""), str(call_name or ""))
        if any((c["id"], c["name"]) == key for c in pending_response_mcp_calls):
            return
        pending_response_mcp_calls.append({
            "id": call_id,
            "name": call_name,
            "arguments": call_args,
        })

    def _response_item_fields(item):
        if isinstance(item, dict):
            return (
                item.get("type"),
                item.get("call_id") or item.get("id"),
                item.get("name"),
                item.get("arguments"),
            )
        return (
            getattr(item, "type", None),
            getattr(item, "call_id", None) or getattr(item, "id", None),
            getattr(item, "name", None),
            getattr(item, "arguments", None),
        )

    def _consume_response_mcp_followup(followup_stream):
        """MCP結果を渡したResponses APIの続きのストリームを読む。"""
        nonlocal response_id, full_res, thought_accumulated
        next_calls = []

        def _queue_next_call(call_id, call_name, call_args):
            if not str(call_name or "").startswith("mcp__"):
                return
            key = (str(call_id or ""), str(call_name or ""))
            if any((c["id"], c["name"]) == key for c in next_calls):
                return
            next_calls.append({
                "id": call_id,
                "name": call_name,
                "arguments": call_args,
            })

        for followup_chunk in followup_stream:
            _latency_mark_once(job_id, "provider_first_chunk_ms")
            if check_stop():
                break
            if isinstance(followup_chunk, dict):
                followup_type = followup_chunk.get("type")
                followup_response = followup_chunk.get("response")
                followup_item = followup_chunk.get("item")
                followup_delta = followup_chunk.get("delta")
            else:
                followup_type = getattr(followup_chunk, "type", None)
                followup_response = getattr(followup_chunk, "response", None)
                followup_item = getattr(followup_chunk, "item", None)
                followup_delta = getattr(followup_chunk, "delta", None)

            if followup_type == "response.created":
                response_id = (
                    (followup_response.get("id") if isinstance(followup_response, dict) else getattr(followup_response, "id", None))
                    or response_id
                )
            elif followup_type == "response.output_text.delta" and followup_delta:
                full_res += followup_delta
                pub("content", followup_delta)
            elif followup_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta") and followup_delta:
                thought_accumulated += followup_delta
                pub("thought", followup_delta)
            elif followup_type == "response.output_item.done":
                item_type, item_call_id, item_name, item_args = _response_item_fields(followup_item)
                if item_type == "function_call":
                    _queue_next_call(item_call_id, item_name, item_args)
            elif followup_type == "response.completed":
                response_id = (
                    (followup_response.get("id") if isinstance(followup_response, dict) else getattr(followup_response, "id", None))
                    or response_id
                )
                output_items = (
                    followup_response.get("output")
                    if isinstance(followup_response, dict)
                    else getattr(followup_response, "output", None)
                )
                for output_item in output_items or []:
                    item_type, item_call_id, item_name, item_args = _response_item_fields(output_item)
                    if item_type == "function_call":
                        _queue_next_call(item_call_id, item_name, item_args)
        return next_calls

    def _execute_response_mcp_call(call):
        """MCP callを実行し、Responses APIへ返すfunction_call_outputを作る。"""
        nonlocal full_res
        call_id = call.get("id")
        call_name = call.get("name")
        raw_args = call.get("arguments")
        try:
            args_json = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            if not isinstance(args_json, dict):
                raise ValueError("arguments must be a JSON object")
            if resp_mcp_env is None:
                result = f"Error: MCP tools are not available: {call_name}"
            else:
                mcp_text, mcp_out = resp_mcp_env.execute(call_name, args_json)
                result = mcp_text
                if mcp_out.get("ok"):
                    _md = f"\n\n> 🔧 **MCPツール実行:** `{call_name}` を実行しました。\n"
                    full_res += _md
                    pub("content", _md)
                elif mcp_out.get("rejected"):
                    _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{call_name}`\n"
                    full_res += _md
                    pub("content", _md)
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(result or "Tool executed."),
            }
        except Exception as exc:
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"Error executing MCP tool: {exc}",
            }

    def _add_source(title, url):
        if not url or url in seen_source_urls:
            return
        seen_source_urls.add(url)
        collected_sources.append((title or url, url))

    def _collect_sources_from_annotations(ann_list):
        for ann in ann_list or []:
            if isinstance(ann, dict):
                a_type = ann.get('type')
                a_url = ann.get('url') or ann.get('source') or ann.get('link')
                a_title = ann.get('title') or a_url
            else:
                a_type = getattr(ann, 'type', None)
                a_url = getattr(ann, 'url', None)
                a_title = getattr(ann, 'title', None) or a_url
            if a_url and (a_type is None or "citation" in str(a_type).lower() or "annotation" in str(a_type).lower()):
                _add_source(a_title, a_url)

    def _collect_sources_from_web_search_call(item):
        action = item.get('action') if isinstance(item, dict) else getattr(item, 'action', None)
        sources = None
        if isinstance(action, dict):
            sources = action.get('sources')
        else:
            sources = getattr(action, 'sources', None)
        for src in sources or []:
            if isinstance(src, dict):
                _add_source(src.get('title') or src.get('name'), src.get('url'))
            else:
                _add_source(getattr(src, 'title', None) or getattr(src, 'name', None), getattr(src, 'url', None))

    def _emit_sources_once():
        nonlocal sources_emitted
        if sources_emitted or not collected_sources:
            return
        sources_emitted = True
        sources_text = "\n\n**Sources:**\n"
        for title, url in collected_sources:
            sources_text += f"- [{title}]({url})\n"
        full_res_add = sources_text
        pub("content", sources_text)
        return full_res_add

    # region: code interpreter outputs
    def _append_agentic_text(text):
        nonlocal full_res
        full_res += text

    def _append_agentic_content(text):
        nonlocal full_res
        full_res += text
        pub("content", text)

    code_interpreter = OpenAICodeInterpreterOutputs(
        client,
        pub,
        user_id,
        user_config,
        agentic_image_digests,
        agentic_saved_urls,
        agentic_filename_url_map,
        _append_agentic_content,
        _append_agentic_text,
        lambda: full_res,
    )
    _consume_container_citation = code_interpreter.consume_container_citation
    _publish_local_python_image = code_interpreter.publish_local_python_image
    _publish_code_interpreter_call = code_interpreter.publish_code_interpreter_call

    # region: stream consumption
    ci_sandbox_buf = [""]
    for chunk in stream:
        _latency_mark_once(job_id, "provider_first_chunk_ms")
        if check_stop(): break
        if isinstance(chunk, dict):
            event_type = chunk.get('type')
            usage = chunk.get('usage')
        else:
            event_type = getattr(chunk, 'type', None)
            usage = getattr(chunk, 'usage', None)

        if usage:
            final_openai_usage = usage

        if response_id is None:
            if isinstance(chunk, dict):
                response_id = chunk.get('response_id') or response_id
            else:
                response_id = getattr(chunk, 'response_id', None) or response_id

        if event_type == "response.created":
            resp = chunk.get('response') if isinstance(chunk, dict) else getattr(chunk, 'response', None)
            if isinstance(resp, dict):
                response_id = resp.get('id') or response_id
            else:
                response_id = getattr(resp, 'id', None) or response_id
            continue

        if event_type in ("response.web_search_call.in_progress", "response.web_search_call.searching"):
            if not search_reported:
                pub("search_status", "searching")
                search_reported = True
        elif event_type == "response.web_search_call.completed":
            if search_reported:
                pub("search_status", "done")
                search_reported = False
        elif event_type == "response.output_text.delta":
            text_delta = chunk.get('delta') if isinstance(chunk, dict) else getattr(chunk, 'delta', None)
            if text_delta and not is_grok and options.get('enable_python'):
                text_delta = _rewrite_streamed_sandbox_refs(
                    text_delta, ci_sandbox_buf, agentic_saved_urls, agentic_consumed_urls, agentic_filename_url_map,
                )
            if text_delta:
                if search_reported:
                    pub("search_status", "done")
                    search_reported = False
                full_res += text_delta
                pub("content", text_delta)
        elif event_type == "response.output_text.annotation.added":
            ann = chunk.get('annotation') if isinstance(chunk, dict) else getattr(chunk, 'annotation', None)
            if ann and not _consume_container_citation(ann):
                _collect_sources_from_annotations([ann])
        elif event_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            reasoning_delta = chunk.get('delta') if isinstance(chunk, dict) else getattr(chunk, 'delta', None)
            if reasoning_delta:
                if event_type == "response.reasoning_summary_text.delta":
                    saw_reasoning_summary_delta = True
                thought_accumulated += reasoning_delta
                pub("thought", reasoning_delta)
        elif event_type in ("response.reasoning_text.done", "response.reasoning_summary_text.done"):
            reasoning_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)
            if reasoning_text:
                if event_type == "response.reasoning_summary_text.done":
                    saw_reasoning_summary_delta = True
                thought_accumulated += reasoning_text
                pub("thought", reasoning_text)
        elif event_type in ("response.reasoning_summary_part.added", "response.reasoning_summary_part.done"):
            part = chunk.get('part') if isinstance(chunk, dict) else getattr(chunk, 'part', None)
            if isinstance(part, dict):
                part_type = part.get('type')
                part_text = part.get('text')
            else:
                part_type = getattr(part, 'type', None) if part else None
                part_text = getattr(part, 'text', None) if part else None
            if part_type == "summary_text" and part_text:
                if not saw_reasoning_summary_delta:
                    thought_accumulated += part_text
                    pub("thought", part_text)
        elif event_type in ("response.content_part.added", "response.content_part.done"):
            part = chunk.get('part') if isinstance(chunk, dict) else getattr(chunk, 'part', None)
            if isinstance(part, dict):
                part_type = part.get('type')
                part_text = part.get('text')
            else:
                part_type = getattr(part, 'type', None) if part else None
                part_text = getattr(part, 'text', None) if part else None
            if part_type in ("summary_text", "reasoning_text") and part_text:
                thought_accumulated += part_text
                pub("thought", part_text)
        elif event_type == "response.output_item.added":
            item = chunk.get('item') if isinstance(chunk, dict) else getattr(chunk, 'item', None)
            if item:
                if isinstance(item, dict):
                    i_type = item.get('type')
                    i_name = item.get('name')
                else:
                    i_type = getattr(item, 'type', None)
                    i_name = getattr(item, 'name', None)

                if i_type in ("function_call", "tool_call") or (i_name and "search" in i_name.lower()):
                    if not search_reported:
                        pub("search_status", "searching")
                        search_reported = True
                if i_type == "reasoning":
                    summary_parts = item.get('summary') if isinstance(item, dict) else getattr(item, 'summary', None)
                    if summary_parts:
                        for part in summary_parts:
                            if isinstance(part, dict):
                                p_type = part.get('type')
                                p_text = part.get('text')
                            else:
                                p_type = getattr(part, 'type', None)
                                p_text = getattr(part, 'text', None)
                            if p_type == "summary_text" and p_text:
                                saw_reasoning_summary_delta = True
                                thought_accumulated += p_text
                                pub("thought", p_text)

        elif event_type == "response.output_item.done":
            item = chunk.get('item') if isinstance(chunk, dict) else getattr(chunk, 'item', None)
            if isinstance(item, dict):
                item_type = item.get('type')
                summary_parts = item.get('summary')
                tool_call_id = item.get('call_id') or item.get('id')
                call_name = item.get('name')
                call_args = item.get('arguments')
            else:
                item_type = getattr(item, 'type', None)
                summary_parts = getattr(item, 'summary', None)
                tool_call_id = getattr(item, 'call_id', None) or getattr(item, 'id', None)
                call_name = getattr(item, 'name', None)
                call_args = getattr(item, 'arguments', None)

            # MCPの継続リクエストは、元のResponsesストリームを読み切って
            # response.completed を受け取ってから送る。ストリーム途中で
            # responses.create を再入すると、ツール後の回答が欠落する。
            if item_type == "code_interpreter_call" and not is_grok:
                _publish_code_interpreter_call(item)
                continue

            if item_type == "function_call" and str(call_name or "").startswith("mcp__"):
                _queue_response_mcp_call(tool_call_id, call_name, call_args)
                continue

            if item_type == "function_call" and call_name in (SKILL_TOOL_NAME, "execute_python", "create_file", "edit_file", "list_branch_images", "import_branch_images"):
                try:
                    def _handle_direct_tool_call(name, args_str, call_id):
                        nonlocal full_res, generated_images
                        args_json = json.loads(args_str or "{}") if args_str else {}
                        if name == "create_file":
                            create_result = _execute_create_file_tool(
                                user_id, args_json, user_config.get('enable_e2ee')
                            )
                            res_text = _create_file_tool_result_text(create_result)
                            if create_result.get("ok"):
                                created_file_rel = f"{user_id}/{create_result['filename']}"
                                if created_file_rel not in generated_images:
                                    generated_images.append(created_file_rel)
                                file_link_md = (
                                    f"\n📄 **ファイルを作成しました:** [{create_result.get('display_name')}]({create_result.get('url')})\n"
                                )
                                full_res += file_link_md
                                pub("content", file_link_md)
                            return res_text
                        elif name == "edit_file":
                            edit_result = _execute_edit_file_tool(
                                user_id,
                                args_json,
                                user_config.get('enable_e2ee'),
                                loaded_files=loaded_files,
                                history=history,
                                thread_id=thread_id,
                            )
                            res_text = _create_file_tool_result_text(edit_result, "edit_file", "編集")
                            if edit_result.get("ok"):
                                edited_file_rel = f"{user_id}/{edit_result['filename']}"
                                if edited_file_rel not in generated_images:
                                    generated_images.append(edited_file_rel)
                                file_link_md = (
                                    f"\n📄 **ファイルを編集しました:** [{edit_result.get('display_name')}]({edit_result.get('url')})\n"
                                )
                                full_res += file_link_md
                                pub("content", file_link_md)
                            return res_text
                        elif name == SKILL_TOOL_NAME:
                            return _execute_skill_tool(str(args_json.get("name") or ""))
                        elif name == "list_branch_images":
                            imported_keys = set(openai_ci_keys)
                            return _execute_list_branch_images(
                                history=history,
                                loaded_files=loaded_files,
                                user_id=user_id,
                                imported_keys=imported_keys,
                            )
                        elif name == "import_branch_images":
                            def _on_import_openai(img):
                                img_bytes = img.get('bytes') or b''
                                img_mime = img.get('mime')
                                img_key = img.get('ref') or img.get('name') or img.get('image_id')
                                _remember_python_image(img_bytes, img_mime)
                                _queue_openai_code_interpreter_image(img_bytes, img_mime, img_key)
                                ext = _guess_image_extension(img_mime, img.get('name'))
                                return f"input_{len(openai_ci_file_ids)}.{ext}"

                            imported_keys = set(openai_ci_keys)
                            return _execute_import_branch_images(
                                args_json.get("image_ids", []),
                                history=history,
                                loaded_files=loaded_files,
                                user_id=user_id,
                                imported_keys=imported_keys,
                                on_import_callback=_on_import_openai,
                            )
                        elif name == "execute_python":
                            code = args_json.get('code', '')
                            if code:
                                pub("content", f"\n```python\n{code}\n```\n")
                                res_text = safe_execute_python(
                                    code,
                                    on_image=_publish_local_python_image,
                                    input_images=response_python_images,
                                )
                                pub("content", f"\n**Output:**\n```\n{res_text}\n```\n")
                                full_res += f"\n```python\n{code}\n```\n\n**Output:**\n```\n{res_text}\n```\n"
                                full_res += f"\n```pyexec\n{json.dumps({'code': code, 'output': res_text})}\n```\n"
                                pub("python", {"id": call_id or f"py_{int(time.time()*1000)}_{os.urandom(3).hex()}", "code": code, "output": res_text})
                                return res_text
                            return "Error: No code provided."
                        return f"Error: Unknown tool {name}"

                    cur_tool_id = tool_call_id
                    cur_call_name = call_name
                    cur_call_args = call_args

                    for _t_round in range(8):
                        result = _handle_direct_tool_call(cur_call_name, cur_call_args, cur_tool_id)
                        if not (response_id and cur_tool_id):
                            break
                        _mark_provider_request_started()
                        followup_kwargs = {
                            "model": model_key,
                            "previous_response_id": response_id,
                            "input": [{
                                "type": "function_call_output",
                                "call_id": cur_tool_id,
                                "output": result
                            }],
                            "stream": True,
                        }
                        if (options.get('enable_python') and not is_grok) or _skills_tool_enabled:
                            followup_tools = []
                            for t in kwargs.get('tools', []):
                                t_copy = dict(t)
                                if t_copy.get('type') == 'code_interpreter':
                                    container_copy = dict(t_copy.get('container', {}))
                                    if openai_ci_file_ids:
                                        container_copy["file_ids"] = list(openai_ci_file_ids)
                                    t_copy['container'] = container_copy
                                followup_tools.append(t_copy)
                            followup_kwargs["tools"] = followup_tools

                        tool_stream = client.responses.create(**followup_kwargs)
                        next_call = None
                        for tchunk in tool_stream:
                            _latency_mark_once(job_id, "provider_first_chunk_ms")
                            if check_stop(): break
                            t_event = tchunk.get('type') if isinstance(tchunk, dict) else getattr(tchunk, 'type', None)
                            if t_event == "response.created":
                                t_resp = tchunk.get('response') if isinstance(tchunk, dict) else getattr(tchunk, 'response', None)
                                if t_resp:
                                    response_id = (t_resp.get('id') if isinstance(t_resp, dict) else getattr(t_resp, 'id', None)) or response_id
                            elif t_event == "response.output_text.delta":
                                t_delta = tchunk.get('delta') if isinstance(tchunk, dict) else getattr(tchunk, 'delta', None)
                                if t_delta and not is_grok and options.get('enable_python'):
                                    t_delta = _rewrite_streamed_sandbox_refs(
                                        t_delta, ci_sandbox_buf, agentic_saved_urls, agentic_consumed_urls, agentic_filename_url_map,
                                    )
                                if t_delta:
                                    full_res += t_delta
                                    pub("content", t_delta)
                            elif t_event in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                                t_reason = tchunk.get('delta') if isinstance(tchunk, dict) else getattr(tchunk, 'delta', None)
                                if t_reason:
                                    thought_accumulated += t_reason
                                    pub("thought", t_reason)
                            elif t_event == "response.output_item.done":
                                t_item = tchunk.get('item') if isinstance(tchunk, dict) else getattr(tchunk, 'item', None)
                                t_type, t_cid, t_cname, t_cargs = _response_item_fields(t_item)
                                if t_type == "code_interpreter_call" and not is_grok:
                                    _publish_code_interpreter_call(t_item)
                                elif t_type == "function_call":
                                    if str(t_cname or "").startswith("mcp__"):
                                        _queue_response_mcp_call(t_cid, t_cname, t_cargs)
                                    elif t_cname in (SKILL_TOOL_NAME, "execute_python", "create_file", "edit_file", "list_branch_images", "import_branch_images"):
                                        next_call = (t_cid, t_cname, t_cargs)
                            elif t_event == "response.completed":
                                t_resp = tchunk.get('response') if isinstance(tchunk, dict) else getattr(tchunk, 'response', None)
                                if t_resp:
                                    response_id = (t_resp.get('id') if isinstance(t_resp, dict) else getattr(t_resp, 'id', None)) or response_id
                                    t_usage = t_resp.get('usage') if isinstance(t_resp, dict) else getattr(t_resp, 'usage', None)
                                    if t_usage:
                                        final_openai_usage = t_usage
                                    t_outputs = t_resp.get('output') if isinstance(t_resp, dict) else getattr(t_resp, 'output', None)
                                    for out_item in t_outputs or []:
                                        out_type, out_cid, out_cname, out_cargs = _response_item_fields(out_item)
                                        if out_type == "code_interpreter_call" and not is_grok:
                                            _publish_code_interpreter_call(out_item)
                                        elif out_type == "function_call":
                                            if str(out_cname or "").startswith("mcp__"):
                                                _queue_response_mcp_call(out_cid, out_cname, out_cargs)
                                            elif out_cname in (SKILL_TOOL_NAME, "execute_python", "create_file", "edit_file", "list_branch_images", "import_branch_images") and not next_call:
                                                next_call = (out_cid, out_cname, out_cargs)
                        if next_call:
                            cur_tool_id, cur_call_name, cur_call_args = next_call
                        else:
                            break
                except Exception as e:
                    pub("error", f"Tool Error: {e}")

            if item_type == "reasoning" and summary_parts:
                for part in summary_parts:
                    if isinstance(part, dict):
                        part_type = part.get('type')
                        part_text = part.get('text')
                    else:
                        part_type = getattr(part, 'type', None)
                        part_text = getattr(part, 'text', None)
                    if part_type == "summary_text" and part_text:
                        saw_reasoning_summary_delta = True
                        thought_accumulated += part_text
                        pub("thought", part_text)
            if item_type == "reasoning":
                content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                if content_parts:
                    for part in content_parts:
                        if isinstance(part, dict):
                            p_type = part.get('type')
                            p_text = part.get('text')
                        else:
                            p_type = getattr(part, 'type', None)
                            p_text = getattr(part, 'text', None)
                        if p_type == "reasoning_text" and p_text:
                            thought_accumulated += p_text
                            pub("thought", p_text)
        else:
            if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
                if search_reported:
                    pub("search_status", "done")
                    search_reported = False
                full_res += chunk.output_text_delta
                pub("content", chunk.output_text_delta)

            if hasattr(chunk, 'citations') and chunk.citations:
                citations_text = "\n\n**Sources:**\n"
                for c in chunk.citations:
                    title = getattr(c, 'title', 'Source')
                    url = getattr(c, 'url', '#')
                    citations_text += f"- [{title}]({url})\n"
                full_res += citations_text
                pub("content", citations_text)

            reasoning_delta = getattr(chunk, 'output_reasoning_text_delta', None)
            if reasoning_delta:
                thought_accumulated += reasoning_delta
                pub("thought", reasoning_delta)
        if event_type == "response.completed":
            resp = chunk.get('response') if isinstance(chunk, dict) else getattr(chunk, 'response', None)
            if isinstance(resp, dict):
                response_id = resp.get('id') or response_id
                resp_usage = resp.get('usage')
                output_items = resp.get('output')
            else:
                response_id = getattr(resp, 'id', None) or response_id
                resp_usage = getattr(resp, 'usage', None) if resp else None
                output_items = getattr(resp, 'output', None) if resp else None
            if resp_usage:
                final_openai_usage = resp_usage
            if output_items:
                for item in output_items:
                    item_type, item_call_id, item_name, item_args = _response_item_fields(item)
                    content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                    if item_type == "code_interpreter_call" and not is_grok:
                        _publish_code_interpreter_call(item)
                    if item_type == "function_call":
                        _queue_response_mcp_call(item_call_id, item_name, item_args)
                    if item_type == "web_search_call":
                        _collect_sources_from_web_search_call(item)
                    if content_parts:
                        for part in content_parts:
                            if isinstance(part, dict):
                                p_type = part.get('type')
                                p_anns = part.get('annotations')
                            else:
                                p_type = getattr(part, 'type', None)
                                p_anns = getattr(part, 'annotations', None)
                            if p_type in ("output_text", "text") and p_anns:
                                for ann in p_anns:
                                    if not _consume_container_citation(ann):
                                        _collect_sources_from_annotations([ann])
            if output_items and not saw_reasoning_summary_delta:
                for item in output_items:
                    if isinstance(item, dict):
                        item_type = item.get('type')
                        summary_parts = item.get('summary')
                    else:
                        item_type = getattr(item, 'type', None)
                        summary_parts = getattr(item, 'summary', None)
                    if item_type == "reasoning":
                        if summary_parts:
                            for part in summary_parts:
                                if isinstance(part, dict):
                                    text = part.get('text')
                                else:
                                    text = getattr(part, 'text', None)
                                if text:
                                    thought_accumulated += text
                                    pub("thought", text)
                        content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                        if content_parts:
                            for part in content_parts:
                                if isinstance(part, dict):
                                    p_type = part.get('type')
                                    p_text = part.get('text')
                                else:
                                    p_type = getattr(part, 'type', None)
                                    p_text = getattr(part, 'text', None)
                                if p_type == "reasoning_text" and p_text:
                                    thought_accumulated += p_text
                                    pub("thought", p_text)
            if collected_sources:
                appended = _emit_sources_once()
                if appended:
                    full_res += appended

    # MCP function callは元のストリーム完了後に結果を返し、回答生成を
    # 必要な回数だけ継続する。1回だけの継続では、ツール結果を受けた
    # モデルが別のMCPツールを続けて呼ぶケースで再び無言終了する。
    for _mcp_round in range(8):
        if not pending_response_mcp_calls:
            break
        if not response_id:
            pub("error", "MCPツールの実行後にResponses APIの応答IDを取得できませんでした。")
            break
        response_tool_outputs = [
            _execute_response_mcp_call(call)
            for call in pending_response_mcp_calls
        ]
        pending_response_mcp_calls = []
        try:
            _mark_provider_request_started()
            followup_stream = client.responses.create(
                model=model_key,
                previous_response_id=response_id,
                input=response_tool_outputs,
                stream=True,
                store=store_flag,
            )
            pending_response_mcp_calls = _consume_response_mcp_followup(followup_stream)
        except Exception as exc:
            pub("error", f"MCPツール実行後の回答生成に失敗しました: {exc}")
            break
    else:
        if pending_response_mcp_calls:
            pub("error", "MCPツール呼び出しが上限回数に達しました。")

    # Fallback: retrieve full response if no reasoning summary surfaced in stream
    if enable_reasoning and not thought_accumulated and response_id:
        try:
            _mark_provider_request_started()
            resp_full = client.responses.retrieve(response_id)
            resp_usage = getattr(resp_full, 'usage', None)
            if resp_usage:
                final_openai_usage = resp_usage
            output_items = getattr(resp_full, 'output', None)
            if output_items:
                for item in output_items:
                    if isinstance(item, dict):
                        item_type = item.get('type')
                        summary_parts = item.get('summary')
                        content_parts = item.get('content')
                    else:
                        item_type = getattr(item, 'type', None)
                        summary_parts = getattr(item, 'summary', None)
                        content_parts = getattr(item, 'content', None)
                    if item_type == "reasoning":
                        if summary_parts:
                            for part in summary_parts:
                                if isinstance(part, dict):
                                    text = part.get('text')
                                else:
                                    text = getattr(part, 'text', None)
                                if text:
                                    thought_accumulated += text
                                    pub("thought", text)
                        if content_parts:
                            for part in content_parts:
                                if isinstance(part, dict):
                                    p_type = part.get('type')
                                    p_text = part.get('text')
                                else:
                                    p_type = getattr(part, 'type', None)
                                    p_text = getattr(part, 'text', None)
                                if p_type == "reasoning_text" and p_text:
                                    thought_accumulated += p_text
                                    pub("thought", p_text)
                    if item_type == "code_interpreter_call" and not is_grok:
                        _publish_code_interpreter_call(item)
                    if item_type == "web_search_call":
                        _collect_sources_from_web_search_call(item)
                    if content_parts:
                        for part in content_parts:
                            if isinstance(part, dict):
                                p_type = part.get('type')
                                p_anns = part.get('annotations')
                            else:
                                p_type = getattr(part, 'type', None)
                                p_anns = getattr(part, 'annotations', None)
                            if p_type in ("output_text", "text") and p_anns:
                                for ann in p_anns:
                                    if not _consume_container_citation(ann):
                                        _collect_sources_from_annotations([ann])
                if collected_sources and not sources_emitted:
                    appended = _emit_sources_once()
                    if appended:
                        full_res += appended
        except Exception as e:
            log_force(f"Reasoning retrieve fallback failed: {e}")
    elif enable_reasoning and not thought_accumulated:
        log_force("Reasoning summary missing after stream and retrieve fallback.")
    if not is_grok and options.get('enable_python') and ci_sandbox_buf[0]:
        full_res += ci_sandbox_buf[0]
        pub("content", ci_sandbox_buf[0])
        ci_sandbox_buf[0] = ""

    return (full_res, m, search_used, thought_accumulated, usage, final_openai_usage)
