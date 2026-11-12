"""xAI Grok native SDK branch of `background_chat_task`.

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns the updated answer text and accumulated thoughts.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_grok_native(local):
    """Run the native Grok branch and return ``(full_res, thought_accumulated)``."""

    _grok_reasoning_effort = local.get('_grok_reasoning_effort')
    _grok_system_prompt = local.get('_grok_system_prompt')
    _load_message_history_images = local.get('_load_message_history_images')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    grok_enable_search = local.get('grok_enable_search')
    grok_reasoning_effort_supported = local.get('grok_reasoning_effort_supported')
    grok_reasoning_supported = local.get('grok_reasoning_supported')
    history = local.get('history')
    job_id = local.get('job_id')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    model_key_l = local.get('model_key_l')
    options = local.get('options')
    pub = local.get('pub')
    thought_accumulated = local.get('thought_accumulated')
    usage_ledger = local.get('usage_ledger')
    x_client = local.get('x_client')

    # --- 2. xAI Grok (Native SDK) ---
    log_force("Routing: Grok Branch (Native SDK)")
    if options.get('enable_thinking') and not grok_reasoning_supported:
        # Grok non-reasoning models should not emit thought events (avoids UI thought box).
        log_force("Grok non-reasoning: skip thought stream")
    search_params = None
    tools = []
    include = []
    if grok_enable_search:
        try:
            tools = [x_web_search(), x_x_search()]
            include = ["verbose_streaming", "inline_citations"]
            log_force("Enabled Grok Search Tools (Web + X)")
        except Exception as e:
            log_force(f"Grok Search Tools Config Error: {e}")
            try:
                search_params = SearchParameters(
                    sources=[web_source(), x_source()],
                    mode="on",
                    return_citations=True
                )
                log_force("Enabled Grok Search (Legacy SearchParameters)")
            except Exception as e2:
                log_force(f"Grok Search Config Error (Legacy): {e2}")

    create_kwargs = {"model": model_key}
    if search_params: create_kwargs["search_parameters"] = search_params
    if tools: create_kwargs["tools"] = tools
    if include: create_kwargs["include"] = include
    forced_grok_reasoning = ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l)
    if (options.get('enable_thinking') or forced_grok_reasoning) and grok_reasoning_effort_supported:
        grok_effort = _grok_reasoning_effort()
        if grok_effort:
            create_kwargs["reasoning_effort"] = grok_effort
    elif (options.get('enable_thinking') or forced_grok_reasoning) and grok_reasoning_supported:
        log_force("Grok reasoning_effort not supported for this model; skipping parameter")
    def _optional_float(option_key, minimum, maximum):
        raw_value = options.get(option_key)
        if raw_value is None or str(raw_value).strip() == "":
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        return max(minimum, min(maximum, value))
    def _optional_int(option_key, minimum=None, maximum=None):
        raw_value = options.get(option_key)
        if raw_value is None or str(raw_value).strip() == "":
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        if minimum is not None: value = max(minimum, value)
        if maximum is not None: value = min(maximum, value)
        return value
    xai_sampling = {
        "temperature": _optional_float("xai_temperature", 0.0, 2.0),
        "top_p": _optional_float("xai_top_p", 0.0, 1.0),
        "frequency_penalty": _optional_float("xai_frequency_penalty", -2.0, 2.0),
        "presence_penalty": _optional_float("xai_presence_penalty", -2.0, 2.0),
        "seed": _optional_int("xai_seed"),
        # xAI's current Chat API calls this max_completion_tokens; the
        # installed native SDK serializes the equivalent max_tokens field.
        "max_tokens": _optional_int("xai_max_completion_tokens", 1),
    }
    for option_key, option_value in xai_sampling.items():
        if option_value is not None:
            create_kwargs[option_key] = option_value
    stop_raw = str(options.get("xai_stop") or "").strip()
    if stop_raw:
        create_kwargs["stop"] = [part.strip() for part in stop_raw.split(",") if part.strip()][:4]
    response_format = str(options.get("xai_response_format") or "").strip()
    if response_format in ("text", "json_object") and response_format != "text":
        create_kwargs["response_format"] = response_format
    tool_choice = str(options.get("xai_tool_choice") or "auto").strip().lower()
    if tools and tool_choice in ("auto", "none", "required"):
        create_kwargs["tool_choice"] = tool_choice
    if tools:
        create_kwargs["parallel_tool_calls"] = bool(options.get("xai_parallel_tool_calls", True))
    if not model_key_l.startswith("grok-4.20"):
        if options.get("xai_logprobs"):
            create_kwargs["logprobs"] = True
        top_logprobs = _optional_int("xai_top_logprobs", 0, 8)
        if top_logprobs is not None and options.get("xai_logprobs"):
            create_kwargs["top_logprobs"] = top_logprobs
    create_kwargs["use_encrypted_content"] = True # Request encrypted reasoning if available
    if options.get('enable_python') and XAI_SDK_AVAILABLE:
        create_kwargs["tools"] = [x_code_execution()]
    if options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
        create_kwargs["conversation_id"] = options.get('prompt_cache_key')
        log_force(f"Grok Prompt Caching conversation_id={options.get('prompt_cache_key')}")

    _mark_provider_request_started()
    chat_session = x_client.chat.create(**create_kwargs)

    grok_sys = _grok_system_prompt(options.get('system_prompt'), grok_enable_search)
    if grok_sys: chat_session.append(x_system(grok_sys))

    history_img_seen = set()
    history_img_bytes = 0
    for m in history:
        if m['role'] in ('user', 'assistant'):
            content_parts = [m['content']]
            if m.get('image_url'):
                try:
                    msg_images, history_img_bytes = _load_message_history_images(
                        m.get('image_url'),
                        seen=history_img_seen,
                        total_bytes=history_img_bytes,
                        include_only_images=True
                    )
                    for msg_img in msg_images:
                        d_uri = f"data:{msg_img['mime']};base64,{base64.b64encode(msg_img['bytes']).decode('utf-8')}"
                        content_parts.append(x_image(d_uri))
                except: pass
            if m['role'] == 'user':
                chat_session.append(x_user(*content_parts))
            else:
                chat_session.append(x_assistant(*content_parts))
        else:
            chat_session.append(x_user(m['content']) if m['role'] == 'user' else x_assistant(m['content']))

    curr_user_content = []
    if final_message_text and str(final_message_text).strip():
        curr_user_content.append(final_message_text)
    current_image_names = []
    for fi in loaded_files:
        if fi.get('text'):
            file_text_block = f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
            if curr_user_content:
                curr_user_content[0] += file_text_block
            else:
                curr_user_content.append(file_text_block.strip())
        elif fi.get('bytes') and fi.get('mime', '').startswith('image/'):
            d_uri = f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"
            curr_user_content.append(x_image(d_uri))
            img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
            current_image_names.append(os.path.basename(str(img_label)))

    chat_session.append(x_user(*curr_user_content))

    _mark_provider_request_started()
    grok_usage_call = usage_ledger.begin(model_key, 'xai')
    stream = chat_session.stream()
    search_reported = False
    last_response = None
    if grok_reasoning_supported:
        # Ensure thought box is created even if Grok doesn't stream reasoning text.
        pub("thought", " ")
    for resp, chunk in stream:
        _latency_mark_once(job_id, "provider_first_chunk_ms")
        last_response = resp
        if getattr(resp, 'usage', None) is not None:
            grok_usage_call.update_usage(resp.usage)
        if check_stop(): break
        tool_calls = getattr(chunk, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                tc_type = getattr(tc, 'type', None)
                tc_fn = getattr(getattr(tc, 'function', None), 'name', None)
                tc_type_str = str(tc_type) if tc_type is not None else ""
                if (tc_fn and "search" in tc_fn.lower()) or ("SEARCH" in tc_type_str):
                    if not search_reported:
                        pub("search_status", "searching")
                        search_reported = True
                    break
        r_content = getattr(chunk, 'reasoning_content', None)
        if r_content:
            thought_accumulated += r_content
            if grok_reasoning_supported:
                pub("thought", r_content)

        # Also log encrypted content presence for debugging
        if getattr(chunk, 'encrypted_content', None):
             log_force("Received encrypted reasoning content")

        c_content = getattr(chunk, 'content', None)
        if c_content:
            full_res += c_content
            pub("content", c_content)
    if search_reported:
        pub("search_status", "done")
    grok_usage_call.finished = not check_stop()
    if last_response and getattr(last_response, 'citations', None):
        citations_text = "\n\n**Sources:**\n"
        inline_citations = getattr(last_response, 'inline_citations', None)
        if inline_citations:
            for c in inline_citations:
                cid = getattr(c, 'id', None)
                web_cit = getattr(c, 'web_citation', None)
                url = None
                title = None
                if web_cit:
                    url = getattr(web_cit, 'url', None)
                    title = getattr(web_cit, 'title', None)
                if url:
                    label = title or url
                    if cid is not None:
                        citations_text += f"- [{cid}] {label} ({url})\n"
                    else:
                        citations_text += f"- {label} ({url})\n"
        else:
            for c in last_response.citations:
                if hasattr(c, 'url'):
                    url = c.url
                    title = getattr(c, 'title', None)
                else:
                    url = str(c)
                    title = None
                label = title or url
                citations_text += f"- {label}\n"
        full_res += citations_text
        pub("content", citations_text)
    return (full_res, thought_accumulated)
