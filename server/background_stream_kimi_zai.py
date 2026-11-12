"""background_stream_kimi_zai branch of `background_chat_task` (Chat Completions streaming).

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


def background_stream_kimi_zai(local):
    """Run the branch and report the values the stream tail needs."""
    _auto_notice_text = local.get('_auto_notice_text')
    _execute_skill_tool = local.get('_execute_skill_tool')
    _kimi_reasoning_effort = local.get('_kimi_reasoning_effort')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    _persist_batch_failure = local.get('_persist_batch_failure')
    _refresh_pending_job = local.get('_refresh_pending_job')
    _skills_tool_enabled = local.get('_skills_tool_enabled')
    _submit_zai_batch = local.get('_submit_zai_batch')
    api_keys = local.get('api_keys')
    batch_mode = local.get('batch_mode')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    final_openai_usage = local.get('final_openai_usage')
    full_res = local.get('full_res') or ""
    history = local.get('history')
    is_kimi = local.get('is_kimi')
    is_zai = local.get('is_zai')
    job_id = local.get('job_id')
    loaded_files = local.get('loaded_files')
    m = local.get('m')
    model_key = local.get('model_key')
    model_key_l = local.get('model_key_l')
    o_client = local.get('o_client')
    options = local.get('options')
    pub = local.get('pub')
    req_reasoning_effort = local.get('req_reasoning_effort')
    search_used = local.get('search_used')
    thought_accumulated = local.get('thought_accumulated') or ""
    usage = local.get('usage')

    log_force(f"Routing: {'Z.AI GLM' if is_zai else 'Kimi K3'} Branch (Chat Completions)")
    try:
        if check_stop():
            return BACKGROUND_STREAM_EARLY_RETURN
        client = o_client
        messages = []
        sys_prompt = options.get('system_prompt') or ""
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        for m in history:
            messages.append({"role": m['role'], "content": m['content']})

        user_text = final_message_text

        # Separate image files from text files
        image_files = [fi for fi in loaded_files if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')]
        for fi in loaded_files:
            if fi.get('text') and not (fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')):
                user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"

        zai_accepts_images = is_zai and (model_key_l in {
            "glm-5.3-flash", "glm-5.3-flashx", "glm-4.6v",
            "glm-4.6v-flashx", "glm-4.6v-flash", "glm-4.5v",
        })
        # Text-only models use the configured vision model for image descriptions.
        if image_files and not zai_accepts_images:
            vision_model = (options.get('image_vision_model') or "").strip()
            analysis_prompt = _auto_notice_text("image_analysis") or DEFAULT_IMAGE_ANALYSIS_PROMPT
            if not vision_model:
                pub("error", "Kimi K3 supports images directly, but no Vision Model is configured for automatic image analysis. Please select a Vision Model in Settings > Default Vision Model.")
                return BACKGROUND_STREAM_EARLY_RETURN
            pub("status", "画像を vision model で解析中...")
            analysis_texts = []
            for idx, fi in enumerate(image_files):
                pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析中...")
                if check_stop():
                    return BACKGROUND_STREAM_EARLY_RETURN
                img_data = fi.get('bytes')
                img_mime = str(fi.get('mime', 'image/png'))
                analysis_result = _analyze_image_with_vision_model(
                    vision_model, img_data, img_mime, analysis_prompt,
                    api_keys
                )
                if analysis_result:
                    analysis_texts.append(f"--- Image {idx+1} ---\n{analysis_result}")
                    pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析完了")
                else:
                    pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析失敗")
                if check_stop():
                    return BACKGROUND_STREAM_EARLY_RETURN
            if analysis_texts:
                analysis_block = "The user attached the following image(s). Detailed descriptions are provided below:\n\n" + "\n\n".join(analysis_texts)
                if user_text.strip():
                    messages.append({"role": "system", "content": analysis_block})
                else:
                    # Image-only send: use the vision analysis as the user turn.
                    user_text = analysis_block
                pub("image_analysis", f"全{len(analysis_texts)}枚の画像解析完了。Kimi K3 で応答生成中...")
            else:
                pub("error", "画像の解析に失敗しました。Vision Model の API 設定を確認してください。")
                return BACKGROUND_STREAM_EARLY_RETURN

        if not user_text.strip() and not (zai_accepts_images and image_files):
            pub("error", "Kimi K3 request is empty.")
            return BACKGROUND_STREAM_EARLY_RETURN

        if zai_accepts_images and image_files:
            user_parts = [{"type": "text", "text": user_text}]
            for fi in image_files:
                user_parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{fi.get('mime') or 'image/png'};base64,"
                    + base64.b64encode(fi['bytes']).decode('ascii')
                }})
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": user_text})
        kimi_kwargs = {
            "model": model_key,
            "messages": messages,
            "stream": True,
        }
        if is_kimi:
            kimi_kwargs["stream_options"] = {"include_usage": True}
        # Kimi K3 always thinks; reasoning_effort is top-level (low/high/max, default max)
        enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
        if enable_reasoning and is_kimi:
            kimi_kwargs["reasoning_effort"] = _kimi_reasoning_effort()
        if is_kimi and options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
            kimi_kwargs["extra_body"] = dict(kimi_kwargs.get("extra_body") or {})
            kimi_kwargs["extra_body"]["prompt_cache_key"] = options.get('prompt_cache_key')
            log_force(f"Kimi Prompt Caching key={options.get('prompt_cache_key')}")

        if batch_mode and is_zai:
            _mark_provider_request_started()
            try:
                _submit_zai_batch(client, kimi_kwargs)
            except Exception as batch_error:
                logger.exception('Z.AI Batch submission failed')
                _persist_batch_failure(str(batch_error))
            return BACKGROUND_STREAM_EARLY_RETURN

        kimi_tools = []
        if _skills_tool_enabled:
            kimi_tools.append(build_skill_tool_openai_chat())
        _mark_provider_request_started()
        final_openai_usage = None
        kimi_max_rounds = 8
        for kimi_round in range(kimi_max_rounds + 1):
            if kimi_tools:
                kimi_kwargs["tools"] = kimi_tools
                kimi_kwargs["tool_choice"] = "none" if kimi_round == kimi_max_rounds else "auto"
            stream = client.chat.completions.create(**kimi_kwargs)
            chunk_count = 0
            round_content = ""
            streamed_tool_calls = {}
            for chunk in stream:
                _latency_mark_once(job_id, "provider_first_chunk_ms")
                if check_stop():
                    break

                chunk_count += 1
                if chunk_count % 20 == 0:
                    _refresh_pending_job()

                usage = getattr(chunk, 'usage', None)
                if usage:
                    final_openai_usage = usage

                choices = getattr(chunk, 'choices', None)
                if not choices:
                    continue

                delta = choices[0].delta

                # Extract reasoning_content
                r_content = None
                try:
                    r_content = delta.reasoning_content
                except AttributeError:
                    pass
                if not r_content:
                    try:
                        extra = getattr(delta, '__pydantic_extra__', None) or {}
                        r_content = extra.get('reasoning_content')
                    except Exception:
                        pass
                if r_content:
                    thought_accumulated += r_content
                    pub("thought", r_content)

                c_content = getattr(delta, 'content', None)
                if c_content:
                    round_content += c_content
                    full_res += c_content
                    pub("content", c_content)

                accumulate_deepseek_tool_call_deltas(
                    streamed_tool_calls,
                    getattr(delta, 'tool_calls', None) or [],
                )

            if not streamed_tool_calls:
                break
            if kimi_round == kimi_max_rounds:
                break

            assistant_tool_calls = []
            for tool_index in sorted(streamed_tool_calls):
                tool_state = streamed_tool_calls[tool_index]
                assistant_tool_calls.append({
                    "id": tool_state["id"] or f"kimi_tool_{kimi_round}_{tool_index}",
                    "type": "function",
                    "function": {
                        "name": tool_state["name"],
                        "arguments": tool_state["arguments"],
                    },
                })
            messages.append({
                "role": "assistant",
                "content": round_content,
                "tool_calls": assistant_tool_calls,
            })
            for tool_call in assistant_tool_calls:
                call_id = tool_call["id"]
                function_data = tool_call["function"]
                call_name = function_data.get("name")
                call_arguments = function_data.get("arguments") or ""
                if call_name == SKILL_TOOL_NAME:
                    try:
                        parsed_arguments = json.loads(call_arguments or "{}")
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("arguments must be a JSON object")
                        result = _execute_skill_tool(
                            str(parsed_arguments.get("name") or "")
                        )
                    except Exception as tool_exc:
                        result = f"Error: Invalid use_skill arguments: {tool_exc}"
                else:
                    result = f"Error: Unsupported tool: {call_name or '(missing name)'}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })
    except Exception as e:
        pub("error", f"{'Z.AI' if is_zai else 'Kimi'} Error: {str(e)}")

    return (full_res, m, search_used, thought_accumulated, usage, final_openai_usage)
