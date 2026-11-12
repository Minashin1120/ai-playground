"""DeepSeek branch of `background_chat_task` (Chat Completions streaming).

`background.py` keeps the provider dispatch and the shared stream tail.  This
part owns the DeepSeek request, streaming tool rounds, image/vision handling
and search citations.

The branch used to run inline in `background_chat_task`, so it shared that
function's locals.  It now receives a `locals()` snapshot as ``local`` and
returns the values the shared tail still needs.  Module-level helpers are
resolved through the `app` namespace because `app.py` execs every server part
into one module.
"""

# Returned to background_chat_task when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_deepseek(local):
    """Run the DeepSeek branch and report the values the stream tail needs."""
    _auto_notice_enabled = local.get("_auto_notice_enabled")
    _auto_notice_text = local.get("_auto_notice_text")
    _deepseek_reasoning_effort = local.get("_deepseek_reasoning_effort")
    _ensure_mcp_env = local.get("_ensure_mcp_env")
    _execute_skill_tool = local.get("_execute_skill_tool")
    _load_message_history_images = local.get("_load_message_history_images")
    _mark_provider_request_started = local.get("_mark_provider_request_started")
    _refresh_pending_job = local.get("_refresh_pending_job")
    _remember_python_image = local.get("_remember_python_image")
    _skills_tool_enabled = local.get("_skills_tool_enabled")
    agentic_image_digests = local.get("agentic_image_digests")
    agentic_saved_urls = local.get("agentic_saved_urls")
    api_keys = local.get("api_keys")
    data = local.get("data")
    deepseek_tool_context = local.get("deepseek_tool_context")
    e = local.get("e")
    final_message_text = local.get("final_message_text")
    generated_images = local.get("generated_images")
    history = local.get("history")
    img = local.get("img")
    job_id = local.get("job_id")
    key = local.get("key")
    loaded_files = local.get("loaded_files")
    model_key = local.get("model_key")
    model_key_l = local.get("model_key_l")
    o_client = local.get("o_client")
    options = local.get("options")
    pub = local.get("pub")
    req_reasoning_effort = local.get("req_reasoning_effort")
    thread_id = local.get("thread_id")
    usage_ledger = local.get("usage_ledger")
    user_config = local.get("user_config")
    user_id = local.get("user_id")

    # State the shared stream tail still reads back.
    check_stop = local.get("check_stop")
    final_openai_usage = local.get("final_openai_usage")
    full_res = local.get("full_res") or ""
    m = local.get("m")
    search_used = local.get("search_used") or False
    thought_accumulated = local.get("thought_accumulated") or ""
    usage = local.get("usage")

    log_force("Routing: DeepSeek V4 Branch (Chat Completions)")
    try:
        if check_stop():
            return BACKGROUND_STREAM_EARLY_RETURN
        client = o_client
        messages = []
        sys_prompt = options.get('system_prompt') or ""
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # region: image inputs
        # DeepSeek V4.1 Flash accepts images natively via OpenAI-compatible
        # image_url content blocks (official Vision guide). The retired Vision
        # Exp alias remains native-vision capable for history compatibility.
        deepseek_native_vision = model_key_l in {"deepseek-v4.1-flash", "deepseek-v4-flash-vision-exp"}
        ds_history_img_seen = set()
        ds_history_img_bytes = 0

        for m in history:
            saved_tool_context = m.get("deepseek_tool_context")
            if m.get("role") == "assistant" and isinstance(saved_tool_context, list):
                for saved_message in saved_tool_context:
                    if isinstance(saved_message, dict) and saved_message.get("role") in {"assistant", "tool"}:
                        messages.append(saved_message)
            # DeepSeek accepts image content in user messages only.  Rebuilding
            # assistant/system history entries with image_url blocks causes a
            # provider-side 400 even when the current request is valid.
            elif deepseek_native_vision and m.get('role') == 'user' and m.get('image_url'):
                try:
                    ds_content_parts = [{"type": "text", "text": m['content'] or ""}]
                    ds_msg_images, ds_history_img_bytes = _load_message_history_images(
                        m.get('image_url'),
                        seen=ds_history_img_seen,
                        total_bytes=ds_history_img_bytes,
                        include_only_images=True
                    )
                    for ds_msg_img in ds_msg_images:
                        ds_bytes, ds_mime = _compact_vision_image(ds_msg_img['bytes'], ds_msg_img['mime'])
                        ds_b64 = base64.b64encode(ds_bytes).decode('utf-8')
                        ds_content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{ds_mime};base64,{ds_b64}"}
                        })
                    messages.append({"role": m['role'], "content": ds_content_parts})
                except Exception as ds_hist_err:
                    log_force(f"Error processing history image in DeepSeek branch: {ds_hist_err}")
                    messages.append({"role": m['role'], "content": m['content']})
            else:
                messages.append({"role": m['role'], "content": m['content']})

        user_text = final_message_text

        # Separate image files from text files
        image_files = [fi for fi in loaded_files if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')]
        for fi in loaded_files:
            if fi.get('text') and not (fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')):
                user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"

        # region: search
        # DeepSeek Search runs before generation. The official Search API
        # works with every DeepSeek model, unlike the Responses API web_search
        # tool (deepseek-v4-pro only) which is ignored by deepseek-flash.
        deepseek_search_sources = []
        deepseek_search_calls = 0
        if options.get('enable_search') and key and str(final_message_text or "").strip():
            pub("search_status", "searching")
            try:
                ds_results, ds_search_error = _deepseek_search_request(
                    key, final_message_text
                )
                if ds_results:
                    search_block = _format_deepseek_search_results(ds_results)
                    user_text += (
                        "\n\n[Web search results from DeepSeek Search]\n"
                        f"{search_block}"
                    )
                    seen_search_urls = set()
                    for ds_item in ds_results:
                        ds_url = str(ds_item.get("url") or "").strip()
                        if not ds_url or ds_url in seen_search_urls:
                            continue
                        seen_search_urls.add(ds_url)
                        deepseek_search_sources.append(
                            (str(ds_item.get("title") or ds_url).strip() or ds_url, ds_url)
                        )
                    deepseek_search_calls += 1
                    search_used = True
                    log_force(
                        f"DeepSeek Search: {len(ds_results)} results, "
                        f"{len(deepseek_search_sources)} sources"
                    )
                elif ds_search_error:
                    log_force(f"DeepSeek Search skipped: {ds_search_error}")
            except Exception as ds_search_exc:
                log_force(f"DeepSeek Search failed: {ds_search_exc}")
            finally:
                pub("search_status", "done")

        # If images are present, analyze them with a vision model unless the selected
        # DeepSeek model accepts image inputs natively (DeepSeek V4.1 Flash).
        if image_files and not deepseek_native_vision:
            vision_model = (options.get('image_vision_model') or "").strip()
            analysis_prompt = _auto_notice_text("image_analysis") or DEFAULT_IMAGE_ANALYSIS_PROMPT
            if not vision_model:
                pub("error", "This DeepSeek model does not support images. Please select a Vision Model in Settings > Default Vision Model to enable automatic image analysis.")
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
                pub("image_analysis", f"全{len(analysis_texts)}枚の画像解析完了。DeepSeek で応答生成中...")
            else:
                pub("error", "画像の解析に失敗しました。Vision Model の API 設定を確認してください。")
                return BACKGROUND_STREAM_EARLY_RETURN

        deepseek_user_content = None
        if image_files and deepseek_native_vision:
            # Native vision path: attach images inline as OpenAI-compatible
            # base64 data URL blocks alongside the text turn.
            deepseek_user_content = []
            if user_text.strip():
                deepseek_user_content.append({"type": "text", "text": user_text})
            for ds_idx, ds_file in enumerate(image_files):
                ds_img_bytes, ds_img_mime = _compact_vision_image(ds_file.get('bytes') or b"", str(ds_file.get('mime', 'image/png')) or "image/png")
                ds_img_b64 = base64.b64encode(ds_img_bytes).decode("utf-8")
                deepseek_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{ds_img_mime};base64,{ds_img_b64}"}
                })
                pub("image_analysis", f"画像 {ds_idx+1}/{len(image_files)} を送信対象に追加")

        if not user_text.strip() and deepseek_user_content is None:
            pub("error", "DeepSeek request is empty.")
            return BACKGROUND_STREAM_EARLY_RETURN

        if deepseek_user_content is not None:
            messages.append({"role": "user", "content": deepseek_user_content})
        else:
            messages.append({"role": "user", "content": user_text})
        enable_reasoning = (
            req_reasoning_effort != "none"
            and (bool(options.get('enable_thinking')) or bool(req_reasoning_effort))
        )
        def publish_python_image(data):
            nonlocal full_res
            image_data, ext = _prepare_agentic_image_bytes(data)
            digest = hashlib.sha256(image_data).hexdigest()
            if digest in agentic_image_digests:
                return
            agentic_image_digests.add(digest)
            _, url = _save_user_generated_bytes_verified(
                user_id, image_data,
                lambda: f"agentic_{int(time.time() * 1000)}_{os.urandom(4).hex()}.{ext}",
                user_config.get('enable_e2ee'),
            )
            agentic_saved_urls.append(url)
            markdown = f"\n![Agentic View]({url})\n"
            full_res += markdown
            pub("content", markdown)

        python_tools = []
        if options.get("enable_python"):
            if _auto_notice_enabled("deepseek_python"):
                messages.append({
                    "role": "system",
                    "content": _auto_notice_text("deepseek_python"),
                })
            python_tools.append({
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": (
                        "Execute Python code for calculations, verification, or data analysis. "
                        "The isolated environment has no network access. "
                        "Print text results. Save output images in /work as PNG, JPEG, WebP or SVG; they are displayed immediately. "
                        "Attached images are /work/input_1.<ext>, input_2.<ext>, etc.; use glob to locate them."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python source code to execute. Use print(...) for every value "
                                    "you need in the tool result."
                                ),
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            })
            python_tools.append(_build_list_branch_images_tool_schema())
            python_tools.append(_build_import_branch_images_tool_schema())
        if options.get("enable_file_creation"):
            python_tools.append(_build_create_file_tool_schema())
            python_tools.append(_build_edit_file_tool_schema())
        if _skills_tool_enabled:
            python_tools.append(build_skill_tool_openai_chat())

        # region: tool schemas
        # MCP外部ツールの定義を追加（ユーザーが有効化＆認証済みのサーバーのみ）
        ds_mcp_env = None
        try:
            ds_mcp_env = _ensure_mcp_env()
        except Exception:
            ds_mcp_env = None
        if ds_mcp_env is not None:
            try:
                python_tools.extend(ds_mcp_env.serialize_chat_completions())
            except Exception as _mcp_e:
                log_force(f"DeepSeek MCP tool attach failed: {_mcp_e}")

        deepseek_usage_totals = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        saw_deepseek_usage = False
        # region: tool rounds and streaming
        max_tool_rounds = 8
        for tool_round in range(max_tool_rounds + 1):
            if tool_round == max_tool_rounds:
                pub("status", "ツールの実行結果から回答を生成中...")
            deepseek_kwargs = {
                "model": _deepseek_api_model_id(model_key),
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if python_tools:
                deepseek_kwargs["tools"] = python_tools
                # Reserve one final request for an answer using the collected results.
                deepseek_kwargs["tool_choice"] = "none" if tool_round == max_tool_rounds else "auto"
            if enable_reasoning:
                deepseek_kwargs["reasoning_effort"] = _deepseek_reasoning_effort()
                deepseek_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                deepseek_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            # Official per-user isolation for safety, scheduling, and KV cache.
            deepseek_kwargs["extra_body"]["user_id"] = f"app_user_{user_id}"

            _mark_provider_request_started()
            stream = client.chat.completions.create(**deepseek_kwargs)
            chunk_count = 0
            round_reasoning = ""
            round_content = ""
            streamed_tool_calls = {}

            for chunk in stream:
                _latency_mark_once(job_id, "provider_first_chunk_ms")
                if check_stop():
                    return BACKGROUND_STREAM_EARLY_RETURN

                # Refresh pending_job TTL periodically to prevent expiry during long thinking
                chunk_count += 1
                if chunk_count % 20 == 0:
                    _refresh_pending_job()

                usage = getattr(chunk, 'usage', None)
                if usage:
                    saw_deepseek_usage = True
                    for usage_key in ("completion_tokens", "prompt_tokens", "total_tokens"):
                        usage_value = (
                            usage.get(usage_key)
                            if isinstance(usage, dict)
                            else getattr(usage, usage_key, None)
                        )
                        if usage_value is not None:
                            deepseek_usage_totals[usage_key] += int(usage_value or 0)
                    usage_details = (
                        usage.get("completion_tokens_details")
                        if isinstance(usage, dict)
                        else getattr(usage, "completion_tokens_details", None)
                    )
                    reasoning_tokens = (
                        usage_details.get("reasoning_tokens")
                        if isinstance(usage_details, dict)
                        else getattr(usage_details, "reasoning_tokens", None)
                        if usage_details is not None
                        else None
                    )
                    if reasoning_tokens is not None:
                        deepseek_usage_totals["completion_tokens_details"]["reasoning_tokens"] += int(reasoning_tokens or 0)

                choices = getattr(chunk, 'choices', None)
                if not choices:
                    continue

                delta = choices[0].delta

                # DeepSeek exposes reasoning_content as an OpenAI-compatible extension.
                r_content = getattr(delta, "reasoning_content", None)
                if not r_content:
                    extra = getattr(delta, '__pydantic_extra__', None) or {}
                    r_content = extra.get('reasoning_content')
                if r_content:
                    round_reasoning += r_content
                    thought_accumulated += r_content
                    pub("thought", r_content)

                c_content = getattr(delta, 'content', None)
                if c_content:
                    round_content += c_content
                    full_res += c_content
                    pub("content", c_content)

                accumulate_deepseek_tool_call_deltas(
                    streamed_tool_calls,
                    getattr(delta, "tool_calls", None) or [],
                )

            if not streamed_tool_calls:
                if deepseek_tool_context:
                    final_assistant_message = {
                        "role": "assistant",
                        "content": round_content,
                    }
                    if enable_reasoning:
                        final_assistant_message["reasoning_content"] = round_reasoning
                    deepseek_tool_context.append(final_assistant_message)
                break

            if tool_round == max_tool_rounds:
                raise RuntimeError("DeepSeek returned tool calls after tools were disabled.")

            assistant_tool_calls = []
            for tool_index in sorted(streamed_tool_calls):
                tool_state = streamed_tool_calls[tool_index]
                assistant_tool_calls.append({
                    "id": tool_state["id"] or f"deepseek_tool_{tool_round}_{tool_index}",
                    "type": "function",
                    "function": {
                        "name": tool_state["name"],
                        "arguments": tool_state["arguments"],
                    },
                })

            assistant_tool_message = {
                "role": "assistant",
                "content": round_content,
                "tool_calls": assistant_tool_calls,
            }
            if enable_reasoning:
                # Required by DeepSeek for every subsequent request after a thinking tool call.
                assistant_tool_message["reasoning_content"] = round_reasoning
            messages.append(assistant_tool_message)
            deepseek_tool_context.append(assistant_tool_message)

            for tool_call in assistant_tool_calls:
                call_id = tool_call["id"]
                function_data = tool_call["function"]
                call_name = function_data.get("name")
                call_arguments = function_data.get("arguments") or ""
                code = ""
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
                elif call_name == "create_file":
                    try:
                        parsed_arguments = json.loads(call_arguments)
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("arguments must be a JSON object")
                        create_result = _execute_create_file_tool(
                            user_id, parsed_arguments, user_config.get('enable_e2ee')
                        )
                        result = _create_file_tool_result_text(create_result)
                        if create_result.get("ok"):
                            created_file_rel = f"{user_id}/{create_result['filename']}"
                            if created_file_rel not in generated_images:
                                generated_images.append(created_file_rel)
                            file_link_md = (
                                f"\n📄 **ファイルを作成しました:** [{create_result.get('display_name')}]({create_result.get('url')})\n"
                            )
                            full_res += file_link_md
                            pub("content", file_link_md)
                    except Exception as tool_exc:
                        result = f"Error: Invalid create_file arguments: {tool_exc}"
                elif call_name == "edit_file":
                    try:
                        parsed_arguments = json.loads(call_arguments)
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("arguments must be a JSON object")
                        edit_result = _execute_edit_file_tool(
                            user_id,
                            parsed_arguments,
                            user_config.get('enable_e2ee'),
                            loaded_files=loaded_files,
                            history=history,
                            thread_id=thread_id,
                        )
                        result = _create_file_tool_result_text(edit_result, "edit_file", "編集")
                        if edit_result.get("ok"):
                            edited_file_rel = f"{user_id}/{edit_result['filename']}"
                            if edited_file_rel not in generated_images:
                                generated_images.append(edited_file_rel)
                            file_link_md = (
                                f"\n📄 **ファイルを編集しました:** [{edit_result.get('display_name')}]({edit_result.get('url')})\n"
                            )
                            full_res += file_link_md
                            pub("content", file_link_md)
                    except Exception as tool_exc:
                        result = f"Error: Invalid edit_file arguments: {tool_exc}"
                elif str(call_name or "").startswith("mcp__"):
                    try:
                        parsed_arguments = json.loads(call_arguments)
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("arguments must be a JSON object")
                        if ds_mcp_env is None:
                            try:
                                ds_mcp_env = _ensure_mcp_env()
                            except Exception:
                                ds_mcp_env = None
                        if ds_mcp_env is None:
                            result = f"Error: MCP tools are not available: {call_name}"
                        else:
                            mcp_text, mcp_out = ds_mcp_env.execute(call_name, parsed_arguments)
                            result = mcp_text
                            if mcp_out.get("ok"):
                                _md = f"\n\n> 🔧 **MCPツール実行:** `{call_name}` を実行しました。\n"
                                full_res += _md
                                pub("content", _md)
                            elif mcp_out.get("rejected"):
                                _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{call_name}`\n"
                                full_res += _md
                                pub("content", _md)
                    except Exception as tool_exc:
                        result = f"Error: Invalid MCP tool arguments: {tool_exc}"
                elif call_name == "list_branch_images":
                    imported_keys = {str(fi.get('path') or fi.get('name') or '') for fi in image_files}
                    result = _execute_list_branch_images(
                        history=history,
                        loaded_files=loaded_files,
                        user_id=user_id,
                        imported_keys=imported_keys,
                    )
                elif call_name == "import_branch_images":
                    def _on_import_ds(img):
                        img_data = img.get('bytes') or b''
                        img_mime = img.get('mime')
                        img_name = img.get('name')
                        _remember_python_image(img_data, img_mime)
                        image_files.append({
                            'bytes': img_data,
                            'mime': img_mime,
                            'name': img_name,
                        })
                        ext = _guess_image_extension(img_mime, img_name)
                        return f"/work/input_{len(image_files)}.{ext}"

                    try:
                        parsed_arguments = json.loads(call_arguments) if call_arguments else {}
                        if not isinstance(parsed_arguments, dict):
                            parsed_arguments = {}
                    except Exception:
                        parsed_arguments = {}
                    imported_keys = {str(fi.get('path') or fi.get('name') or '') for fi in image_files}
                    result = _execute_import_branch_images(
                        parsed_arguments.get("image_ids", []),
                        history=history,
                        loaded_files=loaded_files,
                        user_id=user_id,
                        imported_keys=imported_keys,
                        on_import_callback=_on_import_ds,
                    )
                elif call_name != "execute_python":
                    result = f"Error: Unsupported tool: {call_name or '(missing name)'}"
                else:
                    try:
                        parsed_arguments = json.loads(call_arguments)
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("arguments must be a JSON object")
                        code = parsed_arguments.get("code")
                        if not isinstance(code, str) or not code.strip():
                            raise ValueError("code must be a non-empty string")
                        result = safe_execute_python(code, on_image=publish_python_image, input_images=image_files)
                    except Exception as tool_exc:
                        result = f"Error: Invalid execute_python arguments: {tool_exc}"

                if code:
                    code_fence = _markdown_fence_for_code(code)
                    output_fence = _markdown_fence_for_code(result)
                    visible_tool_output = (
                        f"\n{code_fence}python\n{code}\n{code_fence}\n"
                        f"\n**Output:**\n{output_fence}\n{result}\n{output_fence}\n"
                    )
                    full_res += visible_tool_output
                    pub("content", visible_tool_output)
                    full_res += (
                        f"\n```pyexec\n"
                        f"{json.dumps({'code': code, 'output': result}, ensure_ascii=False)}"
                        f"\n```\n"
                    )
                    pub("python", {"id": call_id, "code": code, "output": result})

                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }
                messages.append(tool_message)
                deepseek_tool_context.append(tool_message)
            continue
        else:
            raise RuntimeError(f"DeepSeek Python tool exceeded {max_tool_rounds} rounds.")

        if deepseek_search_sources:
            citations_text = "\n\n**Sources:**\n"
            for source_title, source_url in deepseek_search_sources:
                citations_text += f"- [{source_title}]({source_url})\n"
            full_res += citations_text
            pub("content", citations_text)

        if deepseek_search_calls:
            # The Search API is billed per successful request and is not
            # part of the model's token usage; attach it to the latest
            # DeepSeek call so the cost estimate includes the tool fee.
            for usage_call in reversed(usage_ledger.calls):
                call_model = str(usage_call.record.get('model') or '').lower()
                if call_model.startswith('deepseek'):
                    usage_call.record['web_search_calls'] = (
                        usage_call.record.get('web_search_calls') or 0
                    ) + deepseek_search_calls
                    break

        if saw_deepseek_usage:
            final_openai_usage = deepseek_usage_totals
    except Exception as e:
        pub("error", f"DeepSeek Error: {str(e)}")

    return (full_res, m, search_used, thought_accumulated, usage, final_openai_usage)
