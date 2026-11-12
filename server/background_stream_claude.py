"""Claude branch of `background_chat_task` (Anthropic Messages API streaming).

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns the updated answer text and accumulated thoughts.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_claude(local):
    """Run the Claude branch and return ``(full_res, thought_accumulated)``."""

    _ensure_mcp_env = local.get('_ensure_mcp_env')
    _execute_skill_tool = local.get('_execute_skill_tool')
    _load_message_history_images = local.get('_load_message_history_images')
    _mark_provider_request_started = local.get('_mark_provider_request_started')
    _skills_tool_enabled = local.get('_skills_tool_enabled')
    c_client = local.get('c_client')
    check_stop = local.get('check_stop')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    history = local.get('history')
    loaded_files = local.get('loaded_files')
    model_key = local.get('model_key')
    options = local.get('options')
    pub = local.get('pub')
    thought_accumulated = local.get('thought_accumulated')

    # --- 1.5 Anthropic Claude ---
    log_force("Routing: Claude Branch")
    try:
        claude_messages = []
        # Convert history
        for m in history:
            role = 'assistant' if m['role'] == 'assistant' else 'user'
            content = m['content'] or ""

            msg_parts = []
            if content:
                msg_parts.append({"type": "text", "text": content})

            if m.get('image_url'):
                msg_images, _ = _load_message_history_images(m.get('image_url'), include_only_images=True)
                for img in msg_images:
                    msg_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img['mime'],
                            "data": base64.b64encode(img['bytes']).decode('utf-8')
                        }
                    })
            if msg_parts:
                claude_messages.append({"role": role, "content": msg_parts})

        # Current message
        curr_parts = []
        if final_message_text and str(final_message_text).strip():
            curr_parts.append({"type": "text", "text": final_message_text})
        for fi in loaded_files:
            if fi.get('bytes'):
                if fi.get('mime', '').startswith('image/'):
                    curr_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": fi['mime'],
                            "data": base64.b64encode(fi['bytes']).decode('utf-8')
                        }
                    })
                elif fi.get('is_pdf') or fi.get('mime') == 'application/pdf':
                    curr_parts.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.b64encode(fi['bytes']).decode('utf-8')
                        }
                    })
        claude_messages.append({"role": "user", "content": curr_parts})

        # Claude Parameters
        claude_kwargs = {
            "model": model_key,
            "messages": claude_messages,
            "max_tokens": 8192,
        }
        sys_prompt_claude = options.get('system_prompt')
        if options.get('enable_prompt_caching'):
            # Automatic prompt caching (Claude): top-level cache_control
            claude_kwargs["cache_control"] = {"type": "ephemeral"}
            if sys_prompt_claude:
                claude_kwargs["system"] = [{
                    "type": "text",
                    "text": sys_prompt_claude,
                    "cache_control": {"type": "ephemeral"},
                }]
            log_force("Claude Prompt Caching enabled")
        elif sys_prompt_claude:
            claude_kwargs["system"] = sys_prompt_claude

        if options.get('enable_thinking'):
            budget = 0
            try:
                budget = int(options.get('thinking_budget') or 4096)
            except: budget = 4096
            if budget < 1024: budget = 1024
            claude_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            claude_kwargs["max_tokens"] = budget + 4096

        # MCP外部ツール（Claude）の付与
        claude_mcp_env = None
        claude_mcp_tools = []
        try:
            claude_mcp_env = _ensure_mcp_env()
        except Exception:
            claude_mcp_env = None
        if claude_mcp_env is not None:
            try:
                claude_mcp_tools = claude_mcp_env.serialize_anthropic()
            except Exception:
                claude_mcp_tools = []
        claude_skill_tools = (
            [build_skill_tool_anthropic()] if _skills_tool_enabled else []
        )
        claude_tools = list(claude_mcp_tools) + claude_skill_tools
        if claude_tools:
            claude_kwargs["tools"] = claude_tools
            log_force(
                f"Claude tools attached: mcp={len(claude_mcp_tools)} "
                f"skill={len(claude_skill_tools)}"
            )

        claude_max_rounds = 20 if claude_tools else 1
        for _claude_round in range(claude_max_rounds):
            _mark_provider_request_started()
            _stopped_claude = False
            with c_client.messages.stream(**claude_kwargs) as stream:
                for event in stream:
                    if check_stop():
                        _stopped_claude = True
                        break
                    if event.type == "content_block_start":
                        if event.content_block.type == "thinking":
                            pub("thought_start", "")
                    elif event.type == "content_block_delta":
                        if event.delta.type == "thinking_delta":
                            thought = event.delta.thinking
                            thought_accumulated += thought
                            pub("thought", thought)
                        elif event.delta.type == "text_delta":
                            txt = event.delta.text
                            full_res += txt
                            pub("content", txt)
                    elif event.type == "content_block_stop":
                        pass
                # After stream, finalize thoughts if any
                if thought_accumulated:
                    pub("thought_stop", "")
                final_message = None
                if claude_tools and not _stopped_claude:
                    try:
                        final_message = stream.get_final_message()
                    except Exception:
                        final_message = None
            if _stopped_claude or not claude_tools or final_message is None:
                break
            stop_reason = getattr(final_message, "stop_reason", None)
            if stop_reason != "tool_use":
                break
            content_blocks = list(getattr(final_message, "content", None) or [])
            tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break
            # アシスタント履歴へ追加（thinkingブロックはAPIへ送らない）
            history_content = []
            for b in content_blocks:
                bt = getattr(b, "type", None)
                if bt == "text":
                    history_content.append({"type": "text", "text": getattr(b, "text", "") or ""})
                elif bt == "tool_use":
                    history_content.append({
                        "type": "tool_use",
                        "id": getattr(b, "id", "") or "",
                        "name": getattr(b, "name", "") or "",
                        "input": getattr(b, "input", None) or {},
                    })
            claude_messages.append({"role": "assistant", "content": history_content})
            tool_result_blocks = []
            for tu in tool_uses:
                tu_id = getattr(tu, "id", "") or ""
                tu_name = getattr(tu, "name", "") or ""
                _args_in = getattr(tu, "input", None) or {}
                if tu_name == SKILL_TOOL_NAME:
                    try:
                        _skill_text = _execute_skill_tool(str(_args_in.get("name") or ""))
                        tool_result_blocks.append({
                            "type": "tool_result", "tool_use_id": tu_id,
                            "content": _skill_text or "Tool executed.",
                        })
                    except Exception as _te:
                        tool_result_blocks.append({
                            "type": "tool_result", "tool_use_id": tu_id,
                            "content": f"Error executing skill tool: {_te}",
                            "is_error": True,
                        })
                    continue
                if not tu_id or not str(tu_name).startswith("mcp__"):
                    tool_result_blocks.append({
                        "type": "tool_result", "tool_use_id": tu_id,
                        "content": "Error: unknown tool",
                        "is_error": True,
                    })
                    continue
                try:
                    mcp_txt, mcp_out = claude_mcp_env.execute(tu_name, _args_in)
                    if mcp_out.get("ok"):
                        _md = f"\n\n> 🔧 **MCPツール実行:** `{tu_name}` を実行しました。\n"
                        full_res += _md
                        pub("content", _md)
                    elif mcp_out.get("rejected"):
                        _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{tu_name}`\n"
                        full_res += _md
                        pub("content", _md)
                    _result_content = mcp_txt if str(mcp_txt).strip() else "Tool executed."
                    tool_result_blocks.append({
                        "type": "tool_result", "tool_use_id": tu_id,
                        "content": _result_content,
                    })
                except Exception as _te:
                    tool_result_blocks.append({
                        "type": "tool_result", "tool_use_id": tu_id,
                        "content": f"Error executing MCP tool: {_te}",
                        "is_error": True,
                    })
            claude_messages.append({"role": "user", "content": tool_result_blocks})
            claude_kwargs["messages"] = claude_messages

        if not full_res and not thought_accumulated:
            pub("error", "Claude returned empty response.")

    except Exception as e:
        logger.exception("Claude Branch Error")
        pub("error", f"Claude Error: {str(e)}")
    return (full_res, thought_accumulated)
