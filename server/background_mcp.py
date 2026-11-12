"""MCP runtime setup for `background_chat_task`.

Extracted from `server/background.py` so the worker entry point keeps only the
job setup and provider dispatch.  The builder returns a lazy `ensure()` closure
so every text-LLM provider branch shares one `tools/list` result and the
guidance text is injected into the system prompt before the branch reads it.
Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_mcp_env_builder(user_id, job_id, pub, check_stop, options, auto_notice_enabled, auto_notice_text):
    """Return the lazy MCP environment builder used by the provider branches.

    ---- MCP（外部モデル連携）実行環境の遅延構築ヘルパ ----
    テキストLLM分岐（Gemini / Claude / DeepSeek / OpenAI系Responses）が
    必要になった時点で tools/list を取得し、モデルへ付与する。
    案内文はシステムプロンプトへ先に入れ、各プロバイダ分岐が
    system / system_instruction を組む前にモデルが MCP の存在を知る。
    プロンプトバーのMCPスイッチ（enable_mcp）が OFF の間は、
    MCPツールの付与と案内文注入の両方を無効化する。
    値が無い（None）従来リクエストは従来どおり有効扱いにする。
    """
    state = {"env": None, "prompt_injected": False}
    request_disabled = options.get('enable_mcp') is False

    def ensure():
        if state["env"] is not None:
            return state["env"]
        if request_disabled:
            return None
        from mcp_service.execution import McpRuntime
        _env = McpRuntime(
            user_id,
            job_id=job_id,
            pub=pub,
            check_stop=check_stop,
            log=log_force,
        )
        try:
            _env.load()
        except Exception as _mcp_exc:
            log_force(f"MCP load failed: {_mcp_exc}")
        state["env"] = _env if not _env.empty() else None
        if state["env"] is not None:
            log_force(f"MCP enabled: {len(state['env'].tool_metas())} tools across {len(state['env'].servers)} servers")
            if not state["prompt_injected"]:
                try:
                    # 案内文は「自動注入システムプロンプト（ユーザー単位）」の
                    # mcp 項目の文面に従う。項目のオン・オフはプロンプトバーの
                    # MCPスイッチに連動しており（ここまで到達 = 有効）、
                    # ここでは全体適用トグルと文面だけを確認する。
                    if auto_notice_enabled("mcp"):
                        _mcp_note = state["env"].guidance_text(preamble=auto_notice_text("mcp"))
                        if _mcp_note:
                            _cur_sys = options.get('system_prompt') or ""
                            if "Model Context Protocol" not in str(_cur_sys):
                                options['system_prompt'] = (
                                    f"{_mcp_note}\n\n{_cur_sys}".strip() if str(_cur_sys).strip() else _mcp_note
                                )
                    state["prompt_injected"] = True
                except Exception as _mcp_note_exc:
                    log_force(f"MCP system prompt inject failed: {_mcp_note_exc}")
        return state["env"]

    return ensure


def background_mcp_summary_md(meta_out, internal_name):
    """MCP実行結果をメッセージ本文へ追記するMarkdownを作る。"""
    try:
        if meta_out.get("ok") and meta_out.get("id"):
            return ""
        if meta_out.get("rejected"):
            return f"\n\n> 🚫 **MCPツール実行（ユーザーが拒否）:** `{internal_name}`\n"
        if meta_out.get("is_error"):
            return ""
        return ""
    except Exception:
        return ""
