"""Gemini chat and streaming branch of `background_chat_task`.

Extracted from `server/background_stream_gemini.py`.  The branch used to run
inline in `background_chat_task`, so it shared that function's locals.  It now
receives a `locals()` snapshot as ``local`` and returns (updated answer text, accumulated thought text).

File upload / Files API cache helpers live in `background_stream_gemini_files.py`
and the grounding / citation / thought helpers in
`background_stream_gemini_citations.py`, so this part keeps only the stream flow.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to the caller when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_gemini_chat(local):
    """Stream a chat answer and return the updated answer text and thought text."""
    # region: Stream setup, request configuration, and tools

    _auto_notice_enabled = local.get("_auto_notice_enabled")
    _auto_notice_text = local.get("_auto_notice_text")
    _ensure_mcp_env = local.get("_ensure_mcp_env")
    _execute_skill_tool = local.get("_execute_skill_tool")
    _load_message_history_images = local.get("_load_message_history_images")
    _mark_provider_request_started = local.get("_mark_provider_request_started")
    _persist_batch_failure = local.get("_persist_batch_failure")
    _record_skill_meta = local.get("_record_skill_meta")
    _skills_for_tools = local.get("_skills_for_tools")
    _skills_tool_enabled = local.get("_skills_tool_enabled")
    _submit_gemini_batch = local.get("_submit_gemini_batch")
    agentic_consumed_urls = local.get("agentic_consumed_urls")
    agentic_filename_url_map = local.get("agentic_filename_url_map")
    agentic_image_digests = local.get("agentic_image_digests")
    agentic_saved_urls = local.get("agentic_saved_urls")
    auto_enable_maps = local.get("auto_enable_maps")
    auto_enable_search = local.get("auto_enable_search")
    auto_enable_url_context = local.get("auto_enable_url_context")
    batch_mode = local.get("batch_mode")
    check_stop = local.get("check_stop")
    final_message_text = local.get("final_message_text")
    full_res = local.get("full_res")
    g_client = local.get("g_client")
    gemini_backend_mode = local.get("gemini_backend_mode")
    gemini_files_api_enabled = gemini_backend_mode != "vertex_ai"
    gemini_local_python = local.get("gemini_local_python")
    generated_images = local.get("generated_images")
    history = local.get("history")
    is_gemini_3 = local.get("is_gemini_3")
    job_id = local.get("job_id")
    loaded_files = local.get("loaded_files")
    mime = local.get("mime")
    model_key = local.get("model_key")
    options = local.get("options")
    prior_sandbox_names = local.get("prior_sandbox_names")
    pub = local.get("pub")
    sandbox_text_buffer = local.get("sandbox_text_buffer")
    signature_parts = local.get("signature_parts")
    supports_docx_inputs = local.get("supports_docx_inputs")
    supports_pdf_inputs = local.get("supports_pdf_inputs")
    supports_text_file_inputs = local.get("supports_text_file_inputs")
    thought_accumulated = local.get("thought_accumulated")
    thread_id = local.get("thread_id")
    user_config = local.get("user_config")
    user_id = local.get("user_id")

    # Text/Chat generation mode
    rm = model_key
    if model_key == "gemini-3.8-flash-cyber":
        rm = "gemini-3.8-flash-cyber"
    elif "gemini-4-argon" in model_key:
        rm = "gemini-4-argon"
    elif "gemini-3.8-flash" in model_key:
        rm = "gemini-3.8-flash"
    elif "gemini-3.7-flash" in model_key:
        rm = "gemini-3.7-flash"
    elif "gemini-3.6-flash" in model_key:
        rm = "gemini-3.6-flash"
    elif "gemini-3.5-flash-lite" in model_key:
        rm = "gemini-3.5-flash-lite"
    elif "gemini-3.5-flash" in model_key:
        rm = "gemini-3.5-flash"
    elif "gemini-3.1-pro" in model_key:
        rm = "gemini-3.1-pro-preview"
    elif model_key == "gemini-3.1-flash-lite":
        rm = "gemini-3.1-flash-lite"
    elif "gemini-3.1-flash-lite-preview" in model_key:
        rm = "gemini-3.1-flash-lite-preview"
    elif "gemini-3-flash" in model_key or "gemini-3.0-flash" in model_key:
        rm = "gemini-3-flash-preview"
    elif "gemini-3-pro" in model_key or "gemini-3.0-pro" in model_key:
        rm = "gemini-3-pro-preview"
    elif "gemini-2.5-pro" in model_key:
        rm = "gemini-2.5-pro"
    elif "gemini-2.5-flash-lite" in model_key:
        rm = model_key
    elif "gemini-2.5" in model_key:
        rm = "gemini-2.5-flash"

    is_latest_flash = rm in ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")
    # Gemini 4 Argon, 3.8/3.7/3.6 Flash and 3.5 Flash-Lite deprecate sampling
    # parameters. Omit them so future API generations do not
    # reject otherwise valid requests.
    conf = {} if is_latest_flash else {'temperature': 0.7}
    if is_gemini_3:
        # Gemini 3 does not support fully disabling thinking; force enabled.
        options['enable_thinking'] = True
    if options.get('enable_thinking'):
        raw_lvl = (options.get('thinking_level') or 'high').lower()
        lvl = raw_lvl if raw_lvl in ("minimal", "low", "medium", "high") else "high"
        if rm == "gemini-4-argon" and lvl not in ("low", "medium", "high"):
            lvl = "medium"
        elif rm == "gemini-3.8-flash" and lvl not in ("low", "medium", "high"):
            lvl = "medium"
        elif rm == "gemini-3.7-flash" and lvl not in ("low", "medium", "high"):
            lvl = "medium"
        elif rm == "gemini-3.6-flash" and lvl not in ("medium", "high"):
            lvl = "medium"
        elif rm == "gemini-3.5-flash-lite" and lvl not in ("minimal", "medium", "high"):
            lvl = "minimal"
        if "gemini-2.5" in model_key:
            budget_map = {"low": 1024, "medium": 4096, "high": 8192}
            manual_budget = options.get('thinking_budget')
            budget_val = None
            if manual_budget is not None and str(manual_budget).strip() != "":
                try:
                    budget_val = int(manual_budget)
                    if budget_val < 0: budget_val = 0
                    if budget_val > 32768: budget_val = 32768
                except Exception:
                    budget_val = None
            conf['thinking_config'] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=budget_val if budget_val is not None else budget_map.get(raw_lvl, 4096)
            )
        else:
            conf['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_level=lvl)
    # Avoid forcing "minimal" when users disable thinking, because Gemini 3 does not
    # support fully turning thinking off and defaults are higher per docs.

    # Gemini 3 supports combining its built-in code execution tool
    # with custom function tools. Older Gemini models do not, so
    # when File is also enabled we expose the same restricted local
    # executor as a custom function instead of sending incompatible
    # built-in and custom tools together.
    _gemini_python_function_active = bool(
        options.get('enable_python')
        and options.get('enable_file_creation')
        and not is_gemini_3
        and not gemini_local_python
    )
    _gemini_manual_function_tools = bool(
        options.get('enable_python')
        and options.get('enable_file_creation')
        and is_gemini_3
        and not gemini_local_python
    )
    _gemini_code_exec_active = False
    if auto_enable_search:
        conf['tools'] = [types.Tool(google_search=types.GoogleSearch())]
    if auto_enable_url_context:
        if 'tools' not in conf: conf['tools'] = []
        conf['tools'].append(types.Tool(url_context=types.UrlContext()))
    if auto_enable_maps:
        if 'tools' not in conf: conf['tools'] = []
        conf['tools'].append(types.Tool(google_maps=types.GoogleMaps()))
    if options.get('enable_python') and not gemini_local_python:
        if 'tools' not in conf: conf['tools'] = []
        if _gemini_python_function_active:
            log_force(
                "Gemini: using local Python function because File creation "
                "is enabled on a pre-Gemini-3 model"
            )
        else:
            conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))
            _gemini_code_exec_active = True
            # Agentic View runs Python server-side; give it a longer
            # deadline so it doesn't hit 504 DEADLINE_EXCEEDED.
            conf['http_options'] = types.HttpOptions(timeout=_GEMINI_AGENTIC_TIMEOUT_MS)
            # The sandbox still has its own hard runtime limit (~30s) that
            # the request timeout cannot extend, so also guide the model to
            # write code that finishes within it (especially for image edits).
            # Both texts are the user's auto system prompt notices (each can be turned off or edited).
            code_exec_notices = []
            if _auto_notice_enabled("gemini_code_execution"):
                code_exec_notices.append(_auto_notice_text("gemini_code_execution"))
            if options.get('enable_file_creation') and _auto_notice_enabled("python_file_creation"):
                code_exec_notices.append(_auto_notice_text("python_file_creation"))
            if code_exec_notices:
                code_exec_guidance = "\n\n".join(code_exec_notices)
                if options.get('system_prompt'):
                    conf['system_instruction'] = (
                        f"{options.get('system_prompt')}\n\n{code_exec_guidance}"
                    )
                else:
                    conf['system_instruction'] = code_exec_guidance
    last_file_tool_error = None
    if _gemini_python_function_active:
        def _gemini_execute_python_tool(code: str) -> str:
            """併用時に使う、アプリ側の制限付きPython実行ツール。"""
            nonlocal full_res
            safe_output = _sanitize_python_sandbox_output(safe_execute_python(code))
            code_md = f"\n```python\n{code}\n```\n"
            output_md = f"\n**Output:**\n```\n{safe_output}\n```\n"
            full_res += code_md + output_md
            full_res += f"\n```pyexec\n{json.dumps({'code': code, 'output': safe_output})}\n```\n"
            pub("content", code_md)
            pub("content", output_md)
            pub(
                "python",
                {
                    "id": f"gem_function_py_{int(time.time()*1000)}_{os.urandom(3).hex()}",
                    "code": code,
                    "output": safe_output,
                },
            )
            return safe_output

        _gemini_execute_python_tool.__name__ = "execute_python"
        if 'tools' not in conf: conf['tools'] = []
        conf['tools'].append(_gemini_execute_python_tool)
    if options.get('enable_file_creation'):
        def _gemini_create_file_tool(
            filename: str,
            content: str,
            format: str = "",
        ) -> str:
            """テキスト・コード・Markdown・PDF・Word(docx)・Excel(xlsx)ファイルを作成し、ユーザーのファイルライブラリに保存します。保存後のURLを回答内のリンクとして提示してください。format は省略可能（拡張子から自動判定）。PDF/DOCX の content は Markdown 形式、XLSX の content は TSV（タブ区切り、1行目ヘッダー）です。添付済みの既存ファイルの編集を求められた場合は、新規作成せず edit_file を使用してください。"""
            nonlocal last_file_tool_error
            result = _execute_create_file_tool(
                user_id,
                {
                    "filename": filename,
                    "content": content,
                    "format": format,
                },
                user_config.get("enable_e2ee"),
            )
            if result.get("ok"):
                created_file_rel = f"{user_id}/{result['filename']}"
                if created_file_rel not in generated_images:
                    generated_images.append(created_file_rel)
                file_link_md = (
                    f"\n📄 **ファイルを作成しました:** [{result.get('display_name')}]({result.get('url')})\n"
                )
                nonlocal full_res
                full_res += file_link_md
                pub("content", file_link_md)
            else:
                last_file_tool_error = _create_file_tool_result_text(result)
            return _create_file_tool_result_text(result)

        _gemini_create_file_tool.__name__ = "create_file"
        if not _gemini_manual_function_tools:
            if 'tools' not in conf: conf['tools'] = []
            conf['tools'].append(_gemini_create_file_tool)

        def _gemini_edit_file_tool(
            source: str,
            content: Optional[str] = None,
            cell_edits: Optional[list[dict]] = None,
            paragraph_edits: Optional[list[dict]] = None,
            text_edits: Optional[list[dict]] = None,
        ) -> str:
            """会話に添付された既存のファイルや過去に作成・編集されたファイル（source に添付ファイル名またはURL）を編集し、編集後のファイルをユーザーのファイルライブラリに保存します。元ファイルの書式・構造・内容を保ったまま変更してください。Excel(xlsx/xlsm)は cell_edits で変更するセルのリストを指定します。各要素は { cell: セル番地（添付時に表示される列名ヘッダー A, B, C... を参照、例 B5）, value: 新しい値, sheet: シート名(省略時は最初のシート), style: 任意の新しい書式 } です。style は指定した項目だけを上書きします: fill: { color: '#RRGGBB' 塗りつぶし色, fillType: 'solid'(既定)/'none'(解除) }, font: { bold, italic, strikethrough, underline: 'none'/'single'/'double', color: 文字色, size: サイズpt, name: フォント名 }, border: { style: 'none'/'thin'/'medium'/'thick'/'dashed'/'dotted'/'double'（四辺）, color: 罫線色, left/right/top/bottom: { style, color } で辺ごとに上書き }, alignment: { horizontal: 'left'/'center'/'right'/'justify', vertical: 'top'/'center'/'bottom', wrapText }, numberFormat: 表示形式コード。Word(docx)は paragraph_edits で変更する段落のリストを指定します（元の書式を維持し、編集後に PDF 版も生成されます）。各要素は { paragraph: [N]番号または段落テキスト(部分一致), text: 新しいテキスト(省略可), style: { font: { bold, italic, strikethrough, underline: 'none'/'single'/'double', color: 文字色, size: サイズpt, name: フォント名, highlight: 'yellow'/'green'/'cyan'/'magenta'/'red'/'blue'/'gray'/'none' }, alignment: 'left'/'center'/'right'/'justify'/'default' } } です。PDF は text_edits で { find: 検索文字列, replace: 置換文字列, page: ページ番号(省略可) } のリストを指定すると、レイアウトを保ったままベストエフォートで置換します（見つからない場合はエラー）。テキスト系は content に編集後の全文、PDF/DOCX の全文置き換えは content に Markdown 形式の全文を指定します。"""
            nonlocal last_file_tool_error
            result = _execute_edit_file_tool(
                user_id,
                {
                    "source": source,
                    "content": content,
                    "cell_edits": cell_edits,
                    "paragraph_edits": paragraph_edits,
                    "text_edits": text_edits,
                },
                user_config.get("enable_e2ee"),
                loaded_files=loaded_files,
                history=history,
                thread_id=thread_id,
            )
            if result.get("ok"):
                edited_file_rel = f"{user_id}/{result['filename']}"
                if edited_file_rel not in generated_images:
                    generated_images.append(edited_file_rel)
                file_link_md = (
                    f"\n📄 **ファイルを編集しました:** [{result.get('display_name')}]({result.get('url')})\n"
                )
                nonlocal full_res
                full_res += file_link_md
                pub("content", file_link_md)
            else:
                last_file_tool_error = _create_file_tool_result_text(result, "edit_file", "編集")
            return _create_file_tool_result_text(result, "edit_file", "編集")

        _gemini_edit_file_tool.__name__ = "edit_file"
        if not _gemini_manual_function_tools:
            if 'tools' not in conf: conf['tools'] = []
            conf['tools'].append(_gemini_edit_file_tool)

        if _gemini_manual_function_tools:
            create_schema = _build_create_file_tool_schema()['function']
            edit_schema = _build_edit_file_tool_schema()['function']

            def _strip_additional_properties(_node):
                # Gemini API は additionalProperties を
                # additional_properties として送ると 400 で拒否する。
                if isinstance(_node, dict):
                    return {
                        _k: _strip_additional_properties(_v)
                        for _k, _v in _node.items()
                        if _k not in ("additionalProperties", "additional_properties")
                    }
                if isinstance(_node, list):
                    return [_strip_additional_properties(_v) for _v in _node]
                return _node

            if 'tools' not in conf: conf['tools'] = []
            _gemini_manual_decls = [
                {
                    "name": create_schema["name"],
                    "description": create_schema.get("description", ""),
                    "parameters": _strip_additional_properties(create_schema.get("parameters", {})),
                },
                {
                    "name": edit_schema["name"],
                    "description": edit_schema.get("description", ""),
                    "parameters": _strip_additional_properties(edit_schema.get("parameters", {})),
                },
            ]
            if _skills_tool_enabled:
                _skill_decl = build_skill_tool_definition()
                _gemini_manual_decls.append({
                    "name": _skill_decl["name"],
                    "description": _skill_decl["description"],
                    "parameters": _strip_additional_properties(_skill_decl["parameters"]),
                })
            conf['tools'].append(types.Tool(function_declarations=_gemini_manual_decls))

    # MCP外部ツール（google-genai の Automatic Function Calling 用の
    # callable として追加。function_declarations は AFC を無効化するため使わない）
    _mcp_runtime = None
    _gemini_mcp_callables = []
    if not gemini_local_python:
        try:
            _mcp_runtime = _ensure_mcp_env()
        except Exception:
            _mcp_runtime = None
    if _mcp_runtime is not None:
        try:
            from mcp_service.gemini_tools import build_gemini_mcp_tools

            def _on_gemini_mcp_result(_text, _mout, _iname):
                nonlocal full_res
                try:
                    if _mout.get("ok"):
                        _md = f"\n\n> 🔧 **MCPツール実行:** `{_iname}` を実行しました。\n"
                        full_res += _md
                        pub("content", _md)
                    elif _mout.get("rejected"):
                        _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{_iname}`\n"
                        full_res += _md
                        pub("content", _md)
                except Exception:
                    pass

            _gemini_mcp_callables = build_gemini_mcp_tools(_mcp_runtime, on_result=_on_gemini_mcp_result)
            if _gemini_mcp_callables:
                if 'tools' not in conf: conf['tools'] = []
                conf['tools'].extend(_gemini_mcp_callables)
                log_force(f"Gemini MCP tools attached: {len(_gemini_mcp_callables)}")
        except Exception as _mcp_e:
            log_force(f"Gemini MCP tool attach failed: {_mcp_e}")
    if _skills_tool_enabled and not _gemini_manual_function_tools:
        try:
            if 'tools' not in conf: conf['tools'] = []
            conf['tools'].append(build_gemini_skill_tool(
                _skills_for_tools,
                on_result=lambda _text, _meta: _record_skill_meta(_meta),
            ))
            log_force(f"Gemini skill tool attached: {len(_skills_for_tools)}")
        except Exception as _skill_tool_exc:
            log_force(f"Gemini skill tool attach failed: {_skill_tool_exc}")
    # Gemini 3 は、組み込みツール（検索・URL Context・マップ・コード実行）と
    # 関数ツール（File・MCP）を同時に渡すとき、この指定がないと 400 を返す。
    # 応答の toolCall / toolResponse パートは、SDK の自動関数呼び出しと
    # _gemini_manual_function_stream が次のラウンドへそのまま返す。
    if is_gemini_3 and _gemini_combines_builtin_and_function_tools(conf.get('tools')):
        conf['tool_config'] = types.ToolConfig(include_server_side_tool_invocations=True)
    if options.get('system_prompt') and 'system_instruction' not in conf:
        conf['system_instruction'] = options.get('system_prompt')

    contents = []
    history_img_seen = set()
    history_img_bytes = 0
    for m in history:
        parts = []
        if m.get('signature'):
            sig_val = m.get('signature')
            sig_list = None
            if isinstance(sig_val, str):
                try:
                    parsed = json.loads(sig_val)
                    if isinstance(parsed, list):
                        sig_list = parsed
                    elif isinstance(parsed, str):
                        sig_list = [parsed]
                except Exception:
                    sig_list = [sig_val]
            elif isinstance(sig_val, list):
                sig_list = sig_val
            if sig_list:
                for s in sig_list:
                    try:
                        parts.append(types.Part(thought_signature=base64.b64decode(s)))
                    except Exception:
                        pass
        if m['content']:
            parts.append(types.Part(text=m['content']))
        if m.get('image_url'):
            try:
                msg_images, history_img_bytes = _load_message_history_images(
                    m.get('image_url'),
                    seen=history_img_seen,
                    total_bytes=history_img_bytes,
                    include_only_images=True
                )
                for msg_img in msg_images:
                    parts.append(types.Part.from_bytes(data=msg_img['bytes'], mime_type=msg_img['mime']))
            except: pass
        if parts: contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

    curr_parts = []
    if final_message_text and str(final_message_text).strip():
        curr_parts.append(types.Part(text=final_message_text))
    media_inline_limit = 20 * 1024 * 1024  # 20MiB limit for inline audio
    pending_file_error = None

    # region: File parts, uploads, and cache
    # The upload/cache helpers live in `background_stream_gemini_files.py`.
    # Only the values that belong to this request are bound here.
    def _gemini_get_cached_part(rel_path, mime, size=None, mtime=None, label=""):
        return _gemini_get_cached_part_impl(
            user_id, gemini_files_api_enabled, g_client,
            rel_path, mime, size=size, mtime=mtime, label=label,
        )

    def _gemini_upload_with_retry(data, mime, suffix, rel_path, label="", display_name=None):
        return _gemini_upload_with_retry_impl(
            user_id, gemini_files_api_enabled, g_client,
            data, mime, suffix, rel_path, label=label, display_name=display_name,
        )

    current_image_names = []
    inline_image_bytes = 0
    for fi in loaded_files:
        if fi.get('is_pdf') and supports_pdf_inputs and fi.get('bytes'):
            try:
                pdf_bytes = fi['bytes']
                pdf_mime = fi.get('mime') or 'application/pdf'
                pdf_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.pdf')
                rel_path = fi.get('path') or fi.get('name') or pdf_name
                if len(pdf_bytes) <= media_inline_limit:
                    curr_parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type=pdf_mime))
                else:
                    cached_part = _gemini_get_cached_part(
                        rel_path,
                        pdf_mime,
                        size=fi.get('size'),
                        mtime=fi.get('mtime'),
                        label=f"pdf:{pdf_name}"
                    )
                    if cached_part:
                        curr_parts.append(cached_part)
                    else:
                        up, up_state, up_err = _gemini_upload_with_retry(
                            pdf_bytes,
                            pdf_mime,
                            os.path.splitext(pdf_name)[1] or '.pdf',
                            rel_path,
                            label=f"pdf:{pdf_name}",
                            display_name=pdf_name
                        )
                        if not up or up_err:
                            pending_file_error = f"PDF({pdf_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                            break
                        file_uri = _gemini_file_uri(up)
                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or pdf_mime
                        part = _make_gemini_uri_part(file_uri, up_mime)
                        if part:
                            curr_parts.append(part)
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "gemini",
                                file_id=_gemini_file_name(up),
                                file_uri=file_uri,
                                state=up_state or "ACTIVE",
                                last_error=None,
                                size_bytes=fi.get('size'),
                                mtime=fi.get('mtime'),
                                mime_type=up_mime,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        else:
                            pending_file_error = f"PDF({pdf_name})参照の生成に失敗しました。再送してください。"
                            break
            except Exception as e:
                log_force(f"Gemini PDF upload failed: {e}")
                if fi.get('text'):
                    curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
            continue
        if fi.get('is_docx') and supports_docx_inputs and fi.get('bytes'):
            try:
                docx_bytes = fi['bytes']
                docx_mime = fi.get('mime') or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                docx_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.docx')
                rel_path = fi.get('path') or fi.get('name') or docx_name
                if len(docx_bytes) <= media_inline_limit:
                    curr_parts.append(types.Part.from_bytes(data=docx_bytes, mime_type=docx_mime))
                else:
                    cached_part = _gemini_get_cached_part(
                        rel_path,
                        docx_mime,
                        size=fi.get('size'),
                        mtime=fi.get('mtime'),
                        label=f"docx:{docx_name}"
                    )
                    if cached_part:
                        curr_parts.append(cached_part)
                    else:
                        up, up_state, up_err = _gemini_upload_with_retry(
                            docx_bytes,
                            docx_mime,
                            os.path.splitext(docx_name)[1] or '.docx',
                            rel_path,
                            label=f"docx:{docx_name}"
                        )
                        if not up or up_err:
                            pending_file_error = f"Word({docx_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                            break
                        file_uri = _gemini_file_uri(up)
                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or docx_mime
                        part = _make_gemini_uri_part(file_uri, up_mime)
                        if part:
                            curr_parts.append(part)
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "gemini",
                                file_id=_gemini_file_name(up),
                                file_uri=file_uri,
                                state=up_state or "ACTIVE",
                                last_error=None,
                                size_bytes=fi.get('size'),
                                mtime=fi.get('mtime'),
                                mime_type=up_mime,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        else:
                            pending_file_error = f"Word({docx_name})参照の生成に失敗しました。再送してください。"
                            break
            except Exception as e:
                log_force(f"Gemini docx upload failed: {e}")
                if fi.get('text'):
                    curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
            continue
        if fi.get('is_text') and supports_text_file_inputs and fi.get('bytes'):
            attached = False
            try:
                txt_bytes = fi['bytes']
                txt_mime = fi.get('mime') or 'text/plain'
                txt_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.txt')
                rel_path = fi.get('path') or fi.get('name') or txt_name
                if len(txt_bytes) <= media_inline_limit:
                    curr_parts.append(types.Part.from_bytes(data=txt_bytes, mime_type=txt_mime))
                else:
                    cached_part = _gemini_get_cached_part(
                        rel_path,
                        txt_mime,
                        size=fi.get('size'),
                        mtime=fi.get('mtime'),
                        label=f"text:{txt_name}"
                    )
                    if cached_part:
                        curr_parts.append(cached_part)
                    else:
                        up, up_state, up_err = _gemini_upload_with_retry(
                            txt_bytes,
                            txt_mime,
                            os.path.splitext(txt_name)[1] or '.txt',
                            rel_path,
                            label=f"text:{txt_name}"
                        )
                        if not up or up_err:
                            pending_file_error = f"テキスト({txt_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                            break
                        file_uri = _gemini_file_uri(up)
                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or txt_mime
                        part = _make_gemini_uri_part(file_uri, up_mime)
                        if part:
                            curr_parts.append(part)
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "gemini",
                                file_id=_gemini_file_name(up),
                                file_uri=file_uri,
                                state=up_state or "ACTIVE",
                                last_error=None,
                                size_bytes=fi.get('size'),
                                mtime=fi.get('mtime'),
                                mime_type=up_mime,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        else:
                            pending_file_error = f"テキスト({txt_name})参照の生成に失敗しました。再送してください。"
                            break
                attached = True
            except Exception as e:
                log_force(f"Gemini text file upload failed: {e}")
            if attached:
                continue
        if fi.get('text'):
            curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
            continue
        if not fi.get('bytes'):
            continue
        mime = (fi.get('mime') or 'application/octet-stream').lower()
        if mime.startswith('image/'):
            img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"image{len(current_image_names) + 1}")
            rel_path = fi.get('path') or fi.get('name') or img_name
            img_size = len(fi['bytes'])

            attached = False
            # Small images are latency-sensitive and fit safely in the
            # GenerateContent payload. Avoid a separate Files API round trip.
            if inline_image_bytes + img_size <= _GEMINI_INLINE_IMAGE_MAX_BYTES:
                curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                inline_image_bytes += img_size
                attached = True
            elif gemini_files_api_enabled:
                try:
                    cached_part = _gemini_get_cached_part(
                        rel_path,
                        fi['mime'],
                        size=img_size,
                        mtime=fi.get('mtime'),
                        label=f"image:{img_name}"
                    )
                    if cached_part:
                        curr_parts.append(cached_part)
                        attached = True
                    else:
                        up, up_state, up_err = _gemini_upload_with_retry(
                            fi['bytes'],
                            fi['mime'],
                            os.path.splitext(img_name)[1] or '.png',
                            rel_path,
                            label=f"image:{img_name}",
                            display_name=img_name
                        )
                        if up and not up_err:
                            file_uri = _gemini_file_uri(up)
                            up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or fi['mime']
                            part = _make_gemini_uri_part(file_uri, up_mime)
                            if part:
                                curr_parts.append(part)
                                _upsert_file_cache(
                                    user_id,
                                    rel_path,
                                    "gemini",
                                    file_id=_gemini_file_name(up),
                                    file_uri=file_uri,
                                    state=up_state or "ACTIVE",
                                    last_error=None,
                                    size_bytes=img_size,
                                    mtime=fi.get('mtime'),
                                    mime_type=up_mime,
                                    last_checked_at=datetime.utcnow()
                                )
                                safe_db_commit()
                                attached = True
                except Exception as e:
                    log_force(f"Gemini image upload failed: {e}")

            if not attached:
                curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                inline_image_bytes += img_size
            img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
            current_image_names.append(os.path.basename(str(img_label)))
            continue
        if mime.startswith('audio/'):
            try:
                audio_bytes, audio_mime, audio_name = _normalize_gemini_audio(fi['bytes'], fi.get('mime') or mime, fi.get('send_name') or fi.get('name') or "")
                rel_path = fi.get('path') or fi.get('name') or audio_name
                audio_size = len(audio_bytes) if audio_bytes is not None else fi.get('size')
                if len(audio_bytes) <= media_inline_limit:
                    curr_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime))
                else:
                    cached_part = _gemini_get_cached_part(
                        rel_path,
                        audio_mime,
                        size=audio_size,
                        mtime=fi.get('mtime'),
                        label=f"audio:{audio_name}"
                    )
                    if cached_part:
                        curr_parts.append(cached_part)
                    else:
                        up, up_state, up_err = _gemini_upload_with_retry(
                            audio_bytes,
                            audio_mime,
                            os.path.splitext(audio_name or '')[1] or '.bin',
                            rel_path,
                            label=f"audio:{audio_name}",
                            display_name=audio_name
                        )
                        if not up or up_err:
                            pending_file_error = f"音声({audio_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                            break
                        file_uri = _gemini_file_uri(up)
                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or audio_mime
                        part = _make_gemini_uri_part(file_uri, up_mime)
                        if part:
                            curr_parts.append(part)
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "gemini",
                                file_id=_gemini_file_name(up),
                                file_uri=file_uri,
                                state=up_state or "ACTIVE",
                                last_error=None,
                                size_bytes=fi.get('size'),
                                mtime=fi.get('mtime'),
                                mime_type=up_mime,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        else:
                            pending_file_error = f"音声({audio_name})参照の生成に失敗しました。再送してください。"
                            break
            except Exception as e:
                log_force(f"Gemini audio upload failed: {e}")
                pending_file_error = f"音声({fi.get('send_name') or fi.get('name') or 'file'})のアップロードに失敗しました。再送してください。"
                break
            continue
        if mime.startswith('video/'):
            try:
                video_bytes = fi['bytes']
                video_mime = fi.get('mime') or mime
                video_name = fi.get('send_name') or fi.get('name') or "video"
                video_size = len(video_bytes) if video_bytes is not None else fi.get('size')
                rel_path = fi.get('path') or fi.get('name') or video_name
                cached_part = _gemini_get_cached_part(
                    rel_path,
                    video_mime,
                    size=video_size,
                    mtime=fi.get('mtime'),
                    label=f"video:{video_name}"
                )
                if cached_part:
                    curr_parts.append(cached_part)
                else:
                    up, up_state, up_err = _gemini_upload_with_retry(
                        video_bytes,
                        video_mime,
                        os.path.splitext(video_name or '')[1] or '.bin',
                        rel_path,
                        label=f"video:{video_name}",
                        display_name=video_name
                    )
                    if not up or up_err:
                        pending_file_error = f"動画({video_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                        break
                    file_uri = _gemini_file_uri(up)
                    up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or video_mime
                    part = _make_gemini_uri_part(file_uri, up_mime)
                    if part:
                        curr_parts.append(part)
                        _upsert_file_cache(
                            user_id,
                            rel_path,
                            "gemini",
                            file_id=_gemini_file_name(up),
                            file_uri=file_uri,
                            state=up_state or "ACTIVE",
                            last_error=None,
                            size_bytes=video_size,
                            mtime=fi.get('mtime'),
                            mime_type=up_mime,
                            last_checked_at=datetime.utcnow()
                        )
                        safe_db_commit()
                    else:
                        pending_file_error = f"動画({video_name})参照の生成に失敗しました。再送してください。"
                        break
            except Exception as e:
                log_force(f"Gemini video upload failed: {e}")
                pending_file_error = f"動画({fi.get('send_name') or fi.get('name') or 'file'})のアップロードに失敗しました。再送してください。"
                break
            continue
        # Skip unsupported binary inputs for Gemini text models
        pass


    # region: Batch submission and grounding helpers
    # URL/grounding の抽出はここで行い、引用の差し込み・思考テキストの解析は
    # `background_stream_gemini_citations.py` の関数を使う。
    if pending_file_error:
        if batch_mode:
            _persist_batch_failure(pending_file_error)
        else:
            pub("error", pending_file_error)
        return BACKGROUND_STREAM_EARLY_RETURN

    contents.append(types.Content(role='user', parts=curr_parts))

    if batch_mode:
        _mark_provider_request_started()
        try:
            _submit_gemini_batch(rm, contents, conf)
        except Exception as batch_error:
            logger.exception('Gemini Batch submission failed')
            _persist_batch_failure(str(batch_error))
        return BACKGROUND_STREAM_EARLY_RETURN

    grounding_chunks = None
    grounding_supports = None
    url_context_chunks = None
    # 出典の位置（UTF-8バイト）を求めるための、モデルが返したテキストだけの本文
    grounding_model_text = []

    def _collect_grounding(gm):
        nonlocal grounding_chunks, grounding_supports
        if not gm:
            return
        g_chunks = getattr(gm, 'grounding_chunks', None) or getattr(gm, 'groundingChunks', None) or []
        if g_chunks and grounding_chunks is None:
            grounding_chunks = g_chunks
        g_supports = getattr(gm, 'grounding_supports', None) or getattr(gm, 'groundingSupports', None) or []
        if g_supports and grounding_supports is None:
            grounding_supports = g_supports

    def _collect_url_context(ucm):
        nonlocal url_context_chunks
        if not ucm:
            return
        u_metadata = getattr(ucm, 'url_metadata', None) or getattr(ucm, 'urlMetadata', None) or []
        if u_metadata and url_context_chunks is None:
            url_context_chunks = u_metadata

    # region: Manual function and MCP aware stream
    def _gemini_manual_function_stream():
        """Gemini 3のツール併用でFileのfunction responseを循環させる。

        各ラウンドをストリームで受け取り、チャンクは届いた順にそのまま
        返す。function callはラウンドの終わりにまとめて実行する。
        """
        request_contents = list(contents)
        for _tool_round in range(8):
            _mark_provider_request_started()
            model_parts = []
            calls = []
            for chunk in g_client.models.generate_content_stream(
                model=rm,
                contents=request_contents,
                config=types.GenerateContentConfig(**conf),
            ):
                yield chunk
                candidates = getattr(chunk, "candidates", None) or []
                chunk_content = getattr(candidates[0], "content", None) if candidates else None
                for part in (getattr(chunk_content, "parts", None) or []):
                    model_parts.append(part)
                    function_call = getattr(part, "function_call", None)
                    if not function_call and isinstance(part, dict):
                        function_call = part.get("function_call") or part.get("functionCall")
                    if not function_call:
                        continue
                    if isinstance(function_call, dict):
                        call_name = function_call.get("name")
                        call_args = function_call.get("args") or function_call.get("arguments") or {}
                        call_id = function_call.get("id")
                    else:
                        call_name = getattr(function_call, "name", None)
                        call_args = getattr(function_call, "args", None) or getattr(function_call, "arguments", None) or {}
                        call_id = getattr(function_call, "id", None)
                    if call_name in ("create_file", "edit_file") or (
                        call_name == SKILL_TOOL_NAME and _skills_tool_enabled
                    ):
                        calls.append((call_name, call_args, call_id))
            if not calls:
                break

            request_contents.append(types.Content(role="model", parts=model_parts))
            response_parts = []
            for call_name, call_args, call_id in calls:
                try:
                    if not isinstance(call_args, dict):
                        raise ValueError("function arguments must be an object")
                    if call_name == "create_file":
                        result_text = _gemini_create_file_tool(**call_args)
                    elif call_name == SKILL_TOOL_NAME:
                        result_text = _execute_skill_tool(
                            str((call_args or {}).get("name") or "")
                        )
                    else:
                        result_text = _gemini_edit_file_tool(**call_args)
                except Exception as tool_exc:
                    result_text = f"Error executing {call_name}: {tool_exc}"
                function_response = types.FunctionResponse(
                    name=call_name,
                    response={"response": str(result_text or "Tool executed.")},
                    id=call_id,
                )
                response_parts.append(types.Part(function_response=function_response))
            request_contents.append(types.Content(role="user", parts=response_parts))

    # MCP/Fileのcallableツールは、SDKのAutomatic Function Calling
    # (generate_content_stream)に任せる。function callを含むラウンドは
    # SDKがツールを実行して次のリクエストへ進み、それ以外のチャンクは
    # 届いた順にストリームされる。
    def _gemini_mcp_aware_stream():
        if _gemini_manual_function_tools:
            return _gemini_manual_function_stream()
        return g_client.models.generate_content_stream(
            model=rm,
            contents=contents,
            config=types.GenerateContentConfig(**conf),
        )

    _mark_provider_request_started()
    log_force(f"STREAM-TRACE: Gemini stream starting for {job_id} model={rm}")
    # The streaming generator performs the HTTP request only when the
    # first chunk is pulled. Google intermittently returns 504
    # DEADLINE_EXCEEDED on the initial response of code-execution
    # requests before any content is generated; a plain retry usually
    # succeeds. Retry the first-chunk pull (nothing has been streamed
    # yet, so no user-visible content is duplicated).
    # region: Stream loop, citations, and final answer
    _gemini_stream_attempts = 0
    while True:
        try:
            _gemini_stream = _gemini_mcp_aware_stream()
            _gemini_stream_iter = iter(_gemini_stream)
            _gemini_first_chunk = next(_gemini_stream_iter)
            break
        except Exception as _stream_exc:
            _is_deadline = (
                "504" in str(_stream_exc) or "DEADLINE_EXCEEDED" in str(_stream_exc)
            )
            if (
                _gemini_code_exec_active
                and _is_deadline
                and _gemini_stream_attempts < _GEMINI_STREAM_DEADLINE_RETRIES
                and not check_stop()
            ):
                _gemini_stream_attempts += 1
                log_force(
                    f"STREAM-TRACE: Gemini 504 before first chunk, retry "
                    f"{_gemini_stream_attempts}/{_GEMINI_STREAM_DEADLINE_RETRIES} for {job_id}"
                )
                time.sleep(2)
                continue
            raise
    current_py_id = None
    current_py_code = None
    final_usage_metadata = None
    log_force(f"STREAM-TRACE: Gemini stream loop start for {job_id}")
    for chunk in itertools.chain([_gemini_first_chunk], _gemini_stream_iter):
        _latency_mark_once(job_id, "provider_first_chunk_ms")
        if check_stop():
            log_force(f"STREAM-TRACE: Gemini stream breaking due to stop for {job_id}")
            break
        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
            final_usage_metadata = chunk.usage_metadata

        if hasattr(chunk, 'candidates') and chunk.candidates:
            for cand in chunk.candidates:
                gm = getattr(cand, 'grounding_metadata', None)
                _collect_grounding(gm)
                ucm = getattr(cand, 'url_context_metadata', None)
                _collect_url_context(ucm)

                parts = getattr(getattr(cand, 'content', None), 'parts', None) or []
                for part in parts:
                    if hasattr(part, 'thought_signature') and part.thought_signature:
                        signature_parts.append(base64.b64encode(part.thought_signature).decode('utf-8'))

                    if hasattr(part, 'thought') and part.thought:
                        t_text = _extract_gemini_thought_text(part)
                        if t_text:
                            thought_accumulated += t_text
                            pub("thought", t_text)
                        continue

                    if hasattr(part, 'executable_code') and part.executable_code:
                        c_txt = f"\n```python\n{part.executable_code.code}\n```\n"
                        full_res += c_txt
                        pub("content", c_txt)
                        current_py_id = f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                        current_py_code = part.executable_code.code
                        # Reset rather than extend: the stream delivers the
                        # executable_code part before that turn's inline_data
                        # images, so only this execution's save-names should be
                        # eligible for the images that follow it.
                        pending_sandbox_filenames = _extract_sandbox_image_filenames(part.executable_code.code)
                        pub("python", {"id": current_py_id, "code": part.executable_code.code})
                        continue

                    if hasattr(part, 'code_execution_result') and part.code_execution_result:
                        r_txt = f"\n**Output:**\n```\n{part.code_execution_result.output}\n```\n"
                        full_res += r_txt
                        pub("content", r_txt)
                        py_id = current_py_id or f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                        pub("python", {"id": py_id, "output": part.code_execution_result.output})
                        py_payload = {"code": current_py_code or "", "output": part.code_execution_result.output}
                        full_res += f"\n```pyexec\n{json.dumps(py_payload)}\n```\n"
                        continue

                    if hasattr(part, 'inline_data') and part.inline_data:
                        try:
                            mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                            img_data = part.inline_data.data
                            img_data, ext = _prepare_agentic_image_bytes(img_data, mime)
                            image_digest = hashlib.sha256(img_data).hexdigest()
                            if image_digest in agentic_image_digests:
                                continue
                            agentic_image_digests.add(image_digest)

                            def _new_agentic_filename():
                                return (
                                    f"agentic_{int(time.time() * 1000)}_"
                                    f"{os.urandom(4).hex()}.{ext}"
                                )

                            fn2, saved_agentic_url = _save_user_generated_bytes_verified(
                                user_id,
                                img_data,
                                _new_agentic_filename,
                                user_config.get('enable_e2ee'),
                            )
                            log_force(
                                f"Agentic image saved: {fn2} ({len(img_data)} bytes)"
                            )
                            agentic_saved_urls.append(saved_agentic_url)

                            # Associate the sandbox filename with the saved URL when
                            # known so a bare ![alt](result.png) reference in the
                            # model's final answer is rewritten to the served image.
                            sandbox_name = None
                            display_name = getattr(part.inline_data, "display_name", None)
                            if display_name:
                                dname = os.path.basename(str(display_name)).strip()
                                if _sandbox_ref_basename(dname):
                                    sandbox_name = dname
                            if not sandbox_name and pending_sandbox_filenames:
                                sandbox_name = pending_sandbox_filenames.pop(0)
                            elif sandbox_name and pending_sandbox_filenames:
                                # A display_name was used; drop a matching pending
                                # entry so the fallback queue does not drift.
                                lower = sandbox_name.lower()
                                if lower in pending_sandbox_filenames:
                                    pending_sandbox_filenames.remove(lower)
                            if sandbox_name:
                                agentic_filename_url_map[sandbox_name.lower()] = saved_agentic_url

                            img_md = f"\n![Agentic View]({saved_agentic_url})\n"
                            full_res += img_md
                            pub("content", img_md)
                        except Exception as e:
                            log_force(f"Agentic Vision Image Error: {e}")
                            # Never leave a silent hole in the answer: tell the
                            # user the image was generated but could not be kept,
                            # instead of streaming a URL that points nowhere.
                            err_md = "\n（※生成画像の保存に失敗したため、画像を表示できません）\n"
                            full_res += err_md
                            pub("content", err_md)
                        continue

                    if hasattr(part, 'text') and part.text:
                        t_delta = part.text
                        grounding_model_text.append(t_delta)
                        if prior_sandbox_names is None and _has_bare_sandbox_image_ref(sandbox_text_buffer[0] + t_delta):
                            # The model may cite a file an earlier turn's code saved.
                            prior_sandbox_names = _prior_sandbox_image_names(thread_id)
                            for _prior_name, _prior_url in prior_sandbox_names.items():
                                if _prior_url:
                                    agentic_filename_url_map.setdefault(_prior_name, _prior_url)
                        rewritten = _rewrite_streamed_sandbox_refs(
                            t_delta,
                            sandbox_text_buffer,
                            agentic_saved_urls,
                            agentic_consumed_urls,
                            agentic_filename_url_map,
                        )
                        if rewritten:
                            full_res += rewritten
                            pub("content", rewritten)

        # Fallback to chunk.text if parts didn't cover it (unlikely but safe)
        # but be careful not to double-publish.


    full_res = background_stream_gemini_tail(
        local,
        full_res,
        agentic_saved_urls,
        agentic_consumed_urls,
        agentic_filename_url_map,
        sandbox_text_buffer,
        auto_enable_url_context,
        gemini_local_python,
        grounding_chunks,
        grounding_supports,
        grounding_model_text,
        url_context_chunks,
        last_file_tool_error,
        prior_sandbox_names,
        thread_id,
        options,
        _chunk_grounding_info,
        _add_gemini_citations,
        _url_retrieval_status_label,
    )

    return (full_res, thought_accumulated)
    # endregion
