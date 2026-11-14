# --- AI Settings Prompt (tool use in settings modal) ---

def _build_ai_settings_tool_schema():
    """Return tool schemas for updating settings or inspecting current values."""
    # Common property descriptions (Japanese for better JP prompt understanding)
    props = {
        "default_model": {"type": "string", "description": "既定モデルID", "enum": sorted(ALL_VALID_MODEL_IDS)},
        "default_enable_search": {"type": "boolean", "description": "Searchツールの既定ON/OFF"},
        "default_gemini_free_tier": {"type": "boolean", "description": "Gemini無料枠APIを既定で使うか（オンで無料枠キー、オフで有料枠キー）"},
        "default_gemini_flex": {"type": "boolean", "description": "Gemini Flex推論（低優先度・標準の50%料金）を既定で使うか"},
        "default_enable_url_context": {"type": "boolean", "description": "URLs (URLコンテキスト) の既定ON/OFF"},
        "default_enable_maps": {"type": "boolean", "description": "Maps (Google Maps grounding) の既定ON/OFF"},
        "default_enable_python": {"type": "boolean", "description": "Python実行ツールの既定ON/OFF"},
        "default_enable_file_creation": {"type": "boolean", "description": "ファイル作成ツールの既定ON/OFF"},
        "default_enable_thinking": {"type": "boolean", "description": "Thinking (拡張思考) の既定ON/OFF"},
        "default_thinking_level": {"type": "string", "description": "Thinkingレベル", "enum": sorted(VALID_THINKING_LEVELS)},
        "default_thinking_budget": {"type": "integer", "description": "Thinking budget (トークン数, 例: 4096)"},
        "default_reasoning_effort": {"type": "string", "description": "Reasoning effort", "enum": sorted(VALID_REASONING_EFFORTS)},
        "default_enable_system_prompt": {"type": "boolean", "description": "既定でシステムプロンプト(ユーザー定義)を使用するか"},
        "default_enable_mcp": {"type": "boolean", "description": "MCP（外部ツール接続）の既定ON/OFF（プロンプトバーのスイッチに反映）"},
        "default_safety_setting": {"type": "string", "description": "安全設定", "enum": sorted(VALID_SAFETY_SETTINGS)},
        "system_prompt": {"type": "string", "description": "ユーザー個別システムプロンプト本文 (自然言語で詳細指示可)"},
        "system_prompt_enabled": {"type": "boolean", "description": "ユーザー個別システムプロンプトのON/OFF"},
        "apply_global_system_prompt": {"type": "boolean", "description": "全体システムプロンプトを適用するか (ユーザー設定)"},
        "apply_auto_system_prompt_notices": {"type": "boolean", "description": "自動注入システムプロンプト (Python/Search等) を適用するか"},
        "auto_system_prompt_notices_config": {"type": "object", "description": "自動注入の種類別ON/OFF設定 (JSONオブジェクト)"},
        "mic_transcribe_mode": {"type": "string", "description": "マイク文字起こし方式: stt_api または llm"},
        "stt_model": {"type": "string", "description": "STTに使用するモデル名", "enum": sorted(VALID_STT_MODELS)},
        "llm_transcribe_prompt": {"type": "string", "description": "LLM文字起こし用の追加プロンプト"},
        "enter_to_send": {"type": "boolean", "description": "Enterキーで送信するか (Shift+Enterで改行)"},
        "use_sw_cache": {"type": "boolean", "description": "Service Workerキャッシュ使用"},
        "compact_prompt_mode": {"type": "boolean", "description": "プロンプトバーをコンパクト表示 (モデル選択のみ)"},
        "minimal_prompt_mode": {"type": "boolean", "description": "プロンプトバーをミニマル表示 (送信・プラスのみ、モデル選択は上部)"},
        "voice_studio_ui": {"type": "boolean", "description": "音声系モデルで専用の音声スタジオUIを使う (OFFで従来のマイクUI)"},
        "math_steps_enabled": {"type": "boolean", "description": "数学ステップ解説を有効にする (全体スイッチ、既定OFF)"},
        "math_steps_config": {"type": "object", "description": "数学ステップの詳細設定 (basics / level / step_size / diagrams / animation)"},
        "default_math_steps": {"type": "boolean", "description": "プロンプトバーで数学ステップを既定でONにするか"},
        "auto_search_on_links": {"type": "boolean", "description": "リンク検出時の自動検索"},
        "use_last_chat_settings": {"type": "boolean", "description": "前回の送信設定を継続使用"},
        "temp_chat_timeout_seconds": {"type": "integer", "description": "一時チャットの切断タイムアウト秒数 (30-86400)"},
        "theme_color": {"type": "string", "description": "テーマカラー (HEX, 例: #10a37f)"},
        "liquid_glass_enabled": {"type": "boolean", "description": "Liquid Glass表示モードのON/OFF"},
        "light_mode_enabled": {"type": "boolean", "description": "ライトモードのON/OFF"},
        "rich_paste_prompt_default": {"type": "string", "description": "リッチ貼り付け用カスタムプロンプト既定値"},
        "rich_paste_prompt_use_custom_default": {"type": "boolean", "description": "リッチ貼り付けでカスタムプロンプトを使用"},
        "enable_latency_metrics": {"type": "boolean", "description": "初回トークンレイテンシ計測を有効化"},
        "enable_client_debug_log": {"type": "boolean", "description": "クライアントデバッグログのサーバー送信を有効化"},
        "bot_detection_enabled": {"type": "boolean", "description": "Bot検出 (ユーザー単位) を有効化"},
        "skip_2fa_on_google_login": {"type": "boolean", "description": "Googleログイン時に2FAをスキップ"},
        "default_2fa_method": {"type": "string", "description": "既定2FA方式: totp または webauthn"},
    }
    # OpenAI / compat format
    openai_tool = {
        "type": "function",
        "function": {
            "name": "update_settings",
            "description": "ユーザーの指示に基づき、設定モーダルで管理可能な安全な設定項目のみを更新します。管理者専用項目やAPIキーなどは一切変更しません。指示が曖昧な場合は最も自然な1つの解釈を選んで適用してください。",
            "parameters": {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
            },
        },
    }
    inspect_description = (
        "設定を一切変更せず、ユーザーが確認・表示・質問した現在の設定項目を返します。"
        "特定項目の質問ではfieldsに該当項目だけを指定し、設定全体の確認ではallをtrueにしてください。"
    )
    openai_inspect_tool = {
        "type": "function",
        "function": {
            "name": "inspect_settings",
            "description": inspect_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(AI_SAFE_EDITABLE_FIELDS)},
                        "description": "確認する設定フィールド名",
                    },
                    "all": {"type": "boolean", "description": "安全な設定項目をすべて確認する場合はtrue"},
                },
                "additionalProperties": False,
            },
        },
    }
    # Gemini format (uses google.genai.types)
    # NOTE: We must NOT include additional_properties / additionalProperties here.
    # Google Gemini API rejects unknown fields like "additional_properties" in tool schemas (400 INVALID_ARGUMENT).
    # OpenAI side keeps additionalProperties: false (valid for OpenAI).
    gemini_func_decl = None
    gemini_inspect_decl = None
    try:
        from google.genai import types as gtypes
        gemini_properties = {}
        for k, v in props.items():
            t = str(v.get("type", "string")).upper()
            if t == "BOOLEAN":
                t = "BOOLEAN"
            elif t == "INTEGER":
                t = "INTEGER"
            elif t == "OBJECT":
                t = "OBJECT"
            else:
                t = "STRING"
            gemini_properties[k] = gtypes.Schema(
                type=t,
                description=v.get("description", ""),
                enum=v.get("enum"),
            )
        gemini_func_decl = gtypes.FunctionDeclaration(
            name="update_settings",
            description=openai_tool["function"]["description"],
            parameters=gtypes.Schema(
                type="OBJECT",
                properties=gemini_properties,
                # deliberately omit additional_properties
            ),
        )
        gemini_inspect_decl = gtypes.FunctionDeclaration(
            name="inspect_settings",
            description=inspect_description,
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "fields": gtypes.Schema(
                        type="ARRAY",
                        description="確認する設定フィールド名",
                        items=gtypes.Schema(type="STRING", enum=sorted(AI_SAFE_EDITABLE_FIELDS)),
                    ),
                    "all": gtypes.Schema(type="BOOLEAN", description="安全な設定項目をすべて確認する場合はtrue"),
                },
            ),
        )
    except Exception as e:
        log_force(f"AI-SETTINGS-GEMINI-SCHEMA-BUILD-ERR: {e}")
        gemini_func_decl = None
        gemini_inspect_decl = None
    return [openai_tool, openai_inspect_tool], [decl for decl in (gemini_func_decl, gemini_inspect_decl) if decl]


def _call_llm_for_settings_ai(model_id, instruction, current_settings_snapshot, user, conversation_history=None):
    """
    Perform a one-shot tool call to interpret the natural language instruction and return
    an action dict with the selected tool name and arguments. Returns (action or None, error_msg or None).
    Uses the user's API key resolution (same as chat) for the chosen model.
    Supports Gemini and OpenAI-compatible families (GPT, DeepSeek, Grok via compat).
    """
    if not model_id:
        return None, "モデルが指定されていません"
    mid = str(model_id).lower()
    is_gemini = "gemini" in mid
    is_claude = "claude" in mid
    is_grok = "grok" in mid and "gpt" not in mid
    is_deepseek = "deepseek" in mid
    is_kimi = "kimi" in mid

    if is_claude:
        return None, "Claude (Anthropic) モデルは現在の設定AIツール呼び出しで未対応です。Gemini または GPT/DeepSeek/Grok 系モデルを選択してください。"

    # Build messages for LLM
    sys_prompt = (
        "あなたはチャットアプリの『設定アシスタント』です。ユーザーが日本語または英語で自然言語の指示を出し、"
        "設定モーダル内の安全な項目（APIキー・管理者専用・パスワードなどを除く）について操作します。"
        "変更・有効化・無効化を求める指示ではupdate_settingsを、現在値の確認・表示・質問ではinspect_settingsを、必ずどちらか1回だけ呼び出してください。"
        "inspect_settingsは設定を変更しません。特定項目の確認ではfieldsに必要な項目だけを、設定全体の確認ではall=trueを指定してください。"
        "update_settingsには不要な項目を含めず、変更指示に合致するものだけを指定してください。曖昧さがある場合は最も妥当な1解釈を選んでください。"
        "ツール呼び出し以外の応答は一切返さず、必ずツールを使用してください。"
    )
    history_lines = []
    if isinstance(conversation_history, list):
        for item in conversation_history[-10:]:
            if not isinstance(item, dict):
                continue
            role = 'ユーザー' if item.get('role') == 'user' else '設定アシスタント'
            content = str(item.get('content') or '').strip()
            if content:
                history_lines.append(f"{role}: {content[:1200]}")
    history_context = '\n'.join(history_lines) if history_lines else '（なし）'
    user_content = (
        f"過去の設定会話（参考。今回の指示を最優先）:\n{history_context}\n\n"
        f"今回の指示: {instruction}\n\n"
        f"現在の設定 (参考・この値が最新):\n{json.dumps(current_settings_snapshot or {}, ensure_ascii=False, indent=2)}"
    )

    openai_tools, gemini_decls = _build_ai_settings_tool_schema()

    try:
        if is_gemini:
            # Gemini path (google.genai)
            api_key = _get_model_specific_api_key(user, model_id) or decrypt_val(user.gemini_api_key)
            if not api_key and _admin_env_fallback_enabled(user):
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None, "Gemini APIキーが設定されていません。設定モーダルでキーを入力するか、モデルを変更してください。"
            client = genai.Client(api_key=api_key)
            # Use top-level import (already present at module level)
            contents = [types.Part(text=user_content)]
            tools_arg = None
            cfg = None
            if gemini_decls:
                try:
                    from google.genai import types as gtypes
                    tools_arg = [gtypes.Tool(function_declarations=gemini_decls)]
                    cfg = gtypes.GenerateContentConfig(
                        tools=tools_arg,
                        system_instruction=sys_prompt,
                    )
                except Exception:
                    tools_arg = None
                    cfg = None

            # Modern SDK pattern (config=) already used everywhere else in app.py for generate_content
            gemini_fallback_used = None
            try:
                if cfg is not None:
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=cfg,
                    )
                else:
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=types.GenerateContentConfig(system_instruction=sys_prompt),
                    )
            except Exception as gemini_err:
                err_str = str(gemini_err)
                # Common Google 404 for retired preview models (e.g. gemini-3.1-flash-lite-preview)
                is_model_not_found = "404" in err_str or "NOT_FOUND" in err_str or "no longer available" in err_str.lower() or "model" in err_str.lower() and "not" in err_str.lower()
                if is_model_not_found:
                    # Temporary fallback to a currently stable model that supports function calling
                    fallback_model = "gemini-2.5-flash"
                    log_force(f"AI-SETTINGS-GEMINI-FALLBACK: original={model_id} -> {fallback_model} reason={err_str[:200]}")
                    gemini_fallback_used = fallback_model
                    try:
                        if cfg is not None:
                            resp = client.models.generate_content(
                                model=fallback_model,
                                contents=contents,
                                config=cfg,
                            )
                        else:
                            resp = client.models.generate_content(
                                model=fallback_model,
                                contents=contents,
                                config=types.GenerateContentConfig(system_instruction=sys_prompt),
                            )
                    except Exception as fb_err:
                        # Re-raise the fallback error with context
                        raise RuntimeError(f"Gemini fallback also failed ({fallback_model}): {fb_err}") from gemini_err
                else:
                    raise
            # Parse function call
            try:
                part = resp.candidates[0].content.parts[0]
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if hasattr(fc, "args") else {}
                    return {"action": getattr(fc, "name", "update_settings"), "arguments": args}, None
            except Exception as parse_e:
                log_force(f"AI-SETTINGS-GEMINI-PARSE-ERR: {parse_e} resp={getattr(resp,'text',None)}")
                return None, "Geminiからのツール呼び出し解析に失敗しました。"
            # fallback if no tool call
            text = getattr(resp, "text", "") or ""
            return None, f"モデルがツール呼び出しを行いませんでした: {text[:200]}"
        else:
            # OpenAI compat path (gpt-*, deepseek, grok via x.ai openai compat)
            api_key = _get_model_specific_api_key(user, model_id) or decrypt_val(user.openai_api_key)
            base_url = None
            if is_grok:
                api_key = _get_model_specific_api_key(user, model_id) or decrypt_val(user.xai_api_key)
                base_url = f"https://{_XAI_API_HOST}/v1"
            elif is_deepseek:
                api_key = _get_model_specific_api_key(user, model_id) or decrypt_val(user.deepseek_api_key)
                base_url = "https://api.deepseek.com"
            elif is_kimi:
                api_key = _get_model_specific_api_key(user, model_id) or decrypt_val(user.kimi_api_key)
                base_url = "https://api.moonshot.ai/v1"
            if not api_key and _admin_env_fallback_enabled(user):
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                return None, "選択したモデルのAPIキーがありません。設定でキーを入力するか、別のモデルを選んでください。"
            oai_client = _get_openai_client(api_key, base_url=base_url)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ]
            resp = oai_client.chat.completions.create(
                model=_deepseek_api_model_id(model_id) if is_deepseek else model_id,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=800,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    return {"action": tc.function.name or "update_settings", "arguments": args}, None
                except Exception as je:
                    log_force(f"AI-SETTINGS-OPENAI-JSON-ERR: {je}")
                    return None, "ツール引数のJSON解析に失敗しました。"
            return None, "モデルがツール呼び出しを行いませんでした。"
    except Exception as e:
        log_force(f"AI-SETTINGS-LLM-ERR model={model_id} err={e}")
        return None, f"LLM呼び出しエラー: {str(e)[:150]}"


@app.route('/api/settings/apply-ai-prompt', methods=['POST'])
@login_required
def apply_ai_settings_prompt():
    """Interpret a natural-language settings request, then inspect or safely update allowlisted fields."""
    try:
        d = request.get_json(silent=True) or {}
        instruction = (d.get('prompt') or '').strip()
        model_id = (d.get('model') or '').strip()
        if not instruction:
            return jsonify({'error': 'prompt_required'}), 400
        if len(instruction) > 2000:
            return jsonify({'error': 'prompt_too_long'}), 400
        if not model_id:
            return jsonify({'error': 'model_required'}), 400

        # The browser keeps a short, session-scoped conversation so follow-up
        # settings instructions can refer to the previous result. Treat it as
        # untrusted context and bound both its shape and size before sending it
        # to a model.
        raw_history = d.get('conversation')
        conversation_history = []
        if isinstance(raw_history, list):
            total_chars = 0
            for item in raw_history[-10:]:
                if not isinstance(item, dict):
                    continue
                role = item.get('role')
                if role not in ('user', 'assistant'):
                    continue
                content = str(item.get('content') or '').strip()
                if not content:
                    continue
                content = content[:1200]
                if total_chars + len(content) > 10000:
                    break
                conversation_history.append({'role': role, 'content': content})
                total_chars += len(content)

        db.session.refresh(current_user)
        current_values = _get_ai_safe_settings_snapshot(current_user)
        model_snapshot = _summarize_ai_settings_for_model(current_values)
        decision, err = _call_llm_for_settings_ai(
            model_id, instruction, model_snapshot, current_user,
            conversation_history=conversation_history,
        )
        if err:
            # For admin accounts, return the raw detailed error so they can debug model availability etc.
            is_admin = bool(getattr(current_user, 'is_admin', False))
            payload = {
                'error': 'llm_error',
                'message': err,
            }
            if is_admin:
                payload['raw_error'] = err
                payload['original_model'] = model_id
                payload['admin_note'] = "This detailed error is only shown to administrators."
            return jsonify(payload), 200  # 200 so client can show nicely

        if not decision or not isinstance(decision, dict):
            return jsonify({'error': 'no_action', 'message': 'モデルが有効な設定操作を選択しませんでした。指示をより具体的にしてみてください。'}), 200

        action = str(decision.get('action') or '').strip()
        arguments = decision.get('arguments') if isinstance(decision.get('arguments'), dict) else {}
        if action == 'inspect_settings':
            requested = arguments.get('fields')
            if arguments.get('all') is True or not isinstance(requested, list) or not requested:
                selected_fields = sorted(AI_SAFE_EDITABLE_FIELDS)
            else:
                selected_fields = []
                for field in requested:
                    field = str(field or '').strip()
                    if field in AI_SAFE_EDITABLE_FIELDS and field not in selected_fields:
                        selected_fields.append(field)
            inspected = {field: current_values.get(field) for field in selected_fields}
            log_force(f"AI-SETTINGS-INSPECTED user={current_user.id} model={model_id} keys={selected_fields}")
            return jsonify({
                'status': 'ok',
                'mode': 'inspect',
                'current': inspected,
                'message': '現在の設定を確認しました。',
                'checked_count': len(inspected),
            })

        if action != 'update_settings':
            return jsonify({'error': 'invalid_action', 'message': 'モデルが未対応の設定操作を選択しました。'}), 200

        delta = arguments
        if not delta:
            return jsonify({'error': 'no_changes', 'message': 'モデルが有効な設定変更を提案しませんでした。指示をより具体的にしてみてください。'}), 200

        # Apply (safe only by construction of the func)
        applied = _apply_ai_settings_update(current_user, delta)
        if not applied:
            return jsonify({'error': 'no_valid_changes', 'message': '有効な安全設定の変更がありませんでした。'}), 200

        safe_db_commit()
        log_force(f"AI-SETTINGS-APPLIED user={current_user.id} model={model_id} keys={list(applied.keys())}")

        # Re-fetch fresh snapshot for UI refresh (reuse part of GET logic is overkill, return applied + success)
        return jsonify({
            'status': 'ok',
            'mode': 'update',
            'applied': applied,
            'message': 'AIが設定を更新しました。',
            'changed_count': len(applied),
        })
    except Exception as e:
        log_force(f"AI-SETTINGS-ENDPOINT-ERR: {e}")
        return jsonify({'error': 'internal', 'message': str(e)[:120]}), 500
