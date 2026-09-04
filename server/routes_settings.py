@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        # Ensure we have the latest data from DB
        db.session.refresh(current_user)
        try:
            status = redis_conn.get(f"migration_status:{current_user.id}")
            mig_status = status.decode() if status else "idle"
            prog = redis_conn.get(f"migration_progress:{current_user.id}")
            mig_progress = prog.decode() if prog else ""
        except Exception:
            mig_status = "idle"
            mig_progress = ""
        sp = current_user.system_prompt
        if current_user.enable_e2ee and sp: sp = decrypt_val(sp)
        # 2FA Status
        has_totp = bool(current_user.totp_secret)
        passkeys = _load_user_webauthn_credentials(current_user)
        has_webauthn = bool(passkeys)
        
        global_prompt_value = get_app_setting("global_system_prompt", "") or ""
        global_prompt_enabled = get_bool_app_setting("global_system_prompt_enabled", True)
        global_prompt_uses_time_fallback = bool(global_prompt_enabled and not str(global_prompt_value).strip())
        global_prompt_effective = ""
        if global_prompt_enabled:
            if str(global_prompt_value).strip():
                global_prompt_effective = str(global_prompt_value)
            else:
                global_prompt_effective = build_global_system_prompt()
        
        auto_notices_config = get_user_auto_system_prompt_notices_config(current_user)

        payload = {
            'system_prompt': sp or "",
            'system_prompt_enabled': current_user.system_prompt_enabled if current_user.system_prompt_enabled is not None else True,
            'apply_global_system_prompt': current_user.apply_global_system_prompt if current_user.apply_global_system_prompt is not None else True,
            'apply_auto_system_prompt_notices': get_user_auto_system_prompt_notices_enabled(current_user),
            'auto_system_prompt_notices_preview': build_auto_system_prompt_notices_preview(current_user),
            'auto_system_prompt_notices_config': auto_notices_config,
            'global_system_prompt': global_prompt_value,
            'global_system_prompt_enabled': global_prompt_enabled,
            'global_system_prompt_effective': global_prompt_effective,
            'global_system_prompt_uses_time_fallback': global_prompt_uses_time_fallback,
            'username': current_user.username, 
            'openai_key': _masked_secret(current_user.openai_api_key),
            'gemini_key': _masked_secret(current_user.gemini_api_key),
            'anthropic_key': _masked_secret(current_user.anthropic_api_key),
            'deepseek_key': _masked_secret(current_user.deepseek_api_key),
            'kimi_key': _masked_secret(current_user.kimi_api_key),
            'mistral_key': _masked_secret(current_user.mistral_api_key),
            'model_api_keys': {
                model_key: _SECRET_MASK for model_key in _load_user_model_api_key_map(current_user)
            },
            'gemini_backend': _normalize_gemini_backend(current_user.gemini_backend),
            'gemini_vertex_project': decrypt_val(current_user.gemini_vertex_project) or "",
            'gemini_vertex_location': _normalize_gemini_vertex_location(current_user.gemini_vertex_location),
            'gemini_vertex_credentials_json': _masked_secret(current_user.gemini_vertex_credentials_json),
            'xai_key': _masked_secret(current_user.xai_api_key),
            'google_key': _masked_secret(current_user.google_api_key),
            'google_project': decrypt_val(current_user.google_cloud_project) or "",
            'mic_transcribe_mode': _normalize_mic_transcribe_mode(getattr(current_user, 'mic_transcribe_mode', None)),
            'stt_model': current_user.stt_model or "gpt-4o-mini-transcribe",
            'llm_transcribe_prompt': _normalize_llm_transcribe_prompt(getattr(current_user, 'llm_transcribe_prompt', None)) or "",
            'llm_transcribe_prompt_default': DEFAULT_LLM_TRANSCRIBE_PROMPT,
            'enter_to_send': current_user.enter_to_send,
            'use_sw_cache': current_user.use_sw_cache,
            'clear_cache_on_version_update': current_user.clear_cache_on_version_update if current_user.clear_cache_on_version_update is not None else False,
            'theme_color': current_user.theme_color or "",
            'liquid_glass_enabled': bool(getattr(current_user, 'liquid_glass_enabled', False)),
            'auto_search_on_links': current_user.auto_search_on_links,
            'compact_prompt_mode': current_user.compact_prompt_mode if current_user.compact_prompt_mode is not None else False,
            'minimal_prompt_mode': current_user.minimal_prompt_mode if getattr(current_user, 'minimal_prompt_mode', None) is not None else False,
            'use_last_chat_settings': current_user.use_last_chat_settings,
            'voice_studio_ui': current_user.voice_studio_ui if getattr(current_user, 'voice_studio_ui', None) is not None else True,
            'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(current_user),
            'default_model': current_user.default_model or "gemini-3.6-flash",
            'default_enable_search': current_user.default_enable_search,
            'default_enable_url_context': current_user.default_enable_url_context,
            'default_enable_maps': current_user.default_enable_maps,
            'default_enable_python': current_user.default_enable_python,
            'default_enable_file_creation': current_user.default_enable_file_creation,
            'default_enable_thinking': current_user.default_enable_thinking,
            'default_thinking_level': current_user.default_thinking_level or "high",
            'default_thinking_budget': current_user.default_thinking_budget if current_user.default_thinking_budget is not None else 4096,
            'default_reasoning_effort': current_user.default_reasoning_effort or "medium",
            'default_enable_system_prompt': current_user.default_enable_system_prompt,
            'default_safety_setting': current_user.default_safety_setting or "default",
            'rich_paste_prompt_default': current_user.rich_paste_prompt_default or "",
            'rich_paste_prompt_use_custom_default': current_user.rich_paste_prompt_use_custom_default if current_user.rich_paste_prompt_use_custom_default is not None else False,
            'default_vision_model': current_user.default_vision_model or "gemini-3-flash-preview",
            'last_model': current_user.last_model or "gemini-3.6-flash",
            'last_gem_uuid': current_user.last_gem_uuid,
            'last_enable_search': current_user.last_enable_search,
            'last_enable_url_context': current_user.last_enable_url_context,
            'last_enable_maps': current_user.last_enable_maps,
            'last_enable_python': current_user.last_enable_python,
            'last_enable_file_creation': current_user.last_enable_file_creation,
            'last_enable_thinking': current_user.last_enable_thinking,
            'last_thinking_level': current_user.last_thinking_level or "high",
            'last_thinking_budget': current_user.last_thinking_budget if current_user.last_thinking_budget is not None else 4096,
            'last_reasoning_effort': current_user.last_reasoning_effort or "medium",
            'google_id': current_user.google_id,
            'google_email': current_user.google_email,
            'minashin_sub': current_user.minashin_sub,
            'minashin_email': current_user.minashin_email,
            'last_enable_system_prompt': current_user.last_enable_system_prompt,
            'last_safety_setting': current_user.last_safety_setting or "default",
            'enable_e2ee': current_user.enable_e2ee,
            'migration_status': mig_status,
            'migration_progress': mig_progress,
            'is_2fa_enabled': current_user.is_2fa_enabled,
            'has_totp': has_totp,
            'has_webauthn': has_webauthn,
            'passkey_credentials': _serialize_public_webauthn_credentials(passkeys),
            'passkey_count': len(passkeys),
            'passkey_only_login': current_user.passkey_only_login,
            'skip_2fa_on_google_login': current_user.skip_2fa_on_google_login,
            'default_2fa_method': current_user.default_2fa_method or 'totp',
            'bot_detection_enabled': current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
            'bot_detection_global_enabled': get_bot_detection_global_enabled(),
            'is_bot_banned': current_user.is_bot_banned,
            'bot_ban_reason': current_user.bot_ban_reason,
            'enable_latency_metrics': current_user.enable_latency_metrics if current_user.enable_latency_metrics is not None else False,
            'enable_client_debug_log': current_user.enable_client_debug_log if current_user.enable_client_debug_log is not None else False
        }
        if getattr(current_user, 'is_admin', False):
            payload['admin_api_key_mode'] = _normalize_admin_api_key_mode(current_user.admin_api_key_mode)
        return jsonify(payload)
    d = request.get_json(silent=True) or {}
    if not isinstance(d, dict):
        return jsonify({'error': 'invalid_payload'}), 400
    if len(json.dumps(d, ensure_ascii=False, default=str)) > 2 * 1024 * 1024:
        return jsonify({'error': 'payload_too_large'}), 413
    for secret_key in (
        'openai_key', 'gemini_key', 'anthropic_key', 'deepseek_key', 'xai_key',
        'kimi_key', 'mistral_key', 'google_key', 'google_project', 'gemini_vertex_project'
    ):
        if secret_key in d and len(str(d.get(secret_key) or '')) > 4096:
            return jsonify({'error': f'{secret_key}_too_large'}), 400
    for text_key, max_chars in (
        ('system_prompt', 500_000), ('llm_transcribe_prompt', 100_000),
        ('rich_paste_prompt_default', 100_000), ('gemini_vertex_credentials_json', 100_000)
    ):
        if text_key in d and len(str(d.get(text_key) or '')) > max_chars:
            return jsonify({'error': f'{text_key}_too_large'}), 400
    if 'default_model' in d and d.get('default_model') not in ALL_VALID_MODEL_IDS:
        return jsonify({'error': 'invalid_default_model'}), 400
    if 'default_vision_model' in d and d.get('default_vision_model') not in ALL_VALID_MODEL_IDS:
        return jsonify({'error': 'invalid_vision_model'}), 400
    if 'stt_model' in d and d.get('stt_model') not in VALID_STT_MODELS:
        return jsonify({'error': 'invalid_stt_model'}), 400
    if 'system_prompt' in d: 
        if current_user.enable_e2ee: current_user.system_prompt = encrypt_val(d['system_prompt'])
        else: current_user.system_prompt = d['system_prompt']
    if 'system_prompt_enabled' in d:
        current_user.system_prompt_enabled = bool(d['system_prompt_enabled'])
    if 'apply_global_system_prompt' in d:
        current_user.apply_global_system_prompt = bool(d['apply_global_system_prompt'])
    if 'apply_auto_system_prompt_notices' in d:
        current_user.apply_auto_system_prompt_notices = bool(d['apply_auto_system_prompt_notices'])
    if 'auto_system_prompt_notices_config' in d:
        set_user_auto_system_prompt_notices_config(current_user, d.get('auto_system_prompt_notices_config'))
    if 'openai_key' in d and d['openai_key'] != _SECRET_MASK: current_user.openai_api_key = encrypt_val(d['openai_key'])
    if 'gemini_key' in d and d['gemini_key'] != _SECRET_MASK: current_user.gemini_api_key = encrypt_val(d['gemini_key'])
    if 'anthropic_key' in d and d['anthropic_key'] != _SECRET_MASK: current_user.anthropic_api_key = encrypt_val(d['anthropic_key'])
    if 'deepseek_key' in d and d['deepseek_key'] != _SECRET_MASK: current_user.deepseek_api_key = encrypt_val(d['deepseek_key'])
    if 'kimi_key' in d and d['kimi_key'] != _SECRET_MASK: current_user.kimi_api_key = encrypt_val(d['kimi_key'])
    if 'mistral_key' in d and d['mistral_key'] != _SECRET_MASK: current_user.mistral_api_key = encrypt_val(d['mistral_key'])
    if 'model_api_keys' in d: _merge_masked_model_api_key_map(current_user, d.get('model_api_keys'))
    if 'gemini_backend' in d: current_user.gemini_backend = _normalize_gemini_backend(d['gemini_backend'])
    if 'gemini_vertex_project' in d: current_user.gemini_vertex_project = encrypt_val(d['gemini_vertex_project'])
    if 'gemini_vertex_location' in d: current_user.gemini_vertex_location = _normalize_gemini_vertex_location(d['gemini_vertex_location'])
    if 'gemini_vertex_credentials_json' in d and d['gemini_vertex_credentials_json'] != _SECRET_MASK:
        try:
            normalized_vertex_json = _normalize_gemini_vertex_credentials_json(d['gemini_vertex_credentials_json'])
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        current_user.gemini_vertex_credentials_json = encrypt_val(normalized_vertex_json)
    if 'xai_key' in d and d['xai_key'] != _SECRET_MASK: current_user.xai_api_key = encrypt_val(d['xai_key'])
    if 'google_key' in d and d['google_key'] != _SECRET_MASK: current_user.google_api_key = encrypt_val(d['google_key'])
    if 'google_project' in d: current_user.google_cloud_project = encrypt_val(d['google_project'])
    if 'mic_transcribe_mode' in d:
        current_user.mic_transcribe_mode = _normalize_mic_transcribe_mode(d['mic_transcribe_mode'])
    if 'stt_model' in d: current_user.stt_model = d['stt_model']
    if 'llm_transcribe_prompt' in d:
        current_user.llm_transcribe_prompt = _normalize_llm_transcribe_prompt(d.get('llm_transcribe_prompt'))
    if 'enter_to_send' in d: current_user.enter_to_send = bool(d['enter_to_send'])
    if 'use_sw_cache' in d: current_user.use_sw_cache = bool(d['use_sw_cache'])
    if 'clear_cache_on_version_update' in d: current_user.clear_cache_on_version_update = bool(d['clear_cache_on_version_update'])
    if 'compact_prompt_mode' in d: current_user.compact_prompt_mode = bool(d['compact_prompt_mode'])
    if 'minimal_prompt_mode' in d: current_user.minimal_prompt_mode = bool(d['minimal_prompt_mode'])
    if 'compact_prompt_mode' in d and current_user.compact_prompt_mode:
        current_user.minimal_prompt_mode = False
    if 'minimal_prompt_mode' in d and current_user.minimal_prompt_mode:
        current_user.compact_prompt_mode = False
    if 'theme_color' in d: current_user.theme_color = normalize_theme_color(d.get('theme_color'))
    if 'liquid_glass_enabled' in d: current_user.liquid_glass_enabled = bool(d['liquid_glass_enabled'])
    if 'auto_search_on_links' in d: current_user.auto_search_on_links = bool(d['auto_search_on_links'])
    if 'use_last_chat_settings' in d: current_user.use_last_chat_settings = bool(d['use_last_chat_settings'])
    if 'voice_studio_ui' in d: current_user.voice_studio_ui = bool(d['voice_studio_ui'])
    if 'default_model' in d: current_user.default_model = d['default_model']
    if 'last_gem_uuid' in d:
        val = d['last_gem_uuid']
        if val is not None:
            gem = Gem.query.filter_by(uuid=val).first()
            if not gem or gem.user_id != current_user.id:
                return jsonify({'error': 'Invalid gem UUID'}), 403
        thread_id = d.get('thread_id')
        if thread_id:
            th = resolve_thread_for_user(thread_id, current_user.id)
            if th:
                th.last_gem_uuid = val
        else:
            current_user.last_gem_uuid = val
    if 'temp_chat_timeout_seconds' in d:
        current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(
            d.get('temp_chat_timeout_seconds')
        )
    if 'default_enable_search' in d: current_user.default_enable_search = bool(d['default_enable_search'])
    if 'default_enable_url_context' in d: current_user.default_enable_url_context = bool(d['default_enable_url_context'])
    if 'default_enable_maps' in d: current_user.default_enable_maps = bool(d['default_enable_maps'])
    if 'default_enable_python' in d: current_user.default_enable_python = bool(d['default_enable_python'])
    if 'default_enable_file_creation' in d: current_user.default_enable_file_creation = bool(d['default_enable_file_creation'])
    if 'default_enable_thinking' in d: current_user.default_enable_thinking = bool(d['default_enable_thinking'])
    if 'default_thinking_level' in d: current_user.default_thinking_level = d['default_thinking_level'] or "high"
    if 'default_thinking_budget' in d:
        try:
            current_user.default_thinking_budget = int(d['default_thinking_budget'])
        except Exception:
            pass
    if 'default_reasoning_effort' in d: current_user.default_reasoning_effort = d['default_reasoning_effort'] or "medium"
    if 'default_enable_system_prompt' in d: current_user.default_enable_system_prompt = bool(d['default_enable_system_prompt'])
    if 'default_safety_setting' in d: current_user.default_safety_setting = d['default_safety_setting'] or "default"
    if 'rich_paste_prompt_default' in d: current_user.rich_paste_prompt_default = d['rich_paste_prompt_default'] or ""
    if 'rich_paste_prompt_use_custom_default' in d: current_user.rich_paste_prompt_use_custom_default = bool(d['rich_paste_prompt_use_custom_default'])
    if 'default_vision_model' in d: current_user.default_vision_model = d['default_vision_model']
    if 'passkey_only_login' in d:
        target = bool(d['passkey_only_login'])
        if target:
            creds = _load_user_webauthn_credentials(current_user)
            if not creds:
                return jsonify({'error': 'No passkey registered'}), 400
        current_user.passkey_only_login = target
    if 'skip_2fa_on_google_login' in d:
        current_user.skip_2fa_on_google_login = bool(d['skip_2fa_on_google_login'])
    if 'default_2fa_method' in d:
        current_user.default_2fa_method = str(d['default_2fa_method'])
    if 'bot_detection_enabled' in d and d['bot_detection_enabled'] is not None:
        current_user.bot_detection_enabled = bool(d['bot_detection_enabled'])
    if 'enable_latency_metrics' in d:
        current_user.enable_latency_metrics = bool(d['enable_latency_metrics'])
    if 'enable_client_debug_log' in d:
        current_user.enable_client_debug_log = bool(d['enable_client_debug_log'])
        log_force(f"SETTINGS-UPDATE: user={current_user.id} enable_client_debug_log={current_user.enable_client_debug_log}")
    if getattr(current_user, 'is_admin', False) and 'admin_api_key_mode' in d:
        current_user.admin_api_key_mode = _normalize_admin_api_key_mode(d['admin_api_key_mode'])
    if getattr(current_user, 'is_admin', False) and 'bot_detection_global_enabled' in d:
        set_app_setting("bot_detection_global_enabled", "1" if d['bot_detection_global_enabled'] else "0")
    
    log_force(f"DEBUG: handle_settings processing keys={sorted(d.keys())}")
    result_message = None
    if d.get('new_password'):
        new_password = str(d['new_password'])
        if len(new_password) < 8 or len(new_password) > 256:
            return jsonify({'error': 'Password must be 8-256 characters'}), 400
        current_user.set_password(new_password)
        revoke_user_sessions(current_user.id, exclude_session_id=session.get('session_id'))
    if d.get('new_username') and d['new_username'] != current_user.username:
        new_username = str(d['new_username']).replace('\x00', '').strip()
        if len(new_username) < 3 or len(new_username) > 80 or re.search(r'[\x00-\x1f\x7f]', new_username):
            return jsonify({'error': 'Username must be 3-80 characters'}), 400
        if _is_primary_admin_username(new_username) and not getattr(current_user, "is_admin", False):
            pass
        elif not User.query.filter_by(username=new_username).first(): current_user.username = new_username
    if 'enable_e2ee' in d and d['enable_e2ee'] != current_user.enable_e2ee:
        target_enable = d['enable_e2ee']
        task_queue.enqueue(migrate_e2ee_task, current_user.id, target_enable)
        result_message = "暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。"
    if 'disable_2fa' in d and d['disable_2fa']:
        current_user.is_2fa_enabled = False
        current_user.totp_secret = None
        current_user.webauthn_credentials = None
        current_user.passkey_only_login = False
        current_user.default_2fa_method = 'totp'
        result_message = "2FAを無効化しました。"
    else:
        log_force("DEBUG: handle_settings calling _refresh_user_2fa_state")
        _refresh_user_2fa_state(current_user)
        log_force("DEBUG: handle_settings calling safe_db_commit")
        safe_db_commit()
        log_force("DEBUG: handle_settings safe_db_commit finished")
        if result_message is None:
            result_message = "設定を保存しました"
    # /api/settings is AJAX-only (the client always posts with fetch/apiFetch and
    # shows its own toast), so the result is returned in the JSON response.
    # Previously this endpoint used flash(), whose messages are only consumed on
    # the next full page render -- so a save's message (including background
    # auto-saves such as rich-paste prompts, Gem application, etc.) leaked onto
    # the next reload as a stale "設定を保存しました" toast even when the user
    # never opened the settings modal.  Any leftover settings-save flashes from
    # sessions created before that change are purged on every request by
    # _purge_stale_settings_flashes() (see _LEAKY_SETTINGS_FLASHES above), so
    # nothing is flashed here.
    log_force("DEBUG: handle_settings returning ok")
    return jsonify({'status': 'ok', 'message': result_message})

@app.route('/api/debug/client_log', methods=['POST'])
@login_required
def receive_client_log():
    if not getattr(current_user, 'enable_client_debug_log', False):
        return jsonify({'status': 'ignored', 'reason': 'disabled'}), 200
    try:
        d = request.get_json(silent=True) or {}
        level = str(d.get('level') or 'info').upper()
        if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR'}:
            level = 'INFO'
        msg = str(d.get('message') or '')
        if not msg:
            return jsonify({'status': 'ignored', 'reason': 'empty'}), 200
        msg = re.sub(r'[\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', ' ', msg)[:8192]
        log_force(f"CLIENT-DEBUG [{level}]: {msg}")
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_force(f"CLIENT-DEBUG-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 400

# --- AI Settings Prompt (tool use in settings modal) ---

def _build_ai_settings_tool_schema():
    """Return tool schemas for updating settings or inspecting current values."""
    # Common property descriptions (Japanese for better JP prompt understanding)
    props = {
        "default_model": {"type": "string", "description": "既定モデルID", "enum": sorted(ALL_VALID_MODEL_IDS)},
        "default_enable_search": {"type": "boolean", "description": "Searchツールの既定ON/OFF"},
        "default_enable_url_context": {"type": "boolean", "description": "URLs (URLコンテキスト) の既定ON/OFF"},
        "default_enable_maps": {"type": "boolean", "description": "Maps (Google Maps grounding) の既定ON/OFF"},
        "default_enable_python": {"type": "boolean", "description": "Python実行ツールの既定ON/OFF"},
        "default_enable_file_creation": {"type": "boolean", "description": "ファイル作成ツールの既定ON/OFF"},
        "default_enable_thinking": {"type": "boolean", "description": "Thinking (拡張思考) の既定ON/OFF"},
        "default_thinking_level": {"type": "string", "description": "Thinkingレベル", "enum": sorted(VALID_THINKING_LEVELS)},
        "default_thinking_budget": {"type": "integer", "description": "Thinking budget (トークン数, 例: 4096)"},
        "default_reasoning_effort": {"type": "string", "description": "Reasoning effort", "enum": sorted(VALID_REASONING_EFFORTS)},
        "default_enable_system_prompt": {"type": "boolean", "description": "既定でシステムプロンプト(ユーザー定義)を使用するか"},
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
        "auto_search_on_links": {"type": "boolean", "description": "リンク検出時の自動検索"},
        "use_last_chat_settings": {"type": "boolean", "description": "前回の送信設定を継続使用"},
        "temp_chat_timeout_seconds": {"type": "integer", "description": "一時チャットの切断タイムアウト秒数 (30-86400)"},
        "theme_color": {"type": "string", "description": "テーマカラー (HEX, 例: #10a37f)"},
        "liquid_glass_enabled": {"type": "boolean", "description": "Liquid Glass表示モードのON/OFF"},
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


# --- Session Management ---

@app.route('/api/sessions', methods=['GET'])
@login_required
def list_sessions():
    sid = session.get('session_id')
    rows = UserSession.query.filter_by(user_id=current_user.id).order_by(UserSession.last_seen_at.desc()).limit(50).all()
    return jsonify({
        'sessions': [
            {
                'id': s.id,
                'created_at': s.created_at.isoformat(),
                'last_seen_at': s.last_seen_at.isoformat() if s.last_seen_at else None,
                'ip_address': s.ip_address,
                'user_agent': s.user_agent,
                'is_current': s.session_id == sid,
                'is_revoked': s.is_revoked
            } for s in rows
        ]
    })

@app.route('/api/sessions/revoke', methods=['POST'])
@login_required
def revoke_session():
    data = request.json or {}
    sess_id = data.get('id')
    if not sess_id:
        return jsonify({'error': 'id_required'}), 400
    user_sess = UserSession.query.filter_by(id=sess_id, user_id=current_user.id).first()
    if not user_sess:
        return jsonify({'error': 'not_found'}), 404
    if not user_sess.is_revoked:
        user_sess.is_revoked = True
        user_sess.revoked_at = datetime.utcnow()
        safe_db_commit()
    logged_out = False
    if user_sess.session_id == session.get('session_id'):
        session.pop('session_id', None)
        logout_user()
        logged_out = True
    return jsonify({'status': 'ok', 'logged_out': logged_out})

@app.route('/api/sessions/revoke_others', methods=['POST'])
@login_required
def revoke_other_sessions():
    sid = session.get('session_id')
    revoke_user_sessions(current_user.id, exclude_session_id=sid)
    return jsonify({'status': 'ok'})

@app.route('/api/sessions/revoke_all', methods=['POST'])
@login_required
def revoke_all_sessions():
    revoke_user_sessions(current_user.id, exclude_session_id=None)
    session.pop('session_id', None)
    logout_user()
    return jsonify({'status': 'ok', 'logged_out': True})

# --- 2FA Settings Routes ---

@app.route('/api/2fa/totp/setup', methods=['POST'])
@login_required
def totp_setup():
    secret = pyotp.random_base32()
    # Save temporarily encrypted or just send back? Ideally verify first.
    # We will send back the secret and QR, but not enable it until verified.
    session['temp_totp_secret'] = secret
    
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.username, issuer_name="AI Chat Playground")
    img = qrcode.make(uri)
    buf = BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    return jsonify({'secret': secret, 'qr_image': f"data:image/png;base64,{b64}"})

@app.route('/api/2fa/totp/enable', methods=['POST'])
@login_required
def totp_enable():
    code = request.json.get('code')
    secret = session.get('temp_totp_secret')
    if not secret: return jsonify({'error': 'Setup session expired'}), 400
    
    if pyotp.TOTP(secret).verify(code):
        current_user.totp_secret = encrypt_val(secret)
        current_user.is_2fa_enabled = True
        session.pop('temp_totp_secret', None)
        safe_db_commit()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Invalid code'}), 400

@app.route('/api/2fa/webauthn/register/options', methods=['POST'])
@login_required
def webauthn_reg_options():
    existing = _load_user_webauthn_credentials(current_user)
    options_kwargs = {
        "rp_name": "AI Chat Playground",
        "rp_id": request.host.split(':')[0],
        "user_id": str(current_user.id).encode(),
        "user_name": current_user.username,
        "authenticator_selection": AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED
        )
    }
    if existing:
        options_kwargs["exclude_credentials"] = [
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in existing
        ]
    options = generate_registration_options(**options_kwargs)
    session['webauthn_reg_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    return options_to_json(options)

@app.route('/api/2fa/webauthn/register/verify', methods=['POST'])
@login_required
def webauthn_reg_verify():
    try:
        data = request.json or {}
        challenge = session.get('webauthn_reg_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        verification = verify_registration_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            require_user_verification=False 
        )
        
        creds = _load_user_webauthn_credentials(current_user)
        cred_id = base64.b64encode(verification.credential_id).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
        cred_name = str(data.get('name') or '').strip()
        if not cred_name:
            cred_name = f"Security Key {len(creds) + 1}"
        existing = next((c for c in creds if c['id'] == cred_id), None)
        if existing:
            existing['public_key'] = base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
            existing['sign_count'] = verification.sign_count
            existing['name'] = cred_name
        else:
            creds.append({
                'id': cred_id,
                'public_key': base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('='),
                'sign_count': verification.sign_count,
                'name': cred_name,
                'created_at': datetime.utcnow().isoformat() + "Z"
            })
        _save_user_webauthn_credentials(current_user, creds)
        current_user.is_2fa_enabled = True
        session.pop('webauthn_reg_challenge', None)
        safe_db_commit()
        return jsonify({'status': 'ok', 'passkey_credentials': _serialize_public_webauthn_credentials(creds)})
    except Exception as e:
        logger.error(f"WebAuthn Reg Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/2fa/webauthn/remove', methods=['POST'])
@login_required
def webauthn_remove():
    data = request.json or {}
    cred_id = str(data.get('id') or '').strip()
    if not cred_id:
        return jsonify({'error': 'id_required'}), 400
    creds = _load_user_webauthn_credentials(current_user)
    filtered = [c for c in creds if c['id'] != cred_id]
    if len(filtered) == len(creds):
        return jsonify({'error': 'not_found'}), 404
    _save_user_webauthn_credentials(current_user, filtered)
    _refresh_user_2fa_state(current_user)
    safe_db_commit()
    return jsonify({
        'status': 'ok',
        'has_webauthn': bool(filtered),
        'passkey_only_login': bool(current_user.passkey_only_login),
        'is_2fa_enabled': bool(current_user.is_2fa_enabled),
        'passkey_count': len(filtered)
    })

@app.route('/api/gems', methods=['GET', 'POST'])
@login_required
def handle_gems():
    if request.method == 'GET':
        gems = Gem.query.filter_by(user_id=current_user.id).order_by(Gem.created_at.desc()).all()
        return jsonify([{'uuid': g.uuid, 'id': g.id, 'name': g.name, 'description': g.description, 'instruction': g.instruction, 'fixed_prompts': g.fixed_prompts_json, 'default_model': g.default_model} for g in gems])
    d = request.get_json(silent=True) or {}
    try:
        payload = _normalize_gem_payload(d)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    import uuid as _uuid
    gem = Gem(uuid=str(_uuid.uuid4()), user_id=current_user.id, **payload)
    db.session.add(gem)
    safe_db_commit()
    return jsonify({'uuid': gem.uuid, 'id': gem.id, 'name': gem.name})

@app.route('/api/gems/<string:gem_uuid>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def handle_gem_item(gem_uuid):
    gem = Gem.query.filter_by(uuid=gem_uuid).first_or_404()
    if gem.user_id != current_user.id: return jsonify({'error': '403'}), 403

    if request.method == 'GET':
        return jsonify({'uuid': gem.uuid, 'id': gem.id, 'name': gem.name, 'description': gem.description, 'instruction': gem.instruction, 'fixed_prompts': gem.fixed_prompts_json, 'default_model': gem.default_model})

    if request.method == 'PUT':
        d = request.get_json(silent=True) or {}
        try:
            payload = _normalize_gem_payload(d, existing=gem)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        for key, value in payload.items():
            setattr(gem, key, value)
        safe_db_commit()
        return jsonify({'uuid': gem.uuid, 'id': gem.id, 'name': gem.name})
    if request.method == 'DELETE':
        db.session.delete(gem)
        safe_db_commit()
        return jsonify({'status': 'deleted'})

@app.route('/api/maintenance', methods=['POST'])
@login_required
def toggle_maintenance():
    if not getattr(current_user, "is_admin", False): return abort(403)
    lock_file = os.path.join(os.path.dirname(__file__), 'maintenance.lock')
    if request.json.get('enabled'):
        with open(lock_file, 'w') as f: f.write('locked')
        app.config['MAINTENANCE_MODE'] = True
    else:
        if os.path.exists(lock_file): os.remove(lock_file)
        app.config['MAINTENANCE_MODE'] = False
    return jsonify({'status': 'ok', 'mode': app.config['MAINTENANCE_MODE']})

