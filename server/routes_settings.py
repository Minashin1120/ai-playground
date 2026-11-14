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
            'gemini_free_key': _masked_secret(getattr(current_user, 'gemini_free_api_key', None)),
            'anthropic_key': _masked_secret(current_user.anthropic_api_key),
            'deepseek_key': _masked_secret(current_user.deepseek_api_key),
            'zai_key': _masked_secret(current_user.zai_api_key),
            'kimi_key': _masked_secret(current_user.kimi_api_key),
            'mistral_key': _masked_secret(current_user.mistral_api_key),
            'ideogram_key': _masked_secret(current_user.ideogram_api_key),
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
            'light_mode_enabled': bool(getattr(current_user, 'light_mode_enabled', False)),
            'auto_search_on_links': current_user.auto_search_on_links,
            'compact_prompt_mode': current_user.compact_prompt_mode if current_user.compact_prompt_mode is not None else False,
            'minimal_prompt_mode': current_user.minimal_prompt_mode if getattr(current_user, 'minimal_prompt_mode', None) is not None else False,
            'use_last_chat_settings': current_user.use_last_chat_settings,
            'voice_studio_ui': current_user.voice_studio_ui if getattr(current_user, 'voice_studio_ui', None) is not None else True,
            'math_steps_enabled': bool(getattr(current_user, 'math_steps_enabled', False)),
            'math_steps_config': normalize_math_steps_config(getattr(current_user, 'math_steps_config', None)),
            'default_math_steps': bool(getattr(current_user, 'default_math_steps', False)),
            'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(current_user),
            'default_model': current_user.default_model or "gemini-3.6-flash",
            'default_enable_search': current_user.default_enable_search,
            'default_gemini_free_tier': bool(getattr(current_user, 'default_gemini_free_tier', False)),
            'default_gemini_flex': bool(getattr(current_user, 'default_gemini_flex', False)),
            'default_enable_url_context': current_user.default_enable_url_context,
            'default_enable_maps': current_user.default_enable_maps,
            'default_enable_python': current_user.default_enable_python,
            'default_enable_file_creation': current_user.default_enable_file_creation,
            'default_enable_thinking': current_user.default_enable_thinking,
            'default_thinking_level': current_user.default_thinking_level or "high",
            'default_thinking_budget': current_user.default_thinking_budget if current_user.default_thinking_budget is not None else 4096,
            'default_reasoning_effort': current_user.default_reasoning_effort or "medium",
            'default_enable_system_prompt': current_user.default_enable_system_prompt,
            'default_enable_mcp': current_user.default_enable_mcp if current_user.default_enable_mcp is not None else True,
            'default_safety_setting': current_user.default_safety_setting or "default",
            'rich_paste_prompt_default': current_user.rich_paste_prompt_default or "",
            'rich_paste_prompt_use_custom_default': current_user.rich_paste_prompt_use_custom_default if current_user.rich_paste_prompt_use_custom_default is not None else False,
            'default_vision_model': current_user.default_vision_model or "gemini-3-flash-preview",
            'last_model': current_user.last_model or "gemini-3.6-flash",
            'last_skill_uuid': current_user.last_skill_uuid,
            # Legacy alias for clients that still read the old field name.
            'last_gem_uuid': current_user.last_skill_uuid,
            'last_enable_search': current_user.last_enable_search,
            'last_gemini_free_tier': bool(getattr(current_user, 'last_gemini_free_tier', False)),
            'last_gemini_flex': bool(getattr(current_user, 'last_gemini_flex', False)),
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
            'last_enable_mcp': current_user.last_enable_mcp if current_user.last_enable_mcp is not None else True,
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
            'enable_client_debug_log': current_user.enable_client_debug_log if current_user.enable_client_debug_log is not None else False,
            'enable_server_diagnostic_log': bool(current_user.enable_server_diagnostic_log),
            'server_diagnostic_log_retention_hours': current_user.server_diagnostic_log_retention_hours or 1
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
        'openai_key', 'gemini_key', 'gemini_free_key', 'anthropic_key', 'deepseek_key', 'xai_key',
        'kimi_key', 'mistral_key', 'ideogram_key', 'zai_key', 'google_key', 'google_project', 'gemini_vertex_project'
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
    if 'gemini_free_key' in d and d['gemini_free_key'] != _SECRET_MASK: current_user.gemini_free_api_key = encrypt_val(d['gemini_free_key'])
    if 'anthropic_key' in d and d['anthropic_key'] != _SECRET_MASK: current_user.anthropic_api_key = encrypt_val(d['anthropic_key'])
    if 'deepseek_key' in d and d['deepseek_key'] != _SECRET_MASK: current_user.deepseek_api_key = encrypt_val(d['deepseek_key'])
    if 'zai_key' in d and d['zai_key'] != _SECRET_MASK: current_user.zai_api_key = encrypt_val(d['zai_key'])
    if 'kimi_key' in d and d['kimi_key'] != _SECRET_MASK: current_user.kimi_api_key = encrypt_val(d['kimi_key'])
    if 'mistral_key' in d and d['mistral_key'] != _SECRET_MASK: current_user.mistral_api_key = encrypt_val(d['mistral_key'])
    if 'ideogram_key' in d and d['ideogram_key'] != _SECRET_MASK: current_user.ideogram_api_key = encrypt_val(d['ideogram_key'])
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
    if 'light_mode_enabled' in d: current_user.light_mode_enabled = bool(d['light_mode_enabled'])
    if 'auto_search_on_links' in d: current_user.auto_search_on_links = bool(d['auto_search_on_links'])
    if 'use_last_chat_settings' in d: current_user.use_last_chat_settings = bool(d['use_last_chat_settings'])
    if 'voice_studio_ui' in d: current_user.voice_studio_ui = bool(d['voice_studio_ui'])
    if 'math_steps_enabled' in d: current_user.math_steps_enabled = bool(d['math_steps_enabled'])
    if 'math_steps_config' in d:
        current_user.math_steps_config = dump_math_steps_config(d.get('math_steps_config'))
    if 'default_math_steps' in d: current_user.default_math_steps = bool(d['default_math_steps'])
    if 'default_model' in d: current_user.default_model = d['default_model']
    if 'last_skill_uuid' in d or 'last_gem_uuid' in d:
        val = d.get('last_skill_uuid', d.get('last_gem_uuid'))
        if val is not None:
            skill = Skill.query.filter_by(uuid=val).first()
            if not skill or skill.user_id != current_user.id:
                return jsonify({'error': 'Invalid skill UUID'}), 403
        thread_id = d.get('thread_id')
        if thread_id:
            th = resolve_thread_for_user(thread_id, current_user.id)
            if th:
                th.last_skill_uuid = val
        else:
            current_user.last_skill_uuid = val
    if 'temp_chat_timeout_seconds' in d:
        current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(
            d.get('temp_chat_timeout_seconds')
        )
    if 'default_enable_search' in d: current_user.default_enable_search = bool(d['default_enable_search'])
    if 'default_gemini_free_tier' in d: current_user.default_gemini_free_tier = bool(d['default_gemini_free_tier'])
    if 'default_gemini_flex' in d: current_user.default_gemini_flex = bool(d['default_gemini_flex'])
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
    if 'default_enable_mcp' in d: current_user.default_enable_mcp = bool(d['default_enable_mcp'])
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
    diagnostic_retention_changed = False
    if 'server_diagnostic_log_retention_hours' in d:
        hours = d['server_diagnostic_log_retention_hours']
        if type(hours) is not int or not 1 <= hours <= 8760:
            return jsonify({'error': 'ログの保存時間は1〜8760時間の整数で指定してください。'}), 400
        diagnostic_retention_changed = current_user.server_diagnostic_log_retention_hours != hours
        current_user.server_diagnostic_log_retention_hours = hours
    diagnostic_consent_changed = False
    if 'enable_server_diagnostic_log' in d:
        if type(d['enable_server_diagnostic_log']) is not bool:
            return jsonify({'error': 'invalid_payload'}), 400
        diagnostic_consent_changed = bool(current_user.enable_server_diagnostic_log) != d['enable_server_diagnostic_log']
        current_user.enable_server_diagnostic_log = d['enable_server_diagnostic_log']
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
        if 'enable_server_diagnostic_log' in d and (diagnostic_consent_changed or d['enable_server_diagnostic_log'] is False):
            _clear_server_diagnostics(current_user.id)
        if diagnostic_retention_changed:
            _update_server_diagnostic_retention(current_user.id)
        log_force("DEBUG: handle_settings safe_db_commit finished")
        if result_message is None:
            result_message = "設定を保存しました"
    # /api/settings is AJAX-only (the client always posts with fetch/apiFetch and
    # shows its own toast), so the result is returned in the JSON response.
    # Previously this endpoint used flash(), whose messages are only consumed on
    # the next full page render -- so a save's message (including background
    # auto-saves such as rich-paste prompts, Skill application, etc.) leaked onto
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
        _record_server_diagnostic(current_user.id, 'server.client_log')
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_force(f"CLIENT-DEBUG-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 400
