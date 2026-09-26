"""Android pairing and token endpoints; grants are transient Redis records."""

# CAS approval and one-time redemption are atomic across Gunicorn workers.
_MOBILE_APPROVE_SCRIPT = """
local value = redis.call('GET', KEYS[1])
if not value or cjson.decode(value).status ~= 'pending' then return 0 end
local ttl = redis.call('TTL', KEYS[1])
if ttl <= 0 then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ttl)
return 1
"""
_MOBILE_REDEEM_SCRIPT = """
local value = redis.call('GET', KEYS[1])
if not value then return '' end
local status = cjson.decode(value).status
if status ~= 'pending' then redis.call('DEL', KEYS[1]) end
return value
"""

_MOBILE_DEPRECATED_MODELS = {
    'gemini-3.1-flash-lite-preview', 'gemini-3-pro-preview',
    'gemini-3.1-flash-image-preview', 'gemini-3-pro-image-preview',
    'deepseek-v4-flash-vision-exp', 'deepseek-v4-flash-0731', 'deepseek-v4-flash',
    'grok-imagine-image-pro', 'grok-voice-think-fast-1.0', 'grok-voice-fast-1.0',
    'grok-voice-agent', 'grok-4-1-fast-reasoning', 'grok-4-1-fast-non-reasoning',
    'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning',
}

# The native Android Realtime studio uses the persistent conversation session
# API.  OpenAI Realtime Whisper is a transcription-only session and must stay
# unavailable until the Android client has a matching transcription flow.
_MOBILE_REALTIME_TRANSCRIPTION_ONLY_MODELS = {'gpt-realtime-whisper'}


def _mobile_model_mode(model_id):
    """Classify Web model IDs without exposing provider credentials or settings."""
    model_id = str(model_id or '').lower()
    if model_id == 'mistral-ocr-4-0':
        return 'ocr'
    if model_id == 'gemini-embedding-2':
        return 'embedding'
    if model_id in XAI_LIVE_STT_MODELS:
        return 'realtime_audio'
    if 'transcribe' in model_id:
        return 'transcription'
    if model_id in STS_MODELS or model_id in XAI_STS_MODEL_ALIASES or 'realtime' in model_id or 'voice' in model_id or 'live-transcribe' in model_id:
        return 'realtime_audio'
    if 'tts' in model_id:
        return 'tts'
    if model_id.startswith('lyria-'):
        return 'music'
    if 'video' in model_id or model_id.startswith('veo-') or model_id.startswith('gemini-omni'):
        return 'video'
    if 'image' in model_id:
        return 'image'
    if any(marker in model_id for marker in ('deep-research', 'antigravity', 'computer-use', 'robotics')):
        return 'agent'
    return 'chat'


def _mobile_model_name(model_id):
    words = str(model_id).replace('_', '-').split('-')
    labels = {'gpt': 'GPT', 'tts': 'TTS', 'ocr': 'OCR', 'ai': 'AI', 'xai': 'xAI',
              'gemini': 'Gemini', 'grok': 'Grok', 'claude': 'Claude',
              'deepseek': 'DeepSeek', 'kimi': 'Kimi', 'mistral': 'Mistral',
              'veo': 'Veo', 'lyria': 'Lyria'}
    return ' '.join(labels.get(word.lower(), word.capitalize()) for word in words)


def _mobile_model_metadata(model_id):
    provider = get_model_api_provider(model_id) or 'unknown'
    provider_label = _PROVIDER_LABELS.get(provider, provider.title())
    mode = _mobile_model_mode(model_id)
    deprecated = model_id in _MOBILE_DEPRECATED_MODELS
    capabilities = [mode]
    if mode == 'chat':
        capabilities += ['attachments', 'thinking']
        if model_id != 'gemini-3.8-flash-cyber':
            capabilities.append('search')
        if provider in {'gemini', 'openai', 'anthropic'}:
            capabilities.append('prompt_cache')
        if model_id != 'gemini-3.8-flash-cyber':
            capabilities += ['python', 'mcp']
    if mode in {'chat', 'image'} and globals().get('_is_batch_model', lambda _model: False)(model_id):
        capabilities.append('batch')
    native_modes = {'chat', 'image', 'video', 'ocr', 'tts', 'transcription', 'realtime_audio', 'agent'}
    realtime_transcription_only = model_id in _MOBILE_REALTIME_TRANSCRIPTION_ONLY_MODELS
    return {
        'id': model_id,
        'name': _mobile_model_name(model_id),
        'provider': provider,
        'provider_label': provider_label,
        'mode': mode,
        'capabilities': capabilities,
        'deprecated': deprecated,
        'selectable': mode in native_modes and not deprecated and not realtime_transcription_only,
    }


@app.route('/api/mobile/v1/config')
def mobile_config():
    return jsonify({
        'api_version': 1, 'system_version': app.config['SYSTEM_VERSION'],
        'client_id': 'official-android',
        'auth_flow': 'native_credentials_v1',
        'legacy_auth_flow': 'device_pairing_v1',
        'legacy_auth_deprecated': True,
        'native_signup_endpoint': '/api/mobile/v1/auth/signup',
        'native_login_endpoint': '/api/mobile/v1/auth/login',
        'native_google_endpoint': '/api/mobile/v1/auth/google',
        'google_server_client_id': os.getenv('GOOGLE_CLIENT_ID', ''),
        'play_integrity_cloud_project_number': os.getenv('PLAY_INTEGRITY_CLOUD_PROJECT_NUMBER', ''),
        'play_integrity_enabled': bool(os.getenv('PLAY_INTEGRITY_CLOUD_PROJECT_NUMBER') and os.getenv('PLAY_INTEGRITY_SERVICE_ACCOUNT_FILE')),
        'native_totp_endpoint': '/api/mobile/v1/auth/totp',
        'native_passkey_options_endpoint': '/api/mobile/v1/auth/passkey/options',
        'native_passkey_verify_endpoint': '/api/mobile/v1/auth/passkey/verify',
        'native_2fa_webauthn_options_endpoint': '/api/mobile/v1/auth/2fa/webauthn/options',
        'native_2fa_webauthn_verify_endpoint': '/api/mobile/v1/auth/2fa/webauthn/verify',
        'security_endpoint': '/api/mobile/v1/security',
        'account_import_endpoints': {
            'start': '/api/account/import/upload/start',
            'chunk': '/api/account/import/upload/<upload_id>/chunk',
            'complete': '/api/account/import/upload/<upload_id>/complete',
            'cancel': '/api/account/import/upload/<upload_id>',
            'import': '/api/account/import',
        },
        'native_setup_endpoint': '/api/mobile/v1/setup',
        'device_endpoint': '/api/mobile/v1/device',
        'token_endpoint': '/api/mobile/v1/token',
        'verification_uri': '/android/connect',
        'me_endpoint': '/api/mobile/v1/me', 'revoke_endpoint': '/api/mobile/v1/revoke',
        'token_expires_in': MOBILE_TOKEN_TTL, 'grant_expires_in': MOBILE_GRANT_TTL,
        'poll_interval': MOBILE_POLL_INTERVAL, 'stream_format': 'application/x-ndjson',
        'e2ee_supported': True,
        'encryption_mode': 'server_managed_at_rest',
        'allowed_endpoints': [
            {'path': rule.rule, 'methods': sorted(MOBILE_ENDPOINT_METHODS[rule.endpoint])}
            for rule in app.url_map.iter_rules() if rule.endpoint in MOBILE_ENDPOINT_METHODS
        ],
    })


@app.route('/api/mobile/v1/device', methods=['POST'])
def mobile_device():
    body = request.get_json()
    if body.get('client_id') != 'official-android':
        return _mobile_error('invalid_client')
    label = body.get('device_name', 'Android')
    if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80 or re.search(r'[\x00-\x1f\x7f]', label):
        return _mobile_error('invalid_device_name')
    try:
        # Fail closed when Redis is unavailable (unlike general app rate limits).
        key = 'mobile:limit:device:' + _mobile_digest(get_client_ip() or 'unknown')
        count = redis_conn.incr(key)
        if count == 1:
            redis_conn.expire(key, 600)
        if count > 10:
            return _mobile_error('rate_limited', 429, 600)
        device_code = secrets.token_urlsafe(32)
        user_code = secrets.token_hex(6).upper()
        grant_key = 'mobile:grant:' + _mobile_digest(device_code)
        code_key = 'mobile:code:' + _mobile_digest(user_code)
        pending = json.dumps({'status': 'pending', 'device_name': label.strip()})
        # Index collisions are rejected, never overwrite another request.
        if not redis_conn.set(code_key, grant_key, ex=MOBILE_GRANT_TTL, nx=True):
            return _mobile_error('temporarily_unavailable', 503)
        redis_conn.set(grant_key, pending, ex=MOBILE_GRANT_TTL)
    except Exception:
        return _mobile_error('temporarily_unavailable', 503)
    return jsonify({'device_code': device_code, 'user_code': user_code,
                    'verification_uri': '/android/connect', 'expires_in': MOBILE_GRANT_TTL,
                    'interval': MOBILE_POLL_INTERVAL})


def _normalize_mobile_user_code(value):
    code = re.sub(r'[\s-]', '', str(value or '')).upper()
    return code if re.fullmatch(r'[0-9A-F]{12}', code) else ''


def _mobile_grant_for_user_code(code):
    if not code:
        return None
    code_key = 'mobile:code:' + _mobile_digest(code)
    grant_key = redis_conn.get(code_key)
    raw = redis_conn.get(grant_key) if grant_key else None
    grant = json.loads(raw) if raw else None
    if not grant or grant.get('status') != 'pending':
        return None
    return grant_key, grant


@app.route('/android/connect', methods=['GET', 'POST'])
def mobile_connect():
    requested_code = _normalize_mobile_user_code(request.args.get('code'))
    if not current_user.is_authenticated:
        if requested_code:
            session['mobile_connect_code'] = requested_code
        else:
            session.pop('mobile_connect_code', None)
        session['mobile_connect_pending'] = True
        return redirect(url_for('login'))
    if not current_user.is_setup_completed:
        if requested_code:
            session['mobile_connect_code'] = requested_code
        session['mobile_connect_pending'] = True
        return redirect(url_for('setup'))
    session.pop('mobile_connect_pending', None)
    stored_code = _normalize_mobile_user_code(session.pop('mobile_connect_code', ''))
    code = requested_code if request.args.get('code') is not None else stored_code
    if request.method == 'GET':
        error = None
        device_name = None
        if request.args.get('code') and not requested_code:
            error = '連携リンクの確認コードが正しくありません。アプリからもう一度開いてください。'
        elif code:
            try:
                pending = _mobile_grant_for_user_code(code)
                if pending:
                    _, grant = pending
                    device_name = grant.get('device_name') or 'Android'
                else:
                    error = 'コードが無効、使用済み、または期限切れです。アプリから新しく連携を開始してください。'
            except Exception:
                error = '連携サービスに接続できません。しばらくしてからやり直してください。'
        return render_template('android_connect.html', error=error, user_code=code, device_name=device_name)
    if not rate_limit('mobile:approve:' + str(current_user.id), 20, 600):
        return render_template('android_connect.html', error='試行回数が多いため、10分後にやり直してください。'), 429
    code = _normalize_mobile_user_code(request.form.get('user_code'))
    if not code:
        return render_template('android_connect.html', error='12文字の確認コードを入力してください。'), 400
    code_key = 'mobile:code:' + _mobile_digest(code)
    new_session = None
    try:
        grant_key = redis_conn.get(code_key)
        raw = redis_conn.get(grant_key) if grant_key else None
        grant = json.loads(raw) if raw else None
        if not grant or grant.get('status') != 'pending':
            return render_template('android_connect.html', error='コードが無効、使用済み、または期限切れです。'), 400
        decision = request.form.get('decision')
        if decision not in {'approve', 'deny'}:
            return render_template('android_connect.html', user_code=code, device_name=grant['device_name'])
        result = {'status': 'denied'}
        if decision == 'approve':
            token = MOBILE_TOKEN_PREFIX + secrets.token_urlsafe(32)
            new_session = UserSession(
                user_id=current_user.id, session_id='android:' + _mobile_digest(token),
                user_agent='Official Android: ' + grant['device_name'], ip_address=get_client_ip(),
            )
            db.session.add(new_session)
            safe_db_commit()
            result = {'status': 'approved', 'encrypted_token': encrypt_val(token),
                      'expires_at': int(time.time()) + MOBILE_TOKEN_TTL}
        approved = redis_conn.eval(_MOBILE_APPROVE_SCRIPT, 1, grant_key, json.dumps(result))
        if not approved:
            if new_session:
                new_session.is_revoked = True
                new_session.revoked_at = datetime.utcnow()
                safe_db_commit()
            return render_template('android_connect.html', error='コードが使用済み、または期限切れです。'), 400
        redis_conn.delete(code_key)
        return render_template('android_connect.html', completed=True, approved=decision == 'approve')
    except Exception:
        db.session.rollback()
        if new_session is not None:
            # Never leave a usable credential behind after incomplete approval.
            new_session.is_revoked = True
            new_session.revoked_at = datetime.utcnow()
            safe_db_commit()
        return render_template('android_connect.html', error='連携サービスに接続できません。しばらくしてからやり直してください。'), 503


@app.route('/api/mobile/v1/token', methods=['POST'])
def mobile_token():
    body = request.get_json()
    code = body.get('device_code')
    if body.get('client_id') != 'official-android':
        return _mobile_error('invalid_client')
    if not isinstance(code, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', code):
        return _mobile_error('invalid_grant')
    digest = _mobile_digest(code)
    try:
        ip_key = 'mobile:limit:poll:' + _mobile_digest(get_client_ip() or 'unknown')
        count = redis_conn.incr(ip_key)
        if count == 1:
            redis_conn.expire(ip_key, 60)
        if count > 120:
            return _mobile_error('rate_limited', 429, 60)
        if not redis_conn.set('mobile:poll:' + digest, '1', ex=MOBILE_POLL_INTERVAL, nx=True):
            return _mobile_error('slow_down', 429)
        raw = redis_conn.eval(_MOBILE_REDEEM_SCRIPT, 1, 'mobile:grant:' + digest)
        if not raw:
            return _mobile_error('expired_token')
        grant = json.loads(raw)
        if grant['status'] == 'pending':
            return _mobile_error('authorization_pending')
        if grant['status'] == 'denied':
            return _mobile_error('access_denied')
        token = decrypt_val(grant['encrypted_token'])
        if not token or not token.startswith(MOBILE_TOKEN_PREFIX):
            return _mobile_error('temporarily_unavailable', 503)
        return jsonify({'access_token': token, 'token_type': 'Bearer',
                        'expires_in': max(0, grant['expires_at'] - int(time.time())),
                        'scope': 'chat files', 'api_version': 1})
    except Exception:
        return _mobile_error('temporarily_unavailable', 503)


@app.route('/api/mobile/v1/me')
def mobile_me():
    models = [_mobile_model_metadata(model_id) for model_id in sorted(ALL_VALID_MODEL_IDS)]
    return jsonify({'id': current_user.id, 'username': current_user.username,
                    'expires_at': (g.mobile_session.created_at + timedelta(seconds=MOBILE_TOKEN_TTL)).isoformat() + 'Z',
                    'e2ee_enabled': bool(current_user.enable_e2ee),
                    'encryption_mode': 'server_managed_at_rest',
                    'default_model': current_user.default_model,
                    'model_ids': [model['id'] for model in models],
                    'model_catalog_version': 1,
                    'models': models})


@app.route('/api/mobile/v1/revoke', methods=['POST'])
def mobile_revoke():
    g.mobile_session.is_revoked = True
    g.mobile_session.revoked_at = datetime.utcnow()
    safe_db_commit()
    return jsonify({'status': 'revoked'})


# Settings shared with the Web settings modal. Provider API keys are write-only
# (masked on read, like Web). MCP OAuth secrets stay on Web.
_MOBILE_PREFERENCE_BOOLS = (
    'default_enable_thinking', 'default_enable_search', 'enter_to_send',
    'light_mode_enabled', 'auto_search_on_links', 'default_enable_url_context',
    'default_enable_maps', 'default_enable_python', 'default_enable_file_creation',
    'default_enable_system_prompt', 'default_enable_mcp',
    'use_last_chat_settings', 'voice_studio_ui', 'liquid_glass_enabled',
    'system_prompt_enabled', 'apply_global_system_prompt',
    'apply_auto_system_prompt_notices', 'skip_2fa_on_google_login',
    'rich_paste_prompt_use_custom_default', 'enable_latency_metrics', 'enable_client_debug_log',
)
# Provider credentials follow Web /api/settings: GET returns only the mask, and a
# submitted mask keeps the stored value, so a native client never sees plaintext keys.
_MOBILE_SECRET_FIELDS = (
    ('openai_key', 'openai_api_key'), ('gemini_key', 'gemini_api_key'),
    ('anthropic_key', 'anthropic_api_key'), ('deepseek_key', 'deepseek_api_key'),
    ('kimi_key', 'kimi_api_key'), ('mistral_key', 'mistral_api_key'),
    ('xai_key', 'xai_api_key'), ('google_key', 'google_api_key'),
)
_MOBILE_STT_MODELS = VALID_STT_MODELS
_MOBILE_PROMPT_BAR_MODES = {'normal', 'compact', 'minimal'}
_MOBILE_2FA_METHODS = {'totp', 'webauthn'}


def _mobile_user_system_prompt(user):
    text = user.system_prompt or ""
    if user.enable_e2ee and text:
        text = decrypt_val(text) or ""
    return text


def _mobile_prompt_bar_mode(user):
    if getattr(user, 'minimal_prompt_mode', False):
        return 'minimal'
    if getattr(user, 'compact_prompt_mode', False):
        return 'compact'
    return 'normal'


def _mobile_apply_prompt_bar_mode(mode):
    current_user.compact_prompt_mode = mode == 'compact'
    current_user.minimal_prompt_mode = mode == 'minimal'


def _mobile_global_prompt_status():
    value = get_app_setting("global_system_prompt", "") or ""
    enabled = get_bool_app_setting("global_system_prompt_enabled", True)
    effective = ""
    if enabled:
        effective = str(value) if str(value).strip() else build_global_system_prompt()
    return value, enabled, effective, bool(enabled and not str(value).strip())


def _mobile_preferences_payload():
    user = current_user
    global_prompt_value, global_prompt_enabled, global_prompt_effective, global_prompt_time_fallback = _mobile_global_prompt_status()
    payload = {
        'api_version': 1,
        'username': user.username,
        'google_email': user.google_email or "",
        'minashin_email': user.minashin_email or "",
        'default_model': user.default_model or "gemini-3.6-flash",
        'default_vision_model': user.default_vision_model or "gemini-3-flash-preview",
        'default_enable_thinking': bool(user.default_enable_thinking),
        'default_enable_search': bool(user.default_enable_search),
        'default_enable_url_context': bool(user.default_enable_url_context),
        'default_enable_maps': bool(user.default_enable_maps),
        'default_enable_python': bool(user.default_enable_python),
        'default_enable_file_creation': bool(user.default_enable_file_creation),
        'default_enable_system_prompt': bool(user.default_enable_system_prompt),
        'default_enable_mcp': bool(user.default_enable_mcp) if user.default_enable_mcp is not None else True,
        'enter_to_send': bool(user.enter_to_send),
        'light_mode_enabled': bool(getattr(user, 'light_mode_enabled', False)),
        'liquid_glass_enabled': bool(getattr(user, 'liquid_glass_enabled', False)),
        'auto_search_on_links': bool(user.auto_search_on_links) if user.auto_search_on_links is not None else True,
        'use_last_chat_settings': bool(user.use_last_chat_settings),
        'voice_studio_ui': bool(user.voice_studio_ui) if getattr(user, 'voice_studio_ui', None) is not None else True,
        'compact_prompt_mode': bool(getattr(user, 'compact_prompt_mode', False)),
        'minimal_prompt_mode': bool(getattr(user, 'minimal_prompt_mode', False)),
        'prompt_bar_mode': _mobile_prompt_bar_mode(user),
        'default_thinking_level': user.default_thinking_level or 'high',
        'default_thinking_budget': int(user.default_thinking_budget if user.default_thinking_budget is not None else 4096),
        'default_reasoning_effort': user.default_reasoning_effort or 'medium',
        'default_safety_setting': user.default_safety_setting or 'default',
        'theme_color': normalize_theme_color(user.theme_color or ""),
        'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(user),
        'mic_transcribe_mode': _normalize_mic_transcribe_mode(getattr(user, 'mic_transcribe_mode', None)),
        'stt_model': user.stt_model or "gpt-4o-mini-transcribe",
        'system_prompt': _mobile_user_system_prompt(user),
        'system_prompt_enabled': user.system_prompt_enabled if user.system_prompt_enabled is not None else True,
        'apply_global_system_prompt': user.apply_global_system_prompt if user.apply_global_system_prompt is not None else True,
        'apply_auto_system_prompt_notices': get_user_auto_system_prompt_notices_enabled(user),
        'global_system_prompt': global_prompt_value,
        'global_system_prompt_enabled': global_prompt_enabled,
        'rich_paste_prompt_default': user.rich_paste_prompt_default or "",
        'rich_paste_prompt_use_custom_default': bool(getattr(user, 'rich_paste_prompt_use_custom_default', False)),
        'last_model': user.last_model or "",
        'last_enable_search': bool(user.last_enable_search),
        'last_enable_url_context': bool(user.last_enable_url_context),
        'last_enable_maps': bool(user.last_enable_maps),
        'last_enable_python': bool(user.last_enable_python) if user.last_enable_python is not None else True,
        'last_enable_file_creation': bool(user.last_enable_file_creation) if user.last_enable_file_creation is not None else True,
        'last_enable_thinking': bool(user.last_enable_thinking),
        'last_thinking_level': user.last_thinking_level or 'high',
        'last_thinking_budget': int(user.last_thinking_budget if user.last_thinking_budget is not None else 4096),
        'last_reasoning_effort': user.last_reasoning_effort or 'medium',
        'last_enable_system_prompt': bool(user.last_enable_system_prompt),
        'last_enable_mcp': bool(user.last_enable_mcp) if user.last_enable_mcp is not None else True,
        'last_safety_setting': user.last_safety_setting or 'default',
        'enable_e2ee': bool(user.enable_e2ee),
        'is_2fa_enabled': bool(user.is_2fa_enabled),
        'has_totp': bool(user.totp_secret),
        'has_webauthn': bool(_load_user_webauthn_credentials(user)),
        'skip_2fa_on_google_login': bool(user.skip_2fa_on_google_login),
        'default_2fa_method': user.default_2fa_method or 'totp',
        'session_created_at': g.mobile_session.created_at.isoformat() + 'Z',
        'session_expires_at': (g.mobile_session.created_at + timedelta(seconds=MOBILE_TOKEN_TTL)).isoformat() + 'Z',
        'device_name': (g.mobile_session.user_agent or '').replace('Official Android: ', '').strip() or None,
        'global_system_prompt_effective': global_prompt_effective,
        'global_system_prompt_uses_time_fallback': global_prompt_time_fallback,
        'auto_system_prompt_notices_config': get_user_auto_system_prompt_notices_config(user),
        'llm_transcribe_prompt': _normalize_llm_transcribe_prompt(getattr(user, 'llm_transcribe_prompt', None)) or "",
        'llm_transcribe_prompt_default': DEFAULT_LLM_TRANSCRIBE_PROMPT,
        'enable_latency_metrics': bool(user.enable_latency_metrics),
        'enable_client_debug_log': bool(user.enable_client_debug_log),
        'passkey_only_login': bool(user.passkey_only_login),
        'model_api_keys': {model_key: _SECRET_MASK for model_key in _load_user_model_api_key_map(user)},
        'gemini_backend': _normalize_gemini_backend(user.gemini_backend),
        'gemini_vertex_project': decrypt_val(user.gemini_vertex_project) or "",
        'gemini_vertex_location': _normalize_gemini_vertex_location(user.gemini_vertex_location),
        'gemini_vertex_credentials_json': _masked_secret(user.gemini_vertex_credentials_json),
        'google_project': decrypt_val(user.google_cloud_project) or "",
        'is_admin': bool(getattr(user, 'is_admin', False)),
        'google_linked': bool(user.google_id),
        'minashin_linked': bool(user.minashin_sub),
    }
    try:
        status = redis_conn.get(f"migration_status:{user.id}")
        progress = redis_conn.get(f"migration_progress:{user.id}")
        payload['migration_status'] = status.decode() if status else "idle"
        payload['migration_progress'] = progress.decode() if progress else ""
    except Exception:
        payload['migration_status'] = "idle"
        payload['migration_progress'] = ""
    for field, column in _MOBILE_SECRET_FIELDS:
        payload[field] = _masked_secret(getattr(user, column, None))
    return payload


def _mobile_apply_provider_settings(data):
    """Web /api/settings credential rules; returns an error code or None."""
    for field, _column in _MOBILE_SECRET_FIELDS + (('google_project', None), ('gemini_vertex_project', None)):
        if field in data and len(str(data.get(field) or '')) > 4096:
            return f'{field}_too_large'
    if 'gemini_vertex_credentials_json' in data and len(str(data.get('gemini_vertex_credentials_json') or '')) > 100_000:
        return 'gemini_vertex_credentials_json_too_large'
    if 'gemini_vertex_credentials_json' in data and data['gemini_vertex_credentials_json'] != _SECRET_MASK:
        try:
            normalized_vertex_json = _normalize_gemini_vertex_credentials_json(data['gemini_vertex_credentials_json'])
        except ValueError as e:
            return str(e)
        current_user.gemini_vertex_credentials_json = encrypt_val(normalized_vertex_json)
    for field, column in _MOBILE_SECRET_FIELDS:
        if field in data and data[field] != _SECRET_MASK:
            setattr(current_user, column, encrypt_val(str(data[field] or '')))
    if 'model_api_keys' in data:
        _merge_masked_model_api_key_map(current_user, data.get('model_api_keys'))
    if 'gemini_backend' in data:
        current_user.gemini_backend = _normalize_gemini_backend(data['gemini_backend'])
    if 'gemini_vertex_project' in data:
        current_user.gemini_vertex_project = encrypt_val(str(data['gemini_vertex_project'] or ''))
    if 'gemini_vertex_location' in data:
        current_user.gemini_vertex_location = _normalize_gemini_vertex_location(data['gemini_vertex_location'])
    if 'google_project' in data:
        current_user.google_cloud_project = encrypt_val(str(data['google_project'] or ''))
    return None


@app.route('/api/mobile/v1/preferences', methods=['GET', 'PUT'])
def mobile_preferences():
    if request.method == 'GET':
        return jsonify(_mobile_preferences_payload())
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return _mobile_error('invalid_request')
    if 'default_model' in data and data.get('default_model') not in ALL_VALID_MODEL_IDS:
        return _mobile_error('invalid_default_model')
    if 'default_vision_model' in data and data.get('default_vision_model') not in ALL_VALID_MODEL_IDS:
        return _mobile_error('invalid_default_vision_model')
    if 'default_model' in data:
        current_user.default_model = data['default_model']
    if 'default_vision_model' in data:
        current_user.default_vision_model = data['default_vision_model']
    for key in _MOBILE_PREFERENCE_BOOLS:
        if key in data:
            setattr(current_user, key, bool(data[key]))
    if 'prompt_bar_mode' in data:
        mode = str(data.get('prompt_bar_mode') or '').strip()
        if mode not in _MOBILE_PROMPT_BAR_MODES:
            return _mobile_error('invalid_prompt_bar_mode')
        _mobile_apply_prompt_bar_mode(mode)
    else:
        if 'compact_prompt_mode' in data:
            current_user.compact_prompt_mode = bool(data['compact_prompt_mode'])
        if 'minimal_prompt_mode' in data:
            current_user.minimal_prompt_mode = bool(data['minimal_prompt_mode'])
        if current_user.minimal_prompt_mode:
            current_user.compact_prompt_mode = False
    if 'default_thinking_level' in data and str(data.get('default_thinking_level')) in {'minimal', 'low', 'medium', 'high'}:
        current_user.default_thinking_level = str(data['default_thinking_level'])
    if 'default_thinking_budget' in data:
        try:
            current_user.default_thinking_budget = max(0, min(32768, int(data['default_thinking_budget'])))
        except (TypeError, ValueError):
            return _mobile_error('invalid_thinking_budget')
    if 'default_reasoning_effort' in data and str(data.get('default_reasoning_effort')) in {'none', 'low', 'medium', 'high', 'xhigh', 'max'}:
        current_user.default_reasoning_effort = str(data['default_reasoning_effort'])
    if 'default_safety_setting' in data and str(data.get('default_safety_setting')) in {'default', 'none'}:
        current_user.default_safety_setting = str(data['default_safety_setting'])
    if 'theme_color' in data:
        current_user.theme_color = normalize_theme_color(data.get('theme_color'))
    if 'temp_chat_timeout_seconds' in data:
        current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(
            data.get('temp_chat_timeout_seconds')
        )
    if 'mic_transcribe_mode' in data:
        current_user.mic_transcribe_mode = _normalize_mic_transcribe_mode(data.get('mic_transcribe_mode'))
    if 'stt_model' in data:
        stt = str(data.get('stt_model') or '').strip()
        if stt not in _MOBILE_STT_MODELS:
            return _mobile_error('invalid_stt_model')
        current_user.stt_model = stt
    if 'default_2fa_method' in data:
        method = str(data.get('default_2fa_method') or '').strip()
        if method not in _MOBILE_2FA_METHODS:
            return _mobile_error('invalid_default_2fa_method')
        current_user.default_2fa_method = method
    if 'system_prompt' in data:
        if not isinstance(data.get('system_prompt'), str):
            return _mobile_error('invalid_system_prompt')
        text = data['system_prompt'].replace('\x00', '')
        if len(text) > 500_000:
            return _mobile_error('system_prompt_too_long')
        current_user.system_prompt = encrypt_val(text) if current_user.enable_e2ee else text
    if 'rich_paste_prompt_default' in data:
        if not isinstance(data.get('rich_paste_prompt_default'), str):
            return _mobile_error('invalid_rich_paste_prompt')
        prompt = data['rich_paste_prompt_default'].replace('\x00', '')[:20_000]
        current_user.rich_paste_prompt_default = prompt
    if 'llm_transcribe_prompt' in data:
        if len(str(data.get('llm_transcribe_prompt') or '')) > 100_000:
            return _mobile_error('llm_transcribe_prompt_too_large')
        current_user.llm_transcribe_prompt = _normalize_llm_transcribe_prompt(data.get('llm_transcribe_prompt'))
    if 'auto_system_prompt_notices_config' in data:
        if not isinstance(data.get('auto_system_prompt_notices_config'), dict):
            return _mobile_error('invalid_auto_system_prompt_notices_config')
        set_user_auto_system_prompt_notices_config(current_user, data['auto_system_prompt_notices_config'])
    if 'passkey_only_login' in data:
        target = bool(data['passkey_only_login'])
        if target and not _load_user_webauthn_credentials(current_user):
            return _mobile_error('passkey_required')
        current_user.passkey_only_login = target
    provider_error = _mobile_apply_provider_settings(data)
    if provider_error:
        db.session.rollback()
        return _mobile_error(provider_error)
    safe_db_commit()
    return jsonify(_mobile_preferences_payload())
