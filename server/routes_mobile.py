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


def _mobile_model_mode(model_id):
    """Classify Web model IDs without exposing provider credentials or settings."""
    model_id = str(model_id or '').lower()
    if model_id == 'mistral-ocr-4-0':
        return 'ocr'
    if model_id == 'gemini-embedding-2':
        return 'embedding'
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
        capabilities += ['attachments', 'search', 'thinking']
        if provider in {'gemini', 'openai', 'anthropic'}:
            capabilities.append('prompt_cache')
        capabilities += ['python', 'mcp']
    if mode in {'chat', 'image'} and globals().get('_is_batch_model', lambda _model: False)(model_id):
        capabilities.append('batch')
    native_modes = {'chat', 'image', 'video', 'ocr', 'tts', 'transcription', 'agent'}
    return {
        'id': model_id,
        'name': _mobile_model_name(model_id),
        'provider': provider,
        'provider_label': provider_label,
        'mode': mode,
        'capabilities': capabilities,
        'deprecated': deprecated,
        'selectable': mode in native_modes and not deprecated,
    }


@app.route('/api/mobile/v1/config')
def mobile_config():
    return jsonify({
        'api_version': 1, 'system_version': app.config['SYSTEM_VERSION'],
        'client_id': 'official-android', 'auth_flow': 'device_pairing_v1',
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


@app.route('/android/connect', methods=['GET', 'POST'])
def mobile_connect():
    if not current_user.is_authenticated:
        session['mobile_connect_pending'] = True
        return redirect(url_for('login'))
    session.pop('mobile_connect_pending', None)
    if not current_user.is_setup_completed:
        return redirect(url_for('setup'))
    if request.method == 'GET':
        return render_template('android_connect.html')
    if not rate_limit('mobile:approve:' + str(current_user.id), 20, 600):
        return render_template('android_connect.html', error='試行回数が多いため、10分後にやり直してください。'), 429
    code = re.sub(r'[\s-]', '', request.form.get('user_code', '')).upper()
    if not re.fullmatch(r'[0-9A-F]{12}', code):
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


# Only non-secret, Android-relevant preferences are exposed. Provider API keys
# and security settings stay on the Web settings screen.
_MOBILE_PREFERENCE_BOOLS = (
    'default_enable_thinking', 'default_enable_search', 'enter_to_send',
    'light_mode_enabled', 'auto_search_on_links',
)


def _mobile_preferences_payload():
    user = current_user
    return {
        'api_version': 1,
        'username': user.username,
        'default_model': user.default_model or "gemini-3.6-flash",
        'default_enable_thinking': bool(user.default_enable_thinking),
        'default_enable_search': bool(user.default_enable_search),
        'enter_to_send': bool(user.enter_to_send),
        'light_mode_enabled': bool(getattr(user, 'light_mode_enabled', False)),
        'auto_search_on_links': bool(user.auto_search_on_links) if user.auto_search_on_links is not None else True,
        'theme_color': normalize_theme_color(user.theme_color or ""),
        'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(user),
        'enable_e2ee': bool(user.enable_e2ee),
        'is_2fa_enabled': bool(user.is_2fa_enabled),
        'has_totp': bool(user.totp_secret),
        'has_webauthn': bool(_load_user_webauthn_credentials(user)),
        'session_created_at': g.mobile_session.created_at.isoformat() + 'Z',
        'session_expires_at': (g.mobile_session.created_at + timedelta(seconds=MOBILE_TOKEN_TTL)).isoformat() + 'Z',
        'device_name': (g.mobile_session.user_agent or '').replace('Official Android: ', '').strip() or None,
    }


@app.route('/api/mobile/v1/preferences', methods=['GET', 'PUT'])
def mobile_preferences():
    if request.method == 'GET':
        return jsonify(_mobile_preferences_payload())
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return _mobile_error('invalid_request')
    if 'default_model' in data and data.get('default_model') not in ALL_VALID_MODEL_IDS:
        return _mobile_error('invalid_default_model')
    if 'default_model' in data:
        current_user.default_model = data['default_model']
    for key in _MOBILE_PREFERENCE_BOOLS:
        if key in data:
            setattr(current_user, key, bool(data[key]))
    if 'theme_color' in data:
        current_user.theme_color = normalize_theme_color(data.get('theme_color'))
    if 'temp_chat_timeout_seconds' in data:
        current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(
            data.get('temp_chat_timeout_seconds')
        )
    safe_db_commit()
    return jsonify(_mobile_preferences_payload())
