"""Native Android authentication and first-run account setup endpoints.

The Android client uses short-lived public JSON requests for signup/login and
then a normal UserSession-backed bearer token for setup and API access.  The
existing browser pairing flow is intentionally left in routes_mobile.py as a
deprecated compatibility path.
"""

_MOBILE_AUTH_TX_TTL = 300
_MOBILE_AUTH_TX_MAX_FAILURES = 5
_MOBILE_NATIVE_CODE_TTL = 300
_MOBILE_SETUP_SECRET_FIELDS = {
    'openai_api_key': 'openai_api_key',
    'gemini_api_key': 'gemini_api_key',
    'gemini_free_api_key': 'gemini_free_api_key',
    'anthropic_api_key': 'anthropic_api_key',
    'deepseek_api_key': 'deepseek_api_key',
    'kimi_api_key': 'kimi_api_key',
    'mistral_api_key': 'mistral_api_key',
    'ideogram_api_key': 'ideogram_api_key',
    'zai_api_key': 'zai_api_key',
    'xai_api_key': 'xai_api_key',
    'google_api_key': 'google_api_key',
    'google_cloud_project': 'google_cloud_project',
    'gemini_vertex_project': 'gemini_vertex_project',
    'gemini_vertex_credentials_json': 'gemini_vertex_credentials_json',
}


def _mobile_auth_tx_key(transaction_id):
    return 'mobile:auth:tx:' + _mobile_digest(str(transaction_id))


def _mobile_auth_transaction(user, device_name):
    transaction_id = secrets.token_urlsafe(24)
    redis_conn.set(
        _mobile_auth_tx_key(transaction_id),
        json.dumps({'user_id': user.id, 'device_name': device_name}),
        ex=_MOBILE_AUTH_TX_TTL,
    )
    return transaction_id


def _mobile_auth_transaction_user(transaction_id):
    if not isinstance(transaction_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32}', transaction_id):
        return None, None
    raw = redis_conn.get(_mobile_auth_tx_key(transaction_id))
    if not raw:
        return None, None
    try:
        payload = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
        user = db.session.get(User, int(payload.get('user_id')))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, None
    return user, payload


def _mobile_native_code_key(code):
    return 'mobile:auth:code:' + _mobile_digest(str(code))


def _mobile_consume_native_code(code):
    key = _mobile_native_code_key(code)
    try:
        return redis_conn.getdel(key)
    except AttributeError:
        # Redis versions predating GETDEL are still supported; the code is
        # random, short-lived, and only accepted once in the normal path.
        raw = redis_conn.get(key)
        if raw:
            redis_conn.delete(key)
        return raw


def _mobile_native_auth_code(user, device_name, provider=None, code_challenge=None):
    code = secrets.token_urlsafe(32)
    redis_conn.set(
        _mobile_native_code_key(code),
        json.dumps({'user_id': user.id, 'device_name': device_name, 'provider': provider,
                    'code_challenge': code_challenge}),
        ex=_MOBILE_NATIVE_CODE_TTL,
    )
    return code


def _mobile_native_code_challenge(args):
    """PKCE (S256) challenge sent by the app when it opens a browser login."""
    challenge = str(args.get('code_challenge') or '')
    if args.get('code_challenge_method', 'S256') != 'S256' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', challenge):
        return None
    return challenge


def _mobile_native_code_verified(payload, verifier):
    """Bind the App Link code to the app that started the browser login.

    Codes minted without a challenge are accepted only from clients that send no
    verifier (Android 1.13.67), so they cannot be injected into newer clients.
    """
    challenge = payload.get('code_challenge')
    if not challenge and verifier is None:
        return True
    if not challenge or not isinstance(verifier, str) or not re.fullmatch(r'[A-Za-z0-9._~-]{43,128}', verifier):
        return False
    computed = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode('ascii')
    return secrets.compare_digest(computed, str(challenge))


def _mobile_native_redirect(code=None, error=None):
    params = {}
    if code:
        params['code'] = code
    if error:
        params['error'] = error
    return _mobile_callback_redirect(params, session.pop('mobile_native_return', None))


def _mobile_native_start_return(args):
    """Remember how this browser login returns; the app scheme is only allowed with PKCE."""
    return_mode = _mobile_return_mode(args.get('return'))
    session['mobile_native_return'] = return_mode
    if return_mode == 'app' and not session.get('mobile_native_code_challenge'):
        return _mobile_native_redirect(error='pkce_required')
    return None


def _mobile_native_device_name(value):
    return _mobile_validate_device_name(value) or 'Android'


def _mobile_native_rate_limit(kind, limit, window):
    client = get_client_ip() or request.remote_addr or 'unknown'
    return rate_limit(f'rl:mobile:{kind}:{_mobile_digest(client)}', limit, window)


def _mobile_validate_device_name(value):
    value = str(value or 'Android').strip()
    if not 1 <= len(value) <= 80 or re.search(r'[\x00-\x1f\x7f]', value):
        return None
    return value


def _mobile_issue_token(user, device_name):
    token = MOBILE_TOKEN_PREFIX + secrets.token_urlsafe(32)
    session_id = 'android:' + _mobile_digest(token)
    now = datetime.utcnow()
    db.session.add(UserSession(
        user_id=user.id,
        session_id=session_id,
        user_agent='Official Android: ' + device_name,
        ip_address=get_client_ip(),
        created_at=now,
        last_seen_at=now,
    ))
    safe_db_commit()
    return token


def _mobile_auth_response(user, token, device_name):
    return jsonify({
        'status': 'ok',
        'access_token': token,
        'token_type': 'Bearer',
        'expires_in': MOBILE_TOKEN_TTL,
        'setup_required': not bool(user.is_setup_completed),
        'user': {'id': user.id, 'username': user.username},
        'device_name': device_name,
    })


def _mobile_lookup_credentials(body):
    username = str(body.get('username') or '').replace('\x00', '').strip()
    password = str(body.get('password') or '')
    if not 1 <= len(username) <= 80 or not 1 <= len(password) <= 512:
        return None, None
    return username, password


@app.route('/api/mobile/v1/auth/signup', methods=['POST'])
def mobile_auth_signup():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/signup')
    if gate: return gate
    if not _mobile_native_rate_limit('signup', 10, 3600):
        return _mobile_error('rate_limited', 429, 600)
    username = str(body.get('username') or '').replace('\x00', '').strip()
    password = str(body.get('password') or '')
    device_name = _mobile_validate_device_name(body.get('device_name'))
    if device_name is None:
        return _mobile_error('invalid_device_name')
    if len(username) < 3 or len(username) > 80 or re.search(r'[\x00-\x1f\x7f]', username):
        return _mobile_error('invalid_username')
    if '@' in username:
        return _mobile_error('invalid_username')
    if len(password) < 8 or len(password) > 256:
        return _mobile_error('invalid_password')
    if is_request_banned_identifier() or _is_primary_admin_username(username):
        return _mobile_error('signup_blocked', 403)
    if User.query.filter_by(username=username).first():
        return _mobile_error('username_taken', 409)
    user = User(username=username, is_setup_completed=False)
    user.set_password(password)
    db.session.add(user)
    safe_db_commit()
    token = _mobile_issue_token(user, device_name)
    return _mobile_auth_response(user, token, device_name), 201


@app.route('/api/mobile/v1/auth/login', methods=['POST'])
def mobile_auth_login():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/login')
    if gate: return gate
    if not _mobile_native_rate_limit('login', 20, 300):
        return _mobile_error('rate_limited', 429, 300)
    username, password = _mobile_lookup_credentials(body)
    device_name = _mobile_validate_device_name(body.get('device_name'))
    if not username or not password or device_name is None:
        return _mobile_error('invalid_credentials', 401)
    user = User.query.filter_by(username=username).first()
    if user and (not rate_limit(f'rl:mobile:login:user:{user.id}', 10, 300)
                 or auth_failures_exceeded('password', user.id)):
        return _mobile_error('rate_limited', 429, 300)
    if not user or not user.password_hash or not user.check_password(password):
        if user and user.password_hash:
            record_auth_failure('password', user.id)
        return _mobile_error('invalid_credentials', 401)
    clear_auth_failures('password', user.id)
    if user.is_2fa_enabled:
        transaction_id = _mobile_auth_transaction(user, device_name)
        return jsonify({
            'status': '2fa_required',
            'transaction_id': transaction_id,
            'default_method': user.default_2fa_method or 'totp',
            'expires_in': _MOBILE_AUTH_TX_TTL,
        })
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


@app.route('/api/mobile/v1/auth/google', methods=['POST'])
def mobile_auth_google():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/google')
    if gate: return gate
    if not _mobile_native_rate_limit('google', 20, 300):
        return _mobile_error('rate_limited', 429, 300)
    credential = body.get('id_token')
    request_nonce = body.get('nonce')
    device_name = _mobile_validate_device_name(body.get('device_name'))
    if (not isinstance(credential, str) or not 1 <= len(credential) <= 4096 or
            not isinstance(request_nonce, str) or not 16 <= len(request_nonce) <= 128 or
            device_name is None):
        return _mobile_error('invalid_google_token', 401)
    server_client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    if not server_client_id:
        return _mobile_error('google_login_unavailable', 503)
    try:
        idinfo = id_token.verify_oauth2_token(
            credential, google_requests.Request(), server_client_id
        )
        if idinfo.get('email_verified') is not True:
            return _mobile_error('google_email_not_verified', 401)
        token_nonce = str(idinfo.get('nonce') or '')
        if not token_nonce or not secrets.compare_digest(token_nonce, request_nonce):
            return _mobile_error('invalid_google_token', 401)
        google_id = str(idinfo.get('sub') or '').strip()
        email = str(idinfo.get('email') or '').strip().lower()
        if not google_id or not email or len(email) > 128:
            return _mobile_error('invalid_google_identity', 401)
        user = _resolve_or_create_google_user(google_id, email)
    except Exception:
        logger.exception('Native Google ID token verification failed')
        return _mobile_error('invalid_google_token', 401)
    if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
        return jsonify({
            'status': '2fa_required',
            'transaction_id': _mobile_auth_transaction(user, device_name),
            'default_method': user.default_2fa_method or 'totp',
            'expires_in': _MOBILE_AUTH_TX_TTL,
        })
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


@app.route('/api/mobile/v1/auth/totp', methods=['POST'])
def mobile_auth_totp():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/totp')
    if gate: return gate
    if not _mobile_native_rate_limit('totp', 20, 300):
        return _mobile_error('rate_limited', 429, 300)
    transaction_id = str(body.get('transaction_id') or '')
    code = re.sub(r'\s+', '', str(body.get('code') or ''))
    user, payload = _mobile_auth_transaction_user(transaction_id)
    if not user or not re.fullmatch(r'\d{6,8}', code):
        return _mobile_error('invalid_2fa', 401)
    if not rate_limit(f'rl:mobile:totp:user:{user.id}', 8, 300) or auth_failures_exceeded('totp', user.id):
        return _mobile_error('rate_limited', 429, 300)
    try:
        secret = decrypt_val(user.totp_secret) if user.totp_secret else None
    except Exception:
        secret = None
    if not verify_totp_code(user.id, secret, code):
        record_auth_failure('totp', user.id)
        # A sign-in transaction allows only a few wrong codes; then the password is needed again.
        fail_key = _mobile_auth_tx_key(transaction_id) + ':fail'
        failures = redis_conn.incr(fail_key)
        if failures == 1:
            redis_conn.expire(fail_key, _MOBILE_AUTH_TX_TTL)
        if failures >= _MOBILE_AUTH_TX_MAX_FAILURES:
            redis_conn.delete(_mobile_auth_tx_key(transaction_id), fail_key)
        return _mobile_error('invalid_2fa', 401)
    clear_auth_failures('totp', user.id)
    redis_conn.delete(_mobile_auth_tx_key(transaction_id), _mobile_auth_tx_key(transaction_id) + ':fail')
    device_name = _mobile_validate_device_name(payload.get('device_name')) or 'Android'
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


@app.route('/api/mobile/v1/auth/exchange', methods=['POST'])
def mobile_auth_exchange():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/exchange')
    if gate: return gate
    if not _mobile_native_rate_limit('exchange', 30, 300):
        return _mobile_error('rate_limited', 429, 300)
    code = str(body.get('code') or '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{43}', code):
        return _mobile_error('invalid_auth_code', 401)
    raw = _mobile_consume_native_code(code)
    if not raw:
        return _mobile_error('invalid_auth_code', 401)
    try:
        payload = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
        user = db.session.get(User, int(payload.get('user_id')))
    except (TypeError, ValueError, json.JSONDecodeError):
        user = None
        payload = {}
    if not user or not _mobile_native_code_verified(payload, body.get('code_verifier')):
        return _mobile_error('invalid_auth_code', 401)
    device_name = _mobile_native_device_name(payload.get('device_name'))
    provider = payload.get('provider')
    if user.is_2fa_enabled and not (provider == 'google' and user.skip_2fa_on_google_login):
        return jsonify({
            'status': '2fa_required',
            'transaction_id': _mobile_auth_transaction(user, device_name),
            'default_method': user.default_2fa_method or 'totp',
            'expires_in': _MOBILE_AUTH_TX_TTL,
        })
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


@app.route('/android/auth/callback', methods=['GET'])
def mobile_auth_callback():
    if request.args.get('linked'):
        return "連携が完了しました。AI Playgroundアプリに戻ってください。", 200
    if request.args.get('code'):
        return "認証が完了しました。AI Playgroundアプリに戻ってください。", 200
    return "認証に失敗しました。AI Playgroundアプリで再試行してください。", 400


@app.route('/android/auth/google/start', methods=['GET'])
def mobile_google_start():
    # A sign-in never completes an abandoned settings link started in the same browser.
    session.pop('mobile_native_link_user', None)
    session.pop('mobile_native_link_provider', None)
    session['mobile_native_google'] = True
    session['mobile_native_device_name'] = _mobile_native_device_name(request.args.get('device_name'))
    session['mobile_native_code_challenge'] = _mobile_native_code_challenge(request.args)
    rejected = _mobile_native_start_return(request.args)
    if rejected is not None:
        return rejected
    redirect_uri = url_for('mobile_google_callback', _external=True, _scheme='https')
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/android/auth/google/callback', methods=['GET'])
def mobile_google_callback():
    link_user = _mobile_native_link_target('google')
    device_name = _mobile_native_device_name(session.pop('mobile_native_device_name', None))
    code_challenge = session.pop('mobile_native_code_challenge', None)
    session.pop('mobile_native_google', None)
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info or user_info.get('email_verified') is not True:
            return _mobile_native_redirect(error='google_identity_invalid')
        google_id = str(user_info.get('sub') or '').strip()
        email = str(user_info.get('email') or '').strip().lower()
        if not google_id or not email or len(email) > 128:
            return _mobile_native_redirect(error='google_identity_invalid')
        if link_user is not None:
            return _mobile_native_link_google(link_user, google_id, email)
        user = _resolve_or_create_google_user(google_id, email)
        if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
            # The one-time code is exchanged by the app into its TOTP transaction;
            # no account identifier or session token is placed in the URL.
            return _mobile_native_redirect(code=_mobile_native_auth_code(user, device_name, 'google', code_challenge))
        return _mobile_native_redirect(code=_mobile_native_auth_code(user, device_name, 'google', code_challenge))
    except Exception:
        logger.exception('Native Google login callback failed')
        return _mobile_native_redirect(error='google_login_failed')


@app.route('/android/auth/minashin/start', methods=['GET'])
def mobile_minashin_start():
    # A sign-in never completes an abandoned settings link started in the same browser.
    session.pop('mobile_native_link_user', None)
    session.pop('mobile_native_link_provider', None)
    session['mobile_native_auth'] = True
    session['mobile_native_device_name'] = _mobile_native_device_name(request.args.get('device_name'))
    session['mobile_native_code_challenge'] = _mobile_native_code_challenge(request.args)
    rejected = _mobile_native_start_return(request.args)
    if rejected is not None:
        return rejected
    # Reuse the existing PKCE generator and central-account authorization URL.
    return login_minashin()


def _mobile_minashin_callback_native():
    link_user = _mobile_native_link_target('minashin')
    device_name = _mobile_native_device_name(session.pop('mobile_native_device_name', None))
    code_challenge = session.pop('mobile_native_code_challenge', None)
    session.pop('mobile_native_auth', None)
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        expected_state = session.pop('minashin_oauth_state', None)
        code_verifier = session.pop('minashin_code_verifier', None)
        if not code or not state or not expected_state or not secrets.compare_digest(state, expected_state) or not code_verifier:
            return _mobile_native_redirect(error='minashin_state_invalid')
        client_id, redirect_uri = _minashin_client_identity()
        token_response = requests.post(
            f"{MINASHIN_ACCOUNT_BASE_URL}/oauth/token",
            json={'grant_type': 'authorization_code', 'code': code, 'redirect_uri': redirect_uri,
                  'client_id': client_id, 'code_verifier': code_verifier},
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            timeout=MINASHIN_REQUEST_TIMEOUT,
        )
        if not token_response.ok:
            return _mobile_native_redirect(error='minashin_token_exchange_failed')
        access_token = token_response.json().get('access_token')
        if not access_token:
            return _mobile_native_redirect(error='minashin_token_exchange_failed')
        userinfo_response = requests.get(
            f"{MINASHIN_ACCOUNT_BASE_URL}/api/userinfo",
            headers={'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'},
            timeout=MINASHIN_REQUEST_TIMEOUT,
        )
        if not userinfo_response.ok:
            return _mobile_native_redirect(error='minashin_userinfo_failed')
        user_data = userinfo_response.json()
        sub = str(user_data.get('sub') or '').strip()
        email = str(user_data.get('email') or '').strip().lower()
        if not sub or len(sub) > 128:
            return _mobile_native_redirect(error='minashin_identity_invalid')
        if user_data.get('email_verified') is False or str(user_data.get('email_verified')) == 'False':
            email = ''
        if link_user is not None:
            return _mobile_native_link_minashin(link_user, sub, email)
        user = _resolve_or_create_minashin_user(sub, email, user_data)
        return _mobile_native_redirect(code=_mobile_native_auth_code(user, device_name, 'minashin', code_challenge))
    except Exception:
        logger.exception('Native Minashin login callback failed')
        return _mobile_native_redirect(error='minashin_login_failed')


@app.route('/.well-known/assetlinks.json', methods=['GET'])
def android_assetlinks():
    fingerprints = [item.strip() for item in (os.getenv('ANDROID_APP_LINK_SHA256') or '').split(',') if item.strip()]
    package_name = os.getenv('ANDROID_APP_ID', 'com.minashin1120.aiplayground')
    return jsonify([{
        # get_login_creds lets Credential Manager use this site's passkeys in the app.
        'relation': ['delegate_permission/common.handle_all_urls', 'delegate_permission/common.get_login_creds'],
        'target': {'namespace': 'android_app', 'package_name': package_name, 'sha256_cert_fingerprints': fingerprints},
    }])


def _mobile_setup_payload(user):
    return {
        'status': 'setup_required',
        'username': user.username,
        'default_model': user.default_model or 'gemini-3.6-flash',
        'gemini_backend': _normalize_gemini_backend(user.gemini_backend),
        'gemini_vertex_location': _normalize_gemini_vertex_location(user.gemini_vertex_location),
        'e2ee_enabled': bool(user.enable_e2ee),
        'models': [_mobile_model_metadata(model_id) for model_id in ALL_VALID_MODEL_IDS],
        'import': {
            'web_url': '/setup',
            'supported': True,
            'max_bytes': _ACCOUNT_IMPORT_MAX_BYTES,
            'chunk_bytes': _ACCOUNT_IMPORT_CHUNK_BYTES,
            'endpoints': {
                'start': '/api/account/import/upload/start',
                'chunk': '/api/account/import/upload/<upload_id>/chunk',
                'complete': '/api/account/import/upload/<upload_id>/complete',
                'cancel': '/api/account/import/upload/<upload_id>',
                'import': '/api/account/import',
            },
            'message': 'アカウントのZIPインポートはアプリ内またはWebのセットアップ画面で続行できます。',
        },
    }


@app.route('/api/mobile/v1/setup', methods=['GET', 'PUT'])
def mobile_setup():
    if request.method == 'GET':
        return jsonify(_mobile_setup_payload(current_user))
    if current_user.is_setup_completed:
        # Like Web /setup, first-run values never overwrite a configured account
        # (enable_e2ee here would bypass the encryption migration).
        return _mobile_error('setup_already_completed', 409)
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return _mobile_error('invalid_request')
    default_model = str(body.get('default_model') or 'gemini-3.6-flash')
    if default_model not in ALL_VALID_MODEL_IDS:
        return _mobile_error('invalid_default_model')
    try:
        vertex_credentials_json = _normalize_gemini_vertex_credentials_json(body.get('gemini_vertex_credentials_json'))
    except ValueError:
        return _mobile_error('invalid_vertex_credentials')
    for request_key, field in _MOBILE_SETUP_SECRET_FIELDS.items():
        if request_key not in body:
            continue
        value = str(body.get(request_key) or '').strip()
        if value:
            setattr(current_user, field, encrypt_val(value))
    current_user.gemini_backend = _normalize_gemini_backend(body.get('gemini_backend'))
    current_user.gemini_vertex_location = _normalize_gemini_vertex_location(body.get('gemini_vertex_location'))
    current_user.default_model = default_model
    if vertex_credentials_json:
        current_user.gemini_vertex_credentials_json = encrypt_val(vertex_credentials_json)
    current_user.enable_e2ee = bool(body.get('enable_e2ee'))
    current_user.is_setup_completed = True
    safe_db_commit()
    if current_user.enable_e2ee and _user_has_unencrypted_data(current_user):
        task_queue.enqueue(migrate_e2ee_task, current_user.id, True)
    return jsonify({'status': 'ok', 'setup_required': False, 'default_model': default_model})
