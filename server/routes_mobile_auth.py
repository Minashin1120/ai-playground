"""Native Android authentication and first-run account setup endpoints.

The Android client uses short-lived public JSON requests for signup/login and
then a normal UserSession-backed bearer token for setup and API access.  The
existing browser pairing flow is intentionally left in routes_mobile.py as a
deprecated compatibility path.
"""

_MOBILE_AUTH_TX_TTL = 300
_MOBILE_NATIVE_CODE_TTL = 300
_MOBILE_SETUP_SECRET_FIELDS = {
    'openai_api_key': 'openai_api_key',
    'gemini_api_key': 'gemini_api_key',
    'anthropic_api_key': 'anthropic_api_key',
    'deepseek_api_key': 'deepseek_api_key',
    'kimi_api_key': 'kimi_api_key',
    'mistral_api_key': 'mistral_api_key',
    'xai_api_key': 'xai_api_key',
    'google_api_key': 'google_api_key',
    'google_cloud_project': 'google_cloud_project',
    'gemini_vertex_project': 'gemini_vertex_project',
    'gemini_vertex_credentials_json': 'gemini_vertex_credentials_json',
}

_MOBILE_INTEGRITY_PACKAGE = 'com.minashin1120.aiplayground'
_MOBILE_INTEGRITY_TTL_MS = 2 * 60 * 1000


def _mobile_integrity_hash(request_id, endpoint, body):
    relevant = {key: value for key, value in body.items()
                if key not in {'integrity_token', 'integrity_request_id', 'integrity_turnstile_ticket'}}
    canonical_body = json.dumps(relevant, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    body_hash = base64.urlsafe_b64encode(hashlib.sha256(canonical_body.encode('utf-8')).digest()).rstrip(b'=').decode('ascii')
    value = f'official-android\nPOST\n{endpoint}\n{request_id}\n{body_hash}'.encode('utf-8')
    return base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode('ascii')


def _mobile_integrity_turnstile_url(challenge):
    return '/android/integrity/turnstile?challenge=' + quote(challenge)


def _mobile_integrity_consume_challenge(key):
    try:
        return redis_conn.getdel(key)
    except AttributeError:
        return redis_conn.eval("local value=redis.call('GET',KEYS[1]); if value then redis.call('DEL',KEYS[1]); end; return value", 1, key)


def _mobile_integrity_consume_ticket(key, endpoint):
    script = "local value=redis.call('GET',KEYS[1]); if value == ARGV[1] then redis.call('DEL',KEYS[1]); return 1; end; return 0"
    return bool(redis_conn.eval(script, 1, key, endpoint))


def _mobile_integrity_decode(token):
    key_file = os.getenv('PLAY_INTEGRITY_SERVICE_ACCOUNT_FILE', '').strip()
    project_number = os.getenv('PLAY_INTEGRITY_CLOUD_PROJECT_NUMBER', '').strip()
    if not key_file or not project_number or not os.path.isfile(key_file):
        raise RuntimeError('Play Integrity server verification is not configured')
    credentials = service_account.Credentials.from_service_account_file(
        key_file, scopes=['https://www.googleapis.com/auth/playintegrity']
    )
    credentials.refresh(google_requests.Request())
    response = requests.post(
        f'https://playintegrity.googleapis.com/v1/{_MOBILE_INTEGRITY_PACKAGE}:decodeIntegrityToken',
        headers={'Authorization': 'Bearer ' + credentials.token},
        json={'integrity_token': token}, timeout=8,
    )
    response.raise_for_status()
    return response.json().get('tokenPayloadExternal') or {}


def _mobile_integrity_gate(body, endpoint):
    """Return a JSON gate response when a native client needs Turnstile.

    Omitting the Integrity fields is treated as an unverifiable device, the same
    as Web login/signup which always require Turnstile.
    """
    ticket = body.get('integrity_turnstile_ticket')
    if isinstance(ticket, str) and 20 <= len(ticket) <= 128:
        key = 'mobile:integrity:turnstile:' + _mobile_digest(ticket)
        if _mobile_integrity_consume_ticket(key, endpoint):
            body['_integrity_turnstile_passed'] = True
            return None
    token = body.get('integrity_token')
    request_id = body.get('integrity_request_id')
    window, threshold = (3600, 6) if endpoint.endswith('/signup') else (300, 12)
    if endpoint.endswith('/totp') or '/2fa/' in endpoint:
        window, threshold = 300, 8
    attempt_key = 'mobile:integrity:attempt:' + _mobile_digest(
        endpoint + ':' + (get_client_ip() or request.remote_addr or 'unknown')
    )
    attempts = redis_conn.incr(attempt_key)
    if attempts == 1:
        redis_conn.expire(attempt_key, window)
    concentrated = attempts > threshold
    risk = True
    if isinstance(token, str) and 20 <= len(token) <= 20_000 and isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{20,80}', request_id):
        try:
            request_key = 'mobile:integrity:req:' + _mobile_digest(request_id)
            if not redis_conn.set(request_key, '1', ex=300, nx=True):
                risk = True
            else:
                payload = _mobile_integrity_decode(token)
                details = payload.get('requestDetails') or {}
                app = payload.get('appIntegrity') or {}
                device = payload.get('deviceIntegrity') or {}
                expected_hash = _mobile_integrity_hash(request_id, endpoint, body)
                package_ok = details.get('requestPackageName') == _MOBILE_INTEGRITY_PACKAGE
                hash_ok = secrets.compare_digest(str(details.get('requestHash') or ''), expected_hash)
                try:
                    timestamp_ok = abs(int(details.get('timestampMillis')) - int(time.time() * 1000)) <= _MOBILE_INTEGRITY_TTL_MS
                except (TypeError, ValueError):
                    timestamp_ok = False
                app_package_ok = app.get('packageName') == _MOBILE_INTEGRITY_PACKAGE
                # A repackaged build keeps the package name but not the signing certificate.
                expected_certs = _mobile_android_cert_digests()
                cert_ok = any(_mobile_sha256_bytes(value) in expected_certs
                              for value in app.get('certificateSha256Digest') or [])
                verdicts = set(device.get('deviceRecognitionVerdict') or [])
                # Sideloaded builds intentionally do not require PLAY_RECOGNIZED or LICENSED.
                risk = concentrated or not (package_ok and app_package_ok and cert_ok and hash_ok and timestamp_ok and 'MEETS_DEVICE_INTEGRITY' in verdicts)
        except Exception:
            logger.info('Play Integrity unavailable or invalid for native auth endpoint %s', endpoint)
            risk = True
    if not risk:
        return None
    challenge = secrets.token_urlsafe(32)
    redis_conn.set('mobile:integrity:challenge:' + _mobile_digest(challenge), endpoint, ex=300, nx=True)
    return jsonify({'code': 'turnstile_required', 'error': 'turnstile_required',
                    'turnstile_url': _mobile_integrity_turnstile_url(challenge), 'expires_in': 300}), 428


@app.route('/android/integrity/turnstile', methods=['GET'])
def mobile_integrity_turnstile():
    challenge = request.args.get('challenge', '')
    key = 'mobile:integrity:challenge:' + _mobile_digest(challenge)
    if not re.fullmatch(r'[A-Za-z0-9_-]{40,60}', challenge) or not redis_conn.get(key):
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    site_key = html.escape(os.getenv('TURNSTILE_SITE_KEY', ''), quote=True)
    csrf_token = html.escape(get_csrf_token(), quote=True)
    action = '/android/integrity/turnstile/verify'
    return Response(f'''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>安全性の確認</title>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<h1>安全性の確認</h1><p>確認後、AI Playgroundへ戻ります。</p><form method="post" action="{action}">
<input type="hidden" name="csrf_token" value="{csrf_token}"><input type="hidden" name="challenge" value="{html.escape(challenge, quote=True)}"><div class="cf-turnstile" data-sitekey="{site_key}"></div>
<button type="submit">続行</button></form>''', mimetype='text/html', headers={'Cache-Control': 'no-store'})


@app.route('/android/integrity/turnstile/verify', methods=['POST'])
def mobile_integrity_turnstile_verify():
    challenge = request.form.get('challenge', '')
    challenge_key = 'mobile:integrity:challenge:' + _mobile_digest(challenge)
    if not re.fullmatch(r'[A-Za-z0-9_-]{40,60}', challenge) or not redis_conn.get(challenge_key):
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    if not verify_turnstile(request.form.get('cf-turnstile-response')):
        return '安全性を確認できませんでした。アプリに戻って再試行してください。', 403
    challenge_action = _mobile_integrity_consume_challenge(challenge_key)
    if not challenge_action:
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    challenge_action = challenge_action.decode('utf-8') if isinstance(challenge_action, bytes) else str(challenge_action)
    if challenge_action not in {
        '/api/mobile/v1/auth/signup', '/api/mobile/v1/auth/login', '/api/mobile/v1/auth/google',
        '/api/mobile/v1/auth/totp', '/api/mobile/v1/auth/exchange',
        '/api/mobile/v1/auth/passkey/options', '/api/mobile/v1/auth/passkey/verify',
        '/api/mobile/v1/auth/2fa/webauthn/options', '/api/mobile/v1/auth/2fa/webauthn/verify',
    }:
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    ticket = secrets.token_urlsafe(32)
    redis_conn.set('mobile:integrity:turnstile:' + _mobile_digest(ticket), challenge_action, ex=300, nx=True)
    return redirect('https://ai.minashin1120.com/android/auth/callback?integrity_ticket=' + quote(ticket), code=303)


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
    return redirect(url_for('mobile_auth_callback', _external=True, _scheme='https', **params))


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
    if user and not rate_limit(f'rl:mobile:login:user:{user.id}', 10, 300):
        return _mobile_error('rate_limited', 429, 300)
    if not user or not user.password_hash or not user.check_password(password):
        return _mobile_error('invalid_credentials', 401)
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
    if not rate_limit(f'rl:mobile:totp:user:{user.id}', 8, 300):
        return _mobile_error('rate_limited', 429, 300)
    try:
        secret = decrypt_val(user.totp_secret) if user.totp_secret else None
    except Exception:
        secret = None
    if not secret or not pyotp.TOTP(secret).verify(code):
        return _mobile_error('invalid_2fa', 401)
    redis_conn.delete(_mobile_auth_tx_key(transaction_id))
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
        'relation': ['delegate_permission/common.handle_all_urls'],
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


# --- Native passkey (WebAuthn) sign-in and 2FA -------------------------------

_MOBILE_PASSKEY_TX_TTL = 300
_MOBILE_SEC_TTL = 600


def _mobile_redis_json(key):
    raw = redis_conn.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _mobile_android_origins():
    """Allowed WebAuthn origins for the official Android app.

    Android Credential Manager reports the origin as
    ``android:apk-key-hash:<base64url(sha256(signing certificate))>``. The App
    Links fingerprints in ``ANDROID_APP_LINK_SHA256`` use the equivalent
    colon-separated hex form, so both representations are derived from one
    configured value. When nothing is configured no Android origin is allowed
    and only the web origin can complete a passkey ceremony.
    """
    return ['android:apk-key-hash:' + base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')
            for digest in _mobile_android_cert_digests()]


def _mobile_sha256_bytes(value):
    """Decode a certificate SHA-256 given as hex (colons allowed) or base64/base64url."""
    text = str(value or '').strip()
    digest_hex = text.replace(':', '')
    if re.fullmatch(r'[0-9a-fA-F]{64}', digest_hex):
        return bytes.fromhex(digest_hex)
    try:
        raw = base64.urlsafe_b64decode(text.replace('+', '-').replace('/', '_') + '=' * (-len(text) % 4))
    except ValueError:
        return None
    return raw if len(raw) == 32 else None


def _mobile_android_cert_digests():
    """Official signing certificate digests configured for App Links."""
    digests = []
    for raw in (os.getenv('ANDROID_APP_LINK_SHA256') or '').split(','):
        digest_hex = raw.strip().replace(':', '')
        if re.fullmatch(r'[0-9a-fA-F]{64}', digest_hex):
            digests.append(bytes.fromhex(digest_hex))
    return digests


def _mobile_expected_origins():
    return [request.url_root.rstrip('/')] + _mobile_android_origins()


def _mobile_passkey_transaction(transaction_id):
    if not isinstance(transaction_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32}', transaction_id):
        return None, None
    return _mobile_redis_json('mobile:auth:pk:' + _mobile_digest(transaction_id)), transaction_id


def _mobile_webauthn_credentials_payload(user):
    creds = _load_user_webauthn_credentials(user)
    return {
        'is_2fa_enabled': bool(user.is_2fa_enabled),
        'has_totp': bool(user.totp_secret),
        'has_webauthn': bool(creds),
        'default_2fa_method': user.default_2fa_method or 'totp',
        'passkey_only_login': bool(user.passkey_only_login),
        'skip_2fa_on_google_login': bool(user.skip_2fa_on_google_login),
        'passkeys': _serialize_public_webauthn_credentials(creds),
    }


@app.route('/api/mobile/v1/auth/passkey/options', methods=['POST'])
def mobile_auth_passkey_options():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/passkey/options')
    if gate: return gate
    if not _mobile_native_rate_limit('passkey', 20, 300):
        return _mobile_error('rate_limited', 429, 300)
    username = str(body.get('username') or '').replace('\x00', '').strip()
    device_name = _mobile_validate_device_name(body.get('device_name'))
    if device_name is None:
        return _mobile_error('invalid_device_name')
    user = User.query.filter_by(username=username).first() if username else None
    creds = _load_user_webauthn_credentials(user) if user else []
    # Like Web, a passkey is a sole login factor only after passkey-only login
    # is enabled; otherwise it remains a second factor after the password.
    if not user or not creds or not user.passkey_only_login:
        # Do not reveal whether the account exists or has passkeys.
        return _mobile_error('passkey_unavailable', 401)
    if not rate_limit(f'rl:mobile:passkey:user:{user.id}', 10, 300):
        return _mobile_error('rate_limited', 429, 300)
    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    transaction_id = secrets.token_urlsafe(24)
    redis_conn.set(
        'mobile:auth:pk:' + _mobile_digest(transaction_id),
        json.dumps({
            'user_id': user.id,
            'device_name': device_name,
            'challenge': base64.b64encode(options.challenge).decode('utf-8'),
        }),
        ex=_MOBILE_PASSKEY_TX_TTL,
    )
    return jsonify({
        'status': 'ok',
        'transaction_id': transaction_id,
        'expires_in': _MOBILE_PASSKEY_TX_TTL,
        'public_key': json.loads(options_to_json(options)),
    })


@app.route('/api/mobile/v1/auth/passkey/verify', methods=['POST'])
def mobile_auth_passkey_verify():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/passkey/verify')
    if gate: return gate
    if not _mobile_native_rate_limit('passkey_verify', 30, 300):
        return _mobile_error('rate_limited', 429, 300)
    payload, transaction_id = _mobile_passkey_transaction(body.get('transaction_id'))
    credential = body.get('credential')
    if not payload or not isinstance(credential, dict):
        return _mobile_error('invalid_passkey', 401)
    user = db.session.get(User, int(payload.get('user_id') or 0))
    if not user or not user.passkey_only_login:
        return _mobile_error('invalid_passkey', 401)
    if not rate_limit(f'rl:mobile:passkey:user:{user.id}', 10, 300):
        return _mobile_error('rate_limited', 429, 300)
    try:
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(credential.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
        if not current_cred:
            return _mobile_error('invalid_passkey', 401)
        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=base64.b64decode(payload.get('challenge') or ''),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=_mobile_expected_origins(),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=True,
        )
        current_cred['sign_count'] = verification.new_sign_count
        _save_user_webauthn_credentials(user, creds)
        safe_db_commit()
    except Exception:
        logger.exception('Native passkey verification failed')
        return _mobile_error('invalid_passkey', 401)
    redis_conn.delete('mobile:auth:pk:' + _mobile_digest(transaction_id))
    device_name = _mobile_native_device_name(payload.get('device_name'))
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


@app.route('/api/mobile/v1/auth/2fa/webauthn/options', methods=['POST'])
def mobile_auth_2fa_webauthn_options():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/2fa/webauthn/options')
    if gate: return gate
    if not _mobile_native_rate_limit('2fa_webauthn', 30, 300):
        return _mobile_error('rate_limited', 429, 300)
    user, payload = _mobile_auth_transaction_user(body.get('transaction_id'))
    if not user or not payload:
        return _mobile_error('invalid_2fa', 401)
    creds = _load_user_webauthn_credentials(user)
    if not creds:
        return _mobile_error('no_passkeys', 400)
    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    redis_conn.set(
        'mobile:auth:wa:' + _mobile_digest(str(body.get('transaction_id'))),
        json.dumps({'challenge': base64.b64encode(options.challenge).decode('utf-8')}),
        ex=_MOBILE_AUTH_TX_TTL,
    )
    return jsonify({
        'status': 'ok',
        'transaction_id': body.get('transaction_id'),
        'public_key': json.loads(options_to_json(options)),
    })


@app.route('/api/mobile/v1/auth/2fa/webauthn/verify', methods=['POST'])
def mobile_auth_2fa_webauthn_verify():
    body = request.get_json(silent=True) or {}
    gate = _mobile_integrity_gate(body, '/api/mobile/v1/auth/2fa/webauthn/verify')
    if gate: return gate
    if not _mobile_native_rate_limit('2fa_webauthn_verify', 30, 300):
        return _mobile_error('rate_limited', 429, 300)
    transaction_id = str(body.get('transaction_id') or '')
    credential = body.get('credential')
    user, payload = _mobile_auth_transaction_user(transaction_id)
    stored = _mobile_redis_json('mobile:auth:wa:' + _mobile_digest(transaction_id))
    if not user or not payload or not stored or not isinstance(credential, dict):
        return _mobile_error('invalid_2fa', 401)
    if not rate_limit(f'rl:mobile:webauthn:user:{user.id}', 8, 300):
        return _mobile_error('rate_limited', 429, 300)
    try:
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(credential.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
        if not current_cred:
            return _mobile_error('invalid_2fa', 401)
        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=base64.b64decode(stored.get('challenge') or ''),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=_mobile_expected_origins(),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=False,
        )
        current_cred['sign_count'] = verification.new_sign_count
        _save_user_webauthn_credentials(user, creds)
        safe_db_commit()
    except Exception:
        logger.exception('Native WebAuthn 2FA verification failed')
        return _mobile_error('invalid_2fa', 401)
    redis_conn.delete(_mobile_auth_tx_key(transaction_id))
    redis_conn.delete('mobile:auth:wa:' + _mobile_digest(transaction_id))
    device_name = _mobile_validate_device_name(payload.get('device_name')) or 'Android'
    return _mobile_auth_response(user, _mobile_issue_token(user, device_name), device_name)


# --- Native 2FA / passkey management (bearer authenticated) ------------------

@app.route('/api/mobile/v1/security', methods=['GET'])
def mobile_security():
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/totp/setup', methods=['POST'])
def mobile_security_totp_setup():
    if not rate_limit(f'rl:mobile:totp_setup:user:{current_user.id}', 6, 3600):
        return _mobile_error('rate_limited', 429, 600)
    secret = pyotp.random_base32()
    redis_conn.set('mobile:sec:totp:' + str(current_user.id), json.dumps({'secret': secret}), ex=_MOBILE_SEC_TTL)
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.username, issuer_name='AI Chat Playground')
    # Same QR image as Web `/api/2fa/totp/setup` (rendered by image_tools in a subprocess).
    qr_b64 = base64.b64encode(_qr_png_bytes(uri)).decode()
    return jsonify({'status': 'ok', 'secret': secret, 'otpauth_uri': uri, 'expires_in': _MOBILE_SEC_TTL,
                    'qr_image': f"data:image/png;base64,{qr_b64}"})


@app.route('/api/mobile/v1/security/totp/enable', methods=['POST'])
def mobile_security_totp_enable():
    if not rate_limit(f'rl:mobile:totp_enable:user:{current_user.id}', 10, 300):
        return _mobile_error('rate_limited', 429, 300)
    body = request.get_json(silent=True) or {}
    code = re.sub(r'\s+', '', str(body.get('code') or ''))
    stored = _mobile_redis_json('mobile:sec:totp:' + str(current_user.id))
    secret = (stored or {}).get('secret')
    if not secret or not re.fullmatch(r'\d{6,8}', code):
        return _mobile_error('totp_setup_required', 400)
    if not pyotp.TOTP(secret).verify(code):
        return _mobile_error('invalid_code', 401)
    current_user.totp_secret = encrypt_val(secret)
    current_user.is_2fa_enabled = True
    if not current_user.default_2fa_method:
        current_user.default_2fa_method = 'totp'
    redis_conn.delete('mobile:sec:totp:' + str(current_user.id))
    safe_db_commit()
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/totp/disable', methods=['POST'])
def mobile_security_totp_disable():
    if not current_user.totp_secret:
        return _mobile_error('totp_not_registered', 400)
    body = request.get_json(silent=True) or {}
    code = re.sub(r'\s+', '', str(body.get('code') or ''))
    try:
        secret = decrypt_val(current_user.totp_secret)
    except Exception:
        secret = None
    if not secret or not re.fullmatch(r'\d{6,8}', code) or not pyotp.TOTP(secret).verify(code):
        return _mobile_error('invalid_code', 401)
    current_user.totp_secret = None
    _refresh_user_2fa_state(current_user)
    safe_db_commit()
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/passkeys/options', methods=['POST'])
def mobile_security_passkey_options():
    if not rate_limit(f'rl:mobile:passkey_reg:user:{current_user.id}', 10, 600):
        return _mobile_error('rate_limited', 429, 600)
    existing = _load_user_webauthn_credentials(current_user)
    options_kwargs = {
        'rp_name': 'AI Chat Playground',
        'rp_id': request.host.split(':')[0],
        'user_id': str(current_user.id).encode(),
        'user_name': current_user.username,
        'authenticator_selection': AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED,
        ),
    }
    if existing:
        options_kwargs['exclude_credentials'] = [
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in existing
        ]
    options = generate_registration_options(**options_kwargs)
    redis_conn.set(
        'mobile:sec:reg:' + str(current_user.id),
        json.dumps({'challenge': base64.b64encode(options.challenge).decode('utf-8')}),
        ex=_MOBILE_SEC_TTL,
    )
    return jsonify({'status': 'ok', 'public_key': json.loads(options_to_json(options))})


@app.route('/api/mobile/v1/security/passkeys/verify', methods=['POST'])
def mobile_security_passkey_verify():
    if not rate_limit(f'rl:mobile:passkey_reg:user:{current_user.id}', 10, 600):
        return _mobile_error('rate_limited', 429, 600)
    body = request.get_json(silent=True) or {}
    credential = body.get('credential')
    stored = _mobile_redis_json('mobile:sec:reg:' + str(current_user.id))
    if not isinstance(credential, dict) or not stored:
        return _mobile_error('passkey_registration_expired', 400)
    try:
        verification = verify_registration_response(
            credential=credential,
            expected_challenge=base64.b64decode(stored.get('challenge') or ''),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=_mobile_expected_origins(),
            require_user_verification=False,
        )
        creds = _load_user_webauthn_credentials(current_user)
        cred_id = base64.b64encode(verification.credential_id).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
        cred_name = str(body.get('name') or '').strip()[:80] or f'Passkey {len(creds) + 1}'
        public_key = base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
        existing = next((c for c in creds if c['id'] == cred_id), None)
        if existing:
            existing.update(public_key=public_key, sign_count=verification.sign_count, name=cred_name)
        else:
            creds.append({
                'id': cred_id,
                'public_key': public_key,
                'sign_count': verification.sign_count,
                'name': cred_name,
                'created_at': datetime.utcnow().isoformat() + 'Z',
            })
        _save_user_webauthn_credentials(current_user, creds)
        current_user.is_2fa_enabled = True
        if not current_user.default_2fa_method:
            current_user.default_2fa_method = 'webauthn'
    except Exception:
        logger.exception('Native passkey registration failed')
        return _mobile_error('passkey_registration_failed', 400)
    redis_conn.delete('mobile:sec:reg:' + str(current_user.id))
    safe_db_commit()
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/passkeys/remove', methods=['POST'])
def mobile_security_passkey_remove():
    body = request.get_json(silent=True) or {}
    cred_id = str(body.get('id') or '').strip()
    if not cred_id:
        return _mobile_error('id_required')
    creds = _load_user_webauthn_credentials(current_user)
    filtered = [c for c in creds if c['id'] != cred_id]
    if len(filtered) == len(creds):
        return _mobile_error('not_found', 404)
    _save_user_webauthn_credentials(current_user, filtered)
    _refresh_user_2fa_state(current_user)
    safe_db_commit()
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/preferences', methods=['POST'])
def mobile_security_preferences():
    body = request.get_json(silent=True) or {}
    default_method = str(body.get('default_2fa_method') or '').strip()
    if default_method and default_method not in ('totp', 'webauthn'):
        return _mobile_error('invalid_2fa_method')
    if default_method:
        current_user.default_2fa_method = default_method
    if 'passkey_only_login' in body:
        requested = bool(body.get('passkey_only_login'))
        if requested and not _load_user_webauthn_credentials(current_user):
            return _mobile_error('passkey_required', 400)
        current_user.passkey_only_login = requested
    if 'skip_2fa_on_google_login' in body:
        current_user.skip_2fa_on_google_login = bool(body.get('skip_2fa_on_google_login'))
    safe_db_commit()
    return jsonify({'status': 'ok', **_mobile_webauthn_credentials_payload(current_user)})


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
