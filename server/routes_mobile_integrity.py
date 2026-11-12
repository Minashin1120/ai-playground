"""Android app integrity gate, Turnstile fallback, and chat Turnstile tickets.

The Play Integrity / Turnstile checks guard the public native auth endpoints.
Every name here is evaluated in the shared server namespace built by app.py.
"""

# region: Integrity verification and gate helpers
_MOBILE_INTEGRITY_PACKAGE = 'com.minashin1120.aiplayground'
# Browser flows on servers other than the official host cannot use the App Link (the Android
# manifest can only verify one fixed host), so the app asks to come back through its own scheme.
# Login codes stay PKCE-bound, so an app that intercepts this redirect cannot redeem them.
_MOBILE_APP_CALLBACK = 'com.minashin1120.aiplayground.auth:/callback'
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


def _mobile_return_mode(value):
    """`return=app` asks a browser flow to come back through the app scheme; anything else is ignored."""
    return 'app' if value == 'app' else None


def _mobile_callback_redirect(params, return_mode=None, code=302):
    """Send the browser back to the app: the https App Link, or the app scheme when requested."""
    if return_mode == 'app':
        query = urlencode(params)
        return redirect(_MOBILE_APP_CALLBACK + ('?' + query if query else ''), code=code)
    return redirect(url_for('mobile_auth_callback', _external=True, _scheme='https', **params), code=code)


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


def _mobile_native_gate_optional():
    """True when the operator set MOBILE_NATIVE_GATE=turnstile_optional and Turnstile is not configured.

    Without either Turnstile or Play Integrity a native login could never pass the gate; the
    official server keeps the default strict behaviour.
    """
    return (os.getenv('MOBILE_NATIVE_GATE', '').strip() == 'turnstile_optional'
            and not os.getenv('TURNSTILE_SECRET_KEY'))


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
    if _mobile_native_gate_optional():
        # Operator opt-out for self-hosted servers without Turnstile or Play Integrity:
        # only the per-IP attempt limit still applies.
        if concentrated:
            return _mobile_error('rate_limit', 429)
        return None
    challenge = secrets.token_urlsafe(32)
    redis_conn.set('mobile:integrity:challenge:' + _mobile_digest(challenge), endpoint, ex=300, nx=True)
    return jsonify({'code': 'turnstile_required', 'error': 'turnstile_required',
                    'turnstile_url': _mobile_integrity_turnstile_url(challenge), 'expires_in': 300}), 428


# region: Turnstile pages
@app.route('/android/integrity/turnstile', methods=['GET'])
def mobile_integrity_turnstile():
    challenge = request.args.get('challenge', '')
    key = 'mobile:integrity:challenge:' + _mobile_digest(challenge)
    if not re.fullmatch(r'[A-Za-z0-9_-]{40,60}', challenge) or not redis_conn.get(key):
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    site_key = html.escape(os.getenv('TURNSTILE_SITE_KEY', ''), quote=True)
    csrf_token = html.escape(get_csrf_token(), quote=True)
    return_field = '<input type="hidden" name="return" value="app">' if _mobile_return_mode(request.args.get('return')) else ''
    action = '/android/integrity/turnstile/verify'
    return Response(f'''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>安全性の確認</title>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<h1>安全性の確認</h1><p>確認後、AI Playgroundへ戻ります。</p><form method="post" action="{action}">
<input type="hidden" name="csrf_token" value="{csrf_token}"><input type="hidden" name="challenge" value="{html.escape(challenge, quote=True)}">{return_field}<div class="cf-turnstile" data-sitekey="{site_key}"></div>
<button type="submit">続行</button></form>''', mimetype='text/html', headers={'Cache-Control': 'no-store'})


@app.route('/android/integrity/turnstile/verify', methods=['POST'])
def mobile_integrity_turnstile_verify():
    challenge = request.form.get('challenge', '')
    return_mode = _mobile_return_mode(request.form.get('return'))
    challenge_key = 'mobile:integrity:challenge:' + _mobile_digest(challenge)
    if not re.fullmatch(r'[A-Za-z0-9_-]{40,60}', challenge) or not redis_conn.get(challenge_key):
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    if not verify_turnstile(request.form.get('cf-turnstile-response')):
        return '安全性を確認できませんでした。アプリに戻って再試行してください。', 403
    challenge_action = _mobile_integrity_consume_challenge(challenge_key)
    if not challenge_action:
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    challenge_action = challenge_action.decode('utf-8') if isinstance(challenge_action, bytes) else str(challenge_action)
    session_user = re.fullmatch(r'session:(\d+)', challenge_action)
    if session_user:
        # A signed-in app asked to pass the chat Turnstile gate; the app redeems
        # this ticket with its own bearer token (mobile_security_turnstile_complete).
        ticket = secrets.token_urlsafe(32)
        redis_conn.set(_mobile_session_turnstile_key(ticket), session_user.group(1), ex=300, nx=True)
        return _mobile_callback_redirect({'turnstile_ticket': ticket}, return_mode, code=303)
    if challenge_action not in {
        '/api/mobile/v1/auth/signup', '/api/mobile/v1/auth/login', '/api/mobile/v1/auth/google',
        '/api/mobile/v1/auth/totp', '/api/mobile/v1/auth/exchange',
        '/api/mobile/v1/auth/passkey/options', '/api/mobile/v1/auth/passkey/verify',
        '/api/mobile/v1/auth/2fa/webauthn/options', '/api/mobile/v1/auth/2fa/webauthn/verify',
    }:
        return '確認リンクの期限が切れました。アプリからやり直してください。', 410
    ticket = secrets.token_urlsafe(32)
    redis_conn.set('mobile:integrity:turnstile:' + _mobile_digest(ticket), challenge_action, ex=300, nx=True)
    return _mobile_callback_redirect({'integrity_ticket': ticket}, return_mode, code=303)


# region: Signed-in chat Turnstile tickets
def _mobile_session_turnstile_key(ticket):
    return 'mobile:turnstile:session:' + _mobile_digest(str(ticket))


@app.route('/api/mobile/v1/security/turnstile', methods=['POST'])
def mobile_security_turnstile():
    """Start the chat Turnstile check for a signed-in app (Web shows the widget in place).

    The app opens the returned page in the browser; after Turnstile it comes back
    with a one-time ticket that only this account can redeem.
    """
    if not _bot_turnstile_active():
        return jsonify({'status': 'ok', 'skipped': True})
    if current_user.is_bot_banned:
        return _mobile_error('banned', 403)
    if _bot_turnstile_verified():
        return jsonify({'status': 'ok', 'already': True})
    if not rate_limit(f'rl:mobile:turnstile:user:{current_user.id}', 10, 300):
        return _mobile_error('rate_limit', 429)
    challenge = secrets.token_urlsafe(32)
    redis_conn.set('mobile:integrity:challenge:' + _mobile_digest(challenge), f'session:{current_user.id}', ex=300, nx=True)
    return jsonify({'turnstile_url': _mobile_integrity_turnstile_url(challenge), 'expires_in': 300})


@app.route('/api/mobile/v1/security/turnstile/complete', methods=['POST'])
def mobile_security_turnstile_complete():
    body = request.get_json(silent=True) or {}
    ticket = body.get('ticket')
    if not isinstance(ticket, str) or not re.fullmatch(r'[A-Za-z0-9_-]{40,60}', ticket):
        return _mobile_error('turnstile_expired', 410)
    owner = _mobile_integrity_consume_challenge(_mobile_session_turnstile_key(ticket))
    owner = owner.decode('utf-8') if isinstance(owner, bytes) else owner
    if owner != str(current_user.id):
        return _mobile_error('turnstile_expired', 410)
    if not _bot_turnstile_active():
        return jsonify({'status': 'ok', 'skipped': True})
    if current_user.is_bot_banned:
        return _mobile_error('banned', 403)
    if _bot_turnstile_register_success():
        return _mobile_error('banned', 403)
    _log_bot_evidence('verify_ok', details='android')
    return jsonify({'status': 'ok'})

