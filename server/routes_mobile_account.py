# Native settings for the Android client: re-authentication, account credentials,
# login sessions, 2FA reset, E2EE switch and easy login. The Web settings modal
# offers the same operations through the cookie session (routes_settings.py,
# routes_account.py, routes_admin.py). Operations that can take over or erase
# the account additionally require a recent re-authentication on this device
# (MOBILE_REAUTH_ENDPOINTS in mobile_auth.py).

_MOBILE_REAUTH_PASSKEY_TTL = 300
_MOBILE_REAUTH_FRESH_LOGIN_SECONDS = 600


def _mobile_reauth_passkey_key():
    return 'mobile:reauth:pk:' + _mobile_digest(g.mobile_session.session_id)


def _mobile_reauth_methods(user):
    methods = []
    if user.password_hash:
        methods.append('password')
    if user.totp_secret:
        methods.append('totp')
    if _load_user_webauthn_credentials(user):
        methods.append('passkey')
    return methods


def _mobile_mark_reauthenticated():
    redis_conn.set(_mobile_reauth_key(g.mobile_session), '1', ex=MOBILE_REAUTH_TTL)


@app.route('/api/mobile/v1/reauth/options', methods=['POST'])
def mobile_reauth_options():
    methods = _mobile_reauth_methods(current_user)
    payload = {'status': 'ok', 'methods': methods, 'reauthenticated': _mobile_recently_reauthenticated()}
    if 'passkey' in methods:
        creds = _load_user_webauthn_credentials(current_user)
        options = generate_authentication_options(
            rp_id=request.host.split(':')[0],
            allow_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds],
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        redis_conn.set(
            _mobile_reauth_passkey_key(),
            json.dumps({'challenge': base64.b64encode(options.challenge).decode('utf-8')}),
            ex=_MOBILE_REAUTH_PASSKEY_TTL,
        )
        payload['public_key'] = json.loads(options_to_json(options))
    if not methods:
        # Accounts without a password, TOTP or passkey (e.g. Google-only) prove
        # themselves by signing in again: a freshly issued token counts.
        fresh = g.mobile_session.created_at + timedelta(seconds=_MOBILE_REAUTH_FRESH_LOGIN_SECONDS) > datetime.utcnow()
        if fresh:
            _mobile_mark_reauthenticated()
        payload['reauthenticated'] = fresh
        payload['sign_in_again'] = not fresh
    return jsonify(payload)


@app.route('/api/mobile/v1/reauth', methods=['POST'])
def mobile_reauth():
    if not rate_limit(f'rl:mobile:reauth:user:{current_user.id}', 10, 300):
        return _mobile_error('rate_limited', 429, 300)
    body = request.get_json(silent=True) or {}
    method = str(body.get('method') or '')
    if method not in _mobile_reauth_methods(current_user):
        return _mobile_error('invalid_method')
    ok = False
    if method == 'password':
        password = str(body.get('password') or '')
        ok = bool(password) and len(password) <= 256 and current_user.check_password(password)
    elif method == 'totp':
        code = re.sub(r'\s+', '', str(body.get('code') or ''))
        try:
            secret = decrypt_val(current_user.totp_secret)
        except Exception:
            secret = None
        ok = bool(secret) and bool(re.fullmatch(r'\d{6,8}', code)) and pyotp.TOTP(secret).verify(code)
    elif method == 'passkey':
        credential = body.get('credential')
        stored = _mobile_redis_json(_mobile_reauth_passkey_key())
        if isinstance(credential, dict) and stored:
            try:
                creds = _load_user_webauthn_credentials(current_user)
                credential_id = str(credential.get('id') or '').strip()
                current_cred = next((c for c in creds if c['id'] == credential_id), None)
                if current_cred:
                    verification = verify_authentication_response(
                        credential=credential,
                        expected_challenge=base64.b64decode(stored.get('challenge') or ''),
                        expected_rp_id=request.host.split(':')[0],
                        expected_origin=_mobile_expected_origins(),
                        credential_public_key=base64url_to_bytes(current_cred['public_key']),
                        credential_current_sign_count=current_cred['sign_count'],
                        require_user_verification=True,
                    )
                    current_cred['sign_count'] = verification.new_sign_count
                    _save_user_webauthn_credentials(current_user, creds)
                    safe_db_commit()
                    ok = True
            except Exception:
                logger.exception('Native re-authentication passkey verification failed')
                ok = False
        redis_conn.delete(_mobile_reauth_passkey_key())
    if not ok:
        return _mobile_error('invalid_credentials', 401)
    _mobile_mark_reauthenticated()
    return jsonify({'status': 'ok', 'expires_in': MOBILE_REAUTH_TTL})


@app.route('/api/mobile/v1/account/credentials', methods=['POST'])
def mobile_account_credentials():
    """Web settings `new_username` / `new_password`; the password needs a re-authentication."""
    body = request.get_json(silent=True) or {}
    new_username = str(body.get('new_username') or '').replace('\x00', '').strip()
    new_password = str(body.get('new_password') or '')
    result = {'status': 'ok', 'username_changed': False, 'password_changed': False}
    if new_password:
        if not _mobile_recently_reauthenticated():
            return _mobile_error('reauth_required', 403)
        if len(new_password) < 8 or len(new_password) > 256:
            return _mobile_error('invalid_password')
        current_user.set_password(new_password)
        revoke_user_sessions(current_user.id, exclude_session_id=g.mobile_session.session_id)
        result['password_changed'] = True
    if new_username and new_username != current_user.username:
        if len(new_username) < 3 or len(new_username) > 80 or re.search(r'[\x00-\x1f\x7f]', new_username):
            return _mobile_error('invalid_username')
        if _is_primary_admin_username(new_username) and not getattr(current_user, 'is_admin', False):
            return _mobile_error('username_unavailable', 409)
        if User.query.filter_by(username=new_username).first():
            return _mobile_error('username_unavailable', 409)
        current_user.username = new_username
        result['username_changed'] = True
    safe_db_commit()
    result['username'] = current_user.username
    return jsonify(result)


@app.route('/api/mobile/v1/account/delete', methods=['POST'])
def mobile_account_delete():
    if getattr(current_user, 'is_admin', False):
        # Web shows the same restriction instead of the delete button.
        return _mobile_error('admin_account', 403)
    try:
        _delete_user_account_immediately(current_user)
    except Exception:
        logger.exception('Native account deletion failed for user %s', getattr(current_user, 'id', None))
        return _mobile_error('delete_failed', 500)
    return jsonify({'status': 'ok'})


@app.route('/api/mobile/v1/account/easy-login', methods=['POST'])
def mobile_easy_login():
    """Web `/api/easy_login`: a one-time password valid for 1–120 minutes, or cancel it."""
    body = request.get_json(silent=True) or {}
    if body.get('cancel'):
        current_user.easy_login_hash = None
        current_user.easy_login_expires_at = None
        safe_db_commit()
        return jsonify({'status': 'ok', 'cancelled': True})
    try:
        minutes = int(body.get('minutes', 5))
    except (TypeError, ValueError):
        minutes = 5
    minutes = max(1, min(120, minutes))
    temp_pw = secrets.token_urlsafe(16)
    current_user.easy_login_hash = generate_password_hash(temp_pw)
    current_user.easy_login_expires_at = datetime.utcnow() + timedelta(minutes=minutes)
    safe_db_commit()
    return jsonify({
        'status': 'ok',
        'temp_password': temp_pw,
        'expires_at': current_user.easy_login_expires_at.isoformat() + 'Z',
        'minutes': minutes,
    })


def _mobile_session_rows():
    return UserSession.query.filter_by(user_id=current_user.id).order_by(UserSession.last_seen_at.desc()).limit(50).all()


@app.route('/api/mobile/v1/sessions', methods=['GET'])
def mobile_sessions():
    current_sid = g.mobile_session.session_id
    return jsonify({'sessions': [
        {
            'id': s.id,
            'created_at': s.created_at.isoformat(),
            'last_seen_at': s.last_seen_at.isoformat() if s.last_seen_at else None,
            'ip_address': s.ip_address,
            'user_agent': s.user_agent,
            'is_current': s.session_id == current_sid,
            'is_revoked': s.is_revoked,
        } for s in _mobile_session_rows()
    ]})


@app.route('/api/mobile/v1/sessions/revoke', methods=['POST'])
def mobile_sessions_revoke():
    body = request.get_json(silent=True) or {}
    try:
        sess_id = int(body.get('id'))
    except (TypeError, ValueError):
        return _mobile_error('id_required')
    row = UserSession.query.filter_by(id=sess_id, user_id=current_user.id).first()
    if not row:
        return _mobile_error('not_found', 404)
    if not row.is_revoked:
        row.is_revoked = True
        row.revoked_at = datetime.utcnow()
        safe_db_commit()
    return jsonify({'status': 'ok', 'logged_out': row.session_id == g.mobile_session.session_id})


@app.route('/api/mobile/v1/sessions/revoke_others', methods=['POST'])
def mobile_sessions_revoke_others():
    revoke_user_sessions(current_user.id, exclude_session_id=g.mobile_session.session_id)
    return jsonify({'status': 'ok'})


@app.route('/api/mobile/v1/sessions/revoke_all', methods=['POST'])
def mobile_sessions_revoke_all():
    revoke_user_sessions(current_user.id, exclude_session_id=None)
    return jsonify({'status': 'ok', 'logged_out': True})


@app.route('/api/mobile/v1/security/2fa/disable', methods=['POST'])
def mobile_security_disable_2fa():
    """Web settings `disable_2fa`: removes TOTP and every passkey."""
    current_user.is_2fa_enabled = False
    current_user.totp_secret = None
    current_user.webauthn_credentials = None
    current_user.passkey_only_login = False
    current_user.default_2fa_method = 'totp'
    safe_db_commit()
    return jsonify({'status': 'ok', 'message': '2FAを無効化しました。', **_mobile_webauthn_credentials_payload(current_user)})


@app.route('/api/mobile/v1/security/e2ee', methods=['POST'])
def mobile_security_e2ee():
    """Web settings `enable_e2ee`: starts the encryption migration when the value changes."""
    body = request.get_json(silent=True) or {}
    if 'enabled' not in body:
        return _mobile_error('enabled_required')
    target = bool(body.get('enabled'))
    if target == bool(current_user.enable_e2ee):
        return jsonify({'status': 'ok', 'message': None})
    task_queue.enqueue(migrate_e2ee_task, current_user.id, target)
    return jsonify({'status': 'ok', 'message': '暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。'})


# --- Google / Minashin linking from the settings Account tab ------------------
# The provider sign-in runs in a browser tab without the app's bearer token, so
# the app first obtains a one-time grant (after a re-authentication) and opens
# /android/link/<provider>?grant=…; the native OAuth callbacks then link the
# verified identity to that user exactly like the Web settings link flow.

_MOBILE_LINK_TTL = 600
_MOBILE_LINK_PROVIDERS = ('google', 'minashin')


def _mobile_link_grant_key(grant):
    return 'mobile:link:' + _mobile_digest(grant)


@app.route('/api/mobile/v1/account/link/<provider>/start', methods=['POST'])
def mobile_account_link_start(provider):
    if provider not in _MOBILE_LINK_PROVIDERS:
        return _mobile_error('invalid_provider', 404)
    grant = secrets.token_urlsafe(32)
    redis_conn.set(_mobile_link_grant_key(grant), json.dumps({'user_id': current_user.id, 'provider': provider}),
                   ex=_MOBILE_LINK_TTL)
    return jsonify({'status': 'ok', 'path': url_for('mobile_account_link_open', provider=provider, grant=grant),
                    'expires_in': _MOBILE_LINK_TTL})


@app.route('/android/link/<provider>', methods=['GET'])
def mobile_account_link_open(provider):
    grant = str(request.args.get('grant') or '')
    if provider not in _MOBILE_LINK_PROVIDERS or not re.fullmatch(r'[A-Za-z0-9_-]{43}', grant):
        return _mobile_native_redirect(error='link_invalid')
    key = _mobile_link_grant_key(grant)
    payload = _mobile_redis_json(key)
    redis_conn.delete(key)
    if not payload or payload.get('provider') != provider:
        return _mobile_native_redirect(error='link_expired')
    for stale in ('mobile_native_google', 'mobile_native_auth', 'mobile_native_device_name', 'mobile_native_code_challenge'):
        session.pop(stale, None)
    session['mobile_native_link_user'] = int(payload.get('user_id') or 0)
    session['mobile_native_link_provider'] = provider
    if provider == 'google':
        session['mobile_native_google'] = True
        return oauth.google.authorize_redirect(url_for('mobile_google_callback', _external=True, _scheme='https'))
    session['mobile_native_auth'] = True
    return login_minashin()


def _mobile_native_link_target(provider):
    """Pops the pending link for [provider]; returns the user to link or None."""
    user_id = session.pop('mobile_native_link_user', None)
    linked_provider = session.pop('mobile_native_link_provider', None)
    if not user_id or linked_provider != provider:
        return None
    return db.session.get(User, int(user_id))


def _mobile_native_link_redirect(provider, error=None):
    params = {'error': error} if error else {'linked': provider}
    return redirect(url_for('mobile_auth_callback', _external=True, _scheme='https', **params))


def _mobile_native_link_google(user, google_id, email):
    existing = User.query.filter_by(google_id=google_id).first()
    if existing and existing.id != user.id:
        return _mobile_native_link_redirect('google', 'google_already_linked')
    user.google_id = google_id
    if not user.google_email:
        user.google_email = email
    safe_db_commit()
    return _mobile_native_link_redirect('google')


def _mobile_native_link_minashin(user, sub, email):
    existing = User.query.filter_by(minashin_sub=sub).first()
    if existing and existing.id != user.id:
        return _mobile_native_link_redirect('minashin', 'minashin_already_linked')
    user.minashin_sub = sub
    if not user.minashin_email:
        user.minashin_email = email or None
    safe_db_commit()
    return _mobile_native_link_redirect('minashin')
