"""Native Android passkey (WebAuthn) sign-in, 2FA, and security preferences.

These routes run in the shared server namespace built by app.py next to
routes_mobile_auth.py, so helpers and constants are shared without imports.
"""

# region: Passkey storage and origin helpers

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


# region: Passkey sign-in and 2FA
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

# region: Security settings and preferences
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
    if not verify_totp_code(current_user.id, secret, code):
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
    if (not rate_limit(f'rl:mobile:totp_disable:user:{current_user.id}', 8, 300)
            or auth_failures_exceeded('totp', current_user.id)):
        return _mobile_error('rate_limited', 429, 300)
    body = request.get_json(silent=True) or {}
    try:
        secret = decrypt_val(current_user.totp_secret)
    except Exception:
        secret = None
    if not verify_totp_code(current_user.id, secret, body.get('code')):
        record_auth_failure('totp', current_user.id)
        return _mobile_error('invalid_code', 401)
    clear_auth_failures('totp', current_user.id)
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

