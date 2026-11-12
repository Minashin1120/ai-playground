
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
    b64 = base64.b64encode(_qr_png_bytes(uri)).decode()
    
    return jsonify({'secret': secret, 'qr_image': f"data:image/png;base64,{b64}"})

@app.route('/api/2fa/totp/enable', methods=['POST'])
@login_required
def totp_enable():
    code = request.json.get('code')
    secret = session.get('temp_totp_secret')
    if not secret: return jsonify({'error': 'Setup session expired'}), 400
    
    if verify_totp_code(current_user.id, secret, code):
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
