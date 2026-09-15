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
    return jsonify({'id': current_user.id, 'username': current_user.username,
                    'expires_at': (g.mobile_session.created_at + timedelta(seconds=MOBILE_TOKEN_TTL)).isoformat() + 'Z',
                    'e2ee_enabled': bool(current_user.enable_e2ee),
                    'encryption_mode': 'server_managed_at_rest',
                    'default_model': current_user.default_model,
                    'model_ids': sorted(ALL_VALID_MODEL_IDS)})


@app.route('/api/mobile/v1/revoke', methods=['POST'])
def mobile_revoke():
    g.mobile_session.is_revoked = True
    g.mobile_session.revoked_at = datetime.utcnow()
    safe_db_commit()
    return jsonify({'status': 'revoked'})
