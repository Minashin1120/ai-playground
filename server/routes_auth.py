# -----------------------------------------------------------
# Auth Routes
# -----------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    auth_success = request.args.get('auth_success') == '1'
    next_url = request.args.get('next') or url_for('index')
    if not (isinstance(next_url, str) and next_url.startswith('/') and not next_url.startswith('//')):
        next_url = url_for('index')

    if current_user.is_authenticated:
        if auth_success:
            g_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
            return render_template(
                'login.html',
                site_key=os.getenv('TURNSTILE_SITE_KEY'),
                google_client_id=g_client_id,
                auth_success=True,
                auth_success_redirect=next_url
            )
        return redirect(url_for('index'))
    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                  'application/json' in request.headers.get('Accept', '')
        
        if request.is_json:
            form_data = request.get_json(silent=True) or {}
        else:
            form_data = request.form

        login_ip = get_client_ip() or request.remote_addr or 'unknown'
        if not rate_limit(f"rl:login:ip:{login_ip}", 20, 300):
            if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")

        if not verify_turnstile(form_data.get('cf-turnstile-response')):
            if is_ajax: return jsonify({'error': "認証エラーが発生しました"}), 401
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="認証エラーが発生しました")
            
        username = (form_data.get('username') or '').strip()
        pw = form_data.get('password') or ""
        if len(username) > 80 or len(pw) > 512:
            if is_ajax:
                return jsonify({'error': "ユーザー名またはパスワードが正しくありません"}), 401
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="ユーザー名またはパスワードが正しくありません"), 401
        user = User.query.filter_by(username=username).first()
        # Allow login even if IP/Cookie is banned; ban screen will handle after login.
        if user:
            if not rate_limit(f"rl:login:user:{user.id}", 10, 300):
                if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
                return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
            
            now = datetime.utcnow()
            easy_ok = False
            try:
                if user.easy_login_hash and user.easy_login_expires_at and now <= user.easy_login_expires_at:
                    easy_ok = check_password_hash(user.easy_login_hash, pw)
            except Exception:
                easy_ok = False
                
            if easy_ok:
                # One-time easy login: disable after first successful use
                user.easy_login_hash = None
                user.easy_login_expires_at = None
                safe_db_commit()
                session['easy_login_used'] = True
                session.pop('_flashes', None)
                remember = bool(form_data.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
                
            if user.easy_login_hash and user.easy_login_expires_at and now > user.easy_login_expires_at:
                user.easy_login_hash = None
                user.easy_login_expires_at = None
                safe_db_commit()
                
            if user.check_password(pw):
                if user.is_2fa_enabled:
                    session['remember_me'] = bool(form_data.get('remember'))
                    session['pre_2fa_user_id'] = user.id
                    if is_ajax: return jsonify({
                        'status': '2fa_required',
                        'default_method': user.default_2fa_method or 'totp'
                    })
                    return redirect(url_for('verify_2fa'))
                
                session.pop('_flashes', None)
                remember = bool(form_data.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
                
        if is_ajax: return jsonify({'error': "ユーザー名またはパスワードが正しくありません"}), 401
        g_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
        if not g_client_id:
            log_force("DEBUG: GOOGLE_CLIENT_ID is missing in .env")
        return render_template('login.html', 
                               site_key=os.getenv('TURNSTILE_SITE_KEY'), 
                               google_client_id=g_client_id, 
                               error="ユーザー名またはパスワードが正しくありません")
    
    g_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
    return render_template('login.html', 
                           site_key=os.getenv('TURNSTILE_SITE_KEY'), 
                           google_client_id=g_client_id)

def _resolve_or_create_google_user(google_id, email):
    """Resolve or create the account for a verified Google identity.

    Accounts are matched strictly by Google identity (``google_id``) and by
    the verified Google email already recorded on the account
    (``google_email``).  Usernames are never used for linking, so an account
    whose username happens to equal another person's Google email cannot
    absorb that person's Google login (pre-account-takeover defense).  When
    the email-as-username is already taken by an unrelated registration, a
    unique username is generated so a legitimate Google signup is never
    silently routed into that account.
    """
    user = User.query.filter_by(google_id=google_id).first()
    if user:
        return user
    user = User.query.filter_by(google_email=email).first()
    if user:
        if user.google_id and user.google_id != google_id:
            raise ValueError("Invalid Google identity")
        user.google_id = google_id
        if not user.google_email:
            user.google_email = email
        safe_db_commit()
        return user
    username = email
    if User.query.filter_by(username=username).first():
        local = (email.split('@', 1)[0] or 'user')[:40]
        while True:
            username = f"{local}-{secrets.token_hex(4)}"
            if not User.query.filter_by(username=username).first():
                break
    user = User(
        username=username,
        google_id=google_id,
        google_email=email,
        is_setup_completed=False,
    )
    db.session.add(user)
    safe_db_commit()
    return user

@app.route('/login/google')
def login_google():
    if current_user.is_authenticated:
        # If already logged in, we are linking
        session['google_link_mode'] = True
    else:
        session.pop('google_link_mode', None)
    redirect_uri = url_for('login_google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def login_google_callback():
    link_mode = session.pop('google_link_mode', False)
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info or user_info.get('email_verified') is not True:
            flash("Google からユーザー情報を取得できませんでした。")
            return redirect(url_for('login' if not current_user.is_authenticated else 'index'))
        
        google_id = str(user_info.get('sub') or '').strip()
        email = str(user_info.get('email') or '').strip().lower()
        if not google_id or not email or len(email) > 128:
            raise ValueError("Invalid Google identity")
        
        if current_user.is_authenticated:
            # Explicit linking from settings
            existing_with_id = User.query.filter_by(google_id=google_id).first()
            if existing_with_id and existing_with_id.id != current_user.id:
                flash("この Google アカウントは既に他のユーザーに紐付けられています。")
                return redirect(url_for('index'))
            
            current_user.google_id = google_id
            if not current_user.google_email:
                current_user.google_email = email
            safe_db_commit()
            flash("Google アカウントと連携しました。")
            return redirect(url_for('index'))

        # Login/Signup flow
        user = _resolve_or_create_google_user(google_id, email)

        if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
            session['pre_2fa_user_id'] = user.id
            session['remember_me'] = True
            return redirect(url_for('verify_2fa'))

        session.pop('_flashes', None)
        login_user(user, remember=True)
        create_user_session(user)
        record_user_client_token(user)
        
        target_url = url_for('setup') if not user.is_setup_completed else url_for('index')
        return redirect(url_for('login', auth_success='1', next=target_url))
    except Exception as e:
        logger.error(f"Google Login Callback Error: {e}")
        flash("Google 連携中にエラーが発生しました。")
        return redirect(url_for('login' if not current_user.is_authenticated else 'index'))

@app.route('/login/google/one-tap', methods=['POST'])
def login_google_one_tap():
    token = request.form.get('credential')
    if not token:
        return jsonify({'error': 'No credential provided'}), 400
    
    try:
        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), os.getenv('GOOGLE_CLIENT_ID'))

        if idinfo.get('email_verified') is not True:
            return jsonify({'error': 'Google email is not verified'}), 400
        google_id = str(idinfo.get('sub') or '').strip()
        email = str(idinfo.get('email') or '').strip().lower()
        if not google_id or not email or len(email) > 128:
            return jsonify({'error': 'Invalid Google identity'}), 400
        
        user = _resolve_or_create_google_user(google_id, email)

        if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
            session['pre_2fa_user_id'] = user.id
            session['remember_me'] = True
            return jsonify({
                'status': '2fa_required', 
                'redirect': url_for('verify_2fa'),
                'default_method': user.default_2fa_method or 'totp'
            })

        session.pop('_flashes', None)
        login_user(user, remember=True)
        create_user_session(user)
        record_user_client_token(user)
        
        if not user.is_setup_completed:
            return jsonify({'status': 'ok', 'redirect': url_for('setup')})
        return jsonify({'status': 'ok', 'redirect': url_for('index')})

    except Exception as e:
        logger.error(f"Google One Tap Login Error: {e}")
        return jsonify({'error': 'Google One Tap 認証中にエラーが発生しました'}), 400

@app.route('/api/account/unlink_google', methods=['POST'])
@login_required
def unlink_google():
    if not current_user.google_id:
        return jsonify({'error': 'Not linked'}), 400
    
    # Optional: Prevent unlinking if no password or other login method exists
    # but here we allow it.
    current_user.google_id = None
    current_user.google_email = None
    safe_db_commit()
    return jsonify({'status': 'ok'})

# ============================================================
# Minashin 中央アカウント連携 (OAuth 2.0 + PKCE)
# ============================================================
def _minashin_code_verifier(length=128):
    """暗号論的に安全な PKCE code_verifier を生成する（43〜128文字）。"""
    if length < 43 or length > 128:
        raise ValueError(f"code_verifier の長さは43〜128である必要があります（指定: {length}）")
    return ''.join(secrets.choice(_PKCE_CHARSET) for _ in range(length))


def _minashin_code_challenge(code_verifier):
    """code_verifier から code_challenge を生成する（SHA-256 + Base64URL）。"""
    digest = hashlib.sha256(code_verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def _minashin_state(length=32):
    """CSRF 対策用の state パラメータを生成する。"""
    return secrets.token_hex(length)


def _minashin_client_identity():
    """リクエストから (client_id, redirect_uri) を導出する。

    Origin-Based 自動登録のため、client_id は連携サイト自身の Origin URL
    （= redirect_uri の Origin）でなければならない。
    """
    redirect_uri = url_for('minashin_callback', _external=True, _scheme='https')
    parsed = urlparse(redirect_uri)
    client_id = f"{parsed.scheme}://{parsed.netloc}"
    return client_id, redirect_uri


def _resolve_or_create_minashin_user(minashin_sub, email, user_data):
    """Minashin アカウントからローカルユーザーを解決または作成する。

    Google 連携と同じく、アカウントは Minashin の ``sub`` と、アカウントに
    記録済みの ``minashin_email`` でのみ照合する。ユーザー名でリンクすることは
    しない（別ユーザーのメールと同じユーザー名のアカウントがその人の
    Minashin ログインを吸収しないための、アカウント乗っ取り対策）。
    """
    user = User.query.filter_by(minashin_sub=minashin_sub).first()
    if user:
        return user
    if email:
        user = User.query.filter_by(minashin_email=email).first()
        if user:
            if user.minashin_sub and user.minashin_sub != minashin_sub:
                raise ValueError("Invalid Minashin identity")
            user.minashin_sub = minashin_sub
            if not user.minashin_email:
                user.minashin_email = email
            safe_db_commit()
            return user
    # 新しいアカウントを作成。preferred_username / nickname / メールローカル部から
    # ユーザー名を生成し、重複があれば数値を足してユニークにする。
    base = (
        user_data.get('preferred_username')
        or user_data.get('nickname')
        or (email.split('@', 1)[0] if email else '')
        or 'minashin'
    )
    base = re.sub(r'[\x00-\x1f\x7f@]', '', str(base)).strip()[:40] or 'minashin'
    username = base
    suffix = 2
    while User.query.filter_by(username=username).first():
        username = f"{base}{suffix}"
        suffix += 1
    user = User(
        username=username,
        minashin_sub=minashin_sub,
        minashin_email=email or None,
        is_setup_completed=False,
    )
    db.session.add(user)
    safe_db_commit()
    return user


@app.route('/login/minashin')
def login_minashin():
    """Minashin アカウントでログイン（または設定画面からの連携）を開始する。"""
    if current_user.is_authenticated:
        # 設定画面からログイン済みユーザーの連携を開始する場合
        session['minashin_link_mode'] = True
    else:
        session.pop('minashin_link_mode', None)

    code_verifier = _minashin_code_verifier(128)
    code_challenge = _minashin_code_challenge(code_verifier)
    state = _minashin_state(32)

    session['minashin_oauth_state'] = state
    session['minashin_code_verifier'] = code_verifier

    client_id, redirect_uri = _minashin_client_identity()

    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
        'state': state,
        'scope': 'openid profile email',
    }
    authorize_url = f"{MINASHIN_ACCOUNT_BASE_URL}/oauth/authorize?{urlencode(params)}"
    return redirect(authorize_url)


@app.route('/auth/minashin/callback')
def minashin_callback():
    """Minashin アカウントからの OAuth コールバックを処理する。"""
    link_mode = session.pop('minashin_link_mode', False)
    redirect_target = 'index' if current_user.is_authenticated else 'login'
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        error = request.args.get('error')
        error_description = request.args.get('error_description')

        if error:
            flash(f"Minashin アカウント認証に失敗しました: {error_description or error}")
            return redirect(url_for(redirect_target))

        if not code:
            flash("Minashin アカウント認証に失敗しました（認可コードがありません）。")
            return redirect(url_for(redirect_target))

        # 並行リクエストやリトライ時の二重交換防止（アトミックキー取得）
        code_lock_key = f"oauth:minashin:code:{hashlib.sha256(code.encode()).hexdigest()}"
        is_first_exchange = True
        try:
            if redis_client:
                is_first_exchange = bool(redis_client.set(code_lock_key, "1", nx=True, ex=120))
        except Exception:
            is_first_exchange = True

        if not is_first_exchange:
            # 既に別スレッド/リクエストで交換処理中または交換済み
            if session.get('pre_2fa_user_id'):
                return redirect(url_for('verify_2fa'))
            if current_user.is_authenticated:
                return redirect(url_for('index'))
            return redirect(url_for('login'))

        expected_state = session.pop('minashin_oauth_state', None)
        if not state or not expected_state or not secrets.compare_digest(state, expected_state):
            if session.get('pre_2fa_user_id'):
                return redirect(url_for('verify_2fa'))
            if current_user.is_authenticated:
                return redirect(url_for('index'))
            flash("Minashin アカウント認証に失敗しました（state 検証エラー）。")
            return redirect(url_for(redirect_target))

        code_verifier = session.pop('minashin_code_verifier', None)
        if not code_verifier:
            if session.get('pre_2fa_user_id'):
                return redirect(url_for('verify_2fa'))
            if current_user.is_authenticated:
                return redirect(url_for('index'))
            flash("セッション情報が失われました。もう一度お試しください。")
            return redirect(url_for(redirect_target))

        client_id, redirect_uri = _minashin_client_identity()

        # 認可コードをアクセストークンに交換する
        token_response = requests.post(
            f"{MINASHIN_ACCOUNT_BASE_URL}/oauth/token",
            json={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': redirect_uri,
                'client_id': client_id,
                'code_verifier': code_verifier,
            },
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            timeout=MINASHIN_REQUEST_TIMEOUT,
        )
        if not token_response.ok:
            if session.get('pre_2fa_user_id'):
                return redirect(url_for('verify_2fa'))
            if current_user.is_authenticated:
                return redirect(url_for('index'))
            try:
                error_data = token_response.json()
            except Exception:
                error_data = {}
            error_code = error_data.get('error', 'unknown')
            logger.error(f"Minashin token exchange failed: {token_response.status_code} - {error_data}")
            if error_code == 'invalid_grant':
                flash("認証コードの有効期限が切れているか、すでに使用されています。もう一度お試しください。")
            else:
                flash("Minashin アカウントとの連携に失敗しました。")
            return redirect(url_for(redirect_target))

        token_data = token_response.json()
        access_token = token_data.get('access_token')
        if not access_token:
            flash("Minashin アカウントとの連携に失敗しました（トークンが取得できませんでした）。")
            return redirect(url_for(redirect_target))

        # ユーザー情報を取得する
        userinfo_response = requests.get(
            f"{MINASHIN_ACCOUNT_BASE_URL}/api/userinfo",
            headers={'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'},
            timeout=MINASHIN_REQUEST_TIMEOUT,
        )
        if not userinfo_response.ok:
            logger.error(f"Minashin userinfo failed: {userinfo_response.status_code}")
            flash("Minashin アカウント情報の取得に失敗しました。")
            return redirect(url_for(redirect_target))

        user_data = userinfo_response.json()
        sub = str(user_data.get('sub') or '').strip()
        email = str(user_data.get('email') or '').strip().lower()
        if not sub:
            raise ValueError("No 'sub' in Minashin userinfo response")
        if len(sub) > 128:
            raise ValueError("Invalid Minashin sub")
        # メール未確認（email_verified=False）の場合はメールを使わない。
        # 未確認メールでの既存アカウント吸収（乗っ取り）を防ぐため、連携は
        # sub ベースでのみ行い、email による照合・保存は行わない。
        if str(user_data.get('email_verified')) == 'False' or user_data.get('email_verified') is False:
            email = ''

        if current_user.is_authenticated:
            # 設定画面からの連携
            existing_with_sub = User.query.filter_by(minashin_sub=sub).first()
            if existing_with_sub and existing_with_sub.id != current_user.id:
                flash("この Minashin アカウントは既に他のユーザーに紐付けられています。")
                return redirect(url_for('index'))
            current_user.minashin_sub = sub
            if not current_user.minashin_email:
                current_user.minashin_email = email or None
            safe_db_commit()
            flash("Minashin アカウントと連携しました。")
            return redirect(url_for('index'))

        # ログイン / アカウント作成フロー
        user = _resolve_or_create_minashin_user(sub, email, user_data)

        if user.is_2fa_enabled:
            session['pre_2fa_user_id'] = user.id
            session['remember_me'] = True
            return redirect(url_for('verify_2fa'))

        session.pop('_flashes', None)
        login_user(user, remember=True)
        create_user_session(user)
        record_user_client_token(user)

        target_url = url_for('setup') if not user.is_setup_completed else url_for('index')
        return redirect(url_for('login', auth_success='1', next=target_url))

    except requests.RequestException as e:
        logger.error(f"Minashin login connection error: {e}")
        flash("Minashin アカウントシステムに接続できませんでした。時間をおいて再度お試しください。")
        return redirect(url_for(redirect_target))
    except Exception as e:
        logger.error(f"Minashin Login Callback Error: {e}")
        flash("Minashin 連携中にエラーが発生しました。")
        return redirect(url_for(redirect_target))


@app.route('/api/account/unlink_minashin', methods=['POST'])
@login_required
def unlink_minashin():
    if not current_user.minashin_sub:
        return jsonify({'error': 'Not linked'}), 400

    # Google 連携と同じく、解除を許可する（パスワード等が未設定の場合は
    # 別のログイン手段が残っていない可能性に注意する旨を UI 側で案内）。
    current_user.minashin_sub = None
    current_user.minashin_email = None
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/login/passkey/options', methods=['POST'])
def login_passkey_options():
    if current_user.is_authenticated:
        return jsonify({'error': 'already_authenticated'}), 400
    login_ip = get_client_ip() or request.remote_addr or 'unknown'
    if not rate_limit(f"rl:login:ip:{login_ip}", 20, 300):
        return jsonify({'error': 'Too many attempts. Try again later.'}), 429
    data = request.json or {}
    if not verify_turnstile(data.get('turnstile')):
        return jsonify({'error': 'Auth Error'}), 401
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': 'Username required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not getattr(user, "passkey_only_login", False):
        return jsonify({'error': 'Invalid credentials'}), 400
    # Allow passkey login even if IP/Cookie is banned; ban screen will handle after login.
    if not rate_limit(f"rl:login:user:{user.id}", 10, 300):
        return jsonify({'error': 'Too many attempts. Try again later.'}), 429
    creds = _load_user_webauthn_credentials(user)
    if not creds:
        return jsonify({'error': 'No credentials'}), 400
    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.REQUIRED
    )
    session['passkey_login_user_id'] = user.id
    session['webauthn_login_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    session['passkey_login_remember'] = bool(data.get('remember'))
    return options_to_json(options)

@app.route('/login/passkey/verify', methods=['POST'])
def login_passkey_verify():
    if current_user.is_authenticated:
        return jsonify({'error': 'already_authenticated'}), 400
    user_id = session.get('passkey_login_user_id')
    if not user_id:
        return jsonify({'error': 'Session expired'}), 401
    if not rate_limit(f"rl:webauthn:user:{user_id}", 8, 300):
        return jsonify({'error': 'Too many attempts'}), 429
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'Invalid user'}), 400
    try:
        data = request.json or {}
        challenge = session.get('webauthn_login_challenge')
        if not challenge:
            return jsonify({'error': 'Challenge missing'}), 400
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(data.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
        if not current_cred:
            return jsonify({'error': 'Credential not found'}), 400
        verification = verify_authentication_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=True
        )
        current_cred['sign_count'] = verification.new_sign_count
        _save_user_webauthn_credentials(user, creds)
        db.session.commit()
        session.pop('passkey_login_user_id', None)
        session.pop('webauthn_login_challenge', None)
        session.pop('_flashes', None)
        remember = bool(session.pop('passkey_login_remember', False))
        login_user(user, remember=remember)
        create_user_session(user)
        record_user_client_token(user)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Passkey Login Verify Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    if current_user.is_authenticated: return redirect(url_for('index'))
    user_id = session.get('pre_2fa_user_id')
    if not user_id: return redirect(url_for('login'))
    
    user = User.query.get(user_id)
    if not user: return redirect(url_for('login'))

    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                  'application/json' in request.headers.get('Accept', '')
        
        if not rate_limit(f"rl:2fa:user:{user.id}", 8, 300):
            if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
            return render_template('verify_2fa.html', error="Too many attempts. Try again later.")
            
        code = None
        if is_ajax:
            data = request.json or {}
            code = data.get('totp_code')
        
        if not code:
            code = request.form.get('totp_code')
            
        if code:
            secret = decrypt_val(user.totp_secret)
            if secret and pyotp.TOTP(secret).verify(code):
                session.pop('pre_2fa_user_id', None)
                session.pop('_flashes', None)
                remember = bool(session.pop('remember_me', False))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
            
            if is_ajax: return jsonify({'error': "Invalid Code"}), 400
            return render_template('verify_2fa.html', error="Invalid Code")
        
        if is_ajax: return jsonify({'error': "Code required"}), 400
            
    has_totp = bool(user.totp_secret)
    has_webauthn = bool(_load_user_webauthn_credentials(user))
    default_method = user.default_2fa_method or 'totp'
    
    # If the default method is not available, switch to the one that is
    if default_method == 'totp' and not has_totp and has_webauthn:
        default_method = 'webauthn'
    elif default_method == 'webauthn' and not has_webauthn and has_totp:
        default_method = 'totp'

    return render_template('verify_2fa.html', 
                           has_totp=has_totp, 
                           has_webauthn=has_webauthn, 
                           default_method=default_method)

@app.route('/verify-2fa/webauthn/options', methods=['POST'])
def verify_2fa_webauthn_options():
    user_id = session.get('pre_2fa_user_id')
    logger.info(f"WebAuthn Options Req: user_id={user_id}, session={session.keys()}")
    if not user_id: return jsonify({'error': 'Session expired'}), 401
    user = User.query.get(user_id)
    
    creds = _load_user_webauthn_credentials(user)
    
    logger.info(f"User Creds Count: {len(creds)}")
    if not creds: return jsonify({'error': 'No credentials'}), 400

    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.PREFERRED
    )
    
    session['webauthn_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    return options_to_json(options)

@app.route('/verify-2fa/webauthn/verify', methods=['POST'])
def verify_2fa_webauthn_verify():
    user_id = session.get('pre_2fa_user_id')
    if not user_id: return jsonify({'error': 'Session expired'}), 401
    user = User.query.get(user_id)
    if not rate_limit(f"rl:webauthn:user:{user_id}", 8, 300):
        return jsonify({'error': 'Too many attempts'}), 429
    
    try:
        data = request.json or {}
        challenge = session.get('webauthn_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(data.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
        if not current_cred: return jsonify({'error': 'Credential not found'}), 400

        verification = verify_authentication_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=False # Depends on device
        )
        
        current_cred['sign_count'] = verification.new_sign_count
        _save_user_webauthn_credentials(user, creds)
        db.session.commit()
        
        session.pop('pre_2fa_user_id', None)
        session.pop('_flashes', None)
        remember = bool(session.pop('remember_me', False))
        login_user(user, remember=remember)
        create_user_session(user)
        record_user_client_token(user)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"WebAuthn Verify Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        signup_ip = get_client_ip() or request.remote_addr or 'unknown'
        if not rate_limit(f"rl:signup:ip:{signup_ip}", 10, 3600):
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
        if not verify_turnstile(request.form.get('cf-turnstile-response')): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
        if is_request_banned_identifier():
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Signup blocked.")
        username = str(request.form.get('username') or '').replace('\x00', '').strip()
        password = str(request.form.get('password') or '')
        if len(username) < 3 or len(username) > 80 or re.search(r'[\x00-\x1f\x7f]', username):
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username must be 3-80 characters.")
        if '@' in username:
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username cannot contain @.")
        if len(password) < 8 or len(password) > 256:
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Password must be 8-256 characters.")
        if _is_primary_admin_username(username): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username taken")
        if User.query.filter_by(username=username).first(): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username taken")
        new_user = User(username=username, is_setup_completed=False)
        new_user.set_password(password)
        db.session.add(new_user)
        safe_db_commit()
        session.pop('_flashes', None)
        login_user(new_user)
        # Establish the server-side session before redirecting.  Deferring this
        # to ensure_active_session on the first /setup request allowed parallel
        # browser requests (manifest/service-worker, etc.) to create competing
        # sessions; one could revoke the other and send the user back to /login.
        create_user_session(new_user)
        record_user_client_token(new_user)
        return redirect(url_for('setup'))
    return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'))

def _user_has_unencrypted_data(user):
    """True when the account still stores plaintext messages or files.

    During setup a freshly imported account may already contain chats/files
    before the user picks E2EE on the final keys step.  When E2EE is enabled at
    that point the imported data must be re-encrypted to match, so the setup
    handler can detect whether a migration job is actually needed.
    """
    try:
        has_unencrypted_message = db.session.query(Message.id).join(
            Thread, Message.thread_id == Thread.id
        ).filter(
            Thread.user_id == user.id,
            or_(Message.is_encrypted.is_(None), Message.is_encrypted.is_(False)),
        ).first() is not None
    except Exception:
        has_unencrypted_message = False
    has_unencrypted_file = False
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user.id))
    if os.path.isdir(user_dir):
        try:
            for root, _, names in os.walk(user_dir):
                if any(not name.endswith('.enc') for name in names):
                    has_unencrypted_file = True
                    break
        except Exception:
            pass
    return has_unencrypted_message or has_unencrypted_file

@app.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if current_user.is_setup_completed: return redirect(url_for('index'))
    if request.method == 'POST':
        try:
            vertex_credentials_json = _normalize_gemini_vertex_credentials_json(request.form.get('gemini_vertex_credentials_json'))
        except ValueError as e:
            return render_template('setup.html', error=str(e))
        # Preserve secrets imported during the setup step when a key field is
        # left empty here, instead of overwriting them with an empty value.
        for field, raw_value in (
            ('openai_api_key', request.form.get('openai_key')),
            ('gemini_api_key', request.form.get('gemini_key')),
            ('deepseek_api_key', request.form.get('deepseek_key')),
            ('kimi_api_key', request.form.get('kimi_key')),
            ('mistral_api_key', request.form.get('mistral_key')),
            ('xai_api_key', request.form.get('xai_key')),
            ('google_api_key', request.form.get('google_key')),
            ('google_cloud_project', request.form.get('google_project')),
            ('gemini_vertex_project', request.form.get('gemini_vertex_project')),
            ('gemini_vertex_credentials_json', vertex_credentials_json),
        ):
            value = str(raw_value or '').strip()
            if value:
                setattr(current_user, field, encrypt_val(value))
        current_user.gemini_backend = _normalize_gemini_backend(request.form.get('gemini_backend'))
        current_user.gemini_vertex_location = _normalize_gemini_vertex_location(request.form.get('gemini_vertex_location'))
        current_user.default_model = request.form.get('default_model') or "gemini-3.6-flash"
        current_user.enable_e2ee = (request.form.get('enable_e2ee') == 'on')
        current_user.is_setup_completed = True
        safe_db_commit()
        # Data imported during the setup step is stored with the account's
        # pre-setup E2EE state (plaintext).  When the user enables E2EE here,
        # re-encrypt the imported content in the background so it matches, in
        # the same way toggling E2EE on the settings screen triggers a migration.
        if current_user.enable_e2ee and _user_has_unencrypted_data(current_user):
            task_queue.enqueue(migrate_e2ee_task, current_user.id, True)
        return redirect(url_for('index'))
    return render_template(
        'setup.html',
        default_model=current_user.default_model or 'gemini-3.6-flash',
        gemini_backend=_normalize_gemini_backend(current_user.gemini_backend),
    )

@app.route('/logout', methods=['POST'])
def logout():
    if current_user.is_authenticated:
        sid = session.get('session_id')
        if sid:
            user_sess = UserSession.query.filter_by(user_id=current_user.id, session_id=sid, is_revoked=False).first()
            if user_sess:
                user_sess.is_revoked = True
                user_sess.revoked_at = datetime.utcnow()
                try:
                    safe_db_commit()
                except Exception:
                    pass
    logout_user()
    session.pop('session_id', None)
    session.pop('pre_2fa_user_id', None)
    return redirect(url_for('index'))

