CLIENT_TOKEN_COOKIE = "ai_client_token"

def _is_secure_request():
    if request.is_secure:
        return True
    proto = request.headers.get('X-Forwarded-Proto', '')
    return proto.lower() == 'https'

def get_client_token():
    token = request.cookies.get(CLIENT_TOKEN_COOKIE)
    if not token:
        token = secrets.token_urlsafe(24)
        g.new_client_token = token
    g.client_token = token
    return token

def is_request_banned_identifier():
    ip = get_client_ip()
    token = request.cookies.get(CLIENT_TOKEN_COOKIE)
    clauses = []
    if ip:
        clauses.append((BannedIdentifier.kind == 'ip') & (BannedIdentifier.value == ip))
    if token:
        clauses.append((BannedIdentifier.kind == 'cookie') & (BannedIdentifier.value == token))
    if not clauses:
        return False
    try:
        return BannedIdentifier.query.filter(or_(*clauses)).first() is not None
    except Exception:
        return False

def _is_admin_exempt(user):
    return bool(getattr(user, "is_admin", False)) or _is_primary_admin_user(user)

def record_user_client_token(user):
    if not user:
        return
    token = get_client_token()
    if not token:
        return
    now = datetime.utcnow()
    ip = get_client_ip()
    row = UserClientToken.query.filter_by(user_id=user.id, token=token).first()
    if row:
        if not row.last_seen_at or (now - row.last_seen_at) > timedelta(minutes=5):
            row.last_seen_at = now
            if ip:
                row.ip_address = ip
            safe_db_commit()
        return
    row = UserClientToken(
        user_id=user.id,
        token=token,
        ip_address=ip,
        created_at=now,
        last_seen_at=now
    )
    db.session.add(row)
    safe_db_commit()

def _ensure_banned_identifier(kind, value, reason, source_user):
    if not value:
        return
    existing = BannedIdentifier.query.filter_by(kind=kind, value=value).first()
    if existing:
        return
    entry = BannedIdentifier(
        kind=kind,
        value=value,
        reason=reason,
        source_user_id=getattr(source_user, "id", None),
        source_username=getattr(source_user, "username", None)
    )
    db.session.add(entry)

def ban_related_accounts(user, reason):
    if not user or _is_admin_exempt(user):
        return
    ban_reason = reason or "Linked ban"
    ips = set()
    tokens = set()
    try:
        if current_user.is_authenticated and current_user.id == user.id:
            ip_now = get_client_ip()
            if ip_now:
                ips.add(ip_now)
            token_now = get_client_token()
            if token_now:
                tokens.add(token_now)
    except Exception:
        pass
    for s in UserSession.query.filter_by(user_id=user.id).all():
        if s.ip_address:
            ips.add(s.ip_address)
    for t in UserClientToken.query.filter_by(user_id=user.id).all():
        if t.token:
            tokens.add(t.token)

    for ip in ips:
        _ensure_banned_identifier("ip", ip, ban_reason, user)
    for token in tokens:
        _ensure_banned_identifier("cookie", token, ban_reason, user)

    user_ids = set()
    if ips:
        for s in UserSession.query.filter(UserSession.ip_address.in_(list(ips))).all():
            if s.user_id:
                user_ids.add(s.user_id)
    if tokens:
        for t in UserClientToken.query.filter(UserClientToken.token.in_(list(tokens))).all():
            if t.user_id:
                user_ids.add(t.user_id)
    now = datetime.utcnow()
    for uid in user_ids:
        u = User.query.get(uid)
        if not u or _is_admin_exempt(u):
            continue
        if not u.is_bot_banned:
            u.is_bot_banned = True
            u.bot_banned_at = now
        if not u.bot_ban_reason:
            u.bot_ban_reason = ban_reason
    safe_db_commit()

def _get_user_identifiers(user):
    ips = set()
    tokens = set()
    if not user:
        return ips, tokens
    for s in UserSession.query.filter_by(user_id=user.id).all():
        if s.ip_address:
            ips.add(s.ip_address)
    for t in UserClientToken.query.filter_by(user_id=user.id).all():
        if t.token:
            tokens.add(t.token)
    if current_user.is_authenticated and current_user.id == user.id:
        try:
            ip_now = get_client_ip()
            if ip_now:
                ips.add(ip_now)
        except Exception:
            pass
        try:
            token_now = get_client_token()
            if token_now:
                tokens.add(token_now)
        except Exception:
            pass
    return ips, tokens

def _unban_user(user):
    if not user:
        return
    user.is_bot_banned = False
    user.bot_ban_reason = None
    user.bot_banned_at = None
    user.bot_unbanned_at = datetime.utcnow()
    user.bot_unban_notice = True

def _unblock_identifiers(ips, tokens):
    if not ips and not tokens:
        return
    clauses = []
    if ips:
        clauses.append((BannedIdentifier.kind == 'ip') & (BannedIdentifier.value.in_(list(ips))))
    if tokens:
        clauses.append((BannedIdentifier.kind == 'cookie') & (BannedIdentifier.value.in_(list(tokens))))
    if not clauses:
        return
    BannedIdentifier.query.filter(or_(*clauses)).delete(synchronize_session=False)

def unban_single_account(user):
    if not user or _is_admin_exempt(user):
        return
    ips, tokens = _get_user_identifiers(user)
    _unban_user(user)
    _unblock_identifiers(ips, tokens)
    safe_db_commit()

def unban_linked_accounts(user):
    if not user or _is_admin_exempt(user):
        return
    ips, tokens = _get_user_identifiers(user)
    user_ids = set()
    if ips:
        for s in UserSession.query.filter(UserSession.ip_address.in_(list(ips))).all():
            if s.user_id:
                user_ids.add(s.user_id)
    if tokens:
        for t in UserClientToken.query.filter(UserClientToken.token.in_(list(tokens))).all():
            if t.user_id:
                user_ids.add(t.user_id)
    for uid in user_ids:
        u = User.query.get(uid)
        if not u or _is_admin_exempt(u):
            continue
        _unban_user(u)
    _unblock_identifiers(ips, tokens)
    safe_db_commit()

def _secure_delete_tree(path):
    if not path or not os.path.exists(path):
        return
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                secure_delete(os.path.join(root, name))
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass
    except Exception:
        pass

def _delete_user_account_immediately(user):
    if not user:
        raise ValueError("user_required")
    user_id = int(user.id)

    try:
        ips, tokens = _get_user_identifiers(user)
    except Exception:
        ips, tokens = set(), set()

    Feedback.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    BanAppeal.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    UserClientToken.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    UserSession.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    FileCache.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    BannedIdentifier.query.filter_by(source_user_id=user_id).delete(synchronize_session=False)
    ChatLatencyTrace.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    FirstTokenLatencyMetric.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    _unblock_identifiers(ips, tokens)

    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    _secure_delete_tree(user_dir)
    _secure_delete_tree(_chunk_user_dir(user_id))
    _delete_all_account_export_artifacts(user_id)

    try:
        redis_conn.delete(f"migration_status:{user_id}")
        redis_conn.delete(f"migration_progress:{user_id}")
        redis_conn.delete(f"bot:score:{user_id}")
    except Exception:
        pass

    try:
        from mcp_service.registry import delete_user_mcp_data
        delete_user_mcp_data(user_id)
    except Exception:
        pass

    db.session.delete(user)
    safe_db_commit()

