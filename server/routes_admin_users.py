# Admin user tools: bot user search, bot evidence listings, account clearing and the admin user
# update, followed by the theme colour and WebAuthn helpers those admin actions use. Split from
# routes_admin.py (see server/README.md); every server part shares the app module globals.

def _is_browser_admin():
    """Administrative rights are not extended to Android tokens (android/ANDROID_ONLY.md)."""
    return bool(getattr(current_user, "is_admin", False)) and getattr(g, 'mobile_session', None) is None

@app.route('/api/bot/users', methods=['GET'])
@login_required
def bot_users():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    q = (request.args.get('q') or '').strip()
    limit = request.args.get('limit') or '100'
    try:
        limit = max(1, min(500, int(limit)))
    except Exception:
        limit = 100
    query = User.query
    if q:
        query = query.filter(User.username.like(f"%{q}%"))
    users = query.order_by(User.id.desc()).limit(limit).all()
    res = []
    for u in users:
        try:
            lock_remaining = max(0, redis_conn.ttl(f"bot:lock:{u.id}") or 0)
        except Exception:
            lock_remaining = 0
        res.append({
            'user_id': u.id,
            'username': u.username,
            'is_admin': bool(_is_admin_exempt(u)),
            'bot_detection_enabled': u.bot_detection_enabled if u.bot_detection_enabled is not None else True,
            'is_bot_banned': bool(u.is_bot_banned),
            'bot_ban_reason': u.bot_ban_reason,
            'bot_banned_at': u.bot_banned_at.isoformat() + "Z" if u.bot_banned_at else None,
            'lock_remaining_seconds': lock_remaining,
        })
    return jsonify({'users': res})

@app.route('/api/bot/evidence/accounts', methods=['GET'])
@login_required
def bot_evidence_accounts():
    """Accounts with bot-detection records, newest activity first (deleted accounts included).

    Banned accounts are listed even without records so a missing BAN record is visible.
    """
    if not _is_browser_admin():
        return jsonify({'error': '403'}), 403
    q = (request.args.get('q') or '').strip()
    try:
        limit = max(1, min(500, int(request.args.get('limit') or 200)))
    except Exception:
        limit = 200
    rows_query = db.session.query(
        BotEvidenceLog.user_id,
        func.max(BotEvidenceLog.username),
        func.count(BotEvidenceLog.id),
        func.max(BotEvidenceLog.created_at),
    )
    if q:
        rows_query = rows_query.filter(BotEvidenceLog.username.like(f"%{q}%"))
    rows = rows_query.group_by(BotEvidenceLog.user_id)\
        .order_by(func.max(BotEvidenceLog.created_at).desc()).limit(limit).all()
    entries = {uid: {'user_id': uid, 'username': username or '', 'evidence_count': int(count or 0), 'last': last}
               for uid, username, count, last in rows}
    banned_query = User.query.filter(User.is_bot_banned.is_(True))
    if q:
        banned_query = banned_query.filter(User.username.like(f"%{q}%"))
    for u in banned_query.limit(limit).all():
        entries.setdefault(u.id, {'user_id': u.id, 'username': u.username, 'evidence_count': 0, 'last': None})
    users = {u.id: u for u in User.query.filter(User.id.in_(list(entries))).all()} if entries else {}
    accounts = []
    for entry in entries.values():
        user = users.get(entry['user_id'])
        try:
            lock_remaining = max(0, redis_conn.ttl(f"bot:lock:{entry['user_id']}") or 0)
        except Exception:
            lock_remaining = 0
        accounts.append({
            'user_id': entry['user_id'],
            'username': user.username if user else entry['username'],
            'exists': user is not None,
            'is_bot_banned': bool(user and user.is_bot_banned),
            'lock_remaining_seconds': lock_remaining,
            'evidence_count': entry['evidence_count'],
            'last_event_at': entry['last'].isoformat() + "Z" if entry['last'] else None,
        })
    accounts.sort(key=lambda a: a['last_event_at'] or '', reverse=True)
    response = jsonify({'accounts': accounts[:limit]})
    response.headers["Cache-Control"] = "no-store"
    return response

def _bot_evidence_item(row):
    # Where the event came from: the user's browser/app, an administrator, or the
    # server itself (a ban that spread from a linked account).
    if row.event_type == 'related_ban':
        client = 'server'
    elif not row.user_agent and row.event_type in ('admin_action', 'ban', 'unban', 'account_deleted'):
        client = 'admin'
    else:
        client = _bot_client_from_user_agent(row.user_agent)
    return {
        'id': row.id,
        'event_type': row.event_type,
        'created_at': row.created_at.isoformat() + "Z" if row.created_at else None,
        'score': row.score,
        'behavior_score': row.behavior_score,
        'reasons': row.reasons or '',
        'details': row.details or '',
        'ip_address': row.ip_address or '',
        'user_agent': row.user_agent or '',
        'client': client,
    }

def _bot_request_user_id(data):
    try:
        user_id = int(data.get('user_id') or 0)
    except Exception:
        return 0
    return user_id if user_id > 0 else 0

def _bot_account_ref(user_id):
    """The account (or a stand-in for a deleted one) that admin events are recorded against."""
    from types import SimpleNamespace
    user = db.session.get(User, user_id)
    if user is not None:
        return user
    latest = BotEvidenceLog.query.filter(BotEvidenceLog.user_id == user_id)\
        .order_by(BotEvidenceLog.id.desc()).first()
    return SimpleNamespace(id=user_id, username=latest.username if latest else None)

@app.route('/api/bot/evidence', methods=['GET'])
@login_required
def bot_evidence():
    """Bot-detection history of one account (also after the account was deleted)."""
    if not _is_browser_admin():
        return jsonify({'error': '403'}), 403
    user_id = _bot_request_user_id(request.args)
    if not user_id:
        return jsonify({'error': 'user_id_required'}), 400
    try:
        limit = max(1, min(200, int(request.args.get('limit') or 50)))
    except Exception:
        limit = 50
    try:
        before_id = int(request.args.get('before_id') or 0)
    except Exception:
        before_id = 0
    event_type = (request.args.get('type') or '').strip()[:32]
    base = BotEvidenceLog.query.filter(BotEvidenceLog.user_id == user_id)
    counts = {
        etype: int(count or 0)
        for etype, count in db.session.query(BotEvidenceLog.event_type, func.count(BotEvidenceLog.id))
        .filter(BotEvidenceLog.user_id == user_id).group_by(BotEvidenceLog.event_type).all()
    }
    query = base
    if event_type:
        query = query.filter(BotEvidenceLog.event_type == event_type)
    if before_id > 0:
        query = query.filter(BotEvidenceLog.id < before_id)
    rows = query.order_by(BotEvidenceLog.id.desc()).limit(limit + 1).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    user = db.session.get(User, user_id)
    if user is not None:
        username = user.username
    else:
        latest = base.order_by(BotEvidenceLog.id.desc()).first()
        username = latest.username if latest else ''
    payload = {
        'user': {'user_id': user_id, 'username': username or '', 'exists': user is not None},
        'counts': counts,
        'total': sum(counts.values()),
        'items': [_bot_evidence_item(r) for r in rows],
        'has_more': has_more,
    }
    if before_id <= 0:
        payload['state'] = _bot_admin_state(user_id, user)
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route('/api/bot/evidence/delete', methods=['POST'])
@login_required
def bot_evidence_delete():
    """Delete bot-detection records: selected ``ids`` or ``all`` of one ``user_id``,
    or every record of several accounts at once (``user_ids``)."""
    if not _is_browser_admin():
        return jsonify({'error': '403'}), 403
    data = request.get_json(silent=True) or {}
    user_ids = data.get('user_ids')
    if user_ids is not None:
        if not isinstance(user_ids, list) or not user_ids:
            return jsonify({'error': 'user_ids_required'}), 400
        try:
            user_ids = [int(i) for i in user_ids][:500]
        except Exception:
            return jsonify({'error': 'bad_user_ids'}), 400
        query = BotEvidenceLog.query.filter(BotEvidenceLog.user_id.in_(user_ids))
    else:
        user_id = _bot_request_user_id(data)
        if not user_id:
            return jsonify({'error': 'user_id_required'}), 400
        query = BotEvidenceLog.query.filter(BotEvidenceLog.user_id == user_id)
        if not data.get('all'):
            ids = data.get('ids')
            if not isinstance(ids, list) or not ids:
                return jsonify({'error': 'ids_required'}), 400
            try:
                ids = [int(i) for i in ids][:1000]
            except Exception:
                return jsonify({'error': 'bad_ids'}), 400
            query = query.filter(BotEvidenceLog.id.in_(ids))
    deleted = query.delete(synchronize_session=False)
    safe_db_commit()
    return jsonify({'status': 'ok', 'deleted': int(deleted or 0)})

@app.route('/api/bot/account/clear-lock', methods=['POST'])
@login_required
def bot_account_clear_lock():
    """Lift the temporary lock of an account, including after it was deleted."""
    if not _is_browser_admin():
        return jsonify({'error': '403'}), 403
    user_id = _bot_request_user_id(request.get_json(silent=True) or {})
    if not user_id:
        return jsonify({'error': 'user_id_required'}), 400
    account = _bot_account_ref(user_id)
    _clear_bot_lock(user_id, account if isinstance(account, User) else None)
    _log_bot_evidence('admin_action', details={'action': 'unlock', 'admin': current_user.username},
                      user=account, from_request=False)
    return jsonify({'status': 'ok'})

@app.route('/api/bot/account/clear-identifiers', methods=['POST'])
@login_required
def bot_account_clear_identifiers():
    """Lift the IP/device bans an account is the source of (they block new sign-ups)."""
    if not _is_browser_admin():
        return jsonify({'error': '403'}), 403
    user_id = _bot_request_user_id(request.get_json(silent=True) or {})
    if not user_id:
        return jsonify({'error': 'user_id_required'}), 400
    account = _bot_account_ref(user_id)
    removed = BannedIdentifier.query.filter_by(source_user_id=user_id).delete(synchronize_session=False)
    safe_db_commit()
    _log_bot_evidence('admin_action', details={
        'action': 'unblock_identifiers', 'admin': current_user.username, 'count': int(removed or 0),
    }, user=account, from_request=False)
    return jsonify({'status': 'ok', 'removed': int(removed or 0)})

@app.route('/api/bot/update', methods=['POST'])
@login_required
def bot_update():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    username = (data.get('username') or '').strip()
    action = (data.get('action') or '').strip()
    if not username or not action:
        return jsonify({'error': 'bad_request'}), 400
    if _is_primary_admin_username(username):
        return jsonify({'error': 'protected'}), 400
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'not_found'}), 404
    
    details = {'action': action, 'admin': current_user.username}
    if action == 'toggle_detection':
        enabled = bool(data.get('enabled'))
        user.bot_detection_enabled = enabled
        details['enabled'] = enabled
    elif action == 'ban':
        user.is_bot_banned = True
        user.bot_banned_at = datetime.utcnow()
        user.bot_ban_reason = data.get('reason') or "Manual ban"
        user.bot_unban_notice = False
        safe_db_commit()
        _log_bot_evidence('ban', reasons=user.bot_ban_reason, details={
            'by': 'admin', 'admin': current_user.username,
        }, user=user, from_request=False)
        ban_related_accounts(user, user.bot_ban_reason)
    elif action == 'unban':
        unban_single_account(user)  # records an 'unban' event
    elif action == 'unban_linked':
        unban_linked_accounts(user)  # records an 'unban' event per account
    elif action == 'unlock':
        _clear_bot_lock_for_user(user)
    elif action == 'delete_account':
        # Administrator accounts cannot be deleted from the account management screen.
        if _is_admin_exempt(user):
            return jsonify({'error': 'admin_account'}), 403
        _delete_user_account_immediately(user)
        return jsonify({'status': 'ok', 'username': username, 'action': action})
    else:
        return jsonify({'error': 'bad_action'}), 400
    
    safe_db_commit()
    if action in ('toggle_detection', 'unlock'):
        _log_bot_evidence('admin_action', details=details, user=user, from_request=False)
    return jsonify({'status': 'ok', 'username': username, 'action': action})

def normalize_theme_color(value):
    if not value:
        return ""
    v = str(value).strip()
    if not v:
        return ""
    if not v.startswith('#'):
        v = f"#{v}"
    if len(v) == 4:
        v = f"#{v[1]}{v[1]}{v[2]}{v[2]}{v[3]}{v[3]}"
    if len(v) != 7:
        return ""
    if any(c not in "0123456789abcdefABCDEF" for c in v[1:]):
        return ""
    return v.lower()

def build_theme_css_vars(value):
    hex_value = normalize_theme_color(value)
    if not hex_value:
        return ""

    r = int(hex_value[1:3], 16)
    g = int(hex_value[3:5], 16)
    b = int(hex_value[5:7], 16)

    def mix(channel, target, pct):
        return round(channel + (target - channel) * pct)

    def to_hex(red, green, blue):
        return f"#{red:02x}{green:02x}{blue:02x}"

    light = to_hex(mix(r, 255, 0.45), mix(g, 255, 0.45), mix(b, 255, 0.45))
    lighter = to_hex(mix(r, 255, 0.7), mix(g, 255, 0.7), mix(b, 255, 0.7))
    dark = to_hex(mix(r, 0, 0.18), mix(g, 0, 0.18), mix(b, 0, 0.18))
    darker = to_hex(mix(r, 0, 0.32), mix(g, 0, 0.32), mix(b, 0, 0.32))
    rgb = f"{r}, {g}, {b}"
    return (
        ":root{"
        f"--theme-500:{hex_value};"
        f"--theme-600:{dark};"
        f"--theme-700:{darker};"
        f"--theme-300:{light};"
        f"--theme-200:{lighter};"
        f"--theme-rgb:{rgb};"
        "}"
    )

def _normalize_webauthn_credentials(raw):
    if not raw:
        return []
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    creds = []
    seen_ids = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        cred_id = str(item.get('id') or '').strip()
        public_key = str(item.get('public_key') or '').strip()
        if not cred_id or not public_key or cred_id in seen_ids:
            continue
        try:
            base64url_to_bytes(cred_id)
            base64url_to_bytes(public_key)
        except Exception:
            continue
        try:
            sign_count = int(item.get('sign_count', 0) or 0)
        except Exception:
            sign_count = 0
        if sign_count < 0:
            sign_count = 0
        name = str(item.get('name') or '').strip() or 'Security Key'
        created_at = item.get('created_at')
        created_at_val = None
        if created_at is not None:
            created_at_val = str(created_at).strip() or None
        creds.append({
            'id': cred_id,
            'public_key': public_key,
            'sign_count': sign_count,
            'name': name,
            'created_at': created_at_val
        })
        seen_ids.add(cred_id)
    return creds

def _load_user_webauthn_credentials(user):
    if not user:
        return []
    return _normalize_webauthn_credentials(getattr(user, "webauthn_credentials", None))

def _save_user_webauthn_credentials(user, creds):
    normalized = _normalize_webauthn_credentials(creds)
    if not normalized:
        user.webauthn_credentials = None
        return []
    user.webauthn_credentials = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return normalized

def _serialize_public_webauthn_credentials(creds):
    rows = []
    for c in _normalize_webauthn_credentials(creds):
        rows.append({
            'id': c['id'],
            'name': c.get('name') or 'Security Key',
            'created_at': c.get('created_at')
        })
    return rows

def _refresh_user_2fa_state(user):
    has_totp = bool(getattr(user, 'totp_secret', None))
    has_webauthn = bool(_load_user_webauthn_credentials(user))
    if getattr(user, 'passkey_only_login', False) and not has_webauthn:
        user.passkey_only_login = False
    if not has_totp and not has_webauthn:
        user.is_2fa_enabled = False

