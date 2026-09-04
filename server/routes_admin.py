@app.route('/api/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        data = request.json or {}
        title = (data.get('title') or "").strip()[:200]
        message = (data.get('message') or "").replace('\x00', '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        if len(message) > 100_000:
            return jsonify({'error': 'Message is too long'}), 400
        if not rate_limit(f"rl:feedback:user:{current_user.id}", 10, 3600):
            return jsonify({'error': 'rate_limit'}), 429
        fb = Feedback(user_id=current_user.id, title=title, message=message)
        db.session.add(fb)
        safe_db_commit()
        return jsonify({'status': 'ok'})

    # GET
    is_admin = bool(getattr(current_user, "is_admin", False))
    if is_admin and request.args.get('all') == '1':
        items = Feedback.query.order_by(Feedback.created_at.desc()).all()
    else:
        items = Feedback.query.filter_by(user_id=current_user.id).order_by(Feedback.created_at.desc()).all()

    res = []
    for f in items:
        res.append({
            'id': f.id,
            'user_id': f.user_id,
            'title': f.title or "",
            'message': f.message,
            'status': f.status,
            'admin_reply': f.admin_reply or "",
            'handled_by': f.handled_by or "",
            'created_at': f.created_at.isoformat(),
            'updated_at': f.updated_at.isoformat() if f.updated_at else None
        })
    return jsonify({'items': res, 'is_admin': is_admin})

@app.route('/api/easy_login', methods=['POST'])
@login_required
def create_easy_login():
    try:
        data = request.json or {}
        if data.get('cancel'):
            current_user.easy_login_hash = None
            current_user.easy_login_expires_at = None
            safe_db_commit()
            return jsonify({'status': 'ok', 'cancelled': True})
        minutes = data.get('minutes', 5)
        try:
            minutes = int(minutes)
        except Exception:
            minutes = 5
        if minutes < 1: minutes = 1
        if minutes > 120: minutes = 120

        temp_pw = secrets.token_urlsafe(16)
        current_user.easy_login_hash = generate_password_hash(temp_pw)
        current_user.easy_login_expires_at = datetime.utcnow() + timedelta(minutes=minutes)
        safe_db_commit()
        return jsonify({
            'status': 'ok',
            'temp_password': temp_pw,
            'expires_at': current_user.easy_login_expires_at.isoformat() + "Z",
            'minutes': minutes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/<int:fid>/update', methods=['POST'])
@login_required
def feedback_update(fid):
    if not getattr(current_user, "is_admin", False): return jsonify({'error': '403'}), 403
    fb = Feedback.query.get_or_404(fid)
    data = request.json or {}
    status = data.get('status')
    reply = data.get('admin_reply')
    if status:
        fb.status = status
    if reply is not None:
        fb.admin_reply = reply
    fb.handled_by = current_user.username
    fb.updated_at = datetime.utcnow()
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/ban/appeals/summary', methods=['GET'])
@login_required
def api_ban_appeals_summary():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    unread = BanAppeal.query.filter(BanAppeal.admin_read_at.is_(None)).count()
    return jsonify({'unread_count': unread})

@app.route('/api/ban/appeals', methods=['GET'])
@login_required
def api_ban_appeals():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    limit = request.args.get('limit') or '50'
    try:
        limit = max(1, min(200, int(limit)))
    except Exception:
        limit = 50
    items = BanAppeal.query.order_by(BanAppeal.created_at.desc()).limit(limit).all()
    res = []
    for a in items:
        res.append({
            'id': a.id,
            'user_id': a.user_id,
            'username': a.username,
            'message': a.message,
            'status': a.status,
            'admin_note': a.admin_note or "",
            'admin_reply': a.admin_reply or "",
            'admin_read_at': a.admin_read_at.isoformat() + "Z" if a.admin_read_at else None,
            'replied_at': a.replied_at.isoformat() + "Z" if a.replied_at else None,
            'handled_at': a.handled_at.isoformat() + "Z" if a.handled_at else None,
            'handled_by': a.handled_by or "",
            'ban_reason': a.ban_reason or "",
            'ban_at': a.ban_at.isoformat() + "Z" if a.ban_at else None,
            'ip_address': a.ip_address or "",
            'user_agent': a.user_agent or "",
            'evidence': a.evidence or "",
            'created_at': a.created_at.isoformat() + "Z" if a.created_at else None,
            'updated_at': a.updated_at.isoformat() + "Z" if a.updated_at else None
        })
    return jsonify({'items': res})

@app.route('/api/ban/appeals/mark_read', methods=['POST'])
@login_required
def api_ban_appeals_mark_read():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    ids = data.get('ids') or []
    mark_all = bool(data.get('all'))
    now = datetime.utcnow()
    if mark_all:
        BanAppeal.query.filter(BanAppeal.admin_read_at.is_(None)).update({'admin_read_at': now, 'updated_at': now})
        safe_db_commit()
        return jsonify({'status': 'ok', 'all': True})
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': 'ids_required'}), 400
    items = BanAppeal.query.filter(BanAppeal.id.in_(ids)).all()
    for a in items:
        if not a.admin_read_at:
            a.admin_read_at = now
        a.updated_at = now
    safe_db_commit()
    return jsonify({'status': 'ok', 'count': len(items)})

@app.route('/api/ban/appeals/update', methods=['POST'])
@login_required
def api_ban_appeals_update():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    appeal_id = data.get('id')
    if not appeal_id:
        return jsonify({'error': 'id_required'}), 400
    appeal = BanAppeal.query.get_or_404(int(appeal_id))
    status = data.get('status')
    admin_note = data.get('admin_note')
    admin_reply = data.get('admin_reply')
    block_user = bool(data.get('block_user')) if 'block_user' in data else False
    unblock_user = bool(data.get('unblock_user')) if 'unblock_user' in data else False
    block_reason = (data.get('block_reason') or '').strip()
    now = datetime.utcnow()
    if status in ['new', 'in_review', 'replied', 'resolved', 'rejected']:
        appeal.status = status
        if status in ['resolved', 'rejected']:
            appeal.handled_at = now
            appeal.handled_by = current_user.username
    if admin_note is not None:
        appeal.admin_note = admin_note.strip()[:2000]
    if admin_reply is not None:
        reply_text = admin_reply.strip()
        appeal.admin_reply = reply_text[:3000]
        appeal.replied_at = now if reply_text else None
        if not status:
            appeal.status = 'replied' if reply_text else appeal.status
    if not appeal.admin_read_at:
        appeal.admin_read_at = now
    appeal.updated_at = now
    if block_user or unblock_user:
        target_user = User.query.get(appeal.user_id)
        if target_user:
            if unblock_user:
                target_user.appeal_blocked = False
                target_user.appeal_block_reason = None
                target_user.appeal_blocked_at = None
            else:
                target_user.appeal_blocked = True
                target_user.appeal_block_reason = block_reason or "異議申し立てはブロックされています。"
                target_user.appeal_blocked_at = now
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/bot-telemetry', methods=['POST'])
@login_required
def bot_telemetry():
    if getattr(current_user, 'is_admin', False):
        return jsonify({'status': 'ok', 'skipped': True})
    if not get_bot_detection_global_enabled() or not current_user.bot_detection_enabled:
        return jsonify({'status': 'disabled'})
    if current_user.is_bot_banned:
        return jsonify({'error': 'banned'}), 403
    data = request.json or {}
    # Script-injected synthetic input events (isTrusted === false) cannot be
    # produced by a normal user. Treat them as definitive bot evidence and ban
    # immediately, regardless of Turnstile status.
    if data.get('untrusted_input'):
        _apply_bot_ban("Synthetic (script-injected) input events detected")
        return jsonify({'error': 'banned', 'reasons': ['untrusted_input']}), 403
    if not verify_turnstile(data.get('turnstile_token')):
        if not data.get('turnstile_failed'):
            return jsonify({'error': 'turnstile_failed'}), 403
        # A Turnstile failure is only counted toward a ban when the client was
        # actually shown the verification dialog (challenged). The silent phase
        # (no dialog) must never accumulate failures that could ban a user who
        # was never given the chance to complete the challenge. Already-verified
        # users also never accumulate (handled inside _bot_turnstile_register_failure).
        if data.get('challenged') and not _bot_turnstile_verified() and _bot_turnstile_register_failure():
            return jsonify({'error': 'banned'}), 403
    raw_behavior, reasons = evaluate_bot_score(data)
    if data.get('turnstile_failed'):
        total_score = raw_behavior + 2
        reasons.append('turnstile_failed')
    else:
        total_score = raw_behavior
    if raw_behavior > 0 or data.get('turnstile_failed'):
        try:
            if rate_limit(f"rl:bot_ev_log:{current_user.id}", 60, 60):
                _log_bot_evidence(
                    'telemetry',
                    score=total_score,
                    behavior_score=raw_behavior,
                    reasons=",".join(reasons),
                    details=", ".join(f"{k}={data.get(k)}" for k in (
                        'window_ms', 'clicks', 'keys', 'fast_clicks', 'fast_keys',
                        'click_burst', 'key_burst', 'event_rate', 'avg_click_ms',
                        'click_cv', 'pointer_speed_max'
                    ) if data.get(k) is not None)
                )
        except Exception:
            pass
    if total_score <= 0:
        return jsonify({'status': 'ok', 'score': 0})
    key = f"bot:score:{current_user.id}"
    behavior_key = f"bot:behavior:{current_user.id}"
    try:
        new_score = redis_conn.incrbyfloat(key, float(total_score))
        redis_conn.expire(key, 300)
        new_behavior = redis_conn.incrbyfloat(behavior_key, float(raw_behavior))
        redis_conn.expire(behavior_key, 300)
    except Exception:
        new_score = float(total_score)
        new_behavior = float(raw_behavior)
    # Score-based ban: require genuine automated-behavior evidence. (A separate
    # Turnstile-failure ban is handled by _bot_turnstile_register_failure when
    # the challenge fails repeatedly.)
    if new_score >= 8 and new_behavior >= 6:
        _apply_bot_ban("Automated behavior detected (fast clicks/inputs)")
        return jsonify({'error': 'banned', 'score': new_score, 'reasons': reasons}), 403
    return jsonify({'status': 'ok', 'score': new_score, 'reasons': reasons})

@app.route('/api/bot/turnstile-verify', methods=['POST'])
@login_required
def bot_turnstile_verify():
    """Verify a client-provided Turnstile token and mark the user as verified.

    Turnstile tokens are single-use. Concurrent client paths (widget callback +
    gate + getTurnstileToken) can submit the same token several times; the first
    succeeds and the rest fail with Cloudflare's timeout-or-duplicate. Those
    races must not ban a user who just verified successfully.
    """
    if not _bot_turnstile_active():
        return jsonify({'status': 'ok', 'skipped': True})
    if current_user.is_bot_banned:
        return jsonify({'error': 'banned'}), 403
    # Already verified: never re-verify or count failures for this user.
    if _bot_turnstile_verified():
        return jsonify({'status': 'ok', 'already': True})
    if not rate_limit(f"rl:bot_tst_verify:{current_user.id}", 30, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    token = data.get('turnstile_token')
    # Per-token dedup: only the first request for a given token may call
    # siteverify / count a failure. Concurrent duplicates of a spent token
    # used to produce verify_ok + N×verify_fail in the same second and ban.
    token_seen_key = None
    if token:
        try:
            token_fp = hashlib.sha256(str(token).encode('utf-8', errors='ignore')).hexdigest()[:32]
            token_seen_key = f"bot:tst:tok:{current_user.id}:{token_fp}"
            claimed = redis_conn.set(token_seen_key, "pending", nx=True, ex=120)
            if not claimed:
                prev = redis_conn.get(token_seen_key)
                prev_s = prev.decode('utf-8', errors='ignore') if isinstance(prev, (bytes, bytearray)) else str(prev or '')
                if prev_s in ('ok', 'pending') or _bot_turnstile_verified():
                    # Another request is using / already accepted this token.
                    if prev_s == 'ok' or _bot_turnstile_verified():
                        return jsonify({'status': 'ok', 'already': True})
                    # Still pending elsewhere — soft-fail without counting.
                    return jsonify({'error': 'turnstile_failed', 'dedup': True}), 403
                # Previous attempt with this token failed: soft-fail, no re-count.
                return jsonify({'error': 'turnstile_failed', 'dedup': True}), 403
        except Exception:
            token_seen_key = None
    if not verify_turnstile(token):
        if token_seen_key:
            try:
                redis_conn.set(token_seen_key, "fail", ex=120)
            except Exception:
                pass
        # A concurrent sibling request may have just marked the user verified.
        if _bot_turnstile_verified():
            return jsonify({'status': 'ok', 'already': True})
        _log_bot_evidence('verify_fail', reasons='turnstile_failed')
        # Only count toward a ban when the client actually showed the dialog
        # (challenged). Silent background verification failures must not ban a
        # user who was never shown the challenge.
        if data.get('challenged') and _bot_turnstile_register_failure():
            return jsonify({'error': 'banned'}), 403
        return jsonify({'error': 'turnstile_failed'}), 403
    if token_seen_key:
        try:
            redis_conn.set(token_seen_key, "ok", ex=120)
        except Exception:
            pass
    if _bot_turnstile_register_success():
        return jsonify({'error': 'banned'}), 403
    _log_bot_evidence('verify_ok')
    return jsonify({'status': 'ok'})

@app.route('/api/bot/lock', methods=['POST'])
@login_required
def bot_lock():
    """Apply a temporary lock after suspicious rapid operation (e.g. send spam).

    The client reports rapid send-button clicking; the server locks the account
    for _BOT_LOCK_TTL seconds and returns the reason. Repeated locks escalate
    to a ban.
    """
    if _is_admin_exempt(current_user):
        return jsonify({'status': 'skipped', 'skipped': True})
    if not get_bot_detection_global_enabled() or not current_user.bot_detection_enabled:
        return jsonify({'status': 'disabled'})
    if current_user.is_bot_banned:
        return jsonify({'error': 'banned'}), 403
    if not rate_limit(f"rl:bot_lock:{current_user.id}", 5, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    reason = str(data.get('reason') or '送信操作が速すぎるため、一時的にロックしています。')[:300]
    result = _apply_bot_lock(reason)
    if result.get('status') == 'skipped':
        return jsonify({'status': 'skipped', 'skipped': True})
    if result.get('status') == 'banned':
        return jsonify({'error': 'banned', 'message': 'ロックが繰り返されたため、BANされました。'}), 403
    active, active_reason, remaining = _bot_lock_info()
    return jsonify({
        'status': 'locked',
        'message': active_reason or reason,
        'remaining_seconds': remaining,
    })

@app.route('/api/bot/lock-status', methods=['GET'])
@login_required
def bot_lock_status():
    """Return the current lock state so the UI can render the lock screen."""
    active, reason, remaining = _bot_lock_info()
    return jsonify({
        'active': active,
        'message': reason or '',
        'remaining_seconds': remaining,
    })

@app.route('/api/bot/unban', methods=['POST'])
@login_required
def bot_unban():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': 'username_required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'not_found'}), 404
    mode = (data.get('mode') or 'single').strip().lower()
    if mode == 'linked':
        unban_linked_accounts(user)
    else:
        unban_single_account(user)
    return jsonify({'status': 'ok', 'username': username, 'mode': mode})

def _deny_unless_primary_admin_for_speedtest():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    if not _is_primary_admin_user(current_user):
        return jsonify({'error': 'primary_admin_only'}), 403
    return None

def _mark_speedtest_no_store(resp):
    try:
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp

@app.route('/api/speedtest/ping', methods=['GET'])
@login_required
def speedtest_ping():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    resp = jsonify({
        'status': 'ok',
        'server_time_ms': int(time.time() * 1000)
    })
    return _mark_speedtest_no_store(resp)

@app.route('/api/speedtest/download', methods=['GET'])
@login_required
def speedtest_download():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    try:
        size = int(request.args.get('bytes') or 0)
    except Exception:
        size = 0
    size = max(64 * 1024, min(32 * 1024 * 1024, size or (8 * 1024 * 1024)))
    chunk = (b'0123456789abcdef' * 4096)  # 64KB deterministic payload
    chunk_len = len(chunk)

    def generate():
        remaining = size
        while remaining > 0:
            n = min(chunk_len, remaining)
            yield chunk[:n]
            remaining -= n

    resp = Response(stream_with_context(generate()), mimetype='application/octet-stream')
    resp.headers['Content-Length'] = str(size)
    resp.headers['X-Speedtest-Bytes'] = str(size)
    return _mark_speedtest_no_store(resp)

@app.route('/api/speedtest/upload', methods=['POST'])
@login_required
def speedtest_upload():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    start = time.perf_counter()
    total = 0
    max_bytes = 32 * 1024 * 1024
    try:
        while True:
            chunk = request.stream.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return jsonify({'error': 'payload_too_large', 'max_bytes': max_bytes}), 413
    except RequestEntityTooLarge:
        return jsonify({'error': 'payload_too_large', 'max_bytes': max_bytes}), 413
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    resp = jsonify({
        'status': 'ok',
        'bytes_received': total,
        'server_elapsed_ms': elapsed_ms
    })
    return _mark_speedtest_no_store(resp)

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
        res.append({
            'username': u.username,
            'bot_detection_enabled': u.bot_detection_enabled if u.bot_detection_enabled is not None else True,
            'is_bot_banned': bool(u.is_bot_banned),
            'bot_ban_reason': u.bot_ban_reason,
            'bot_banned_at': u.bot_banned_at.isoformat() + "Z" if u.bot_banned_at else None
        })
    return jsonify({'users': res})

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
    
    if action == 'toggle_detection':
        enabled = bool(data.get('enabled'))
        user.bot_detection_enabled = enabled
    elif action == 'ban':
        user.is_bot_banned = True
        user.bot_banned_at = datetime.utcnow()
        user.bot_ban_reason = data.get('reason') or "Manual ban"
        user.bot_unban_notice = False
        ban_related_accounts(user, user.bot_ban_reason)
    elif action == 'unban':
        unban_single_account(user)
    elif action == 'unban_linked':
        unban_linked_accounts(user)
    elif action == 'delete_account':
        _delete_user_account_immediately(user)
        return jsonify({'status': 'ok', 'username': username, 'action': action})
    else:
        return jsonify({'error': 'bad_action'}), 400
    
    safe_db_commit()
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

