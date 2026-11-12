# Admin protection: BAN appeals, bot telemetry, Turnstile verification, bot lock/unban and the
# primary-administrator speed test. Split from routes_admin.py (see server/README.md); every
# server part is exec()'d into the app module globals, so names stay shared.

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
    # The Android app has no in-page Turnstile widget; its bearer token already passed
    # Play Integrity or Turnstile, and the chat Turnstile gate still applies to it.
    if getattr(g, 'mobile_bearer_request', False):
        data.pop('turnstile_failed', None)
        data.pop('challenged', None)
    elif not verify_turnstile(data.get('turnstile_token')):
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

