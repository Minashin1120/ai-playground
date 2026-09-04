def generate_thread_public_id():
    for _ in range(8):
        candidate = secrets.token_urlsafe(32)
        if not Thread.query.filter_by(public_id=candidate).first():
            return candidate
    return secrets.token_urlsafe(32)

_JOB_ID_RE = re.compile(r"^job_[0-9]{10}_[0-9]+(?:_[0-9a-f]{16})?$")

def _is_valid_job_id(job_id):
    return bool(_JOB_ID_RE.fullmatch(str(job_id or '')))

def _pending_job_id_for_thread(user_id, thread_db_id):
    try:
        pending_raw = redis_conn.get(f"pending_job:{user_id}:{thread_db_id}")
    except Exception:
        return None
    if not pending_raw:
        return None
    try:
        pending_obj = json.loads(pending_raw)
        return str((pending_obj or {}).get('job_id') or '') or None
    except Exception:
        try:
            return pending_raw.decode("utf-8", "ignore") or None
        except Exception:
            return None

def _chat_submission_key(user_id, client_request_id):
    return f"chat_submission:{int(user_id)}:{client_request_id}"

def _claim_chat_submission(user_id, client_request_id):
    """Claim one client send ID or return its existing processing/accepted state."""
    if not client_request_id:
        return True, None
    key = _chat_submission_key(user_id, client_request_id)
    try:
        claimed = redis_conn.set(key, "processing", nx=True, ex=600)
        if claimed:
            return True, None
        raw = redis_conn.get(key)
        if not raw:
            return False, {"state": "processing"}
        text_value = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        if text_value == "processing":
            return False, {"state": "processing"}
        parsed = json.loads(text_value)
        return False, parsed if isinstance(parsed, dict) else {"state": "processing"}
    except Exception:
        # Availability must not depend on the dedupe cache. Redis is also
        # required by chat dispatch, so a wider outage will still fail safely.
        return True, None

def _complete_chat_submission(user_id, client_request_id, job_id, thread_public_id, message_id, model):
    if not client_request_id:
        return
    payload = {
        "state": "accepted",
        "job_id": str(job_id),
        "thread_id": str(thread_public_id),
        "message_id": int(message_id),
        "model": str(model or "")[:80],
    }
    _store_idempotent_submission(user_id, client_request_id, payload)

def _store_idempotent_submission(user_id, client_request_id, payload):
    if not client_request_id:
        return
    stored = dict(payload or {})
    stored["state"] = "accepted"
    try:
        redis_conn.setex(
            _chat_submission_key(user_id, client_request_id),
            600,
            json.dumps(stored, ensure_ascii=False),
        )
    except Exception:
        pass

def _release_chat_submission(user_id, client_request_id):
    if not client_request_id:
        return
    try:
        redis_conn.delete(_chat_submission_key(user_id, client_request_id))
    except Exception:
        pass

def resolve_thread_for_user(identifier, user_id):
    if identifier is None:
        return None
    ident_str = str(identifier).strip()
    if not ident_str:
        return None
    
    # Try public_id first
    t = Thread.query.filter_by(public_id=ident_str, user_id=user_id).first()
    if t:
        return t
        
    # Try numerical id
    if ident_str.isdigit():
        t = Thread.query.get(int(ident_str))
        if t and t.user_id == user_id:
            return t
            
    return None

def create_user_session(user):
    ip = get_client_ip()
    ua = get_request_user_agent()
    now = datetime.utcnow()
    old_sessions = UserSession.query.filter(
        UserSession.user_id == user.id,
        UserSession.is_revoked == False,
        UserSession.ip_address == ip,
        UserSession.user_agent == ua
    ).all()
    for s in old_sessions:
        s.is_revoked = True
        s.revoked_at = now
    sid = secrets.token_urlsafe(32)
    session['session_id'] = sid
    user_sess = UserSession(
        user_id=user.id,
        session_id=sid,
        user_agent=ua,
        ip_address=ip
    )
    db.session.add(user_sess)
    safe_db_commit()
    return user_sess

def revoke_user_sessions(user_id, exclude_session_id=None):
    q = UserSession.query.filter_by(user_id=user_id, is_revoked=False)
    if exclude_session_id:
        q = q.filter(UserSession.session_id != exclude_session_id)
    now = datetime.utcnow()
    for s in q.all():
        s.is_revoked = True
        s.revoked_at = now
    safe_db_commit()

