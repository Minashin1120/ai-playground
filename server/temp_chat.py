def _normalize_attachment_source(value):
    raw = str(value or "").strip().lower()
    if raw in ("library", "lib", "library_attach"):
        return "library"
    if raw in ("upload", "uploaded", "new_upload"):
        return "upload"
    return "unknown"

def _iter_message_attachment_refs(raw_value):
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except Exception:
        parsed = [raw_value]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return parsed

def _delete_user_upload_ref(user_id, ref):
    norm = _normalize_upload_ref(ref)
    if not norm:
        return False
    if norm.startswith("..") or os.path.isabs(norm):
        return False
    if not norm.startswith(f"{user_id}/"):
        return False
    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
    if not _path_is_within(app.config['UPLOAD_FOLDER'], fp):
        return False
    secure_delete(fp)
    secure_delete(fp + '.enc')
    _delete_file_cache_for_path(user_id, norm)
    return True

def _temp_chat_member(thread):
    if not thread:
        return None
    return str(thread.public_id or thread.id)

def _temp_chat_state_key(member):
    return f"{_TEMP_CHAT_STATE_PREFIX}{member}"

def _temp_chat_uploads_key(member):
    return f"{_TEMP_CHAT_UPLOADS_PREFIX}{member}"

def _decode_redis_value(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return str(v) if v is not None else None

def _resolve_temp_chat_thread(member):
    if not member:
        return None
    ident = str(member).strip()
    if not ident:
        return None
    t = Thread.query.filter_by(public_id=ident).first()
    if t:
        return t
    if ident.isdigit():
        return Thread.query.get(int(ident))
    return None

def _clear_temp_chat_tracking(member):
    if not member:
        return
    try:
        pipe = redis_conn.pipeline()
        pipe.zrem(_TEMP_CHAT_LAST_SEEN_ZSET, member)
        pipe.delete(_temp_chat_state_key(member))
        pipe.delete(_temp_chat_uploads_key(member))
        pipe.execute()
    except Exception:
        pass

def _clear_temp_chat_tracking_for_thread(thread):
    member = _temp_chat_member(thread)
    if member:
        _clear_temp_chat_tracking(member)

def _normalize_temp_chat_timeout_seconds(value, fallback=None):
    base = _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS if fallback is None else fallback
    sec = _coerce_int_or_none(value)
    if sec is None:
        sec = base
    if sec < _TEMP_CHAT_TIMEOUT_MIN_SECONDS:
        sec = _TEMP_CHAT_TIMEOUT_MIN_SECONDS
    if sec > _TEMP_CHAT_TIMEOUT_MAX_SECONDS:
        sec = _TEMP_CHAT_TIMEOUT_MAX_SECONDS
    return int(sec)

def _get_user_temp_chat_timeout_seconds(user):
    if not user:
        return _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS
    return _normalize_temp_chat_timeout_seconds(getattr(user, "temp_chat_timeout_seconds", None))

def _resolve_temp_chat_timeout_seconds(thread=None, user_id=None, timeout_seconds=None):
    if timeout_seconds is not None:
        return _normalize_temp_chat_timeout_seconds(timeout_seconds)
    if thread is not None:
        try:
            if getattr(thread, "user", None) is not None:
                return _get_user_temp_chat_timeout_seconds(thread.user)
        except Exception:
            pass
        if user_id is None:
            user_id = thread.user_id
    uid = _coerce_int_or_none(user_id)
    if uid is not None:
        try:
            u = User.query.get(int(uid))
        except Exception:
            u = None
        if u:
            return _get_user_temp_chat_timeout_seconds(u)
    return _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS

def _resolve_temp_chat_member_timeout_seconds(member, thread=None):
    raw = None
    try:
        raw = redis_conn.hget(_temp_chat_state_key(member), "timeout_seconds")
    except Exception:
        raw = None
    parsed = _coerce_int_or_none(_decode_redis_value(raw))
    if parsed is not None:
        return _normalize_temp_chat_timeout_seconds(parsed)
    uid = thread.user_id if thread is not None else None
    return _resolve_temp_chat_timeout_seconds(thread=thread, user_id=uid)

def _mark_temp_chat_presence(thread, user_id=None, timeout_seconds=None):
    if not thread:
        return
    member = _temp_chat_member(thread)
    if not member:
        return
    uid = user_id if user_id is not None else thread.user_id
    timeout_val = _resolve_temp_chat_timeout_seconds(thread=thread, user_id=uid, timeout_seconds=timeout_seconds)
    now_ts = int(time.time())
    expires_at = now_ts + int(timeout_val)
    try:
        pipe = redis_conn.pipeline()
        pipe.zadd(_TEMP_CHAT_LAST_SEEN_ZSET, {member: now_ts})
        pipe.hset(_temp_chat_state_key(member), mapping={
            "user_id": str(uid or ""),
            "thread_id": str(thread.id or ""),
            "thread_public_id": str(thread.public_id or ""),
            "last_seen": str(now_ts),
            "timeout_seconds": str(timeout_val),
            "expires_at": str(expires_at),
        })
        pipe.expire(_temp_chat_state_key(member), _TEMP_CHAT_TRACK_TTL_SECONDS)
        pipe.execute()
    except Exception:
        pass

def _get_temp_chat_runtime_meta(thread, user=None):
    timeout_seconds = _resolve_temp_chat_timeout_seconds(
        thread=thread,
        user_id=(thread.user_id if thread is not None else None),
        timeout_seconds=(_get_user_temp_chat_timeout_seconds(user) if user is not None else None),
    )
    meta = {
        "timeout_seconds": int(timeout_seconds),
        "temp_chat_expires_at": None,
        "temp_chat_remaining_seconds": None,
    }
    if not thread or not bool(getattr(thread, "is_temporary", False)):
        return meta
    member = _temp_chat_member(thread)
    if not member:
        return meta
    now_ts = int(time.time())
    try:
        score = redis_conn.zscore(_TEMP_CHAT_LAST_SEEN_ZSET, member)
    except Exception:
        score = None
    if score is None:
        meta["temp_chat_expires_at"] = now_ts + int(timeout_seconds)
        meta["temp_chat_remaining_seconds"] = int(timeout_seconds)
        return meta
    try:
        timeout_seconds = _resolve_temp_chat_member_timeout_seconds(member, thread=thread)
    except Exception:
        timeout_seconds = _resolve_temp_chat_timeout_seconds(thread=thread, user_id=thread.user_id)
    expires_at = int(float(score) + float(timeout_seconds))
    remaining = max(0, int(expires_at - now_ts))
    meta["timeout_seconds"] = int(timeout_seconds)
    meta["temp_chat_expires_at"] = expires_at
    meta["temp_chat_remaining_seconds"] = remaining
    return meta

def _track_temp_chat_uploaded_refs(thread, user_id, refs):
    if not thread or not bool(getattr(thread, "is_temporary", False)):
        return
    if not refs:
        return
    member = _temp_chat_member(thread)
    if not member:
        return
    normalized = _normalize_attachment_list(refs, user_id)
    if not normalized:
        return
    try:
        pipe = redis_conn.pipeline()
        pipe.sadd(_temp_chat_uploads_key(member), *normalized)
        pipe.expire(_temp_chat_uploads_key(member), _TEMP_CHAT_TRACK_TTL_SECONDS)
        pipe.execute()
    except Exception:
        pass

def _cleanup_temp_chat_member(member, now_ts=None, force=False):
    if not member:
        return False
    member = str(member).strip()
    if not member:
        return False

    thread = _resolve_temp_chat_thread(member)
    if not thread:
        _clear_temp_chat_tracking(member)
        return False
    if not bool(getattr(thread, "is_temporary", False)) and not force:
        _clear_temp_chat_tracking(member)
        return False
    if now_ts is not None and not force:
        try:
            score = redis_conn.zscore(_TEMP_CHAT_LAST_SEEN_ZSET, member)
        except Exception:
            score = None
        if score is None:
            return False
        timeout_seconds = _resolve_temp_chat_member_timeout_seconds(member, thread=thread)
        expires_at = float(score) + float(timeout_seconds)
        if expires_at > float(now_ts):
            return False

    user_id = thread.user_id
    uploaded_paths = []
    try:
        raw_paths = redis_conn.smembers(_temp_chat_uploads_key(member)) or set()
        for raw in raw_paths:
            val = _decode_redis_value(raw)
            if val:
                uploaded_paths.append(val)
    except Exception:
        uploaded_paths = []

    for p in uploaded_paths:
        _delete_user_upload_ref(user_id, p)

    try:
        redis_conn.delete(f"pending_job:{user_id}:{thread.id}")
    except Exception:
        pass

    try:
        db.session.delete(thread)
        safe_db_commit()
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        logger.error(f"Temporary chat cleanup failed ({member}): {e}")
        return False

    _clear_temp_chat_tracking(member)
    log_force(
        f"Temporary chat auto-deleted: member={member}, user_id={user_id}, "
        f"deleted_uploads={len(uploaded_paths)}"
    )
    return True

def _cleanup_stale_temp_chats():
    while True:
        now_ts = int(time.time())
        cutoff = now_ts - _TEMP_CHAT_TIMEOUT_MIN_SECONDS
        try:
            stale_members = redis_conn.zrangebyscore(
                _TEMP_CHAT_LAST_SEEN_ZSET,
                "-inf",
                cutoff,
                start=0,
                num=50,
            )
        except Exception:
            return
        if not stale_members:
            return
        for raw_member in stale_members:
            member = _decode_redis_value(raw_member)
            if member:
                _cleanup_temp_chat_member(member, now_ts=now_ts)

def _temp_chat_monitor_has_lead():
    global _TEMP_CHAT_MONITOR_TOKEN
    if not _TEMP_CHAT_MONITOR_TOKEN:
        _TEMP_CHAT_MONITOR_TOKEN = f"{os.getpid()}:{threading.get_ident()}"
    token = _TEMP_CHAT_MONITOR_TOKEN
    try:
        if redis_conn.set(_TEMP_CHAT_MONITOR_LEADER_KEY, token, nx=True, ex=_TEMP_CHAT_MONITOR_LEASE_SECONDS):
            return True
        cur = redis_conn.get(_TEMP_CHAT_MONITOR_LEADER_KEY)
        cur_val = _decode_redis_value(cur)
        if cur_val == token:
            redis_conn.expire(_TEMP_CHAT_MONITOR_LEADER_KEY, _TEMP_CHAT_MONITOR_LEASE_SECONDS)
            return True
        return False
    except Exception:
        return True

def _temp_chat_monitor_loop():
    while True:
        try:
            if _temp_chat_monitor_has_lead():
                with app.app_context():
                    _cleanup_stale_temp_chats()
                    _maybe_sweep_stale_chunk_uploads()
                    _cleanup_all_stale_account_import_uploads()
        except Exception as e:
            logger.error(f"Temporary chat monitor error: {e}")
        time.sleep(_TEMP_CHAT_MONITOR_INTERVAL)

def _ensure_temp_chat_monitor_running():
    global _TEMP_CHAT_MONITOR_THREAD, _TEMP_CHAT_MONITOR_PID
    pid = os.getpid()
    with _TEMP_CHAT_MONITOR_LOCK:
        if (
            _TEMP_CHAT_MONITOR_THREAD
            and _TEMP_CHAT_MONITOR_THREAD.is_alive()
            and _TEMP_CHAT_MONITOR_PID == pid
        ):
            return
        th = threading.Thread(
            target=_temp_chat_monitor_loop,
            name=f"temp-chat-monitor-{pid}",
            daemon=True,
        )
        _TEMP_CHAT_MONITOR_THREAD = th
        _TEMP_CHAT_MONITOR_PID = pid
        th.start()

