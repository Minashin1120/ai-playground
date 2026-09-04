@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

def get_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

def get_client_ip():
    # Apache appends its verified client address to the right side of XFF.
    forwarded = [part.strip() for part in request.headers.get('X-Forwarded-For', '').split(',') if part.strip()]
    candidates = list(reversed(forwarded))
    if request.remote_addr:
        candidates.append(str(request.remote_addr).strip())
    for candidate in candidates:
        try:
            return ip_address(candidate).compressed
        except ValueError:
            continue
    return None

def get_request_user_agent():
    value = str(request.headers.get('User-Agent', '') or '')
    return re.sub(r'[\r\n\x00-\x1f\x7f]+', ' ', value)[:512]

def _now_epoch_ms():
    return int(time.time() * 1000)

def _latency_trace_key(job_id):
    return f"{_LATENCY_TRACE_PREFIX}{job_id}"

def _latency_mark(job_id, phase, ts_ms=None, only_if_missing=False):
    if not job_id or not phase:
        return None
    try:
        v = int(ts_ms if ts_ms is not None else _now_epoch_ms())
    except Exception:
        v = _now_epoch_ms()
    try:
        k = _latency_trace_key(job_id)
        if only_if_missing:
            redis_conn.hsetnx(k, phase, v)
        else:
            redis_conn.hset(k, phase, v)
        redis_conn.expire(k, _LATENCY_TRACE_TTL_SECONDS)
    except Exception:
        pass
    return v

def _latency_mark_once(job_id, phase, ts_ms=None):
    return _latency_mark(job_id, phase, ts_ms=ts_ms, only_if_missing=True)

def _latency_read(job_id):
    if not job_id:
        return {}
    out = {}
    try:
        raw = redis_conn.hgetall(_latency_trace_key(job_id)) or {}
    except Exception:
        raw = {}
    for k, v in raw.items():
        try:
            ks = k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)
            vs = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
            out[ks] = int(vs)
        except Exception:
            continue
    return out

def _epoch_ms_to_utc_datetime(ms):
    try:
        ms_i = int(ms)
    except Exception:
        return None
    if ms_i < 946684800000 or ms_i > 4102444800000:
        return None
    try:
        return datetime.utcfromtimestamp(ms_i / 1000.0)
    except Exception:
        return None

def _upsert_chat_latency_trace(job_id, user_id, thread_public_id=None, model=None, execution_path=None, client_sent_at_ms=None, client_first_event_type=None, client_first_latency_ms=None, client_done_at_ms=None, client_total_latency_ms=None):
    if not job_id or not user_id:
        return None
    try:
        trace = ChatLatencyTrace.query.filter_by(job_id=job_id).first()
        if not trace:
            trace = ChatLatencyTrace(job_id=job_id, user_id=user_id)
        if thread_public_id:
            trace.thread_public_id = str(thread_public_id)[:64]
        if model:
            trace.model = str(model)[:80]
        if execution_path:
            trace.execution_path = str(execution_path)[:24]
        if client_sent_at_ms is not None:
            dt = _epoch_ms_to_utc_datetime(client_sent_at_ms)
            if dt:
                trace.client_sent_at = dt
        if client_first_event_type:
            trace.client_first_event_type = str(client_first_event_type)[:32]
        if client_first_latency_ms is not None:
            try:
                c_ms = max(0, int(client_first_latency_ms))
            except Exception:
                c_ms = None
            if c_ms is not None and (trace.client_first_latency_ms is None or c_ms < trace.client_first_latency_ms):
                trace.client_first_latency_ms = c_ms
        
        if client_done_at_ms is not None:
            dt = _epoch_ms_to_utc_datetime(client_done_at_ms)
            if dt:
                trace.client_done_at = dt
        if client_total_latency_ms is not None:
            try:
                t_ms = max(0, int(client_total_latency_ms))
            except Exception:
                t_ms = None
            if t_ms is not None:
                trace.client_total_latency_ms = t_ms

        phases = _latency_read(job_id)
        for phase_key, field_name in _LATENCY_PHASE_TO_FIELD.items():
            if getattr(trace, field_name, None) is not None:
                continue
            dt = _epoch_ms_to_utc_datetime(phases.get(phase_key))
            if dt:
                setattr(trace, field_name, dt)
        db.session.add(trace)
        safe_db_commit()
        return trace
    except Exception:
        db.session.rollback()
        return None

def _trace_delta_ms(trace, start_field, end_field):
    if not trace:
        return None
    a = getattr(trace, start_field, None)
    b = getattr(trace, end_field, None)
    if not a or not b:
        return None
    try:
        return int((b - a).total_seconds() * 1000)
    except Exception:
        return None

