"""Opt-in account diagnostics; Redis TTL never resets an existing event's age.

Loaded into app's shared namespace after request_identity. No diagnostic entries
are sent to operational files, stdout or the Python logger.
"""
from server.logging_policy import safe_entry as _safe_log_entry, safe_version as _safe_log_version

_SERVER_DIAGNOSTIC_SECONDS = 3600
_SERVER_DIAGNOSTIC_MAX_ENTRIES = 2000


def _server_diagnostic_key(user_id):
    return f'diagnostic:v2:{int(user_id)}'


def _server_diagnostic_enabled(user_id):
    # Execute a fresh column query: long-running workers must observe revocation.
    with app.app_context(), db.engine.connect() as connection:
        return bool(connection.execute(db.select(User.enable_server_diagnostic_log).where(User.id == int(user_id))).scalar())


def _clear_server_diagnostics(user_id):
    try:
        key = _server_diagnostic_key(user_id)
        generation = redis_conn.get(key + ':generation')
        # Rotate namespace first so a previously started writer cannot resurrect visible entries.
        redis_conn.set(key + ':generation', secrets.token_hex(16), ex=_SERVER_DIAGNOSTIC_SECONDS)
        if generation:
            generation = generation.decode() if isinstance(generation, bytes) else generation
            index = key + ':' + generation
            members = redis_conn.zrange(index, 0, -1)
            pipe = redis_conn.pipeline()
            for member in members:
                member = member.decode() if isinstance(member, bytes) else member
                pipe.delete(index + ':' + member)
            pipe.delete(index)
            pipe.execute()
    except Exception:
        pass


def _record_server_diagnostic(user_id, event, **fields):
    try:
        if not _server_diagnostic_enabled(user_id):
            return
        now = time.time()
        entry = _safe_log_entry(dict(fields, ev=event, t=int(now * 1000)))
        if entry is None:
            return
        base = _server_diagnostic_key(user_id)
        candidate = secrets.token_hex(16)
        redis_conn.set(base + ':generation', candidate, nx=True, ex=_SERVER_DIAGNOSTIC_SECONDS)
        generation = redis_conn.get(base + ':generation')
        generation = generation.decode() if isinstance(generation, bytes) else generation
        redis_conn.expire(base + ':generation', _SERVER_DIAGNOSTIC_SECONDS)
        index = base + ':' + generation
        member = secrets.token_hex(16)
        if not _server_diagnostic_enabled(user_id):
            return
        redis_conn.eval("""
            if redis.call('GET', KEYS[3]) ~= ARGV[6] then return 0 end
            redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[4])
            redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
            local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[2]) - tonumber(ARGV[4]))
            for _, id in ipairs(expired) do
                redis.call('DEL', KEYS[1] .. ':' .. id)
                redis.call('ZREM', KEYS[1], id)
            end
            local extra = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[5])
            if extra > 0 then
                local old = redis.call('ZRANGE', KEYS[1], 0, extra - 1)
                for _, id in ipairs(old) do
                    redis.call('DEL', KEYS[1] .. ':' .. id)
                    redis.call('ZREM', KEYS[1], id)
                end
            end
            redis.call('EXPIRE', KEYS[1], ARGV[4])
            return 1
        """, 3, index, index + ':' + member, base + ':generation', member, now,
            json.dumps(entry, separators=(',', ':')), _SERVER_DIAGNOSTIC_SECONDS,
            _SERVER_DIAGNOSTIC_MAX_ENTRIES, generation)
    except Exception:
        # Logging is best effort and must not change application behavior.
        pass


def _recent_server_diagnostics(user_id):
    try:
        if not _server_diagnostic_enabled(user_id):
            return []
        base = _server_diagnostic_key(user_id)
        generation = redis_conn.get(base + ':generation')
        if not generation:
            return []
        generation = generation.decode() if isinstance(generation, bytes) else generation
        index = base + ':' + generation
        cutoff = time.time() - _SERVER_DIAGNOSTIC_SECONDS
        members = redis_conn.zrangebyscore(index, cutoff, '+inf', start=0, num=_SERVER_DIAGNOSTIC_MAX_ENTRIES)
        if not members:
            return []
        keys = [index + ':' + (m.decode() if isinstance(m, bytes) else m) for m in members]
        entries = []
        for raw in redis_conn.mget(keys):
            entry = _safe_log_entry(json.loads(raw)) if raw else None
            if entry and entry.get('t', 0) > cutoff * 1000:
                entries.append(entry)
        return entries
    except Exception:
        return []


@app.before_request
def _diagnostic_request_start():
    try:
        if not current_user.is_authenticated or not _server_diagnostic_enabled(current_user.id):
            return
        g.diagnostic_started = time.monotonic()
        # Generated locally, never trust a content-bearing request header.
        g.diagnostic_op = secrets.token_hex(16)
        _record_server_diagnostic(current_user.id, 'server.request.start', op=g.diagnostic_op,
                                  route=request.url_rule.rule if request.url_rule else None, method=request.method)
    except Exception:
        pass


@app.after_request
def _diagnostic_request_end(response):
    try:
        if hasattr(g, 'diagnostic_started'):
            _record_server_diagnostic(current_user.id, 'server.request.end', op=g.diagnostic_op,
                route=request.url_rule.rule if request.url_rule else None, method=request.method,
                status=response.status_code, ms=round((time.monotonic() - g.diagnostic_started) * 1000))
            response.headers['X-Diagnostic-ID'] = g.diagnostic_op
    except Exception:
        pass
    return response


def _diagnostic_operation(name):
    """Wrap named implementation stages without observing arguments or return values."""
    from functools import wraps
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                uid = getattr(g, 'diagnostic_user_id', None)
                if uid is None and current_user.is_authenticated:
                    uid = current_user.id
            except (RuntimeError, AttributeError):
                uid = None
            if uid is None:
                return function(*args, **kwargs)
            started = time.monotonic()
            op = secrets.token_hex(16)
            _record_server_diagnostic(uid, name + '.start', op=op)
            try:
                result = function(*args, **kwargs)
            except Exception:
                _record_server_diagnostic(uid, name + '.error', op=op,
                    ms=round((time.monotonic() - started) * 1000))
                raise
            _record_server_diagnostic(uid, name + '.done', op=op,
                ms=round((time.monotonic() - started) * 1000))
            return result
        return wrapped
    return decorate
