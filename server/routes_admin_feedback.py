# --- Files kept with each feedback ---
# ``feedback/<public_id>/`` under the app directory holds what the server keeps about one feedback
# besides its database row: ``feedback.json`` (title, message, sender, client, status and
# replies), ``logs/activity.jsonl`` (the client's activity log, below), ``logs/reply-<id>.jsonl``
# (the activity log sent with a reply, when the person ticked the box) and the copy of the open
# chat (``chat.jsonl`` and ``chat.files/``, server/feedback_chat_copy.py).
#
# Client activity logs ("ログの収集を強化"): the Web and Android clients record their own
# operations (never prompts, answers or file contents) while the user enables it, and attach
# the last hour to a feedback.
_FEEDBACK_LOG_MAX_ENTRIES = 50_000
_FEEDBACK_LOG_MAX_BYTES = 3_500_000
_FEEDBACK_LOG_MAX_LINE = 64 * 1024
_FEEDBACK_LOG_KEEP_FILES = 500
_FEEDBACK_INFO_NAME = 'feedback.json'
# WSGI environ key through which scripts/update_feedback_status.py names the agent recorded in ``handled_by``.
FEEDBACK_AGENT_ENVIRON = 'aiplayground.feedback_agent'
_FEEDBACK_ACTIVITY_LOG = os.path.join('logs', 'activity.jsonl')


def _feedback_root_dir():
    return app.config.get('FEEDBACK_DIR') or os.path.join(app.root_path, 'feedback')


# A feedback is identified to people and on disk by ``Feedback.public_id`` (random, 12 lowercase
# hex characters); the database ``id`` stays internal. Older clients still address a feedback by
# that number, so the routes accept either (``_feedback_by_ref``).
_FEEDBACK_PUBLIC_ID_RE = re.compile(r'[0-9a-f]{12}')


def _feedback_dir(public_id):
    public_id = str(public_id or '')
    if not _FEEDBACK_PUBLIC_ID_RE.fullmatch(public_id):
        raise ValueError('invalid feedback id')
    return os.path.join(_feedback_root_dir(), public_id)


def _feedback_dir_ids():
    try:
        return [name for name in os.listdir(_feedback_root_dir()) if _FEEDBACK_PUBLIC_ID_RE.fullmatch(name)]
    except OSError:
        return []


def _ensure_feedback_public_id(fb):
    """Gives a feedback created before public ids a random one and renames its old numbered directory."""
    if fb.public_id:
        return fb.public_id
    for _attempt in range(10):
        candidate = _new_feedback_public_id()
        if not Feedback.query.filter_by(public_id=candidate).first():
            break
    fb.public_id = candidate
    legacy = os.path.join(_feedback_root_dir(), str(int(fb.id)))
    try:
        if os.path.isdir(legacy):
            os.rename(legacy, _feedback_dir(candidate))
    except OSError as e:
        log_force(f"FEEDBACK-DIR-RENAME-ERROR: feedback={fb.id} err={e}")
    return candidate


def _feedback_by_ref(ref):
    """The feedback named by a public id, or by the old database number (older Android apps)."""
    ref = str(ref or '')
    if _FEEDBACK_PUBLIC_ID_RE.fullmatch(ref):
        return Feedback.query.filter_by(public_id=ref).first()
    if ref.isdigit() and len(ref) < 12:
        return db.session.get(Feedback, int(ref))
    return None


def _feedback_activity_log_path(feedback_id):
    return os.path.join(_feedback_dir(feedback_id), _FEEDBACK_ACTIVITY_LOG)


def _feedback_reply_log_path(feedback_id, reply_id):
    return os.path.join(_feedback_dir(feedback_id), 'logs', f'reply-{int(reply_id)}.jsonl')


def _feedback_log_files(feedback_id):
    """The activity logs kept with a feedback: the one sent with it and those sent with replies."""
    directory = os.path.dirname(_feedback_activity_log_path(feedback_id))
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith('.jsonl'))
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names if os.path.isfile(os.path.join(directory, n))]


def _feedback_replies(feedbacks):
    """{feedback id: replies, oldest first}. A reply saved before the history existed (only
    ``Feedback.admin_reply``) becomes the first entry; the caller commits it."""
    feedbacks = list(feedbacks)
    groups = {fb.id: [] for fb in feedbacks}
    if groups:
        rows = FeedbackReply.query.filter(FeedbackReply.feedback_id.in_(list(groups))).order_by(
            FeedbackReply.created_at.asc(), FeedbackReply.id.asc()).all()
        for row in rows:
            groups[row.feedback_id].append(row)
    for fb in feedbacks:
        if not groups[fb.id] and (fb.admin_reply or '').strip():
            legacy = FeedbackReply(feedback_id=fb.id, role='admin', author=fb.handled_by, message=fb.admin_reply,
                                   created_at=fb.updated_at or fb.created_at)
            db.session.add(legacy)
            groups[fb.id].append(legacy)
    return groups


def _feedback_reply_json(reply, public_id, is_admin):
    log_path = _feedback_reply_log_path(public_id, reply.id) if is_admin and reply.id else None
    image_dir = _feedback_reply_images_path(public_id, reply.id) if reply.id else None
    images = _feedback_image_names_in(image_dir) if image_dir else []
    return {
        'id': reply.id, 'role': reply.role, 'message': reply.message,
        # The administrator's name is shown to administrators only, like ``log_file``.
        'author': (reply.author or '') if is_admin else '',
        'created_at': reply.created_at.isoformat() if reply.created_at else None,
        'log_file': _feedback_display_path(log_path) if log_path else None,
        'chat_file': _feedback_display_path(_feedback_chat_copy_path(public_id, reply.id)) if is_admin and reply.id else None,
        'image_count': len(images),
        'image_dir': os.path.relpath(image_dir, app.root_path) if is_admin and images else None,
    }


def _feedback_display_path(path):
    """Path shown to administrators (relative to the app directory), or None when it does not exist."""
    return os.path.relpath(path, app.root_path) if os.path.isfile(path) else None


def _reset_feedback_dir(feedback_id):
    """Starts an empty directory for a new feedback (a reused id never inherits older files)."""
    path = _feedback_dir(feedback_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(_feedback_root_dir(), mode=0o700, exist_ok=True)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _write_feedback_info(fb, **extra):
    """Writes ``feedback.json`` for ``fb``, keeping what an earlier write recorded about the submission."""
    directory = _feedback_dir(fb.public_id)
    path = os.path.join(directory, _FEEDBACK_INFO_NAME)
    info = {}
    try:
        with open(path, encoding='utf-8') as handle:
            info = json.load(handle)
    except (OSError, ValueError):
        pass
    if not isinstance(info, dict):
        info = {}
    user = db.session.get(User, fb.user_id)
    replies = _feedback_replies([fb])[fb.id]
    if db.session.new:
        safe_db_commit()
    info.update(extra)
    info.update({
        'feedback_id': fb.public_id, 'user_id': fb.user_id, 'username': user.username if user else None,
        'title': fb.title or '', 'message': fb.message, 'status': fb.status,
        'admin_reply': fb.admin_reply or '', 'handled_by': fb.handled_by or '', 'agent_name': fb.agent_name or '',
        'created_at': fb.created_at.isoformat(timespec='seconds') + 'Z' if fb.created_at else None,
        'updated_at': fb.updated_at.isoformat(timespec='seconds') + 'Z' if fb.updated_at else None,
        'replies': [{
            'id': r.id, 'role': r.role, 'user_id': r.user_id, 'author': r.author or '', 'message': r.message,
            'created_at': r.created_at.isoformat(timespec='seconds') + 'Z' if r.created_at else None,
            'activity_log': f'logs/reply-{r.id}.jsonl' if r.activity_log_saved else None,
            'chat_copy': os.path.basename(_feedback_chat_copy_path(fb.public_id, r.id)) if r.id and os.path.isfile(_feedback_chat_copy_path(fb.public_id, r.id)) else None,
            'images': [f'images/reply-{r.id}/{name}' for name in _feedback_image_names_in(
                _feedback_reply_images_path(fb.public_id, r.id))] if r.id else [],
        } for r in replies],
    })
    os.makedirs(_feedback_root_dir(), mode=0o700, exist_ok=True)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    temp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2, default=str)
        handle.write('\n')
    os.replace(temp, path)


def _prune_feedback_log_files():
    paths = [p for fid in _feedback_dir_ids() for p in _feedback_log_files(fid)]
    if len(paths) <= _FEEDBACK_LOG_KEEP_FILES:
        return
    dated = []
    for path in paths:
        try:
            dated.append((os.path.getmtime(path), path))
        except OSError:
            pass
    dated.sort()
    for _mtime, path in dated[:len(dated) - _FEEDBACK_LOG_KEEP_FILES]:
        try:
            os.remove(path)
            os.rmdir(os.path.dirname(path))
        except OSError:
            pass


def _feedback_client(data, client_logs, chat_copy):
    """(client, client_version) of a feedback, also when neither logs nor a chat copy went with it."""
    sent = next((p for p in (data, client_logs, chat_copy) if isinstance(p, dict) and p.get('client')), {})
    client = str(sent.get('client') or '').lower()
    version = str(sent.get('version') or '')[:64] or None
    if not client:
        # The Android app names itself in every request's User-Agent (data/PlaygroundApi.kt).
        android = re.match(r'AIPlayground-Android/(\S+)', str(request.headers.get('User-Agent') or ''))
        client, version = ('android', android.group(1)[:64]) if android else ('unknown', None)
    if not re.fullmatch(r'[a-z]{1,16}', client):
        client = 'unknown'
    return client, version


def _save_feedback_client_logs(feedback_id, user_id, payload, reply_id=None):
    """Writes the client's activity log entries for ``feedback_id`` (or its reply ``reply_id``); returns the file path (None when empty)."""
    entries = payload.get('entries')
    if not isinstance(entries, list):
        return None
    entries = entries + _recent_server_diagnostics(user_id)
    if not entries:
        return None
    client = str(payload.get('client') or 'unknown').lower()
    if not re.fullmatch(r'[a-z]{1,16}', client):
        client = 'unknown'
    received = datetime.utcnow()
    lines = []
    total = 0
    dropped = 0
    for entry in entries[-_FEEDBACK_LOG_MAX_ENTRIES:]:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        entry = _safe_log_entry(entry)
        if entry is None:
            dropped += 1
            continue
        line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
        if len(line) > _FEEDBACK_LOG_MAX_LINE:
            line = json.dumps({'t': entry.get('t'), 'ev': str(entry.get('ev') or '')[:80], 'truncated': len(line)},
                              ensure_ascii=False, default=str, separators=(',', ':'))
        total += len(line) + 1
        if total > _FEEDBACK_LOG_MAX_BYTES:
            dropped += 1
            continue
        lines.append(line)
    dropped += max(0, len(entries) - _FEEDBACK_LOG_MAX_ENTRIES)
    meta = {
        'kind': 'meta', 'feedback_id': feedback_id, 'user_id': user_id, 'client': client,
        'client_version': _safe_log_version(payload.get('version')),
        'server_version': app.config.get('SYSTEM_VERSION'),
        'user_agent': 'android' if client == 'android' else 'web',
        'received_at': received.isoformat(timespec='seconds') + 'Z',
        'window_seconds': payload.get('window_seconds') if isinstance(payload.get('window_seconds'), int) else None,
        'entries': len(lines), 'dropped': dropped,
    }
    if reply_id is not None:
        meta['reply_id'] = reply_id
    g.feedback_logs_count = len(lines)
    path = _feedback_activity_log_path(feedback_id) if reply_id is None else _feedback_reply_log_path(feedback_id, reply_id)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(meta, ensure_ascii=False, default=str) + '\n')
        if lines:
            handle.write('\n'.join(lines) + '\n')
    _prune_feedback_log_files()
    return path


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
        client_logs = data.get('client_logs')
        if client_logs is not None and not isinstance(client_logs, dict):
            return jsonify({'error': 'Invalid client logs'}), 400
        chat_copy = data.get('chat_copy')
        if chat_copy is not None and not isinstance(chat_copy, dict):
            return jsonify({'error': 'Invalid chat copy'}), 400
        fb = Feedback(user_id=current_user.id, title=title, message=message)
        db.session.add(fb)
        safe_db_commit()
        # ``feedback_id`` is the number older apps use for the upload of device files (still accepted).
        result = {'status': 'ok', 'feedback_id': fb.id, 'public_id': fb.public_id}
        try:
            _reset_feedback_dir(fb.public_id)
        except OSError as e:
            log_force(f"FEEDBACK-DIR-ERROR: user={current_user.id} feedback={fb.public_id} err={e}")
        if client_logs is not None:
            try:
                result['logs_saved'] = _save_feedback_client_logs(fb.public_id, current_user.id, client_logs) is not None
                result['logs_count'] = getattr(g, 'feedback_logs_count', 0)
            except Exception as e:
                log_force(f"FEEDBACK-LOGS-ERROR: user={current_user.id} feedback={fb.public_id} err={e}")
                result['logs_saved'] = False
        if chat_copy is not None:
            try:
                result['chat_copy_saved'] = _save_feedback_chat_copy(fb.public_id, current_user.id, chat_copy) is not None
            except Exception as e:
                log_force(f"FEEDBACK-CHAT-ERROR: user={current_user.id} feedback={fb.public_id} err={e}")
                result['chat_copy_saved'] = False
        client, client_version = _feedback_client(data, client_logs, chat_copy)
        try:
            _write_feedback_info(
                fb, client=client, client_version=client_version,
                server_version=app.config.get('SYSTEM_VERSION'),
                user_agent=str(request.headers.get('User-Agent') or '')[:300],
                activity_log_saved=result.get('logs_saved'), chat_copy_saved=result.get('chat_copy_saved'))
        except Exception as e:
            log_force(f"FEEDBACK-INFO-ERROR: user={current_user.id} feedback={fb.public_id} err={e}")
        return jsonify(result)

    # GET
    is_admin = bool(getattr(current_user, "is_admin", False))
    if is_admin and request.args.get('all') == '1':
        items = Feedback.query.order_by(Feedback.created_at.desc()).all()
    else:
        items = Feedback.query.filter_by(user_id=current_user.id).order_by(Feedback.created_at.desc()).all()

    for f in items:
        _ensure_feedback_public_id(f)
    replies = _feedback_replies(items)
    if db.session.new or db.session.dirty:
        safe_db_commit()
    res = []
    for f in items:
        res.append({
            'log_file': _feedback_display_path(_feedback_activity_log_path(f.public_id)) if is_admin else None,
            'chat_file': _feedback_display_path(_feedback_chat_copy_path(f.public_id)) if is_admin else None,
            'image_count': len(_feedback_image_names(f.public_id)),
            'image_dir': os.path.relpath(_feedback_images_path(f.public_id), app.root_path) if is_admin and os.path.isdir(_feedback_images_path(f.public_id)) else None,
            'id': f.id,
            'public_id': f.public_id,
            'user_id': f.user_id,
            'title': f.title or "",
            'message': f.message,
            'status': f.status,
            'admin_reply': f.admin_reply or "",
            'replies': [_feedback_reply_json(r, f.public_id, is_admin) for r in replies[f.id]],
            'handled_by': f.handled_by or "",
            'agent_name': f.agent_name or "",
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

def _feedback_agent_name():
    """The AI agent named by scripts/update_feedback_status.py, which calls this API in-process.

    It travels in a WSGI environ key that an HTTP request cannot set (headers arrive as ``HTTP_*``).
    """
    name = str(request.environ.get(FEEDBACK_AGENT_ENVIRON) or '').strip()
    return name[:40] if name else None

@app.route('/api/feedback/<fid>/update', methods=['POST'])
@login_required
def feedback_update(fid):
    if not getattr(current_user, "is_admin", False): return jsonify({'error': '403'}), 403
    fb = _feedback_by_ref(fid)
    if fb is None:
        return jsonify({'error': 'not_found'}), 404
    _ensure_feedback_public_id(fb)
    data = request.json or {}
    status = data.get('status')
    reply = data.get('admin_reply')
    handler = _feedback_agent_name() or current_user.username
    now = datetime.utcnow()
    if status:
        fb.status = status
    if reply is not None:
        # A changed reply is added to the history too (older apps and scripts/update_feedback_status.py).
        if isinstance(reply, str) and reply.strip() and reply != (fb.admin_reply or ''):
            _feedback_replies([fb])
            db.session.add(FeedbackReply(feedback_id=fb.id, user_id=current_user.id, role='admin', author=handler,
                                         message=reply.strip(), created_at=now))
        fb.admin_reply = reply
    fb.handled_by = handler
    fb.updated_at = now
    safe_db_commit()
    try:
        _write_feedback_info(fb)
    except Exception as e:
        log_force(f"FEEDBACK-INFO-ERROR: feedback={fb.public_id} err={e}")
    return jsonify({'status': 'ok'})

_FEEDBACK_SENDER_REOPENS = ('replied', 'resolved', 'resolved_unreleased', 'rejected')


@app.route('/api/feedback/<fid>/replies', methods=['POST'])
@login_required
def feedback_reply(fid):
    """Adds a reply to a feedback: the sender on their own feedback, administrators on any.

    ``client_logs`` (the last hour of the activity log) goes with it only when the person ticked the box.
    A reply by the sender puts a feedback that was replied to or closed back to ``new``."""
    fb = _feedback_by_ref(fid)
    is_admin = bool(getattr(current_user, 'is_admin', False))
    if fb is None or (fb.user_id != current_user.id and not is_admin):
        return jsonify({'error': 'not_found'}), 404
    data = request.json or {}
    message = (data.get('message') or '').replace('\x00', '').strip()
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    if len(message) > 100_000:
        return jsonify({'error': 'Message is too long'}), 400
    client_logs = data.get('client_logs')
    if client_logs is not None and not isinstance(client_logs, dict):
        return jsonify({'error': 'Invalid client logs'}), 400
    chat_copy = data.get('chat_copy')
    if chat_copy is not None and not isinstance(chat_copy, dict):
        return jsonify({'error': 'Invalid chat copy'}), 400
    if not rate_limit(f"rl:feedback_reply:user:{current_user.id}", 30, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    _ensure_feedback_public_id(fb)
    # The reply saved before the history existed stays before this one.
    _feedback_replies([fb])
    # An administrator answers as the sender on their own feedback, except on one an AI agent sent.
    as_admin = is_admin and (fb.user_id != current_user.id or bool(fb.agent_name))
    now = datetime.utcnow()
    reply = FeedbackReply(feedback_id=fb.id, user_id=current_user.id, role='admin' if as_admin else 'user',
                          author=current_user.username, message=message, created_at=now)
    db.session.add(reply)
    if as_admin:
        fb.admin_reply = message
        fb.handled_by = _feedback_agent_name() or current_user.username
    elif fb.status in _FEEDBACK_SENDER_REOPENS:
        fb.status = 'new'
    fb.updated_at = now
    safe_db_commit()
    result = {'status': 'ok', 'reply_id': reply.id, 'feedback_status': fb.status}
    if client_logs is not None:
        try:
            result['logs_saved'] = _save_feedback_client_logs(fb.public_id, current_user.id, client_logs, reply_id=reply.id) is not None
            result['logs_count'] = getattr(g, 'feedback_logs_count', 0)
        except Exception as e:
            log_force(f"FEEDBACK-LOGS-ERROR: user={current_user.id} feedback={fb.public_id} reply={reply.id} err={e}")
            result['logs_saved'] = False
        reply.activity_log_saved = result['logs_saved']
        safe_db_commit()
    if chat_copy is not None:
        try:
            result['chat_copy_saved'] = _save_feedback_chat_copy(fb.public_id, current_user.id, chat_copy, reply_id=reply.id) is not None
        except Exception as e:
            log_force(f"FEEDBACK-CHAT-ERROR: user={current_user.id} feedback={fb.public_id} reply={reply.id} err={e}")
            result['chat_copy_saved'] = False
    try:
        _write_feedback_info(fb)
    except Exception as e:
        log_force(f"FEEDBACK-INFO-ERROR: feedback={fb.public_id} err={e}")
    return jsonify(result)

@app.route('/api/feedback/<fid>', methods=['DELETE'])
@login_required
def feedback_delete(fid):
    """Deletes a feedback with its directory (information, activity log, chat copy).

    The sender deletes their own; administrators delete any, on Web and in the Android app."""
    fb = _feedback_by_ref(fid)
    is_admin = bool(getattr(current_user, 'is_admin', False))
    if fb is None or (fb.user_id != current_user.id and not is_admin):
        return jsonify({'error': 'not_found'}), 404
    _ensure_feedback_public_id(fb)
    public_id = fb.public_id
    FeedbackReply.query.filter_by(feedback_id=fb.id).delete(synchronize_session=False)
    db.session.delete(fb)
    safe_db_commit()
    _delete_feedback_files([public_id])
    return jsonify({'status': 'ok'})

