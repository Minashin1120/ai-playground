# --- Copy of the open chat attached to feedback ---
# Sent only when the user ticks the box in the feedback form. Everything the server can find
# from the chat's id goes into one JSON Lines file next to the activity logs
# (``logs/feedback-<id>-<client>-<time>.chat.jsonl``): the chat and its messages decrypted, its
# Batch jobs, latency traces, sync records, attachment cache state, the Gems it used, the Redis
# state of its answers, its temporary-chat state, and the server log lines (debug.log,
# access.log, the service journal, Android diagnostics, activity logs of other feedback) that
# name the chat or one of its answers. Attachments are decrypted into the sibling
# ``.chat.files`` directory. Clients add what only they hold: their view of the chat, storage
# keyed by the chat, and attachments kept only on an Android device (``/api/feedback/<id>/chat_files``).
_FEEDBACK_CHAT_MAX_BYTES = 256 * 1024 * 1024
_FEEDBACK_CHAT_FILES_MAX_BYTES = 1024 * 1024 * 1024
_FEEDBACK_CHAT_TOTAL_MAX_BYTES = 3 * 1024 * 1024 * 1024
_FEEDBACK_CHAT_MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
_FEEDBACK_CHAT_KEEP_FILES = 200
_FEEDBACK_CHAT_LOG_MAX_LINES = 20_000
_FEEDBACK_CHAT_LOG_LINE_MAX = 64 * 1024
_FEEDBACK_CHAT_JOURNAL_TIMEOUT = 30
_FEEDBACK_CHAT_REDIS_VALUE_MAX = 8 * 1024 * 1024
_FEEDBACK_CHAT_DEVICE_FILE_WINDOW = 3600
_FEEDBACK_CHAT_DEVICE_FILES_MAX = 500
_FEEDBACK_CHAT_JOURNAL_UNITS = ('ai-chat.service', 'ai-chat-worker@*.service')
_FEEDBACK_CHAT_NAME_RE = re.compile(r'^feedback-(\d+)-[a-z]{1,16}-\d{8}T\d{6}Z\.chat\.jsonl$')
_FEEDBACK_CHAT_FILES_RE = re.compile(r'^feedback-(\d+)-[a-z]{1,16}-\d{8}T\d{6}Z\.chat\.files$')
_FEEDBACK_CHAT_FILE_URL_RE = re.compile(r'/files/(?:thumb/)?([^\s"\'()<>?#\\]+)')


def _feedback_chat_file_index():
    """Feedback id -> name of its chat copy file (administrators' feedback list)."""
    index = {}
    try:
        names = os.listdir(_feedback_logs_dir())
    except OSError:
        return index
    for name in names:
        match = _FEEDBACK_CHAT_NAME_RE.match(name)
        if match:
            index[int(match.group(1))] = name
    return index


def _feedback_chat_files_dir(path):
    return path[:-len('.jsonl')] + '.files'


def _feedback_chat_tree_bytes(path):
    total = 0
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _remove_feedback_chat_copy(path):
    try:
        os.remove(path)
    except OSError:
        pass
    shutil.rmtree(_feedback_chat_files_dir(path), ignore_errors=True)


def _prune_feedback_chat_files(directory, keep=None):
    """Keeps the newest copies within the count and total size limits (never ``keep``)."""
    try:
        paths = [os.path.join(directory, n) for n in os.listdir(directory) if _FEEDBACK_CHAT_NAME_RE.match(n)]
    except OSError:
        return
    entries = []
    for path in paths:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        size = _feedback_chat_tree_bytes(path) + _feedback_chat_tree_bytes(_feedback_chat_files_dir(path))
        entries.append((mtime, path, size))
    entries.sort()
    total = sum(size for _mtime, _path, size in entries)
    count = len(entries)
    for _mtime, path, size in entries:
        if count <= _FEEDBACK_CHAT_KEEP_FILES and total <= _FEEDBACK_CHAT_TOTAL_MAX_BYTES:
            break
        if path == keep:
            continue
        _remove_feedback_chat_copy(path)
        count -= 1
        total -= size


def _delete_feedback_files(feedback_ids):
    """Removes the activity logs and chat copies of ``feedback_ids`` (account deletion)."""
    wanted = {int(fid) for fid in feedback_ids}
    if not wanted:
        return
    directory = _feedback_logs_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        match = (_FEEDBACK_LOG_NAME_RE.match(name) or _FEEDBACK_CHAT_NAME_RE.match(name)
                 or _FEEDBACK_CHAT_FILES_RE.match(name))
        if not match or int(match.group(1)) not in wanted:
            continue
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _feedback_chat_value(value):
    if isinstance(value, datetime):
        return value.isoformat() + 'Z'
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _feedback_chat_row(row, kind, decrypted=()):
    data = {'kind': kind}
    for column in row.__table__.columns:
        value = getattr(row, column.key)
        if column.key in decrypted and value:
            value = decrypt_val(value)
        data[column.key] = _feedback_chat_value(value)
    return data


def _feedback_chat_file_budget(files_dir, used, size):
    """None when ``size`` more bytes may be written to this copy, else why not."""
    if used + size > _FEEDBACK_CHAT_FILES_MAX_BYTES:
        return 'copy_limit'
    try:
        free = shutil.disk_usage(os.path.dirname(files_dir)).free
    except OSError:
        return 'disk_unknown'
    if free - size < _FEEDBACK_CHAT_MIN_FREE_BYTES:
        return 'disk_space'
    return None


def _feedback_chat_file_name(files_dir, name):
    try:
        index = len(os.listdir(files_dir)) + 1
    except OSError:
        index = 1
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.basename(str(name or 'file'))).lstrip('.')[-120:] or 'file'
    return f"{index:04d}-{safe}"


def _feedback_chat_write_stream(files_dir, name, chunks, used):
    """Writes ``chunks`` to a new file in ``files_dir``; returns (stored name, bytes, sha256) or the reason it stopped."""
    os.makedirs(files_dir, mode=0o700, exist_ok=True)
    stored = _feedback_chat_file_name(files_dir, name)
    target = os.path.join(files_dir, stored)
    digest = hashlib.sha256()
    written = 0
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                reason = _feedback_chat_file_budget(files_dir, used, written)
                if reason:
                    raise _FeedbackChatFileStop(reason)
                digest.update(chunk)
                handle.write(chunk)
    except BaseException:
        try:
            os.remove(target)
        except OSError:
            pass
        raise
    return stored, written, digest.hexdigest()


class _FeedbackChatFileStop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _feedback_chat_read_chunks(path, size=1024 * 1024):
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(size)
            if not chunk:
                return
            yield chunk


def _feedback_chat_server_file(files_dir, used, rel_path):
    """Decrypted copy of one of the user's uploads; returns (row, bytes written)."""
    row = {'kind': 'file', 'source': 'server', 'ref': rel_path,
           'mime': mimetypes.guess_type(rel_path)[0] or 'application/octet-stream'}
    info = _get_file_disk_info(rel_path)
    if not info.get('exists'):
        row['status'] = 'missing'
        return row, 0
    row.update(encrypted_at_rest=bool(info.get('is_encrypted')), stored_bytes=info.get('size'))
    reason = _feedback_chat_file_budget(files_dir, used, int(info.get('size') or 0))
    if reason:
        row['status'] = 'skipped_' + reason
        return row, 0
    try:
        if info.get('is_encrypted'):
            with open(info['disk_path'], 'rb') as handle:
                plain = decrypt_bytes(handle.read())
            if plain is None:
                row['status'] = 'decrypt_failed'
                return row, 0
            chunks = [plain]
        else:
            chunks = _feedback_chat_read_chunks(info['disk_path'])
        stored, written, sha = _feedback_chat_write_stream(files_dir, rel_path, chunks, used)
    except _FeedbackChatFileStop as stop:
        row['status'] = 'skipped_' + stop.reason
        return row, 0
    except Exception as e:
        row.update(status='error', error=f"{type(e).__name__}: {e}"[:500])
        return row, 0
    row.update(status='copied', file=stored, bytes=written, sha256=sha)
    return row, written


def _feedback_chat_refs(user_id, messages, extra_texts):
    """Upload references of the chat: message attachments and /files/ links in the text."""
    refs = []
    seen = set()

    def add(raw):
        rel = _resolve_user_upload_rel_path(raw, user_id)
        if rel and rel not in seen:
            seen.add(rel)
            refs.append(rel)

    for message in messages:
        for item in _iter_message_attachment_refs(message.get('image_url')):
            add(unquote(str(_normalize_upload_ref(item) or '')))
    for text in extra_texts:
        if isinstance(text, str):
            for match in _FEEDBACK_CHAT_FILE_URL_RE.finditer(text):
                add(unquote(match.group(1)))
    return refs


def _feedback_chat_redis_value(key, kind):
    def text(value):
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        return value[:_FEEDBACK_CHAT_REDIS_VALUE_MAX] if isinstance(value, str) else value
    if kind == 'string':
        return text(redis_conn.get(key))
    if kind == 'list':
        return [text(v) for v in redis_conn.lrange(key, 0, 9999)]
    if kind == 'hash':
        return {text(k): text(v) for k, v in redis_conn.hgetall(key).items()}
    if kind == 'set':
        return sorted(text(v) for v in redis_conn.smembers(key))
    if kind == 'zset':
        return [[text(v), score] for v, score in redis_conn.zrange(key, 0, 9999, withscores=True)]
    if kind == 'stream':
        return [[text(i), {text(k): text(v) for k, v in fields.items()}] for i, fields in redis_conn.xrange(key, count=10000)]
    return None


def _feedback_chat_redis_rows(exact_keys, substrings):
    try:
        keys = list(redis_conn.scan_iter(count=1000))
    except Exception as e:
        yield {'kind': 'redis_error', 'error': f"{type(e).__name__}: {e}"[:500]}
        return
    for key in keys:
        name = key.decode('utf-8', 'replace') if isinstance(key, bytes) else str(key)
        if name not in exact_keys and not any(s in name for s in substrings):
            continue
        try:
            kind = redis_conn.type(key)
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            yield {'kind': 'redis', 'key': name, 'type': kind, 'ttl': redis_conn.ttl(key),
                   'value': _feedback_chat_redis_value(key, kind)}
        except Exception as e:
            yield {'kind': 'redis', 'key': name, 'error': f"{type(e).__name__}: {e}"[:500]}


def _feedback_chat_log_pattern(thread, job_ids):
    """Log lines that name the chat (public or database id) or one of its answers (job id)."""
    alts = [re.escape(thread.public_id)] if thread.public_id else []
    alts += [re.escape(job) for job in sorted(job_ids) if len(job) >= 8]
    alts.append(rf"thread(?:[ _-]?id)?[\s=:\"'#]*{int(thread.id)}(?![0-9])")
    return '|'.join(alts)


def _feedback_chat_log_root():
    return app.config.get('FEEDBACK_CHAT_LOG_ROOT') or app.root_path


def _feedback_chat_log_files(user_id):
    root = _feedback_chat_log_root()
    paths = sorted(glob.glob(os.path.join(root, 'debug.log*'))) + sorted(glob.glob(os.path.join(root, 'access.log*')))
    logs_dir = _feedback_logs_dir()
    try:
        names = sorted(os.listdir(logs_dir))
    except OSError:
        names = []
    # Activity logs record the sender's own operations, so only this user's feedback can name the chat.
    own = {fid for (fid,) in db.session.query(Feedback.id).filter_by(user_id=user_id).all()}
    for n in names:
        match = _FEEDBACK_LOG_NAME_RE.match(n)
        if n.startswith('android-diagnostics.log') or (match and int(match.group(1)) in own):
            paths.append(os.path.join(logs_dir, n))
    return [p for p in paths if os.path.isfile(p) and not p.endswith('.gz')]


def _feedback_chat_log_rows(pattern, user_id):
    regex = re.compile(pattern, re.IGNORECASE)
    root = _feedback_chat_log_root()
    for path in _feedback_chat_log_files(user_id):
        source = os.path.relpath(path, root)
        found = 0
        try:
            with open(path, encoding='utf-8', errors='replace') as handle:
                for number, line in enumerate(handle, 1):
                    if not regex.search(line):
                        continue
                    found += 1
                    if found > _FEEDBACK_CHAT_LOG_MAX_LINES:
                        continue
                    yield {'kind': 'log', 'source': source, 'line_no': number,
                           'line': line.rstrip('\n')[:_FEEDBACK_CHAT_LOG_LINE_MAX]}
        except OSError as e:
            yield {'kind': 'log_error', 'source': source, 'error': str(e)[:500]}
            continue
        if found > _FEEDBACK_CHAT_LOG_MAX_LINES:
            yield {'kind': 'log_truncated', 'source': source, 'matched': found, 'kept': _FEEDBACK_CHAT_LOG_MAX_LINES}


def _feedback_chat_journal_rows(pattern, since):
    command = ['journalctl', '--no-pager', '-o', 'short-iso-precise', '--case-sensitive=false', '--grep', pattern]
    for unit in _FEEDBACK_CHAT_JOURNAL_UNITS:
        command += ['-u', unit]
    if since:
        command += ['--since', since.strftime('%Y-%m-%d %H:%M:%S') + ' UTC']
    try:
        done = subprocess.run(command, capture_output=True, timeout=_FEEDBACK_CHAT_JOURNAL_TIMEOUT)
    except Exception as e:
        yield {'kind': 'log_error', 'source': 'journal', 'error': f"{type(e).__name__}: {e}"[:500]}
        return
    lines = done.stdout.decode('utf-8', 'replace').splitlines()
    if done.returncode not in (0, 1):
        yield {'kind': 'log_error', 'source': 'journal', 'returncode': done.returncode,
               'error': done.stderr.decode('utf-8', 'replace')[-500:]}
    for line in lines[-_FEEDBACK_CHAT_LOG_MAX_LINES:]:
        if line.startswith('-- '):
            continue
        yield {'kind': 'log', 'source': 'journal', 'line': line[:_FEEDBACK_CHAT_LOG_LINE_MAX]}
    if len(lines) > _FEEDBACK_CHAT_LOG_MAX_LINES:
        yield {'kind': 'log_truncated', 'source': 'journal', 'matched': len(lines), 'kept': _FEEDBACK_CHAT_LOG_MAX_LINES}


def _feedback_chat_server_rows(thread, user_id, files_dir, totals):
    """Every server record of ``thread`` (see the module comment); ``totals`` counts the copied files."""
    yield _feedback_chat_row(thread, 'thread')
    messages = []
    query = Message.query.filter_by(thread_id=thread.id).order_by(Message.timestamp, Message.id)
    for message in query.yield_per(200):
        decrypted = ('content', 'thought_data') if message.is_encrypted else ()
        messages.append(_feedback_chat_row(message, 'message', decrypted))
    # Answers show files saved by an earlier turn's code execution under rewritten links.
    displayed = [{'id': m['id'], 'role': m['role'], 'content': m['content']} for m in messages]
    _apply_thread_sandbox_image_refs(thread.id, displayed)
    for row, shown in zip(messages, displayed):
        if shown['content'] != row['content']:
            row['content_displayed'] = shown['content']
        yield row

    job_ids = set()
    for job in GeminiBatchJob.query.filter_by(thread_id=thread.id).order_by(GeminiBatchJob.id):
        job_ids.add(job.job_id)
        yield _feedback_chat_row(job, 'batch_job')
    public_id = thread.public_id or str(thread.id)
    for model, kind in ((FirstTokenLatencyMetric, 'latency_first_token'), (ChatLatencyTrace, 'latency_trace')):
        for row in model.query.filter_by(user_id=user_id, thread_public_id=public_id).order_by(model.id):
            if row.job_id:
                job_ids.add(row.job_id)
            yield _feedback_chat_row(row, kind)
    for row in SyncThreadState.query.filter_by(thread_id=thread.id):
        yield _feedback_chat_row(row, 'sync_thread_state')
    for row in SyncMessageRef.query.filter_by(thread_id=thread.id).order_by(SyncMessageRef.message_id):
        yield _feedback_chat_row(row, 'sync_message_ref')
    for row in SyncTombstone.query.filter_by(user_id=user_id, public_id=public_id):
        yield _feedback_chat_row(row, 'sync_tombstone')

    gem_uuids = {m.get('gem_uuid') for m in messages if m.get('gem_uuid')}
    if thread.last_gem_uuid:
        gem_uuids.add(thread.last_gem_uuid)
    if gem_uuids:
        for gem in Gem.query.filter(Gem.user_id == user_id, Gem.uuid.in_(sorted(gem_uuids))):
            yield _feedback_chat_row(gem, 'gem')

    user = db.session.get(User, user_id)
    try:
        yield {'kind': 'temp_chat_state', **_get_temp_chat_runtime_meta(thread, user=user)}
    except Exception as e:
        yield {'kind': 'temp_chat_state', 'error': f"{type(e).__name__}: {e}"[:500]}

    pending_key = f"pending_job:{user_id}:{thread.id}"
    try:
        pending_raw = redis_conn.get(pending_key)
        if pending_raw:
            pending = json.loads(pending_raw)
            if isinstance(pending, dict) and pending.get('job_id'):
                job_ids.add(str(pending['job_id']))
    except Exception:
        pass
    yield from _feedback_chat_redis_rows({pending_key}, [public_id] + sorted(j for j in job_ids if len(j) >= 8))

    refs = _feedback_chat_refs(user_id, messages, [
        text for m in messages for text in (m.get('content'), m.get('content_displayed'), m.get('thought_data'))])
    if refs:
        for row in FileCache.query.filter(FileCache.user_id == user_id, FileCache.rel_path.in_(refs)).order_by(FileCache.id):
            yield _feedback_chat_row(row, 'file_cache')
    for rel_path in refs:
        row, written = _feedback_chat_server_file(files_dir, totals['file_bytes'], rel_path)
        totals['file_bytes'] += written
        yield row

    pattern = _feedback_chat_log_pattern(thread, job_ids)
    yield {'kind': 'log_search', 'pattern': pattern}
    yield from _feedback_chat_log_rows(pattern, user_id)
    stamps = [m['timestamp'] for m in messages if m.get('timestamp')]
    since = None
    if stamps:
        since = datetime.fromisoformat(min(stamps).rstrip('Z')) - timedelta(minutes=10)
    yield from _feedback_chat_journal_rows(pattern, since or datetime.utcnow() - timedelta(days=30))


def _feedback_chat_client_rows(payload):
    """What only the client holds: its view of the chat, storage keyed by it, its own logs of it."""
    client_thread = payload.get('thread')
    if isinstance(client_thread, dict) and isinstance(client_thread.get('messages'), list):
        row = {key: value for key, value in client_thread.items() if key != 'messages'}
        row['kind'] = 'client_thread'
        yield row
        for message in client_thread['messages']:
            if isinstance(message, dict):
                yield dict(message, kind='client_message')
    for field, kind in (('client_state', 'client_state'), ('offline_cache', 'client_offline_cache')):
        value = payload.get(field)
        if isinstance(value, dict) and value:
            yield {'kind': kind, 'value': value}
    entries = payload.get('diagnostics')
    if isinstance(entries, list):
        for entry in entries[:_FEEDBACK_CHAT_LOG_MAX_LINES]:
            if isinstance(entry, dict):
                yield {'kind': 'client_diagnostics', 'entry': entry}


def _save_feedback_chat_copy(feedback_id, user_id, payload):
    """Writes the copy of the chat in ``payload``; returns the file name (None when there is no chat)."""
    client = str(payload.get('client') or 'unknown').lower()
    if not re.fullmatch(r'[a-z]{1,16}', client):
        client = 'unknown'
    thread = resolve_thread_for_user(payload.get('thread_id'), user_id)
    client_thread = payload.get('thread')
    has_client_thread = isinstance(client_thread, dict) and isinstance(client_thread.get('messages'), list)
    if not thread and not has_client_thread:
        return None
    received = datetime.utcnow()
    directory = _feedback_logs_dir()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    name = f"feedback-{int(feedback_id)}-{client}-{received.strftime('%Y%m%dT%H%M%SZ')}.chat.jsonl"
    path = os.path.join(directory, name)
    files_dir = _feedback_chat_files_dir(path)
    # Rows go to a side file first so a long chat is never held in memory; the meta line needs the counts.
    body_path = path + '.part'
    totals = {'file_bytes': 0}
    counts = {}
    size_total = 0
    dropped = 0
    fd = os.open(body_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as body:
            rows = itertools.chain(
                _feedback_chat_server_rows(thread, user_id, files_dir, totals) if thread else (),
                _feedback_chat_client_rows(payload))
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, default=str, separators=(',', ':')) + '\n'
                size = len(line.encode('utf-8'))
                if size_total + size > _FEEDBACK_CHAT_MAX_BYTES:
                    dropped += 1
                    continue
                size_total += size
                kind = str(row.get('kind'))
                counts[kind] = counts.get(kind, 0) + 1
                body.write(line)
        meta = {
            'kind': 'meta', 'feedback_id': feedback_id, 'user_id': user_id, 'client': client,
            'client_version': str(payload.get('version') or '')[:64],
            'server_version': app.config.get('SYSTEM_VERSION'),
            'source': 'server' if thread else 'device',
            'thread_id': (thread.public_id or str(thread.id)) if thread else str(payload.get('thread_id') or '')[:64],
            'received_at': received.isoformat(timespec='seconds') + 'Z',
            'rows': counts, 'files_bytes': totals['file_bytes'], 'dropped': dropped,
            'device_files_expected': payload.get('device_files') if isinstance(payload.get('device_files'), int) else 0,
        }
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle, open(body_path, encoding='utf-8') as body:
            handle.write(json.dumps(meta, ensure_ascii=False, default=str) + '\n')
            shutil.copyfileobj(body, handle)
    except BaseException:
        shutil.rmtree(files_dir, ignore_errors=True)
        raise
    finally:
        try:
            os.remove(body_path)
        except OSError:
            pass
    _prune_feedback_chat_files(directory, keep=path)
    return name


@app.route('/api/feedback/<int:fid>/chat_files', methods=['POST'])
@login_required
def feedback_chat_file(fid):
    """An attachment kept only on the device, added to the chat copy of the user's feedback ``fid``."""
    fb = Feedback.query.filter_by(id=fid, user_id=current_user.id).first()
    name = _feedback_chat_file_index().get(fid) if fb else None
    if not name:
        return jsonify({'error': 'not_found'}), 404
    if fb.created_at and (datetime.utcnow() - fb.created_at).total_seconds() > _FEEDBACK_CHAT_DEVICE_FILE_WINDOW:
        return jsonify({'error': 'expired'}), 409
    upload = request.files.get('file')
    if upload is None:
        return jsonify({'error': 'file_required'}), 400
    path = os.path.join(_feedback_logs_dir(), name)
    files_dir = _feedback_chat_files_dir(path)
    try:
        existing = os.listdir(files_dir)
    except OSError:
        existing = []
    if len(existing) >= _FEEDBACK_CHAT_DEVICE_FILES_MAX:
        return jsonify({'error': 'too_many_files'}), 429
    row = {'kind': 'file', 'source': 'device', 'ref': str(request.form.get('ref') or '')[:200],
           'name': str(upload.filename or '')[:200], 'mime': str(upload.mimetype or '')[:128]}
    try:
        stored, written, sha = _feedback_chat_write_stream(
            files_dir, upload.filename or row['ref'],
            iter(lambda: upload.stream.read(1024 * 1024), b''), _feedback_chat_tree_bytes(files_dir))
        row.update(status='copied', file=stored, bytes=written, sha256=sha)
    except _FeedbackChatFileStop as stop:
        row['status'] = 'skipped_' + stop.reason
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    _prune_feedback_chat_files(_feedback_logs_dir(), keep=path)
    return jsonify({'saved': row['status'] == 'copied', 'status': row['status']})
