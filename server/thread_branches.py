"""Chat snapshots, origin metadata and reference-aware attachment cleanup."""

_BRANCH_WARNING = '分岐先は通常のチャットとして保存され、自動削除されません。コピー対象の添付・生成ファイルも保存対象になります。'
_FILE_LINK_RE = re.compile(r'''/files/(?!thumb/)([^\s<>"'\)\]\}]+)''')


def _lock_chat_files(user_id):
    # All branch/delete paths acquire this row before reading chat state. The
    # transaction owns the lock until commit (also works across web workers).
    db.session.query(User.id).filter(User.id == user_id).with_for_update().one()


def _message_file_refs(message, user_id):
    refs = set()
    raw = message.get('image_url') if isinstance(message, dict) else message.image_url
    for value in _iter_message_attachment_refs(raw):
        ref = _resolve_user_upload_rel_path(value, user_id)
        if ref and ref.startswith(f'{user_id}/') and '..' not in ref.split('/'):
            refs.add(ref)
    content = message.get('content') if isinstance(message, dict) else message.content
    encrypted = message.get('is_encrypted', False) if isinstance(message, dict) else message.is_encrypted
    if encrypted and content:
        if not cipher:
            raise ValueError('暗号化された会話のファイル参照を確認できません')
        try:
            content = _ring_decrypt_bytes(content.encode()).decode()
        except Exception:
            # No key in the ring opens it (e.g. a key lost before rotation was
            # kept), so it can never be displayed or link a file. Count only
            # the plain attachment column instead of failing every delete.
            if str(content).startswith('gAAAA'):
                return refs
    for value in _FILE_LINK_RE.findall(content or ''):
        ref = _resolve_user_upload_rel_path(value, user_id)
        if ref and ref.startswith(f'{user_id}/') and '..' not in ref.split('/'):
            refs.add(ref)
    return refs


def _missing_message_files(message, user_id):
    return sorted(ref for ref in _message_file_refs(message, user_id)
                  if not _get_file_disk_info(ref).get('exists'))


def _delete_unreferenced_upload_refs(user_id, refs, exclude_threads=(), exclude_messages=()):
    candidates = {ref for value in refs if (ref := _normalize_upload_ref(value))
                  and ref.startswith(f'{user_id}/')}
    if not candidates:
        return
    # Scan once per cleanup, with bounded payload reads. Include generated file
    # links and encrypted answers, not only image_url attachments.
    cursor = 0
    while candidates:
        query = db.session.query(Message.id, Message.image_url, Message.content, Message.is_encrypted).join(
            Thread, Thread.id == Message.thread_id).filter(Thread.user_id == user_id, Message.id > cursor)
        if exclude_threads:
            query = query.filter(~Message.thread_id.in_(exclude_threads))
        if exclude_messages:
            query = query.filter(~Message.id.in_(exclude_messages))
        rows = query.order_by(Message.id).limit(100).with_for_update().all()
        if not rows:
            break
        for row in rows:
            candidates.difference_update(_message_file_refs(row, user_id))
        cursor = rows[-1].id
    for ref in candidates:
        _delete_file_cache_for_path(user_id, ref)
    db.session.info.setdefault('branch_file_cleanup', set()).update((user_id, ref) for ref in candidates)


def _branch_files_after_commit(session):
    for user_id, ref in session.info.pop('branch_file_cleanup', set()):
        _delete_user_upload_ref(user_id, ref)


def _branch_files_after_rollback(session):
    session.info.pop('branch_file_cleanup', None)


_sa_event.listen(_SyncOrmSession, 'after_commit', _branch_files_after_commit)
_sa_event.listen(_SyncOrmSession, 'after_rollback', _branch_files_after_rollback)


def _branch_origin(thread):
    record = db.session.get(ThreadBranch, thread.id)
    if not record:
        return None
    origin = json.loads(record.origin)
    source = resolve_thread_for_user(str(origin.get('thread_id') or ''), thread.user_id)
    if not source and origin.get('thread_client_uuid'):
        state = SyncThreadState.query.filter_by(user_id=thread.user_id, client_uuid=origin['thread_client_uuid']).first()
        source = db.session.get(Thread, state.thread_id) if state else None
    mid = origin.get('message_id')
    if source and not mid and origin.get('message_client_uuid'):
        ref = SyncMessageRef.query.filter_by(user_id=thread.user_id, client_uuid=origin['message_client_uuid']).first()
        mid = ref.message_id if ref and ref.thread_id == source.id else None
    message = db.session.get(Message, mid) if isinstance(mid, int) else None
    available = bool(source and message and message.thread_id == source.id)
    return {**origin, 'thread_id': (source.public_id or str(source.id)) if source else None,
            'message_id': mid, 'available': available}


def _store_branch_origin(thread, origin):
    if not isinstance(origin, dict):
        return
    # Source identifiers are opaque; they are resolved with ownership checks on
    # every read. Never accept a title/URL from the client for rendering.
    clean = {key: str(origin[key])[:80] for key in ('thread_id', 'thread_client_uuid', 'message_client_uuid')
             if origin.get(key)}
    mid = origin.get('message_id')
    if isinstance(mid, int) and not isinstance(mid, bool) and mid > 0:
        clean['message_id'] = mid
    if not clean:
        return
    row = db.session.get(ThreadBranch, thread.id)
    if not row:
        row = ThreadBranch(thread_id=thread.id, user_id=thread.user_id)
        db.session.add(row)
    row.origin = json.dumps(clean)


@app.route('/api/threads/<thread_id>/branch', methods=['POST'])
@login_required
def branch_thread(thread_id):
    _lock_chat_files(current_user.id)
    source = resolve_thread_for_user(thread_id, current_user.id)
    if source:
        source = Thread.query.filter_by(id=source.id, user_id=current_user.id).with_for_update().populate_existing().first()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': '分岐リクエストが不正です'}), 400
    key = str(data.get('request_id') or '')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{16,80}', key):
        return jsonify({'error': '分岐リクエストが不正です'}), 400
    previous = ThreadBranch.query.filter_by(user_id=current_user.id, request_id=key).with_for_update().first()
    if previous:
        dest = db.session.get(Thread, previous.thread_id)
        origin = json.loads(previous.origin)
        if origin.get('thread_id') not in {str(thread_id), (source.public_id or str(source.id)) if source else None} or origin.get('message_id') != data.get('message_id'):
            return jsonify({'error': '分岐リクエストが一致しません'}), 409
        return jsonify({'id': dest.public_id, 'title': dest.title, 'leaf_id': previous.selected_message_id})
    if not source or str(source.title or '').startswith('[LIBRARY]'):
        return jsonify({'error': 'チャットがありません'}), 403
    mid = data.get('message_id')
    if not isinstance(mid, int) or isinstance(mid, bool):
        return jsonify({'error': 'メッセージが不正です'}), 400
    selected = Message.query.filter_by(id=mid, thread_id=source.id).with_for_update().first()
    if not selected or selected.thread_id != source.id:
        return jsonify({'error': 'メッセージがありません'}), 400
    if source.is_temporary and data.get('confirm_persist_temporary') is not True:
        return jsonify({'error': _BRANCH_WARNING, 'code': 'confirmation_required'}), 409
    cutoff = selected.timestamp or datetime.min
    metadata = db.session.query(Message.id, Message.parent_id, Message.timestamp).filter_by(thread_id=source.id).with_for_update().all()
    included = {row.id: row for row in metadata if ((row.timestamp or datetime.min), row.id) <= (cutoff, mid)}
    ordered, visited, visiting = [], set(), set()
    def visit(message_id):
        if message_id in visited:
            return
        if message_id in visiting or message_id not in included:
            raise ValueError('invalid tree')
        visiting.add(message_id)
        parent = included[message_id].parent_id
        if parent is not None:
            visit(parent)
        visiting.remove(message_id)
        visited.add(message_id)
        ordered.append(message_id)
    try:
        # Iteration is chronological in normal data, so recursion only checks
        # the immediate parent. Malformed chains are rejected atomically.
        for row in sorted(included.values(), key=lambda r: ((r.timestamp or datetime.min), r.id)):
            visit(row.id)
    except (ValueError, RecursionError):
        return jsonify({'error': '会話の親子関係が不正です'}), 409
    active = GeminiBatchJob.query.filter(GeminiBatchJob.thread_id == source.id,
        GeminiBatchJob.assistant_message_id.in_(ordered),
        ~GeminiBatchJob.state.in_(BATCH_TERMINAL_STATES)).with_for_update().first()
    if active or redis_conn.get(f'pending_job:{current_user.id}:{source.id}'):
        return jsonify({'error': '回答の生成が完了してから分岐してください'}), 409
    dest = Thread(user_id=current_user.id, public_id=generate_thread_public_id(),
        title=(str(source.title or 'New Chat')[:196] + '（分岐）'), is_temporary=False,
        custom_instruction=source.custom_instruction, include_global_instruction=source.include_global_instruction,
        last_model=source.last_model, last_gem_uuid=source.last_gem_uuid,
        enable_prompt_caching=source.enable_prompt_caching, prompt_cache_nonce=generate_thread_public_id()[:32])
    db.session.add(dest)
    db.session.flush()
    mapped = {}
    columns = [c.name for c in Message.__table__.columns if c.name not in ('id', 'thread_id', 'parent_id')]
    # Fetch payloads in bounded batches; retain only the ID mapping in memory.
    for offset in range(0, len(ordered), 100):
        batch_ids = ordered[offset:offset + 100]
        rows = {row.id: row for row in Message.query.filter(Message.id.in_(batch_ids)).with_for_update().all()}
        for old_id in batch_ids:
            old = rows[old_id]
            clone = Message(thread_id=dest.id, parent_id=mapped.get(old.parent_id),
                            **{name: getattr(old, name) for name in columns})
            db.session.add(clone)
            db.session.flush()
            mapped[old_id] = clone.id
    record = ThreadBranch(thread_id=dest.id, user_id=current_user.id, request_id=key,
        selected_message_id=mapped[mid], origin=json.dumps({'thread_id': source.public_id or str(source.id), 'message_id': mid}))
    db.session.add(record)
    safe_db_commit()
    return jsonify({'id': dest.public_id, 'title': dest.title, 'leaf_id': mapped[mid]}), 201
