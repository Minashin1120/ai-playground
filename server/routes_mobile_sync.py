"""Android serverless mode: chat sync and API key export.

The app answers chats on the device and keeps them in its own store; these endpoints
let it upload those chats (idempotently, by per-record UUIDs) and learn which server
chats changed or were deleted since the last sync.  Message bodies of changed chats
are read with the existing ``GET /api/threads/<id>``.
"""

_SYNC_PAGE_SIZE = 200
_SYNC_PUSH_MAX_MESSAGES = 100
_SYNC_PUSH_MAX_BYTES = 8 * 1024 * 1024
_SYNC_TOMBSTONE_DAYS = 180
_SYNC_UUID_RE = re.compile(r'^[0-9a-fA-F-]{16,36}$')
_SYNC_SECRET_FIELDS = (
    ('openai_key', 'openai_api_key'), ('gemini_key', 'gemini_api_key'),
    ('anthropic_key', 'anthropic_api_key'), ('deepseek_key', 'deepseek_api_key'),
    ('kimi_key', 'kimi_api_key'), ('mistral_key', 'mistral_api_key'),
    ('ideogram_key', 'ideogram_api_key'),
    ('zai_key', 'zai_api_key'),
    ('xai_key', 'xai_api_key'), ('google_key', 'google_api_key'),
)


def _sync_ms(value):
    """Epoch milliseconds from an int/str, or None."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _sync_from_ms(value):
    return datetime.utcfromtimestamp(value / 1000.0)


def _sync_to_ms(value):
    if not value:
        return None
    return int((value - datetime(1970, 1, 1)).total_seconds() * 1000)


def _sync_uuid(value):
    text_value = str(value or '').strip()
    return text_value if _SYNC_UUID_RE.fullmatch(text_value) else None


def _sync_thread_row(thread, state):
    return {
        'id': thread.public_id or str(thread.id),
        'client_uuid': state.client_uuid if state else None,
        'title': thread.title,
        'is_bookmarked': bool(thread.is_bookmarked),
        'custom_instruction': thread.custom_instruction or '',
        'include_global_instruction': thread.include_global_instruction if thread.include_global_instruction is not None else True,
        'last_model': thread.last_model,
        'last_gem_uuid': thread.last_gem_uuid,
        'updated_at_ms': _sync_to_ms(thread.updated_at),
        'changed_at_ms': _sync_to_ms(state.changed_at) if state else _sync_to_ms(thread.updated_at),
    }


def _sync_thread_visible(thread):
    return bool(thread) and not thread.is_temporary and not str(thread.title or '').startswith('[LIBRARY]')


@app.route('/api/mobile/v1/sync/changes', methods=['GET'])
def mobile_sync_changes():
    """Changed chats since ``since`` (epoch ms), or every chat when it is absent, plus deletions."""
    if not rate_limit(f'rl:mobile:sync:changes:{current_user.id}', 120, 60):
        return _mobile_error('rate_limit', 429)
    since = _sync_ms(request.args.get('since'))
    cursor = request.args.get('cursor', type=int) or 0
    server_time = datetime.utcnow()
    rows = []
    has_more = False
    next_cursor = None
    if since is None:
        # First sync: every chat, paged by id.
        threads = (Thread.query.filter(Thread.user_id == current_user.id, Thread.id > cursor)
                   .order_by(Thread.id).limit(_SYNC_PAGE_SIZE + 1).all())
        has_more = len(threads) > _SYNC_PAGE_SIZE
        threads = threads[:_SYNC_PAGE_SIZE]
        states = {s.thread_id: s for s in SyncThreadState.query.filter(
            SyncThreadState.thread_id.in_([t.id for t in threads] or [0])).all()}
        rows = [_sync_thread_row(t, states.get(t.id)) for t in threads if _sync_thread_visible(t)]
        next_cursor = threads[-1].id if has_more and threads else None
        tombstones = []
    else:
        since_at = _sync_from_ms(since)
        query = (db.session.query(SyncThreadState, Thread)
                 .join(Thread, Thread.id == SyncThreadState.thread_id)
                 .filter(SyncThreadState.user_id == current_user.id, SyncThreadState.changed_at >= since_at,
                         SyncThreadState.thread_id > cursor)
                 .order_by(SyncThreadState.thread_id).limit(_SYNC_PAGE_SIZE + 1))
        pairs = query.all()
        has_more = len(pairs) > _SYNC_PAGE_SIZE
        pairs = pairs[:_SYNC_PAGE_SIZE]
        rows = [_sync_thread_row(t, s) for s, t in pairs if _sync_thread_visible(t)]
        next_cursor = pairs[-1][0].thread_id if has_more and pairs else None
        tombstones = [{'id': t.public_id, 'client_uuid': t.client_uuid, 'deleted_at_ms': _sync_to_ms(t.deleted_at)}
                      for t in SyncTombstone.query.filter(SyncTombstone.user_id == current_user.id,
                                                          SyncTombstone.deleted_at >= since_at)
                      .order_by(SyncTombstone.id).limit(2000).all()]
    try:
        cutoff = server_time - timedelta(days=_SYNC_TOMBSTONE_DAYS)
        SyncTombstone.query.filter(SyncTombstone.user_id == current_user.id,
                                   SyncTombstone.deleted_at < cutoff).delete(synchronize_session=False)
        safe_db_commit()
    except Exception:
        db.session.rollback()
    return jsonify({'status': 'ok', 'threads': rows, 'tombstones': tombstones, 'has_more': has_more,
                    'next_cursor': next_cursor, 'server_time_ms': _sync_to_ms(server_time)})


def _sync_resolve_thread(entry):
    public_id = str(entry.get('id') or '').strip()
    client_uuid = _sync_uuid(entry.get('client_uuid'))
    thread = resolve_thread_for_user(public_id, current_user.id) if public_id else None
    state = None
    if thread is None and client_uuid:
        state = SyncThreadState.query.filter_by(user_id=current_user.id, client_uuid=client_uuid).first()
        if state:
            thread = db.session.get(Thread, state.thread_id)
    if thread is None and public_id:
        return None, None, 'thread_not_found'
    if thread is None:
        thread = Thread(user_id=current_user.id, public_id=generate_thread_public_id(),
                        title=_normalize_thread_title(entry.get('title') or 'New Chat'))
        db.session.add(thread)
        db.session.flush()
    if state is None:
        state = db.session.get(SyncThreadState, thread.id)
    if state is not None and client_uuid and not state.client_uuid:
        clash = SyncThreadState.query.filter_by(user_id=current_user.id, client_uuid=client_uuid).first()
        if clash is None:
            state.client_uuid = client_uuid
    elif state is None:
        state = SyncThreadState(thread_id=thread.id, user_id=current_user.id, client_uuid=client_uuid, changed_at=datetime.utcnow())
        db.session.add(state)
    return thread, state, None


def _sync_apply_meta(thread, state, entry):
    """Thread settings: the later edit wins (device ``meta_changed_at_ms`` vs server change time)."""
    changed_ms = _sync_ms(entry.get('meta_changed_at_ms'))
    server_ms = _sync_to_ms(state.changed_at) if state and state.changed_at else 0
    if changed_ms is None or (server_ms and changed_ms < server_ms):
        return
    if 'title' in entry:
        thread.title = _normalize_thread_title(entry.get('title') or 'New Chat')
    if 'is_bookmarked' in entry:
        bookmarked = bool(entry.get('is_bookmarked'))
        if bookmarked != bool(thread.is_bookmarked):
            thread.is_bookmarked = bookmarked
            thread.bookmarked_at = datetime.utcnow() if bookmarked else None
    if 'custom_instruction' in entry:
        thread.custom_instruction = str(entry.get('custom_instruction') or '')[:100_000]
    if 'include_global_instruction' in entry:
        thread.include_global_instruction = bool(entry.get('include_global_instruction'))
    if 'last_gem_uuid' in entry:
        value = str(entry.get('last_gem_uuid') or '').strip()[:36]
        thread.last_gem_uuid = value or None


def _sync_insert_messages(thread, entry, result):
    """Inserts the device messages of one chat (parents first); returns the number inserted."""
    inserted = 0
    known = {}
    is_enc = bool(getattr(current_user, 'enable_e2ee', False))
    now = datetime.utcnow()
    latest = thread.updated_at
    for item in entry.get('messages') or []:
        if not isinstance(item, dict):
            continue
        client_uuid = _sync_uuid(item.get('client_uuid'))
        if not client_uuid:
            result['rejected'].append({'client_uuid': item.get('client_uuid'), 'reason': 'invalid_uuid'})
            continue
        existing = SyncMessageRef.query.filter_by(user_id=current_user.id, client_uuid=client_uuid).first()
        if existing:
            known[client_uuid] = existing.message_id
            result['messages'].append({'client_uuid': client_uuid, 'id': existing.message_id})
            continue
        role = item.get('role')
        if role not in ('user', 'assistant'):
            result['rejected'].append({'client_uuid': client_uuid, 'reason': 'invalid_role'})
            continue
        model = str(item.get('model') or '').strip()
        if model and model not in ALL_VALID_MODEL_IDS:
            model = ''
        content = str(item.get('content') or '')
        thought = str(item.get('thought') or '')
        if len(content) > (500_000 if role == 'user' else 2_000_000) or len(thought) > 2_000_000:
            result['rejected'].append({'client_uuid': client_uuid, 'reason': 'too_large'})
            continue
        parent_id = None
        parent = item.get('parent') or {}
        if isinstance(parent, dict) and (parent.get('id') is not None or parent.get('client_uuid')):
            if parent.get('id') is not None:
                try:
                    candidate = int(parent.get('id'))
                except (TypeError, ValueError):
                    candidate = None
                row = db.session.get(Message, candidate) if candidate else None
                parent_id = row.id if row and row.thread_id == thread.id else None
            else:
                parent_uuid = _sync_uuid(parent.get('client_uuid'))
                parent_id = known.get(parent_uuid)
                if parent_id is None and parent_uuid:
                    ref = SyncMessageRef.query.filter_by(user_id=current_user.id, client_uuid=parent_uuid).first()
                    parent_id = ref.message_id if ref and ref.thread_id == thread.id else None
            if parent_id is None:
                result['rejected'].append({'client_uuid': client_uuid, 'reason': 'parent_missing'})
                continue
        raw_files = item.get('files') or []
        if not isinstance(raw_files, list):
            raw_files = [raw_files]
        files = _normalize_attachment_list(raw_files, current_user.id)
        if len(files) != len(raw_files) or any(not _get_file_disk_info(ref).get('exists') for ref in files):
            result['rejected'].append({'client_uuid': client_uuid, 'reason': 'invalid_files'})
            continue
        created_ms = _sync_ms(item.get('created_at_ms'))
        created = _sync_from_ms(created_ms) if created_ms else now
        if created > now:
            created = now
        if parent_id:
            parent_row = db.session.get(Message, parent_id)
            if parent_row and parent_row.timestamp and created <= parent_row.timestamp:
                created = parent_row.timestamp + timedelta(milliseconds=1)
        quote = str(item.get('quote_text') or '')[:200_000] or None
        thought_payload = json.dumps({'text': thought}, ensure_ascii=False) if thought else None
        message = Message(
            thread_id=thread.id, role=role,
            content=encrypt_val(content) if is_enc else content,
            model=model or None,
            image_url=json.dumps(files) if files else None,
            timestamp=created,
            tokens=count_tokens(content, model or 'gpt-4') if role == 'user' else count_tokens_for_display(content, model or 'gpt-4', thought or None),
            tokens_in=_sync_ms(item.get('tokens_in')) or 0,
            tokens_out=_sync_ms(item.get('tokens_out')) or 0,
            thought_data=(encrypt_val(thought_payload) if is_enc and thought_payload else thought_payload),
            quote_text=quote, is_encrypted=is_enc,
            gem_uuid=(str(item.get('gem_uuid') or '').strip()[:36] or None),
            gem_name=(str(item.get('gem_name') or '').strip()[:100] or None),
            parent_id=parent_id,
        )
        db.session.add(message)
        db.session.flush()
        db.session.add(SyncMessageRef(message_id=message.id, thread_id=thread.id, user_id=current_user.id, client_uuid=client_uuid))
        known[client_uuid] = message.id
        result['messages'].append({'client_uuid': client_uuid, 'id': message.id})
        inserted += 1
        if model and role == 'assistant':
            thread.last_model = model
        if latest is None or created > latest:
            latest = created
    if inserted:
        thread.updated_at = latest or now
    return inserted


@app.route('/api/mobile/v1/sync/push', methods=['POST'])
def mobile_sync_push():
    """Uploads device chats: new chats and messages (idempotent by UUID), setting changes and deletions."""
    if request.content_length and request.content_length > _SYNC_PUSH_MAX_BYTES:
        return _mobile_error('payload_too_large', 413)
    if not rate_limit(f'rl:mobile:sync:push:{current_user.id}', 60, 60):
        return _mobile_error('rate_limit', 429)
    try:
        status = redis_conn.get(f"migration_status:{current_user.id}")
        if status and (status.decode() if isinstance(status, bytes) else str(status)) == 'processing':
            return _mobile_error('e2ee_migration_in_progress', 409)
    except Exception:
        pass
    body = request.get_json(silent=True) or {}
    entries = body.get('threads') or []
    if not isinstance(entries, list):
        return _mobile_error('invalid_request')
    if sum(len(e.get('messages') or []) for e in entries if isinstance(e, dict)) > _SYNC_PUSH_MAX_MESSAGES:
        return _mobile_error('too_many_messages', 413)
    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result = {'client_uuid': entry.get('client_uuid'), 'messages': [], 'rejected': []}
        thread, state, error = _sync_resolve_thread(entry)
        if error:
            result['error'] = error
            results.append(result)
            continue
        if not _sync_thread_visible(thread) and thread.is_temporary:
            result['error'] = 'temporary_chat'
            results.append(result)
            continue
        _sync_apply_meta(thread, state, entry)
        _sync_insert_messages(thread, entry, result)
        result['id'] = thread.public_id
        results.append(result)
    deleted_threads = []
    for public_id in (body.get('deleted_threads') or [])[:200]:
        thread = resolve_thread_for_user(str(public_id), current_user.id)
        if thread is None:
            deleted_threads.append(str(public_id))
            continue
        for message in thread.messages:
            for ref in _iter_message_attachment_refs(message.image_url):
                try:
                    _delete_user_upload_ref(current_user.id, ref)
                except Exception:
                    pass
        db.session.delete(thread)
        deleted_threads.append(str(public_id))
    deleted_messages = []
    for raw_id in (body.get('deleted_messages') or [])[:200]:
        try:
            message = db.session.get(Message, int(raw_id))
        except (TypeError, ValueError):
            message = None
        if message is None:
            deleted_messages.append(raw_id)
            continue
        if message.thread.user_id != current_user.id:
            continue
        _delete_message_and_following(message, current_user.id)
        deleted_messages.append(raw_id)
    safe_db_commit()
    return jsonify({'status': 'ok', 'threads': results, 'deleted_threads': deleted_threads,
                    'deleted_messages': deleted_messages, 'server_time_ms': _sync_to_ms(datetime.utcnow())})


@app.route('/api/mobile/v1/secrets/export', methods=['POST'])
def mobile_secrets_export():
    """The signed-in user's own API keys, for serverless mode (requires a recent re-authentication).

    Operator environment keys are never included; only the user's stored columns are read.
    """
    if not rate_limit(f'rl:mobile:secrets:export:{current_user.id}', 10, 3600):
        return _mobile_error('rate_limit', 429)
    secrets_payload = {}
    for field, column in _SYNC_SECRET_FIELDS:
        value = getattr(current_user, column, None)
        secrets_payload[field] = (decrypt_val(value) or '') if value else ''
    secrets_payload['model_api_keys'] = _load_user_model_api_key_map(current_user)
    logger.info('Android secrets export for user %s (%d provider keys)', current_user.id,
                sum(1 for field, _column in _SYNC_SECRET_FIELDS if secrets_payload.get(field)))
    response = jsonify({'status': 'ok', 'secrets': secrets_payload})
    response.headers['Cache-Control'] = 'no-store'
    return response
