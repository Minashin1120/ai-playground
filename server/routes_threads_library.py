import heapq


@app.route('/api/encryption_scan', methods=['GET'])
@login_required
def encryption_scan():
    thread_id = request.args.get('thread_id')
    q = Message.query.join(Thread, Message.thread_id == Thread.id).filter(Thread.user_id == current_user.id)
    target_thread = None
    if thread_id:
        target_thread = resolve_thread_for_user(thread_id, current_user.id)
        if not target_thread:
            return jsonify({'error': 'Invalid thread'}), 403
        q = q.filter(Message.thread_id == target_thread.id)
    try:
        total = q.count()
        encrypted = q.filter(Message.is_encrypted.is_(True)).count()
        unencrypted = q.filter((Message.is_encrypted.is_(False)) | (Message.is_encrypted.is_(None))).order_by(Message.timestamp.desc()).limit(100).all()
        unenc_list = []
        for m in unencrypted:
            unenc_list.append({
                "id": m.id,
                "thread_id": m.thread.public_id if m.thread else None,
                "role": m.role,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            })
        return jsonify({
            "thread_id": target_thread.public_id if target_thread else None,
            "total": total,
            "encrypted": encrypted,
            "unencrypted": total - encrypted,
            "samples": unenc_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/threads', methods=['GET'])
@login_required
def admin_threads_list():
    """List the current admin account's own threads with encryption status.

    Username selection was removed: only the logged-in admin's chats are in scope.
    """
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    q = (request.args.get('q') or '').strip()
    query = Thread.query.filter_by(user_id=current_user.id)
    if q:
        query = query.filter(Thread.title.contains(q))
    threads = query.order_by(Thread.updated_at.desc()).limit(500).all()
    res = []
    for t in threads:
        msgs = Message.query.filter_by(thread_id=t.id).all()
        total = len(msgs)
        enc = sum(1 for m in msgs if m.is_encrypted)
        res.append({
            'thread_id': t.public_id or t.id,
            'title': t.title,
            'updated_at': t.updated_at.isoformat() if t.updated_at else None,
            'message_count': total,
            'encrypted_count': enc,
            'encrypted': enc > 0,
        })
    return jsonify({
        'user': {
            'username': current_user.username,
            'enable_e2ee': bool(getattr(current_user, 'enable_e2ee', False)),
        },
        'threads': res
    })

@app.route('/api/admin/threads/<thread_id>/encryption', methods=['POST'])
@login_required
def admin_toggle_thread_encryption(thread_id):
    """Decrypt or re-encrypt a single thread owned by the current admin account."""
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.get_json(silent=True) or {}
    enable = bool(data.get('enable'))
    # Scope to the admin's own threads only (other users' chats are never targeted).
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'thread_not_found'}), 404
    changed = 0
    for m in t.messages:
        if enable and not m.is_encrypted:
            if m.content:
                m.content = encrypt_val(m.content)
            if m.thought_data:
                m.thought_data = encrypt_val(m.thought_data)
            m.is_encrypted = True
            changed += 1
        elif not enable and m.is_encrypted:
            if m.content:
                m.content = decrypt_val(m.content)
            if m.thought_data:
                m.thought_data = decrypt_val(m.thought_data)
            m.is_encrypted = False
            changed += 1
    safe_db_commit()
    return jsonify({
        'status': 'ok',
        'thread_id': t.public_id or t.id,
        'enable': enable,
        'changed': changed,
        'total': len(t.messages)
    })

@app.route('/api/threads/<thread_id>/settings', methods=['GET', 'PUT'])
@login_required
def update_thread_settings(thread_id):
    log_force(f"DEBUG: update_thread_settings started for {thread_id}, method={request.method}")
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        log_force(f"DEBUG: thread not found for {thread_id}")
        return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        log_force(f"DEBUG: update_thread_settings GET returning data for {thread_id}")
        return jsonify({
            'custom_instruction': t.custom_instruction,
            'include_global_instruction': t.include_global_instruction if t.include_global_instruction is not None else True,
            'is_temporary': bool(getattr(t, "is_temporary", False)),
            **_get_temp_chat_runtime_meta(t, user=current_user)
        })
    d = request.json or {}
    log_force(f"DEBUG: update_thread_settings received keys: {sorted(d.keys())}")
    if 'custom_instruction' in d:
        custom_instruction = str(d.get('custom_instruction') or '')
        if len(custom_instruction) > 100_000:
            return jsonify({'error': 'payload_too_large'}), 413
        t.custom_instruction = custom_instruction
    if 'include_global_instruction' in d:
        t.include_global_instruction = bool(d['include_global_instruction'])
    if 'is_temporary' in d:
        requested_temp = _coerce_bool_or_none(d.get('is_temporary'))
        t.is_temporary = bool(requested_temp)
    
    t.updated_at = datetime.utcnow()
    db.session.add(t)
    log_force(f"DEBUG: calling safe_db_commit")
    safe_db_commit()
    log_force(f"DEBUG: safe_db_commit finished")
    if bool(getattr(t, "is_temporary", False)):
        _mark_temp_chat_presence(
            t,
            current_user.id,
            timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
        )
    else:
        _clear_temp_chat_tracking_for_thread(t)
    log_force(f"DEBUG: update_thread_settings returning ok")
    temp_meta = _get_temp_chat_runtime_meta(t, user=current_user)
    return jsonify({
        'status': 'ok',
        'is_temporary': bool(getattr(t, "is_temporary", False)),
        'timeout_seconds': temp_meta.get('timeout_seconds'),
        'temp_chat_expires_at': temp_meta.get('temp_chat_expires_at'),
        'temp_chat_remaining_seconds': temp_meta.get('temp_chat_remaining_seconds')
    })

@app.route('/api/threads/<thread_id>/title', methods=['PUT'])
@login_required
def update_title(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    data = request.get_json(silent=True) or {}
    t.title = _normalize_thread_title(data.get('title', 'Untitled'))
    safe_db_commit()
    return jsonify({'status': 'ok', 'title': t.title})

@app.route('/api/threads/<thread_id>/bookmark', methods=['POST'])
@login_required
def toggle_bookmark(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    t.is_bookmarked = not bool(t.is_bookmarked)
    t.bookmarked_at = datetime.utcnow() if t.is_bookmarked else None
    safe_db_commit()
    return jsonify({'status': 'ok', 'is_bookmarked': t.is_bookmarked})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    if msg.thread.user_id != current_user.id: return jsonify({'error': '403'}), 403
    
    msgs_to_delete = Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).all()
    for m in msgs_to_delete:
        if m.image_url:
            try:
                for p in _iter_message_attachment_refs(m.image_url):
                    _delete_user_upload_ref(current_user.id, p)
            except Exception:
                pass

    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    safe_db_commit()
    return jsonify({'status': 'ok'})

class _LibraryHeapEntry:
    """Keep the worst selected item at the heap root."""

    __slots__ = ('key', 'item', 'reverse')

    def __init__(self, key, item, reverse):
        self.key = key
        self.item = item
        self.reverse = reverse

    def __lt__(self, other):
        if self.reverse:
            return self.key < other.key
        return self.key > other.key


def _library_sort_spec(sort_order, item):
    filename = str(item.get('filename') or '').casefold()
    filepath = str(item.get('filepath') or '')
    timestamp = int(item.get('ts') or 0)
    if sort_order == 'oldest':
        return (timestamp, filename, filepath), False
    if sort_order == 'name_asc':
        return (filename, -timestamp, filepath), False
    if sort_order == 'name_desc':
        return (filename, -timestamp, filepath), True
    return (timestamp, filename, filepath), True


def _library_item_is_better(candidate, current, reverse):
    return candidate > current if reverse else candidate < current


@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    """Return a bounded page without loading the account's messages."""
    try:
        limit = max(1, min(request.args.get('limit', 40, type=int) or 40, 40))
        offset = max(0, min(request.args.get('offset', 0, type=int) or 0, 10_000))
        sort_order = (request.args.get('sort') or 'newest').strip().lower()
        if sort_order not in {'newest', 'oldest', 'name_asc', 'name_desc'}:
            sort_order = 'newest'
        search = (request.args.get('q') or '').strip().casefold()
        favorites_only = str(request.args.get('favorites_only', '')).lower() in {'1', 'true', 'yes', 'on'}

        label_map = _get_user_file_label_map(current_user.id)
        favorite_paths = set()
        favorite_query = FileCache.query.with_entities(FileCache.rel_path).filter_by(
            user_id=current_user.id, provider='favorite'
        ).yield_per(500)
        for (rel_path,) in favorite_query:
            if rel_path:
                favorite_paths.add(rel_path)

        # Keep only the requested page window in memory while scandir streams
        # directory entries.  This bounds both ORM and Python allocations.
        page_window = offset + limit
        selected = []
        total = 0
        image_exts = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'svg', 'heic'}
        upload_root = app.config['UPLOAD_FOLDER']
        user_dir = os.path.join(upload_root, str(current_user.id))
        if os.path.isdir(user_dir) and _path_is_within(upload_root, user_dir):
            with os.scandir(user_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        name = entry.name
                        if not name or name.startswith('.'):
                            continue
                        if name.endswith('.enc'):
                            base_name = name[:-4]
                            if not base_name or os.path.exists(os.path.join(user_dir, base_name)):
                                continue
                        else:
                            base_name = name
                        rel_path = f'{current_user.id}/{base_name}'
                        display_name = label_map.get(rel_path) or base_name
                        if search and search not in display_name.casefold():
                            continue
                        is_favorite = rel_path in favorite_paths
                        if favorites_only and not is_favorite:
                            continue
                        try:
                            timestamp = int(entry.stat(follow_symlinks=False).st_mtime)
                        except Exception:
                            timestamp = 0
                        ext = os.path.splitext(base_name)[1].lower().lstrip('.')
                        item = {
                            'filename': display_name,
                            'original_filename': base_name,
                            'filepath': rel_path,
                            'url': url_for('serve_file', filename=rel_path),
                            'thumbnail_url': url_for('serve_file_thumb', filename=rel_path) if ext in image_exts else None,
                            'type': 'image' if ext in image_exts else 'file',
                            'ext': ext,
                            'is_favorite': is_favorite,
                            'ts': timestamp,
                        }
                        total += 1
                        key, reverse = _library_sort_spec(sort_order, item)
                        heap_entry = _LibraryHeapEntry(key, item, reverse)
                        if len(selected) < page_window:
                            heapq.heappush(selected, heap_entry)
                        elif _library_item_is_better(key, selected[0].key, reverse):
                            heapq.heapreplace(selected, heap_entry)
                    except (OSError, ValueError):
                        continue

        selected.sort(key=lambda entry: entry.key, reverse=selected[0].reverse if selected else False)
        page_items = [entry.item for entry in selected[offset:offset + limit]]
        return jsonify({
            'files': page_items,
            'total': total,
            'offset': offset,
            'limit': limit,
            'has_more': total > offset + len(page_items),
        })
    except Exception as exc:
        log_force(f'get_files_lib failed: {exc}')
        return jsonify({'error': 'library_load_failed', 'files': [], 'total': 0, 'has_more': False}), 500


@app.route('/api/files/usage', methods=['GET'])
@login_required
def get_file_usage_chats():
    """List chats that reference one library file without loading message bodies."""
    try:
        rel_path = _normalize_upload_ref(request.args.get('filepath') or request.args.get('path'))
        if not rel_path:
            return jsonify({'error': 'invalid filepath'}), 400
        if not rel_path.startswith(f'{current_user.id}/'):
            return jsonify({'error': 'forbidden'}), 403
        info = _get_file_disk_info(rel_path)
        if not info or not info.get('exists'):
            return jsonify({'error': 'file not found'}), 404

        matched_thread_ids = []
        seen_thread_ids = set()
        # image_url is the attachment reference column and is not encrypted.
        # Select only the two small columns and stop after the bounded result
        # set; large message body columns are intentionally never selected.
        message_query = db.session.query(Message.thread_id, Message.image_url).join(
            Thread, Message.thread_id == Thread.id
        ).filter(
            Thread.user_id == current_user.id,
            Message.image_url.contains(rel_path),
        ).order_by(Message.id.desc()).yield_per(500)
        has_more = False
        for thread_id, raw_refs in message_query:
            if thread_id in seen_thread_ids:
                continue
            refs = _iter_message_attachment_refs(raw_refs)
            if not any(_normalize_upload_ref(ref) == rel_path for ref in refs):
                continue
            seen_thread_ids.add(thread_id)
            matched_thread_ids.append(thread_id)
            if len(matched_thread_ids) >= 101:
                break

        has_more = len(matched_thread_ids) > 100
        matched_thread_ids = matched_thread_ids[:100]
        threads = []
        if matched_thread_ids:
            thread_query = Thread.query.with_entities(
                Thread.id, Thread.public_id, Thread.title, Thread.updated_at
            ).filter(
                Thread.user_id == current_user.id,
                Thread.id.in_(matched_thread_ids),
            ).order_by(Thread.updated_at.desc(), Thread.id.desc())
            for thread_id, public_id, title, updated_at in thread_query:
                threads.append({
                    'id': public_id or thread_id,
                    'title': title or '新しいチャット',
                    'updated_at': updated_at.isoformat() if updated_at else None,
                })
        return jsonify({
            'filepath': rel_path,
            'chats': threads,
            'has_more': has_more,
        })
    except Exception as exc:
        log_force(f'get_file_usage_chats failed: {exc}')
        return jsonify({'error': 'file_usage_lookup_failed'}), 500


@app.route('/api/files/favorite', methods=['POST'])
@login_required
def toggle_file_favorite():
    data = request.get_json(silent=True) or {}
    rel_path = _normalize_upload_ref(data.get('filepath') or data.get('path'))
    if not rel_path:
        return jsonify({'error': 'invalid filepath'}), 400
    if rel_path.startswith('..') or os.path.isabs(rel_path) or not rel_path.startswith(f'{current_user.id}/'):
        return jsonify({'error': 'forbidden'}), 403
    info = _get_file_disk_info(rel_path)
    if not info or not info.get('exists'):
        return jsonify({'error': 'file not found'}), 404
    try:
        favorite = FileCache.query.filter_by(
            user_id=current_user.id, rel_path=rel_path, provider='favorite'
        ).order_by(FileCache.id.desc()).first()
        if favorite:
            db.session.delete(favorite)
            is_favorite = False
        else:
            _upsert_file_cache(
                current_user.id,
                rel_path,
                'favorite',
                state='ready',
                last_error=None,
            )
            is_favorite = True
        safe_db_commit()
        return jsonify({'status': 'ok', 'filepath': rel_path, 'is_favorite': is_favorite})
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'favorite update failed'}), 500


@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    for f in request.json.get('filenames', []):
        norm = _normalize_upload_ref(f)
        if not norm:
            continue
        if norm.startswith("..") or os.path.isabs(norm): continue
        if norm.startswith(f"{current_user.id}/"):
            fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
            if not _path_is_within(app.config['UPLOAD_FOLDER'], fp): continue
            secure_delete(fp)
            secure_delete(fp + '.enc')
            _delete_file_cache_for_path(current_user.id, norm)
    return jsonify({'status': 'ok'})

@app.route('/api/files/rename', methods=['POST'])
@login_required
def rename_library_file():
    data = request.json or {}
    rel_path = _normalize_upload_ref(data.get('filepath') or data.get('path'))
    if not rel_path:
        return jsonify({'error': 'invalid filepath'}), 400
    if rel_path.startswith("..") or os.path.isabs(rel_path) or not rel_path.startswith(f"{current_user.id}/"):
        return jsonify({'error': 'forbidden'}), 403
    info = _get_file_disk_info(rel_path)
    if not info or not info.get("exists"):
        return jsonify({'error': 'file not found'}), 404
    base_name = os.path.basename(rel_path)
    display_name = _normalize_display_name_for_path(rel_path, data.get('filename') or data.get('name'))
    if not display_name:
        return jsonify({'error': 'invalid filename'}), 400
    try:
        if display_name == base_name:
            FileCache.query.filter_by(user_id=current_user.id, rel_path=rel_path, provider="label").delete()
        else:
            _upsert_file_cache(
                current_user.id,
                rel_path,
                "label",
                file_uri=display_name,
                state="ready",
                last_error=None
            )
        safe_db_commit()
        return jsonify({'status': 'ok', 'filepath': rel_path, 'filename': display_name, 'original_filename': base_name})
    except Exception:
        return jsonify({'error': 'rename failed'}), 500
