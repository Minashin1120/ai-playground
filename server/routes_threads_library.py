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

@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    try:
        label_map = _get_user_file_label_map(current_user.id)
        favorite_paths = {
            row.rel_path for row in FileCache.query.filter_by(
                user_id=current_user.id, provider="favorite"
            ).all() if row.rel_path
        }
        msgs = Message.query.join(Thread).filter(Thread.user_id == current_user.id, Message.image_url != None).order_by(Message.timestamp.desc()).all()
        files = []
        seen = set()
        image_exts = {'png','jpg','jpeg','webp','gif'}
        for m in msgs:
            if not m.image_url: continue
            try:
                l = json.loads(m.image_url)
                if not isinstance(l, list): l = [m.image_url]
            except: l = [m.image_url]
            msg_ts = None
            try:
                msg_ts = int(m.timestamp.timestamp())
            except:
                msg_ts = None
            for p in l:
                norm = _normalize_upload_ref(p)
                if norm and norm not in seen:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
                    if os.path.exists(fp) or os.path.exists(fp + '.enc'):
                        seen.add(norm)
                        ext = os.path.splitext(norm)[1].lower().replace('.', '')
                        base_name = os.path.basename(norm)
                        display_name = label_map.get(norm) or base_name
                        files.append({
                            'filename': display_name,
                            'original_filename': base_name,
                            'filepath': norm,
                            'url': url_for('serve_file', filename=norm),
                            'thumbnail_url': url_for('serve_file_thumb', filename=norm) if ext in image_exts else None,
                            'type': 'image' if ext in image_exts else 'file',
                            'ext': ext,
                            'is_favorite': norm in favorite_paths,
                            'ts': msg_ts
                        })
        # Include uploaded files that are not yet attached to any message
        ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.isdir(ud):
            for entry in os.scandir(ud):
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith('.'):
                    continue
                is_enc = name.endswith('.enc')
                base_name = name[:-4] if is_enc else name
                if not base_name:
                    continue
                rel_path = f"{current_user.id}/{base_name}"
                if rel_path in seen:
                    continue
                ext = os.path.splitext(base_name)[1].lower().replace('.', '')
                seen.add(rel_path)
                ts = None
                try:
                    ts = int(entry.stat().st_mtime)
                except:
                    ts = None
                display_name = label_map.get(rel_path) or os.path.basename(rel_path)
                files.append({
                    'filename': display_name,
                    'original_filename': os.path.basename(rel_path),
                    'filepath': rel_path,
                    'url': url_for('serve_file', filename=rel_path),
                    'thumbnail_url': url_for('serve_file_thumb', filename=rel_path) if ext in image_exts else None,
                    'type': 'image' if ext in image_exts else 'file',
                    'ext': ext,
                    'is_favorite': rel_path in favorite_paths,
                    'ts': ts
                })
        return jsonify(files)
    except: return jsonify([])


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
