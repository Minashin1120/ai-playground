"""Media upload routes extracted from ``routes_media.py``.

``/upload``, ``/upload/init``, ``/upload/chunk``, ``/upload/complete`` and the
413 handler. Loaded into the shared ``app`` namespace by ``_SERVER_PARTS``,
after ``routes_media.py`` and before ``routes_media_storage.py``.
"""

# region: File upload
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    ALLOWED_EXTENSIONS = _UPLOAD_ALLOWED_EXTENSIONS
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    try:
        # Reclaim abandoned chunk sessions before they are included in the
        # storage-capacity check for a new upload.
        _cleanup_stale_chunk_uploads(current_user.id)
        if not _is_primary_admin_user(current_user):
            hard_limit = _get_user_storage_limit_bytes(current_user)
            if hard_limit:
                for f in files:
                    size = _get_filestorage_size(f)
                    if size is not None and size > hard_limit:
                        limit_mb = _bytes_to_mb_str(hard_limit)
                        return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        total_incoming = 0
        for f in files:
            size = _get_filestorage_size(f)
            if size is None:
                continue
            total_incoming += size
        if not _is_primary_admin_user(current_user):
            ok, used, limit = _check_storage_capacity(current_user, total_incoming)
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
    except Exception:
        return jsonify({'error': 'Unable to validate upload size'}), 400
    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass
    res = []
    cache_updated = False
    for f in files:
        if f.filename:
            orig_name, ext = _sanitize_upload_filename(f.filename)
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({'error': f'File type {ext} not allowed'}), 400
            
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            save_path = os.path.join(ud, fname)
            if current_user.enable_e2ee:
                with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                    ef.write(encrypt_bytes(f.read()))
            else:
                f.save(save_path)
            rel_path = f"{current_user.id}/{fname}"
            res.append(rel_path)
            try:
                disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
                size = None
                mtime = None
                try:
                    size = os.path.getsize(disk_path)
                except Exception:
                    size = None
                try:
                    mtime = int(os.path.getmtime(disk_path))
                except Exception:
                    mtime = None
                mime_guess = mimetypes.guess_type(fname)[0]
                mime = _normalize_media_mime(fname, mime_guess)
                _upsert_file_cache(
                    current_user.id,
                    rel_path,
                    "local",
                    size_bytes=size,
                    mtime=mtime,
                    mime_type=mime,
                    state="stored",
                    last_error=None
                )
                cache_updated = True
            except Exception:
                pass
    if cache_updated:
        try:
            safe_db_commit()
        except Exception:
            pass
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/upload/init', methods=['POST'])
@login_required
def upload_init():
    if not rate_limit(f"rl:upload_init:user:{current_user.id}", 30, 60):
        return jsonify({'error': 'Too many upload requests'}), 429
    data = request.get_json(silent=True) or {}
    filename, ext = _sanitize_upload_filename(data.get('filename') or '')
    try:
        total_size = int(data.get('size') or 0)
    except (TypeError, ValueError):
        total_size = 0
    if not filename or total_size <= 0:
        return jsonify({'error': 'Invalid upload'}), 400

    allowed = _UPLOAD_ALLOWED_EXTENSIONS
    if ext not in allowed:
        return jsonify({'error': f'File type {ext} not allowed'}), 400

    # This must run before the quota scan. Otherwise an abandoned partial upload
    # can keep a user over quota and prevent the very request that would clean it.
    active_uploads = _cleanup_stale_chunk_uploads(current_user.id)

    if not _is_primary_admin_user(current_user):
        hard_limit = _get_user_storage_limit_bytes(current_user)
        if hard_limit and total_size > hard_limit:
            limit_mb = _bytes_to_mb_str(hard_limit)
            return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        ok, used, limit = _check_storage_capacity(current_user, total_size)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413

    if active_uploads >= _CHUNK_UPLOAD_MAX_ACTIVE:
        return jsonify({'error': 'Too many active uploads'}), 429

    upload_id = f"up_{int(time.time())}_{os.urandom(4).hex()}"
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir:
        return jsonify({'error': 'Init failed'}), 500
    os.makedirs(session_dir, exist_ok=True)
    os.chmod(session_dir, 0o700)
    meta = {
        "filename": filename,
        "size": total_size,
        "received": 0,
        "created": int(time.time()),
        "updated": int(time.time()),
        "ext": ext,
        "chunk_size": _CHUNK_SIZE_BYTES,
        "total_chunks": (total_size + _CHUNK_SIZE_BYTES - 1) // _CHUNK_SIZE_BYTES,
        "state": "receiving"
    }
    if not _save_chunk_meta(os.path.join(session_dir, 'meta.json'), meta):
        return jsonify({'error': 'Init failed'}), 500
    return jsonify({'upload_id': upload_id, 'chunk_size': _CHUNK_SIZE_BYTES})

@app.route('/upload/chunk', methods=['POST'])
@login_required
def upload_chunk():
    if not rate_limit(f"rl:upload_chunk:user:{current_user.id}", 240, 60):
        return jsonify({'error': 'Too many upload requests'}), 429
    upload_id = (request.form.get('upload_id') or '').strip()
    index = request.form.get('index')
    total = request.form.get('total')
    f = request.files.get('chunk')
    if not _is_valid_chunk_upload_id(upload_id) or f is None:
        return jsonify({'error': 'Invalid chunk'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir or not os.path.isdir(session_dir):
        return jsonify({'error': 'Upload not found'}), 404
    meta_path = os.path.join(session_dir, 'meta.json')
    try:
        index = int(index) if index is not None else 0
        total = int(total) if total is not None else 0
    except Exception:
        return jsonify({'error': 'Invalid chunk index'}), 400
    part_path = os.path.join(session_dir, 'data.part')
    try:
        with _chunk_upload_lock(session_dir):
            meta = _load_chunk_meta(meta_path)
            if not meta:
                return jsonify({'error': 'Upload not found'}), 404
            if meta.get('state') != 'receiving':
                return jsonify({'error': 'Upload is not accepting chunks'}), 409
            declared_size = int(meta.get('size') or 0)
            received = int(meta.get('received') or 0)
            chunk_size = int(meta.get('chunk_size') or _CHUNK_SIZE_BYTES)
            expected_total = int(meta.get('total_chunks') or 0)
            if received >= declared_size:
                return jsonify({'error': 'Upload already complete'}), 409
            if chunk_size != _CHUNK_SIZE_BYTES or expected_total <= 0 or total != expected_total:
                return jsonify({'error': 'Invalid chunk count'}), 400
            expected_index = received // chunk_size
            if index != expected_index or index < 0 or index >= expected_total:
                return jsonify({'error': 'Invalid chunk order'}), 409
            current_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if current_size != received:
                return jsonify({'error': 'Upload state mismatch'}), 409
            remaining = declared_size - received
            expected_bytes = min(chunk_size, remaining)
            chunk_data = f.read(chunk_size + 1)
            if expected_bytes <= 0 or len(chunk_data) != expected_bytes:
                return jsonify({'error': 'Invalid chunk size'}), 400
            if not _is_primary_admin_user(current_user):
                limit = _get_user_storage_limit_bytes(current_user)
                used = _get_user_storage_usage_bytes(current_user.id)
                if limit and used + len(chunk_data) > limit:
                    return jsonify({'error': 'Storage limit exceeded'}), 413
            with open(part_path, 'ab') as out:
                out.write(chunk_data)
                out.flush()
            meta['received'] = received + len(chunk_data)
            meta['updated'] = int(time.time())
            if not _save_chunk_meta(meta_path, meta):
                with open(part_path, 'r+b') as out:
                    out.truncate(received)
                return jsonify({'error': 'Chunk state write failed'}), 500
    except Exception:
        return jsonify({'error': 'Chunk write failed'}), 500

    return jsonify({'received': meta['received'], 'total': meta.get('size', 0), 'index': index, 'chunks': total})

@app.route('/upload/complete', methods=['POST'])
@login_required
def upload_complete():
    data = request.get_json(silent=True) or {}
    upload_id = (data.get('upload_id') or '').strip()
    if not _is_valid_chunk_upload_id(upload_id):
        return jsonify({'error': 'Invalid upload'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    if not session_dir or not os.path.isdir(session_dir):
        return jsonify({'error': 'Upload not found'}), 404
    meta_path = os.path.join(session_dir, 'meta.json')
    part_path = os.path.join(session_dir, 'data.part')
    with _chunk_upload_lock(session_dir):
        meta = _load_chunk_meta(meta_path)
        if not meta:
            return jsonify({'error': 'Upload not found'}), 404
        if meta.get('state') != 'receiving':
            return jsonify({'error': 'Upload is already finalizing'}), 409
        if not os.path.exists(part_path):
            return jsonify({'error': 'Upload missing'}), 400
        declared_size = int(meta.get('size') or 0)
        if int(meta.get('received') or 0) != declared_size or os.path.getsize(part_path) != declared_size:
            return jsonify({'error': 'Upload incomplete'}), 400
        meta['state'] = 'finalizing'
        if not _save_chunk_meta(meta_path, meta):
            return jsonify({'error': 'Finalize state failed'}), 500

    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass

    ext = (meta.get('ext') or '').lower()
    if not ext.startswith('.'):
        # Fall back to the sanitized filename's extension for sessions created
        # before the meta["ext"] value started carrying the original extension.
        ext = os.path.splitext(str(meta.get('filename') or 'file'))[1].lower()
    fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
    fname = f"{fname_base}{ext}"
    save_path = os.path.join(ud, fname)
    res = []
    cache_updated = False
    try:
        if current_user.enable_e2ee:
            with open(part_path, 'rb') as rf:
                with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                    ef.write(encrypt_bytes(rf.read()))
        else:
            os.replace(part_path, save_path)
        res.append(f"{current_user.id}/{fname}")
        rel_path = f"{current_user.id}/{fname}"
        try:
            disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
            size = None
            mtime = None
            try:
                size = os.path.getsize(disk_path)
            except Exception:
                size = None
            try:
                mtime = int(os.path.getmtime(disk_path))
            except Exception:
                mtime = None
            mime_guess = mimetypes.guess_type(fname)[0]
            mime = _normalize_media_mime(fname, mime_guess)
            _upsert_file_cache(
                current_user.id,
                rel_path,
                "local",
                size_bytes=size,
                mtime=mtime,
                mime_type=mime,
                state="stored",
                last_error=None
            )
            cache_updated = True
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Chunk finalize failed: {e}")
        return jsonify({'error': 'Finalize failed'}), 500
    finally:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except Exception:
            pass
        try:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            lock_path = os.path.join(session_dir, '.lock')
            if os.path.exists(lock_path):
                os.remove(lock_path)
            os.rmdir(session_dir)
        except Exception:
            pass
    if cache_updated:
        try:
            safe_db_commit()
        except Exception:
            pass
    return jsonify({'filename': res[0] if res else '', 'filenames': res})
