@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    try:
        _delete_user_account_immediately(current_user)
        logout_user()
        return jsonify({'status': 'ok'})
    except Exception as e: return jsonify({'error': str(e)}), 500


@app.route('/api/account/transfer/<job_id>', methods=['GET'])
@login_required
def get_account_transfer_status(job_id):
    if not _valid_account_transfer_job_id(job_id):
        return jsonify({'error': 'invalid_job_id'}), 400
    try:
        response = jsonify(_account_transfer_status_payload(current_user.id, job_id))
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception:
        return jsonify({'error': 'status_unavailable'}), 503


@app.route('/api/account/transfer/<job_id>/cancel', methods=['POST'])
@login_required
def cancel_account_transfer(job_id):
    if not _valid_account_transfer_job_id(job_id):
        return jsonify({'error': 'invalid_job_id'}), 400
    try:
        existing = _account_transfer_status_payload(current_user.id, job_id)
        if existing.get("state") == "ready" and existing.get("available"):
            _delete_account_export_artifact(current_user.id, job_id, state="cancelled")
            return jsonify({'status': 'cancelled'})
        if existing:
            if existing.get("state") in {"completed", "failed", "cancelled", "expired"}:
                return jsonify({'status': existing.get("state")})
        redis_conn.setex(
            _account_transfer_cancel_key(current_user.id, job_id),
            ACCOUNT_TRANSFER_STATUS_TTL,
            "1",
        )
        _set_account_transfer_status(
            current_user.id, job_id, "cancelling", 0, "cancelling", "キャンセルしています"
        )
        return jsonify({'status': 'ok'})
    except Exception:
        return jsonify({'error': 'cancel_failed'}), 503


@app.route('/api/account/export/latest', methods=['GET'])
@login_required
def get_latest_account_export():
    _cleanup_expired_account_export_files()
    try:
        raw = redis_conn.get(_account_export_latest_key(current_user.id))
        job_id = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
    except Exception:
        return jsonify({'error': 'status_unavailable'}), 503
    if not _valid_account_transfer_job_id(job_id):
        response = jsonify({'state': 'idle', 'available': False})
        response.headers["Cache-Control"] = "no-store"
        return response
    response = jsonify(_account_transfer_status_payload(current_user.id, job_id))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route('/api/account/export', methods=['POST'])
@login_required
def export_account_data():
    body = request.get_json(silent=True) or {}
    job_id = str(body.get("job_id") or secrets.token_hex(16)).lower()
    if not _valid_account_transfer_job_id(job_id):
        return jsonify({'error': 'invalid_job_id'}), 400
    try:
        _cleanup_expired_account_export_files()
        claimed = redis_conn.set(
            _account_export_active_key(current_user.id), job_id,
            nx=True, ex=ACCOUNT_EXPORT_RETENTION_SECONDS,
        )
        if not claimed:
            active_raw = redis_conn.get(_account_export_active_key(current_user.id))
            active_job_id = active_raw.decode("utf-8") if isinstance(active_raw, bytes) else str(active_raw or "")
            return jsonify({'error': 'export_in_progress', 'job_id': active_job_id}), 409
        if not rate_limit(f"rl:account_export:user:{current_user.id}", 3, 3600):
            redis_conn.delete(_account_export_active_key(current_user.id))
            _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "エクスポート回数の上限に達しました")
            return jsonify({'error': 'rate_limit'}), 429
        old_raw = redis_conn.get(_account_export_latest_key(current_user.id))
        old_job_id = old_raw.decode("utf-8") if isinstance(old_raw, bytes) else str(old_raw or "")
        if _valid_account_transfer_job_id(old_job_id) and old_job_id != job_id:
            _delete_account_export_artifact(current_user.id, old_job_id)
        redis_conn.delete(_account_transfer_cancel_key(current_user.id, job_id))
        redis_conn.setex(_account_export_latest_key(current_user.id), ACCOUNT_TRANSFER_STATUS_TTL, job_id)
        _set_account_transfer_status(
            current_user.id, job_id, "queued", 0, "queued", "エクスポートを受け付けました",
            available=False,
        )
        task_queue.enqueue(
            build_account_export_task, current_user.id, job_id,
            job_timeout=ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS, result_ttl=ACCOUNT_TRANSFER_STATUS_TTL,
            failure_ttl=ACCOUNT_TRANSFER_STATUS_TTL,
            on_failure=account_export_job_failure,
        )
        response = jsonify({'status': 'accepted', 'job_id': job_id})
        response.headers["Cache-Control"] = "no-store"
        return response, 202
    except Exception:
        try:
            redis_conn.delete(_account_export_active_key(current_user.id))
        except Exception:
            pass
        logger.exception("Failed to enqueue account export for user %s", current_user.id)
        _set_account_transfer_status(
            current_user.id, job_id, "failed", 0, "failed", "エクスポートを開始できませんでした",
            available=False,
        )
        return jsonify({'error': 'export_enqueue_failed'}), 503


@app.route('/api/account/export/<job_id>/download', methods=['GET'])
@login_required
def download_account_export(job_id):
    if not _valid_account_transfer_job_id(job_id):
        return jsonify({'error': 'invalid_job_id'}), 400
    artifact = _account_export_artifact(current_user.id, job_id, destroy_on_missing=False)
    if not artifact:
        # Non-destructive: a failed download must not wipe the archive/record,
        # otherwise a single 404 makes the file look like it disappeared and
        # prevents any retry. State transitions are owned by the status
        # endpoint and the scheduled cleanup.
        return jsonify({'error': 'export_not_available'}), 404
    if not rate_limit(f"rl:account_export_download:user:{current_user.id}", 30, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        response = send_file(
            artifact["path"], mimetype="application/zip", as_attachment=True,
            download_name=artifact.get("filename") or "ai-playground-account.zip",
            conditional=True,
        )
    except OSError:
        # The archive vanished between the availability check and the actual
        # send (e.g. scheduled cleanup ran). Return 404 without destroying the
        # record so the UI can refresh to an accurate state.
        return jsonify({'error': 'export_not_available'}), 404
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


_ACCOUNT_IMPORT_UPLOAD_ID_RE = re.compile(r"^imp_[0-9]{10}_[0-9a-f]{16}$")
_ACCOUNT_IMPORT_CHUNK_BYTES = 10 * 1024 * 1024
_ACCOUNT_IMPORT_MAX_BYTES = 2 * 1024 * 1024 * 1024

def _account_import_upload_root():
    path = os.path.join(app.instance_path, "account_import_uploads")
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path

def _is_valid_account_import_upload_id(upload_id):
    return bool(_ACCOUNT_IMPORT_UPLOAD_ID_RE.fullmatch(str(upload_id or "")))

def _account_import_upload_dir(user_id, upload_id):
    if not _is_valid_account_import_upload_id(upload_id):
        return None
    user_root = os.path.realpath(os.path.join(_account_import_upload_root(), str(user_id)))
    candidate = os.path.realpath(os.path.join(user_root, upload_id))
    try:
        return candidate if os.path.commonpath((user_root, candidate)) == user_root else None
    except Exception:
        return None

def _cleanup_stale_account_import_uploads(user_id):
    root = os.path.join(_account_import_upload_root(), str(user_id))
    if not os.path.isdir(root):
        return
    cutoff = time.time() - 3600
    try:
        for entry in os.scandir(root):
            if entry.is_dir(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                _secure_delete_tree(entry.path)
    except Exception:
        logger.debug("Unable to sweep stale account import uploads", exc_info=True)

def _cleanup_all_stale_account_import_uploads():
    root = _account_import_upload_root()
    if not os.path.isdir(root):
        return
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            int(entry.name)
        except (TypeError, ValueError):
            continue
        _cleanup_stale_account_import_uploads(entry.name)

@app.route('/api/account/import/upload/start', methods=['POST'])
@login_required
def start_account_import_upload():
    data = request.get_json(silent=True) or {}
    try:
        total_size = int(data.get('size') or 0)
    except (TypeError, ValueError):
        total_size = 0
    if total_size <= 0 or total_size > _ACCOUNT_IMPORT_MAX_BYTES:
        return jsonify({'error': 'invalid_upload_size'}), 413
    if not rate_limit(f"rl:account_import_upload_start:user:{current_user.id}", 6, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    _cleanup_stale_account_import_uploads(current_user.id)
    user_root = os.path.join(_account_import_upload_root(), str(current_user.id))
    os.makedirs(user_root, mode=0o700, exist_ok=True)
    try:
        os.chmod(user_root, 0o700)
    except OSError:
        pass
    upload_id = f"imp_{int(time.time())}_{secrets.token_hex(8)}"
    upload_dir = _account_import_upload_dir(current_user.id, upload_id)
    os.makedirs(upload_dir, mode=0o700, exist_ok=False)
    try:
        os.chmod(upload_dir, 0o700)
    except OSError:
        pass
    meta_path = os.path.join(upload_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        json.dump({'size': total_size, 'received': 0, 'received_chunks': [], 'chunks': (total_size + _ACCOUNT_IMPORT_CHUNK_BYTES - 1) // _ACCOUNT_IMPORT_CHUNK_BYTES,
                   'updated': int(time.time()), 'state': 'receiving'}, meta_file)
    try:
        os.chmod(meta_path, 0o600)
    except OSError:
        pass
    return jsonify({'upload_id': upload_id, 'chunk_size': _ACCOUNT_IMPORT_CHUNK_BYTES,
                    'total_chunks': (total_size + _ACCOUNT_IMPORT_CHUNK_BYTES - 1) // _ACCOUNT_IMPORT_CHUNK_BYTES})

@app.route('/api/account/import/upload/<upload_id>/chunk', methods=['POST'])
@login_required
def account_import_upload_chunk(upload_id):
    if not _is_valid_account_import_upload_id(upload_id):
        return jsonify({'error': 'invalid_upload_id'}), 400
    upload_dir = _account_import_upload_dir(current_user.id, upload_id)
    meta_path = os.path.join(upload_dir, 'meta.json') if upload_dir else None
    chunk_file = request.files.get('chunk')
    try:
        index = int(request.form.get('index') or -1)
    except (TypeError, ValueError):
        index = -1
    if not upload_dir or not os.path.isfile(meta_path) or not chunk_file or index < 0:
        return jsonify({'error': 'invalid_chunk'}), 400
    meta = _load_chunk_meta(meta_path) or {}
    total_size = int(meta.get('size') or 0)
    received = int(meta.get('received') or 0)
    total_chunks = int(meta.get('chunks') or 0)
    received_chunks = {int(value) for value in (meta.get('received_chunks') or [])}
    if meta.get('state') != 'receiving' or index >= total_chunks:
        return jsonify({'error': 'invalid_chunk_index'}), 409
    expected = min(_ACCOUNT_IMPORT_CHUNK_BYTES, total_size - (index * _ACCOUNT_IMPORT_CHUNK_BYTES))
    part_path = os.path.join(upload_dir, f'chunk_{index:08d}.part')
    if index in received_chunks and os.path.isfile(part_path) and os.path.getsize(part_path) == expected:
        return jsonify({'received': received, 'total': total_size, 'index': index, 'duplicate': True})
    payload = chunk_file.read(_ACCOUNT_IMPORT_CHUNK_BYTES + 1)
    if len(payload) != expected:
        return jsonify({'error': 'invalid_chunk_size'}), 400
    with _chunk_upload_lock(upload_dir):
        meta = _load_chunk_meta(meta_path) or {}
        received_chunks = {int(value) for value in (meta.get('received_chunks') or [])}
        if index in received_chunks and os.path.isfile(part_path) and os.path.getsize(part_path) == expected:
            return jsonify({'received': int(meta.get('received') or 0), 'total': total_size, 'index': index, 'duplicate': True})
        if index in received_chunks:
            return jsonify({'error': 'chunk_already_received'}), 409
        with open(part_path, 'wb') as part_file:
            part_file.write(payload)
        try:
            os.chmod(part_path, 0o600)
        except OSError:
            pass
        received_chunks.add(index)
        meta.update(received=int(meta.get('received') or 0) + len(payload), received_chunks=sorted(received_chunks), updated=int(time.time()))
        _save_chunk_meta(meta_path, meta)
    return jsonify({'received': meta['received'], 'total': total_size, 'index': index})

@app.route('/api/account/import/upload/<upload_id>/complete', methods=['POST'])
@login_required
def complete_account_import_upload(upload_id):
    if not _is_valid_account_import_upload_id(upload_id):
        return jsonify({'error': 'invalid_upload_id'}), 400
    upload_dir = _account_import_upload_dir(current_user.id, upload_id)
    meta_path = os.path.join(upload_dir, 'meta.json') if upload_dir else None
    part_path = os.path.join(upload_dir, 'data.part') if upload_dir else None
    meta = _load_chunk_meta(meta_path) if meta_path else None
    if not meta:
        return jsonify({'error': 'upload_not_found'}), 404
    total_size = int(meta.get('size') or 0)
    total_chunks = int(meta.get('chunks') or 0)
    received_chunks = {int(value) for value in (meta.get('received_chunks') or [])}
    if len(received_chunks) != total_chunks or received_chunks != set(range(total_chunks)) or int(meta.get('received') or 0) != total_size:
        return jsonify({'error': 'upload_incomplete'}), 400
    for index in range(total_chunks):
        chunk_path = os.path.join(upload_dir, f'chunk_{index:08d}.part')
        expected = min(_ACCOUNT_IMPORT_CHUNK_BYTES, total_size - (index * _ACCOUNT_IMPORT_CHUNK_BYTES))
        if not os.path.isfile(chunk_path) or os.path.getsize(chunk_path) != expected:
            return jsonify({'error': 'upload_incomplete'}), 400
    temp_path = f'{part_path}.tmp'
    try:
        with open(temp_path, 'wb') as output:
            for index in range(total_chunks):
                chunk_path = os.path.join(upload_dir, f'chunk_{index:08d}.part')
                with open(chunk_path, 'rb') as chunk_input:
                    shutil.copyfileobj(chunk_input, output, length=1024 * 1024)
    except Exception:
        try:
            if os.path.lexists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise
    os.replace(temp_path, part_path)
    try:
        os.chmod(part_path, 0o600)
    except OSError:
        pass
    if os.path.getsize(part_path) != total_size:
        return jsonify({'error': 'upload_incomplete'}), 400
    meta.update(state='ready', updated=int(time.time()))
    _save_chunk_meta(meta_path, meta)
    return jsonify({'upload_id': upload_id, 'size': meta['size'], 'status': 'ready'})

@app.route('/api/account/import/upload/<upload_id>', methods=['DELETE'])
@login_required
def cancel_account_import_upload(upload_id):
    if not _is_valid_account_import_upload_id(upload_id):
        return jsonify({'error': 'invalid_upload_id'}), 400
    upload_dir = _account_import_upload_dir(current_user.id, upload_id)
    if upload_dir:
        _secure_delete_tree(upload_dir)
    return jsonify({'status': 'cancelled'})


@app.route('/api/account/import', methods=['POST'])
@login_required
def import_account_data():
    upload_file = request.files.get("file")
    body = request.get_json(silent=True) if request.is_json else {}
    body = body if isinstance(body, dict) else {}
    upload_id = str(request.form.get("upload_id") or body.get("upload_id") or "").strip()
    categories = _coerce_account_import_categories(request.form.get("categories", body.get("categories", "")))
    # When true, files imported for the CURRENT user are written back to their
    # original rel_path (overwriting the existing entry) instead of being copied
    # to a fresh "import-<token>" path.  This is how an export is used to restore
    # local files that have become unreadable (e.g. after an encryption-key loss).
    restore_inplace = (
        request.form.get("restore_inplace") in ("1", "true", "on")
        or body.get("restore_inplace") is True
    )
    # When falsy and the "settings" category is selected, the import pauses after
    # the upload so the client can show which settings would change and ask for a
    # confirmation before anything is written.  True bypasses that confirmation.
    confirm_settings = (
        request.form.get("confirm_settings") in ("1", "true", "on")
        or body.get("confirm_settings") is True
    )
    job_id = str(request.form.get("job_id") or body.get("job_id") or secrets.token_hex(16)).lower()
    if not _valid_account_transfer_job_id(job_id):
        return jsonify({'error': 'invalid_job_id'}), 400
    if not rate_limit(f"rl:account_import:user:{current_user.id}", 6, 3600):
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "インポート回数の上限に達しました")
        return jsonify({'error': 'rate_limit'}), 429
    import_upload_dir = None
    import_stream = upload_file.stream if upload_file and upload_file.filename else None
    if upload_id:
        if not _is_valid_account_import_upload_id(upload_id):
            return jsonify({'error': 'invalid_upload_id'}), 400
        import_upload_dir = _account_import_upload_dir(current_user.id, upload_id)
        import_stream = os.path.join(import_upload_dir, 'data.part') if import_upload_dir else None
        if not import_stream or not os.path.isfile(import_stream):
            _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "アップロードデータがありません")
            return jsonify({'error': 'upload_not_found'}), 404
        import_stream = open(import_stream, 'rb')
    if not import_stream:
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "ZIPファイルがありません")
        return jsonify({'error': 'file_required'}), 400
    if not categories:
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "インポート対象が選択されていません")
        return jsonify({'error': 'categories_required'}), 400

    created_paths = []
    try:
        _account_transfer_checkpoint(current_user.id, job_id, 36, "validating", "ZIPを検証しています")
        with zipfile.ZipFile(import_stream, "r") as archive:
            members = archive.infolist()
            if len(members) > 10_000:
                raise ValueError("too_many_archive_entries")
            by_name = {item.filename: item for item in members}
            manifest_info = by_name.get("account_data.json")
            if not manifest_info or manifest_info.file_size > 128 * 1024 * 1024:
                raise ValueError("invalid_manifest")
            manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
            if not isinstance(manifest, dict) or manifest.get("format") != ACCOUNT_EXPORT_FORMAT:
                raise ValueError("unsupported_export_format")
            if int(manifest.get("format_version") or 0) != ACCOUNT_EXPORT_VERSION:
                raise ValueError("unsupported_export_version")
            data = manifest.get("data")
            if not isinstance(data, dict):
                raise ValueError("invalid_manifest")
            _account_transfer_checkpoint(current_user.id, job_id, 38, "validating", "データ構成を確認しています")

            if "settings" in categories and not confirm_settings:
                settings_changes = _account_import_settings_changes(current_user, data.get("settings") or {})
                if settings_changes:
                    _set_account_transfer_status(
                        current_user.id, job_id, "needs_settings_confirmation", 55, "importing_settings",
                        "設定のインポート内容を確認してください",
                        settings_changes=settings_changes,
                    )
                    if hasattr(import_stream, 'close'):
                        import_stream.close()
                    response = jsonify({
                        "status": "settings_confirmation",
                        "error": "settings_confirmation_required",
                        "message": "設定のインポート内容を確認してください",
                        "settings_changes": settings_changes,
                    })
                    response.headers["Cache-Control"] = "no-store"
                    return response

            # Pre-load per-user dedupe lookups so re-importing the same source
            # data is detected and skipped instead of creating duplicates.
            existing_thread_pids = set()
            existing_thread_sigs = set()
            existing_gem_sigs = set()
            existing_gem_uuids = set()
            existing_feedback_sigs = set()
            existing_metric_sigs = set()
            existing_trace_sigs = set()
            existing_file_hashes = {}
            duplicates = {category: 0 for category in ACCOUNT_IMPORT_CATEGORIES}
            if "chats" in categories:
                for (pid,) in db.session.query(Thread.public_id).filter(
                        Thread.user_id == current_user.id, Thread.public_id.isnot(None)).all():
                    existing_thread_pids.add(pid)
                for (sig,) in db.session.query(Thread.import_signature).filter(
                        Thread.user_id == current_user.id, Thread.import_signature.isnot(None)).all():
                    existing_thread_sigs.add(sig)
            if "gems" in categories:
                for (uid,) in db.session.query(Gem.uuid).filter(Gem.user_id == current_user.id).all():
                    existing_gem_uuids.add(uid)
                for (sig,) in db.session.query(Gem.import_signature).filter(
                        Gem.user_id == current_user.id, Gem.import_signature.isnot(None)).all():
                    existing_gem_sigs.add(sig)
            if "files" in categories:
                for cache in FileCache.query.filter_by(user_id=current_user.id).all():
                    sig = cache.import_signature or ""
                    if sig.startswith("sha256:"):
                        existing_file_hashes.setdefault(sig[len("sha256:"):], cache.rel_path)
            if "feedback" in categories:
                for (sig,) in db.session.query(Feedback.import_signature).filter(
                        Feedback.user_id == current_user.id, Feedback.import_signature.isnot(None)).all():
                    existing_feedback_sigs.add(sig)
            if "diagnostics" in categories:
                for (sig,) in db.session.query(FirstTokenLatencyMetric.import_signature).filter(
                        FirstTokenLatencyMetric.user_id == current_user.id,
                        FirstTokenLatencyMetric.import_signature.isnot(None)).all():
                    existing_metric_sigs.add(sig)
                for (sig,) in db.session.query(ChatLatencyTrace.import_signature).filter(
                        ChatLatencyTrace.user_id == current_user.id,
                        ChatLatencyTrace.import_signature.isnot(None)).all():
                    existing_trace_sigs.add(sig)

            file_map = {}
            imported_files = []
            if "files" in categories:
                file_items = data.get("files") or []
                if not isinstance(file_items, list) or len(file_items) > 10_000:
                    raise ValueError("invalid_files")
                selected_files = _coerce_import_selected_files(
                    request.form.get("selected_files", body.get("selected_files", ""))
                )
                file_entries = []
                total_file_bytes = 0
                total_stored_bytes = 0
                for item_index, item in enumerate(file_items, start=1):
                    if item_index == 1 or item_index % 25 == 0:
                        progress = 38 + int(4 * item_index / max(1, len(file_items)))
                        _account_transfer_checkpoint(
                            current_user.id, job_id, progress, "validating_files",
                            f"ファイル情報を確認しています（{item_index}/{len(file_items)}）",
                        )
                    if not isinstance(item, dict):
                        raise ValueError("invalid_files")
                    archive_path = str(item.get("archive_path") or "")
                    entry = by_name.get(archive_path)
                    if not entry or not archive_path.startswith("files/") or entry.is_dir():
                        raise ValueError("missing_archive_file")
                    if entry.file_size < 0 or entry.file_size > (app.config.get("MAX_CONTENT_LENGTH") or 512 * 1024 * 1024):
                        raise ValueError("archive_file_too_large")
                    if selected_files is not None and archive_path not in selected_files:
                        continue
                    file_entries.append((item, archive_path, entry))
                    total_file_bytes += entry.file_size
                    total_stored_bytes += (
                        _fernet_encrypted_size(entry.file_size)
                        if current_user.enable_e2ee else entry.file_size
                    )
                capacity_ok, used, limit = _check_storage_capacity(current_user, total_file_bytes)
                stored_ok, _, _ = _check_storage_capacity(current_user, total_stored_bytes)
                if not capacity_ok or not stored_ok:
                    # Ask the client to choose which files to import instead of
                    # failing the whole import.  The upload is kept on disk so the
                    # follow-up import with selected_files can reuse it.  The
                    # selection data is also recorded in the transfer status so the
                    # client can recover the picker even when this (potentially
                    # large) response body is lost on a flaky connection.
                    selection_payload = _account_storage_limit_payload(file_items, used, limit)
                    _set_account_transfer_status(
                        current_user.id, job_id, "needs_selection", 42, "validating_files",
                        selection_payload["message"],
                        files=selection_payload["files"],
                        used_bytes=selection_payload["used_bytes"],
                        limit_bytes=selection_payload["limit_bytes"],
                        available_bytes=selection_payload["available_bytes"],
                    )
                    if hasattr(import_stream, 'close'):
                        import_stream.close()
                    return jsonify(selection_payload), 409
                for file_index, (item, archive_path, entry) in enumerate(file_entries, start=1):
                    progress = 43 + int(12 * (file_index - 1) / max(1, len(file_entries)))
                    _account_transfer_checkpoint(
                        current_user.id, job_id, progress, "reading_files",
                        f"ファイルを読み込んでいます（{file_index}/{len(file_entries)}）",
                    )
                    raw = archive.read(entry)
                    expected_size_raw = item.get("size_bytes")
                    expected_size = int(expected_size_raw) if expected_size_raw is not None else -1
                    expected_hash = str(item.get("sha256") or "")
                    if len(raw) != expected_size or not secrets.compare_digest(hashlib.sha256(raw).hexdigest(), expected_hash):
                        raise ValueError("file_integrity_error")
                    old_rel = _normalize_upload_ref(item.get("rel_path"))
                    if not old_rel:
                        raise ValueError("invalid_file_path")
                    # In-place restore only applies to the importing user's own
                    # files, so a foreign export can never overwrite local data.
                    target_inplace = restore_inplace and old_rel.startswith(str(current_user.id) + "/")
                    existing_rel = existing_file_hashes.get(expected_hash)
                    if existing_rel and not _get_file_disk_info(existing_rel).get("exists"):
                        existing_rel = None
                    if not target_inplace and existing_rel:
                        # The exact same content is already stored for this user
                        # (from a previous import).  Reuse it instead of copying
                        # the file again so repeat imports never duplicate files.
                        file_map[old_rel] = existing_rel
                        duplicates["files"] += 1
                        del raw
                        continue
                    if target_inplace:
                        new_rel = old_rel
                    else:
                        new_rel = _unique_import_file_rel_path(current_user.id, old_rel)
                    disk_data = encrypt_bytes(raw) if current_user.enable_e2ee else raw
                    destination = os.path.join(app.config["UPLOAD_FOLDER"], new_rel)
                    if not _path_is_within(app.config["UPLOAD_FOLDER"], destination):
                        raise ValueError("invalid_file_path")
                    os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
                    disk_destination = destination + ".enc" if current_user.enable_e2ee else destination
                    # Overwrite the existing (possibly broken) entry on restore.
                    open_mode = "wb" if target_inplace else "xb"
                    handle = open(disk_destination, open_mode)
                    created_paths.append(disk_destination)
                    with handle:
                        handle.write(disk_data)
                    # Restore the original file mtime (recorded in the export) so
                    # the library's "newest first" ordering is preserved.
                    if item.get("mtime") is not None:
                        try:
                            mt = int(item["mtime"])
                            if mt > 0:
                                os.utime(disk_destination, (mt, mt))
                        except Exception:
                            pass
                    imported_files.append((old_rel, new_rel, item.get("display_name"), expected_hash))
                    file_map[old_rel] = new_rel
                    del raw, disk_data

            gem_uuid_map = {}
            counts = {category: 0 for category in ACCOUNT_IMPORT_CATEGORIES}
            if "settings" in categories:
                _account_transfer_checkpoint(current_user.id, job_id, 57, "importing_settings", "設定を取り込んでいます")
                _apply_imported_account_settings(current_user, data.get("settings") or {})
                counts["settings"] = 1
            if "api_credentials" in categories:
                _account_transfer_checkpoint(current_user.id, job_id, 59, "importing_credentials", "認証情報を取り込んでいます")
                _apply_imported_account_secrets(current_user, data.get("api_credentials") or {})
                counts["api_credentials"] = 1

            if "gems" in categories:
                gem_items = data.get("gems") or []
                if not isinstance(gem_items, list) or len(gem_items) > 10_000:
                    raise ValueError("invalid_gems")
                import uuid as _uuid_import
                for item_index, item in enumerate(gem_items, start=1):
                    if item_index == 1 or item_index % 25 == 0:
                        progress = 60 + int(6 * item_index / max(1, len(gem_items)))
                        _account_transfer_checkpoint(
                            current_user.id, job_id, progress, "importing_gems",
                            f"Gemを取り込んでいます（{item_index}/{len(gem_items)}）",
                        )
                    payload = _normalize_gem_payload(item)
                    old_uuid = str(item.get("uuid") or "")
                    gem_sig = f"gem:{old_uuid}" if old_uuid else None
                    if gem_sig and (gem_sig in existing_gem_sigs or old_uuid in existing_gem_uuids):
                        duplicates["gems"] += 1
                        continue
                    new_uuid = str(_uuid_import.uuid4())
                    gem = Gem(
                        uuid=new_uuid,
                        user_id=current_user.id,
                        created_at=_parse_portable_datetime(item.get("created_at")) or datetime.utcnow(),
                        import_signature=gem_sig,
                        **payload,
                    )
                    db.session.add(gem)
                    if old_uuid:
                        gem_uuid_map[old_uuid] = new_uuid
                        existing_gem_uuids.add(old_uuid)
                    if gem_sig:
                        existing_gem_sigs.add(gem_sig)
                    counts["gems"] += 1

            if "files" in categories:
                for file_index, (old_rel, new_rel, display_name, file_hash) in enumerate(imported_files, start=1):
                    progress = 67 + int(11 * (file_index - 1) / max(1, len(imported_files)))
                    _account_transfer_checkpoint(
                        current_user.id, job_id, progress, "saving_files",
                        f"ファイル情報を登録しています（{file_index}/{len(imported_files)}）",
                    )
                    clean_label = _sanitize_file_display_name(display_name)
                    if clean_label:
                        _upsert_file_cache(
                            current_user.id, new_rel, "label", file_uri=clean_label,
                            state="ready", last_error=None,
                            import_signature=f"sha256:{file_hash}" if file_hash else None,
                        )
                    counts["files"] += 1

            if "chats" in categories:
                chat_items = data.get("chats") or []
                if not isinstance(chat_items, list) or len(chat_items) > 100_000:
                    raise ValueError("invalid_chats")
                total_messages = 0
                for chat_index, item in enumerate(chat_items, start=1):
                    progress = 79 + int(12 * (chat_index - 1) / max(1, len(chat_items)))
                    _account_transfer_checkpoint(
                        current_user.id, job_id, progress, "importing_chats",
                        f"チャット履歴を取り込んでいます（{chat_index}/{len(chat_items)}）",
                    )
                    if not isinstance(item, dict):
                        raise ValueError("invalid_chats")
                    src_public_id = str(item.get("public_id") or "").strip()
                    thread_sig = f"thread:{src_public_id}" if src_public_id else None
                    if thread_sig and (thread_sig in existing_thread_sigs or src_public_id in existing_thread_pids):
                        duplicates["chats"] += 1
                        continue
                    messages = item.get("messages") or []
                    if not isinstance(messages, list):
                        raise ValueError("invalid_messages")
                    total_messages += len(messages)
                    if total_messages > 1_000_000:
                        raise ValueError("too_many_messages")
                    thread = Thread(
                        user_id=current_user.id,
                        public_id=generate_thread_public_id(),
                        title=_normalize_thread_title(item.get("title")),
                        is_bookmarked=bool(item.get("is_bookmarked")),
                        bookmarked_at=_parse_portable_datetime(item.get("bookmarked_at")),
                        is_temporary=False,
                        custom_instruction=_safe_account_import_text(item.get("custom_instruction"), 500_000) if item.get("custom_instruction") else None,
                        include_global_instruction=bool(item.get("include_global_instruction", True)),
                        last_model=item.get("last_model") if item.get("last_model") in ALL_VALID_MODEL_IDS else None,
                        last_gem_uuid=gem_uuid_map.get(str(item.get("last_gem_uuid") or "")),
                        enable_prompt_caching=bool(item.get("enable_prompt_caching")),
                        prompt_cache_provider=_safe_account_import_text(item.get("prompt_cache_provider"), 32) if item.get("prompt_cache_provider") else None,
                        updated_at=_parse_portable_datetime(item.get("updated_at")) or datetime.utcnow(),
                        import_signature=thread_sig,
                    )
                    db.session.add(thread)
                    db.session.flush()
                    if thread_sig:
                        existing_thread_sigs.add(thread_sig)
                    if src_public_id:
                        existing_thread_pids.add(src_public_id)
                    id_map = {}
                    pending_parents = []
                    for message_index, message_item in enumerate(messages, start=1):
                        if message_index % 100 == 0:
                            _account_transfer_checkpoint(
                                current_user.id, job_id, progress, "importing_chats",
                                f"チャット{chat_index}/{len(chat_items)}・メッセージ{message_index}/{len(messages)}",
                            )
                        if not isinstance(message_item, dict):
                            raise ValueError("invalid_messages")
                        content = _safe_account_import_text(message_item.get("content"), 20_000_000)
                        thought = _safe_account_import_text(message_item.get("thought_data"), 20_000_000) if message_item.get("thought_data") else None
                        encrypted = bool(current_user.enable_e2ee)
                        message = Message(
                            thread_id=thread.id,
                            role=_safe_account_import_text(message_item.get("role"), 20)[:20],
                            content=encrypt_val(content) if encrypted and content else content,
                            model=_safe_account_import_text(message_item.get("model"), 50)[:50] if message_item.get("model") else None,
                            image_url=_rewrite_imported_attachment_value(message_item.get("image_url"), file_map, current_user.id),
                            timestamp=_parse_portable_datetime(message_item.get("timestamp")) or datetime.utcnow(),
                            tokens=max(0, int(message_item.get("tokens") or 0)),
                            tokens_in=max(0, int(message_item.get("tokens_in") or 0)),
                            tokens_out=max(0, int(message_item.get("tokens_out") or 0)),
                            tokens_thought=max(0, int(message_item.get("tokens_thought") or 0)),
                            thought_data=encrypt_val(thought) if encrypted and thought else thought,
                            quote_text=_safe_account_import_text(message_item.get("quote_text"), 20_000_000) if message_item.get("quote_text") else None,
                            is_encrypted=encrypted,
                            thought_signature=_safe_account_import_text(message_item.get("thought_signature"), 20_000_000) if message_item.get("thought_signature") else None,
                            gem_uuid=gem_uuid_map.get(str(message_item.get("gem_uuid") or "")),
                            gem_name=_safe_account_import_text(message_item.get("gem_name"), 100)[:100] if message_item.get("gem_name") else None,
                        )
                        db.session.add(message)
                        db.session.flush()
                        export_id = message_item.get("export_id")
                        if export_id is not None:
                            id_map[str(export_id)] = message.id
                        pending_parents.append((message, message_item.get("parent_export_id")))
                    for message, old_parent_id in pending_parents:
                        if old_parent_id is not None:
                            message.parent_id = id_map.get(str(old_parent_id))
                    counts["chats"] += 1

            if "feedback" in categories:
                _account_transfer_checkpoint(current_user.id, job_id, 93, "importing_feedback", "フィードバックを取り込んでいます")
                feedback_items = data.get("feedback") or []
                if not isinstance(feedback_items, list) or len(feedback_items) > 100_000:
                    raise ValueError("invalid_feedback")
                for item in feedback_items:
                    if not isinstance(item, dict):
                        raise ValueError("invalid_feedback")
                    status = str(item.get("status") or "new")
                    if status not in {"new", "in_review", "replied", "rejected", "resolved"}:
                        status = "new"
                    fb_title = _safe_account_import_text(item.get("title"), 200)[:200]
                    fb_message = _safe_account_import_text(item.get("message"), 500_000)
                    fb_admin_reply = _safe_account_import_text(item.get("admin_reply"), 500_000) if item.get("admin_reply") else None
                    fb_created = _parse_portable_datetime(item.get("created_at")) or datetime.utcnow()
                    fb_sig = _import_signature("feedback", [
                        fb_title, fb_message, status,
                        fb_admin_reply or "", _portable_datetime(fb_created) or "",
                    ])
                    if fb_sig in existing_feedback_sigs:
                        duplicates["feedback"] += 1
                        continue
                    db.session.add(Feedback(
                        user_id=current_user.id,
                        title=fb_title,
                        message=fb_message,
                        status=status,
                        admin_reply=fb_admin_reply,
                        created_at=fb_created,
                        updated_at=_parse_portable_datetime(item.get("updated_at")) or datetime.utcnow(),
                        import_signature=fb_sig,
                    ))
                    existing_feedback_sigs.add(fb_sig)
                    counts["feedback"] += 1

            if "diagnostics" in categories:
                _account_transfer_checkpoint(current_user.id, job_id, 95, "importing_diagnostics", "診断データを取り込んでいます")
                diagnostics = data.get("diagnostics") or {}
                metric_items = diagnostics.get("first_token_metrics") or [] if isinstance(diagnostics, dict) else []
                trace_items = diagnostics.get("chat_latency_traces") or [] if isinstance(diagnostics, dict) else []
                if not isinstance(metric_items, list) or not isinstance(trace_items, list) or len(metric_items) + len(trace_items) > 500_000:
                    raise ValueError("invalid_diagnostics")
                for item in metric_items:
                    if not isinstance(item, dict):
                        continue
                    latency_ms = max(0, int(item.get("latency_ms") or 0))
                    metric_public_id = _safe_account_import_text(item.get("thread_public_id"), 64)[:64] if item.get("thread_public_id") else None
                    metric_model = _safe_account_import_text(item.get("model"), 80)[:80] if item.get("model") else None
                    metric_event = _safe_account_import_text(item.get("first_event_type"), 32)[:32] if item.get("first_event_type") else None
                    metric_client_sent = _parse_portable_datetime(item.get("client_sent_at"))
                    metric_created = _parse_portable_datetime(item.get("created_at")) or datetime.utcnow()
                    m_sig = _import_signature("metric", [
                        metric_public_id or "", metric_model or "", metric_event or "",
                        str(latency_ms), _portable_datetime(metric_client_sent) or "",
                        _portable_datetime(metric_created) or "",
                    ])
                    if m_sig in existing_metric_sigs:
                        duplicates["diagnostics"] += 1
                        continue
                    db.session.add(FirstTokenLatencyMetric(
                        user_id=current_user.id,
                        thread_public_id=metric_public_id,
                        job_id=f"import-{secrets.token_hex(8)}",
                        model=metric_model,
                        first_event_type=metric_event,
                        latency_seconds=max(0.0, float(item.get("latency_seconds") or latency_ms / 1000.0)),
                        latency_ms=latency_ms,
                        client_sent_at=metric_client_sent,
                        created_at=metric_created,
                        import_signature=m_sig,
                    ))
                    existing_metric_sigs.add(m_sig)
                    counts["diagnostics"] += 1
                trace_columns = {column.name: column for column in ChatLatencyTrace.__table__.columns}
                datetime_columns = {name for name, column in trace_columns.items() if isinstance(column.type, db.DateTime)}
                for item in trace_items:
                    if not isinstance(item, dict):
                        continue
                    kwargs = {"user_id": current_user.id, "job_id": f"import-{secrets.token_hex(12)}"}
                    for field, column in trace_columns.items():
                        if field in {"id", "user_id", "job_id"} or field not in item:
                            continue
                        value = item.get(field)
                        if field in datetime_columns:
                            value = _parse_portable_datetime(value)
                        elif isinstance(column.type, db.Integer):
                            value = int(value) if value is not None else None
                        elif value is not None:
                            value = _safe_account_import_text(value, getattr(column.type, "length", None) or 4096)
                        kwargs[field] = value
                    trace_sig_parts = []
                    for field, column in trace_columns.items():
                        if field in {"id", "user_id", "job_id", "updated_at"} or field not in item:
                            continue
                        value = item.get(field)
                        if field in datetime_columns:
                            value = _portable_datetime(_parse_portable_datetime(value)) or ""
                        trace_sig_parts.append(f"{field}={value}")
                    t_sig = _import_signature("trace", trace_sig_parts)
                    if t_sig in existing_trace_sigs:
                        duplicates["diagnostics"] += 1
                        continue
                    kwargs["import_signature"] = t_sig
                    db.session.add(ChatLatencyTrace(**kwargs))
                    existing_trace_sigs.add(t_sig)
                    counts["diagnostics"] += 1

            _account_transfer_checkpoint(current_user.id, job_id, 98, "finalizing", "変更を確定しています")
            db.session.commit()
            _set_account_transfer_status(current_user.id, job_id, "completed", 100, "completed", "インポートが完了しました")
            response = jsonify({
                "status": "ok",
                "imported": counts,
                "duplicates": duplicates,
                "selected": sorted(categories),
            })
            response.headers["Cache-Control"] = "no-store"
            if hasattr(import_stream, 'close'):
                import_stream.close()
            if import_upload_dir:
                _secure_delete_tree(import_upload_dir)
            return response
    except AccountTransferCancelled:
        db.session.rollback()
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        for path in created_paths:
            secure_delete(path)
        _set_account_transfer_status(current_user.id, job_id, "cancelled", 0, "cancelled", "インポートをキャンセルしました")
        return jsonify({'error': 'cancelled'}), 409
    except zipfile.BadZipFile:
        db.session.rollback()
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "ZIP形式が正しくありません")
        return jsonify({'error': 'invalid_zip'}), 400
    except StorageLimitError:
        db.session.rollback()
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        for path in created_paths:
            secure_delete(path)
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "ストレージ上限を超えるためインポートできません")
        return jsonify({'error': 'storage_limit_exceeded'}), 413
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        db.session.rollback()
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        for path in created_paths:
            secure_delete(path)
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "インポートデータが正しくありません")
        return jsonify({'error': str(exc) or 'invalid_import'}), 400
    except Exception:
        db.session.rollback()
        if hasattr(import_stream, 'close'):
            import_stream.close()
        if import_upload_dir:
            _secure_delete_tree(import_upload_dir)
        for path in created_paths:
            secure_delete(path)
        logger.exception("Account import failed for user %s", current_user.id)
        _set_account_transfer_status(current_user.id, job_id, "failed", 0, "failed", "インポートに失敗しました")
        return jsonify({'error': 'import_failed'}), 500


def _import_signature(prefix, parts):
    """Deterministic per-user identity for an imported record.

    ``prefix`` is a short category tag (e.g. ``thread`` / ``gem`` / ``feedback``)
    and ``parts`` are the stable fields that define the record.  The same source
    record always yields the same signature, so re-importing it can be detected
    even though local identifiers (public_id, uuid, job_id, ...) are regenerated.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part if part is not None else "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
    return f"{prefix}:{digest.hexdigest()}"


def _decrypt_dedupe_value(value, is_encrypted):
    """Return a comparable plaintext value for dedupe signatures.

    Values that cannot be decrypted (e.g. after an encryption-key change) are
    normalised to a fixed placeholder so identical unreadable records still hash
    the same way.

    This deliberately bypasses the shared decrypt cache (``decrypt_val`` keeps up
    to 4096 plaintext payloads in memory).  A full-account dedupe scan decrypts
    every message, so caching them would exhaust memory on small servers.
    """
    if value is None:
        return ""
    if not is_encrypted:
        return value
    try:
        return _ring_decrypt_bytes(value.encode()).decode()
    except Exception:
        return "\x00unreadable"


def _thread_dedupe_signature(thread):
    """Content signature of a thread, or None when the thread has no messages.

    Empty threads are never treated as duplicates: users commonly have several
    unused "New Chat" rows that only differ by their id.
    """
    digest = hashlib.sha256()
    digest.update(str(thread.title or "").encode("utf-8", "replace"))
    digest.update(b"|")
    digest.update(b"1" if thread.is_bookmarked else b"0")
    digest.update(b"|")
    message_count = 0
    message_query = (
        Message.query
        .with_entities(Message.id, Message.role, Message.content, Message.model,
                       Message.timestamp, Message.gem_uuid, Message.thought_data,
                       Message.is_encrypted)
        .filter(Message.thread_id == thread.id)
        .order_by(Message.id.asc())
        .yield_per(50)
    )
    for message in message_query:
        message_count += 1
        content = _decrypt_dedupe_value(message.content, bool(message.is_encrypted))
        thought = _decrypt_dedupe_value(message.thought_data, bool(message.is_encrypted))
        digest.update(str(message.role or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(content or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(message.model or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update((message.timestamp.isoformat() if message.timestamp else "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(message.gem_uuid or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(thought or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
    if message_count == 0:
        return None
    return digest.hexdigest()


def _gem_dedupe_signature(gem):
    parts = [
        gem.name or "", gem.description or "", gem.instruction or "",
        gem.fixed_prompts_json or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _file_plaintext_sha256(rel_path):
    """Compute the sha256 of a stored file's plaintext content (decrypting if needed)."""
    info = _get_file_disk_info(rel_path)
    if not info.get("exists"):
        return None
    digest = hashlib.sha256()
    try:
        if info.get("is_encrypted"):
            with open(info["disk_path"], "rb") as source:
                token = source.read()
            try:
                data = decrypt_bytes(token)
            finally:
                del token
            digest.update(data)
        else:
            with open(info["disk_path"], "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
    except Exception:
        return None
    return digest.hexdigest()


def _feedback_dedupe_signature(feedback):
    parts = [
        feedback.title or "", feedback.message or "", feedback.status or "",
        feedback.admin_reply or "", _portable_datetime(feedback.created_at) or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _metric_dedupe_signature(metric):
    parts = [
        metric.thread_public_id or "", metric.model or "", metric.first_event_type or "",
        str(int(metric.latency_ms or 0)),
        _portable_datetime(metric.client_sent_at) or "", _portable_datetime(metric.created_at) or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _trace_dedupe_signature(trace):
    fields = []
    for column in ChatLatencyTrace.__table__.columns:
        name = column.name
        if name in {"id", "user_id", "job_id", "import_signature", "updated_at"}:
            continue
        value = getattr(trace, name)
        if isinstance(value, datetime):
            value = _portable_datetime(value)
        fields.append(f"{name}={value}")
    return hashlib.sha256("\x1f".join(fields).encode("utf-8", "replace")).hexdigest()


def _collect_referenced_file_paths(user_id, exclude_thread_ids=None):
    """Return the set of local upload refs referenced by the user's messages.

    ``exclude_thread_ids`` (an iterable of thread ids) is used to ignore messages
    that belong to duplicate threads scheduled for deletion, so files referenced
    only by those copies become eligible for removal.
    """
    excluded = set(exclude_thread_ids or [])
    referenced = set()
    thread_ids = [tid for (tid,) in db.session.query(Thread.id).filter_by(user_id=user_id).all()]
    for start in range(0, len(thread_ids), 200):
        chunk = [tid for tid in thread_ids[start:start + 200] if tid not in excluded]
        if not chunk:
            continue
        for (image_url,) in db.session.query(Message.image_url).filter(Message.thread_id.in_(chunk)).all():
            if not image_url:
                continue
            try:
                parsed = json.loads(image_url) if isinstance(image_url, str) else image_url
            except Exception:
                parsed = image_url

            def walk(item):
                if isinstance(item, list):
                    for child in item:
                        walk(child)
                elif isinstance(item, dict):
                    matched = False
                    for key in ("path", "filepath", "file", "url", "name"):
                        if key in item and item[key]:
                            walk(item[key])
                            matched = True
                            break
                    if not matched:
                        for value in item.values():
                            walk(value)
                elif isinstance(item, str) and item:
                    norm = _normalize_upload_ref(item)
                    if norm:
                        referenced.add(norm)

            walk(parsed)
    return referenced


def _dedupe_plan_for_user(user_id):
    """Compute which duplicate records should be removed for a user.

    Returns a dict with:
      - ``removed``: list of record identifiers to delete per category
      - ``kept_referenced_files``: number of content-duplicate files that must be
        kept because they are still referenced from chat messages
    """
    removed = {"chats": [], "gems": [], "files": [], "feedback": [], "metrics": [], "traces": []}
    kept_referenced_files = 0

    thread_groups = {}
    for thread in Thread.query.filter_by(user_id=user_id).order_by(Thread.id.asc()).yield_per(200):
        signature = _thread_dedupe_signature(thread)
        if signature is not None:
            thread_groups.setdefault(signature, []).append(thread.id)
    for ids in thread_groups.values():
        if len(ids) > 1:
            removed["chats"].extend(ids[1:])

    gem_groups = {}
    for gem in Gem.query.filter_by(user_id=user_id).order_by(Gem.id.asc()).yield_per(200):
        gem_groups.setdefault(_gem_dedupe_signature(gem), []).append(gem.id)
    for ids in gem_groups.values():
        if len(ids) > 1:
            removed["gems"].extend(ids[1:])

    feedback_groups = {}
    for feedback in Feedback.query.filter_by(user_id=user_id).order_by(Feedback.id.asc()).yield_per(200):
        feedback_groups.setdefault(_feedback_dedupe_signature(feedback), []).append(feedback.id)
    for ids in feedback_groups.values():
        if len(ids) > 1:
            removed["feedback"].extend(ids[1:])

    metric_groups = {}
    for metric in FirstTokenLatencyMetric.query.filter_by(user_id=user_id).order_by(FirstTokenLatencyMetric.id.asc()).yield_per(500):
        metric_groups.setdefault(_metric_dedupe_signature(metric), []).append(metric.id)
    for ids in metric_groups.values():
        if len(ids) > 1:
            removed["metrics"].extend(ids[1:])

    trace_groups = {}
    for trace in ChatLatencyTrace.query.filter_by(user_id=user_id).order_by(ChatLatencyTrace.id.asc()).yield_per(500):
        trace_groups.setdefault(_trace_dedupe_signature(trace), []).append(trace.id)
    for ids in trace_groups.values():
        if len(ids) > 1:
            removed["traces"].extend(ids[1:])

    cache_order = {}
    for cache in FileCache.query.filter_by(user_id=user_id).order_by(FileCache.id.asc()).all():
        cache_order.setdefault(cache.rel_path, cache.id)
    file_hashes = {}
    for row in _account_file_rows(user_id):
        rel_path = row.get("rel_path")
        sha = _file_plaintext_sha256(rel_path)
        if not sha:
            continue
        file_hashes.setdefault(sha, []).append(rel_path)
    referenced = _collect_referenced_file_paths(user_id, exclude_thread_ids=removed["chats"])
    for rels in file_hashes.values():
        if len(rels) <= 1:
            continue
        ordered = sorted(rels, key=lambda rel: (cache_order.get(rel, 2 ** 62), rel))
        keep = ordered[0]
        for rel in ordered[1:]:
            if rel in referenced:
                kept_referenced_files += 1
                continue
            removed["files"].append(rel)

    return {"removed": removed, "kept_referenced_files": kept_referenced_files}


def _dedupe_count_payload(plan):
    removed = plan["removed"]
    return {
        "chats": len(removed["chats"]),
        "gems": len(removed["gems"]),
        "files": len(removed["files"]),
        "feedback": len(removed["feedback"]),
        "diagnostics": len(removed["metrics"]) + len(removed["traces"]),
        "kept_referenced_files": plan["kept_referenced_files"],
        "total": (
            len(removed["chats"]) + len(removed["gems"]) + len(removed["files"])
            + len(removed["feedback"]) + len(removed["metrics"]) + len(removed["traces"])
        ),
    }


@app.route('/api/account/dedupe/preview', methods=['POST'])
@login_required
def account_dedupe_preview():
    if not rate_limit(f"rl:account_dedupe:user:{current_user.id}", 10, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    plan = _dedupe_plan_for_user(current_user.id)
    counts = _dedupe_count_payload(plan)
    response = jsonify({
        "status": "ok",
        "duplicates": {key: counts[key] for key in ("chats", "gems", "files", "feedback", "diagnostics")},
        "kept_referenced_files": counts["kept_referenced_files"],
        "total": counts["total"],
        "has_duplicates": counts["total"] > 0,
    })
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route('/api/account/dedupe/execute', methods=['POST'])
@login_required
def account_dedupe_execute():
    if not rate_limit(f"rl:account_dedupe:user:{current_user.id}", 10, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    plan = _dedupe_plan_for_user(current_user.id)
    removed = plan["removed"]
    try:
        if removed["chats"]:
            for thread_id in removed["chats"]:
                thread = db.session.get(Thread, thread_id)
                if thread:
                    db.session.delete(thread)
        if removed["gems"]:
            gem_groups = {}
            for gem in Gem.query.filter_by(user_id=current_user.id).order_by(Gem.id.asc()).all():
                gem_groups.setdefault(_gem_dedupe_signature(gem), []).append(gem)
            gem_twin = {}
            for group in gem_groups.values():
                if len(group) > 1:
                    keep = group[0]
                    for dup in group[1:]:
                        gem_twin[dup.id] = keep.uuid
            thread_ids = [tid for (tid,) in db.session.query(Thread.id).filter_by(user_id=current_user.id).all()]
            for gem_id in removed["gems"]:
                gem = db.session.get(Gem, gem_id)
                if not gem:
                    continue
                # Keep references (thread last_gem_uuid / message gem_uuid)
                # pointing at a live gem when the removed duplicate had the
                # same content.
                keep_uuid = gem_twin.get(gem.id)
                if keep_uuid and keep_uuid != gem.uuid:
                    for start in range(0, len(thread_ids), 200):
                        chunk = thread_ids[start:start + 200]
                        Message.query.filter(Message.thread_id.in_(chunk), Message.gem_uuid == gem.uuid).update(
                            {"gem_uuid": keep_uuid}, synchronize_session=False
                        )
                    Thread.query.filter_by(user_id=current_user.id, last_gem_uuid=gem.uuid).update(
                        {"last_gem_uuid": keep_uuid}, synchronize_session=False
                    )
                db.session.delete(gem)
        if removed["files"]:
            for rel_path in removed["files"]:
                info = _get_file_disk_info(rel_path)
                if info.get("exists"):
                    secure_delete(info["disk_path"])
                _delete_file_cache_for_path(current_user.id, rel_path)
        if removed["feedback"]:
            for feedback_id in removed["feedback"]:
                feedback = db.session.get(Feedback, feedback_id)
                if feedback:
                    db.session.delete(feedback)
        if removed["metrics"]:
            for start in range(0, len(removed["metrics"]), 500):
                chunk = removed["metrics"][start:start + 500]
                FirstTokenLatencyMetric.query.filter(
                    FirstTokenLatencyMetric.id.in_(chunk)
                ).delete(synchronize_session=False)
        if removed["traces"]:
            for start in range(0, len(removed["traces"]), 500):
                chunk = removed["traces"][start:start + 500]
                ChatLatencyTrace.query.filter(
                    ChatLatencyTrace.id.in_(chunk)
                ).delete(synchronize_session=False)
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Account dedupe failed for user %s", current_user.id)
        return jsonify({'error': 'dedupe_failed'}), 500
    counts = _dedupe_count_payload(plan)
    response = jsonify({
        "status": "ok",
        "removed": {key: counts[key] for key in ("chats", "gems", "files", "feedback", "diagnostics")},
        "kept_referenced_files": counts["kept_referenced_files"],
        "total": counts["total"],
    })
    response.headers["Cache-Control"] = "no-store"
    return response

