# region: account deletion, transfer, and export
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

