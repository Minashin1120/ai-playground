@app.before_request
def _clear_stale_flask_login_flash():
    """Drop any leftover Flask-Login login flash from before it was disabled.

    Sessions created before the fix above may still carry
    "Please log in to access this page." in _flashes.  Without this, those
    users would see the misleading toast once more on their next page load."""
    if request.endpoint == 'static':
        return
    flashes = session.get("_flashes") or []
    if not flashes:
        return
    cleaned = [
        f for f in flashes
        if not (isinstance(f, (list, tuple)) and len(f) >= 2 and f[1] == "Please log in to access this page.")
    ]
    if len(cleaned) != len(flashes):
        session["_flashes"] = cleaned

# Settings-save result messages.  /api/settings no longer flashes them (the
# result is returned in the JSON response and the client shows its own toast),
# but sessions created while the old code was live may still carry these stale
# flashes.  They are purged on every request so they can never be rendered on a
# page load as a misleading "設定を保存しました" toast -- e.g. when a user has
# the settings modal open (a client-side history navigation that does not render
# the flash) and then navigates via the URL bar (a full page render that would).
_LEAKY_SETTINGS_FLASHES = {
    "設定を保存しました",
    "暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。",
    "2FAを無効化しました。",
}

@app.before_request
def _purge_stale_settings_flashes():
    """Drop any leftover settings-save flashes from before they were removed.

    Previously this endpoint used flash(), whose messages are only consumed on
    the next full page render.  Sessions created before that fix can still
    carry a stale "設定を保存しました" flash, which would otherwise render on
    the next full page load and claim a save that never happened.  Purging on
    every request (not just inside the settings POST handler) guarantees the
    stale message is gone before any page render can show it.  Unrelated
    flashes, e.g. the bot-unban notice, are kept."""
    if request.endpoint == 'static':
        return
    flashes = session.get("_flashes") or []
    if not flashes:
        return
    cleaned = [
        f for f in flashes
        if not (isinstance(f, (list, tuple)) and len(f) >= 2 and f[1] in _LEAKY_SETTINGS_FLASHES)
    ]
    if len(cleaned) != len(flashes):
        session["_flashes"] = cleaned

@app.before_request
def _apply_per_user_upload_limits():
    endpoint = request.endpoint or ''
    global_limit = app.config.get('MAX_CONTENT_LENGTH')
    if endpoint not in ('upload', 'upload_chunk', 'import_account_data', 'account_import_upload_chunk'):
        endpoint_limit = 4 * 1024 * 1024
        if endpoint in {'transcribe', 'speech_to_speech', 'save_sts_direct'}:
            endpoint_limit = 32 * 1024 * 1024
        elif endpoint == 'speedtest_upload':
            endpoint_limit = 33 * 1024 * 1024
        request.max_content_length = min(global_limit, endpoint_limit) if global_limit else endpoint_limit
        return
    try:
        if endpoint in ('import_account_data', 'account_import_upload_chunk'):
            request.max_content_length = global_limit
        elif current_user.is_authenticated and _is_primary_admin_user(current_user):
            request.max_content_length = min(global_limit or 12 * 1024 * 1024, 12 * 1024 * 1024) if endpoint == 'upload_chunk' else global_limit
        else:
            limit = _get_user_storage_limit_bytes(current_user) if current_user.is_authenticated else None
            if limit:
                if request.content_length and request.content_length > limit:
                    limit_mb = _bytes_to_mb_str(limit)
                    return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
                if request.endpoint == 'upload_chunk':
                    hard_cap = app.config.get('MAX_CONTENT_LENGTH') or limit
                    request.max_content_length = min(hard_cap, limit, 12 * 1024 * 1024)
                else:
                    # Stale partial uploads must not reduce the body limit to one
                    # byte before the upload route gets a chance to reclaim them.
                    _cleanup_stale_chunk_uploads(current_user.id)
                    used = _get_user_storage_usage_bytes(current_user.id)
                    remaining = max(0, limit - used)
                    hard_cap = app.config.get('MAX_CONTENT_LENGTH') or remaining
                    request.max_content_length = min(hard_cap, remaining if remaining > 0 else 1)
            else:
                request.max_content_length = min(app.config.get('MAX_CONTENT_LENGTH') or 12 * 1024 * 1024, 12 * 1024 * 1024) if endpoint == 'upload_chunk' else app.config.get('MAX_CONTENT_LENGTH')
    except Exception:
        request.max_content_length = min(app.config.get('MAX_CONTENT_LENGTH') or 12 * 1024 * 1024, 12 * 1024 * 1024) if endpoint == 'upload_chunk' else app.config.get('MAX_CONTENT_LENGTH')

