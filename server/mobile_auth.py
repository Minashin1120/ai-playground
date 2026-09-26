"""Native client authentication. Loaded before all request hooks; no startup I/O."""

MOBILE_TOKEN_PREFIX = 'aip_android_'
MOBILE_TOKEN_TTL = 30 * 24 * 60 * 60
MOBILE_GRANT_TTL = 600
MOBILE_POLL_INTERVAL = 5
MOBILE_ENDPOINT_METHODS = {
    'mobile_me': {'GET'}, 'mobile_revoke': {'POST'}, 'mobile_preferences': {'GET', 'PUT'},
    # Native sign-in is deliberately separate from the legacy browser pairing
    # endpoints.  The latter remains available as a deprecated fallback.
    'mobile_setup': {'GET', 'PUT'},
    # Passkey / TOTP management for the signed-in native client.
    'mobile_security': {'GET'},
    'mobile_security_totp_setup': {'POST'}, 'mobile_security_totp_enable': {'POST'},
    'mobile_security_totp_disable': {'POST'},
    'mobile_security_passkey_options': {'POST'}, 'mobile_security_passkey_verify': {'POST'},
    'mobile_security_passkey_remove': {'POST'}, 'mobile_security_preferences': {'POST'},
    # In-app account ZIP import during first-run setup reuses the tested Web
    # chunked-upload routes with the native bearer token.
    'start_account_import_upload': {'POST'}, 'account_import_upload_chunk': {'POST'},
    'complete_account_import_upload': {'POST'}, 'cancel_account_import_upload': {'DELETE'},
    'import_account_data': {'POST'},
    'handle_threads': {'GET', 'POST'}, 'handle_thread_item': {'GET', 'DELETE'},
    'update_thread_settings': {'GET', 'PUT'}, 'update_title': {'PUT'},
    'toggle_bookmark': {'POST'},
    # Message delete from the bubble controls (owner-checked in the route, like Web).
    'delete_message': {'DELETE'},
    'chat_stream': {'POST'}, 'chat_stream_resume': {'POST'}, 'stop_chat': {'POST'},
    'temporary_chat_heartbeat': {'POST'}, 'estimate_prompt_tokens_api': {'POST'},
    'upload': {'POST'}, 'upload_init': {'POST'}, 'upload_chunk': {'POST'},
    'upload_complete': {'POST'}, 'serve_file': {'GET', 'HEAD'},
    'serve_file_thumb': {'GET', 'HEAD'}, 'get_storage_usage': {'GET'},
    'get_files_lib': {'GET'},
    'toggle_file_favorite': {'POST'}, 'delete_files_batch': {'POST'},
    'rename_library_file': {'POST'},
    'handle_gems': {'GET', 'POST'}, 'handle_gem_item': {'GET', 'PUT', 'DELETE'},
    'export_thread_pdf': {'GET'},
    'gemini_batch_status_api': {'GET'}, 'list_batch_jobs_api': {'GET'},
    'cancel_batch_job_api': {'POST'}, 'delete_batch_job_api': {'DELETE'},
    'mcp_service.chat_decision': {'POST'},
    'mcp_service.list_servers': {'GET'}, 'mcp_service.update_server': {'PUT'},
    # Unauthenticated custom servers only; Bearer/OAuth secrets stay on Web (ANDROID_ONLY.md).
    'mcp_service.add_custom_server': {'POST'}, 'mcp_service.delete_server': {'DELETE'},
    'mcp_service.test_server': {'POST'}, 'mcp_service.list_server_tools': {'GET'},
    'feedback': {'GET', 'POST'},
    # Native clients use the same authenticated provider sessions as Web for
    # realtime audio/music. These endpoints never return provider API keys.
    # Settings modal parity (routes_mobile_account.py): account, sessions, 2FA and
    # data transfer. Sensitive ones also require a recent re-authentication.
    'mobile_reauth_options': {'POST'}, 'mobile_reauth': {'POST'},
    'mobile_account_credentials': {'POST'}, 'mobile_account_delete': {'POST'},
    'mobile_easy_login': {'POST'},
    'mobile_sessions': {'GET'}, 'mobile_sessions_revoke': {'POST'},
    'mobile_sessions_revoke_others': {'POST'}, 'mobile_sessions_revoke_all': {'POST'},
    'mobile_security_disable_2fa': {'POST'}, 'mobile_security_e2ee': {'POST'},
    'encryption_scan': {'GET'},
    'unlink_google': {'POST'}, 'unlink_minashin': {'POST'},
    'export_account_data': {'POST'}, 'get_latest_account_export': {'GET'},
    'download_account_export': {'GET'}, 'get_account_transfer_status': {'GET'},
    'cancel_account_transfer': {'POST'},
    'account_dedupe_preview': {'POST'}, 'account_dedupe_execute': {'POST'},
    'realtime_start': {'POST'}, 'realtime_stream': {'GET'}, 'realtime_audio': {'POST'},
    'realtime_commit': {'POST'}, 'realtime_cancel': {'POST'}, 'realtime_save': {'POST'},
    'gemini_music_start': {'POST'}, 'gemini_music_stream': {'GET'},
    'gemini_music_command': {'POST'}, 'gemini_music_cancel': {'POST'}, 'gemini_music_save': {'POST'},
}
MOBILE_PUBLIC_ENDPOINTS = {
    'mobile_device', 'mobile_token',
    'mobile_auth_signup', 'mobile_auth_login', 'mobile_auth_google', 'mobile_auth_totp', 'mobile_auth_exchange',
    'mobile_auth_passkey_options', 'mobile_auth_passkey_verify',
    'mobile_auth_2fa_webauthn_options', 'mobile_auth_2fa_webauthn_verify',
}
MOBILE_SETUP_ENDPOINTS = {
    'mobile_setup', 'mobile_security', 'mobile_security_totp_setup', 'mobile_security_totp_enable',
    'mobile_security_totp_disable', 'mobile_security_passkey_options', 'mobile_security_passkey_verify',
    'mobile_security_passkey_remove', 'mobile_security_preferences',
    'start_account_import_upload', 'account_import_upload_chunk', 'complete_account_import_upload',
    'cancel_account_import_upload', 'import_account_data',
}
# Endpoints limited to first-run setup (none: the settings Data tab imports too).
MOBILE_FIRST_RUN_ONLY_ENDPOINTS = set()
# After setup, these need a re-authentication within MOBILE_REAUTH_TTL on this
# device (password, TOTP or passkey; see routes_mobile_account.py). Archives
# contain decrypted provider keys, so creating/downloading/importing them is
# covered too.
MOBILE_REAUTH_TTL = 600
MOBILE_REAUTH_ENDPOINTS = {
    'mobile_account_delete', 'mobile_easy_login', 'mobile_sessions_revoke_all',
    'mobile_security_disable_2fa', 'export_account_data', 'download_account_export',
    'start_account_import_upload', 'import_account_data', 'unlink_google', 'unlink_minashin',
}


def _mobile_reauth_key(session_row):
    return 'mobile:reauth:' + _mobile_digest(session_row.session_id)


def _mobile_recently_reauthenticated():
    row = getattr(g, 'mobile_session', None)
    if row is None:
        return False
    try:
        return bool(redis_conn.get(_mobile_reauth_key(row)))
    except Exception:
        return False


def _mobile_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _mobile_enabled():
    return app.config.get('MOBILE_API_ENABLED', os.getenv('MOBILE_API_ENABLED', '1')) not in (False, '0', 'false', 'off')


def _mobile_error(code, status=400, retry_after=MOBILE_POLL_INTERVAL):
    response = jsonify({'error': code})
    response.status_code = status
    if status == 401:
        response.headers['WWW-Authenticate'] = 'Bearer realm="android"'
    if status == 429 or code == 'slow_down':
        response.headers['Retry-After'] = str(retry_after)
    return response


@login_manager.request_loader
def _load_mobile_user(req):
    # The early guard enforces a cookie-free, explicitly scoped HTTPS request.
    if not getattr(g, 'mobile_bearer_request', False):
        return None
    token = req.headers.get('Authorization', '')[7:]
    sid = 'android:' + _mobile_digest(token)
    row = UserSession.query.filter_by(session_id=sid, is_revoked=False).first()
    if not row or row.created_at + timedelta(seconds=MOBILE_TOKEN_TTL) <= datetime.utcnow():
        return None
    user = db.session.get(User, row.user_id)
    if user is not None:
        g.mobile_session = row
    return user


@app.before_request
def mobile_request_guard():
    endpoint = request.endpoint
    header = request.headers.get('Authorization', '')
    is_mobile_path = request.path.startswith('/api/mobile/') or endpoint == 'mobile_connect'
    is_bearer = header.lower().startswith('bearer ') and (
        header[7:].startswith(MOBILE_TOKEN_PREFIX) or endpoint in MOBILE_ENDPOINT_METHODS or is_mobile_path
    )
    if not is_mobile_path and not is_bearer:
        # Login providers/2FA return to the usual index. Resume the connection
        # page only after authentication, without changing any provider flow.
        if endpoint == 'index' and request.method == 'GET' and session.get('mobile_connect_pending') and current_user.is_authenticated:
            session.pop('mobile_connect_pending', None)
            code = session.pop('mobile_connect_code', None)
            target = url_for('mobile_connect', code=code) if code else url_for('mobile_connect')
            return redirect(target)
        return
    g.mobile_request = True
    if not _mobile_enabled():
        return _mobile_error('mobile_api_disabled', 503)
    if not _is_secure_request():
        return _mobile_error('https_required', 400)
    if endpoint in MOBILE_PUBLIC_ENDPOINTS or is_bearer:
        # Neither cookies nor browser-origin requests may acquire the native
        # CSRF exemption. Native HTTP clients must use a separate cookie jar.
        if request.headers.get('Cookie') or request.headers.get('Origin'):
            return _mobile_error('cookies_or_origin_not_allowed', 400)
    if endpoint in MOBILE_PUBLIC_ENDPOINTS:
        if header:
            return _mobile_error('authorization_not_allowed', 400)
        if request.method != 'POST' or not request.is_json:
            return _mobile_error('json_post_required', 415)
        # Standard Integrity tokens and WebAuthn assertion payloads exceed the
        # legacy 4 KiB native-auth limit. Bound them while allowing both fields.
        request.max_content_length = 32 * 1024
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _mobile_error('invalid_request')
        g.mobile_public_request = True
    if is_bearer:
        allowed = MOBILE_ENDPOINT_METHODS.get(endpoint, set())
        if request.method not in allowed:
            return _mobile_error('insufficient_scope', 403)
        if not re.fullmatch(r'aip_android_[A-Za-z0-9_-]{43}', header[7:]) or not header.startswith('Bearer '):
            return _mobile_error('invalid_token', 401)
        g.mobile_bearer_request = True
        if not current_user.is_authenticated or not getattr(g, 'mobile_session', None):
            return _mobile_error('invalid_token', 401)
        if not current_user.is_setup_completed and endpoint not in MOBILE_SETUP_ENDPOINTS and endpoint != 'mobile_revoke':
            return _mobile_error('setup_required', 403)
        if current_user.is_setup_completed and endpoint in MOBILE_FIRST_RUN_ONLY_ENDPOINTS:
            return _mobile_error('setup_already_completed', 403)
        if current_user.is_setup_completed and endpoint in MOBILE_REAUTH_ENDPOINTS and not _mobile_recently_reauthenticated():
            if endpoint != 'mobile_easy_login' or not (request.get_json(silent=True) or {}).get('cancel'):
                return _mobile_error('reauth_required', 403)
    elif endpoint in MOBILE_ENDPOINT_METHODS:
        return _mobile_error('invalid_token', 401)


@app.after_request
def mobile_response_headers(response):
    if getattr(g, 'mobile_request', False):
        response.headers['Cache-Control'] = 'private, no-store, no-transform, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        # Existing non-API guards sometimes return HTML/redirects. Native
        # callers receive machine-readable errors before a stream is opened.
        if getattr(g, 'mobile_bearer_request', False) and response.status_code >= 300 and response.status_code != 304 and not response.is_json:
            code = 'maintenance' if response.status_code == 503 else 'request_blocked'
            status = response.status_code if response.status_code >= 400 else 403
            replacement = _mobile_error(code, status)
            replacement.headers['Cache-Control'] = 'private, no-store, no-transform, max-age=0'
            replacement.headers['Pragma'] = 'no-cache'
            replacement.headers['Referrer-Policy'] = 'no-referrer'
            return replacement
    return response
