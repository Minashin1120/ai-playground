"""Native client authentication. Loaded before all request hooks; no startup I/O."""

MOBILE_TOKEN_PREFIX = 'aip_android_'
MOBILE_TOKEN_TTL = 30 * 24 * 60 * 60
MOBILE_GRANT_TTL = 600
MOBILE_POLL_INTERVAL = 5
MOBILE_ENDPOINT_METHODS = {
    'mobile_me': {'GET'}, 'mobile_revoke': {'POST'},
    'handle_threads': {'GET', 'POST'}, 'handle_thread_item': {'GET', 'DELETE'},
    'update_thread_settings': {'GET', 'PUT'}, 'update_title': {'PUT'},
    'toggle_bookmark': {'POST'},
    'chat_stream': {'POST'}, 'chat_stream_resume': {'POST'}, 'stop_chat': {'POST'},
    'temporary_chat_heartbeat': {'POST'}, 'estimate_prompt_tokens_api': {'POST'},
    'upload': {'POST'}, 'upload_init': {'POST'}, 'upload_chunk': {'POST'},
    'upload_complete': {'POST'}, 'serve_file': {'GET', 'HEAD'},
    'serve_file_thumb': {'GET', 'HEAD'}, 'get_storage_usage': {'GET'},
    'get_files_lib': {'GET'},
    'toggle_file_favorite': {'POST'}, 'delete_files_batch': {'POST'},
    'rename_library_file': {'POST'},
    'handle_gems': {'GET', 'POST'}, 'handle_gem_item': {'GET', 'PUT', 'DELETE'},
}
MOBILE_PUBLIC_ENDPOINTS = {'mobile_device', 'mobile_token'}


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
            return redirect(url_for('mobile_connect'))
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
        request.max_content_length = 4096
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
        if not current_user.is_setup_completed and endpoint != 'mobile_revoke':
            return _mobile_error('setup_required', 403)
    elif endpoint in {'mobile_me', 'mobile_revoke'}:
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
