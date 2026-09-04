@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        easy_login_used = bool(session.pop('easy_login_used', False))
        bot_config = {
            "username": current_user.username,
            "isAdmin": bool(getattr(current_user, "is_admin", False)),
            "globalEnabled": get_bot_detection_global_enabled(),
            "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
            "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or "",
            "turnstileVerified": _bot_turnstile_active() and _bot_turnstile_verified(),
            "lock": _bot_lock_config()
        }
        return render_template('chat.html', easy_login_used=easy_login_used, bot_config=bot_config)
    return render_template('landing.html')

@app.route('/settings')
@app.route('/upload')
@app.route('/library')
@app.route('/history')
@app.route('/branch')
@app.route('/paste')
@app.route('/camera')
@app.route('/edit-image')
@app.route('/chat-settings')
@app.route('/model')
@app.route('/token-details')
@app.route('/encryption-status')
@app.route('/python-execution')
@app.route('/gem')
@app.route('/compression')
@app.route('/admin-bots')
@login_required
def modal_pages():
    if not current_user.is_setup_completed:
        return redirect(url_for('setup'))
    easy_login_used = bool(session.pop('easy_login_used', False))
    bot_config = {
        "username": current_user.username,
        "isAdmin": bool(getattr(current_user, "is_admin", False)),
        "globalEnabled": get_bot_detection_global_enabled(),
        "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
        "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or "",
        "turnstileVerified": _bot_turnstile_active() and _bot_turnstile_verified(),
        "lock": _bot_lock_config()
    }
    return render_template('chat.html', easy_login_used=easy_login_used, bot_config=bot_config)

@app.route('/c/<thread_id>')
@login_required
def chat_permalink(thread_id):
    thread = resolve_thread_for_user(thread_id, current_user.id)
    if not thread:
        return render_template('404.html', message="指定されたチャットは存在しません。"), 404
    easy_login_used = bool(session.pop('easy_login_used', False))
    bot_config = {
        "username": current_user.username,
        "isAdmin": bool(getattr(current_user, "is_admin", False)),
        "globalEnabled": get_bot_detection_global_enabled(),
        "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
        "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or "",
        "turnstileVerified": _bot_turnstile_active() and _bot_turnstile_verified(),
        "lock": _bot_lock_config()
    }
    initial_thread_id = thread.public_id or thread.id
    return render_template('chat.html', initial_thread_id=initial_thread_id, easy_login_used=easy_login_used, bot_config=bot_config)

def _get_changelogs(page=1, limit=10):
    log_dir = app.config.get('CHANGELOG_FOLDER', os.path.join(os.path.dirname(__file__), 'static/changelogs'))
    all_logs = []
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.md'))
        def _changelog_meta(path):
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r'^(\d{4}-\d{2}-\d{2})_v(.+)$', base)
            if not m:
                m = re.match(r'^(\d{8})_v(.+)$', base)
            if m:
                date_raw, version = m.group(1), m.group(2)
                if len(date_raw) == 8:
                    date_fmt = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date_fmt = date_raw
                date_key = int(date_fmt.replace('-', ''))
                ver_nums = tuple(int(x) for x in re.findall(r'\d+', version)) or (0,)
                title = f"V{version} ({date_fmt})"
                return date_key, ver_nums, title
            return 0, (0,), base
        
        # Sort primarily by date (newest first), then by version as tiebreaker.
        # Date-based sorting is more reliable since version number tuple comparison
        # breaks across different version formats (e.g. (224,) > (4, 8, 608)).
        files.sort(key=lambda p: (_changelog_meta(p)[0], _changelog_meta(p)[1]), reverse=True)
        
        start = (page - 1) * limit
        end = start + limit
        paginated_files = files[start:end]
        
        for f in paginated_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                title = None
                if not content.lstrip().startswith('#'):
                    _, _, title = _changelog_meta(f)
                all_logs.append({'content': content, 'title': title})
            except Exception as e:
                logger.error(f"Error reading changelog file {f}: {e}")
                
        return all_logs, len(files)
    return [], 0

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/changelog')
def changelog():
    logs, total = _get_changelogs(page=1, limit=10)
    return render_template('changelog.html', logs=logs, total=total, limit=10)

@app.route('/api/changelogs')
def api_changelogs():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    logs, total = _get_changelogs(page=page, limit=limit)
    return jsonify({'logs': logs, 'total': total, 'page': page, 'limit': limit})

@app.route('/api/changelogs/search')
def api_changelogs_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'logs': [], 'total': 0})

    log_dir = app.config.get('CHANGELOG_FOLDER', os.path.join(os.path.dirname(__file__), 'static/changelogs'))
    results = []

    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.md'))

        def _changelog_meta(path):
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r'^(\d{4}-\d{2}-\d{2})_v(.+)$', base)
            if not m:
                m = re.match(r'^(\d{8})_v(.+)$', base)
            if m:
                date_raw, version = m.group(1), m.group(2)
                if len(date_raw) == 8:
                    date_fmt = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date_fmt = date_raw
                date_key = int(date_fmt.replace('-', ''))
                ver_nums = tuple(int(x) for x in re.findall(r'\d+', version)) or (0,)
                title = f"V{version} ({date_fmt})"
                return date_key, ver_nums, title
            return 0, (0,), base

        files.sort(key=lambda p: (_changelog_meta(p)[0], _changelog_meta(p)[1]), reverse=True)

        q_lower = q.lower()
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                title = None
                if not content.lstrip().startswith('#'):
                    _, _, title = _changelog_meta(f)

                if q_lower in content.lower() or (title and q_lower in title.lower()):
                    results.append({'content': content, 'title': title})
            except Exception as e:
                logger.error(f"Error reading changelog file {f}: {e}")

    return jsonify({'logs': results, 'total': len(results)})

@app.route('/banned')
@login_required
def banned():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for('index'))
    if not current_user.is_bot_banned:
        return redirect(url_for('index'))
    latest_appeal = None
    try:
        latest_appeal = BanAppeal.query.filter_by(user_id=current_user.id).order_by(BanAppeal.created_at.desc()).first()
    except Exception:
        latest_appeal = None
    return render_template(
        'banned.html',
        reason=current_user.bot_ban_reason,
        banned_at=current_user.bot_banned_at,
        evidence=current_user.bot_evidence,
        latest_appeal=latest_appeal,
        appeal_submitted=session.pop('appeal_submitted', False),
        appeal_error=session.pop('appeal_error', None),
        appeal_blocked=bool(getattr(current_user, "appeal_blocked", False)),
        appeal_block_reason=getattr(current_user, "appeal_block_reason", None)
    )

@app.route('/ban/appeal', methods=['POST'])
@login_required
def submit_ban_appeal():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for('index'))
    if not current_user.is_bot_banned:
        return redirect(url_for('index'))
    if getattr(current_user, "appeal_blocked", False):
        session['appeal_error'] = current_user.appeal_block_reason or "異議申し立てはブロックされています。"
        return redirect(url_for('banned'))
    message = (request.form.get('message') or '').strip()
    if not message or len(message) < 10:
        session['appeal_error'] = "内容は10文字以上で入力してください。"
        return redirect(url_for('banned'))
    if len(message) > 3000:
        session['appeal_error'] = "内容は3000文字以内で入力してください。"
        return redirect(url_for('banned'))
    appeal = BanAppeal(
        user_id=current_user.id,
        username=current_user.username,
        message=message,
        ban_reason=current_user.bot_ban_reason,
        ban_at=current_user.bot_banned_at,
        ip_address=get_client_ip(),
        user_agent=get_request_user_agent(),
        evidence=current_user.bot_evidence
    )
    db.session.add(appeal)
    safe_db_commit()
    session['appeal_submitted'] = True
    return redirect(url_for('banned'))

@app.route('/api/ban/appeal/status')
@login_required
def api_ban_appeal_status():
    if getattr(current_user, 'is_admin', False):
        return jsonify({'error': 'admin_not_allowed'}), 403
    latest = BanAppeal.query.filter_by(user_id=current_user.id).order_by(BanAppeal.created_at.desc()).first()
    if not latest:
        return jsonify({'has_appeal': False})
    return jsonify({
        'has_appeal': True,
        'status': latest.status,
        'created_at': latest.created_at.isoformat() + "Z" if latest.created_at else None,
        'evidence': current_user.bot_evidence or "",
        'ban_reason': current_user.bot_ban_reason or "",
        'banned_at': current_user.bot_banned_at.isoformat() + "Z" if current_user.bot_banned_at else None
    })

@app.route('/api/version')
def api_version():
    if request.args.get('heartbeat') and app.config.get('MAINTENANCE_MODE'):
        resp = jsonify({'error': 'maintenance', 'code': 'maintenance'})
        resp.status_code = 503
        resp.headers['X-AI-Maintenance'] = '1'
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    resp = jsonify({'version': app.config.get('APP_VERSION', '')})
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/csrf_token')
def api_csrf_token():
    resp = jsonify({'csrf_token': get_csrf_token()})
    resp.headers['Cache-Control'] = 'private, no-store, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route('/api/assets/fonts/japanese.ttf')
def serve_japanese_font():
    font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
    if not os.path.exists(font_path):
        # Fallback to other possible locations
        alt_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
        ]
        for p in alt_paths:
            if os.path.exists(p):
                font_path = p
                break
    try:
        return send_file(font_path, mimetype='font/ttf', max_age=31536000)
    except Exception as e:
        log_force(f"Error serving font: {e}")
        abort(404)

@app.route('/sw.js')
def service_worker():
    resp = send_from_directory(app.static_folder, 'sw.js')
    resp.headers['Content-Type'] = 'application/javascript; charset=utf-8'
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['Service-Worker-Allowed'] = '/'
    return resp


@app.route('/manifest.webmanifest')
def site_manifest():
    resp = send_from_directory(app.static_folder, 'manifest.webmanifest')
    resp.headers['Content-Type'] = 'application/manifest+json; charset=utf-8'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.static_folder, 'pwa'),
        'icon-192.png',
        mimetype='image/png'
    )

