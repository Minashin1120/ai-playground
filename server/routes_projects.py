"""Project CRUD, shared file attachment management, and project thread routes."""


def _project_vision_api_keys_for_user(user):
    """Build decrypted API keys map for Vision model image analysis."""
    def _dec(attr):
        val = getattr(user, attr, None)
        if not val:
            return None
        try:
            return decrypt_val(val)
        except Exception:
            return None

    return {
        "gemini": _dec("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
        "openai": _dec("openai_api_key") or os.getenv("OPENAI_API_KEY"),
        "anthropic": _dec("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"),
        "xai": _dec("xai_api_key") or os.getenv("XAI_API_KEY"),
        "zai": _dec("zai_api_key") or os.getenv("ZAI_API_KEY"),
    }


def _make_project_vision_runner(user, project, pub_fn=None):
    """Return a ``vision_fn(img_bytes, img_mime, display_name) -> (text, err)`` callback."""
    vision_model = resolve_project_vision_model(project, user)
    api_keys = _project_vision_api_keys_for_user(user)
    auto_notice_config = get_user_auto_system_prompt_notices_config(user)
    analysis_prompt = get_user_auto_system_prompt_notice_text(user, "image_analysis", auto_notice_config)

    def _run(img_bytes, img_mime, display_name):
        if pub_fn is not None:
            try:
                pub_fn("image_analysis", f"プロジェクト画像 ({display_name}) を Vision モデルで解析中...")
            except Exception:
                pass
        result = _analyze_image_with_vision_model(
            vision_model,
            img_bytes,
            img_mime,
            analysis_prompt,
            api_keys,
        )
        if result and str(result).strip():
            if pub_fn is not None:
                try:
                    pub_fn("image_analysis", f"プロジェクト画像 ({display_name}) の解析完了")
                except Exception:
                    pass
            return str(result).strip(), None
        if pub_fn is not None:
            try:
                pub_fn("image_analysis", f"プロジェクト画像 ({display_name}) の解析に失敗しました")
            except Exception:
                pass
        return None, f"Vision model ({vision_model}) could not analyze the image (check API key settings)."

    return _run


def _project_file_response(row, user_id, label_map=None):
    rel_path = _normalize_upload_ref(row.rel_path) or row.rel_path
    info = _get_file_disk_info(rel_path)
    base_name = os.path.basename(rel_path)
    display_name = str(
        row.display_name
        or ((label_map or {}).get(rel_path) if label_map is not None else None)
        or base_name
    ).strip() or base_name
    ext = os.path.splitext(base_name)[1].lower().lstrip('.')
    mime_guess = mimetypes.guess_type(rel_path)[0] or 'application/octet-stream'
    mime = _normalize_media_mime(rel_path, mime_guess)
    is_image = _is_project_image_path(rel_path, mime)
    return {
        'id': row.id,
        'filepath': rel_path,
        'filename': display_name,
        'original_filename': base_name,
        'note': row.note or '',
        'mime': mime,
        'ext': ext,
        'type': 'image' if is_image else 'file',
        'is_image': is_image,
        'size': int(info.get('size') or 0) if info.get('exists') else 0,
        'exists': bool(info.get('exists')),
        'url': url_for('serve_file', filename=rel_path),
        'thumbnail_url': url_for('serve_file_thumb', filename=rel_path) if is_image else None,
        'created_at': row.created_at.isoformat() + 'Z' if row.created_at else None,
    }


def _project_response(project, user, include_files=False, file_count=None, thread_count=None):
    if file_count is None:
        file_count = ProjectFile.query.filter_by(project_id=project.id, user_id=user.id).count()
    if thread_count is None:
        thread_count = (
            Thread.query
            .filter_by(user_id=user.id, project_id=project.id)
            .filter(~Thread.title.startswith("[LIBRARY]"))
            .count()
        )
    data = {
        'uuid': project.uuid,
        'id': project.id,
        'name': project.name,
        'description': project.description or '',
        'instruction': project.instruction or '',
        'use_vision_for_images': bool(getattr(project, 'use_vision_for_images', False)),
        'vision_model': getattr(project, 'vision_model', None) or '',
        'effective_vision_model': resolve_project_vision_model(project, user),
        'file_count': int(file_count),
        'thread_count': int(thread_count),
        'created_at': project.created_at.isoformat() + 'Z' if project.created_at else None,
        'updated_at': project.updated_at.isoformat() + 'Z' if project.updated_at else None,
    }
    if include_files:
        label_map = _get_user_file_label_map(user.id)
        rows = (
            ProjectFile.query
            .filter_by(project_id=project.id, user_id=user.id)
            .order_by(ProjectFile.created_at.asc(), ProjectFile.id.asc())
            .all()
        )
        data['files'] = [_project_file_response(r, user.id, label_map=label_map) for r in rows]
        data['file_count'] = len(data['files'])
    return data


def _project_name_conflict(name, user_id, exclude_id=None):
    query = Project.query.filter(
        Project.user_id == user_id,
        db.func.lower(Project.name) == str(name or '').lower(),
    )
    if exclude_id is not None:
        query = query.filter(Project.id != exclude_id)
    return query.first() is not None


def _normalize_project_payload(data, existing=None):
    if not isinstance(data, dict):
        raise ValueError('invalid_payload')
    out = {}
    if existing is None or 'name' in data:
        name = str(data.get('name') or '').strip()
        if not name:
            raise ValueError('name_required')
        if len(name) > PROJECT_NAME_MAX_CHARS:
            raise ValueError('name_too_long')
        out['name'] = name
    if existing is None or 'description' in data:
        desc = str(data.get('description') or '').strip()
        if len(desc) > PROJECT_DESCRIPTION_MAX_CHARS:
            raise ValueError('description_too_long')
        out['description'] = desc
    if existing is None or 'instruction' in data:
        inst = str(data.get('instruction') or '')
        if len(inst) > PROJECT_INSTRUCTION_MAX_CHARS:
            raise ValueError('instruction_too_long')
        out['instruction'] = inst
    if existing is None or 'use_vision_for_images' in data:
        raw_use_vision = data.get('use_vision_for_images', False if existing is None else existing.use_vision_for_images)
        out['use_vision_for_images'] = bool(_coerce_bool_or_none(raw_use_vision))
    if existing is None or 'vision_model' in data:
        vm = str(data.get('vision_model') or '').strip()
        if vm:
            if vm not in ALL_VALID_MODEL_IDS:
                raise ValueError('invalid_vision_model')
            out['vision_model'] = vm
        else:
            out['vision_model'] = None
    return out


@app.route('/api/projects', methods=['GET', 'POST'])
@login_required
def handle_projects():
    if request.method == 'GET':
        projects = (
            Project.query
            .filter_by(user_id=current_user.id)
            .order_by(Project.updated_at.desc(), Project.id.desc())
            .all()
        )
        proj_ids = [p.id for p in projects]
        file_counts = {}
        thread_counts = {}
        if proj_ids:
            for pid, cnt in db.session.query(ProjectFile.project_id, func.count(ProjectFile.id)).filter(
                ProjectFile.user_id == current_user.id,
                ProjectFile.project_id.in_(proj_ids),
            ).group_by(ProjectFile.project_id).all():
                file_counts[pid] = int(cnt)
            for pid, cnt in db.session.query(Thread.project_id, func.count(Thread.id)).filter(
                Thread.user_id == current_user.id,
                Thread.project_id.in_(proj_ids),
                ~Thread.title.startswith("[LIBRARY]"),
            ).group_by(Thread.project_id).all():
                thread_counts[pid] = int(cnt)
        return jsonify([
            _project_response(
                p,
                current_user,
                include_files=False,
                file_count=file_counts.get(p.id, 0),
                thread_count=thread_counts.get(p.id, 0),
            )
            for p in projects
        ])

    data = request.get_json(silent=True) or {}
    try:
        payload = _normalize_project_payload(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if _project_name_conflict(payload['name'], current_user.id):
        return jsonify({'error': 'name_taken'}), 400
    import uuid as _uuid
    project = Project(
        uuid=str(_uuid.uuid4()),
        user_id=current_user.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        **payload,
    )
    db.session.add(project)
    safe_db_commit()
    return jsonify(_project_response(project, current_user, include_files=True, file_count=0, thread_count=0)), 201


@app.route('/api/projects/<string:project_uuid>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def handle_project_item(project_uuid):
    project = resolve_project_for_user(project_uuid, current_user.id)
    if not project:
        return jsonify({'error': 'not_found'}), 404

    if request.method == 'GET':
        return jsonify(_project_response(project, current_user, include_files=True))

    if request.method == 'PUT':
        data = request.get_json(silent=True) or {}
        try:
            payload = _normalize_project_payload(data, existing=project)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        if 'name' in payload and _project_name_conflict(payload['name'], current_user.id, exclude_id=project.id):
            return jsonify({'error': 'name_taken'}), 400
        for key, val in payload.items():
            setattr(project, key, val)
        project.updated_at = datetime.utcnow()
        safe_db_commit()
        return jsonify(_project_response(project, current_user, include_files=True))

    # DELETE: unassign all threads cleanly so chats are never lost accidentally.
    assigned_ids = [
        tid for (tid,) in db.session.query(Thread.id).filter_by(
            user_id=current_user.id, project_id=project.id
        ).all()
    ]
    if assigned_ids:
        Thread.query.filter(Thread.id.in_(assigned_ids)).update(
            {Thread.project_id: None}, synchronize_session=False
        )
        _sync_touch_threads(db.session.connection(), assigned_ids)
    db.session.delete(project)
    safe_db_commit()
    return jsonify({'status': 'deleted', 'unassigned_threads': len(assigned_ids)})


@app.route('/api/projects/<string:project_uuid>/threads', methods=['GET'])
@login_required
def handle_project_threads(project_uuid):
    project = resolve_project_for_user(project_uuid, current_user.id)
    if not project:
        return jsonify({'error': 'not_found'}), 404
    q = (request.args.get('q') or '').strip()
    page = max(1, request.args.get('page', 1, type=int) or 1)
    per_page = 20
    query = (
        Thread.query
        .filter_by(user_id=current_user.id, project_id=project.id)
        .filter(~Thread.title.startswith("[LIBRARY]"))
    )
    if q:
        if current_user.enable_e2ee:
            query = query.filter(Thread.title.contains(q))
        else:
            query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
    pagination = query.order_by(
        Thread.is_bookmarked.desc(), Thread.bookmarked_at.desc(), Thread.updated_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    threads = [
        {
            'id': t.public_id or t.id,
            'title': t.title,
            'is_bookmarked': bool(t.is_bookmarked),
            'last_model': t.last_model,
            'is_temporary': bool(getattr(t, 'is_temporary', False)),
            'project_id': project.uuid,
            'project_name': project.name,
        }
        for t in pagination.items
    ]
    return jsonify({
        'project': _project_response(project, current_user, include_files=False),
        'threads': threads,
        'has_next': pagination.has_next,
        'next_page': pagination.next_num,
    })


@app.route('/api/projects/<string:project_uuid>/files', methods=['GET', 'POST'])
@login_required
def handle_project_files(project_uuid):
    project = resolve_project_for_user(project_uuid, current_user.id)
    if not project:
        return jsonify({'error': 'not_found'}), 404
    if request.method == 'GET':
        label_map = _get_user_file_label_map(current_user.id)
        rows = (
            ProjectFile.query
            .filter_by(project_id=project.id, user_id=current_user.id)
            .order_by(ProjectFile.created_at.asc(), ProjectFile.id.asc())
            .all()
        )
        return jsonify({'files': [_project_file_response(r, current_user.id, label_map=label_map) for r in rows]})

    data = request.get_json(silent=True) or {}
    raw_items = data.get('files')
    if not isinstance(raw_items, list):
        raw_paths = data.get('filepaths') or data.get('rel_paths') or ([data.get('filepath')] if data.get('filepath') else [])
        raw_items = [{'filepath': p, 'note': data.get('note'), 'display_name': data.get('display_name')} for p in raw_paths]
    if not raw_items:
        return jsonify({'error': 'files_required'}), 400

    existing_rows = ProjectFile.query.filter_by(project_id=project.id, user_id=current_user.id).all()
    existing_paths = {_normalize_upload_ref(r.rel_path) for r in existing_rows}
    label_map = _get_user_file_label_map(current_user.id)
    added = []
    skipped = []
    for entry in raw_items:
        if isinstance(entry, str):
            entry = {'filepath': entry}
        if not isinstance(entry, dict):
            continue
        raw_ref = entry.get('filepath') or entry.get('path') or entry.get('url')
        rel_path = _resolve_user_upload_rel_path(raw_ref, current_user.id)
        if not rel_path or not rel_path.startswith(f"{current_user.id}/") or '..' in rel_path.split('/'):
            skipped.append({'filepath': str(raw_ref or '')[:120], 'reason': 'forbidden'})
            continue
        info = _get_file_disk_info(rel_path)
        if not info.get('exists'):
            skipped.append({'filepath': rel_path, 'reason': 'not_found'})
            continue
        if rel_path in existing_paths:
            skipped.append({'filepath': rel_path, 'reason': 'already_attached'})
            continue
        if len(existing_paths) >= PROJECT_MAX_FILES:
            return jsonify({
                'error': 'project_file_limit_exceeded',
                'limit': PROJECT_MAX_FILES,
                'added': added,
            }), 400
        raw_display = entry.get('display_name') or entry.get('filename') or label_map.get(rel_path) or os.path.basename(rel_path)
        display_name = _normalize_display_name_for_path(rel_path, raw_display) or os.path.basename(rel_path)
        note = str(entry.get('note') or '').replace('\r\n', '\n').strip()[:PROJECT_FILE_NOTE_MAX_CHARS]
        row = ProjectFile(
            project_id=project.id,
            user_id=current_user.id,
            rel_path=rel_path,
            display_name=display_name[:255],
            note=note or None,
            created_at=datetime.utcnow(),
        )
        db.session.add(row)
        db.session.flush()
        existing_paths.add(rel_path)
        added.append(_project_file_response(row, current_user.id, label_map=label_map))

    if added:
        project.updated_at = datetime.utcnow()
        safe_db_commit()
    elif any(item.get('reason') == 'forbidden' for item in skipped):
        return jsonify({'error': 'forbidden', 'added': added, 'skipped': skipped}), 400
    return jsonify({
        'status': 'ok',
        'added': added,
        'skipped': skipped,
        'file_count': len(existing_paths),
    }), 201 if added else 200


@app.route('/api/projects/<string:project_uuid>/files/<int:file_id>', methods=['PUT', 'DELETE'])
@login_required
def handle_project_file_item(project_uuid, file_id):
    project = resolve_project_for_user(project_uuid, current_user.id)
    if not project:
        return jsonify({'error': 'not_found'}), 404
    row = ProjectFile.query.filter_by(id=file_id, project_id=project.id, user_id=current_user.id).first()
    if not row:
        return jsonify({'error': 'not_found'}), 404

    if request.method == 'DELETE':
        db.session.delete(row)
        project.updated_at = datetime.utcnow()
        safe_db_commit()
        return jsonify({'status': 'deleted'})

    data = request.get_json(silent=True) or {}
    if 'note' in data:
        row.note = str(data.get('note') or '').replace('\r\n', '\n').strip()[:PROJECT_FILE_NOTE_MAX_CHARS] or None
    if 'display_name' in data or 'filename' in data:
        raw_name = data.get('display_name') if 'display_name' in data else data.get('filename')
        fixed = _normalize_display_name_for_path(row.rel_path, raw_name)
        if fixed:
            row.display_name = fixed[:255]
    project.updated_at = datetime.utcnow()
    safe_db_commit()
    return jsonify(_project_file_response(row, current_user.id))


@app.route('/api/projects/<string:project_uuid>/files/read', methods=['POST'])
@login_required
def read_project_file_api(project_uuid):
    """Execute ``read_project_file`` for Web Browser Fast Mode (and direct tool callers)."""
    if not rate_limit(f"rl:project_file_read:user:{current_user.id}", 120, 60):
        return jsonify({'error': 'rate_limit'}), 429
    project = resolve_project_for_user(project_uuid, current_user.id)
    if not project:
        return jsonify({'error': 'not_found'}), 404
    data = request.get_json(silent=True) or {}
    name = str(data.get('name') or '').strip()
    offset = data.get('offset', 0)
    limit = data.get('limit', PROJECT_FILE_READ_DEFAULT_LIMIT)
    files = get_project_files_for_prompt(project, current_user.id)
    vision_fn = (
        _make_project_vision_runner(current_user, project)
        if bool(getattr(project, 'use_vision_for_images', False))
        else None
    )
    text_out, meta = execute_project_file_tool(
        current_user.id,
        project,
        files,
        name=name,
        offset=offset,
        limit=limit,
        vision_fn=vision_fn,
    )
    resp = jsonify({
        'status': 'ok' if meta.get('ok') else 'error',
        'content': text_out,
        'meta': meta,
    })
    resp.headers['Cache-Control'] = 'no-store'
    return resp
