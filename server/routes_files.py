@app.route('/files/thumb/<path:filename>')
@login_required
def serve_file_thumb(filename):
    actual_rel_path = _resolve_user_upload_rel_path(filename, current_user.id)
    if not actual_rel_path:
        abort(403)
    info = _get_file_disk_info(actual_rel_path)
    if not info or not info.get("exists"):
        abort(404)

    ext = os.path.splitext(actual_rel_path)[1].lower()
    if ext not in _IMAGE_THUMB_EXTS:
        return redirect(url_for('serve_file', filename=filename))

    cache_key = (
        actual_rel_path,
        info.get("mtime"),
        info.get("size"),
        1 if info.get("is_encrypted") else 0,
        _THUMBNAIL_SIZE,
        _THUMBNAIL_QUALITY
    )
    etag = hashlib.sha256(repr(cache_key).encode()).hexdigest()
    request_etag = request.headers.get("If-None-Match") or ""
    if request_etag:
        req_tokens = [tok.strip().strip('"') for tok in request_etag.split(",") if tok.strip()]
        if "*" in req_tokens or etag in req_tokens:
            resp = Response(status=304)
            return _add_thumb_cache_headers(resp, etag=etag)

    thumb_bytes = _thumbnail_bytes_cache_get(cache_key)
    if thumb_bytes is None:
        data = _load_user_file_bytes(actual_rel_path, info)
        if data is None:
            # A file present on disk that the current key cannot decrypt is a
            # key-mismatch (unrecoverable) case, not a missing file.
            if info and info.get("exists"):
                return _unreadable_file_http_response(filename, is_thumb=True)
            abort(404)
        try:
            with Image.open(BytesIO(data)) as im:
                if hasattr(Image, "Resampling"):
                    resample_lanczos = Image.Resampling.LANCZOS
                else:
                    resample_lanczos = Image.LANCZOS
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                im.thumbnail((_THUMBNAIL_SIZE, _THUMBNAIL_SIZE), resample=resample_lanczos)
                buf = BytesIO()
                im.save(buf, format="WEBP", quality=_THUMBNAIL_QUALITY, method=4)
                thumb_bytes = buf.getvalue()
        except Exception as e:
            log_force(f"Thumbnail generation failed for {actual_rel_path}: {e}")
            return redirect(url_for('serve_file', filename=filename))
        _thumbnail_bytes_cache_put(cache_key, thumb_bytes)

    resp = send_file(BytesIO(thumb_bytes), mimetype="image/webp")
    return _add_thumb_cache_headers(resp, etag=etag)

@app.route('/files/<path:filename>')
@login_required
def serve_file(filename):
    actual_rel_path = _resolve_user_upload_rel_path(filename, current_user.id)
    if not actual_rel_path:
        abort(403)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], actual_rel_path)
    if not _path_is_within(app.config['UPLOAD_FOLDER'], file_path):
        abort(403)
    
    enc_path = file_path + '.enc'
    mtype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    file_ext = os.path.splitext(file_path)[1].lower()
    force_download = file_ext in _FILE_FORCE_DOWNLOAD_EXTS
    if force_download:
        mtype = 'application/octet-stream'

    if os.path.exists(file_path):
        resp = send_file(file_path, mimetype=mtype, conditional=True, as_attachment=force_download, download_name=os.path.basename(actual_rel_path))
        resp.headers.setdefault("Accept-Ranges", "bytes")
        return _add_file_privacy_headers(resp)
    elif os.path.exists(enc_path):
        info = _get_file_disk_info(actual_rel_path)
        data = _load_user_file_bytes(actual_rel_path, info)
        if data is None:
            if info and info.get("exists"):
                return _unreadable_file_http_response(filename, is_thumb=False)
            abort(404)
        range_header = request.headers.get('Range')
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                size = len(data)
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
                chunk = data[start:end + 1]
                resp = Response(chunk, status=206, mimetype=mtype, direct_passthrough=True)
                resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Content-Length"] = str(end - start + 1)
                return _add_file_privacy_headers(resp)
        resp = send_file(BytesIO(data), download_name=os.path.basename(actual_rel_path), as_attachment=force_download, mimetype=mtype)
        return _add_file_privacy_headers(resp)
    else:
        abort(404)

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    log_force(f"DEBUG: handle_threads started, method={request.method}")
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = 20
        log_force(f"DEBUG: handle_threads query q={q}, page={page}")
        try:
            query = Thread.query.filter_by(user_id=current_user.id).filter(~Thread.title.startswith("[LIBRARY]"))
            if q: 
                if current_user.enable_e2ee: query = query.filter(Thread.title.contains(q))
                else: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
            
            pagination = query.order_by(Thread.is_bookmarked.desc(), Thread.bookmarked_at.desc(), Thread.updated_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
            threads = []
            for t in pagination.items:
                threads.append({
                    'id': t.public_id or t.id,
                    'title': t.title,
                    'is_bookmarked': bool(t.is_bookmarked),
                    'last_model': t.last_model,
                    'is_temporary': bool(getattr(t, "is_temporary", False))
                })
            log_force(f"DEBUG: handle_threads returning {len(threads)} threads")
            return jsonify({
                'threads': threads,
                'has_next': pagination.has_next,
                'next_page': pagination.next_num
            })
        except Exception as e:
            log_force(f"DEBUG: handle_threads GET failed: {e}")
            raise e
    
    # POST logic
    try:
        payload = request.get_json(silent=True) or {}
        requested_temp = _coerce_bool_or_none(payload.get('is_temporary'))
        log_force(f"DEBUG: handle_threads creating new thread, is_temporary={requested_temp}")
        t = Thread(
            user_id=current_user.id,
            public_id=generate_thread_public_id(),
            is_temporary=bool(requested_temp)
        )
        db.session.add(t)
        safe_db_commit()
        log_force(f"DEBUG: handle_threads thread created, public_id={t.public_id}")
        if bool(t.is_temporary):
            _mark_temp_chat_presence(
                t,
                current_user.id,
                timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
            )
        return jsonify({
            'id': t.public_id,
            'title': t.title,
            'is_temporary': bool(t.is_temporary),
            **_get_temp_chat_runtime_meta(t, user=current_user)
        })
    except Exception as e:
        log_force(f"DEBUG: handle_threads POST failed: {e}")
        raise e

@app.route('/api/threads/<thread_id>', methods=['GET', 'DELETE'])
@login_required
def handle_thread_item(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        limit = request.args.get('limit', type=int)
        before_id = request.args.get('before_id', type=int)
        include_meta_raw = str(request.args.get('include_meta', '1')).strip().lower()
        include_meta = include_meta_raw not in ('0', 'false', 'no', 'off')
        if limit is not None:
            limit = max(1, min(limit, 200))
        has_older_messages = False
        total_messages = None
        oldest_loaded_id = None

        q = Message.query.filter_by(thread_id=t.id)
        if before_id:
            q = q.filter(Message.id < before_id)

        if limit:
            # Page by message ID (cursor=before_id) and then restore stable presentation order.
            rows_desc = q.order_by(Message.id.desc()).limit(limit + 1).all()
            if len(rows_desc) > limit:
                has_older_messages = True
                rows_desc = rows_desc[:limit]
            ms = sorted(rows_desc, key=lambda m: ((m.timestamp or datetime.min), m.id))
            if ms:
                oldest_loaded_id = ms[0].id
            if before_id is None:
                total_messages = None
        else:
            # Ensure stable ordering even when timestamps collide (e.g., rapid edit/regenerate).
            ms = Message.query.filter_by(thread_id=t.id).order_by(Message.timestamp, Message.id).all()
            if ms:
                oldest_loaded_id = ms[0].id
        res = []
        for m in ms:
            cnt = decrypt_val(m.content) if m.is_encrypted else m.content
            tht = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
            thought_text = extract_reasoning_text(tht)
            token_in = None
            token_out = None
            token_total = None
            tokens_content = None
            tokens_thought = None
            legacy_token_total = None
            legacy_token_in = None
            legacy_token_out = None
            if (m.tokens_in and m.tokens_in > 0) or (m.tokens_out and m.tokens_out > 0):
                token_in = m.tokens_in if m.tokens_in and m.tokens_in > 0 else None
                token_out = m.tokens_out if m.tokens_out and m.tokens_out > 0 else None
                token_total = sum_token_counts(token_in, token_out)
                stored_tokens_thought = getattr(m, 'tokens_thought', None)
                if stored_tokens_thought is not None and stored_tokens_thought > 0:
                    tokens_thought = stored_tokens_thought
            elif m.tokens is not None and m.tokens > 0 and (should_count_tokens_for_display(m.model) or not m.model):
                if m.role == 'user':
                    legacy_token_in = m.tokens
                else:
                    legacy_token_out = m.tokens
                legacy_token_total = m.tokens
            if token_total is None and should_count_tokens_for_display(m.model):
                details = build_message_token_details(m.role, cnt, thought_text, m.model, token_in, token_out)
                token_in = details["tokens_in"] if details["tokens_in"] is not None else token_in
                token_out = details["tokens_out"] if details["tokens_out"] is not None else token_out
                token_total = details["tokens_total"] if details["tokens_total"] is not None else token_total
                tokens_content = details["tokens_content"]
                tokens_thought = details["tokens_thought"]
            if token_total is None and legacy_token_total is not None:
                token_in = token_in if token_in is not None else legacy_token_in
                token_out = token_out if token_out is not None else legacy_token_out
                token_total = legacy_token_total
            res.append({
                'id': m.id, 
                'role': m.role, 
                'content': cnt, 
                'image_url': m.image_url, 
                'model': m.model, 
                'thought_data': tht,
                'tokens': token_total,
                'tokens_in': token_in,
                'tokens_out': token_out,
                'tokens_content': tokens_content,
                'tokens_thought': tokens_thought,
                'is_encrypted': bool(m.is_encrypted),
                'quote_text': m.quote_text,
                'parent_id': m.parent_id,
                'gem_uuid': m.gem_uuid,
                'gem_name': m.gem_name
            })
        payload = {
            'messages': res,
            'has_older_messages': bool(has_older_messages),
            'oldest_loaded_id': oldest_loaded_id,
            'loaded_count': len(res),
            'total_messages': total_messages
        }
        if include_meta:
            pending_job = None
            try:
                pending_raw = redis_conn.get(f"pending_job:{current_user.id}:{t.id}")
                if pending_raw:
                    try:
                        pending_job = json.loads(pending_raw)
                    except Exception:
                        pending_job = {"job_id": pending_raw.decode("utf-8", "ignore")}
            except Exception:
                pending_job = None
            payload.update({
                'title': t.title,
                'custom_instruction': t.custom_instruction,
                'include_global_instruction': t.include_global_instruction if t.include_global_instruction is not None else True,
                'last_model': t.last_model,
                'last_gem_uuid': t.last_gem_uuid,
                'enable_prompt_caching': bool(getattr(t, "enable_prompt_caching", False)),
                'prompt_cache_provider': getattr(t, "prompt_cache_provider", None),
                'is_temporary': bool(getattr(t, "is_temporary", False)),
                'pending_job': pending_job
            })
            payload.update(_get_temp_chat_runtime_meta(t, user=current_user))
        return jsonify(payload)

    temp_member = _temp_chat_member(t)
    for m in t.messages:
        if m.image_url:
            try:
                for p in _iter_message_attachment_refs(m.image_url):
                    _delete_user_upload_ref(current_user.id, p)
            except Exception:
                pass

    db.session.delete(t)
    safe_db_commit()
    _clear_temp_chat_tracking(temp_member)
    try:
        redis_conn.delete(f"pending_job:{current_user.id}:{t.id}")
    except Exception:
        pass
    return jsonify({'status': 'deleted'})

