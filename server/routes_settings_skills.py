
def _skill_response(skill, legacy=False):
    data = {
        'uuid': skill.uuid,
        'id': skill.id,
        'name': skill.name,
        'description': skill.description,
        'instruction': skill.instruction,
    }
    if legacy:
        # Older clients read these two fields.  New Skill clients ignore them.
        data['fixed_prompts'] = skill.fixed_prompts_json
        data['default_model'] = skill.default_model
    return data


def _skill_name_conflict(name, exclude_id=None):
    query = Skill.query.filter(
        Skill.user_id == current_user.id,
        db.func.lower(Skill.name) == str(name or '').lower(),
    )
    if exclude_id is not None:
        query = query.filter(Skill.id != exclude_id)
    return query.first() is not None


def _handle_skills_request(legacy=False):
    if request.method == 'GET':
        skills = Skill.query.filter_by(user_id=current_user.id).order_by(Skill.created_at.desc()).all()
        return jsonify([_skill_response(skill, legacy=legacy) for skill in skills])
    d = request.get_json(silent=True) or {}
    normalize = _normalize_gem_payload if legacy else _normalize_skill_payload
    try:
        payload = normalize(d)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if _skill_name_conflict(payload.get('name')):
        return jsonify({'error': 'name_taken'}), 400
    import uuid as _uuid
    skill = Skill(uuid=str(_uuid.uuid4()), user_id=current_user.id, **payload)
    db.session.add(skill)
    safe_db_commit()
    return jsonify({'uuid': skill.uuid, 'id': skill.id, 'name': skill.name})


def _handle_skill_item_request(skill_uuid, legacy=False):
    skill = Skill.query.filter_by(uuid=skill_uuid).first_or_404()
    if skill.user_id != current_user.id:
        return jsonify({'error': '403'}), 403

    if request.method == 'GET':
        return jsonify(_skill_response(skill, legacy=legacy))

    if request.method == 'PUT':
        d = request.get_json(silent=True) or {}
        normalize = _normalize_gem_payload if legacy else _normalize_skill_payload
        try:
            payload = normalize(d, existing=skill)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        if 'name' in payload and _skill_name_conflict(payload['name'], exclude_id=skill.id):
            return jsonify({'error': 'name_taken'}), 400
        for key, value in payload.items():
            setattr(skill, key, value)
        safe_db_commit()
        return jsonify({'uuid': skill.uuid, 'id': skill.id, 'name': skill.name})
    if request.method == 'DELETE':
        db.session.delete(skill)
        safe_db_commit()
        return jsonify({'status': 'deleted'})


@app.route('/api/skills', methods=['GET', 'POST'])
@login_required
def handle_skills():
    return _handle_skills_request()


@app.route('/api/skills/<string:skill_uuid>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def handle_skill_item(skill_uuid):
    return _handle_skill_item_request(skill_uuid)


# Legacy aliases: older Web clients, bookmarks and shipped Android builds still
# call the Gem routes.  They keep the legacy response fields.
@app.route('/api/gems', methods=['GET', 'POST'])
@login_required
def handle_gems():
    return _handle_skills_request(legacy=True)


@app.route('/api/gems/<string:gem_uuid>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def handle_gem_item(gem_uuid):
    return _handle_skill_item_request(gem_uuid, legacy=True)

@app.route('/api/maintenance', methods=['POST'])
@login_required
def toggle_maintenance():
    if not getattr(current_user, "is_admin", False): return abort(403)
    lock_file = os.path.join(os.path.dirname(__file__), 'maintenance.lock')
    if request.json.get('enabled'):
        with open(lock_file, 'w') as f: f.write('locked')
        app.config['MAINTENANCE_MODE'] = True
    else:
        if os.path.exists(lock_file): os.remove(lock_file)
        app.config['MAINTENANCE_MODE'] = False
    return jsonify({'status': 'ok', 'mode': app.config['MAINTENANCE_MODE']})
