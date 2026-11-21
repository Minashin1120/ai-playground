"""Account-wide cost inputs for the usage tab.

The browser and Android app already turn saved per-request usage into an estimated
cost.  This endpoint collects the same usage for every chat of the signed-in user
inside one period, including the legacy fallbacks (input inferred from the parent
chain and image edits described only by their prompt options), so both clients can
reuse their existing estimator and show one total instead of per-chat totals.

Only numeric usage, model names, timestamps and pricing switches leave the
database.  Prompts, answers, search queries and file contents are never returned
or logged.
"""

_USAGE_PERIOD_DAYS = {
    '1d': 1,
    '7d': 7,
    '14d': 14,
    '1m': 30,
    '3m': 90,
    '1y': 365,
}
_USAGE_MAX_ANSWERS = 20000
_USAGE_IMAGE_MODELS = None
_USAGE_TTL_QUALITIES = ('very_low', 'low', 'medium', 'high')
_USAGE_IDEOGRAM_SPEEDS = ('turbo', 'default', 'quality')
_USAGE_IDEOGRAM_OPERATIONS = ('remove_object', 'remove_background')


def _usage_number(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _usage_prompt_options(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _usage_iso(value):
    return value.isoformat() + 'Z' if value else None


def _usage_is_image_model(model):
    """True when the official rate table bills this model per generated image."""
    global _USAGE_IMAGE_MODELS
    if _USAGE_IMAGE_MODELS is None:
        image_models = set()
        try:
            pricing_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'model_pricing.json')
            with open(pricing_path, encoding='utf-8') as pricing_file:
                pricing = json.load(pricing_file)
            aliases = pricing.get('aliases') or {}
            for key, rate in (pricing.get('models') or {}).items():
                if isinstance(rate, dict) and rate.get('unit') == 'image':
                    image_models.add(key)
                    for alias, target in aliases.items():
                        if target == key:
                            image_models.add(alias)
        except (OSError, ValueError, TypeError) as exc:
            log_force(f"usage cost image pricing unavailable: {exc}")
        _USAGE_IMAGE_MODELS = image_models
    return bool(model) and (model in _USAGE_IMAGE_MODELS)


def _usage_image_count(value):
    """Same interpretation as Web's `legacyImageRequest`: only a JSON list counts."""
    if not value:
        return 0
    try:
        files = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        files = value
    return len(files) if isinstance(files, list) else 0


def _usage_has_image_source(value):
    if isinstance(value, list):
        return bool(value)
    return bool(value) and value not in ('[]', 'null')


def _usage_legacy_image_request(row, rows_by_id):
    """Mirror Web's `legacyImageRequest` for answers saved before per-call usage."""
    parent = rows_by_id.get(row.parent_id)
    options = _usage_prompt_options(parent.prompt_options if parent else row.prompt_options)
    source = parent.image_url if parent else None
    has_source = _usage_has_image_source(source)
    node = parent
    seen = set()
    while not has_source and node is not None and node.parent_id is not None and node.id not in seen:
        seen.add(node.id)
        node = rows_by_id.get(node.parent_id)
        if node is not None and node.role == 'assistant' and node.image_url not in (None, '', '[]', 'null'):
            has_source = True
    operation = options.get('ideogram_operation')
    if operation not in _USAGE_IDEOGRAM_OPERATIONS:
        operation = 'edit' if has_source and row.model == 'ideogram-4.5' else 'generate'
    quality = options.get('ideogram_quality')
    if quality not in _USAGE_TTL_QUALITIES or (quality == 'very_low' and operation == 'generate'):
        quality = 'medium' if operation == 'edit' else 'high'
    speed = options.get('ideogram_speed')
    if parent is not None:
        speed = speed if speed in _USAGE_IDEOGRAM_SPEEDS else 'default'
    else:
        speed = None
    return {
        'model': row.model,
        'images': _usage_image_count(row.image_url) or None,
        'operation': operation,
        'quality': quality,
        'speed': speed,
        'complete': parent is not None,
        'created_at': _usage_iso(row.timestamp),
    }


def _usage_legacy_input(row, rows_by_id):
    """Mirror the estimator's parent-chain input inference for legacy answers."""
    parent_id = row.parent_id
    if parent_id is None:
        return None
    total = 0
    ancestors = {row.id}
    while parent_id is not None:
        if parent_id in ancestors or parent_id not in rows_by_id:
            return None
        ancestors.add(parent_id)
        ancestor = rows_by_id[parent_id]
        if ancestor.role == 'user':
            count = _usage_number(ancestor.tokens_in)
        else:
            output = _usage_number(ancestor.tokens_out)
            content = None if output is None else max(0, output - (_usage_number(ancestor.tokens_thought) or 0))
            count = content
        if count is None:
            return None
        total += count
        parent_id = ancestor.parent_id
    return total


def _usage_answer_calls(row, rows_by_id):
    """Per-request usage rows shaped like the saved `pricing_calls` entries."""
    thought = row.thought_data
    if row.is_encrypted and thought:
        try:
            thought = decrypt_val(thought)
        except Exception:
            thought = None
    metadata = None
    if thought:
        try:
            metadata = json.loads(thought) if isinstance(thought, str) else thought
        except (TypeError, ValueError):
            metadata = None
    if isinstance(metadata, dict) and isinstance(metadata.get('pricing_calls'), list):
        calls = _sanitize_pricing_calls(metadata['pricing_calls']) or []
        for call in calls:
            if isinstance(call, dict) and not call.get('created_at'):
                call['created_at'] = _usage_iso(row.timestamp)
        return calls
    if _usage_is_image_model(row.model):
        return [_usage_legacy_image_request(row, rows_by_id)]
    output = _usage_number(row.tokens_out)
    if not output:
        return None
    input_tokens = _usage_number(row.tokens_in)
    complete = True
    if not input_tokens:
        input_tokens = _usage_legacy_input(row, rows_by_id)
        complete = input_tokens is not None
    return [{
        'model': row.model,
        'input_tokens': input_tokens,
        'output_tokens': output,
        'complete': complete,
        'created_at': _usage_iso(row.timestamp),
    }]


@app.route('/api/usage/costs', methods=['GET'])
@login_required
def handle_usage_costs():
    period = (request.args.get('period') or '1m').strip()
    if period not in _USAGE_PERIOD_DAYS:
        period = '1m'
    days = _USAGE_PERIOD_DAYS[period]
    since = datetime.utcnow() - timedelta(days=days)
    log_force(f"usage cost summary start user={current_user.id} period={period}")
    try:
        rows = (db.session.query(
            Message.id, Message.thread_id, Message.role, Message.model, Message.timestamp,
            Message.tokens_in, Message.tokens_out, Message.tokens_thought, Message.thought_data,
            Message.image_url, Message.prompt_options, Message.is_encrypted, Message.parent_id,
        ).join(Thread, Thread.id == Message.thread_id)
            .filter(Thread.user_id == current_user.id, ~Thread.title.startswith('[LIBRARY]'))
            .all())
        rows_by_id = {row.id: row for row in rows}
        answers = sorted(
            (row for row in rows
             if row.role == 'assistant' and row.timestamp is not None and row.timestamp >= since),
            key=lambda row: row.timestamp,
        )
        truncated = len(answers) > _USAGE_MAX_ANSWERS
        if truncated:
            answers = answers[-_USAGE_MAX_ANSWERS:]
        batch_states = {}
        answer_ids = [row.id for row in answers]
        if answer_ids:
            for job in GeminiBatchJob.query.filter(
                GeminiBatchJob.user_id == current_user.id,
                GeminiBatchJob.assistant_message_id.in_(answer_ids),
            ).with_entities(GeminiBatchJob.assistant_message_id, GeminiBatchJob.state).all():
                batch_states[job.assistant_message_id] = job.state
        messages = []
        unreadable = 0
        for row in answers:
            batch = batch_states.get(row.id) == 'JOB_STATE_SUCCEEDED'
            parent = rows_by_id.get(row.parent_id)
            options = _usage_prompt_options(parent.prompt_options if parent else row.prompt_options)
            if not options:
                options = _usage_prompt_options(row.prompt_options)
            calls = _usage_answer_calls(row, rows_by_id)
            if calls is None:
                unreadable += 1
                continue
            messages.append({
                'id': row.id,
                'role': 'assistant',
                'model': row.model,
                'created_at': _usage_iso(row.timestamp),
                'batch': batch,
                'flex': options.get('gemini_flex') is True,
                'pricing_calls': calls,
            })
        payload = {
            'period': period,
            'days': days,
            'since': _usage_iso(since),
            'answers': len(messages),
            'unreadable': unreadable,
            'truncated': truncated,
            'messages': messages,
        }
        log_force(f"usage cost summary ok user={current_user.id} period={period} "
                  f"answers={len(messages)} unreadable={unreadable} truncated={truncated}")
        return jsonify(payload)
    except Exception as exc:
        db.session.rollback()
        log_force(f"usage cost summary failed user={current_user.id} period={period}: {exc}")
        return jsonify({'error': 'usage_unavailable'}), 500
