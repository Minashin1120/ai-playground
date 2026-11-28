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


def _usage_pricing_table():
    global _USAGE_IMAGE_MODELS
    if isinstance(_USAGE_IMAGE_MODELS, dict) and 'models' in _USAGE_IMAGE_MODELS:
        return _USAGE_IMAGE_MODELS
    try:
        pricing_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'model_pricing.json')
        with open(pricing_path, encoding='utf-8') as pricing_file:
            _USAGE_IMAGE_MODELS = json.load(pricing_file)
    except (OSError, ValueError, TypeError) as exc:
        log_force(f"usage cost pricing unavailable: {exc}")
        _USAGE_IMAGE_MODELS = {'aliases': {}, 'models': {}}
    return _USAGE_IMAGE_MODELS


def _usage_model_rate(model):
    if not model:
        return None
    pricing = _usage_pricing_table()
    key = (pricing.get('aliases') or {}).get(model, model)
    rate = (pricing.get('models') or {}).get(key)
    return rate if isinstance(rate, dict) else None


def _usage_is_image_model(model):
    """True when the official rate table bills this model per generated image."""
    rate = _usage_model_rate(model)
    return bool(rate and rate.get('unit') == 'image')


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
    rate = _usage_model_rate(row.model)
    unit = rate.get('unit') if isinstance(rate, dict) else None
    if unit == 'image':
        return [_usage_legacy_image_request(row, rows_by_id)]
    if unit == 'song':
        return [{'model': row.model, 'songs': 1, 'complete': True, 'created_at': _usage_iso(row.timestamp)}]
    if unit in ('video', 'page', 'character', 'duration'):
        call = {'model': row.model, 'complete': False, 'created_at': _usage_iso(row.timestamp)}
        if unit == 'character':
            parent = rows_by_id.get(row.parent_id)
            if parent and parent.content and not parent.is_encrypted:
                call['characters'] = len(parent.content)
        return [call]
    output = _usage_number(row.tokens_out)
    if output is None or (output == 0 and not (isinstance(rate, dict) and rate.get('output') == 0)):
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


_USAGE_KEY_SOURCE_LABELS = {
    'openai_common': 'OpenAI 共通キー',
    'gemini_common': 'Gemini 共通キー（有料枠）',
    'gemini_free': 'Gemini 無料枠キー',
    'anthropic_common': 'Anthropic 共通キー',
    'deepseek_common': 'DeepSeek 共通キー',
    'zai_common': 'Z.AI 共通キー',
    'kimi_common': 'Kimi 共通キー',
    'xai_common': 'xAI 共通キー',
    'mistral_common': 'Mistral 共通キー',
    'ideogram_common': 'Ideogram 共通キー',
    'google_common': 'Google 共通キー',
    'model_specific': 'モデル別キー',
    'vertex_ai': 'Vertex AI',
    'env_fallback': 'サーバー環境変数',
}

_USAGE_PROVIDER_LABELS = {
    'openai': 'OpenAI',
    'gemini': 'Gemini',
    'google': 'Google',
    'anthropic': 'Anthropic',
    'deepseek': 'DeepSeek',
    'zai': 'Z.AI',
    'kimi': 'Kimi',
    'xai': 'xAI',
    'mistral': 'Mistral',
    'ideogram': 'Ideogram',
}


def _usage_user_key_labels(user):
    """Map non-reversible key IDs of the user's current keys to safe display labels."""
    labels = {}
    if not user:
        return labels
    common_fields = (
        ('openai_api_key', 'openai_common'),
        ('gemini_api_key', 'gemini_common'),
        ('gemini_free_api_key', 'gemini_free'),
        ('anthropic_api_key', 'anthropic_common'),
        ('deepseek_api_key', 'deepseek_common'),
        ('zai_api_key', 'zai_common'),
        ('kimi_api_key', 'kimi_common'),
        ('mistral_api_key', 'mistral_common'),
        ('ideogram_api_key', 'ideogram_common'),
        ('xai_api_key', 'xai_common'),
        ('google_api_key', 'google_common'),
    )
    for field_name, source in common_fields:
        raw = decrypt_val(getattr(user, field_name, None))
        if raw and str(raw).strip():
            meta = _pricing_key_metadata(str(raw).strip(), source)
            key_id = meta.get('api_key_id')
            if key_id:
                hint = f" ({meta['api_key_hint']})" if meta.get('api_key_hint') else ''
                labels[key_id] = f"{_USAGE_KEY_SOURCE_LABELS[source]}{hint}"
    model_keys = _load_user_model_api_key_map(user)
    model_groups = {}
    for model_name, raw in model_keys.items():
        if not raw or not str(raw).strip():
            continue
        meta = _pricing_key_metadata(str(raw).strip(), 'model_specific')
        key_id = meta.get('api_key_id')
        if not key_id or key_id in labels:
            continue
        model_groups.setdefault(key_id, {'models': [], 'hint': meta.get('api_key_hint')})
        model_groups[key_id]['models'].append(model_name)
    for key_id, info in model_groups.items():
        models = info['models']
        model_desc = models[0] if len(models) == 1 else f"{models[0]} 等{len(models)}件"
        hint = f" ({info['hint']})" if info.get('hint') else ''
        labels[key_id] = f"モデル別キー: {model_desc}{hint}"
    return labels


def _usage_format_key_label(key_id, key_source, key_hint, user_key_labels):
    if key_id and key_id in user_key_labels:
        return user_key_labels[key_id]
    if key_id == 'vertex_ai' or key_source == 'vertex_ai':
        return 'Vertex AI'
    if key_id and key_id.startswith('env:'):
        prov = key_id.split(':', 1)[1]
        prov_label = _USAGE_PROVIDER_LABELS.get(prov, prov)
        return f"サーバー環境変数 ({prov_label})"
    if key_source == 'env_fallback':
        return 'サーバー環境変数'
    base = _USAGE_KEY_SOURCE_LABELS.get(key_source, '過去のAPIキー')
    if key_hint:
        return f"{base}（過去のキー {key_hint}）"
    if key_id:
        return f"{base}（過去のキー）"
    return '未記録（キー未設定）'


def _usage_attach_api_key_labels(calls, user, fallback_model, gemini_free_tier, user_key_labels, auth_cache):
    for call in calls:
        if not isinstance(call, dict):
            continue
        key_id = call.get('api_key_id')
        if key_id:
            call['api_key_label'] = _usage_format_key_label(
                key_id, call.get('api_key_source'), call.get('api_key_hint'), user_key_labels
            )
            continue
        call_model = call.get('model') or fallback_model or ''
        cache_key = (call_model, bool(gemini_free_tier))
        if cache_key not in auth_cache:
            resolved = _resolve_chat_model_auth(user, call_model, gemini_free_tier=gemini_free_tier)
            auth_cache[cache_key] = _pricing_key_metadata_from_auth(resolved)
        meta = auth_cache[cache_key]
        inferred_id = meta.get('api_key_id')
        if inferred_id:
            call['api_key_id'] = inferred_id
            call['api_key_label'] = _usage_format_key_label(
                inferred_id, meta.get('api_key_source'), meta.get('api_key_hint'), user_key_labels
            )
            call['api_key_inferred'] = True
        else:
            call['api_key_id'] = 'unknown'
            call['api_key_label'] = '未記録（キー未設定）'


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
        user_key_labels = _usage_user_key_labels(current_user)
        auth_cache = {}
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
            _usage_attach_api_key_labels(
                calls, current_user, row.model, bool(options.get('gemini_free_tier')),
                user_key_labels, auth_cache,
            )
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
