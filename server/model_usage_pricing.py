"""Numeric cache and hosted-tool usage; transient tool IDs/queries never enter saved metadata."""

_USAGE_PRICE_FIELDS = ('cache_read_tokens', 'cache_write_tokens', 'cache_write_5m_tokens', 'cache_write_1h_tokens',
                       'cache_write_30m_tokens', 'cache_write_12h_tokens', 'web_search_calls', 'file_search_calls',
                       'code_execution_calls', 'container_sessions', 'grounding_queries')


def _usage_field(obj, key, fallback=None):
    return obj.get(key, fallback) if isinstance(obj, dict) else getattr(obj, key, fallback)


def _usage_cache_counts(usage):
    details = _usage_field(usage, 'input_tokens_details') or _usage_field(usage, 'prompt_tokens_details') or {}
    creation = _usage_field(usage, 'cache_creation') or {}
    writes = _usage_field(details, 'cache_write_tokens_details') or {}
    return {
        'cache_read_tokens': _usage_value(details, 'cached_tokens') if _usage_value(details, 'cached_tokens') is not None else
            _usage_value(usage, 'cache_read_input_tokens', 'cachedContentTokenCount', 'cached_content_token_count', 'prompt_cache_hit_tokens', 'cached_prompt_tokens', 'total_cached_tokens'),
        'cache_write_tokens': _usage_value(details, 'cache_write_tokens') if _usage_value(details, 'cache_write_tokens') is not None else _usage_value(usage, 'cache_creation_input_tokens'),
        'cache_write_5m_tokens': _usage_value(creation, 'ephemeral_5m_input_tokens'),
        'cache_write_1h_tokens': _usage_value(creation, 'ephemeral_1h_input_tokens'),
        'cache_write_30m_tokens': _usage_value(writes, 'ttl_30m_tokens'),
        'cache_write_12h_tokens': _usage_value(writes, 'ttl_12h_tokens'),
    }


def _observe_pricing_payload(call, payload, obj):
    if not hasattr(call, '_price_seen'):
        call._price_seen = set()
        call._query_seen = set()
    configured = (obj.get('tools') or [])
    if any(isinstance(t, dict) and str(t.get('type', '')).startswith('web_search_preview') for t in configured):
        call.record['search_preview'] = True
    items = obj.get('output') or []
    item = payload.get('item')
    if isinstance(item, dict):
        items = [item, *items]
    for item in items:
        if not isinstance(item, dict):
            continue
        kind, identity = item.get('type'), item.get('id')
        field = {'web_search_call': 'web_search_calls', 'file_search_call': 'file_search_calls',
                 'code_interpreter_call': 'code_execution_calls'}.get(kind)
        if kind == 'image_generation_call':
            call.record['pricing_incomplete'] = True
        if kind == 'web_search_call' and call.provider == 'openai':
            action = item.get('action', {}).get('type')
            if action not in ('search', 'image_search'):
                continue
        if field is None or not isinstance(identity, str):
            continue
        if item.get('status') in ('failed', 'incomplete'):
            call.record['pricing_incomplete'] = True
            continue
        key = (kind, identity)
        if key not in call._price_seen:
            call._price_seen.add(key)
            call.record[field] = call.record.get(field, 0) + 1
        if kind == 'code_interpreter_call':
            container = item.get('container_id')
            if isinstance(container, str) and ('container', container) not in call._price_seen:
                call._price_seen.add(('container', container))
                call.record['container_sessions'] = call.record.get('container_sessions', 0) + 1
    for candidate in (obj.get('candidates') or []):
        if not isinstance(candidate, dict):
            continue
        grounding = candidate.get('groundingMetadata') or candidate.get('grounding_metadata') or {}
        for query in grounding.get('webSearchQueries', grounding.get('web_search_queries', [])):
            if isinstance(query, str) and query.strip():
                call._query_seen.add(query)
        if call._query_seen:
            call.record['grounding_queries'] = len(call._query_seen)
    # Z.AI supplies search results rather than an authoritative number of billed calls.
    if obj.get('web_search') and str(call.record['model']).startswith('glm-'):
        call.record['web_search_calls'] = max(1, call.record.get('web_search_calls', 0))
        call.record['pricing_incomplete'] = True


def _update_pricing_usage(call, usage):
    counts = _usage_cache_counts(usage)
    server = _usage_field(usage, 'server_tool_use') or {}
    counts.update(grounding_queries=_usage_value(usage, 'grounding_queries'), web_search_calls=_usage_value(server, 'web_search_requests'),
                  code_execution_calls=_usage_value(server, 'code_execution_requests'))
    for key, value in counts.items():
        if value is not None:
            call.record[key] = max(value, call.record.get(key, 0))
    # xAI returns the actual per-request charge, including all cache and tool charges.
    ticks = _usage_field(usage, 'cost_in_usd_ticks')
    if call.provider == 'xai' and isinstance(ticks, int) and not isinstance(ticks, bool) and 0 <= ticks <= 9_000_000_000_000_000:
        call.record['cost_in_usd_ticks'] = max(ticks, call.record.get('cost_in_usd_ticks', 0))


def _configure_pricing_request(call, request):
    try:
        body = request.content
        if len(body) > 262144:
            if re.search(rb'(?<!\\)"(?:cachedContent|cached_content)"\s*:', body):
                call.record['explicit_cache'] = True
            return  # No extra copy/JSON tree of large attachments or chat histories.
        obj = json.loads(body)
    except (ValueError, UnicodeError, AttributeError):
        return
    if not isinstance(obj, dict):
        return
    if obj.get('cachedContent') or obj.get('cached_content'):
        call.record['explicit_cache'] = True
    for tool in (obj.get('tools') or []):
        if not isinstance(tool, dict):
            continue
        kind = tool.get('type', '')
        if str(kind).startswith('web_search_preview'):
            call.record['search_preview'] = True
        if kind in ('web_search_20260209', 'web_fetch_20260209', 'web_search_20260515', 'web_fetch_20260515'):
            call.record['code_execution_free'] = True
        if kind == 'code_interpreter':
            memory = tool.get('container', {})
            if isinstance(memory, dict) and memory.get('memory_limit') in ('1g', '4g', '16g', '64g'):
                call.record['container_memory_gb'] = int(memory['memory_limit'][:-1])
