"""Per-generation usage records. No prompts, response text, URLs or credentials are retained."""
from contextvars import ContextVar
from datetime import datetime as _usage_clock
import hashlib
import json
import re

_MODEL_USAGE_CONTEXT = ContextVar('model_usage', default=None)

_USAGE_KEY_SOURCES = (
    'model_specific', 'gemini_free', 'vertex_ai', 'env_fallback',
    'openai_common', 'gemini_common', 'anthropic_common', 'deepseek_common',
    'zai_common', 'kimi_common', 'xai_common', 'mistral_common', 'ideogram_common', 'google_common',
)


def _pricing_key_metadata(api_key=None, key_source=None, provider=None):
    """Return non-reversible key identity metadata for cost aggregation; never stores raw keys."""
    source = key_source if key_source in _USAGE_KEY_SOURCES else None
    prov = re.sub(r'[^a-z0-9_]', '', str(provider or '').lower())[:24] or 'unknown'
    if source == 'vertex_ai':
        return {'api_key_id': 'vertex_ai', 'api_key_source': 'vertex_ai'}
    if source == 'env_fallback':
        return {'api_key_id': f'env:{prov}', 'api_key_source': 'env_fallback'}
    raw = str(api_key or '').strip()
    if not raw:
        return {}
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]
    info = {'api_key_id': f'k:{digest}'}
    if source:
        info['api_key_source'] = source
    if len(raw) >= 8 and re.fullmatch(r'[A-Za-z0-9._:/-]+', raw):
        info['api_key_hint'] = '...' + raw[-4:]
    return info


def _pricing_key_metadata_from_auth(resolved):
    if not isinstance(resolved, dict):
        return {}
    return _pricing_key_metadata(
        api_key=resolved.get('api_key'),
        key_source=resolved.get('key_source'),
        provider=resolved.get('provider'),
    )


def _usage_value(value, *names):
    for name in names:
        item = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        if isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 2_147_483_647:
            return item
    return None


def _usage_counts(usage, provider):
    inputs = _usage_value(usage, 'promptTokenCount', 'prompt_token_count', 'prompt_tokens', 'input_tokens')
    if provider == 'anthropic' and inputs is not None:
        inputs += (_usage_value(usage, 'cache_read_input_tokens') or 0) + (_usage_value(usage, 'cache_creation_input_tokens') or 0)
    thought = _usage_value(usage, 'thoughtsTokenCount', 'thoughts_token_count', 'reasoning_tokens')
    for name in ('output_tokens_details', 'completion_tokens_details'):
        details = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if details is not None:
            thought = _usage_value(details, 'reasoning_tokens')
            break
    outputs = _usage_value(usage, 'output_tokens', 'completion_tokens')
    candidates = _usage_value(usage, 'candidatesTokenCount', 'candidates_token_count')
    if candidates is not None:
        outputs = candidates + (thought or 0)
    elif provider == 'xai' and outputs is not None:
        total = _usage_value(usage, 'total_tokens')
        if total is not None and inputs is not None and total >= inputs:
            outputs = total - inputs
    return {'input_tokens': inputs, 'output_tokens': outputs, 'thought_tokens': thought}


class ModelUsageCall:
    def __init__(self, model, provider):
        self.record = {'model': model, 'input_tokens': None, 'output_tokens': None, 'thought_tokens': None,
                       'created_at': _usage_clock.utcnow().isoformat() + 'Z'}
        if provider == 'openai' and ('search-preview' in model or model == 'gpt-5-search-api'):
            self.record['web_search_calls'] = 1
            if 'search-preview' in model:
                self.record['search_preview'] = True
        self.provider = provider
        self.finished = False
        self.rejected = False
        self.response_id = None  # Deduplication only; never serialized or logged.

    def observe(self, payload):
        if not isinstance(payload, dict):
            return
        nested = payload.get('response') or payload.get('message')
        obj = nested if isinstance(nested, dict) else payload
        if isinstance(obj.get('id'), str):
            self.response_id = obj['id']
        _observe_pricing_payload(self, payload, obj)
        usage = obj.get('usageMetadata') or obj.get('usage_metadata') or obj.get('usage')
        if not isinstance(usage, dict):
            return
        self.update_usage(usage)

    def update_usage(self, usage):
        _update_pricing_usage(self, usage)
        counts = _usage_counts(usage, self.provider)
        # Claude message_delta omits input counts; usage events within one response are cumulative.
        for key, value in counts.items():
            if value is not None:
                self.record[key] = max(value, self.record.get(key) or 0)

    def snapshot(self):
        known = (
            'cost_in_usd_ticks' in self.record
            or any(self.record.get(k) is not None for k in (
                'images', 'pages', 'characters', 'songs', 'duration_ms',
                'audio_in_ms', 'audio_out_ms', 'standby_ms',
            ))
            or (self.record['input_tokens'] is not None and self.record['output_tokens'] is not None)
        )
        return {**self.record, 'complete': self.finished and known}


class ModelUsageLedger:
    def __init__(self):
        self.calls = []
        self.default_key_info = {}
        self.provider_key_info = {}

    def set_default_key_info(self, info):
        self.default_key_info = dict(info) if isinstance(info, dict) else {}

    def set_provider_key_info(self, provider, info):
        if provider and isinstance(info, dict):
            self.provider_key_info[str(provider)] = dict(info)

    def begin(self, model, provider):
        call = ModelUsageCall(model, provider)
        self.calls.append(call)
        return call

    def snapshot(self):
        records = []
        seen = {}
        containers = set()
        for call in self.calls:
            if call.rejected:
                continue
            identity = (call.provider, call.response_id)
            record = call.snapshot()
            key_info = self.provider_key_info.get(call.provider) or self.default_key_info
            if key_info:
                for k in ('api_key_id', 'api_key_source', 'api_key_hint'):
                    if k not in record and key_info.get(k):
                        record[k] = key_info[k]
            used = {item[1] for item in getattr(call, '_price_seen', set()) if item[0] == 'container'}
            if used:
                record['container_sessions'] = len(used - containers)
                containers.update(used)
            if call.response_id and identity in seen:
                previous = records[seen[identity]]
                for field in ('input_tokens', 'output_tokens', 'thought_tokens', *_USAGE_PRICE_FIELDS, 'cost_in_usd_ticks'):
                    values = [value for value in (previous.get(field), record.get(field)) if isinstance(value, int)]
                    previous[field] = max(values) if values else None
                previous['complete'] = previous['complete'] or record['complete']
                continue
            if call.response_id:
                seen[identity] = len(records)
            records.append(record)
        return records

    def find_response(self, identity):
        return next((call for call in reversed(self.calls) if call.response_id == identity), None)

    def token_totals(self):
        rows = self.snapshot()
        def total(key):
            values = [row[key] for row in rows if isinstance(row.get(key), int)]
            return sum(values) if values else None
        return {key: total(key) for key in ('input_tokens', 'output_tokens', 'thought_tokens')}


_SANITIZE_INT_FIELDS = (
    'input_tokens', 'output_tokens', 'thought_tokens', 'images', 'container_memory_gb',
    'pages', 'characters', 'songs', 'duration_ms', 'audio_in_ms', 'audio_out_ms',
    'standby_ms', 'text_turns', 'input_images',
)


def _sanitize_pricing_calls(value):
    if not isinstance(value, list):
        return None
    rows = []
    for entry in value[:1024]:
        if not isinstance(entry, dict):
            rows.append({'model': '', 'complete': False})
            continue
        model = entry.get('model')
        model = model if isinstance(model, str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,159}', model) else ''
        row = {'model': model, 'complete': entry.get('complete') is True}
        for key in (*_SANITIZE_INT_FIELDS, *_USAGE_PRICE_FIELDS):
            val = _usage_value(entry, key)
            if val is not None or key in ('input_tokens', 'output_tokens', 'thought_tokens', 'images', 'container_memory_gb', *_USAGE_PRICE_FIELDS):
                row[key] = val
        for key, allowed in {'operation': ('generate', 'edit', 'remove_object', 'remove_background'),
                             'speed': ('turbo', 'default', 'quality'),
                             'quality': ('very_low', 'low', 'medium', 'high', 'auto'),
                             'resolution': ('360p', '480p', '512px', '0.5k', '720p', '1080p', '1k', '2k', '4k'),
                             'size': ('1024x1024', '1024x1536', '1536x1024', 'auto')}.items():
            if entry.get(key) in allowed:
                row[key] = entry[key]
        ticks = entry.get('cost_in_usd_ticks')
        if isinstance(ticks, int) and not isinstance(ticks, bool) and 0 <= ticks <= 9_000_000_000_000_000:
            row['cost_in_usd_ticks'] = ticks
        for flag in ('search_preview', 'pricing_incomplete', 'explicit_cache', 'code_execution_free', 'without_audio'):
            if entry.get(flag) is True:
                row[flag] = True
        date = entry.get('created_at')
        if isinstance(date, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T[0-9:.+Z-]{5,30}', date):
            row['created_at'] = date
        key_id = entry.get('api_key_id')
        if isinstance(key_id, str) and re.fullmatch(r'(?:k:[0-9a-f]{12}|env:[a-z0-9_]{1,24}|vertex_ai)', key_id):
            row['api_key_id'] = key_id
        key_source = entry.get('api_key_source')
        if key_source in _USAGE_KEY_SOURCES:
            row['api_key_source'] = key_source
        key_hint = entry.get('api_key_hint')
        if isinstance(key_hint, str) and re.fullmatch(r'\.\.\.[A-Za-z0-9._:/-]{4}', key_hint):
            row['api_key_hint'] = key_hint
        rows.append(row)
    if len(value) > 1024:
        rows.append({'model': '', 'complete': False})
    return rows


def _pcm_duration_ms(pcm_bytes, rate=24000, channels=1):
    if not pcm_bytes or rate <= 0 or channels <= 0:
        return 0
    return max(0, int(round(len(pcm_bytes) / float(rate * 2 * channels) * 1000.0)))


def _probe_duration_ms(data, demuxer):
    if not data or not demuxer:
        return None
    try:
        info = _probe_media_with_ffmpeg(data, demuxer, timeout=10)
        dur = (info or {}).get('duration')
        if isinstance(dur, (int, float)) and dur > 0:
            return int(round(dur * 1000.0))
    except Exception:
        return None
    return None


def _interactions_usage_fields(usage):
    """Normalize Gemini /v1beta/interactions `usage` object into pricing call fields."""
    if not usage:
        return {}
    fields = {}
    in_tok = _usage_value(usage, 'total_input_tokens', 'input_tokens')
    out_tok = _usage_value(usage, 'total_output_tokens', 'output_tokens')
    thought_tok = _usage_value(usage, 'total_thought_tokens', 'thought_tokens', 'thoughts_tokens')
    if thought_tok is not None and out_tok is not None:
        # Interactions total_output_tokens may exclude thoughts in some payloads; align to output = text+thoughts+media
        pass
    if in_tok is not None:
        fields['input_tokens'] = in_tok
    if out_tok is not None:
        fields['output_tokens'] = out_tok
    if thought_tok is not None:
        fields['thought_tokens'] = thought_tok
    for arr_name, mod_map in (
        ('input_tokens_by_modality', {'text': 'text_input_tokens', 'image': 'image_input_tokens', 'audio': 'audio_input_tokens', 'video': 'video_input_tokens'}),
        ('output_tokens_by_modality', {'text': 'text_output_tokens', 'image': 'image_output_tokens', 'audio': 'audio_output_tokens', 'video': 'video_output_tokens'}),
    ):
        arr = _usage_field(usage, arr_name)
        if isinstance(arr, (list, tuple)):
            for item in arr:
                mod = str(_usage_field(item, 'modality', '') or '').lower()
                tok = _usage_value(item, 'tokens', 'token_count', 'tokenCount')
                key = mod_map.get(mod)
                if key and tok is not None:
                    fields[key] = tok
    return fields


def _record_direct_usage(model, provider, enrich_existing=False, usage=None, key_info=None, **fields):
    """Record or enrich a call in the active ModelUsageLedger (no-op if no ledger context)."""
    ledger = _MODEL_USAGE_CONTEXT.get()
    if ledger is None or not model:
        return None
    call = None
    if enrich_existing:
        for existing in reversed(ledger.calls):
            if existing.record.get('model') == model:
                call = existing
                break
    if call is None:
        call = ledger.begin(model, provider)
    if key_info and not ledger.provider_key_info.get(provider):
        ledger.set_provider_key_info(provider, key_info)
    if usage is not None:
        if isinstance(usage, dict) or hasattr(usage, 'total_input_tokens'):
            fields = {**_interactions_usage_fields(usage), **fields}
        call.update_usage(usage)
    for k, v in fields.items():
        if v is not None:
            call.record[k] = v
    call.finished = True
    return call


def _encode_usage_thought_data(calls, is_encrypted=False):
    clean = _sanitize_pricing_calls(calls)
    if not clean:
        return None
    payload = json.dumps({'pricing_calls': clean}, ensure_ascii=False)
    return encrypt_val(payload) if is_encrypted else payload


def _usage_model_from_request(request):
    """Extract the top-level model without parsing/copying large prompt or attachment strings."""
    path = request.url.path
    if '/models/' in path and re.search(r':(?:streamGenerateContent|generateContent)$', path):
        return path.rsplit('/models/', 1)[1].split(':', 1)[0]
    if not path.rstrip('/').endswith(('/responses', '/chat/completions', '/messages')):
        return None
    try:
        data = request.content
    except Exception:
        return None
    index = depth = 0
    punctuation = re.compile(rb'["{}\[\]]')
    while index < len(data):
        match = punctuation.search(data, index)
        if not match:
            break
        start = match.start()
        char = data[start]
        index = start + 1
        if char != 34:
            depth += 1 if char in (123, 91) else -1
            continue
        while True:
            end = data.find(b'"', index)
            if end < 0:
                return None
            escapes = 0
            pos = end - 1
            while pos >= index - 1 and data[pos] == 92:
                escapes += 1
                pos -= 1
            index = end + 1
            if escapes % 2 == 0:
                break
        if depth == 1 and end - start == 6 and data[start:end + 1] == b'"model"':
            value = re.match(rb'\s*:\s*"([a-zA-Z0-9][a-zA-Z0-9._:/-]{0,159})"', data[index:index + 200])
            return value[1].decode('ascii') if value else None
    return None
