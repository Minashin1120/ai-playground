"""Provider token counting for the composer's current text, quote and attachments.

Never generates an answer or uploads a persistent provider file. Large/unsupported
inputs and unavailable counters fall back to the local estimate, labelled as such.
"""
from collections import OrderedDict
from urllib.parse import quote

_DRAFT_COUNT_CACHE = OrderedDict()
_DRAFT_COUNT_LOCK = threading.Lock()
_DRAFT_COUNT_MAX_BYTES = 16 * 1024 * 1024


def _draft_counter_request(provider, model, text, attachments):
    if provider == 'gemini':
        parts = []
        for f in attachments:
            if f.get('text') is not None:
                parts.append({'text': f['text']})
            else:
                parts.append({'inlineData': {'mimeType': f['mime'], 'data': base64.b64encode(f['bytes']).decode('ascii')}})
        parts.append({'text': text})
        return {'contents': [{'role': 'user', 'parts': parts}]}
    if provider == 'anthropic':
        parts = []
        for f in attachments:
            if f.get('text') is not None:
                parts.append({'type': 'text', 'text': f['text']})
            else:
                parts.append({'type': 'document' if f['mime'] == 'application/pdf' else 'image',
                              'source': {'type': 'base64', 'media_type': f['mime'], 'data': base64.b64encode(f['bytes']).decode('ascii')}})
        if text or not parts:
            parts.append({'type': 'text', 'text': text or '(empty)'})
        return {'model': model, 'messages': [{'role': 'user', 'content': parts}]}
    if provider == 'openai':
        parts = []
        for f in attachments:
            if f.get('text') is not None:
                parts.append({'type': 'input_text', 'text': f['text']})
            elif f['mime'].startswith('image/'):
                parts.append({'type': 'input_image', 'image_url': 'data:' + f['mime'] + ';base64,' + base64.b64encode(f['bytes']).decode('ascii')})
            else:
                parts.append({'type': 'input_file', 'filename': f['name'], 'file_data': 'data:' + f['mime'] + ';base64,' + base64.b64encode(f['bytes']).decode('ascii')})
        if text or not parts:
            parts.append({'type': 'input_text', 'text': text})
        return {'model': model, 'input': [{'role': 'user', 'content': parts}]}
    combined = '\n\n'.join([text] + [f['text'] for f in attachments])
    if provider == 'xai':
        return {'model': model, 'text': combined}
    return {'model': model, 'messages': [{'role': 'user', 'content': combined}]}


def _draft_count_call(user, resolved, model, text, attachments):
    provider = resolved['provider']
    payload = _draft_counter_request(provider, model, text, attachments)
    # Only digests and numeric results live in this bounded, short-lived cache.
    identity = json.dumps([user.id, resolved.get('api_key'), resolved.get('gemini_runtime'), payload], sort_keys=True, default=str)
    cache_key = hashlib.sha256(identity.encode()).digest()
    del identity
    now = time.monotonic()
    with _DRAFT_COUNT_LOCK:
        cached = _DRAFT_COUNT_CACHE.get(cache_key)
        if cached and now - cached[0] < 60:
            _DRAFT_COUNT_CACHE.move_to_end(cache_key)
            return cached[1]
    if provider == 'gemini' and (resolved.get('gemini_runtime') or {}).get('backend') == 'vertex_ai':
        runtime = resolved['gemini_runtime']
        client = _get_gemini_client(api_key=resolved.get('api_key'), backend=runtime.get('backend'),
            vertex_project=runtime.get('vertex_project'), vertex_location=runtime.get('vertex_location'),
            vertex_credentials_json=runtime.get('vertex_credentials_json'))
        contents = [types.Content(role='user', parts=[types.Part.from_text(text=f['text']) if f.get('text') is not None
                    else types.Part.from_bytes(data=f['bytes'], mime_type=f['mime']) for f in attachments] + [types.Part.from_text(text=text)])]
        value = client.models.count_tokens(model=model, contents=contents, config=types.CountTokensConfig(http_options=types.HttpOptions(timeout=15000))).total_tokens
    else:
        urls = {'gemini': 'https://generativelanguage.googleapis.com/v1beta/models/' + quote(model, safe='') + ':countTokens',
                'anthropic': 'https://api.anthropic.com/v1/messages/count_tokens',
                'openai': 'https://api.openai.com/v1/responses/input_tokens',
                'xai': 'https://api.x.ai/v1/tokenize-text',
                'kimi': 'https://api.moonshot.ai/v1/tokenizers/estimate-token-count'}
        key = resolved['api_key']
        headers = {'x-goog-api-key': key} if provider == 'gemini' else ({'x-api-key': key, 'anthropic-version': '2023-06-01'}
                    if provider == 'anthropic' else {'Authorization': 'Bearer ' + key})
        response = requests.post(urls[provider], json=payload, headers=headers, timeout=(5, 15), allow_redirects=False)
        try:
            response.raise_for_status()
            data = response.json()
            value = (data.get('totalTokens') if provider == 'gemini' else len(data['token_ids']) if provider == 'xai'
                     else data['data']['total_tokens'] if provider == 'kimi' else data.get('input_tokens'))
        finally:
            response.close()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('invalid token count')
    with _DRAFT_COUNT_LOCK:
        _DRAFT_COUNT_CACHE[cache_key] = (now, value)
        _DRAFT_COUNT_CACHE.move_to_end(cache_key)
        while len(_DRAFT_COUNT_CACHE) > 256:
            _DRAFT_COUNT_CACHE.popitem(last=False)
    return value


def _provider_draft_estimate(user, model, text, references):
    """Return an API count, or None so the caller can supply a labelled approximation."""
    provider = get_model_api_provider(model)
    if provider not in {'gemini', 'anthropic', 'openai', 'xai', 'kimi'}:
        return None
    # Image generation / STT etc. have different input and billing contracts.
    if any(word in model.lower() for word in ('image', 'banana', 'imagine', 'tts', 'transcribe', 'search', 'agent', 'deep-research')):
        return None
    resolved = _resolve_chat_model_auth(user, model)
    if resolved.get('error_code') or (not resolved.get('api_key') and (resolved.get('gemini_runtime') or {}).get('backend') != 'vertex_ai'):
        return None
    op = secrets.token_hex(16)
    _record_server_diagnostic(user.id, 'state', route='/api/token_estimate', provider=provider, phase='start', op=op)
    counted = False
    try:
        attachments = []
        size = len(text.encode())
        if size > _DRAFT_COUNT_MAX_BYTES:
            return None
        for ref in references:
            info = _get_file_disk_info(ref)
            if not info.get('exists') or size + int(info.get('size') or 0) > _DRAFT_COUNT_MAX_BYTES:
                return None
            bits = _load_user_file_bytes(ref, info)
            if bits is None:
                return None
            size += len(bits)
            if size > _DRAFT_COUNT_MAX_BYTES:
                return None
            name = os.path.basename(ref)
            mime = _normalize_media_mime(name, mimetypes.guess_type(name)[0]) or 'application/octet-stream'
            ext = os.path.splitext(name)[1].lower()
            f = {'name': name, 'mime': mime, 'bytes': bits}
            if mime.startswith('text/') or ext in _TEXT_LIKE_UPLOAD_EXTS:
                f['text'] = '[File: ' + name + ']\n' + _decode_text_bytes_for_prompt(bits)
            elif ext == '.docx' and provider != 'openai':
                f['text'] = '[File: ' + name + ']\n' + _extract_docx_as_numbered(bits)
            elif not (mime.startswith('image/') and provider in {'gemini', 'anthropic', 'openai'} or
                      mime == 'application/pdf' and provider in {'gemini', 'anthropic', 'openai'} or
                      provider == 'gemini' and mime.startswith(('audio/', 'video/')) or
                      provider == 'openai' and ext == '.docx'):
                return None
            attachments.append(f)
        prompt = _draft_count_call(user, resolved, model, text, [])
        total = _draft_count_call(user, resolved, model, text, attachments) if attachments else prompt
        if total < prompt:
            return None
        counted = True
        _record_server_diagnostic(user.id, 'state', route='/api/token_estimate', provider=provider, phase='done', op=op, files=len(references))
        return {'countable': True, 'tokens_total': total, 'tokens_prompt': prompt, 'tokens_files': total - prompt,
                'files_total': len(references), 'files_counted': len(references), 'files_non_text': 0, 'files_missing': 0,
                'files_error': 0, 'estimate_method': 'provider', 'estimate_scope': 'draft'}
    except Exception:
        # No provider error text, names, prompt, keys or payloads enter the logs.
        return None
    finally:
        if not counted:
            _record_server_diagnostic(user.id, 'state', route='/api/token_estimate', provider=provider, phase='error', op=op)
