# -----------------------------------------------------------
# Batch API routes (Gemini / OpenAI / xAI / Z.AI)
# 状態取得・履歴一覧・停止・削除。server/routes_chat.py から分割。
# -----------------------------------------------------------
# region: batch model gates and state helpers
def _is_gemini_batch_model(model_key):
    """Gemini API models whose generateContent endpoint supports Batch API."""
    model_l = str(model_key or '').strip().lower()
    if not (model_l.startswith('gemini-') and model_key in ALL_VALID_MODEL_IDS):
        return False
    return not any(marker in model_l for marker in (
        'embedding', 'video', 'veo', 'music', 'lyria', 'native-audio',
        'tts', 'live', 'transcribe', 'agent', 'deep-research', 'robotics',
        'computer-use'
    ))

def _is_openai_batch_model(model_key):
    """OpenAI GPT models with the Batch endpoint documented by OpenAI."""
    model_l = str(model_key or '').strip().lower()
    if model_key not in ALL_VALID_MODEL_IDS or not model_l.startswith('gpt-'):
        return False
    return not any(marker in model_l for marker in (
        'image', 'audio', 'tts', 'transcribe', 'realtime', 'search',
    ))

def _is_xai_batch_model(model_key):
    """xAI text models that can be submitted through the xAI Batch API."""
    model_l = str(model_key or '').strip().lower()
    if model_key not in ALL_VALID_MODEL_IDS or not model_l.startswith('grok-'):
        return False
    return not any(marker in model_l for marker in (
        'image', 'video', 'voice', 'audio', 'tts', 'realtime',
    ))

def _is_zai_batch_model(model_key):
    """Z.AI GLM chat models that use the Batch Chat Completions endpoint."""
    model_l = str(model_key or '').strip().lower()
    if model_key not in ALL_VALID_MODEL_IDS or not model_l.startswith('glm-'):
        return False
    return not any(marker in model_l for marker in (
        'image', 'audio', 'tts', 'transcribe', 'realtime', 'video',
        'embedding', 'ocr',
    ))

def _is_batch_model(model_key):
    return (
        _is_gemini_batch_model(model_key)
        or _is_openai_batch_model(model_key)
        or _is_xai_batch_model(model_key)
        or _is_zai_batch_model(model_key)
    )

def _openai_batch_state_label(raw_status):
    return {
        'validating': 'JOB_STATE_VALIDATING',
        'in_progress': 'JOB_STATE_RUNNING',
        'finalizing': 'JOB_STATE_FINALIZING',
        'completed': 'JOB_STATE_SUCCEEDED',
        'failed': 'JOB_STATE_FAILED',
        'expired': 'JOB_STATE_EXPIRED',
        'cancelling': 'JOB_STATE_CANCELLING',
        'cancelled': 'JOB_STATE_CANCELLED',
    }.get(str(raw_status or '').strip().lower(), 'JOB_STATE_PENDING')

def _batch_json_value(value):
    """Convert google-genai Pydantic objects to REST/SDK JSON safely."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode('ascii')
    if isinstance(value, dict):
        return {str(k): _batch_json_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_batch_json_value(v) for v in value]
    for method_name in ('model_dump', 'to_dict'):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method(mode='json', by_alias=True, exclude_none=True)
            except TypeError:
                try:
                    dumped = method(by_alias=True, exclude_none=True)
                except TypeError:
                    dumped = method()
            return _batch_json_value(dumped)
    return str(value)

def _batch_field(value, *names):
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value.get(name)
    for name in names:
        current = getattr(value, name, None)
        if current is not None:
            return current
    return None

def _normalize_batch_state(raw_state):
    """Normalize REST/SDK enum representations to Gemini's canonical state."""
    value = raw_state
    if isinstance(value, dict):
        value = value.get('name') or value.get('value') or value.get('state')
    else:
        value = getattr(value, 'name', None) or getattr(value, 'value', None) or value
    state = str(value or '').strip()
    if '.' in state:
        state = state.rsplit('.', 1)[-1]
    state = state.replace('-', '_').replace(' ', '_').upper()
    if state.startswith('STATE_'):
        state = state[len('STATE_'):]
    if state.startswith('JOB_STATE_BATCH_STATE_'):
        state = f"JOB_STATE_{state[len('JOB_STATE_BATCH_STATE_') :]}"
    elif state.startswith('BATCH_STATE_'):
        state = f"JOB_STATE_{state[len('BATCH_STATE_') :]}"
    aliases = {
        'COMPLETED': 'JOB_STATE_SUCCEEDED',
        'SUCCESS': 'JOB_STATE_SUCCEEDED',
        'SUCCEEDED': 'JOB_STATE_SUCCEEDED',
    }
    if state in aliases:
        return aliases[state]
    if state.startswith('JOB_STATE_'):
        return state
    return f'JOB_STATE_{state}' if state else ''

def _gemini_operation_state(provider_payload, fallback_state):
    """Resolve Gemini Batch state from either Batch metadata or Operation result."""
    metadata = _batch_field(provider_payload, 'metadata') or {}
    raw_state = (
        _batch_field(provider_payload, 'state')
        or _batch_field(metadata, 'state')
    )
    if raw_state:
        return _normalize_batch_state(raw_state)

    # Operation metadata is service-specific and may be omitted. Per the
    # long-running Operation contract, done=true has exactly one of error or
    # response, so it is sufficient to determine the terminal state.
    if _batch_field(provider_payload, 'done') is True:
        if _batch_field(provider_payload, 'error') is not None:
            return 'JOB_STATE_FAILED'
        if _batch_field(provider_payload, 'response') is not None:
            return 'JOB_STATE_SUCCEEDED'
    return _normalize_batch_state(fallback_state)

def _batch_state_label(state):
    labels = {
        'JOB_STATE_QUEUED': 'Batch APIへの送信待ちです',
        'JOB_STATE_VALIDATING': 'Batch APIで入力を検証中です',
        'JOB_STATE_PENDING': 'Batch APIに送信済みです。処理待機中です',
        'JOB_STATE_RUNNING': 'Batch APIで処理中です',
        'JOB_STATE_FINALIZING': 'Batch処理の結果を準備中です',
        'JOB_STATE_SUCCEEDED': 'Batch処理が完了しました',
        'JOB_STATE_FAILED': 'Batch処理に失敗しました',
        'JOB_STATE_CANCELLING': 'Batch処理をキャンセル中です',
        'JOB_STATE_CANCELLED': 'Batch処理がキャンセルされました',
        'JOB_STATE_EXPIRED': 'Batch処理の有効期限が切れました',
    }
    return labels.get(_normalize_batch_state(state), 'Batch処理の状態を確認中です')

# region: batch routes
@app.route('/api/gemini/batch/status', methods=['GET'])
@login_required
def gemini_batch_status_api():
    """Refresh the user's provider Batch jobs and return banner-worthy completions."""
    rows = GeminiBatchJob.query.filter_by(user_id=current_user.id).filter(
        GeminiBatchJob.notified_at.is_(None) | ~GeminiBatchJob.state.in_({
            'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'
        })
    ).order_by(GeminiBatchJob.created_at.asc()).limit(50).all()
    terminal_states = {
        'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'
    }
    changed_thread_ids = set()
    completed = []
    active = []

    def _set_message_payload(message, content, thought=None, image_refs=None, tokens_out=None, tokens_thought=0, usage=None, billing_payload=None):
        is_enc = bool(message.is_encrypted)
        message.content = encrypt_val(content) if is_enc else content
        thought_record = {'text': thought} if thought else {}
        pricing_input = _cost_input_tokens(usage)
        if pricing_input is not None:
            thought_record['pricing_input_tokens'] = pricing_input
        if usage is not None:
            model = getattr(message, 'model', '') or ''
            provider = 'google' if model.startswith('gemini') else 'xai' if model.startswith('grok') else 'openai'
            call = ModelUsageCall(model, provider)
            call.update_usage(usage)
            if hasattr(billing_payload, 'model_dump'):
                billing_payload = billing_payload.model_dump(mode='json', by_alias=True)
            if isinstance(billing_payload, dict):
                call.observe(billing_payload)
            call.finished = True
            thought_record['pricing_calls'] = [call.snapshot()]
            if call.record['output_tokens'] is not None:
                tokens_out = call.record['output_tokens']
            if call.record['thought_tokens'] is not None:
                tokens_thought = call.record['thought_tokens']
        if thought_record:
            thought_payload = json.dumps(thought_record, ensure_ascii=False)
            message.thought_data = encrypt_val(thought_payload) if is_enc else thought_payload
        message.image_url = json.dumps(image_refs, ensure_ascii=False) if image_refs else None
        if tokens_out is not None:
            message.tokens_out = int(tokens_out or 0)
            message.tokens = sum_token_counts(None, message.tokens_out)
        message.tokens_thought = int(tokens_thought or 0)

    def _terminal_message(row, state, error_text=None, response_payload=None):
        message = Message.query.filter_by(
            id=row.assistant_message_id, thread_id=row.thread_id
        ).first()
        if not message:
            return
        if state == 'JOB_STATE_SUCCEEDED':
            response = _batch_field(response_payload, 'response') or response_payload or {}
            candidates = _batch_field(response, 'candidates') or []
            candidate = candidates[0] if candidates else {}
            content_obj = _batch_field(candidate, 'content') or {}
            parts = _batch_field(content_obj, 'parts') or []
            text_parts = []
            thought_parts = []
            image_refs = []
            for part in parts:
                part_text = _batch_field(part, 'text')
                # Thought images are intermediate reasoning artifacts, just as
                # in the non-Batch Gemini image response path.
                if _batch_field(part, 'thought') is True:
                    if part_text:
                        thought_parts.append(str(part_text))
                    continue
                if part_text:
                    text_parts.append(str(part_text))
                inline_data = _batch_field(part, 'inlineData', 'inline_data')
                if inline_data:
                    raw_data = _batch_field(inline_data, 'data')
                    mime = str(_batch_field(inline_data, 'mimeType', 'mime_type') or 'image/png')
                    if raw_data:
                        try:
                            if isinstance(raw_data, str):
                                raw_data = raw_data.split(',', 1)[-1]
                                raw_data = _decode_base64_limited(raw_data, 50 * 1024 * 1024)
                            ext = {'image/jpeg': 'jpg', 'image/webp': 'webp'}.get(mime, 'png')
                            filename = f"batch_{int(time.time())}_{secrets.token_hex(5)}.{ext}"
                            _save_user_generated_bytes(current_user.id, raw_data, filename, bool(message.is_encrypted))
                            image_refs.append(f"{current_user.id}/{filename}")
                        except Exception as image_error:
                            logger.warning('Gemini Batch image save failed: %s', image_error)
            content = ''.join(text_parts).strip()
            if image_refs:
                content = (content + '\n\n' if content else '') + '\n'.join(
                    f'![Image](/files/{ref})' for ref in image_refs
                )
            if not content:
                content = 'Batch処理が完了しましたが、回答本文はありませんでした。'
            usage = _batch_field(response, 'usageMetadata', 'usage_metadata') or {}
            tokens_out = _batch_field(usage, 'candidatesTokenCount', 'candidates_token_count')
            tokens_thought = _batch_field(usage, 'thoughtsTokenCount', 'thoughts_token_count') or 0
            _set_message_payload(
                message, content,
                '\n'.join(thought_parts).strip() or None,
                image_refs,
                tokens_out,
                tokens_thought,
                usage=usage, billing_payload=response,
            )
        else:
            detail = str(error_text or _batch_state_label(state))[:4000]
            _set_message_payload(message, f'**{_batch_state_label(state)}**\n\n{detail}')

    def _terminal_openai_message(row, state, error_text=None, response_payload=None):
        message = Message.query.filter_by(
            id=row.assistant_message_id, thread_id=row.thread_id
        ).first()
        if not message:
            return
        if state == 'JOB_STATE_SUCCEEDED':
            response = _batch_field(response_payload, 'response') or response_payload or {}
            body = _batch_field(response, 'body') or {}
            content = _extract_openai_response_text(body).strip()
            if not content:
                content = 'Batch処理が完了しましたが、回答本文はありませんでした。'
            usage = _batch_field(body, 'usage') or {}
            tokens_out = _batch_field(usage, 'output_tokens')
            _set_message_payload(message, content, tokens_out=tokens_out, usage=usage, billing_payload=body)
        else:
            detail = str(error_text or _batch_state_label(state))[:4000]
            _set_message_payload(message, f'**{_batch_state_label(state)}**\n\n{detail}')

    def _terminal_zai_message(row, state, error_text=None, response_payload=None):
        message = Message.query.filter_by(
            id=row.assistant_message_id, thread_id=row.thread_id
        ).first()
        if not message:
            return
        if state == 'JOB_STATE_SUCCEEDED':
            response = _batch_field(response_payload, 'response') or response_payload or {}
            body = _batch_field(response, 'body') or {}
            choices = _batch_field(body, 'choices') or []
            choice = choices[0] if choices else {}
            answer = _batch_field(choice, 'message') or {}
            content = _batch_field(answer, 'content')
            if isinstance(content, list):
                content = ''.join(
                    str(_batch_field(part, 'text') or '')
                    for part in content
                    if isinstance(part, (dict, list)) or hasattr(part, 'text')
                )
            content = str(content or '').strip()
            if not content:
                content = 'Batch処理が完了しましたが、回答本文はありませんでした。'
            usage = _batch_field(body, 'usage') or {}
            tokens_out = _batch_field(usage, 'completion_tokens')
            tokens_thought = _batch_field(
                _batch_field(usage, 'completion_tokens_details') or {},
                'reasoning_tokens',
            ) or 0
            thought = _batch_field(answer, 'reasoning_content')
            _set_message_payload(
                message,
                content,
                str(thought).strip() if thought else None,
                tokens_out=tokens_out,
                tokens_thought=tokens_thought,
                usage=usage, billing_payload=body,
            )
        else:
            detail = str(error_text or _batch_state_label(state))[:4000]
            _set_message_payload(message, f'**{_batch_state_label(state)}**\n\n{detail}')

    def _terminal_xai_message(row, state, error_text=None, response_payload=None):
        message = Message.query.filter_by(
            id=row.assistant_message_id, thread_id=row.thread_id
        ).first()
        if not message:
            return
        if state == 'JOB_STATE_SUCCEEDED':
            batch_result = _batch_field(response_payload, 'batch_result') or {}
            response = _batch_field(batch_result, 'response') or {}
            completion = (
                _batch_field(response, 'chat_get_completion')
                or _batch_field(response, 'responses')
                or response
            )
            choices = _batch_field(completion, 'choices') or []
            choice = choices[0] if choices else {}
            answer = _batch_field(choice, 'message') or {}
            content = _batch_field(answer, 'content')
            if isinstance(content, list):
                content = ''.join(
                    str(_batch_field(part, 'text') or '')
                    for part in content
                    if isinstance(part, (dict, list)) or hasattr(part, 'text')
                )
            if not content:
                content = _batch_field(completion, 'output_text')
            if not content:
                output_items = _batch_field(completion, 'output') or []
                output_parts = []
                for item in output_items:
                    for part in (_batch_field(item, 'content') or []):
                        part_text = _batch_field(part, 'text')
                        if part_text:
                            output_parts.append(str(part_text))
                content = ''.join(output_parts)
            thought = _batch_field(answer, 'reasoning_content')
            usage = _batch_field(completion, 'usage') or {}
            tokens_out = (
                _batch_field(usage, 'completion_tokens', 'output_tokens')
            )
            tokens_thought = _batch_field(
                _batch_field(usage, 'completion_tokens_details') or {},
                'reasoning_tokens'
            ) or 0
            content = str(content or '').strip()
            if not content:
                content = 'Batch処理が完了しましたが、回答本文はありませんでした。'
            _set_message_payload(
                message,
                content,
                str(thought).strip() if thought else None,
                tokens_out=tokens_out,
                tokens_thought=tokens_thought,
                usage=usage, billing_payload=completion,
            )
        else:
            detail = str(error_text or _batch_state_label(state))[:4000]
            _set_message_payload(message, f'**{_batch_state_label(state)}**\n\n{detail}')

    def _openai_batch_state(raw_status):
        return {
            'validating': 'JOB_STATE_VALIDATING',
            'in_progress': 'JOB_STATE_RUNNING',
            'finalizing': 'JOB_STATE_FINALIZING',
            'completed': 'JOB_STATE_SUCCEEDED',
            'failed': 'JOB_STATE_FAILED',
            'expired': 'JOB_STATE_EXPIRED',
            'cancelling': 'JOB_STATE_CANCELLING',
            'cancelled': 'JOB_STATE_CANCELLED',
        }.get(str(raw_status or '').strip().lower(), 'JOB_STATE_PENDING')

    def _openai_file_text(client, file_id):
        response = client.files.content(file_id)
        raw = getattr(response, 'text', None)
        if callable(raw):
            raw = raw()
        if raw is None and hasattr(response, 'read'):
            raw = response.read()
        return str(raw or '')

    def _openai_result_line(raw_text):
        for line in str(raw_text or '').splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    return value
        return None

    def _gemini_result_line(raw_text):
        """Normalize one Gemini result-file JSONL line to an inline response."""
        for line in str(raw_text or '').splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            # File results may be wrapped as {key, response}, while the API
            # can also return a bare GenerateContentResponse object.
            if _batch_field(value, 'response') or _batch_field(value, 'error'):
                return value
            return {'response': value}
        return None

    def _gemini_result_payload(provider_payload, api_key):
        """Return the first Gemini result from inline or downloadable output."""
        result_container = (
            _batch_field(provider_payload, 'response')
            or _batch_field(provider_payload, 'dest')
            or {}
        )
        responses = _batch_field(result_container, 'inlinedResponses', 'inlined_responses') or []
        if isinstance(responses, dict):
            responses = _batch_field(
                responses, 'inlinedResponses', 'inlined_responses', 'responses'
            ) or []
        if responses:
            return responses[0]

        result_file = _batch_field(
            result_container,
            'responsesFile', 'responses_file', 'fileName', 'file_name'
        )
        if not result_file:
            return None
        result_file = str(result_file).lstrip('/')
        download_url = (
            'https://generativelanguage.googleapis.com/download/v1beta/'
            f'{result_file}:download?alt=media'
        )
        result_response = requests.get(
            download_url,
            headers={'x-goog-api-key': str(api_key)},
            timeout=30,
        )
        if not result_response.ok:
            try:
                detail = _batch_field(result_response.json().get('error') or {}, 'message')
            except Exception:
                detail = None
            raise RuntimeError(detail or f'Gemini Batch結果ファイル HTTP {result_response.status_code}')
        return _gemini_result_line(result_response.content.decode('utf-8', errors='replace'))

    def _xai_batch_state(provider_payload):
        """Map xAI's aggregate counters to the shared canonical states."""
        state = _batch_field(provider_payload, 'state') or {}
        def _count(name):
            try:
                return max(0, int(_batch_field(state, name) or 0))
            except (TypeError, ValueError):
                return 0
        pending = _count('num_pending')
        if pending > 0:
            return 'JOB_STATE_RUNNING' if _count('num_success') or _count('num_error') else 'JOB_STATE_PENDING'
        if _count('num_error'):
            return 'JOB_STATE_FAILED'
        if _count('num_cancelled') and not _count('num_success'):
            return 'JOB_STATE_CANCELLED'
        if _count('num_success'):
            return 'JOB_STATE_SUCCEEDED'
        if _batch_field(provider_payload, 'cancel_by_xai_message'):
            return 'JOB_STATE_CANCELLED'
        return 'JOB_STATE_PENDING'

    def _xai_batch_result(api_key, batch_id, request_id):
        """Find this chat's result, following xAI's pagination token."""
        base_url = f"https://{_XAI_API_HOST}/v1/batches/{quote(str(batch_id), safe='')}/results"
        pagination_token = None
        for _ in range(1000):
            params = {'limit': 100}
            if pagination_token:
                params['pagination_token'] = pagination_token
            result_response = requests.get(
                base_url,
                headers={'Authorization': f'Bearer {str(api_key)}'},
                params=params,
                timeout=30,
            )
            payload = result_response.json() if result_response.content else {}
            if not result_response.ok:
                detail = _batch_field(payload, 'error') or payload
                if isinstance(detail, dict):
                    detail = _batch_field(detail, 'message') or detail
                raise RuntimeError(detail or f'xAI Batch結果 HTTP {result_response.status_code}')
            for result in _batch_field(payload, 'results') or []:
                if str(_batch_field(result, 'batch_request_id') or '') == str(request_id):
                    return result
            pagination_token = _batch_field(payload, 'pagination_token')
            if not pagination_token:
                return None
        raise RuntimeError('xAI Batch結果のページ数が上限を超えました')

    for row in rows:
        public_thread_id = getattr(getattr(row, 'thread', None), 'public_id', None) or str(row.thread_id)
        old_state = row.state
        old_status = row.status_text
        state = _normalize_batch_state(row.state or 'JOB_STATE_QUEUED')
        provider_payload = None
        stale_without_provider = (
            not row.provider_job_name
            and state in {'JOB_STATE_QUEUED', 'JOB_STATE_RUNNING'}
            and row.updated_at
            and (datetime.utcnow() - row.updated_at).total_seconds() > 180
        )
        if stale_without_provider:
            row.state = 'JOB_STATE_FAILED'
            row.status_text = 'Batch APIへの送信がタイムアウトしました'
            row.error = 'Batch APIのジョブ名を取得できないまま時間切れになりました。再送してください。'
            row.completed_at = row.completed_at or datetime.utcnow()
            _terminal_message(row, row.state, error_text=row.error)
            state = row.state
        # A terminal row with no notification may have reached the provider's
        # completed state before result persistence succeeded. Re-poll it so
        # existing stuck jobs can still retrieve their answer.
        if row.provider_job_name and (state not in terminal_states or row.notified_at is None):
            try:
                provider = row.provider or get_model_api_provider(row.model)
                resolved = _resolve_chat_model_auth(current_user, row.model)
                if provider == 'xai':
                    api_key = resolved.get('api_key')
                    if not api_key:
                        raise RuntimeError('xAI APIキーを確認してください')
                    batch_url = f"https://{_XAI_API_HOST}/v1/batches/{quote(str(row.provider_job_name), safe='')}"
                    provider_response = requests.get(
                        batch_url,
                        headers={'Authorization': f'Bearer {str(api_key)}'},
                        timeout=30,
                    )
                    provider_payload = provider_response.json() if provider_response.content else {}
                    if not provider_response.ok:
                        detail = _batch_field(provider_payload, 'error') or provider_payload
                        if isinstance(detail, dict):
                            detail = _batch_field(detail, 'message') or detail
                        raise RuntimeError(detail or f'xAI Batch API HTTP {provider_response.status_code}')
                    state = _xai_batch_state(provider_payload)
                    row.state = state
                    row.status_text = _batch_state_label(state)
                    if state in terminal_states:
                        if state == 'JOB_STATE_SUCCEEDED':
                            result_payload = _xai_batch_result(api_key, row.provider_job_name, row.job_id)
                            if not result_payload:
                                raise RuntimeError('xAI Batch APIから回答データが返されませんでした')
                            result_error = _batch_field(_batch_field(result_payload, 'batch_result') or {}, 'error')
                            if result_error:
                                row.state = 'JOB_STATE_FAILED'
                                row.error = str(result_error)
                                row.status_text = _batch_state_label(row.state)
                                row.completed_at = row.completed_at or datetime.utcnow()
                                _terminal_xai_message(row, row.state, error_text=row.error)
                            else:
                                row.completed_at = row.completed_at or datetime.utcnow()
                                _terminal_xai_message(row, state, response_payload=result_payload)
                        else:
                            row.completed_at = row.completed_at or datetime.utcnow()
                            error_text = (
                                _batch_field(provider_payload, 'cancel_by_xai_message')
                                or _batch_field(provider_payload, 'error')
                                or row.status_text
                            )
                            _terminal_xai_message(row, state, error_text=error_text)
                elif provider == 'openai':
                    api_key = resolved.get('api_key')
                    if not api_key:
                        raise RuntimeError('OpenAI APIキーを確認してください')
                    client = _get_openai_client(api_key, base_url=None)
                    provider_payload = client.batches.retrieve(row.provider_job_name)
                    state = _openai_batch_state(_batch_field(provider_payload, 'status'))
                    row.state = state
                    row.status_text = _batch_state_label(state)
                    row.output_file_id = _batch_field(provider_payload, 'output_file_id') or row.output_file_id
                    row.error_file_id = _batch_field(provider_payload, 'error_file_id') or row.error_file_id
                    if state in terminal_states:
                        row.completed_at = row.completed_at or datetime.utcnow()
                        if state == 'JOB_STATE_SUCCEEDED' and row.output_file_id:
                            result_line = _openai_result_line(_openai_file_text(client, row.output_file_id))
                            if result_line and _batch_field(result_line, 'response'):
                                _terminal_openai_message(row, state, response_payload=result_line)
                            else:
                                _terminal_openai_message(row, 'JOB_STATE_FAILED', error_text='OpenAI Batch APIから回答データが返されませんでした')
                                row.state = 'JOB_STATE_FAILED'
                        elif state == 'JOB_STATE_SUCCEEDED':
                            _terminal_openai_message(row, 'JOB_STATE_FAILED', error_text='OpenAI Batch APIの結果ファイルが見つかりませんでした')
                            row.state = 'JOB_STATE_FAILED'
                        elif state != 'JOB_STATE_SUCCEEDED':
                            detail = row.status_text
                            if row.error_file_id:
                                error_line = _openai_result_line(_openai_file_text(client, row.error_file_id))
                                error_obj = _batch_field(error_line, 'error') if error_line else None
                                detail = _batch_field(error_obj, 'message') or detail
                            _terminal_openai_message(row, state, error_text=detail)
                elif provider == 'zai':
                    api_key = resolved.get('api_key')
                    if not api_key:
                        raise RuntimeError('Z.AI APIキーを確認してください')
                    client = _get_openai_client(api_key, base_url='https://api.z.ai/api/paas/v4')
                    provider_payload = client.batches.retrieve(row.provider_job_name)
                    state = _openai_batch_state(_batch_field(provider_payload, 'status'))
                    row.state = state
                    row.status_text = _batch_state_label(state)
                    row.output_file_id = _batch_field(provider_payload, 'output_file_id') or row.output_file_id
                    row.error_file_id = _batch_field(provider_payload, 'error_file_id') or row.error_file_id
                    if state in terminal_states:
                        row.completed_at = row.completed_at or datetime.utcnow()
                        if state == 'JOB_STATE_SUCCEEDED' and row.output_file_id:
                            result_line = _openai_result_line(_openai_file_text(client, row.output_file_id))
                            response = _batch_field(result_line, 'response') if result_line else None
                            request_error = _batch_field(result_line, 'error') if result_line else None
                            response_error = _batch_field(response, 'error') if response else None
                            status_code = _batch_field(response, 'status_code') if response else None
                            status_failed = str(status_code or '').isdigit() and int(status_code) >= 400
                            if request_error or response_error or status_failed:
                                detail = (
                                    _batch_field(request_error, 'message')
                                    or _batch_field(response_error, 'message')
                                    or _batch_field(_batch_field(response, 'body') or {}, 'error')
                                    or f'Z.AI Batchリクエスト HTTP {status_code}'
                                )
                                row.state = 'JOB_STATE_FAILED'
                                row.error = str(detail)
                                row.status_text = _batch_state_label(row.state)
                                _terminal_zai_message(row, row.state, error_text=row.error)
                            elif response:
                                _terminal_zai_message(row, state, response_payload=result_line)
                            else:
                                row.state = 'JOB_STATE_FAILED'
                                row.error = 'Z.AI Batch APIから回答データが返されませんでした'
                                row.status_text = _batch_state_label(row.state)
                                _terminal_zai_message(row, row.state, error_text=row.error)
                        elif state == 'JOB_STATE_SUCCEEDED':
                            row.state = 'JOB_STATE_FAILED'
                            row.error = 'Z.AI Batch APIの結果ファイルが見つかりませんでした'
                            row.status_text = _batch_state_label(row.state)
                            _terminal_zai_message(row, row.state, error_text=row.error)
                        else:
                            detail = row.status_text
                            if row.error_file_id:
                                error_line = _openai_result_line(_openai_file_text(client, row.error_file_id))
                                error_obj = _batch_field(error_line, 'error') if error_line else None
                                detail = _batch_field(error_obj, 'message') or detail
                            if state in {'JOB_STATE_FAILED', 'JOB_STATE_EXPIRED'}:
                                row.error = str(detail)[:4000]
                            _terminal_zai_message(row, state, error_text=detail)
                else:
                    runtime = resolved.get('gemini_runtime') or {}
                    api_key = resolved.get('api_key') or runtime.get('api_key')
                    if not api_key or runtime.get('backend') == 'vertex_ai':
                        raise RuntimeError('Gemini APIキーまたは Gemini API モードを確認してください')
                    provider_url = f"https://generativelanguage.googleapis.com/v1beta/{row.provider_job_name.lstrip('/')}"
                    provider_response = requests.get(
                        provider_url,
                        headers={'x-goog-api-key': str(api_key)},
                        timeout=30,
                    )
                    provider_payload = provider_response.json() if provider_response.content else {}
                    if not provider_response.ok:
                        raise RuntimeError(
                            _batch_field(_batch_field(provider_payload, 'error') or {}, 'message')
                            or f'Gemini Batch API HTTP {provider_response.status_code}'
                        )
                    state = _gemini_operation_state(provider_payload, state)
                    row.state = state
                    row.status_text = _batch_state_label(state)
                    if state in terminal_states:
                        if state == 'JOB_STATE_SUCCEEDED':
                            result_container = (
                                _batch_field(provider_payload, 'response')
                                or _batch_field(provider_payload, 'dest')
                                or {}
                            )
                            result_file = _batch_field(
                                result_container,
                                'responsesFile', 'responses_file', 'fileName', 'file_name'
                            )
                            if result_file:
                                row.output_file_id = str(result_file)
                            result_payload = _gemini_result_payload(provider_payload, api_key)
                            if result_payload and _batch_field(result_payload, 'error'):
                                error_obj = _batch_field(result_payload, 'error') or {}
                                row.state = 'JOB_STATE_FAILED'
                                row.error = _batch_field(error_obj, 'message') or str(error_obj)
                                row.completed_at = row.completed_at or datetime.utcnow()
                                row.status_text = _batch_state_label(row.state)
                                _terminal_message(row, row.state, error_text=row.error)
                            elif result_payload:
                                row.completed_at = row.completed_at or datetime.utcnow()
                                _terminal_message(row, state, response_payload=result_payload)
                            else:
                                row.state = 'JOB_STATE_FAILED'
                                row.error = 'Gemini Batch APIから回答データが返されませんでした'
                                row.completed_at = row.completed_at or datetime.utcnow()
                                row.status_text = _batch_state_label(row.state)
                                _terminal_message(row, row.state, error_text=row.error)
                        else:
                            row.completed_at = row.completed_at or datetime.utcnow()
                            error_obj = _batch_field(provider_payload, 'error') or {}
                            _terminal_message(row, state, error_text=_batch_field(error_obj, 'message') or row.status_text)
            except Exception as poll_error:
                logger.warning('Batch status poll failed for %s: %s', row.job_id, poll_error)
                if state in terminal_states:
                    # Keep a completed provider job pollable until its result
                    # has actually been persisted to the assistant message.
                    row.state = 'JOB_STATE_FINALIZING'
                    row.completed_at = None
                    state = row.state
                row.status_text = 'Batch APIの状態を再確認しています'
        if row.state != old_state or row.status_text != old_status:
            changed_thread_ids.add(str(public_thread_id))
        state = _normalize_batch_state(row.state)
        if state in terminal_states:
            if row.notified_at is None:
                row.notified_at = datetime.utcnow()
                completed.append({
                    'job_id': row.job_id,
                    'thread_id': public_thread_id,
                    'model': row.model,
                    'provider': row.provider or get_model_api_provider(row.model),
                    'state': state,
                    'status_text': row.status_text or _batch_state_label(state),
                })
        else:
            active.append({
                'job_id': row.job_id,
                'thread_id': public_thread_id,
                'model': row.model,
                'provider': row.provider or get_model_api_provider(row.model),
                'state': state,
                'status_text': row.status_text or _batch_state_label(state),
            })
    try:
        safe_db_commit()
    except Exception:
        db.session.rollback()
        logger.exception('Gemini Batch status commit failed')
    return jsonify({
        'active': active,
        'completed': completed,
        'changed_thread_ids': list(changed_thread_ids),
    })


BATCH_TERMINAL_STATES = {
    'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'
}


def _serialize_batch_job(row):
    """Return one Batch history row for the management list.

    History is intentionally never dropped here; completed and failed rows stay
    visible until the user deletes them explicitly.
    """
    state = _normalize_batch_state(row.state or 'JOB_STATE_QUEUED')
    terminal = state in BATCH_TERMINAL_STATES
    thread = getattr(row, 'thread', None)
    public_thread_id = getattr(thread, 'public_id', None) or str(row.thread_id)
    return {
        'job_id': row.job_id,
        'thread_id': public_thread_id,
        'thread_title': (getattr(thread, 'title', None) or '無題のチャット'),
        'thread_exists': thread is not None,
        'assistant_message_id': row.assistant_message_id,
        'model': row.model,
        'provider': row.provider or get_model_api_provider(row.model),
        'state': state,
        'status_text': row.status_text or _batch_state_label(state),
        'error': row.error,
        'created_at': row.created_at.isoformat() if row.created_at else None,
        'completed_at': row.completed_at.isoformat() if row.completed_at else None,
        'is_active': not terminal,
        'can_cancel': bool(row.provider_job_name) and not terminal,
    }


@app.route('/api/batch/jobs', methods=['GET'])
@login_required
def list_batch_jobs_api():
    """List every Batch job the current user submitted, newest first."""
    rows = GeminiBatchJob.query.filter_by(user_id=current_user.id).order_by(
        GeminiBatchJob.created_at.desc()
    ).limit(500).all()
    return jsonify({'jobs': [_serialize_batch_job(row) for row in rows]})


@app.route('/api/batch/jobs/<job_id>/cancel', methods=['POST'])
@login_required
def cancel_batch_job_api(job_id):
    """Ask the provider to cancel an active Batch job and stop tracking it."""
    row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=current_user.id).first()
    if not row:
        return jsonify({'error': 'Batchジョブが見つかりません'}), 404
    state = _normalize_batch_state(row.state or '')
    if state in BATCH_TERMINAL_STATES:
        return jsonify({'error': 'このBatch処理はすでに終了しています'}), 400
    if not row.provider_job_name:
        return jsonify({'error': 'Batch APIへ送信中のため停止できません。少し待ってから再度お試しください。'}), 400

    provider = row.provider or get_model_api_provider(row.model)
    resolved = _resolve_chat_model_auth(current_user, row.model)
    try:
        if provider == 'openai':
            api_key = resolved.get('api_key')
            if not api_key:
                raise RuntimeError('OpenAI APIキーを確認してください')
            client = _get_openai_client(api_key, base_url=None)
            client.batches.cancel(row.provider_job_name)
        elif provider == 'zai':
            api_key = resolved.get('api_key')
            if not api_key:
                raise RuntimeError('Z.AI APIキーを確認してください')
            client = _get_openai_client(api_key, base_url='https://api.z.ai/api/paas/v4')
            client.batches.cancel(row.provider_job_name)
        elif provider == 'xai':
            api_key = resolved.get('api_key')
            if not api_key:
                raise RuntimeError('xAI APIキーを確認してください')
            cancel_response = requests.post(
                f"https://{_XAI_API_HOST}/v1/batches/{quote(str(row.provider_job_name), safe='')}:cancel",
                headers={'Authorization': f'Bearer {str(api_key)}'},
                timeout=30,
            )
            if not cancel_response.ok:
                payload = cancel_response.json() if cancel_response.content else {}
                detail = _batch_field(payload, 'error') or payload
                if isinstance(detail, dict):
                    detail = _batch_field(detail, 'message') or detail
                raise RuntimeError(detail or f'xAI Batchキャンセル HTTP {cancel_response.status_code}')
        else:
            runtime = resolved.get('gemini_runtime') or {}
            api_key = resolved.get('api_key') or runtime.get('api_key')
            if not api_key or runtime.get('backend') == 'vertex_ai':
                raise RuntimeError('Gemini APIキーまたは Gemini API モードを確認してください')
            cancel_response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/{str(row.provider_job_name).lstrip('/')}:cancel",
                headers={'x-goog-api-key': str(api_key)},
                timeout=30,
            )
            if not cancel_response.ok:
                payload = cancel_response.json() if cancel_response.content else {}
                raise RuntimeError(
                    _batch_field(_batch_field(payload, 'error') or {}, 'message')
                    or f'Gemini Batchキャンセル HTTP {cancel_response.status_code}'
                )
    except Exception as cancel_error:
        logger.warning('Batch cancel failed for %s: %s', job_id, cancel_error)
        return jsonify({'error': str(cancel_error)}), 502

    row.state = 'JOB_STATE_CANCELLED'
    row.status_text = _batch_state_label(row.state)
    row.completed_at = row.completed_at or datetime.utcnow()
    row.notified_at = datetime.utcnow()
    message = Message.query.filter_by(
        id=row.assistant_message_id, thread_id=row.thread_id
    ).first()
    if message:
        content = f'**{row.status_text}**'
        message.content = encrypt_val(content) if message.is_encrypted else content
    safe_db_commit()
    return jsonify({'ok': True, 'job': _serialize_batch_job(row)})


@app.route('/api/batch/jobs/<job_id>', methods=['DELETE'])
@login_required
def delete_batch_job_api(job_id):
    """Remove one finished Batch history row for the current user."""
    row = GeminiBatchJob.query.filter_by(job_id=job_id, user_id=current_user.id).first()
    if not row:
        return jsonify({'error': 'Batchジョブが見つかりません'}), 404
    state = _normalize_batch_state(row.state or '')
    if state not in BATCH_TERMINAL_STATES:
        return jsonify({'error': '実行中のBatch処理は削除できません。先に停止してください。'}), 400
    try:
        db.session.delete(row)
        safe_db_commit()
    except Exception:
        db.session.rollback()
        logger.exception('Batch history delete failed')
        return jsonify({'error': 'Batch履歴の削除に失敗しました'}), 500
    return jsonify({'ok': True})
