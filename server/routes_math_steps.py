# -----------------------------------------------------------
# Math steps API
# 数学ステップ解説の進捗保存と「もっとやさしく」の再生成。
# 本文・プロンプト・回答はログに出さない。
# -----------------------------------------------------------

def _math_steps_int(value, low, high, default=0):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _load_math_steps_state(raw):
    """Parse a stored state JSON into the clamped canonical shape."""
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw) if raw.strip() else None
        except (TypeError, ValueError):
            data = None
    if not isinstance(data, dict):
        data = {}
    current = _math_steps_int(data.get("current"), 0, MATH_STEPS_MAX_CURRENT, 0)
    max_reached = _math_steps_int(data.get("max_reached"), 0, MATH_STEPS_MAX_CURRENT, current)
    max_reached = max(max_reached, current)
    simplified = {}
    total_chars = 0
    raw_simplified = data.get("simplified")
    if isinstance(raw_simplified, dict):
        for key, value in raw_simplified.items():
            key_text = str(key)
            if not key_text.isdigit() or len(key_text) > 4:
                continue
            value_text = str(value or "")
            if not value_text:
                continue
            remaining = MATH_STEPS_MAX_SIMPLIFIED_TOTAL - total_chars
            if remaining <= 0:
                break
            limit = min(MATH_STEPS_MAX_SIMPLIFIED_PER_STEP, remaining)
            simplified[key_text] = value_text[:limit]
            total_chars += len(simplified[key_text])
    return {"current": current, "max_reached": max_reached, "simplified": simplified}


def _dump_math_steps_state(state):
    return json.dumps(_load_math_steps_state(state), ensure_ascii=False, separators=(",", ":"))


def _parse_math_steps_state_json(raw):
    """Payload helper: the stored state as an object (or None when absent)."""
    if not raw:
        return None
    return _load_math_steps_state(raw)


def _read_math_steps_state(message):
    """Read the state, decrypting it when the message is end-to-end encrypted."""
    raw = getattr(message, 'math_steps_state', None)
    if raw and getattr(message, 'is_encrypted', False):
        try:
            raw = decrypt_val(raw)
        except Exception:
            raw = None
    return _load_math_steps_state(raw)


def _write_math_steps_state(message, state):
    """Store the state, encrypting it when the message is end-to-end encrypted."""
    text = _dump_math_steps_state(state)
    if getattr(message, 'is_encrypted', False):
        text = encrypt_val(text)
    message.math_steps_state = text


def _resolve_math_steps_message(data):
    """Return (message, error) for a message_id or job_id owned by the current user."""
    message_id = data.get("message_id")
    job_id = str(data.get("job_id") or "").strip()
    query = Message.query.join(Thread, Thread.id == Message.thread_id).filter(
        Thread.user_id == current_user.id
    )
    if message_id is not None and str(message_id).strip():
        try:
            message_pk = int(message_id)
        except (TypeError, ValueError):
            return None, "invalid_message"
        return query.filter(Message.id == message_pk).first(), None
    if job_id:
        if len(job_id) > 128:
            return None, "invalid_job"
        return query.filter(Message.job_id == job_id, Message.role == "assistant").first(), None
    return None, "invalid_message"


def _math_steps_message_content(message):
    content = message.content or ""
    if message.is_encrypted and content:
        content = decrypt_val(content) or ""
    return str(content)


def _math_steps_prompt_options(message):
    try:
        value = json.loads(message.prompt_options) if message.prompt_options else None
    except (TypeError, ValueError):
        value = None
    return value if isinstance(value, dict) else {}


def _math_steps_usage_observe(call, usage):
    if call is None or usage is None:
        return
    try:
        call.update_usage(usage)
    except Exception:
        pass


def _append_math_steps_usage(message, call):
    """Best-effort content-free usage record on the assistant message."""
    if call is None:
        return
    try:
        call.finished = True
        snapshot = call.snapshot()
        raw = message.thought_data
        payload = decrypt_val(raw) if (message.is_encrypted and raw) else raw
        metadata = json.loads(payload) if payload else {}
        if not isinstance(metadata, dict):
            metadata = {}
        calls = metadata.get("pricing_calls")
        if not isinstance(calls, list):
            calls = []
        metadata["pricing_calls"] = calls + [snapshot]
        text = json.dumps(metadata, ensure_ascii=False)
        message.thought_data = encrypt_val(text) if message.is_encrypted else text
    except Exception as exc:
        logger.warning("math_steps usage record skipped: %s", exc)


def _call_math_steps_simplify_model(user, model_key, prompt, gemini_free_tier=False):
    """Call the message's model once and return (text, usage_call)."""
    resolved = _resolve_chat_model_auth(user, model_key, gemini_free_tier=gemini_free_tier)
    if resolved.get("error_code"):
        raise RuntimeError(resolved.get("error") or "モデルの認証情報がありません")
    provider = resolved.get("provider")
    api_key = resolved.get("api_key")
    call = ModelUsageCall(model_key, provider)
    try:
        if provider == "gemini":
            runtime = resolved.get("gemini_runtime") or _resolve_gemini_runtime(
                user, use_free_tier=gemini_free_tier
            )
            client = _get_gemini_client(
                api_key=api_key,
                backend=runtime.get("backend"),
                vertex_project=runtime.get("vertex_project"),
                vertex_location=runtime.get("vertex_location"),
                vertex_credentials_json=runtime.get("vertex_credentials_json"),
            )
            response = client.models.generate_content(model=model_key, contents=prompt)
            _math_steps_usage_observe(call, getattr(response, "usage_metadata", None))
            return str(getattr(response, "text", None) or ""), call
        if provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError("Anthropic SDK is not installed")
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_key,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            _math_steps_usage_observe(call, getattr(response, "usage", None))
            return "".join(
                str(getattr(block, "text", "") or "")
                for block in (getattr(response, "content", None) or [])
                if getattr(block, "type", None) == "text"
            ), call
        base_url = None
        if provider == "deepseek":
            base_url = "https://api.deepseek.com"
        elif provider == "zai":
            base_url = "https://api.z.ai/api/paas/v4"
        elif provider == "kimi":
            base_url = "https://api.moonshot.ai/v1"
        elif provider == "xai":
            base_url = f"https://{_XAI_API_HOST}/v1"
        client = _get_openai_client(api_key, base_url=base_url)
        if provider in {"deepseek", "kimi", "xai", "zai"}:
            response = client.chat.completions.create(
                model=_deepseek_api_model_id(model_key) if provider == "deepseek" else model_key,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            _math_steps_usage_observe(call, getattr(response, "usage", None))
            return str(getattr(response.choices[0].message, "content", None) or ""), call
        response = client.responses.create(model=model_key, input=prompt, max_output_tokens=4096, store=False)
        _math_steps_usage_observe(call, getattr(response, "usage", None))
        return _extract_openai_response_text(response), call
    finally:
        call.finished = True


def _is_math_steps_text_model(model_key):
    lowered = str(model_key or "").lower()
    if not lowered:
        return False
    return not any(
        marker in lowered
        for marker in (
            "image", "video", "tts", "audio", "native-audio", "embedding",
            "transcribe", "realtime", "ocr", "music", "voice",
        )
    )


@app.route('/api/math_steps/progress', methods=['POST'])
@login_required
def math_steps_progress():
    if not rate_limit(f"rl:math_steps_progress:user:{current_user.id}", 120, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'invalid_request'}), 400
    message, error = _resolve_math_steps_message(data)
    if error:
        return jsonify({'error': error}), 400
    if not message:
        return jsonify({'error': 'message_not_found'}), 404
    state = _read_math_steps_state(message)
    if 'current' in data:
        state['current'] = _math_steps_int(data.get('current'), 0, MATH_STEPS_MAX_CURRENT, state['current'])
    if 'max_reached' in data:
        given_max = _math_steps_int(
            data.get('max_reached'), 0, MATH_STEPS_MAX_CURRENT, state['max_reached']
        )
        state['max_reached'] = max(state['max_reached'], given_max, state['current'])
    else:
        state['max_reached'] = max(state['max_reached'], state['current'])
    # 「元の説明に戻す」: remove the saved simplified variants for the given steps.
    removals = data.get('remove_simplified')
    if isinstance(removals, list):
        for item in removals[:MATH_STEPS_MAX_CURRENT + 1]:
            try:
                state['simplified'].pop(str(int(item)), None)
            except (TypeError, ValueError):
                continue
    _write_math_steps_state(message, state)
    safe_db_commit()
    logger.info(
        "math_steps progress user=%s message=%s current=%s max=%s",
        current_user.id, message.id, state['current'], state['max_reached'],
    )
    return jsonify({'status': 'ok', 'math_steps_state': _read_math_steps_state(message)})


@app.route('/api/math_steps/simplify', methods=['POST'])
@login_required
def math_steps_simplify():
    if not rate_limit(f"rl:math_steps_simplify:user:{current_user.id}", 20, 60):
        return jsonify({'error': 'rate_limit'}), 429
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'invalid_request'}), 400
    message, error = _resolve_math_steps_message(data)
    if error:
        return jsonify({'error': error}), 400
    if not message or message.role != 'assistant':
        return jsonify({'error': 'message_not_found'}), 404
    try:
        step_index = int(data.get('step_index'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid_step_index'}), 400
    content = _math_steps_message_content(message)
    if not content.strip():
        return jsonify({'error': 'message_has_no_content'}), 409
    steps = extract_math_steps(content)
    if not steps or step_index < 0 or step_index >= len(steps):
        return jsonify({'error': 'invalid_step_index'}), 400
    step = steps[step_index]
    step_text = (step.get('title') + "\n" + step.get('body', '')).strip()
    if not step_text:
        return jsonify({'error': 'empty_step'}), 400
    if len(step_text) > 100_000:
        return jsonify({'error': 'step_too_large'}), 413
    model_key = str(message.model or '').strip()
    if not _is_math_steps_text_model(model_key):
        return jsonify({'error': 'unsupported_model'}), 400
    options = _math_steps_prompt_options(message)
    config = normalize_math_steps_config(getattr(current_user, 'math_steps_config', None))
    prompt = build_math_steps_simplify_prompt(config) + step_text
    logger.info(
        "math_steps simplify start user=%s message=%s step=%s model=%s",
        current_user.id, message.id, step_index, model_key,
    )
    try:
        simplified, usage_call = _call_math_steps_simplify_model(
            current_user,
            model_key,
            prompt,
            gemini_free_tier=options.get('gemini_free_tier') is True,
        )
    except Exception as exc:
        logger.warning(
            "math_steps simplify failed user=%s message=%s step=%s error=%s",
            current_user.id, message.id, step_index, type(exc).__name__,
        )
        return jsonify({'error': 'simplify_failed'}), 502
    simplified = str(simplified or "").strip()
    # The simplify prompt asks for the body only. Strip a stray heading so the
    # client cannot mistake it for an extra step.
    simplified = re.sub(
        r"^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*\d+[^\n]*\n?",
        "",
        simplified,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if not simplified:
        return jsonify({'error': 'empty_result'}), 502
    state = _read_math_steps_state(message)
    key = str(step_index)
    others_total = sum(
        len(value) for other_key, value in state['simplified'].items() if other_key != key
    )
    remaining = MATH_STEPS_MAX_SIMPLIFIED_TOTAL - others_total
    if remaining <= 0:
        return jsonify({'error': 'simplified_limit_reached'}), 409
    state['simplified'][key] = simplified[:min(MATH_STEPS_MAX_SIMPLIFIED_PER_STEP, remaining)]
    _write_math_steps_state(message, state)
    _append_math_steps_usage(message, usage_call)
    safe_db_commit()
    logger.info(
        "math_steps simplify done user=%s message=%s step=%s chars=%s",
        current_user.id, message.id, step_index, len(state['simplified'][key]),
    )
    return jsonify({
        'status': 'ok',
        'text': state['simplified'][key],
        'math_steps_state': _read_math_steps_state(message),
    })
