# -----------------------------------------------------------
# Math steps API
# 数学ステップ解説の進捗保存と「もっとやさしく」の再生成。
# 本文・プロンプト・回答はログに出さない。
# -----------------------------------------------------------

# Context handed to the 「もっとやさしく」 rewrite. The problem and the tail of the
# previous step are all the model needs to explain why an operation is done; the
# full answer is not sent.
MATH_STEPS_SIMPLIFY_PROBLEM_CHARS = 1200
MATH_STEPS_SIMPLIFY_PREVIOUS_CHARS = 600
MATH_STEPS_SIMPLIFY_ANCHOR_CHARS = 80

def _math_steps_int(value, low, high, default=0):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _clean_math_steps_parts(value, fallback_title=""):
    """Normalize one stored/returned replacement list (title + Markdown body)."""
    if not isinstance(value, list):
        return []
    fallback = str(fallback_title or "").strip()[:MATH_STEPS_MAX_EXPANDED_TITLE]
    parts = []
    for item in value[:MATH_STEPS_MAX_EXPANDED_PER_STEP]:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or "").strip()[:MATH_STEPS_MAX_SIMPLIFIED_PER_STEP]
        if not body:
            continue
        title = str(item.get("title") or "").strip()[:MATH_STEPS_MAX_EXPANDED_TITLE]
        parts.append({"title": title or fallback, "body": body})
    return parts


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
    # 「もっとやさしく」 replaces a step with smaller easier steps. The list is
    # keyed by the original step index, exactly like a saved simplified variant.
    expanded = {}
    raw_expanded = data.get("expanded")
    if isinstance(raw_expanded, dict):
        for key, value in raw_expanded.items():
            key_text = str(key)
            if not key_text.isdigit() or len(key_text) > 4:
                continue
            parts = []
            for part in _clean_math_steps_parts(value):
                remaining = MATH_STEPS_MAX_SIMPLIFIED_TOTAL - total_chars
                if remaining <= 0:
                    break
                body = part["body"][:min(MATH_STEPS_MAX_SIMPLIFIED_PER_STEP, remaining)]
                parts.append({"title": part["title"], "body": body})
                total_chars += len(body)
            if parts:
                expanded[key_text] = parts
    return {
        "current": current,
        "max_reached": max_reached,
        "simplified": simplified,
        "expanded": expanded,
    }


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


def _math_steps_head(value, limit):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _math_steps_tail(value, limit):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return "…" + text[-limit:].lstrip()


def _math_steps_first_line(value):
    """First readable line of a step body (anchors, mathsvg fences and tags are skipped)."""
    for line in str(value or "").splitlines():
        text = line.strip()
        if not text or text.startswith("```") or text.startswith("<"):
            continue
        if re.match(r"^元のステップ\d+.*から説明します。$", text):
            continue
        return text
    return ""


def _math_steps_anchor(step_index, step_title, step_body, part_index=None):
    """Sentence that tells the reader which step (or sub-step) this came from."""
    number = step_index + 1
    if part_index is None:
        label = f"元のステップ{number}"
    else:
        label = f"元のステップ{number}の{part_index + 1}つ目の小ステップ"
    title = str(step_title or "").strip()
    if title:
        label += f"「{title}」"
    first = _math_steps_head(
        _math_steps_first_line(step_body), MATH_STEPS_SIMPLIFY_ANCHOR_CHARS
    )
    if first:
        return f"{label}の『{first}』から説明します。"
    return f"{label}から説明します。"


def _math_steps_problem_text(message, steps):
    """Latest question in the thread (or the first step when the question is an image)."""
    question = (
        Message.query
        .filter(
            Message.thread_id == message.thread_id,
            Message.role == "user",
            Message.id < message.id,
        )
        .order_by(Message.id.desc())
        .first()
    )
    text = _math_steps_message_content(question) if question else ""
    text = str(text or "").strip()
    if not text and steps:
        text = str(steps[0].get("body") or "").strip()
    return _math_steps_head(text, MATH_STEPS_SIMPLIFY_PROBLEM_CHARS)


def _math_steps_previous_text(steps, state, step_index, part_index):
    expanded = state.get("expanded") or {}
    if part_index is not None and part_index > 0:
        entry = expanded.get(str(step_index)) or []
        if part_index <= len(entry):
            return _math_steps_tail(
                entry[part_index - 1].get("body"), MATH_STEPS_SIMPLIFY_PREVIOUS_CHARS
            )
    for index in range(step_index - 1, -1, -1):
        parts = expanded.get(str(index)) or []
        body = parts[-1].get("body") if parts else steps[index].get("body")
        text = _math_steps_tail(body, MATH_STEPS_SIMPLIFY_PREVIOUS_CHARS)
        if text:
            return text
    return ""


def _math_steps_next_title(steps, state, step_index, part_index):
    expanded = state.get("expanded") or {}
    if part_index is not None:
        entry = expanded.get(str(step_index)) or []
        if part_index + 1 < len(entry):
            title = str(entry[part_index + 1].get("title") or "").strip()
            if title:
                return title
    if step_index + 1 < len(steps):
        next_index = step_index + 1
        parts = expanded.get(str(next_index)) or []
        if parts:
            title = str(parts[0].get("title") or "").strip()
        else:
            title = str(steps[next_index].get("title") or "").strip()
        return title or f"ステップ{next_index + 1}"
    return "最終ステップ（答え）"


def _math_steps_simplify_context(message, steps, state, step_index, part_index,
                                 step_title, step_body):
    return {
        "problem": _math_steps_problem_text(message, steps),
        "previous": _math_steps_previous_text(steps, state, step_index, part_index),
        "next": _math_steps_next_title(steps, state, step_index, part_index),
        "step_number": step_index + 1,
        "step_title": step_title,
        "step_body": step_body,
        "anchor": _math_steps_anchor(step_index, step_title, step_body, part_index),
    }


def _math_steps_with_anchor(parts, anchor):
    """Guarantee the first body starts with the anchor the model was given."""
    anchor = str(anchor or "").strip()
    if not anchor or not parts:
        return parts
    body = str(parts[0].get("body") or "").strip()
    if body.startswith(anchor):
        return parts
    parts[0] = dict(parts[0])
    parts[0]["body"] = (anchor + "\n\n" + body).strip() if body else anchor
    return parts


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
    # The same button collapses an expanded step back to its original text.
    expanded_removals = data.get('remove_expanded')
    if isinstance(expanded_removals, list):
        for item in expanded_removals[:MATH_STEPS_MAX_CURRENT + 1]:
            try:
                state['expanded'].pop(str(int(item)), None)
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
    part_index = None
    if data.get('part_index') is not None:
        try:
            part_index = int(data.get('part_index'))
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid_part_index'}), 400
        if part_index < 0:
            return jsonify({'error': 'invalid_part_index'}), 400
    content = _math_steps_message_content(message)
    if not content.strip():
        return jsonify({'error': 'message_has_no_content'}), 409
    steps = extract_math_steps(content)
    if not steps or step_index < 0 or step_index >= len(steps):
        return jsonify({'error': 'invalid_step_index'}), 400
    step = steps[step_index]
    state = _read_math_steps_state(message)
    key = str(step_index)
    entry = state['expanded'].get(key)
    if part_index is not None:
        # A sub-step created by an earlier split: make that part even easier too.
        if not entry or part_index >= len(entry):
            return jsonify({'error': 'invalid_part_index'}), 400
        part = entry[part_index]
        step_title = part.get('title') or step.get('title', '')
        step_body = part.get('body', '')
    else:
        step_title = step.get('title', '')
        step_body = step.get('body', '')
    step_text = (str(step_title) + "\n" + str(step_body)).strip()
    if not step_text:
        return jsonify({'error': 'empty_step'}), 400
    if len(step_text) > 100_000:
        return jsonify({'error': 'step_too_large'}), 413
    model_key = str(message.model or '').strip()
    if not _is_math_steps_text_model(model_key):
        return jsonify({'error': 'unsupported_model'}), 400
    options = _math_steps_prompt_options(message)
    config = normalize_math_steps_config(getattr(current_user, 'math_steps_config', None))
    context = _math_steps_simplify_context(
        message, steps, state, step_index, part_index, step_title, step_body
    )
    prompt = build_math_steps_simplify_prompt(config, context)
    logger.info(
        "math_steps simplify start user=%s message=%s step=%s part=%s model=%s "
        "problem_chars=%s previous_chars=%s anchor_chars=%s",
        current_user.id, message.id, step_index, part_index, model_key,
        len(context['problem']), len(context['previous']), len(context['anchor']),
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
    parts = _clean_math_steps_parts(extract_math_steps(simplified), step_title)
    if not parts:
        # A model may ignore the heading rule. Keep the reader moving by treating
        # the whole answer as a single easier replacement for this step.
        fallback = re.sub(
            r"^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*\d+[^\n]*\n?",
            "",
            simplified,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if fallback:
            parts = _clean_math_steps_parts(
                [{"title": step_title, "body": fallback}], step_title
            )
    if not parts:
        return jsonify({'error': 'empty_result'}), 502
    parts = _math_steps_with_anchor(parts, context['anchor'])
    existing = dict(state['expanded'])
    if part_index is None:
        existing[key] = parts
    else:
        merged = list(entry[:part_index]) + parts + list(entry[part_index + 1:])
        if len(merged) > MATH_STEPS_MAX_EXPANDED_PER_STEP:
            return jsonify({'error': 'too_many_steps'}), 409
        existing[key] = merged
    display_total = len(steps)
    for other_key, items in existing.items():
        if not other_key.isdigit() or int(other_key) >= len(steps):
            continue
        display_total += max(0, len(items) - 1)
    if display_total > MATH_STEPS_MAX_STEPS:
        return jsonify({'error': 'too_many_steps'}), 409
    others_total = sum(
        len(value) for other_key, value in state['simplified'].items() if other_key != key
    )
    others_total += sum(
        len(part['body'])
        for other_key, items in existing.items() if other_key != key
        for part in items
    )
    new_total = sum(len(part['body']) for part in existing[key])
    if others_total + new_total > MATH_STEPS_MAX_SIMPLIFIED_TOTAL:
        return jsonify({'error': 'simplified_limit_reached'}), 409
    state['expanded'] = existing
    # The expanded list replaces any older single-body simplification.
    state['simplified'].pop(key, None)
    _write_math_steps_state(message, state)
    _append_math_steps_usage(message, usage_call)
    safe_db_commit()
    logger.info(
        "math_steps simplify done user=%s message=%s step=%s part=%s parts=%s chars=%s",
        current_user.id, message.id, step_index, part_index, len(existing[key]),
        new_total,
    )
    return jsonify({
        'status': 'ok',
        'text': "\n\n".join(part['body'] for part in existing[key]),
        'steps': existing[key],
        'step_index': step_index,
        'math_steps_state': _read_math_steps_state(message),
    })
