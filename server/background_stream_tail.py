"""Shared stream tail of `background_chat_task`.

Runs after the provider branch, composes the final answer metadata, persists
the assistant message, and publishes ``done``.  The tail used to run inline in
`background.py`; it now receives that function's `locals()` snapshot as
``local``.  It returns ``BACKGROUND_STREAM_EARLY_RETURN`` when an already
persisted stream error must end the task without writing a second message.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to background_chat_task when the task must end without a new message.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_tail(local):
    """Persist the assistant message and publish completion for the job."""
    _compact_thought_signature = local.get("_compact_thought_signature")
    _consume_coding_stream_chunk = local.get("_consume_coding_stream_chunk")
    _emit_coding_diff = local.get("_emit_coding_diff")
    _used_skills = local.get("_used_skills")
    agentic_consumed_urls = local.get("agentic_consumed_urls")
    agentic_filename_url_map = local.get("agentic_filename_url_map")
    agentic_saved_urls = local.get("agentic_saved_urls")
    assistant_tokens_out = local.get("assistant_tokens_out")
    c_count = local.get("c_count")
    check_stop = local.get("check_stop")
    coding_repair_outputs = local.get("coding_repair_outputs")
    coding_stream_edit_index = local.get("coding_stream_edit_index")
    coding_target = local.get("coding_target")
    completion_tokens_val = local.get("completion_tokens_val")
    current_working_code = local.get("current_working_code")
    deepseek_tool_context = local.get("deepseek_tool_context")
    details = local.get("details")
    edit_payload = local.get("edit_payload")
    exc = local.get("exc")
    explicitly_selected = local.get("explicitly_selected")
    failure = local.get("failure")
    final_content = local.get("final_content")
    final_openai_usage = local.get("final_openai_usage")
    final_signature = local.get("final_signature")
    final_thought = local.get("final_thought")
    final_usage_metadata = local.get("final_usage_metadata")
    full_res = local.get("full_res")
    generated_images = local.get("generated_images")
    grok_reasoning_supported = local.get("grok_reasoning_supported")
    history = local.get("history")
    image_url = local.get("image_url")
    initial_failure = local.get("initial_failure")
    is_deepseek = local.get("is_deepseek")
    is_enc = local.get("is_enc")
    is_gem = local.get("is_gem")
    is_grok = local.get("is_grok")
    job_id = local.get("job_id")
    last_repair_error = local.get("last_repair_error")
    locked_target_id = local.get("locked_target_id")
    m = local.get("m")
    message_id = local.get("message_id")
    message_text = local.get("message_text")
    meta = local.get("meta")
    model_key = local.get("model_key")
    msg_entry = local.get("msg_entry")
    options = local.get("options")
    original_target_code = local.get("original_target_code")
    output_tokens_val = local.get("output_tokens_val")
    parts = local.get("parts")
    pricing_input = local.get("pricing_input")
    pub = local.get("pub")
    reasoning_tokens_val = local.get("reasoning_tokens_val")
    remaining_edits = local.get("remaining_edits")
    repair_attempt = local.get("repair_attempt")
    repair_exc = local.get("repair_exc")
    repair_failure = local.get("repair_failure")
    repair_output = local.get("repair_output")
    repair_payload = local.get("repair_payload")
    repair_prompt = local.get("repair_prompt")
    repair_steps = local.get("repair_steps")
    repair_target_id = local.get("repair_target_id")
    repaired = local.get("repaired")
    search_used = local.get("search_used")
    selected_candidate = local.get("selected_candidate")
    sig_kept_count = local.get("sig_kept_count")
    sig_original_count = local.get("sig_original_count")
    signature_parts = local.get("signature_parts")
    skill = local.get("skill")
    skill_name_val = local.get("skill_name_val")
    skill_uuid_val = local.get("skill_uuid_val")
    step = local.get("step")
    stream_error_persisted = local.get("stream_error_persisted")
    summary = local.get("summary")
    t_count = local.get("t_count")
    target_language = local.get("target_language")
    th = local.get("th")
    thought_accumulated = local.get("thought_accumulated")
    thought_payload = local.get("thought_payload")
    thread_id = local.get("thread_id")
    tokens_thought_val = local.get("tokens_thought_val")
    usage = local.get("usage")
    usage_ledger = local.get("usage_ledger")
    usage_totals = local.get("usage_totals")
    user = local.get("user")
    user_config = local.get("user_config")


    if (is_gem or is_deepseek) and not full_res.strip() and not generated_images and not stream_error_persisted:
        pub("error", "モデルから回答本文が返されませんでした。時間をおいて再試行してください。")
        # `pub` sets the flag on `background_chat_task`, not on this function's
        # `locals()` copy. Mirror it here so the check below still stops the
        # normal message from being saved behind the persisted error.
        stream_error_persisted = True
    # pub(error) has already saved the partial answer and error as the active leaf.
    # A second normal row would hide that error behind an empty sibling branch.
    if stream_error_persisted:
        log_force("Stream error already persisted; skipping the normal assistant message")
        return BACKGROUND_STREAM_EARLY_RETURN
    if is_grok and grok_reasoning_supported and not thought_accumulated:
        thought_accumulated = " "
    # Container citations can arrive after the text referring to their files.
    # Resolve the completed answer now that every returned image is saved.
    if agentic_filename_url_map or agentic_saved_urls:
        full_res = _rewrite_sandbox_image_refs(
            full_res, agentic_saved_urls, agentic_consumed_urls, agentic_filename_url_map,
        )
        for image_url in agentic_consumed_urls:
            full_res = full_res.replace(f"![Agentic View]({image_url})", "")
    final_content = _strip_invented_agentic_images(
        full_res,
        set(agentic_saved_urls) | set(agentic_consumed_urls),
        [str(m.get('content') or '') for m in history],
    )
    coding_repair_outputs = ""
    if options.get("coding_mode"):
        _consume_coding_stream_chunk("", flush=True)
        coding_target = options.get("coding_target") or {}
        try:
            selected_candidate, edit_payload = _resolve_coding_mode_candidate(
                full_res,
                options.get("coding_candidates") or [],
                coding_target.get("default_target_id") or coding_target.get("id"),
            )
            locked_target_id = str(selected_candidate.get("id") or "")
            target_language = str(selected_candidate.get("language") or "text")
            original_target_code = str(selected_candidate.get("code") or "")
            current_working_code = original_target_code
            summary = edit_payload.get("summary") or "変更を適用しました"
            explicitly_selected = (
                coding_target.get("explicit") is True
                and locked_target_id == str(coding_target.get("id") or "")
            )
            repaired = False
            try:
                current_working_code, _ = _apply_coding_mode_payload(
                    edit_payload,
                    current_working_code,
                )
            except CodingModeEditApplicationError as initial_failure:
                failure = initial_failure
                current_working_code = failure.current_code
                last_repair_error = failure
                for repair_attempt in range(1, 3):
                    if check_stop():
                        raise ValueError("自動修復はユーザーにより停止されました")
                    pub(
                        "status",
                        f"Coding Modeの差分を自動修復中です（{repair_attempt}/2）...",
                    )
                    remaining_edits = edit_payload["edits"][failure.edit_index - 1:]
                    repair_prompt = build_coding_mode_repair_prompt(
                        message_text,
                        locked_target_id,
                        target_language,
                        current_working_code,
                        failure,
                        remaining_edits,
                        explicitly_selected=explicitly_selected,
                        attempt=repair_attempt,
                    )
                    try:
                        repair_output = _call_coding_mode_repair_model(
                            user,
                            model_key,
                            repair_prompt,
                        )
                        coding_repair_outputs += f"\n{repair_output}"
                        repair_payload = _parse_coding_mode_edit_payload(repair_output)
                        repair_target_id = repair_payload.get("target_id")
                        if repair_target_id and repair_target_id != locked_target_id:
                            raise ValueError(
                                "自動修復モデルが固定済みの編集対象を変更しようとしました"
                            )
                        if not repair_payload["edits"]:
                            raise ValueError("自動修復モデルから修正差分が返されませんでした")
                        current_working_code, repair_steps = _apply_coding_mode_payload(
                            repair_payload,
                            current_working_code,
                        )
                        for step in repair_steps:
                            coding_stream_edit_index += 1
                            _emit_coding_diff({
                                "target_id": locked_target_id,
                                "language": target_language,
                                "edit_index": coding_stream_edit_index,
                                "repair_attempt": repair_attempt,
                                "diff": build_coding_mode_unified_diff(
                                    step["before_code"],
                                    step["after_code"],
                                    target_language,
                                ),
                            })
                        if repair_payload.get("summary"):
                            summary = repair_payload["summary"]
                        repaired = True
                        break
                    except CodingModeEditApplicationError as repair_failure:
                        current_working_code = repair_failure.current_code
                        for step in repair_failure.applied_steps:
                            coding_stream_edit_index += 1
                            _emit_coding_diff({
                                "target_id": locked_target_id,
                                "language": target_language,
                                "edit_index": coding_stream_edit_index,
                                "repair_attempt": repair_attempt,
                                "diff": build_coding_mode_unified_diff(
                                    step["before_code"],
                                    step["after_code"],
                                    target_language,
                                ),
                            })
                        failure = repair_failure
                        edit_payload = repair_payload
                        last_repair_error = repair_failure
                        logger.warning(
                            "Coding Mode automatic repair attempt %s failed for job %s: %s",
                            repair_attempt,
                            job_id,
                            repair_failure,
                        )
                    except Exception as repair_exc:
                        last_repair_error = repair_exc
                        logger.warning(
                            "Coding Mode automatic repair attempt %s failed for job %s: %s",
                            repair_attempt,
                            job_id,
                            repair_exc,
                        )
                if not repaired:
                    raise ValueError(
                        f"{initial_failure}。自動修復も完了できませんでした: {last_repair_error}"
                    )
            if repaired:
                summary = f"{summary}（差分を自動修復しました）"
            final_content = build_coding_mode_final_markdown(
                summary,
                original_target_code,
                current_working_code,
                target_language,
            )
        except ValueError as exc:
            logger.warning("Coding Mode edit application failed for job %s: %s", job_id, exc)
            final_content = (
                f"**Coding Modeの差分適用に失敗しました:** {exc}\n\n"
                "安全のため元コードは変更していません。指示を具体化して再送してください。"
            )
        pub("content", final_content, coding_final=True)

    def _compact_thought_signature(parts):
        if not parts:
            return None
        max_json_bytes = 60000
        max_items = 32
        max_item_chars = 4096
        compact = []
        for raw in parts:
            if not isinstance(raw, str) or not raw:
                continue
            # Skip abnormal signatures to protect DB TEXT column and history payload.
            if len(raw) > max_item_chars:
                continue
            compact.append(raw)
            if len(compact) >= max_items:
                break
        if not compact:
            return None
        enc = json.dumps(compact, separators=(",", ":"))
        if len(enc.encode('utf-8')) <= max_json_bytes:
            return enc
        while compact:
            compact.pop()
            if not compact:
                return None
            enc = json.dumps(compact, separators=(",", ":"))
            if len(enc.encode('utf-8')) <= max_json_bytes:
                return enc
        return None

    sig_original_count = len(signature_parts)
    final_signature = _compact_thought_signature(signature_parts)
    if sig_original_count:
        try:
            sig_kept_count = len(json.loads(final_signature)) if final_signature else 0
        except Exception:
            sig_kept_count = 0
        if sig_kept_count < sig_original_count:
            log_force(f"Trimmed thought_signature for DB storage: kept {sig_kept_count}/{sig_original_count}")

    thought_payload = {"search_used": True} if search_used else {}
    if _used_skills:
        thought_payload['used_skills'] = list(_used_skills)
    if usage_ledger.calls:
        thought_payload['pricing_calls'] = usage_ledger.snapshot()
    pricing_input = _cost_input_tokens(locals().get('final_usage_metadata') if is_gem else locals().get('final_openai_usage'))
    if pricing_input is not None:
        thought_payload['pricing_input_tokens'] = pricing_input
    if thought_accumulated:
        thought_payload["text"] = thought_accumulated
    if deepseek_tool_context:
        thought_payload["deepseek_tool_context"] = deepseek_tool_context
    final_thought = json.dumps(thought_payload, ensure_ascii=False) if thought_payload else None
    is_enc = user_config.get('enable_e2ee', False)
    if is_enc:
        final_content = encrypt_val(final_content)
        if final_thought: final_thought = encrypt_val(final_thought)

    assistant_tokens_out = count_tokens_for_display(
        full_res + coding_repair_outputs,
        model_key,
        thought_accumulated,
    )
    tokens_thought_val = count_tokens(thought_accumulated, model_key) if thought_accumulated else 0

    # Gemini Thinking: Use official usage metadata if available
    if is_gem and locals().get('final_usage_metadata'):
        meta = locals().get('final_usage_metadata')
        t_count = getattr(meta, 'thoughts_token_count', 0) or 0
        c_count = getattr(meta, 'candidates_token_count', 0) or 0
        # Official billing: Total Output = candidates + thoughts
        assistant_tokens_out = c_count + t_count
        tokens_thought_val = t_count
    # OpenAI/xAI Responses: Use official usage if available
    elif (not is_gem) and locals().get('final_openai_usage'):
        usage = locals().get('final_openai_usage')
        completion_tokens_val = None
        output_tokens_val = None
        reasoning_tokens_val = None
        if isinstance(usage, dict):
            completion_tokens_val = usage.get('completion_tokens')
            output_tokens_val = usage.get('output_tokens')
            details = usage.get('completion_tokens_details')
            if not isinstance(details, dict):
                details = usage.get('output_tokens_details')
            if isinstance(details, dict):
                reasoning_tokens_val = details.get('reasoning_tokens', 0)
        else:
            completion_tokens_val = getattr(usage, 'completion_tokens', None)
            output_tokens_val = getattr(usage, 'output_tokens', None)
            details = getattr(usage, 'completion_tokens_details', None) or getattr(usage, 'output_tokens_details', None)
            if details:
                reasoning_tokens_val = getattr(details, 'reasoning_tokens', 0)

        if reasoning_tokens_val is not None:
            tokens_thought_val = reasoning_tokens_val

        # xAI usage commonly reports completion_tokens (reasoning separate and included in total only).
        if is_grok and completion_tokens_val is not None:
            assistant_tokens_out = int(completion_tokens_val or 0) + int(reasoning_tokens_val or 0)
        # OpenAI Responses usage reports output_tokens (already total output).
        elif output_tokens_val is not None:
            assistant_tokens_out = output_tokens_val
        elif completion_tokens_val is not None:
            assistant_tokens_out = int(completion_tokens_val or 0) + int(reasoning_tokens_val or 0)

    usage_totals = usage_ledger.token_totals()
    if usage_totals['output_tokens'] is not None:
        assistant_tokens_out = usage_totals['output_tokens']
    if usage_totals['thought_tokens'] is not None:
        tokens_thought_val = usage_totals['thought_tokens']

    skill_uuid_val = options.get('skill_uuid') or options.get('gem_uuid')
    skill_name_val = None
    if skill_uuid_val:
        skill = Skill.query.filter_by(uuid=skill_uuid_val).first()
        if skill:
            skill_name_val = skill.name
    if not skill_uuid_val and _used_skills:
        skill_uuid_val = _used_skills[0].get('uuid')
        skill_name_val = _used_skills[0].get('name')
    msg_entry = Message(
        thread_id=thread_id, role='assistant', content=final_content, 
        model=model_key, job_id=job_id,
        image_url=json.dumps(generated_images) if generated_images else None, 
        thought_data=final_thought, tokens_out=assistant_tokens_out, tokens=sum_token_counts(None, assistant_tokens_out), 
        is_encrypted=is_enc, thought_signature=final_signature,
        parent_id=message_id,
        tokens_thought=tokens_thought_val,
        skill_uuid=skill_uuid_val,
        skill_name=skill_name_val
    )
    db.session.add(msg_entry)
    th = Thread.query.get(thread_id)
    if th:
        th.updated_at = datetime.utcnow()
        th.last_model = model_key
        # A model-invoked use_skill is shown on the message only.  The
        # thread's last_skill_uuid is the user's explicit selection.
    safe_db_commit()
    pub("done", "OK")

    return None
