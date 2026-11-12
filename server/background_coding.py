# --- Background: Coding mode helpers ---

# region: Coding mode prompt
CODING_MODE_SYSTEM_PROMPT = """[Coding Mode]
You are editing the supplied existing code block. Minimize output tokens: never repeat the whole file.
Choose the code block that best matches the user's request unless the candidate list contains only an explicitly selected block.
Your final output must be newline-delimited JSON (NDJSON), with one compact JSON object per line and no Markdown fences or commentary:
{"type":"target","target_id":"candidate id","summary":"short description"}
{"type":"edit","search":"exact existing text","replace":"replacement text"}
{"type":"edit","search":"another exact existing text","replace":"replacement text"}
{"type":"done"}
Emit each edit line immediately after you decide it; do not wait to batch edits. Each search value must be non-empty and occur exactly once in the current code at the time that edit is applied. Include only the smallest exact span needed to make it unique. Use replace="" to delete. For insertion, replace a short unique anchor with the anchor plus the insertion. Apply multiple edits in output order. Preserve indentation and line endings. If no code change is needed, emit target and done without edit lines. You may use an available Python tool to inspect or validate the change, but the final output must still follow this NDJSON protocol."""
# endregion


# region: Code block extraction
def extract_markdown_code_blocks(markdown_text):
    completed = []
    active = None
    for line in str(markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if active is None:
            opening = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
            if not opening:
                continue
            info = str(opening.group(2) or "").strip()
            language = (info.split()[0] if info else "text")
            language = re.sub(r"^\{?\.?", "", language)
            language = re.sub(r"\}$", "", language) or "text"
            active = {
                "marker_char": opening.group(1)[0],
                "marker_length": len(opening.group(1)),
                "language": language,
                "buffer": [],
            }
            continue
        trimmed = str(line or "").strip()
        closing_pattern = rf"^{re.escape(active['marker_char'])}{{{active['marker_length']},}}\s*$"
        if re.match(closing_pattern, trimmed):
            code = "\n".join(active["buffer"])
            if code.strip():
                completed.append({"code": code, "language": active["language"]})
            active = None
            continue
        active["buffer"].append(line)
    return completed

def extract_latest_markdown_code_block(markdown_text):
    completed = extract_markdown_code_blocks(markdown_text)
    return completed[-1] if completed else None
# endregion


# region: Edit payload parsing and formatting
def _parse_coding_mode_edit_payload(model_output):
    raw = str(model_output or "").strip()
    if not raw:
        raise ValueError("モデルから編集内容が返されませんでした")
    ndjson_target_id = None
    ndjson_summary = ""
    ndjson_edits = []
    saw_ndjson_protocol = False
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "").strip().lower()
        if event_type == "target":
            saw_ndjson_protocol = True
            ndjson_target_id = str(event.get("target_id") or "").strip()[:100] or None
            ndjson_summary = str(event.get("summary") or "").strip()[:2000]
        elif event_type == "edit":
            saw_ndjson_protocol = True
            ndjson_edits.append({
                "search": event.get("search"),
                "replace": event.get("replace"),
            })
        elif event_type == "done":
            saw_ndjson_protocol = True
    if saw_ndjson_protocol:
        payload = {
            "target_id": ndjson_target_id,
            "summary": ndjson_summary,
            "edits": ndjson_edits,
        }
    else:
        payload = None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw, count=1)
    if payload is None:
        decoder = json.JSONDecoder()
        scanned_target_id = None
        scanned_summary = ""
        scanned_edits = []
        scanned_protocol = False
        for match in re.finditer(r"\{", raw):
            try:
                candidate, _ = decoder.raw_decode(raw[match.start():])
            except Exception:
                continue
            if not isinstance(candidate, dict):
                continue
            candidate_type = str(candidate.get("type") or "").strip().lower()
            if candidate_type == "target":
                scanned_protocol = True
                scanned_target_id = str(candidate.get("target_id") or "").strip()[:100] or None
                scanned_summary = str(candidate.get("summary") or "").strip()[:2000]
            elif candidate_type == "edit":
                scanned_protocol = True
                scanned_edits.append({
                    "search": candidate.get("search"),
                    "replace": candidate.get("replace"),
                })
            elif isinstance(candidate.get("edits"), list):
                payload = candidate
                break
        if payload is None and scanned_protocol:
            payload = {
                "target_id": scanned_target_id,
                "summary": scanned_summary,
                "edits": scanned_edits,
            }
    if payload is None:
        raise ValueError("editsを含む編集JSONを解析できません")
    edits = payload.get("edits")
    if not isinstance(edits, list) or len(edits) > 50:
        raise ValueError("editsは50件以下の配列である必要があります")
    summary = str(payload.get("summary") or "").strip()[:2000]
    normalized = []
    for item in edits:
        if not isinstance(item, dict):
            raise ValueError("各編集はJSONオブジェクトである必要があります")
        search = item.get("search")
        replace = item.get("replace")
        if not isinstance(search, str) or not search:
            raise ValueError("searchは空にできません")
        if not isinstance(replace, str):
            raise ValueError("replaceは文字列である必要があります")
        if len(search) > 100_000 or len(replace) > 100_000:
            raise ValueError("1件の編集が大きすぎます")
        normalized.append({"search": search, "replace": replace})
    return {
        "target_id": str(payload.get("target_id") or "").strip()[:100] or None,
        "summary": summary,
        "edits": normalized,
    }

def _markdown_fence_for_code(code):
    longest = max((len(run) for run in re.findall(r"`+", str(code or ""))), default=0)
    return "`" * max(3, longest + 1)

def build_coding_mode_unified_diff(before_code, after_code, language="text"):
    safe_language = re.sub(r"[^A-Za-z0-9_+.#-]", "", str(language or "text"))[:40] or "text"
    before_lines = str(before_code or "").splitlines(keepends=True)
    after_lines = str(after_code or "").splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"before.{safe_language}",
        tofile=f"after.{safe_language}",
        lineterm="\n",
    )
    return "".join(diff_lines).rstrip()

def build_coding_mode_final_markdown(summary, before_code, after_code, language="text"):
    safe_language = re.sub(r"[^A-Za-z0-9_+.#-]", "", str(language or "text"))[:40] or "text"
    code_fence = _markdown_fence_for_code(after_code)
    diff = build_coding_mode_unified_diff(before_code, after_code, safe_language)
    diff_section = f"```diff\n{diff}\n```\n\n" if diff else ""
    safe_summary = str(summary or "").strip() or ("変更を適用しました" if diff else "変更はありません")
    return (
        f"**Coding Mode:** {safe_summary}\n\n"
        f"{diff_section}"
        f"**更新後コード:**\n\n"
        f"{code_fence}{safe_language}\n{after_code}\n{code_fence}"
    )
# endregion


# region: Edit application
class CodingModeEditApplicationError(ValueError):
    def __init__(self, edit_index, occurrences, current_code, failed_edit, applied_steps):
        super().__init__(f"編集{edit_index}のsearch一致数が{occurrences}件です（1件必要）")
        self.edit_index = edit_index
        self.occurrences = occurrences
        self.current_code = current_code
        self.failed_edit = failed_edit
        self.applied_steps = applied_steps

def _apply_coding_mode_payload(payload, target_code):
    code = str(target_code or "")
    applied_steps = []
    for index, edit in enumerate(payload["edits"], start=1):
        occurrences = code.count(edit["search"])
        if occurrences != 1:
            raise CodingModeEditApplicationError(
                index,
                occurrences,
                code,
                edit,
                applied_steps,
            )
        before_code = code
        code = code.replace(edit["search"], edit["replace"], 1)
        if len(code) > 500_000:
            raise ValueError("適用後のコードが大きすぎます")
        applied_steps.append({
            "before_code": before_code,
            "after_code": code,
            "edit": edit,
        })
    return code, applied_steps

def apply_coding_mode_edits(model_output, target_code, language="text"):
    original_code = str(target_code or "")
    payload = _parse_coding_mode_edit_payload(model_output)
    code, _ = _apply_coding_mode_payload(payload, original_code)
    return build_coding_mode_final_markdown(
        payload["summary"],
        original_code,
        code,
        language,
    )
# endregion


# region: Target resolution
def _resolve_coding_mode_candidate(model_output, candidates, default_target_id=None):
    payload = _parse_coding_mode_edit_payload(model_output)
    candidate_map = {
        str(item.get("id")): item
        for item in (candidates or [])
        if isinstance(item, dict) and item.get("id")
    }
    target_id = payload.get("target_id")
    selected = candidate_map.get(target_id) if target_id else None
    if selected is None and payload["edits"]:
        first_search = payload["edits"][0]["search"]
        matching = [
            item for item in candidate_map.values()
            if str(item.get("code") or "").count(first_search) == 1
        ]
        if len(matching) == 1:
            selected = matching[0]
        elif target_id:
            raise ValueError("モデルが指定した編集対象IDが無効です")
        else:
            raise ValueError(f"編集対象を一意に判定できません（候補{len(matching)}件）")
    if selected is None:
        selected = candidate_map.get(str(default_target_id or ""))
    if selected is None:
        raise ValueError("編集対象コードを決定できません")
    return selected, payload

def apply_coding_mode_candidate_edits(model_output, candidates, default_target_id=None):
    selected, payload = _resolve_coding_mode_candidate(
        model_output,
        candidates,
        default_target_id,
    )
    normalized_output = json.dumps(payload, ensure_ascii=False)
    return apply_coding_mode_edits(
        normalized_output,
        selected.get("code"),
        selected.get("language"),
    )
# endregion


# region: Repair
def build_coding_mode_repair_prompt(
    user_instruction,
    target_id,
    language,
    current_code,
    failure,
    remaining_edits,
    explicitly_selected=False,
    attempt=1,
):
    selection_note = (
        "This target was explicitly selected by the user and must remain locked."
        if explicitly_selected
        else "The target was already chosen for this edit and must remain locked."
    )
    failed_edit = failure.failed_edit if isinstance(failure, CodingModeEditApplicationError) else {}
    repair_context = {
        "attempt": int(attempt),
        "target_id": str(target_id or ""),
        "language": str(language or "text"),
        "explicitly_selected": bool(explicitly_selected),
        "failure": str(failure),
        "failed_edit": failed_edit,
        "remaining_original_edits": remaining_edits or [],
    }
    return (
        "[Coding Mode Automatic Repair]\n"
        f"{selection_note}\n"
        "Continue from the CURRENT WORKING CODE below. Successful earlier edits are already included. "
        "Do not repeat them. Correct the failed edit and complete the user's original request. "
        "Return NDJSON only, using the same target_id. Emit at least one edit before done.\n\n"
        f"Original user request:\n{str(user_instruction or '')[:100000]}\n\n"
        f"Repair context:\n{json.dumps(repair_context, ensure_ascii=False)}\n\n"
        f"--- BEGIN CURRENT WORKING CODE ---\n{current_code}\n--- END CURRENT WORKING CODE ---"
    )

def _call_coding_mode_repair_model(user, model_key, repair_prompt):
    resolved = _resolve_chat_model_auth(user, model_key)
    if resolved.get("error_code"):
        raise RuntimeError(resolved.get("error") or "修復用モデルの認証情報がありません")
    provider = resolved.get("provider")
    api_key = resolved.get("api_key")
    if provider == "gemini":
        runtime = resolved.get("gemini_runtime") or _resolve_gemini_runtime(user)
        client = _get_gemini_client(
            api_key=api_key,
            backend=runtime.get("backend"),
            vertex_project=runtime.get("vertex_project"),
            vertex_location=runtime.get("vertex_location"),
            vertex_credentials_json=runtime.get("vertex_credentials_json"),
        )
        response = client.models.generate_content(
            model=model_key,
            contents=repair_prompt,
            config=types.GenerateContentConfig(system_instruction=CODING_MODE_SYSTEM_PROMPT),
        )
        return str(getattr(response, "text", None) or "")
    if provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic SDK is not installed")
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_key,
            max_tokens=8192,
            system=CODING_MODE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        return "".join(
            str(getattr(block, "text", "") or "")
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        )
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
            messages=[
                {"role": "system", "content": CODING_MODE_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ],
            max_tokens=8192,
        )
        return str(getattr(response.choices[0].message, "content", None) or "")
    response = client.responses.create(
        model=model_key,
        instructions=CODING_MODE_SYSTEM_PROMPT,
        input=repair_prompt,
        max_output_tokens=8192,
        store=False,
    )
    return _extract_openai_response_text(response)
# endregion

