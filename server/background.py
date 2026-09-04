# --- Background Tasks ---

def migrate_e2ee_task(user_id, target_enable):
    with app.app_context():
        r = redis.from_url(REDIS_URL)
        r.set(f"migration_status:{user_id}", "processing")
        try:
            user = User.query.get(user_id)
            if not user: return
            # Estimate total work units (messages + files)
            total = 0
            done = 0
            threads = Thread.query.filter_by(user_id=user_id).all()
            total += sum(len(t.messages) for t in threads)
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for root, _, files in os.walk(user_dir):
                    total += len(files)
            if total <= 0: total = 1
            r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            user.enable_e2ee = target_enable
            if user.system_prompt:
                if target_enable: user.system_prompt = encrypt_val(user.system_prompt)
                else: user.system_prompt = decrypt_val(user.system_prompt)
            for t in threads:
                for m in t.messages:
                    if m.content:
                        if target_enable and not m.is_encrypted: m.content = encrypt_val(m.content)
                        elif not target_enable and m.is_encrypted: m.content = decrypt_val(m.content)
                    if m.thought_data:
                        if target_enable and not m.is_encrypted: m.thought_data = encrypt_val(m.thought_data)
                        elif not target_enable and m.is_encrypted: m.thought_data = decrypt_val(m.thought_data)
                    m.is_encrypted = target_enable
                    done += 1
                    if done % 10 == 0:
                        r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for root, dirs, files in os.walk(user_dir):
                    for file in files:
                        fp = os.path.join(root, file)
                        if target_enable:
                            if not file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = f.read()
                                with open(fp + '.enc', 'wb') as f: f.write(encrypt_bytes(data))
                                secure_delete(fp)
                        else:
                            if file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = decrypt_bytes(f.read())
                                new_fp = fp[:-4]
                                with open(new_fp, 'wb') as f: f.write(data)
                                secure_delete(fp)
                        done += 1
                        if done % 5 == 0:
                            r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            safe_db_commit()
            r.set(f"migration_progress:{user_id}", f"{total}/{total}")
            r.set(f"migration_status:{user_id}", "done")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            r.set(f"migration_status:{user_id}", "error")
            r.set(f"migration_progress:{user_id}", "error")

def _sanitize_python_sandbox_output(value):
    """Remove host filesystem paths before Python output reaches the client."""
    text = str(value or "")
    text = re.sub(
        r"(?<![A-Za-z0-9_.-])/(?:home|root)/[^\s'\"`<>]+",
        "[host path redacted]",
        text,
    )
    app_root = os.path.abspath(os.path.dirname(__file__))
    text = text.replace(app_root, "[app path redacted]")
    return text


def safe_execute_python(code):
    """Executes Python code in a restricted environment using bubblewrap."""
    import subprocess
    import tempfile
    import os
    import shutil

    # The sandbox only exposes the system filesystem below.  ``shutil.which``
    # normally resolves to the application's venv, but that host path is not
    # mounted into bubblewrap and therefore cannot be executed there.
    py_path = None
    for candidate in ("/usr/bin/python3", "/bin/python3"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            py_path = candidate
            break
    if not py_path:
        candidate = shutil.which("python3")
        if candidate:
            resolved = os.path.realpath(candidate)
            if resolved.startswith(("/usr/", "/bin/")) and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                py_path = resolved
    if not py_path:
        return "Error: python3 not found."

    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "Error: Python execution disabled (sandbox not available)."
    prlimit = shutil.which("prlimit")
    if not prlimit:
        return "Error: Python execution disabled (resource limiter not available)."
    code = str(code or "")
    if len(code.encode("utf-8", errors="ignore")) > 256 * 1024:
        return "Error: Python code exceeds the 256KB limit."

    with tempfile.TemporaryDirectory() as td:
        code_path = os.path.join(td, "code.py")
        with open(code_path, "w") as f:
            f.write(code)
        binds = [
            ("--ro-bind", "/usr", "/usr"),
            ("--ro-bind", "/bin", "/bin"),
        ]
        for p in ["/lib", "/lib64"]:
            if os.path.exists(p):
                binds.append(("--ro-bind", p, p))
        cmd = [
            prlimit,
            "--cpu=15",
            "--as=536870912",
            "--fsize=8388608",
            # RLIMIT_NPROC is enforced for the whole service UID, not only
            # processes spawned by this sandbox.  The gunicorn/RQ threads
            # already consume part of that UID-wide budget, so 64 can make
            # bwrap fail before Python starts (EAGAIN) in a healthy service.
            # Keep a bounded sandbox process budget while leaving headroom for
            # the service's resident threads and the namespace helper.
            "--nproc=256",
            "--nofile=64",
            "--",
            bwrap,
            "--unshare-net",
            "--unshare-uts",
            "--unshare-pid",
            "--unshare-ipc",
            "--die-with-parent",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/home",
            "--tmpfs", "/var",
            "--dir", "/tmp",  # nosec B108 - private path inside the bwrap mount namespace
            "--chdir", "/tmp",  # nosec B108 - private path inside the bwrap mount namespace
        ]
        for b in binds:
            cmd.extend(list(b))
        cmd.extend(["--bind", td, "/work", py_path, "/work/code.py"])
        try:
            with tempfile.TemporaryFile(mode="w+b") as output:
                subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, timeout=30, check=False)
                output.seek(0)
                raw = output.read(2 * 1024 * 1024 + 1)
            truncated = len(raw) > 2 * 1024 * 1024
            out = raw[:2 * 1024 * 1024].decode("utf-8", errors="replace")
            if truncated:
                out += "\n[Output truncated at 2MB]"
            out = _sanitize_python_sandbox_output(out)
            return out if out.strip() else "Success (No output)"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (30s limit)"
        except Exception as e:
            return _sanitize_python_sandbox_output(f"Error: {e}")


def accumulate_deepseek_tool_call_deltas(tool_call_state, delta_tool_calls):
    """Merge streamed OpenAI-compatible tool call fragments by their stable index."""
    for position, tool_delta in enumerate(delta_tool_calls or []):
        if isinstance(tool_delta, dict):
            tool_index = tool_delta.get("index")
            call_id = tool_delta.get("id")
            call_type = tool_delta.get("type")
            function_delta = tool_delta.get("function") or {}
            function_name = function_delta.get("name") if isinstance(function_delta, dict) else None
            function_args = function_delta.get("arguments") if isinstance(function_delta, dict) else None
        else:
            tool_index = getattr(tool_delta, "index", None)
            call_id = getattr(tool_delta, "id", None)
            call_type = getattr(tool_delta, "type", None)
            function_delta = getattr(tool_delta, "function", None)
            function_name = getattr(function_delta, "name", None) if function_delta else None
            function_args = getattr(function_delta, "arguments", None) if function_delta else None
        if tool_index is None:
            tool_index = position
        tool_state = tool_call_state.setdefault(
            int(tool_index),
            {"id": "", "type": "function", "name": "", "arguments": ""},
        )
        if call_id:
            tool_state["id"] = call_id
        if call_type:
            tool_state["type"] = call_type
        if function_name:
            tool_state["name"] += function_name
        if function_args:
            tool_state["arguments"] += function_args
    return tool_call_state


CODING_MODE_SYSTEM_PROMPT = """[Coding Mode]
You are editing the supplied existing code block. Minimize output tokens: never repeat the whole file.
Choose the code block that best matches the user's request unless the candidate list contains only an explicitly selected block.
Your final output must be newline-delimited JSON (NDJSON), with one compact JSON object per line and no Markdown fences or commentary:
{"type":"target","target_id":"candidate id","summary":"short description"}
{"type":"edit","search":"exact existing text","replace":"replacement text"}
{"type":"edit","search":"another exact existing text","replace":"replacement text"}
{"type":"done"}
Emit each edit line immediately after you decide it; do not wait to batch edits. Each search value must be non-empty and occur exactly once in the current code at the time that edit is applied. Include only the smallest exact span needed to make it unique. Use replace="" to delete. For insertion, replace a short unique anchor with the anchor plus the insertion. Apply multiple edits in output order. Preserve indentation and line endings. If no code change is needed, emit target and done without edit lines. You may use an available Python tool to inspect or validate the change, but the final output must still follow this NDJSON protocol."""

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
    elif provider == "kimi":
        base_url = "https://api.moonshot.ai/v1"
    elif provider == "xai":
        base_url = f"https://{_XAI_API_HOST}/v1"
    client = _get_openai_client(api_key, base_url=base_url)
    if provider in {"deepseek", "kimi", "xai"}:
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

def background_chat_task(job_id, thread_id, model_key, message_id, options, user_id, user_config):
    with app.app_context():
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        _latency_mark_once(job_id, "worker_started_ms")
        coding_stream_buffer = ""
        coding_stream_target_id = None
        coding_stream_code = None
        coding_stream_language = "text"
        coding_stream_edit_index = 0
        coding_candidate_map = {
            str(item.get("id")): item
            for item in (options.get("coding_candidates") or [])
            if isinstance(item, dict) and item.get("id")
        }

        def _emit_coding_diff(content):
            event_payload = {"type": "coding_diff", "content": content}
            r.publish(channel, json.dumps(event_payload, ensure_ascii=False))
            try:
                cache_key = f"stream_acc:{job_id}:coding_diff"
                r.rpush(cache_key, json.dumps(content, ensure_ascii=False))
                r.ltrim(cache_key, -50, -1)
                r.expire(cache_key, 600)
            except Exception:
                pass

        def _process_coding_stream_line(raw_line):
            nonlocal coding_stream_target_id
            nonlocal coding_stream_code
            nonlocal coding_stream_language
            nonlocal coding_stream_edit_index
            line = str(raw_line or "").strip()
            if not line or line.startswith("```"):
                return
            try:
                event = json.loads(line)
            except Exception:
                return
            if not isinstance(event, dict):
                return
            event_type = str(event.get("type") or "").strip().lower()
            if event_type == "target":
                target_id = str(event.get("target_id") or "").strip()
                candidate = coding_candidate_map.get(target_id)
                if candidate:
                    coding_stream_target_id = target_id
                    coding_stream_code = str(candidate.get("code") or "")
                    coding_stream_language = str(candidate.get("language") or "text")
                return
            if event_type != "edit":
                return
            search = event.get("search")
            replace = event.get("replace")
            if not isinstance(search, str) or not search or not isinstance(replace, str):
                return
            if coding_stream_code is None:
                matching = [
                    candidate for candidate in coding_candidate_map.values()
                    if str(candidate.get("code") or "").count(search) == 1
                ]
                if len(matching) != 1:
                    return
                candidate = matching[0]
                coding_stream_target_id = str(candidate.get("id") or "")
                coding_stream_code = str(candidate.get("code") or "")
                coding_stream_language = str(candidate.get("language") or "text")
            if coding_stream_code.count(search) != 1:
                return
            before_code = coding_stream_code
            after_code = before_code.replace(search, replace, 1)
            if len(after_code) > 500_000:
                return
            coding_stream_code = after_code
            coding_stream_edit_index += 1
            _latency_mark_once(job_id, "provider_first_content_ms")
            _emit_coding_diff({
                "target_id": coding_stream_target_id,
                "language": coding_stream_language,
                "edit_index": coding_stream_edit_index,
                "diff": build_coding_mode_unified_diff(
                    before_code,
                    after_code,
                    coding_stream_language,
                ),
            })

        def _consume_coding_stream_chunk(chunk, flush=False):
            nonlocal coding_stream_buffer
            coding_stream_buffer += str(chunk or "")
            while "\n" in coding_stream_buffer:
                line, coding_stream_buffer = coding_stream_buffer.split("\n", 1)
                _process_coding_stream_line(line)
            if flush and coding_stream_buffer.strip():
                _process_coding_stream_line(coding_stream_buffer)
                coding_stream_buffer = ""

        def _append_limited(key, chunk, limit=1_000_000):
            try:
                if chunk is None:
                    return
                if not isinstance(chunk, str):
                    chunk = str(chunk)
                r.append(key, chunk)
                size = r.strlen(key)
                if size and size > limit:
                    curr = r.get(key) or b""
                    if len(curr) > limit:
                        r.set(key, curr[-limit:])
                r.expire(key, 600)
            except Exception:
                pass

        # Persist stream errors as assistant messages so they remain visible after reload.
        stream_error_persisted = False

        def _persist_stream_error(error_text):
            """Save an assistant bubble for stream failures so reloads still show the error."""
            nonlocal stream_error_persisted
            if stream_error_persisted:
                return
            stream_error_persisted = True
            try:
                parent = Message.query.get(message_id)
                if not parent or parent.thread_id != thread_id:
                    return
                if getattr(parent, "thread", None) is not None and parent.thread.user_id != user_id:
                    return

                partial = ""
                try:
                    cached = r.get(f"stream_acc:{job_id}:content")
                    if cached:
                        partial = cached.decode("utf-8", "ignore")
                except Exception:
                    partial = ""
                thought_text = ""
                try:
                    cached_thought = r.get(f"stream_acc:{job_id}:thought")
                    if cached_thought:
                        thought_text = cached_thought.decode("utf-8", "ignore")
                except Exception:
                    thought_text = ""

                final_content = format_chat_error_content(error_text, partial)
                is_enc = bool(user_config.get("enable_e2ee", False))
                content_to_store = encrypt_val(final_content) if is_enc else final_content
                thought_data = None
                if thought_text and str(thought_text).strip():
                    thought_payload = json.dumps({"text": thought_text}, ensure_ascii=False)
                    thought_data = encrypt_val(thought_payload) if is_enc else thought_payload

                gem_uuid_val = options.get("gem_uuid")
                gem_name_val = None
                if gem_uuid_val:
                    gem = Gem.query.filter_by(uuid=gem_uuid_val).first()
                    if gem:
                        gem_name_val = gem.name

                msg_entry = Message(
                    thread_id=thread_id,
                    role="assistant",
                    content=content_to_store,
                    model=model_key,
                    thought_data=thought_data,
                    tokens_out=0,
                    tokens=0,
                    is_encrypted=is_enc,
                    parent_id=message_id,
                    gem_uuid=gem_uuid_val,
                    gem_name=gem_name_val,
                )
                db.session.add(msg_entry)
                th = Thread.query.get(thread_id)
                if th:
                    th.updated_at = datetime.utcnow()
                    th.last_model = model_key
                    if gem_uuid_val:
                        th.last_gem_uuid = gem_uuid_val
                safe_db_commit()
            except Exception as exc:
                logger.exception(
                    "Failed to persist stream error message for job %s: %s", job_id, exc
                )
                try:
                    db.session.rollback()
                except Exception:
                    pass

        def pub(dt, d, **metadata):
            if options.get("coding_mode") and dt == "content" and not metadata.get("coding_final"):
                _consume_coding_stream_chunk(d)
                return
            if dt == "status":
                _latency_mark_once(job_id, "provider_first_status_ms")
            elif dt == "thought":
                _latency_mark_once(job_id, "provider_first_thought_ms")
            elif dt == "content":
                _latency_mark_once(job_id, "provider_first_content_ms")
            elif dt in ("done", "error"):
                _latency_mark_once(job_id, "worker_done_ms")
            # Persist before publish so a client reload right after the error event
            # already finds the saved assistant bubble.
            if dt == "error":
                _persist_stream_error(d)
            event_payload = {"type": dt, "content": d}
            event_payload.update(metadata)
            r.publish(channel, json.dumps(event_payload))
            try:
                if dt == "content":
                    _append_limited(f"stream_acc:{job_id}:content", d)
                elif dt == "thought":
                    _append_limited(f"stream_acc:{job_id}:thought", d)
                elif dt == "status":
                    r.setex(f"stream_acc:{job_id}:status", 600, d)
                elif dt == "python":
                    py = d if isinstance(d, dict) else {}
                    py_id = py.get("id") or "default"
                    r.hset(f"stream_acc:{job_id}:python", py_id, json.dumps(py))
                    r.expire(f"stream_acc:{job_id}:python", 600)
                elif dt == "search_status":
                    r.setex(f"stream_acc:{job_id}:search", 600, d)
                elif dt in ("mcp", "mcp_decision_request", "mcp_decision_resolved"):
                    # MCP実行カード・確認ダイアログ用イベントのリプレイ保存（発生順）
                    try:
                        entry = {"type": dt, "content": d}
                        entry.update(metadata)
                        r.rpush(f"stream_acc:{job_id}:mcp", json.dumps(entry, ensure_ascii=False))
                        r.ltrim(f"stream_acc:{job_id}:mcp", -200, -1)
                        r.expire(f"stream_acc:{job_id}:mcp", 600)
                    except Exception:
                        pass
                elif dt in ["error", "done"]:
                    r.setex(f"stream_acc:{job_id}:final", 600, dt)
                    if dt == "error":
                        r.setex(f"stream_acc:{job_id}:error", 600, json.dumps(event_payload))
            except Exception:
                pass
        
        def check_stop():
            try:
                res = r.get(f"stop_job:{job_id}")
                if res:
                    # Clear it immediately to avoid double processing if needed
                    # but actually we want all loops to see it if multiple.
                    # r.delete(f"stop_job:{job_id}") 
                    log_force(f"STREAM-STOP-DETECTED: Job {job_id} stop flag found in Redis.")
                    return True
            except Exception as e:
                log_force(f"STREAM-STOP-ERROR: Failed to check stop flag: {e}")
            return False

        def _refresh_pending_job():
            try:
                pending_data = json.dumps({
                    "job_id": job_id,
                    "message_id": message_id,
                    "created_at": int(time.time()),
                    "model": model_key,
                })
                r.setex(f"pending_job:{user_id}:{thread_id}", 600, pending_data)
            except Exception:
                pass

        def _mark_provider_request_started():
            _latency_mark_once(job_id, "provider_request_started_ms")
        
        def _decode_text_bytes(raw):
            return _decode_text_bytes_for_prompt(raw)

        try:
            log_force(f"Task Start: model={model_key}, user={user_id}")
            pub("status", "ワーカーがジョブを受信しました。入力を処理中です...")
            user = User.query.get(user_id)
            msg = Message.query.get(message_id)
            if not msg or msg.thread_id != thread_id or msg.thread.user_id != user_id:
                pub("error", "Invalid message")
                return
            message_text = decrypt_val(msg.content) if msg.is_encrypted else msg.content
            img_list = []
            if msg.image_url:
                try:
                    img_list = json.loads(msg.image_url)
                    if not isinstance(img_list, list):
                        img_list = [img_list]
                except: pass
            max_files = int(app.config.get('ATTACHMENT_MAX_FILES') or 30)
            if len(img_list) > max_files:
                pub("error", f"添付ファイルは最大{max_files}件です。ファイル数を減らして再送してください。")
                return
            # System Prompt Construction
            base_sys_prompt = options.get('system_prompt')
            if not base_sys_prompt:
                try:
                    sv = r.get(f"sys:{job_id}")
                    if sv: base_sys_prompt = sv.decode('utf-8')
                except: pass
                finally:
                    try: r.delete(f"sys:{job_id}")
                    except: pass
            
            forced_prompt = base_sys_prompt or ""
            global_prompt = None
            user_prompt = None
            use_time_notice = False
            apply_global_prompt = True
            apply_auto_prompt_notices = get_user_auto_system_prompt_notices_enabled(user)
            auto_notice_config = get_user_auto_system_prompt_notices_config(user)
            def _auto_notice_enabled(notice_key):
                return bool(
                    apply_auto_prompt_notices
                    and get_user_auto_system_prompt_notice_enabled(user, notice_key, auto_notice_config)
                )
            def _auto_notice_text(notice_key):
                return get_user_auto_system_prompt_notice_text(user, notice_key, auto_notice_config)
            def _build_attachment_name_block(names):
                if not (is_llm_model and _auto_notice_enabled("attachment_names")):
                    return ""
                template_text = _auto_notice_text("attachment_names")
                return _render_attachment_names_notice(template_text, names)
            # Fetch thread to check instructions
            th = Thread.query.get(thread_id)
            include_global = th.include_global_instruction if th and th.include_global_instruction is not None else True

            try:
                if getattr(user, "apply_global_system_prompt", None) is False:
                    apply_global_prompt = False
            except Exception:
                apply_global_prompt = True
            
            if apply_global_prompt and include_global:
                global_enabled = get_bool_app_setting("global_system_prompt_enabled", True)
                global_value = get_app_setting("global_system_prompt", "") or ""
                if global_enabled:
                    if global_value.strip():
                        global_prompt = global_value
                    else:
                        use_time_notice = True
            
            if options.get('enable_system_prompt') and include_global:
                if user.system_prompt and (user.system_prompt_enabled is None or user.system_prompt_enabled):
                    sp = user.system_prompt
                    if user.enable_e2ee: sp = decrypt_val(sp)
                    user_prompt = sp

            # Thread specific prompt
            local_sys_prompt = None
            if 'thread_custom_instruction' in options:
                raw_local_sys_prompt = options.get('thread_custom_instruction')
            else:
                raw_local_sys_prompt = th.custom_instruction if th else None
            if raw_local_sys_prompt and str(raw_local_sys_prompt).strip():
                local_sys_prompt = str(raw_local_sys_prompt).strip()
            
            combined_prompt = ""
            for part in [forced_prompt, global_prompt, user_prompt]:
                if part and str(part).strip():
                    if combined_prompt:
                        combined_prompt = f"{combined_prompt}\n\n{part}"
                    else:
                        combined_prompt = str(part).strip()
            if local_sys_prompt:
                if combined_prompt:
                    combined_prompt = f"{combined_prompt}\n\n[Chat Specific Instructions]:\n{local_sys_prompt}"
                else:
                    combined_prompt = local_sys_prompt
            if options.get("coding_mode"):
                if combined_prompt:
                    combined_prompt = f"{combined_prompt}\n\n{CODING_MODE_SYSTEM_PROMPT}"
                else:
                    combined_prompt = CODING_MODE_SYSTEM_PROMPT
            
            options['system_prompt'] = combined_prompt

            if _auto_notice_enabled("python") and options.get('enable_python'):
                python_notice = _auto_notice_text("python")
                curr_p = options.get('system_prompt')
                if curr_p and str(curr_p).strip():
                    if python_notice.lower() not in str(curr_p).lower():
                        options['system_prompt'] = f"{python_notice}\n\n{curr_p}"
                else:
                    options['system_prompt'] = python_notice
            if _auto_notice_enabled("marker"):
                marker_prompt = options.get('marker_system_prompt')
                if marker_prompt and str(marker_prompt).strip():
                    marker_notice = _auto_notice_text("marker")
                    curr_p = options.get('system_prompt') or ""
                    if curr_p.strip():
                        if str(marker_notice).strip() not in str(curr_p):
                            options['system_prompt'] = f"{curr_p}\n\n{marker_notice}"
                    else:
                        options['system_prompt'] = marker_notice
            if use_time_notice:
                time_notice = build_global_system_prompt()
                curr_p = options.get('system_prompt') or ""
                if curr_p.strip():
                    options['system_prompt'] = f"{time_notice}\n\n{curr_p}"
                else:
                    options['system_prompt'] = time_notice
            
            if _auto_notice_enabled("mathjax"):
                mathjax_notice = _auto_notice_text("mathjax")
                curr_p = options.get('system_prompt') or ""
                if "MathJax" not in curr_p:
                    if curr_p.strip():
                        options['system_prompt'] = f"{curr_p}\n\n{mathjax_notice}"
                    else:
                        options['system_prompt'] = mathjax_notice

            quote_text = None
            try:
                qv = r.get(f"quote:{job_id}")
                if qv: quote_text = qv.decode('utf-8')
            except: pass
            finally:
                try: r.delete(f"quote:{job_id}")
                except: pass

            # Reconstruct history by traversing UP the tree (parent_id)
            # The current message (msg) is the User's new prompt. We need its ancestors.
            
            history_rev = []
            total_history_tokens = 0
            try:
                # 0 means unlimited.
                MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "0") or "0")
            except Exception:
                MAX_CONTEXT_TOKENS = 0
            if MAX_CONTEXT_TOKENS < 0:
                MAX_CONTEXT_TOKENS = 0
            try:
                # 0 means unlimited.
                MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "0") or "0")
            except Exception:
                MAX_CONTEXT_MESSAGES = 0
            if MAX_CONTEXT_MESSAGES < 0:
                MAX_CONTEXT_MESSAGES = 0
            history_count = 0
            
            # Load all messages for the thread once to avoid N+1 sequential queries when traversing parent_id
            all_thread_msgs = Message.query.filter_by(thread_id=thread_id).all()
            msg_map = {m.id: m for m in all_thread_msgs}
            
            current_node = msg_map.get(msg.parent_id) if msg.parent_id else None
            if current_node and current_node.thread.user_id != user_id:
                current_node = None
            
            messages_to_update = False
            while current_node:
                if MAX_CONTEXT_MESSAGES and history_count >= MAX_CONTEXT_MESSAGES:
                    break
                raw_cnt = current_node.content or ""
                cached_tokens = None
                try:
                    if current_node.role == 'user' and current_node.tokens_in and current_node.tokens_in > 0:
                        cached_tokens = int(current_node.tokens_in)
                    elif current_node.role == 'assistant' and current_node.tokens_out and current_node.tokens_out > 0:
                        cached_tokens = int(current_node.tokens_out)
                except Exception:
                    cached_tokens = None
                if cached_tokens is not None:
                    t_len = cached_tokens
                else:
                    token_src = decrypt_val(raw_cnt) if current_node.is_encrypted else raw_cnt
                    token_model = current_node.model or model_key
                    t_len = max(1, count_tokens(token_src or "", token_model))
                    # Mark for single commit at the end of reconstruction
                    try:
                        if current_node.role == 'user':
                            current_node.tokens_in = t_len
                        else:
                            current_node.tokens_out = t_len
                        messages_to_update = True
                    except: pass
                
                if (not MAX_CONTEXT_TOKENS) or (total_history_tokens + t_len <= MAX_CONTEXT_TOKENS):
                    cnt = decrypt_val(raw_cnt) if current_node.is_encrypted else raw_cnt
                    deepseek_tool_context = None
                    if current_node.role == "assistant" and current_node.thought_data:
                        try:
                            raw_thought = (
                                decrypt_val(current_node.thought_data)
                                if current_node.is_encrypted
                                else current_node.thought_data
                            )
                            parsed_thought = json.loads(raw_thought)
                            if isinstance(parsed_thought, dict):
                                candidate_context = parsed_thought.get("deepseek_tool_context")
                                if isinstance(candidate_context, list):
                                    deepseek_tool_context = candidate_context
                        except Exception:
                            deepseek_tool_context = None
                    history_rev.append({
                        'role': current_node.role, 
                        'content': cnt, 
                        'image_url': current_node.image_url, 
                        'signature': current_node.thought_signature,
                        'deepseek_tool_context': deepseek_tool_context,
                    })
                    total_history_tokens += t_len
                    history_count += 1
                else:
                    break
                
                current_node = msg_map.get(current_node.parent_id) if current_node.parent_id else None
            
            # Commit any token count updates in a single batch
            if messages_to_update:
                try:
                    safe_db_commit()
                except Exception:
                    pass
            
            history = list(reversed(history_rev))

            def _load_history_image_parts(include_roles=None, newest_first=False, include_only_images=True):
                parts = []
                seen = set()
                total_bytes = 0
                src_messages = list(reversed(history)) if newest_first else history
                for m in src_messages:
                    role = str(m.get('role') or '').strip().lower()
                    if include_roles and role not in include_roles:
                        continue
                    raw_urls = m.get('image_url')
                    if not raw_urls:
                        continue
                    try:
                        ref_list = json.loads(raw_urls)
                    except Exception:
                        ref_list = raw_urls
                    if not isinstance(ref_list, list):
                        ref_list = [ref_list]
                    for ref in ref_list:
                        if _HISTORY_IMAGE_MAX_ITEMS and len(parts) >= _HISTORY_IMAGE_MAX_ITEMS:
                            return parts
                        norm_h = _normalize_upload_ref(ref)
                        if not norm_h or norm_h in seen:
                            continue
                        info_h = _get_file_disk_info(norm_h)
                        if not info_h.get("exists"):
                            continue
                        est_size = info_h.get("size") or 0
                        if _HISTORY_IMAGE_MAX_BYTES and est_size and (total_bytes + est_size > _HISTORY_IMAGE_MAX_BYTES):
                            continue
                        data_h = _load_user_file_bytes(norm_h, info_h)
                        if not data_h:
                            continue
                        if _HISTORY_IMAGE_MAX_BYTES and (total_bytes + len(data_h) > _HISTORY_IMAGE_MAX_BYTES):
                            continue
                        mime_h = _normalize_media_mime(norm_h, mimetypes.guess_type(norm_h)[0] or 'application/octet-stream')
                        if include_only_images and not str(mime_h).startswith('image/'):
                            continue
                        parts.append({
                            "role": role,
                            "ref": norm_h,
                            "bytes": data_h,
                            "mime": mime_h,
                            "name": os.path.basename(norm_h),
                            "content": m.get('content') or ""
                        })
                        seen.add(norm_h)
                        total_bytes += len(data_h)
                return parts

            def _load_message_history_images(raw_urls, seen=None, total_bytes=0, include_only_images=True):
                items = []
                if not raw_urls:
                    return items, total_bytes
                if seen is None:
                    seen = set()
                try:
                    ref_list = json.loads(raw_urls)
                except Exception:
                    ref_list = raw_urls
                if not isinstance(ref_list, list):
                    ref_list = [ref_list]
                for ref in ref_list:
                    if _HISTORY_IMAGE_MAX_ITEMS and len(seen) >= _HISTORY_IMAGE_MAX_ITEMS:
                        break
                    norm_h = _normalize_upload_ref(ref)
                    if not norm_h or norm_h in seen:
                        continue
                    info_h = _get_file_disk_info(norm_h)
                    if not info_h.get("exists"):
                        continue
                    est_size = info_h.get("size") or 0
                    if _HISTORY_IMAGE_MAX_BYTES and est_size and (total_bytes + est_size > _HISTORY_IMAGE_MAX_BYTES):
                        continue
                    data_h = _load_user_file_bytes(norm_h, info_h)
                    if not data_h:
                        continue
                    if _HISTORY_IMAGE_MAX_BYTES and (total_bytes + len(data_h) > _HISTORY_IMAGE_MAX_BYTES):
                        continue
                    mime_h = _normalize_media_mime(norm_h, mimetypes.guess_type(norm_h)[0] or 'application/octet-stream')
                    if include_only_images and not str(mime_h).startswith('image/'):
                        continue
                    items.append({
                        "ref": norm_h,
                        "bytes": data_h,
                        "mime": mime_h,
                        "name": os.path.basename(norm_h)
                    })
                    seen.add(norm_h)
                    total_bytes += len(data_h)
                return items, total_bytes

            def _build_non_llm_image_context(current_text, include_assistant_images=True):
                max_context_chars = 12000
                text_lines = []
                for m in history:
                    role = 'User' if m.get('role') == 'user' else 'Assistant'
                    msg_text = (m.get('content') or '').strip()
                    image_count = 0
                    try:
                        raw_urls = m.get('image_url')
                        if raw_urls:
                            parsed_urls = json.loads(raw_urls)
                            if isinstance(parsed_urls, list):
                                image_count = len(parsed_urls)
                            elif parsed_urls:
                                image_count = 1
                    except Exception:
                        image_count = 1 if m.get('image_url') else 0
                    if msg_text:
                        text_lines.append(f"{role}: {msg_text}")
                    elif image_count:
                        text_lines.append(f"{role}: [attached {image_count} image(s)]")
                history_images = _load_history_image_parts(
                    include_roles={"user", "assistant"} if include_assistant_images else {"user"},
                    newest_first=True,
                    include_only_images=True
                )
                if not text_lines and not history_images:
                    return current_text, history_images
                if text_lines:
                    combined_text = "\n".join(text_lines)
                    if len(combined_text) > max_context_chars:
                        combined_text = combined_text[-max_context_chars:]
                        text_lines = ["[earlier context trimmed]", combined_text]
                prompt_sections = [
                    "Conversation context for this image follow-up:",
                    "\n".join(text_lines) if text_lines else "(no prior text context)",
                    "Current user request:",
                    current_text
                ]
                return "\n\n".join([section for section in prompt_sections if section]), history_images

            model_key = model_key.strip()
            model_key_l = model_key.lower()
            is_openai_search_model = model_key_l in ("gpt-5-search-api", "gpt-4o-search-preview", "gpt-4o-mini-search-preview")
            is_gem = is_gemini_model_key(model_key_l)
            is_claude = is_anthropic_model_key(model_key_l)
            is_deepseek = is_deepseek_model_key(model_key_l)
            is_kimi = 'kimi' in model_key_l
            is_grok = 'grok' in model_key_l and 'gpt' not in model_key_l
            is_mistral_ocr = is_mistral_ocr_model_key(model_key_l)
            gemini_backend_mode = "gemini_api"
            def _is_non_llm_model(m):
                mk = str(m or "").lower().strip()
                if not mk:
                    return False
                if is_mistral_ocr_model_key(mk):
                    return True
                if "gpt-image" in mk:
                    return True
                if mk in (
                    "grok-imagine-image-2.0",
                    "grok-imagine-image",
                    "grok-imagine-image-pro",
                    "grok-imagine-image-quality",
                    "grok-imagine-video-1.5",
                    "grok-imagine-video",
                ):
                    return True
                if "tts" in mk:
                    return True
                if is_gemini_image_model_key(mk):
                    return True
                if "gemini" in mk and "native-audio" in mk:
                    return True
                if is_gemini_video_model_key(mk) or is_gemini_music_model_key(mk) or is_gemini_embedding_model_key(mk):
                    return True
                return False
            is_llm_model = not _is_non_llm_model(model_key_l)
            grok_reasoning_supported = ("grok-4.3" in model_key_l) or ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l) or ("grok-build" in model_key_l) or ("grok-3-mini" in model_key_l) or ("reasoning" in model_key_l and "non-reasoning" not in model_key_l) or ("multi-agent" in model_key_l)
            grok_reasoning_effort_supported = ("grok-4.3" in model_key_l) or ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l) or ("grok-build" in model_key_l) or ("grok-3-mini" in model_key_l) or ("grok-4.20-0309-reasoning" in model_key_l) or ("multi-agent" in model_key_l)
            req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
            reasoning_requested = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
            if is_deepseek and req_reasoning_effort == "none":
                reasoning_requested = False
            if reasoning_requested:
                pub("status", "推論プロセスを準備中です。モデルの初回トークンを待機しています...")
            else:
                pub("status", "モデルに接続中です。初回トークンを待機しています...")

            def _is_gemini_text_model(m):
                if "gemini" not in m:
                    return False
                if any(x in m for x in ("image", "nano", "tts", "native-audio")):
                    return False
                return True

            supports_audio_inputs = _is_gemini_text_model(model_key_l)
            supports_video_inputs = supports_audio_inputs
            supports_pdf_inputs = supports_audio_inputs
            supports_docx_inputs = supports_audio_inputs
            supports_text_file_inputs = supports_audio_inputs

            def _openai_cache_fresh(cache, size, mtime, mime):
                if not cache or not cache.file_id:
                    return False
                if size is not None and cache.size_bytes is not None and cache.size_bytes != size:
                    return False
                if mtime is not None and cache.mtime is not None and cache.mtime != mtime:
                    return False
                if mime and cache.mime_type and cache.mime_type != mime:
                    return False
                ttl_hours = 24
                try:
                    ttl_val = os.getenv("OPENAI_FILE_CACHE_TTL_HOURS")
                    if ttl_val and str(ttl_val).strip():
                        ttl_hours = int(ttl_val)
                except Exception:
                    ttl_hours = 24
                try:
                    if ttl_hours > 0 and cache.updated_at:
                        age = (datetime.utcnow() - cache.updated_at).total_seconds()
                        if age > ttl_hours * 3600:
                            return False
                except Exception:
                    pass
                return True

            def _openai_upload_with_retry(client, data, suffix, rel_path, mime=None, size=None, mtime=None):
                max_attempts = 2
                try:
                    max_attempts = int(os.getenv("OPENAI_FILE_UPLOAD_RETRIES", "2") or "2")
                except Exception:
                    max_attempts = 2
                last_err = None
                for attempt in range(max_attempts):
                    try:
                        _upsert_file_cache(
                            user_id,
                            rel_path,
                            "openai",
                            state="UPLOADING",
                            last_error=None,
                            retries=attempt + 1
                        )
                        safe_db_commit()
                        with tempfile.NamedTemporaryFile(suffix=suffix or '.bin') as tmp:
                            tmp.write(data)
                            tmp.flush()
                            tmp.seek(0)
                            up = client.files.create(file=tmp, purpose="user_data")
                        file_id = getattr(up, "id", None) or (up.get("id") if isinstance(up, dict) else None)
                        if not file_id:
                            last_err = "file_id missing"
                            time.sleep(1)
                            continue
                        _upsert_file_cache(
                            user_id,
                            rel_path,
                            "openai",
                            file_id=file_id,
                            file_uri=None,
                            state="ACTIVE",
                            last_error=None,
                            size_bytes=size if size is not None else (len(data) if data is not None else None),
                            mtime=mtime,
                            mime_type=mime,
                            last_checked_at=datetime.utcnow()
                        )
                        safe_db_commit()
                        return file_id, None
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(1)
                        continue
                _upsert_file_cache(
                    user_id,
                    rel_path,
                    "openai",
                    state="FAILED",
                    last_error=last_err
                )
                safe_db_commit()
                return None, last_err

            def _grok_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                is_grok_45 = "grok-4.5" in model_key_l
                is_grok_46 = "grok-4.6" in model_key_l
                if is_grok_45 or is_grok_46:
                    # Grok 4.5 / 4.6: reasoning cannot be disabled (defaults to high).
                    # "xhigh" is only supported by grok-4.6; grok-4.5 treats it as "high".
                    if raw in ("low", "medium", "high"):
                        return raw
                    if raw == "xhigh":
                        return "xhigh" if is_grok_46 else "high"
                    return None
                if "grok-4.20-0309-reasoning" in model_key_l:
                    return raw if raw in ("low", "medium", "high") else "high"
                if raw in ("none", "low", "medium", "high", "xhigh"):
                    return "xhigh" if raw == "xhigh" and is_grok_46 else ("high" if raw == "xhigh" else raw)
                lvl = (options.get('thinking_level') or "low").lower()
                return "high" if lvl == "high" else "low"

            def _deepseek_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if model_key_l in {"deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}:
                    if raw in ("low", "high", "max"):
                        return raw
                    # DeepSeek maps compatibility values medium/xhigh to high.
                    return "high"
                if raw in ("max", "xhigh"):
                    return "max"
                return "high"

            def _kimi_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if raw in ("low", "high", "max"):
                    return raw
                return "max"

            def _grok_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                if not _auto_notice_enabled("grok_search"):
                    return base_prompt
                notice = _auto_notice_text("grok_search")
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice

            def _openai_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                if not _auto_notice_enabled("openai_search"):
                    return base_prompt
                notice = _auto_notice_text("openai_search")
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice
            
            def get_k(db_val, env_key):
                k = decrypt_val(db_val)
                if k and str(k).strip():
                    return k
                if _admin_env_fallback_enabled(user):
                    return os.getenv(env_key)
                return None

            resolved_auth = _resolve_chat_model_auth(user, model_key)
            gemini_runtime = resolved_auth.get("gemini_runtime") or _resolve_gemini_runtime(user)
            model_api_key_override = _get_model_specific_api_key(user, model_key)

            api_keys = {
                'openai': get_k(user.openai_api_key, 'OPENAI_API_KEY'),
                'gemini': gemini_runtime.get('api_key'),
                'anthropic': get_k(user.anthropic_api_key, 'ANTHROPIC_API_KEY'),
                'xai': get_k(user.xai_api_key, 'XAI_API_KEY'),
                'deepseek': get_k(user.deepseek_api_key, 'DEEPSEEK_API_KEY'),
                'kimi': get_k(user.kimi_api_key, 'MOONSHOT_API_KEY'),
                'mistral': get_k(user.mistral_api_key, 'MISTRAL_API_KEY'),
            }

            key = resolved_auth.get("api_key")
            if resolved_auth.get("error_code"):
                pub(
                    "error",
                    resolved_auth.get("error") or "APIキーが設定されていません。",
                    code=resolved_auth["error_code"],
                    model=model_key,
                    provider=resolved_auth.get("provider"),
                )
                return

            g_client = None; o_client = None; x_client = None; c_client = None
            gemini_backend_mode = _normalize_gemini_backend(gemini_runtime.get("backend")) if is_gem else "gemini_api"
            if is_gem:
                try:
                    g_client = _get_gemini_client(
                        api_key=key,
                        backend=gemini_backend_mode,
                        vertex_project=gemini_runtime.get("vertex_project"),
                        vertex_location=gemini_runtime.get("vertex_location"),
                        vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
                    )
                except Exception as e:
                    pub("error", _format_gemini_runtime_error(e, gemini_backend_mode))
                    return
                if not g_client:
                    if gemini_backend_mode == "vertex_ai":
                        pub("error", _gemini_vertex_auth_error_message())
                    else:
                        pub("error", "Gemini client initialization failed. Gemini設定を確認してください。")
                    return
            elif is_claude:
                if not ANTHROPIC_AVAILABLE:
                    pub("error", "Anthropic SDK is not installed on the server.")
                    return
                c_client = Anthropic(api_key=key)
            elif is_grok:
                x_client = _get_xai_client(key)
                o_client = _get_openai_client(key, base_url=f"https://{_XAI_API_HOST}/v1")
            elif is_deepseek:
                o_client = _get_openai_client(key, base_url="https://api.deepseek.com")
            elif is_kimi:
                o_client = _get_openai_client(key, base_url="https://api.moonshot.ai/v1")
            elif is_mistral_ocr:
                o_client = None
            else: o_client = _get_openai_client(key, base_url=None)

            loaded_files = []
            file_errors = []
            cache_updated = False
            total_loaded_bytes = 0
            attachment_name_map = {}
            raw_attachment_name_map = options.get("attachment_name_map") or {}
            if isinstance(raw_attachment_name_map, dict):
                for raw_path, raw_name in raw_attachment_name_map.items():
                    norm_path = _normalize_upload_ref(raw_path)
                    if not norm_path or not norm_path.startswith(f"{user_id}/"):
                        continue
                    norm_name = _normalize_display_name_for_path(norm_path, raw_name)
                    if norm_name:
                        attachment_name_map[norm_path] = norm_name
            label_name_map = _get_user_file_label_map(user_id) if img_list else {}

            def _resolve_send_name(rel_path, mime):
                norm_rel = _normalize_upload_ref(rel_path)
                base_name = os.path.basename(norm_rel or "") or "file"
                explicit = attachment_name_map.get(norm_rel) if norm_rel else None
                if not explicit and norm_rel:
                    explicit = label_name_map.get(norm_rel)
                if explicit and norm_rel:
                    fixed = _normalize_display_name_for_path(norm_rel, explicit)
                    if fixed:
                        return fixed
                if norm_rel:
                    fixed = _normalize_display_name_for_path(norm_rel, base_name)
                    if fixed:
                        return fixed
                return _sanitize_file_display_name(base_name) or "file"
            if img_list:
                try:
                    max_single_mb = int(os.getenv("ATTACHMENT_MAX_MB", str(_upload_max_mb)) or _upload_max_mb)
                except Exception:
                    max_single_mb = _upload_max_mb
                max_single_bytes = max_single_mb * 1024 * 1024 if max_single_mb else 0
                max_total_bytes = 0
                try:
                    max_total_mb = os.getenv("ATTACHMENT_TOTAL_MAX_MB")
                    if max_total_mb and str(max_total_mb).strip():
                        max_total_bytes = int(max_total_mb) * 1024 * 1024
                except Exception:
                    max_total_bytes = 0

                for fn in img_list:
                    clean_fn = _normalize_upload_ref(fn)
                    if not clean_fn:
                        file_errors.append({"name": str(fn)[:80], "reason": "無効な参照"})
                        continue
                    if not clean_fn.startswith(f"{user_id}/"):
                        file_errors.append({"name": clean_fn, "reason": "権限外のパス"})
                        continue
                    info = _get_file_disk_info(clean_fn)
                    if not info.get("exists"):
                        file_errors.append({"name": clean_fn, "reason": "見つかりません"})
                        continue
                    if max_single_bytes and info.get("size") and info["size"] > max_single_bytes:
                        size_mb = info["size"] // (1024 * 1024)
                        file_errors.append({"name": clean_fn, "reason": f"サイズ超過({size_mb}MB)"})
                        continue
                    data = _load_user_file_bytes(clean_fn, info)
                    if data is None:
                        file_errors.append({"name": clean_fn, "reason": "読み込み失敗"})
                        continue
                    if len(data) == 0:
                        file_errors.append({"name": clean_fn, "reason": "空ファイル"})
                        continue
                    if max_total_bytes:
                        total_loaded_bytes += len(data)
                        if total_loaded_bytes > max_total_bytes:
                            file_errors.append({"name": clean_fn, "reason": "合計サイズ超過"})
                            break

                    is_pdf = clean_fn.lower().endswith('.pdf')
                    is_docx = clean_fn.lower().endswith('.docx')
                    is_pptx = clean_fn.lower().endswith('.pptx')
                    is_xlsx = clean_fn.lower().endswith(('.xlsx', '.xlsm'))
                    mime_guess = mimetypes.guess_type(clean_fn)[0]
                    mime = _normalize_media_mime(clean_fn, mime_guess)
                    clean_ext = os.path.splitext(clean_fn)[1].lower()
                    is_text = (mime or '').startswith('text/') or clean_ext in _TEXT_LIKE_UPLOAD_EXTS
                    if is_pdf:
                        mime = 'application/pdf'
                    elif is_docx:
                        mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    elif is_pptx:
                        mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    elif is_text:
                        mime = 'text/plain'
                    send_name = _resolve_send_name(clean_fn, mime) if is_llm_model else None

                    if is_pdf:
                        extracted = None if is_mistral_ocr else _extract_text_from_pdf(data)
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': extracted if extracted else None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': True,
                            'is_docx': False,
                            'is_pptx': False,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    elif is_docx:
                        # Render paragraphs with [N] numbers so the model can
                        # reference them with edit_file's paragraph_edits.
                        extracted = None if is_mistral_ocr else _extract_docx_as_numbered(data)
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': extracted if extracted else None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_docx': True,
                            'is_pptx': False,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    elif is_pptx:
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_docx': False,
                            'is_pptx': True,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    elif is_text:
                        extracted = _decode_text_bytes(data)
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': extracted if extracted else None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_text': True,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    elif is_xlsx:
                        # Spreadsheets are binary, so the model cannot read them
                        # as a native file part.  Render the cells as TSV text
                        # (with a column-letter header row) so the model can see
                        # the content and derive cell addresses for edit_file.
                        xlsx_text = _extract_xlsx_as_tsv(data, include_column_headers=True)
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': xlsx_text,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_xlsx': True,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    else:
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    try:
                        _upsert_file_cache(
                            user_id,
                            clean_fn,
                            "local",
                            size_bytes=len(data),
                            mtime=info.get("mtime"),
                            mime_type=mime,
                            state="loaded",
                            last_error=None
                        )
                        cache_updated = True
                    except Exception:
                        pass

                if cache_updated:
                    try:
                        safe_db_commit()
                    except Exception:
                        pass

            if file_errors:
                parts = []
                for e in file_errors[:5]:
                    nm = e.get("name") or "file"
                    rs = e.get("reason") or "error"
                    parts.append(f"{nm}({rs})")
                if len(file_errors) > 5:
                    parts.append(f"...他{len(file_errors) - 5}件")
                pub("error", "添付ファイルの検証に失敗しました: " + " / ".join(parts))
                return

            if img_list and not loaded_files:
                pub("error", "添付ファイルを読み込めませんでした。再アップロードしてから再送してください。")
                return

            has_audio = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('audio/') for fi in loaded_files)
            has_video = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('video/') for fi in loaded_files)
            has_code_exec_unsupported_doc = any(
                fi.get('bytes') and (
                    fi.get('is_pdf')
                    or fi.get('is_docx')
                    or str(fi.get('mime', '')).lower() == 'application/pdf'
                    or str(fi.get('mime', '')).lower() == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
                for fi in loaded_files
            )
            gemini_local_python = False
            if is_gem and (has_audio or has_video or has_code_exec_unsupported_doc) and options.get('enable_python'):
                # Gemini code_execution does not accept audio/video/PDF/DOCX inputs; fall back to local exec.
                gemini_local_python = True
                log_force("Gemini: local python mode for audio/video/document inputs")
                if _auto_notice_enabled("gemini_local_python"):
                    local_py_notice = _auto_notice_text("gemini_local_python")
                    curr_p = options.get('system_prompt') or ""
                    if local_py_notice not in str(curr_p):
                        options['system_prompt'] = f"{local_py_notice}\n\n{curr_p}" if str(curr_p).strip() else local_py_notice
            if (has_audio and not supports_audio_inputs) or (has_video and not supports_video_inputs):
                pub("error", "This model does not support audio/video inputs. Please remove them and retry.")
                return

            full_res, thought_accumulated, generated_images = "", "", []
            deepseek_tool_context = []
            agentic_image_digests = set()
            agentic_saved_urls = []
            agentic_consumed_urls = []
            sandbox_text_buffer = [""]
            agentic_filename_url_map = {}
            pending_sandbox_filenames = []
            signature_parts = []

            original_quote_text = quote_text
            if options.get("coding_mode"):
                coding_target = options.get("coding_target") or {}
                coding_candidates = options.get("coding_candidates") or []
                target_code = str(coding_target.get("code") or "")
                target_language = str(coding_target.get("language") or "text")
                explicitly_selected = (
                    coding_target.get("explicit") is True
                    and len(coding_candidates) == 1
                    and str(coding_candidates[0].get("id") or "") == str(coding_target.get("id") or "")
                )
                if explicitly_selected:
                    coding_context = (
                        f"[Coding Mode Target]\nYou must edit candidate {coding_target.get('id')}.\n"
                        f"Language: {target_language}\nCharacter count: {len(target_code)}\n"
                        f"--- BEGIN TARGET CODE ---\n{target_code}\n--- END TARGET CODE ---"
                    )
                else:
                    candidate_lines = []
                    for candidate in coding_candidates:
                        candidate_code = str(candidate.get("code") or "")
                        preview = next(
                            (line.strip() for line in candidate_code.splitlines() if line.strip()),
                            "",
                        )[:160]
                        candidate_lines.append(
                            f"- id={candidate.get('id')}; source={candidate.get('source')}; "
                            f"language={candidate.get('language')}; chars={len(candidate_code)}; "
                            f"first_line={preview!r}"
                        )
                    coding_context = (
                        "[Coding Mode Candidates]\n"
                        "Choose the candidate that best matches User Message and return its id as target_id. "
                        "Prompt candidates correspond to Markdown code blocks in User Message; history candidates "
                        "correspond to code blocks in the conversation in chronological order.\n"
                        + "\n".join(candidate_lines)
                    )
                quote_text = f"{quote_text}\n\n{coding_context}" if quote_text else coding_context
            final_message_text = message_text
            if quote_text:
                final_message_text = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"

            auto_enable_search = options.get('enable_search')
            auto_enable_url_context = options.get('enable_url_context')
            is_gemini_3 = "gemini-3" in model_key or "gemini-3.1" in model_key
            auto_enable_maps = bool(options.get('enable_maps')) and is_gemini_3
            grok_enable_search = auto_enable_search
            user_auto_search = True
            try:
                user_auto_search = bool(getattr(user, "auto_search_on_links", True))
            except Exception:
                user_auto_search = True
            disable_auto = bool(options.get('disable_auto_search'))
            if is_mistral_ocr:
                grok_enable_search = False
                auto_enable_search = False
                auto_enable_url_context = False
                auto_enable_maps = False
            if is_grok and not is_mistral_ocr and not grok_enable_search and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        grok_enable_search = True
                        auto_enable_search = True
                        log_force("Auto-enabled Grok search for URL/X post access")
                except Exception:
                    pass
            if not is_mistral_ocr and not is_grok and not auto_enable_search and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        auto_enable_search = True
                        log_force("Auto-enabled Web search for URL/X post access")
                except Exception:
                    pass
            if is_gem and not is_mistral_ocr and not auto_enable_url_context and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {original_quote_text or ''}"
                    if re.search(r'https?://', check_text):
                        auto_enable_url_context = True
                        log_force("Auto-enabled URL context for Gemini URL access")
                except Exception:
                    pass

            # ---- MCP（外部モデル連携）実行環境の遅延構築ヘルパ ----
            # テキストLLM分岐（Gemini / Claude / DeepSeek / OpenAI系Responses）が
            # 必要になった時点で tools/list を取得し、モデルへ付与する。
            # 案内文はシステムプロンプトへ先に入れ、各プロバイダ分岐が
            # system / system_instruction を組む前にモデルが MCP の存在を知る。
            _mcp_env = None
            _mcp_prompt_injected = False
            # プロンプトバーのMCPスイッチ（enable_mcp）が OFF の間は、
            # MCPツールの付与と案内文注入の両方を無効化する。
            # 値が無い（None）従来リクエストは従来どおり有効扱いにする。
            _mcp_request_disabled = options.get('enable_mcp') is False

            def _ensure_mcp_env():
                nonlocal _mcp_env, _mcp_prompt_injected
                if _mcp_env is not None:
                    return _mcp_env
                if _mcp_request_disabled:
                    return None
                from mcp_service.execution import McpRuntime
                _env = McpRuntime(
                    user_id,
                    job_id=job_id,
                    pub=pub,
                    check_stop=check_stop,
                    log=log_force,
                )
                try:
                    _env.load()
                except Exception as _mcp_exc:
                    log_force(f"MCP load failed: {_mcp_exc}")
                _mcp_env = _env if not _env.empty() else None
                if _mcp_env is not None:
                    log_force(f"MCP enabled: {len(_mcp_env.tool_metas())} tools across {len(_mcp_env.servers)} servers")
                    if not _mcp_prompt_injected:
                        try:
                            # 案内文は「自動注入システムプロンプト（ユーザー単位）」の
                            # mcp 項目の文面に従う。項目のオン・オフはプロンプトバーの
                            # MCPスイッチに連動しており（ここまで到達 = 有効）、
                            # ここでは全体適用トグルと文面だけを確認する。
                            if _auto_notice_enabled("mcp"):
                                _mcp_note = _mcp_env.guidance_text(preamble=_auto_notice_text("mcp"))
                                if _mcp_note:
                                    _cur_sys = options.get('system_prompt') or ""
                                    if "Model Context Protocol" not in str(_cur_sys):
                                        options['system_prompt'] = (
                                            f"{_mcp_note}\n\n{_cur_sys}".strip() if str(_cur_sys).strip() else _mcp_note
                                        )
                            _mcp_prompt_injected = True
                        except Exception as _mcp_note_exc:
                            log_force(f"MCP system prompt inject failed: {_mcp_note_exc}")
                return _mcp_env

            if is_llm_model and not gemini_local_python:
                try:
                    _ensure_mcp_env()
                except Exception as _mcp_preload_exc:
                    log_force(f"MCP preload failed: {_mcp_preload_exc}")

            def _mcp_summary_md(meta_out, internal_name):
                """MCP実行結果をメッセージ本文へ追記するMarkdownを作る。"""
                try:
                    if meta_out.get("ok") and meta_out.get("id"):
                        return ""
                    if meta_out.get("rejected"):
                        return f"\n\n> 🚫 **MCPツール実行（ユーザーが拒否）:** `{internal_name}`\n"
                    if meta_out.get("is_error"):
                        return ""
                    return ""
                except Exception:
                    return ""

            # --- 0. Mistral OCR 4 (document-only, no chat history) ---
            if is_mistral_ocr:
                log_force("Routing: Mistral OCR Branch")
                uploaded_file_ids = []
                try:
                    pub("status", "OCR対象の文書を準備中...")
                    table_format = str(options.get("ocr_table_format") or "").strip().lower()
                    if table_format not in ("markdown", "html"):
                        table_format = None
                    extract_header = bool(options.get("ocr_extract_header"))
                    extract_footer = bool(options.get("ocr_extract_footer"))
                    include_blocks = bool(options.get("ocr_include_blocks"))
                    include_image_base64 = options.get("ocr_include_image_base64")
                    if include_image_base64 is None:
                        include_image_base64 = True
                    else:
                        include_image_base64 = bool(include_image_base64)
                    pages_opt = str(options.get("ocr_pages") or "").strip() or None
                    extra = {
                        "include_image_base64": include_image_base64,
                        "include_blocks": include_blocks,
                    }
                    if table_format:
                        extra["table_format"] = table_format
                    if extract_header:
                        extra["extract_header"] = True
                    if extract_footer:
                        extra["extract_footer"] = True
                    if pages_opt:
                        extra["pages"] = pages_opt

                    jobs = []
                    for fi in loaded_files:
                        data = fi.get("bytes")
                        if not data:
                            continue
                        path = str(fi.get("path") or fi.get("name") or "")
                        ext = os.path.splitext(path)[1].lower()
                        mime = str(fi.get("mime") or "")
                        label = fi.get("send_name") or os.path.basename(path) or "document"
                        if ext in MISTRAL_OCR_IMAGE_EXTS or mime.startswith("image/"):
                            jobs.append({
                                "label": label,
                                "document": {
                                    "type": "image_url",
                                    "image_url": _mistral_data_uri(mime or "image/jpeg", data),
                                },
                            })
                        elif (
                            ext in MISTRAL_OCR_DOC_EXTS
                            or fi.get("is_pdf")
                            or fi.get("is_docx")
                            or fi.get("is_pptx")
                        ):
                            file_id = _mistral_upload_ocr_file(key, label, data, mime)
                            uploaded_file_ids.append(file_id)
                            jobs.append({
                                "label": label,
                                "document": {"type": "file", "file_id": file_id},
                            })
                        else:
                            pub("error", f"Mistral OCR は {label} の形式に対応していません。PDF / 画像 / DOCX / PPTX を添付してください。")
                            return

                    for url in _extract_mistral_ocr_urls(final_message_text):
                        kind = _mistral_guess_url_kind(url)
                        if kind == "image":
                            jobs.append({
                                "label": url,
                                "document": {"type": "image_url", "image_url": url},
                            })
                        else:
                            jobs.append({
                                "label": url,
                                "document": {"type": "document_url", "document_url": url},
                            })

                    if not jobs:
                        pub("error", "Mistral OCR は文書専用です。PDF・画像・DOCX・PPTX を添付するか、公開URLを入力してください。会話履歴は送信されません。")
                        return

                    assembled = []
                    pages_total = 0
                    for idx, job in enumerate(jobs, start=1):
                        pub("status", f"OCR処理中です（{idx}/{len(jobs)}）...")
                        _mark_provider_request_started()
                        ocr_json = _mistral_ocr_process_document(key, job["document"], extra)
                        usage = (ocr_json or {}).get("usage_info") or {}
                        try:
                            pages_total += int(usage.get("pages_processed") or 0)
                        except Exception:
                            pass
                        image_url_by_id = {}
                        if include_image_base64:
                            for page in (ocr_json or {}).get("pages") or []:
                                for img in (page or {}).get("images") or []:
                                    if not isinstance(img, dict):
                                        continue
                                    img_id = str(img.get("id") or img.get("image_id") or "").strip()
                                    raw_b64 = img.get("image_base64") or img.get("imageBase64")
                                    img_bytes, img_mime = _decode_mistral_image_base64(raw_b64)
                                    if not img_bytes:
                                        continue
                                    ext = _mistral_ext_from_mime(img_mime, "jpg")
                                    fn2 = f"ocr_{int(time.time())}_{os.urandom(3).hex()}.{ext}"
                                    _save_user_generated_bytes(
                                        user_id, img_bytes, fn2, user_config.get("enable_e2ee")
                                    )
                                    rel = f"{user_id}/{fn2}"
                                    generated_images.append(rel)
                                    local_url = f"/files/{rel}"
                                    if img_id:
                                        image_url_by_id[img_id] = local_url
                        body = _build_mistral_ocr_markdown(
                            ocr_json, image_url_by_id, include_blocks=include_blocks
                        )
                        heading = f"## {job['label']}" if len(jobs) > 1 else ""
                        chunk = f"{heading}\n\n{body}".strip()
                        assembled.append(chunk)
                        pub("content", (chunk if not full_res else "\n\n" + chunk))
                        full_res += (("\n\n" if full_res else "") + chunk)

                    if pages_total:
                        footer = f"\n\n— 合計処理ページ数: {pages_total}"
                        full_res += footer
                        pub("content", footer)
                except Exception as e:
                    logger.exception("Mistral OCR Error")
                    pub("error", f"Mistral OCR Error: {str(e)}")
                    return
                finally:
                    for file_id in uploaded_file_ids:
                        _mistral_delete_file(key, file_id)

            # --- 1A-0. GEMINI OMNI 1.1 FLASH (Interactions API) ---
            elif model_key_l == "gemini-omni-1.1-flash":
                log_force("Routing: Gemini Omni 1.1 Flash (Interactions API)")
                try:
                    if gemini_backend_mode != "gemini_api":
                        pub("error", "Gemini Omni 1.1 Flash は Gemini API バックエンドでのみ利用できます（Vertex AI バックエンドでは利用不可）。")
                    else:
                        pub("content", "**Generating Video (Gemini Omni 1.1 Flash)...**\n")
                        aspect_ratio = str(options.get('gemini_video_aspect') or "16:9")
                        if aspect_ratio not in ("16:9", "9:16"):
                            aspect_ratio = "16:9"
                        resolution = str(options.get('gemini_video_resolution') or "720p")
                        if resolution == "4K":
                            resolution = "4k"
                        elif resolution not in ("360p", "720p", "1080p", "4k"):
                            resolution = "720p"

                        # Build Interactions API input parts: attached images + text prompt.
                        # Omni 1.1 accepts text / image / video / audio inputs via the Interactions API.
                        interaction_input = []
                        for fi in loaded_files:
                            if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                                interaction_input.append({
                                    "type": "image",
                                    "data": base64.b64encode(fi['bytes']).decode('ascii'),
                                    "mime_type": str(fi.get('mime') or 'image/jpeg'),
                                })
                        interaction_input.append({"type": "text", "text": final_message_text})

                        # Synchronous unary generation (background=false / store=false / stream=false).
                        # Use URI delivery so videos larger than 4MB are not clipped by payload limits.
                        _mark_provider_request_started()
                        interaction = g_client.interactions.create(
                            model=model_key,
                            input=interaction_input,
                            response_format={
                                "type": "video",
                                "aspect_ratio": aspect_ratio,
                                "resolution": resolution,
                                "delivery": "uri",
                            },
                            background=False,
                            store=False,
                            stream=False,
                            timeout=600.0,
                        )
                        if getattr(interaction, "status", None) == "failed":
                            raise RuntimeError("Omni 1.1 video generation failed.")
                        video_data = None
                        video_uri = None
                        for out in (getattr(interaction, "outputs", None) or []):
                            if getattr(out, "type", None) == "video":
                                video_data = getattr(out, "data", None)
                                video_uri = getattr(out, "uri", None)
                                break
                        if video_data:
                            video_bytes = base64.b64decode(video_data)
                        elif video_uri:
                            file_id_match = re.search(r'files/([A-Za-z0-9_-]+)', video_uri)
                            if file_id_match:
                                for _i in range(15):
                                    finfo = g_client.files.get(name=f"files/{file_id_match.group(1)}")
                                    if str(getattr(getattr(finfo, "state", None), "name", "")) == "ACTIVE":
                                        break
                                    time.sleep(2)
                            _mark_provider_request_started()
                            video_bytes = g_client.files.download(file=video_uri)
                        else:
                            raise RuntimeError("No video output returned from Omni 1.1.")
                        if not video_bytes:
                            raise RuntimeError("Video download failed.")
                        fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
                        _save_user_generated_bytes(user_id, video_bytes, fn2, user_config.get('enable_e2ee'))
                        video_tag = f'\n<video controls src="/files/{user_id}/{fn2}" class="w-full max-w-2xl rounded-lg"></video>\n'
                        pub("content", video_tag)
                        full_res += f"Generated Video for: {final_message_text}\n"
                        generated_images.append(f"{user_id}/{fn2}")
                except Exception as e:
                    logger.exception("Gemini Video Gen Error")
                    pub("error", f"Gemini Video Gen Error: {str(e)}")

            # --- 1A. GEMINI VIDEO GENERATION (Veo 3.1 / Omni Flash) ---
            elif is_gemini_video_model_key(model_key_l):
                log_force("Routing: Gemini Video Branch")
                try:
                    pub("content", "**Generating Video (Gemini)...**\n")
                    try:
                        video_duration = int(options.get('gemini_video_duration') or 8)
                    except Exception:
                        video_duration = 8
                    if video_duration < 1:
                        video_duration = 1
                    if video_duration > 8:
                        video_duration = 8
                    aspect_ratio = str(options.get('gemini_video_aspect') or "16:9")
                    if aspect_ratio not in ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"):
                        aspect_ratio = "16:9"
                    resolution = str(options.get('gemini_video_resolution') or "720p")
                    if resolution not in ("480p", "720p", "1080p", "4K"):
                        resolution = "720p"
                    if resolution == "4K" and model_key != "veo-3.1-generate-preview":
                        resolution = "1080p"

                    # Image-to-video support: use the first attached image if present
                    src_image = None
                    for fi in loaded_files:
                        if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                            src_image = types.Image(imageBytes=fi['bytes'], mimeType=fi['mime'])
                            break
                    video_source = types.GenerateVideosSource(
                        prompt=final_message_text,
                        image=src_image,
                    )

                    _mark_provider_request_started()
                    op = g_client.models.generate_videos(
                        model=model_key,
                        source=video_source,
                        config=types.GenerateVideosConfig(
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                            duration_seconds=video_duration,
                            number_of_videos=1,
                        ),
                    )
                    pub("content", "生成中です。数分かかる場合があります...\n")
                    max_polls = 120
                    done_op = None
                    for i in range(max_polls):
                        if check_stop():
                            break
                        time.sleep(5)
                        op = g_client.operations.get(op)
                        if op.done:
                            done_op = op
                            break
                    if done_op is None:
                        raise RuntimeError("Video generation timed out.")
                    if getattr(done_op, "error", None):
                        raise RuntimeError(str(done_op.error))
                    gen_videos = (done_op.result.generated_videos or []) if done_op.result else []
                    if not gen_videos:
                        raise RuntimeError("No video output returned.")
                    video_uri = gen_videos[0].video.uri if gen_videos[0].video else None
                    if not video_uri:
                        raise RuntimeError("No video URI in response.")
                    _mark_provider_request_started()
                    video_bytes = _download_public_https_bytes(video_uri, 256 * 1024 * 1024, timeout=180.0)
                    if not video_bytes:
                        raise RuntimeError("Video download failed.")
                    fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
                    _save_user_generated_bytes(user_id, video_bytes, fn2, user_config.get('enable_e2ee'))
                    video_tag = f'\n<video controls src="/files/{user_id}/{fn2}" class="w-full max-w-2xl rounded-lg"></video>\n'
                    pub("content", video_tag)
                    full_res += f"Generated Video for: {final_message_text}\n"
                    generated_images.append(f"{user_id}/{fn2}")
                except Exception as e:
                    logger.exception("Gemini Video Gen Error")
                    pub("error", f"Gemini Video Gen Error: {str(e)}")

            # --- 1B. GEMINI MUSIC GENERATION (Lyria 3) ---
            elif is_gemini_music_model_key(model_key_l):
                log_force("Routing: Gemini Music Branch")
                try:
                    if model_key == "lyria-realtime-exp":
                        pub("error", "Lyria RealTimeはWebSocketによるリアルタイム生成専用の実験モデルです。プロンプトを入力して送信すると、Lyria RealTime Studioが開きます。")
                    else:
                        pub("content", "**Generating Music (Lyria)...**\n")
                        music_prompt = final_message_text
                        if options.get('music_instrumental'):
                            music_prompt = f"{music_prompt} Instrumental only, no vocals."
                        music_parts = [types.Part(text=music_prompt)]
                        for fi in loaded_files:
                            if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/'):
                                music_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                                if len(music_parts) >= 11:
                                    break
                        music_cfg = types.GenerateContentConfig(response_modalities=["AUDIO", "TEXT"])
                        _mark_provider_request_started()
                        m_resp = g_client.models.generate_content(
                            model=model_key,
                            contents=music_parts,
                            config=music_cfg,
                        )
                        lyrics = []
                        audio_data = None
                        audio_mime = "audio/mpeg"
                        cand0 = m_resp.candidates[0] if m_resp.candidates else None
                        for part in (getattr(getattr(cand0, "content", None), "parts", None) or []):
                            if part.text:
                                lyrics.append(part.text)
                            elif getattr(part, 'inline_data', None) and part.inline_data:
                                audio_data = part.inline_data.data
                                audio_mime = part.inline_data.mime_type or "audio/mpeg"
                        if lyrics:
                            lyrics_text = "\n".join(lyrics)
                            pub("content", lyrics_text + "\n")
                            full_res += lyrics_text + "\n"
                        if audio_data:
                            if isinstance(audio_data, str):
                                audio_data = base64.b64decode(audio_data)
                            ext = ".wav" if "wav" in str(audio_mime).lower() else ".mp3"
                            fn2 = f"music_{int(time.time())}_{os.urandom(4).hex()}{ext}"
                            _save_user_generated_bytes(user_id, bytes(audio_data), fn2, user_config.get('enable_e2ee'))
                            audio_tag = f'\n<audio controls src="/files/{user_id}/{fn2}" class="w-full mt-2"></audio>\n'
                            pub("content", audio_tag)
                            full_res += f"Generated Music for: {final_message_text}\n"
                            generated_images.append(f"{user_id}/{fn2}")
                        if not audio_data and not lyrics:
                            pub("error", "Lyria returned no output.")
                except Exception as e:
                    logger.exception("Gemini Music Gen Error")
                    pub("error", f"Gemini Music Gen Error: {str(e)}")

            # --- 1C. GEMINI EMBEDDING ---
            elif is_gemini_embedding_model_key(model_key_l):
                log_force("Routing: Gemini Embedding Branch")
                try:
                    _mark_provider_request_started()
                    emb = g_client.models.embed_content(
                        model=model_key,
                        contents=final_message_text,
                    )
                    emb_list = emb.embeddings if getattr(emb, "embeddings", None) else None
                    values = emb_list[0].values if emb_list else None
                    if values is None:
                        pub("error", "Gemini Embedding returned no values.")
                    else:
                        dims = len(values)
                        preview = ", ".join(f"{float(v):.6f}" for v in values[:12])
                        out = (
                            f"**Gemini Embedding 2**\n\n"
                            f"- 次元数: **{dims}**\n"
                            f"- 先頭12次元: `[{preview}{', ...' if dims > 12 else ''}]`\n\n"
                            f"*入力テキストの埋め込みベクトルを生成しました。*"
                        )
                        pub("content", out)
                        full_res += out
                except Exception as e:
                    logger.exception("Gemini Embedding Error")
                    pub("error", f"Gemini Embedding Error: {str(e)}")

            # --- 1. GEMINI & GEMINI IMAGE ---
            elif is_gem:
                log_force("Routing: Gemini Branch")
                gemini_files_api_enabled = (gemini_backend_mode != "vertex_ai")

                # Gemini Transcribe (audio file -> text, Interactions API)
                if is_gemini_transcribe_model_key(model_key):
                    try:
                        _mark_provider_request_started()
                        if gemini_backend_mode == "vertex_ai":
                            pub("error", "Gemini 3.5 Transcribe: 現在は Gemini API モード（Vertex AI 以外）でのみ利用できます。")
                            return
                        audio_fi = next(
                            (
                                fi for fi in loaded_files
                                if fi.get('bytes') and str(fi.get('mime', '')).startswith('audio/')
                            ),
                            None
                        )
                        if not audio_fi:
                            pub("error", "Gemini 3.5 Transcribe: 文字起こしには音声ファイル（MP3/WAV/M4A/OGG/FLAC等）を添付してください。")
                            return
                        audio_data = audio_fi.get('bytes')
                        audio_mime = audio_fi.get('mime') or "audio/mpeg"
                        audio_name = audio_fi.get('name') or "audio.mp3"

                        # Normalize WebM/OGG/Opus to 16kHz WAV (Gemini inline audio + model expectations)
                        m_low = (audio_mime or '').lower()
                        ext_low = (os.path.splitext(audio_name or '')[1] or '').lower()
                        if m_low in ("audio/webm", "audio/ogg", "audio/oga", "audio/opus") or ext_low in (".webm", ".ogg", ".oga", ".opus"):
                            try:
                                src_suffix = ext_low if ext_low else ".webm"
                                pcm = _convert_audio_to_pcm(audio_data, src_suffix=src_suffix, rate=16000)
                                audio_data = _pcm_to_wav_bytes(pcm, rate=16000)
                                audio_mime = "audio/wav"
                                audio_name = os.path.splitext(audio_name or "audio")[0] + ".wav"
                            except Exception as conv_e:
                                logger.exception("Gemini Transcribe audio conversion failed")
                                pub("error", f"Gemini 3.5 Transcribe: 音声の変換に失敗しました: {str(conv_e)}")
                                return

                        # Upload via Gemini Files API (documented path for transcribe input)
                        file_uri = None
                        try:
                            if not gemini_files_api_enabled:
                                pub("error", "Gemini 3.5 Transcribe: Vertex AI モードではファイルアップロードを利用できません。Gemini API モードへ切り替えてください。")
                                return
                            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_name)[1] or ".wav") as tmp:
                                tmp.write(audio_data)
                                tmp.flush()
                                up = g_client.files.upload(file=tmp.name, config={"mimeType": audio_mime})
                            up_name = getattr(up, "name", None) or (up.get("name") if isinstance(up, dict) else None)
                            deadline = time.time() + 120
                            while time.time() < deadline:
                                if isinstance(up, dict):
                                    st = up.get("state")
                                else:
                                    st = getattr(up, "state", None)
                                if isinstance(st, dict):
                                    st = st.get("name") or st.get("state")
                                else:
                                    st = getattr(st, "name", None) or st
                                if not st or st == "ACTIVE" or st == "FAILED":
                                    break
                                time.sleep(2)
                                try:
                                    if up_name:
                                        up = g_client.files.get(name=up_name)
                                except Exception:
                                    break
                            if isinstance(up, dict):
                                file_uri = up.get("uri") or up.get("file_uri") or up.get("fileUri") or up.get("name")
                            else:
                                file_uri = (
                                    getattr(up, "uri", None)
                                    or getattr(up, "file_uri", None)
                                    or getattr(up, "fileUri", None)
                                    or getattr(up, "name", None)
                                )
                        except Exception as up_e:
                            logger.exception("Gemini Transcribe file upload failed")
                            pub("error", f"Gemini 3.5 Transcribe: 音声のアップロードに失敗しました: {str(up_e)}")
                            return
                        if not file_uri:
                            pub("error", "Gemini 3.5 Transcribe: 音声のアップロードに失敗しました（ファイルURIを取得できません）。")
                            return

                        transcription_config = {}
                        try:
                            lang_codes = options.get('transcription_language_codes') or []
                            if isinstance(lang_codes, (list, tuple)) and lang_codes:
                                transcription_config['language_codes'] = [str(x) for x in lang_codes][:20]
                            else:
                                transcription_config['language_codes'] = []
                            custom_vocab = options.get('transcription_custom_vocabulary') or []
                            if isinstance(custom_vocab, (list, tuple)) and custom_vocab:
                                transcription_config['custom_vocabulary'] = [str(x) for x in custom_vocab][:1000]
                            t_mode = str(options.get('transcription_mode') or 'verbatim').lower()
                            if t_mode == "smart":
                                transcription_config['mode'] = {"type": "smart"}
                            else:
                                verbatim_mode = {"type": "verbatim"}
                                if options.get('transcription_diarization'):
                                    verbatim_mode['diarization_mode'] = "speaker"
                                if options.get('transcription_word_timestamps'):
                                    verbatim_mode['timestamp_granularities'] = ["word"]
                                transcription_config['mode'] = verbatim_mode
                        except Exception as cfg_e:
                            logger.warning(f"Gemini Transcribe config fallback: {cfg_e}")
                            transcription_config = {"language_codes": []}

                        pub("status", "文字起こしを実行中です。音声の長さによって時間がかかります...")
                        transcript = _gemini_transcribe_rest(
                            api_key=key,
                            file_uri=file_uri,
                            mime_type=audio_mime,
                            transcription_config=transcription_config,
                        )
                        if not transcript or not str(transcript).strip():
                            pub("error", "Gemini 3.5 Transcribe: 文字起こし結果が空でした。音声が無声または対応外形式の可能性があります。")
                            return
                        transcript = str(transcript).strip()
                        full_res += transcript
                        pub("content", transcript)
                        log_force(f"Gemini Transcribe completed for {job_id} chars={len(transcript)}")
                    except Exception as e:
                        logger.exception("Gemini Transcribe Error")
                        pub("error", f"Gemini 3.5 Transcribe Error: {str(e)}")

                # Gemini TTS (Preview)
                elif "tts" in model_key:
                    try:
                        voice_name = (options.get('tts_voice') or "Kore").strip()
                        if voice_name not in GEMINI_TTS_VOICES:
                            voice_name = "Kore"
                        tts_lang = (options.get('tts_language') or "").strip() or None
                        _mark_provider_request_started()
                        tts_resp = g_client.models.generate_content(
                            model=model_key,
                            contents=final_message_text,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=voice_name
                                        )
                                    ),
                                    language_code=tts_lang
                                ),
                            ),
                        )
                        audio_bytes = None
                        cand0 = tts_resp.candidates[0] if tts_resp.candidates else None
                        parts0 = getattr(getattr(cand0, "content", None), "parts", None) or []
                        if parts0:
                            p0 = parts0[0]
                            if hasattr(p0, 'inline_data') and p0.inline_data:
                                data = p0.inline_data.data
                                if isinstance(data, (bytes, bytearray)):
                                    audio_bytes = bytes(data)
                                elif isinstance(data, str):
                                    audio_bytes = base64.b64decode(data)

                        if not audio_bytes:
                            pub("error", "Gemini TTS Error: No audio data returned.")
                        else:
                            buf = BytesIO()
                            with wave.open(buf, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(24000)
                                wf.writeframes(audio_bytes)
                            wav_bytes = buf.getvalue()

                            speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.wav"
                            _save_user_generated_bytes(
                                user_id, wav_bytes, speech_file_name, user_config.get('enable_e2ee')
                            )

                            audio_url = f"/files/{user_id}/{speech_file_name}"
                            audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
                            full_res += audio_tag
                            pub("content", audio_tag)
                            generated_images.append(f"{user_id}/{speech_file_name}")
                    except Exception as e:
                        logger.exception("Gemini TTS Error")
                        pub("error", f"Gemini TTS Error: {str(e)}")

                # Image Generation
                elif is_gemini_image_model_key(model_key):
                    try:
                        def _collect_gemini_image_output_parts(resp_obj, keep_only_last_image=False):
                            text_chunks = []
                            image_parts = []
                            seen_part_ids = set()

                            def _append_parts(parts_seq):
                                for _part in parts_seq or []:
                                    # Gemini 3 image models can return intermediate thought
                                    # images/text. They are reasoning artifacts, not user-facing
                                    # output, and must not be saved as the final generated image.
                                    if bool(getattr(_part, "thought", False)):
                                        continue
                                    part_id = id(_part)
                                    if part_id in seen_part_ids:
                                        continue
                                    seen_part_ids.add(part_id)
                                    if hasattr(_part, 'text') and _part.text:
                                        txt = str(_part.text)
                                        if txt.strip():
                                            text_chunks.append(txt)
                                    if hasattr(_part, 'inline_data') and _part.inline_data:
                                        image_parts.append(_part)

                            _append_parts(getattr(resp_obj, 'parts', None) or [])
                            for cand in getattr(resp_obj, 'candidates', None) or []:
                                _append_parts(getattr(getattr(cand, 'content', None), 'parts', None) or [])

                            if keep_only_last_image and len(image_parts) > 1:
                                image_parts = [image_parts[-1]]
                            return text_chunks, image_parts

                        def _save_gemini_image_part(part_obj):
                            mime = getattr(part_obj.inline_data, "mime_type", None) or "image/png"
                            ext_map = {
                                "image/png": "png",
                                "image/jpeg": "jpg",
                                "image/webp": "webp"
                            }
                            ext = ext_map.get(mime, "png")
                            fn2 = f"gen_{int(time.time())}_{len(generated_images)}.{ext}"
                            img_data = part_obj.inline_data.data
                            if isinstance(img_data, str):
                                img_data = _decode_base64_limited(img_data, 50 * 1024 * 1024)
                            _save_user_generated_bytes(
                                user_id, img_data, fn2, user_config.get('enable_e2ee')
                            )
                            generated_images.append(f"{user_id}/{fn2}")
                            return fn2

                        # [FIX] Apply System Prompt to Image Prompts if available
                        img_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
                        if options.get('system_prompt'):
                            img_prompt = f"{options.get('system_prompt')}\n\n{img_prompt}"

                        mk_lower = str(model_key or "").lower()
                        if "gemini-3.1-flash-lite-image" in mk_lower:
                            img_model = "gemini-3.1-flash-lite-image"
                        elif "gemini-3.1-flash-image" in mk_lower:
                            img_model = "gemini-3.1-flash-image"
                        elif "2.5" in mk_lower:
                            img_model = "gemini-2.5-flash-image"
                        else:
                            img_model = "gemini-3-pro-image"
                        aspect_allowed = {
                            "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
                            "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
                            "auto"
                        }
                        size_allowed = {"1K", "2K", "4K"}
                        aspect_val = options.get('gemini_image_aspect')
                        if aspect_val:
                            aspect_val = str(aspect_val).strip()
                            if aspect_val not in aspect_allowed or aspect_val == "auto":
                                aspect_val = None
                        size_val = options.get('gemini_image_size')
                        if size_val:
                            size_val = str(size_val).strip().upper()
                            if size_val not in size_allowed:
                                size_val = None
                        image_cfg_kwargs = {}
                        if aspect_val:
                            image_cfg_kwargs["aspect_ratio"] = aspect_val
                        if img_model == "gemini-3.1-flash-lite-image":
                            # Nano Banana 2 Lite supports 1K output only.
                            image_cfg_kwargs["image_size"] = "1K"
                        elif size_val and (
                            "gemini-3-pro-image" in img_model or img_model == "gemini-3.1-flash-image"
                        ):
                            image_cfg_kwargs["image_size"] = size_val
                        config_kwargs = {
                            "temperature": 0.7,
                            "candidate_count": 1,
                            "response_modalities": ["TEXT", "IMAGE"],
                            "safety_settings": [
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                            ]
                        }
                        if img_model in ("gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"):
                            default_level = "minimal" if img_model == "gemini-3.1-flash-lite-image" else "high"
                            raw_lvl = str(options.get('thinking_level') or default_level).lower()
                            if raw_lvl in ("low", "minimal"):
                                nano_banana2_lvl = "minimal"
                            elif raw_lvl in ("medium", "high"):
                                nano_banana2_lvl = "high"
                            else:
                                nano_banana2_lvl = default_level
                            # Both Gemini 3.1 Flash image models support only minimal/high.
                            # The UI checkbox controls thought output visibility; internal thinking remains model-driven.
                            config_kwargs["thinking_config"] = types.ThinkingConfig(
                                include_thoughts=bool(options.get('enable_thinking')),
                                thinking_level=nano_banana2_lvl
                            )
                            # Google Search grounding is supported by Nano Banana 2,
                            # but explicitly unsupported by Nano Banana 2 Lite.
                            if img_model != "gemini-3.1-flash-lite-image" and options.get('enable_search'):
                                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                        if image_cfg_kwargs:
                            config_kwargs["image_config"] = types.ImageConfig(**image_cfg_kwargs)

                        _mark_provider_request_started()
                        gemini_image_parts = []
                        history_image_refs_included = set()
                        for fi in loaded_files:
                            if fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                                if img_model == "gemini-3.1-flash-lite-image" and len(gemini_image_parts) >= 14:
                                    break
                                gemini_image_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                        for hp in history_image_parts:
                            if img_model == "gemini-3.1-flash-lite-image" and len(gemini_image_parts) >= 14:
                                break
                            ref = hp.get("ref")
                            if ref and ref in history_image_refs_included:
                                continue
                            gemini_image_parts.append(types.Part.from_bytes(data=hp['bytes'], mime_type=hp['mime']))
                            if ref:
                                history_image_refs_included.add(ref)

                        resp = g_client.models.generate_content(
                            model=img_model,
                            contents=[
                                *gemini_image_parts,
                                types.Part(text=img_prompt)
                            ],
                            config=types.GenerateContentConfig(**config_kwargs)
                        )

                        text_outputs, image_outputs = _collect_gemini_image_output_parts(
                            resp,
                            keep_only_last_image=(img_model in ("gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"))
                        )

                        if not image_outputs and img_model in ("gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"):
                            log_force(
                                f"Nano Banana 2 returned text-only output; retrying with image-only mode. "
                                f"thread={thread_id} job={job_id}"
                            )
                            retry_cfg_kwargs = dict(config_kwargs)
                            retry_cfg_kwargs.pop("tools", None)
                            retry_cfg_kwargs["response_modalities"] = ["IMAGE"]
                            retry_prompt = (
                                f"{img_prompt}\n\n"
                                "Return an image for this request. Do not answer with text only."
                            )
                            retry_resp = g_client.models.generate_content(
                                model=img_model,
                                contents=[
                                    *gemini_image_parts,
                                    types.Part(text=retry_prompt)
                                ],
                                config=types.GenerateContentConfig(**retry_cfg_kwargs)
                            )
                            retry_text_outputs, retry_image_outputs = _collect_gemini_image_output_parts(
                                retry_resp,
                                keep_only_last_image=True
                            )
                            if retry_image_outputs:
                                text_outputs, image_outputs = retry_text_outputs, retry_image_outputs

                        for txt in text_outputs:
                            pub("content", txt)
                            full_res += txt + ("\n" if not txt.endswith("\n") else "")

                        for part in image_outputs:
                            fn2 = _save_gemini_image_part(part)
                            pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                            full_res += f"Generated Image for: {final_message_text}\n"

                        if not image_outputs and not text_outputs:
                            pub("error", "No image output returned.")
                    except Exception as e:
                        logger.exception("Gemini Image Gen Error")
                        pub("error", f"Gemini Image Gen Error: {str(e)}")

                else:
                    # Text/Chat generation mode
                    rm = model_key
                    if "gemini-3.7-flash" in model_key:
                        rm = "gemini-3.7-flash"
                    elif "gemini-3.6-flash" in model_key:
                        rm = "gemini-3.6-flash"
                    elif "gemini-3.5-flash-lite" in model_key:
                        rm = "gemini-3.5-flash-lite"
                    elif "gemini-3.5-flash" in model_key:
                        rm = "gemini-3.5-flash"
                    elif "gemini-3.1-pro" in model_key:
                        rm = "gemini-3.1-pro-preview"
                    elif model_key == "gemini-3.1-flash-lite":
                        rm = "gemini-3.1-flash-lite"
                    elif "gemini-3.1-flash-lite-preview" in model_key:
                        rm = "gemini-3.1-flash-lite-preview"
                    elif "gemini-3-flash" in model_key or "gemini-3.0-flash" in model_key:
                        rm = "gemini-3-flash-preview"
                    elif "gemini-3-pro" in model_key or "gemini-3.0-pro" in model_key:
                        rm = "gemini-3-pro-preview"
                    elif "gemini-2.5-pro" in model_key:
                        rm = "gemini-2.5-pro"
                    elif "gemini-2.5-flash-lite" in model_key:
                        rm = model_key
                    elif "gemini-2.5" in model_key:
                        rm = "gemini-2.5-flash"

                    is_latest_flash = rm in ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")
                    # Gemini 3.7/3.6 Flash and 3.5 Flash-Lite deprecate sampling
                    # parameters. Omit them so future API generations do not
                    # reject otherwise valid requests.
                    conf = {} if is_latest_flash else {'temperature': 0.7}
                    if is_gemini_3:
                        # Gemini 3 does not support fully disabling thinking; force enabled.
                        options['enable_thinking'] = True
                    if options.get('enable_thinking'):
                        raw_lvl = (options.get('thinking_level') or 'high').lower()
                        lvl = raw_lvl if raw_lvl in ("minimal", "low", "medium", "high") else "high"
                        if rm == "gemini-3.7-flash" and lvl not in ("low", "medium", "high"):
                            lvl = "medium"
                        elif rm == "gemini-3.6-flash" and lvl not in ("medium", "high"):
                            lvl = "medium"
                        elif rm == "gemini-3.5-flash-lite" and lvl not in ("minimal", "medium", "high"):
                            lvl = "minimal"
                        if "gemini-2.5" in model_key:
                            budget_map = {"low": 1024, "medium": 4096, "high": 8192}
                            manual_budget = options.get('thinking_budget')
                            budget_val = None
                            if manual_budget is not None and str(manual_budget).strip() != "":
                                try:
                                    budget_val = int(manual_budget)
                                    if budget_val < 0: budget_val = 0
                                    if budget_val > 32768: budget_val = 32768
                                except Exception:
                                    budget_val = None
                            conf['thinking_config'] = types.ThinkingConfig(
                                include_thoughts=True,
                                thinking_budget=budget_val if budget_val is not None else budget_map.get(raw_lvl, 4096)
                            )
                        else:
                            conf['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_level=lvl)
                    # Avoid forcing "minimal" when users disable thinking, because Gemini 3 does not
                    # support fully turning thinking off and defaults are higher per docs.

                    # Gemini 3 supports combining its built-in code execution tool
                    # with custom function tools. Older Gemini models do not, so
                    # when File is also enabled we expose the same restricted local
                    # executor as a custom function instead of sending incompatible
                    # built-in and custom tools together.
                    _gemini_python_function_active = bool(
                        options.get('enable_python')
                        and options.get('enable_file_creation')
                        and not is_gemini_3
                        and not gemini_local_python
                    )
                    _gemini_manual_function_tools = bool(
                        options.get('enable_python')
                        and options.get('enable_file_creation')
                        and is_gemini_3
                        and not gemini_local_python
                    )
                    _gemini_code_exec_active = False
                    if auto_enable_search:
                        conf['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if auto_enable_url_context:
                        if 'tools' not in conf: conf['tools'] = []
                        conf['tools'].append(types.Tool(url_context=types.UrlContext()))
                    if auto_enable_maps:
                        if 'tools' not in conf: conf['tools'] = []
                        conf['tools'].append(types.Tool(google_maps=types.GoogleMaps()))
                    if options.get('enable_python') and not gemini_local_python:
                        if 'tools' not in conf: conf['tools'] = []
                        if _gemini_python_function_active:
                            log_force(
                                "Gemini: using local Python function because File creation "
                                "is enabled on a pre-Gemini-3 model"
                            )
                        else:
                            conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))
                            _gemini_code_exec_active = True
                            # Agentic View runs Python server-side; give it a longer
                            # deadline so it doesn't hit 504 DEADLINE_EXCEEDED.
                            conf['http_options'] = types.HttpOptions(timeout=_GEMINI_AGENTIC_TIMEOUT_MS)
                            # The sandbox still has its own hard runtime limit (~30s) that
                            # the request timeout cannot extend, so also guide the model to
                            # write code that finishes within it (especially for image edits).
                            if options.get('system_prompt'):
                                conf['system_instruction'] = (
                                    f"{options.get('system_prompt')}\n\n{GEMINI_CODE_EXECUTION_GUIDANCE}"
                                )
                            else:
                                conf['system_instruction'] = GEMINI_CODE_EXECUTION_GUIDANCE
                    last_file_tool_error = None
                    if _gemini_python_function_active:
                        def _gemini_execute_python_tool(code: str) -> str:
                            """併用時に使う、アプリ側の制限付きPython実行ツール。"""
                            nonlocal full_res
                            safe_output = _sanitize_python_sandbox_output(safe_execute_python(code))
                            code_md = f"\n```python\n{code}\n```\n"
                            output_md = f"\n**Output:**\n```\n{safe_output}\n```\n"
                            full_res += code_md + output_md
                            full_res += f"\n```pyexec\n{json.dumps({'code': code, 'output': safe_output})}\n```\n"
                            pub("content", code_md)
                            pub("content", output_md)
                            pub(
                                "python",
                                {
                                    "id": f"gem_function_py_{int(time.time()*1000)}_{os.urandom(3).hex()}",
                                    "code": code,
                                    "output": safe_output,
                                },
                            )
                            return safe_output

                        _gemini_execute_python_tool.__name__ = "execute_python"
                        if 'tools' not in conf: conf['tools'] = []
                        conf['tools'].append(_gemini_execute_python_tool)
                    if options.get('enable_file_creation'):
                        def _gemini_create_file_tool(
                            filename: str,
                            content: str,
                            format: str = "",
                        ) -> str:
                            """テキスト・コード・Markdown・PDF・Word(docx)・Excel(xlsx)ファイルを作成し、ユーザーのファイルライブラリに保存します。保存後のURLを回答内のリンクとして提示してください。format は省略可能（拡張子から自動判定）。PDF/DOCX の content は Markdown 形式、XLSX の content は TSV（タブ区切り、1行目ヘッダー）です。添付済みの既存ファイルの編集を求められた場合は、新規作成せず edit_file を使用してください。"""
                            nonlocal last_file_tool_error
                            result = _execute_create_file_tool(
                                user_id,
                                {
                                    "filename": filename,
                                    "content": content,
                                    "format": format,
                                },
                                user_config.get("enable_e2ee"),
                            )
                            if result.get("ok"):
                                created_file_rel = f"{user_id}/{result['filename']}"
                                if created_file_rel not in generated_images:
                                    generated_images.append(created_file_rel)
                                file_link_md = (
                                    f"\n📄 **ファイルを作成しました:** [{result.get('display_name')}]({result.get('url')})\n"
                                )
                                nonlocal full_res
                                full_res += file_link_md
                                pub("content", file_link_md)
                            else:
                                last_file_tool_error = _create_file_tool_result_text(result)
                            return _create_file_tool_result_text(result)

                        _gemini_create_file_tool.__name__ = "create_file"
                        if not _gemini_manual_function_tools:
                            if 'tools' not in conf: conf['tools'] = []
                            conf['tools'].append(_gemini_create_file_tool)

                        def _gemini_edit_file_tool(
                            source: str,
                            content: Optional[str] = None,
                            cell_edits: Optional[list[dict]] = None,
                            paragraph_edits: Optional[list[dict]] = None,
                            text_edits: Optional[list[dict]] = None,
                        ) -> str:
                            """会話に添付された既存のファイルや過去に作成・編集されたファイル（source に添付ファイル名またはURL）を編集し、編集後のファイルをユーザーのファイルライブラリに保存します。元ファイルの書式・構造・内容を保ったまま変更してください。Excel(xlsx/xlsm)は cell_edits で変更するセルのリストを指定します。各要素は { cell: セル番地（添付時に表示される列名ヘッダー A, B, C... を参照、例 B5）, value: 新しい値, sheet: シート名(省略時は最初のシート), style: 任意の新しい書式 } です。style は指定した項目だけを上書きします: fill: { color: '#RRGGBB' 塗りつぶし色, fillType: 'solid'(既定)/'none'(解除) }, font: { bold, italic, strikethrough, underline: 'none'/'single'/'double', color: 文字色, size: サイズpt, name: フォント名 }, border: { style: 'none'/'thin'/'medium'/'thick'/'dashed'/'dotted'/'double'（四辺）, color: 罫線色, left/right/top/bottom: { style, color } で辺ごとに上書き }, alignment: { horizontal: 'left'/'center'/'right'/'justify', vertical: 'top'/'center'/'bottom', wrapText }, numberFormat: 表示形式コード。Word(docx)は paragraph_edits で変更する段落のリストを指定します（元の書式を維持し、編集後に PDF 版も生成されます）。各要素は { paragraph: [N]番号または段落テキスト(部分一致), text: 新しいテキスト(省略可), style: { font: { bold, italic, strikethrough, underline: 'none'/'single'/'double', color: 文字色, size: サイズpt, name: フォント名, highlight: 'yellow'/'green'/'cyan'/'magenta'/'red'/'blue'/'gray'/'none' }, alignment: 'left'/'center'/'right'/'justify'/'default' } } です。PDF は text_edits で { find: 検索文字列, replace: 置換文字列, page: ページ番号(省略可) } のリストを指定すると、レイアウトを保ったままベストエフォートで置換します（見つからない場合はエラー）。テキスト系は content に編集後の全文、PDF/DOCX の全文置き換えは content に Markdown 形式の全文を指定します。"""
                            nonlocal last_file_tool_error
                            result = _execute_edit_file_tool(
                                user_id,
                                {
                                    "source": source,
                                    "content": content,
                                    "cell_edits": cell_edits,
                                    "paragraph_edits": paragraph_edits,
                                    "text_edits": text_edits,
                                },
                                user_config.get("enable_e2ee"),
                                loaded_files=loaded_files,
                                history=history,
                                thread_id=thread_id,
                            )
                            if result.get("ok"):
                                edited_file_rel = f"{user_id}/{result['filename']}"
                                if edited_file_rel not in generated_images:
                                    generated_images.append(edited_file_rel)
                                file_link_md = (
                                    f"\n📄 **ファイルを編集しました:** [{result.get('display_name')}]({result.get('url')})\n"
                                )
                                nonlocal full_res
                                full_res += file_link_md
                                pub("content", file_link_md)
                            else:
                                last_file_tool_error = _create_file_tool_result_text(result, "edit_file", "編集")
                            return _create_file_tool_result_text(result, "edit_file", "編集")

                        _gemini_edit_file_tool.__name__ = "edit_file"
                        if not _gemini_manual_function_tools:
                            if 'tools' not in conf: conf['tools'] = []
                            conf['tools'].append(_gemini_edit_file_tool)

                        if _gemini_manual_function_tools:
                            create_schema = _build_create_file_tool_schema()['function']
                            edit_schema = _build_edit_file_tool_schema()['function']
                            if 'tools' not in conf: conf['tools'] = []
                            conf['tools'].append(types.Tool(function_declarations=[
                                {
                                    "name": create_schema["name"],
                                    "description": create_schema.get("description", ""),
                                    "parameters": create_schema.get("parameters", {}),
                                },
                                {
                                    "name": edit_schema["name"],
                                    "description": edit_schema.get("description", ""),
                                    "parameters": edit_schema.get("parameters", {}),
                                },
                            ]))
                            # Gemini 3 requires this flag and manual function
                            # responses to circulate code-execution context.
                            conf['tool_config'] = types.ToolConfig(
                                include_server_side_tool_invocations=True
                            )

                    # MCP外部ツール（google-genai の Automatic Function Calling 用の
                    # callable として追加。function_declarations は AFC を無効化するため使わない）
                    _mcp_runtime = None
                    _gemini_mcp_callables = []
                    if not gemini_local_python:
                        try:
                            _mcp_runtime = _ensure_mcp_env()
                        except Exception:
                            _mcp_runtime = None
                    if _mcp_runtime is not None:
                        try:
                            from mcp_service.gemini_tools import build_gemini_mcp_tools

                            def _on_gemini_mcp_result(_text, _mout, _iname):
                                nonlocal full_res
                                try:
                                    if _mout.get("ok"):
                                        _md = f"\n\n> 🔧 **MCPツール実行:** `{_iname}` を実行しました。\n"
                                        full_res += _md
                                        pub("content", _md)
                                    elif _mout.get("rejected"):
                                        _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{_iname}`\n"
                                        full_res += _md
                                        pub("content", _md)
                                except Exception:
                                    pass

                            _gemini_mcp_callables = build_gemini_mcp_tools(_mcp_runtime, on_result=_on_gemini_mcp_result)
                            if _gemini_mcp_callables:
                                if 'tools' not in conf: conf['tools'] = []
                                conf['tools'].extend(_gemini_mcp_callables)
                                log_force(f"Gemini MCP tools attached: {len(_gemini_mcp_callables)}")
                        except Exception as _mcp_e:
                            log_force(f"Gemini MCP tool attach failed: {_mcp_e}")
                    if options.get('system_prompt') and 'system_instruction' not in conf:
                        conf['system_instruction'] = options.get('system_prompt')
                    
                    contents = []
                    history_img_seen = set()
                    history_img_bytes = 0
                    for m in history:
                        parts = []
                        if m.get('signature'):
                            sig_val = m.get('signature')
                            sig_list = None
                            if isinstance(sig_val, str):
                                try:
                                    parsed = json.loads(sig_val)
                                    if isinstance(parsed, list):
                                        sig_list = parsed
                                    elif isinstance(parsed, str):
                                        sig_list = [parsed]
                                except Exception:
                                    sig_list = [sig_val]
                            elif isinstance(sig_val, list):
                                sig_list = sig_val
                            if sig_list:
                                for s in sig_list:
                                    try:
                                        parts.append(types.Part(thought_signature=base64.b64decode(s)))
                                    except Exception:
                                        pass
                        if m['content']:
                            parts.append(types.Part(text=m['content']))
                        if m.get('image_url'):
                            try:
                                msg_images, history_img_bytes = _load_message_history_images(
                                    m.get('image_url'),
                                    seen=history_img_seen,
                                    total_bytes=history_img_bytes,
                                    include_only_images=True
                                )
                                for msg_img in msg_images:
                                    parts.append(types.Part.from_bytes(data=msg_img['bytes'], mime_type=msg_img['mime']))
                            except: pass
                        if parts: contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

                    curr_parts = []
                    if final_message_text and str(final_message_text).strip():
                        curr_parts.append(types.Part(text=final_message_text))
                    media_inline_limit = 20 * 1024 * 1024  # 20MiB limit for inline audio
                    pending_file_error = None

                    def _gemini_file_state_name(fobj):
                        if not fobj:
                            return None
                        st = fobj.get("state") if isinstance(fobj, dict) else getattr(fobj, "state", None)
                        if isinstance(st, dict):
                            return st.get("name") or st.get("state")
                        return getattr(st, "name", None) or st

                    def _gemini_file_name(fobj):
                        if not fobj:
                            return None
                        return fobj.get("name") if isinstance(fobj, dict) else getattr(fobj, "name", None)

                    def _gemini_file_uri(fobj):
                        if not fobj:
                            return None
                        if isinstance(fobj, dict):
                            return fobj.get("uri") or fobj.get("file_uri") or fobj.get("fileUri") or fobj.get("name")
                        return (
                            getattr(fobj, "uri", None)
                            or getattr(fobj, "file_uri", None)
                            or getattr(fobj, "fileUri", None)
                            or getattr(fobj, "name", None)
                        )

                    def _make_gemini_uri_part(file_uri, mime):
                        if not file_uri:
                            return None
                        if hasattr(types.Part, "from_uri"):
                            try:
                                return types.Part.from_uri(file_uri, mime_type=mime)
                            except TypeError:
                                try:
                                    return types.Part.from_uri(file_uri, mime)
                                except Exception:
                                    try:
                                        return types.Part.from_uri(file_uri=file_uri, mime_type=mime)
                                    except Exception:
                                        try:
                                            return types.Part.from_uri(uri=file_uri, mime_type=mime)
                                        except Exception:
                                            return None
                        if hasattr(types, "FileData"):
                            try:
                                return types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime))
                            except Exception:
                                try:
                                    return types.Part(file_data=types.FileData(file_uri=file_uri))
                                except Exception:
                                    return None
                        return None

                    def _wait_gemini_file_active(fobj, label=""):
                        state = _gemini_file_state_name(fobj)
                        if not state or state == "ACTIVE":
                            return fobj, state
                        if state == "FAILED":
                            return fobj, state
                        name = _gemini_file_name(fobj)
                        deadline = time.time() + 120
                        while state == "PROCESSING" and time.time() < deadline:
                            time.sleep(2)
                            try:
                                if name:
                                    fobj = g_client.files.get(name=name)
                                else:
                                    break
                            except Exception as e:
                                log_force(f"Gemini file poll failed {label}: {e}")
                                break
                            state = _gemini_file_state_name(fobj)
                        return fobj, state

                    def _normalize_gemini_audio(data, mime, name=""):
                        if not data or not mime:
                            return data, mime, name
                        m = (mime or '').lower()
                        ext = (os.path.splitext(name or '')[1] or '').lower()
                        if m in ("audio/webm", "audio/ogg", "audio/oga", "audio/opus") or ext in (".webm", ".ogg", ".oga", ".opus"):
                            try:
                                src_suffix = ext if ext else ".webm"
                                pcm = _convert_audio_to_pcm(data, src_suffix=src_suffix, rate=16000)
                                wav = _pcm_to_wav_bytes(pcm, rate=16000)
                                base = os.path.splitext(name or "audio")[0]
                                return wav, "audio/wav", f"{base}.wav"
                            except Exception as e:
                                log_force(f"Gemini audio convert failed: {e}")
                        return data, mime, name

                    def _gemini_cache_matches(cache, size, mtime, mime):
                        if not cache or not cache.file_uri:
                            return False
                        if size is not None and cache.size_bytes is not None and cache.size_bytes != size:
                            return False
                        if mtime is not None and cache.mtime is not None and cache.mtime != mtime:
                            return False
                        if mime and cache.mime_type and cache.mime_type != mime:
                            return False
                        return True

                    def _gemini_get_cached_part(rel_path, mime, size=None, mtime=None, label=""):
                        if not gemini_files_api_enabled:
                            return None
                        cache = _get_file_cache(user_id, rel_path, "gemini")
                        if not _gemini_cache_matches(cache, size, mtime, mime):
                            return None
                        try:
                            if cache.file_id:
                                fobj = g_client.files.get(name=cache.file_id)
                                state = _gemini_file_state_name(fobj)
                                cache.file_uri = _gemini_file_uri(fobj) or cache.file_uri
                                cache.state = state or cache.state
                                cache.last_checked_at = datetime.utcnow()
                                if state and state != "ACTIVE":
                                    if state == "PROCESSING":
                                        fobj, state = _wait_gemini_file_active(fobj, label=label)
                                        cache.file_uri = _gemini_file_uri(fobj) or cache.file_uri
                                        cache.state = state or cache.state
                                    if state and state != "ACTIVE":
                                        _upsert_file_cache(
                                            user_id,
                                            rel_path,
                                            "gemini",
                                            state=state,
                                            last_error=f"state:{state}",
                                            size_bytes=size,
                                            mtime=mtime,
                                            mime_type=mime,
                                            last_checked_at=datetime.utcnow()
                                        )
                                        safe_db_commit()
                                        return None
                            part = _make_gemini_uri_part(cache.file_uri, mime)
                            if part:
                                _upsert_file_cache(
                                    user_id,
                                    rel_path,
                                    "gemini",
                                    state="ACTIVE",
                                    last_error=None,
                                    size_bytes=size,
                                    mtime=mtime,
                                    mime_type=mime,
                                    last_checked_at=datetime.utcnow()
                                )
                                safe_db_commit()
                                return part
                        except Exception as e:
                            _upsert_file_cache(
                                user_id,
                                rel_path,
                                "gemini",
                                state="FAILED",
                                last_error=str(e),
                                size_bytes=size,
                                mtime=mtime,
                                mime_type=mime,
                                last_checked_at=datetime.utcnow()
                            )
                            safe_db_commit()
                        return None

                    def _gemini_upload_with_retry(data, mime, suffix, rel_path, label="", display_name=None):
                        if not gemini_files_api_enabled:
                            return None, None, "Vertex AI モードではこのアプリの Files API 経路を利用できません（20MB以下にするか Gemini API モードへ切替してください）。"
                        max_attempts = 2
                        try:
                            max_attempts = int(os.getenv("GEMINI_FILE_UPLOAD_RETRIES", "2") or "2")
                        except Exception:
                            max_attempts = 2
                        last_err = None
                        for attempt in range(max_attempts):
                            try:
                                _upsert_file_cache(
                                    user_id,
                                    rel_path,
                                    "gemini",
                                    state="UPLOADING",
                                    last_error=None,
                                    retries=attempt + 1
                                )
                                safe_db_commit()
                                with tempfile.NamedTemporaryFile(suffix=suffix or '.bin') as tmp:
                                    tmp.write(data)
                                    tmp.flush()
                                    config = {"mimeType": mime}
                                    if display_name:
                                        config["display_name"] = display_name
                                        config["displayName"] = display_name
                                        config["name"] = display_name
                                    up = g_client.files.upload(file=tmp.name, config=config)
                                up, up_state = _wait_gemini_file_active(up, label=label)
                                if up_state and up_state != "ACTIVE":
                                    last_err = f"state:{up_state}"
                                    time.sleep(1)
                                    continue
                                return up, up_state, None
                            except Exception as e:
                                last_err = str(e)
                                time.sleep(1)
                                continue
                        return None, None, last_err

                    current_image_names = []
                    inline_image_bytes = 0
                    for fi in loaded_files:
                        if fi.get('is_pdf') and supports_pdf_inputs and fi.get('bytes'):
                            try:
                                pdf_bytes = fi['bytes']
                                pdf_mime = fi.get('mime') or 'application/pdf'
                                pdf_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.pdf')
                                rel_path = fi.get('path') or fi.get('name') or pdf_name
                                if len(pdf_bytes) <= media_inline_limit:
                                    curr_parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type=pdf_mime))
                                else:
                                    cached_part = _gemini_get_cached_part(
                                        rel_path,
                                        pdf_mime,
                                        size=fi.get('size'),
                                        mtime=fi.get('mtime'),
                                        label=f"pdf:{pdf_name}"
                                    )
                                    if cached_part:
                                        curr_parts.append(cached_part)
                                    else:
                                        up, up_state, up_err = _gemini_upload_with_retry(
                                            pdf_bytes,
                                            pdf_mime,
                                            os.path.splitext(pdf_name)[1] or '.pdf',
                                            rel_path,
                                            label=f"pdf:{pdf_name}",
                                            display_name=pdf_name
                                        )
                                        if not up or up_err:
                                            pending_file_error = f"PDF({pdf_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                                            break
                                        file_uri = _gemini_file_uri(up)
                                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or pdf_mime
                                        part = _make_gemini_uri_part(file_uri, up_mime)
                                        if part:
                                            curr_parts.append(part)
                                            _upsert_file_cache(
                                                user_id,
                                                rel_path,
                                                "gemini",
                                                file_id=_gemini_file_name(up),
                                                file_uri=file_uri,
                                                state=up_state or "ACTIVE",
                                                last_error=None,
                                                size_bytes=fi.get('size'),
                                                mtime=fi.get('mtime'),
                                                mime_type=up_mime,
                                                last_checked_at=datetime.utcnow()
                                            )
                                            safe_db_commit()
                                        else:
                                            pending_file_error = f"PDF({pdf_name})参照の生成に失敗しました。再送してください。"
                                            break
                            except Exception as e:
                                log_force(f"Gemini PDF upload failed: {e}")
                                if fi.get('text'):
                                    curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
                            continue
                        if fi.get('is_docx') and supports_docx_inputs and fi.get('bytes'):
                            try:
                                docx_bytes = fi['bytes']
                                docx_mime = fi.get('mime') or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                                docx_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.docx')
                                rel_path = fi.get('path') or fi.get('name') or docx_name
                                if len(docx_bytes) <= media_inline_limit:
                                    curr_parts.append(types.Part.from_bytes(data=docx_bytes, mime_type=docx_mime))
                                else:
                                    cached_part = _gemini_get_cached_part(
                                        rel_path,
                                        docx_mime,
                                        size=fi.get('size'),
                                        mtime=fi.get('mtime'),
                                        label=f"docx:{docx_name}"
                                    )
                                    if cached_part:
                                        curr_parts.append(cached_part)
                                    else:
                                        up, up_state, up_err = _gemini_upload_with_retry(
                                            docx_bytes,
                                            docx_mime,
                                            os.path.splitext(docx_name)[1] or '.docx',
                                            rel_path,
                                            label=f"docx:{docx_name}"
                                        )
                                        if not up or up_err:
                                            pending_file_error = f"Word({docx_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                                            break
                                        file_uri = _gemini_file_uri(up)
                                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or docx_mime
                                        part = _make_gemini_uri_part(file_uri, up_mime)
                                        if part:
                                            curr_parts.append(part)
                                            _upsert_file_cache(
                                                user_id,
                                                rel_path,
                                                "gemini",
                                                file_id=_gemini_file_name(up),
                                                file_uri=file_uri,
                                                state=up_state or "ACTIVE",
                                                last_error=None,
                                                size_bytes=fi.get('size'),
                                                mtime=fi.get('mtime'),
                                                mime_type=up_mime,
                                                last_checked_at=datetime.utcnow()
                                            )
                                            safe_db_commit()
                                        else:
                                            pending_file_error = f"Word({docx_name})参照の生成に失敗しました。再送してください。"
                                            break
                            except Exception as e:
                                log_force(f"Gemini docx upload failed: {e}")
                                if fi.get('text'):
                                    curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
                            continue
                        if fi.get('is_text') and supports_text_file_inputs and fi.get('bytes'):
                            attached = False
                            try:
                                txt_bytes = fi['bytes']
                                txt_mime = fi.get('mime') or 'text/plain'
                                txt_name = os.path.basename(fi.get('send_name') or fi.get('name') or 'document.txt')
                                rel_path = fi.get('path') or fi.get('name') or txt_name
                                if len(txt_bytes) <= media_inline_limit:
                                    curr_parts.append(types.Part.from_bytes(data=txt_bytes, mime_type=txt_mime))
                                else:
                                    cached_part = _gemini_get_cached_part(
                                        rel_path,
                                        txt_mime,
                                        size=fi.get('size'),
                                        mtime=fi.get('mtime'),
                                        label=f"text:{txt_name}"
                                    )
                                    if cached_part:
                                        curr_parts.append(cached_part)
                                    else:
                                        up, up_state, up_err = _gemini_upload_with_retry(
                                            txt_bytes,
                                            txt_mime,
                                            os.path.splitext(txt_name)[1] or '.txt',
                                            rel_path,
                                            label=f"text:{txt_name}"
                                        )
                                        if not up or up_err:
                                            pending_file_error = f"テキスト({txt_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                                            break
                                        file_uri = _gemini_file_uri(up)
                                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or txt_mime
                                        part = _make_gemini_uri_part(file_uri, up_mime)
                                        if part:
                                            curr_parts.append(part)
                                            _upsert_file_cache(
                                                user_id,
                                                rel_path,
                                                "gemini",
                                                file_id=_gemini_file_name(up),
                                                file_uri=file_uri,
                                                state=up_state or "ACTIVE",
                                                last_error=None,
                                                size_bytes=fi.get('size'),
                                                mtime=fi.get('mtime'),
                                                mime_type=up_mime,
                                                last_checked_at=datetime.utcnow()
                                            )
                                            safe_db_commit()
                                        else:
                                            pending_file_error = f"テキスト({txt_name})参照の生成に失敗しました。再送してください。"
                                            break
                                attached = True
                            except Exception as e:
                                log_force(f"Gemini text file upload failed: {e}")
                            if attached:
                                continue
                        if fi.get('text'):
                            curr_parts.append(types.Part(text=f"\nFile: {fi.get('send_name') or fi.get('name') or 'file'}\n{fi['text']}"))
                            continue
                        if not fi.get('bytes'):
                            continue
                        mime = (fi.get('mime') or 'application/octet-stream').lower()
                        if mime.startswith('image/'):
                            img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"image{len(current_image_names) + 1}")
                            rel_path = fi.get('path') or fi.get('name') or img_name
                            img_size = len(fi['bytes'])

                            attached = False
                            # Small images are latency-sensitive and fit safely in the
                            # GenerateContent payload. Avoid a separate Files API round trip.
                            if inline_image_bytes + img_size <= _GEMINI_INLINE_IMAGE_MAX_BYTES:
                                curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                                inline_image_bytes += img_size
                                attached = True
                            elif gemini_files_api_enabled:
                                try:
                                    cached_part = _gemini_get_cached_part(
                                        rel_path,
                                        fi['mime'],
                                        size=img_size,
                                        mtime=fi.get('mtime'),
                                        label=f"image:{img_name}"
                                    )
                                    if cached_part:
                                        curr_parts.append(cached_part)
                                        attached = True
                                    else:
                                        up, up_state, up_err = _gemini_upload_with_retry(
                                            fi['bytes'],
                                            fi['mime'],
                                            os.path.splitext(img_name)[1] or '.png',
                                            rel_path,
                                            label=f"image:{img_name}",
                                            display_name=img_name
                                        )
                                        if up and not up_err:
                                            file_uri = _gemini_file_uri(up)
                                            up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or fi['mime']
                                            part = _make_gemini_uri_part(file_uri, up_mime)
                                            if part:
                                                curr_parts.append(part)
                                                _upsert_file_cache(
                                                    user_id,
                                                    rel_path,
                                                    "gemini",
                                                    file_id=_gemini_file_name(up),
                                                    file_uri=file_uri,
                                                    state=up_state or "ACTIVE",
                                                    last_error=None,
                                                    size_bytes=img_size,
                                                    mtime=fi.get('mtime'),
                                                    mime_type=up_mime,
                                                    last_checked_at=datetime.utcnow()
                                                )
                                                safe_db_commit()
                                                attached = True
                                except Exception as e:
                                    log_force(f"Gemini image upload failed: {e}")

                            if not attached:
                                curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                                inline_image_bytes += img_size
                            img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
                            current_image_names.append(os.path.basename(str(img_label)))
                            continue
                        if mime.startswith('audio/'):
                            try:
                                audio_bytes, audio_mime, audio_name = _normalize_gemini_audio(fi['bytes'], fi.get('mime') or mime, fi.get('send_name') or fi.get('name') or "")
                                rel_path = fi.get('path') or fi.get('name') or audio_name
                                audio_size = len(audio_bytes) if audio_bytes is not None else fi.get('size')
                                if len(audio_bytes) <= media_inline_limit:
                                    curr_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime))
                                else:
                                    cached_part = _gemini_get_cached_part(
                                        rel_path,
                                        audio_mime,
                                        size=audio_size,
                                        mtime=fi.get('mtime'),
                                        label=f"audio:{audio_name}"
                                    )
                                    if cached_part:
                                        curr_parts.append(cached_part)
                                    else:
                                        up, up_state, up_err = _gemini_upload_with_retry(
                                            audio_bytes,
                                            audio_mime,
                                            os.path.splitext(audio_name or '')[1] or '.bin',
                                            rel_path,
                                            label=f"audio:{audio_name}",
                                            display_name=audio_name
                                        )
                                        if not up or up_err:
                                            pending_file_error = f"音声({audio_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                                            break
                                        file_uri = _gemini_file_uri(up)
                                        up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or audio_mime
                                        part = _make_gemini_uri_part(file_uri, up_mime)
                                        if part:
                                            curr_parts.append(part)
                                            _upsert_file_cache(
                                                user_id,
                                                rel_path,
                                                "gemini",
                                                file_id=_gemini_file_name(up),
                                                file_uri=file_uri,
                                                state=up_state or "ACTIVE",
                                                last_error=None,
                                                size_bytes=fi.get('size'),
                                                mtime=fi.get('mtime'),
                                                mime_type=up_mime,
                                                last_checked_at=datetime.utcnow()
                                            )
                                            safe_db_commit()
                                        else:
                                            pending_file_error = f"音声({audio_name})参照の生成に失敗しました。再送してください。"
                                            break
                            except Exception as e:
                                log_force(f"Gemini audio upload failed: {e}")
                                pending_file_error = f"音声({fi.get('send_name') or fi.get('name') or 'file'})のアップロードに失敗しました。再送してください。"
                                break
                            continue
                        if mime.startswith('video/'):
                            try:
                                video_bytes = fi['bytes']
                                video_mime = fi.get('mime') or mime
                                video_name = fi.get('send_name') or fi.get('name') or "video"
                                video_size = len(video_bytes) if video_bytes is not None else fi.get('size')
                                rel_path = fi.get('path') or fi.get('name') or video_name
                                cached_part = _gemini_get_cached_part(
                                    rel_path,
                                    video_mime,
                                    size=video_size,
                                    mtime=fi.get('mtime'),
                                    label=f"video:{video_name}"
                                )
                                if cached_part:
                                    curr_parts.append(cached_part)
                                else:
                                    up, up_state, up_err = _gemini_upload_with_retry(
                                        video_bytes,
                                        video_mime,
                                        os.path.splitext(video_name or '')[1] or '.bin',
                                        rel_path,
                                        label=f"video:{video_name}",
                                        display_name=video_name
                                    )
                                    if not up or up_err:
                                        pending_file_error = f"動画({video_name})のアップロードに失敗しました: {up_err or 'unknown error'}"
                                        break
                                    file_uri = _gemini_file_uri(up)
                                    up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or video_mime
                                    part = _make_gemini_uri_part(file_uri, up_mime)
                                    if part:
                                        curr_parts.append(part)
                                        _upsert_file_cache(
                                            user_id,
                                            rel_path,
                                            "gemini",
                                            file_id=_gemini_file_name(up),
                                            file_uri=file_uri,
                                            state=up_state or "ACTIVE",
                                            last_error=None,
                                            size_bytes=video_size,
                                            mtime=fi.get('mtime'),
                                            mime_type=up_mime,
                                            last_checked_at=datetime.utcnow()
                                        )
                                        safe_db_commit()
                                    else:
                                        pending_file_error = f"動画({video_name})参照の生成に失敗しました。再送してください。"
                                        break
                            except Exception as e:
                                log_force(f"Gemini video upload failed: {e}")
                                pending_file_error = f"動画({fi.get('send_name') or fi.get('name') or 'file'})のアップロードに失敗しました。再送してください。"
                                break
                            continue
                        # Skip unsupported binary inputs for Gemini text models
                        pass

                    name_block = _build_attachment_name_block(current_image_names)
                    if name_block:
                        curr_parts.insert(1, types.Part(text=name_block))

                    if pending_file_error:
                        pub("error", pending_file_error)
                        return

                    contents.append(types.Content(role='user', parts=curr_parts))

                    grounding_chunks = None
                    grounding_supports = None
                    url_context_chunks = None

                    def _collect_grounding(gm):
                        nonlocal grounding_chunks, grounding_supports
                        if not gm:
                            return
                        g_chunks = getattr(gm, 'grounding_chunks', None) or getattr(gm, 'groundingChunks', None) or []
                        if g_chunks and grounding_chunks is None:
                            grounding_chunks = g_chunks
                        g_supports = getattr(gm, 'grounding_supports', None) or getattr(gm, 'groundingSupports', None) or []
                        if g_supports and grounding_supports is None:
                            grounding_supports = g_supports

                    def _collect_url_context(ucm):
                        nonlocal url_context_chunks
                        if not ucm:
                            return
                        u_metadata = getattr(ucm, 'url_metadata', None) or getattr(ucm, 'urlMetadata', None) or []
                        if u_metadata and url_context_chunks is None:
                            url_context_chunks = u_metadata

                    def _chunk_grounding_info(chunk):
                        if not chunk:
                            return None, None
                        candidates = [chunk]
                        if isinstance(chunk, dict):
                            candidates.extend([chunk.get('web'), chunk.get('maps')])
                        else:
                            candidates.extend([getattr(chunk, 'web', None), getattr(chunk, 'maps', None)])
                        for candidate in candidates:
                            if not candidate:
                                continue
                            if isinstance(candidate, dict):
                                title = candidate.get('title') or candidate.get('name') or candidate.get('text')
                                uri = candidate.get('uri') or candidate.get('url')
                            else:
                                title = getattr(candidate, 'title', None) or getattr(candidate, 'name', None) or getattr(candidate, 'text', None)
                                uri = getattr(candidate, 'uri', None) or getattr(candidate, 'url', None)
                            if title or uri:
                                return title, uri
                        if isinstance(chunk, dict):
                            title = chunk.get('title') or chunk.get('name') or chunk.get('text')
                            uri = chunk.get('uri') or chunk.get('url')
                            if not title:
                                place_id = chunk.get('place_id') or chunk.get('placeId')
                                if place_id:
                                    title = place_id
                        else:
                            title = getattr(chunk, 'title', None) or getattr(chunk, 'name', None) or getattr(chunk, 'text', None)
                            uri = getattr(chunk, 'uri', None) or getattr(chunk, 'url', None)
                            if not title:
                                place_id = getattr(chunk, 'place_id', None) or getattr(chunk, 'placeId', None)
                                if place_id:
                                    title = place_id
                        return title, uri

                    def _segment_end_index(segment):
                        if segment is None:
                            return None
                        end_index = getattr(segment, 'end_index', None)
                        if end_index is None:
                            end_index = getattr(segment, 'endIndex', None)
                        return end_index

                    def _add_gemini_citations(text, supports, chunks):
                        if not text or not supports or not chunks:
                            return text
                        try:
                            sorted_supports = sorted(
                                supports,
                                key=lambda s: _segment_end_index(getattr(s, 'segment', None)) or 0,
                                reverse=True
                            )
                        except Exception:
                            sorted_supports = supports
                        for support in sorted_supports:
                            segment = getattr(support, 'segment', None)
                            end_index = _segment_end_index(segment)
                            if end_index is None or end_index > len(text):
                                continue
                            idxs = getattr(support, 'grounding_chunk_indices', None) or getattr(support, 'groundingChunkIndices', None) or []
                            if not idxs:
                                continue
                            citation_links = []
                            for i in idxs:
                                try:
                                    idx = int(i)
                                except Exception:
                                    continue
                                if idx < 0 or idx >= len(chunks):
                                    continue
                                _, uri = _chunk_grounding_info(chunks[idx])
                                if uri:
                                    citation_links.append(f"[{idx + 1}]({uri})")
                            if citation_links:
                                text = text[:end_index] + "".join(citation_links) + text[end_index:]
                        return text

                    def _extract_gemini_thought_text(part):
                        if not part:
                            return ""
                        thought_val = getattr(part, 'thought', None)
                        text_val = getattr(part, 'text', None)
                        if isinstance(thought_val, str):
                            return thought_val
                        if isinstance(thought_val, dict):
                            t_val = thought_val.get("text") or thought_val.get("content") or thought_val.get("value")
                            if t_val is not None:
                                return str(t_val)
                        if thought_val is not None and not isinstance(thought_val, bool):
                            for key in ("text", "content", "value"):
                                t_val = getattr(thought_val, key, None)
                                if t_val:
                                    return str(t_val)
                        if text_val:
                            return str(text_val)
                        return ""

                    def _gemini_manual_function_stream():
                        """Gemini 3のツール併用でFileのfunction responseを循環させる。"""
                        request_contents = list(contents)
                        responses = []
                        for _tool_round in range(8):
                            _mark_provider_request_started()
                            response = g_client.models.generate_content(
                                model=rm,
                                contents=request_contents,
                                config=types.GenerateContentConfig(**conf),
                            )
                            responses.append(response)
                            candidates = getattr(response, "candidates", None) or []
                            model_content = getattr(candidates[0], "content", None) if candidates else None
                            parts = getattr(model_content, "parts", None) or []
                            calls = []
                            for part in parts:
                                function_call = getattr(part, "function_call", None)
                                if not function_call and isinstance(part, dict):
                                    function_call = part.get("function_call") or part.get("functionCall")
                                if not function_call:
                                    continue
                                if isinstance(function_call, dict):
                                    call_name = function_call.get("name")
                                    call_args = function_call.get("args") or function_call.get("arguments") or {}
                                    call_id = function_call.get("id")
                                else:
                                    call_name = getattr(function_call, "name", None)
                                    call_args = getattr(function_call, "args", None) or getattr(function_call, "arguments", None) or {}
                                    call_id = getattr(function_call, "id", None)
                                if call_name in ("create_file", "edit_file"):
                                    calls.append((call_name, call_args, call_id))
                            if not calls:
                                break

                            if model_content is not None:
                                request_contents.append(model_content)
                            response_parts = []
                            for call_name, call_args, call_id in calls:
                                try:
                                    if not isinstance(call_args, dict):
                                        raise ValueError("function arguments must be an object")
                                    if call_name == "create_file":
                                        result_text = _gemini_create_file_tool(**call_args)
                                    else:
                                        result_text = _gemini_edit_file_tool(**call_args)
                                except Exception as tool_exc:
                                    result_text = f"Error executing {call_name}: {tool_exc}"
                                function_response = types.FunctionResponse(
                                    name=call_name,
                                    response={"response": str(result_text or "Tool executed.")},
                                    id=call_id,
                                )
                                response_parts.append(types.Part(function_response=function_response))
                            request_contents.append(types.Content(role="user", parts=response_parts))
                        return iter(responses)

                    # Automatic Function Calling is managed by the Python SDK on the
                    # non-streaming generate_content path. generate_content_stream
                    # exposes the function-call part but does not complete the AFC
                    # request/response cycle, which used to leave MCP/File turns empty.
                    def _gemini_mcp_aware_stream():
                        if _gemini_manual_function_tools:
                            return _gemini_manual_function_stream()
                        if _gemini_mcp_callables or options.get('enable_file_creation'):
                            _mark_provider_request_started()
                            log_force(f"STREAM-TRACE: Gemini custom-tool AFC request for {job_id} model={rm}")
                            response = g_client.models.generate_content(
                                model=rm,
                                contents=contents,
                                config=types.GenerateContentConfig(**conf),
                            )
                            return iter((response,))
                        return g_client.models.generate_content_stream(
                            model=rm,
                            contents=contents,
                            config=types.GenerateContentConfig(**conf),
                        )

                    _mark_provider_request_started()
                    log_force(f"STREAM-TRACE: Gemini stream starting for {job_id} model={rm}")
                    # The streaming generator performs the HTTP request only when the
                    # first chunk is pulled. Google intermittently returns 504
                    # DEADLINE_EXCEEDED on the initial response of code-execution
                    # requests before any content is generated; a plain retry usually
                    # succeeds. Retry the first-chunk pull (nothing has been streamed
                    # yet, so no user-visible content is duplicated).
                    _gemini_stream_attempts = 0
                    while True:
                        try:
                            _gemini_stream = _gemini_mcp_aware_stream()
                            _gemini_stream_iter = iter(_gemini_stream)
                            _gemini_first_chunk = next(_gemini_stream_iter)
                            break
                        except Exception as _stream_exc:
                            _is_deadline = (
                                "504" in str(_stream_exc) or "DEADLINE_EXCEEDED" in str(_stream_exc)
                            )
                            if (
                                _gemini_code_exec_active
                                and _is_deadline
                                and _gemini_stream_attempts < _GEMINI_STREAM_DEADLINE_RETRIES
                                and not check_stop()
                            ):
                                _gemini_stream_attempts += 1
                                log_force(
                                    f"STREAM-TRACE: Gemini 504 before first chunk, retry "
                                    f"{_gemini_stream_attempts}/{_GEMINI_STREAM_DEADLINE_RETRIES} for {job_id}"
                                )
                                time.sleep(2)
                                continue
                            raise
                    current_py_id = None
                    current_py_code = None
                    final_usage_metadata = None
                    log_force(f"STREAM-TRACE: Gemini stream loop start for {job_id}")
                    for chunk in itertools.chain([_gemini_first_chunk], _gemini_stream_iter):
                        _latency_mark_once(job_id, "provider_first_chunk_ms")
                        if check_stop():
                            log_force(f"STREAM-TRACE: Gemini stream breaking due to stop for {job_id}")
                            break
                        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                            final_usage_metadata = chunk.usage_metadata
                        
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for cand in chunk.candidates:
                                gm = getattr(cand, 'grounding_metadata', None)
                                _collect_grounding(gm)
                                ucm = getattr(cand, 'url_context_metadata', None)
                                _collect_url_context(ucm)

                                parts = getattr(getattr(cand, 'content', None), 'parts', None) or []
                                for part in parts:
                                    if hasattr(part, 'thought_signature') and part.thought_signature:
                                        signature_parts.append(base64.b64encode(part.thought_signature).decode('utf-8'))
                                    
                                    if hasattr(part, 'thought') and part.thought:
                                        t_text = _extract_gemini_thought_text(part)
                                        if t_text:
                                            thought_accumulated += t_text
                                            pub("thought", t_text)
                                        continue
                                    
                                    if hasattr(part, 'executable_code') and part.executable_code:
                                        c_txt = f"\n```python\n{part.executable_code.code}\n```\n"
                                        full_res += c_txt
                                        pub("content", c_txt)
                                        current_py_id = f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                                        current_py_code = part.executable_code.code
                                        # Reset rather than extend: the stream delivers the
                                        # executable_code part before that turn's inline_data
                                        # images, so only this execution's save-names should be
                                        # eligible for the images that follow it.
                                        pending_sandbox_filenames = _extract_sandbox_image_filenames(part.executable_code.code)
                                        pub("python", {"id": current_py_id, "code": part.executable_code.code})
                                        continue
                                    
                                    if hasattr(part, 'code_execution_result') and part.code_execution_result:
                                        r_txt = f"\n**Output:**\n```\n{part.code_execution_result.output}\n```\n"
                                        full_res += r_txt
                                        pub("content", r_txt)
                                        py_id = current_py_id or f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                                        pub("python", {"id": py_id, "output": part.code_execution_result.output})
                                        py_payload = {"code": current_py_code or "", "output": part.code_execution_result.output}
                                        full_res += f"\n```pyexec\n{json.dumps(py_payload)}\n```\n"
                                        continue

                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                                            img_data = part.inline_data.data
                                            img_data, ext = _prepare_agentic_image_bytes(img_data, mime)
                                            image_digest = hashlib.sha256(img_data).hexdigest()
                                            if image_digest in agentic_image_digests:
                                                continue
                                            agentic_image_digests.add(image_digest)

                                            def _new_agentic_filename():
                                                return (
                                                    f"agentic_{int(time.time() * 1000)}_"
                                                    f"{os.urandom(4).hex()}.{ext}"
                                                )

                                            fn2, saved_agentic_url = _save_user_generated_bytes_verified(
                                                user_id,
                                                img_data,
                                                _new_agentic_filename,
                                                user_config.get('enable_e2ee'),
                                            )
                                            log_force(
                                                f"Agentic image saved: {fn2} ({len(img_data)} bytes)"
                                            )
                                            agentic_saved_urls.append(saved_agentic_url)

                                            # Associate the sandbox filename with the saved URL when
                                            # known so a bare ![alt](result.png) reference in the
                                            # model's final answer is rewritten to the served image.
                                            sandbox_name = None
                                            display_name = getattr(part.inline_data, "display_name", None)
                                            if display_name:
                                                dname = os.path.basename(str(display_name)).strip()
                                                if _sandbox_ref_basename(dname):
                                                    sandbox_name = dname
                                            if not sandbox_name and pending_sandbox_filenames:
                                                sandbox_name = pending_sandbox_filenames.pop(0)
                                            elif sandbox_name and pending_sandbox_filenames:
                                                # A display_name was used; drop a matching pending
                                                # entry so the fallback queue does not drift.
                                                lower = sandbox_name.lower()
                                                if lower in pending_sandbox_filenames:
                                                    pending_sandbox_filenames.remove(lower)
                                            if sandbox_name:
                                                agentic_filename_url_map[sandbox_name.lower()] = saved_agentic_url

                                            img_md = f"\n![Agentic View]({saved_agentic_url})\n"
                                            full_res += img_md
                                            pub("content", img_md)
                                        except Exception as e:
                                            log_force(f"Agentic Vision Image Error: {e}")
                                            # Never leave a silent hole in the answer: tell the
                                            # user the image was generated but could not be kept,
                                            # instead of streaming a URL that points nowhere.
                                            err_md = "\n（※生成画像の保存に失敗したため、画像を表示できません）\n"
                                            full_res += err_md
                                            pub("content", err_md)
                                        continue

                                    if hasattr(part, 'text') and part.text:
                                        t_delta = part.text
                                        rewritten = _rewrite_streamed_sandbox_refs(
                                            t_delta,
                                            sandbox_text_buffer,
                                            agentic_saved_urls,
                                            agentic_consumed_urls,
                                            agentic_filename_url_map,
                                        )
                                        if rewritten:
                                            full_res += rewritten
                                            pub("content", rewritten)
                        
                        # Fallback to chunk.text if parts didn't cover it (unlikely but safe)
                        # but be careful not to double-publish.

                    if sandbox_text_buffer[0]:
                        _tail = sandbox_text_buffer[0]
                        sandbox_text_buffer[0] = ""
                        if _tail:
                            full_res += _tail
                            pub("content", _tail)

                    if agentic_saved_urls or agentic_consumed_urls or agentic_filename_url_map or "sandbox:" in full_res:
                        full_res = _rewrite_sandbox_image_refs(
                            full_res, agentic_saved_urls, agentic_consumed_urls, agentic_filename_url_map
                        )
                        for _consumed_url in list(agentic_consumed_urls):
                            full_res = full_res.replace(
                                f"![Agentic View]({_consumed_url})", ""
                            )

                    if grounding_chunks and (options.get('enable_search') or options.get('enable_maps')):
                        if grounding_supports:
                            full_res = _add_gemini_citations(full_res, grounding_supports, grounding_chunks)
                        sources_lines = []
                        has_sources = False
                        for i, chunk in enumerate(grounding_chunks):
                            title, uri = _chunk_grounding_info(chunk)
                            if title or uri:
                                has_sources = True
                            if uri:
                                label = title or uri
                                sources_lines.append(f"- [{i + 1}] [{label}]({uri})")
                            elif title:
                                sources_lines.append(f"- [{i + 1}] {title}")
                            else:
                                sources_lines.append(f"- [{i + 1}] (source unavailable)")
                        if has_sources:
                            sources_text = "\n\n**Sources:**\n" + "\n".join(sources_lines) + "\n"
                            full_res += sources_text
                            pub("content", sources_text)

                    if url_context_chunks and (options.get('enable_url_context') or auto_enable_url_context):
                        url_sources = []
                        has_url_sources = False
                        for i, chunk in enumerate(url_context_chunks):
                            uri = None
                            status = None
                            if isinstance(chunk, dict):
                                uri = chunk.get('retrieved_url') or chunk.get('retrievedUrl')
                                status = chunk.get('url_retrieval_status') or chunk.get('urlRetrievalStatus')
                            else:
                                uri = getattr(chunk, 'retrieved_url', None) or getattr(chunk, 'retrievedUrl', None)
                                status = getattr(chunk, 'url_retrieval_status', None) or getattr(chunk, 'urlRetrievalStatus', None)
                            if uri:
                                has_url_sources = True
                                st_str = f" ({status})" if status and str(status) != "ACTIVE" else ""
                                url_sources.append(f"- [{uri}]({uri}){st_str}")
                        if has_url_sources:
                            url_sources_text = "\n\n**URL Context:**\n" + "\n".join(url_sources) + "\n"
                            full_res += url_sources_text
                            pub("content", url_sources_text)

                    if gemini_local_python and options.get('enable_python'):
                        try:
                            def _extract_exec_blocks(text):
                                blocks = []
                                if not text:
                                    return blocks
                                for m in re.finditer(r"```python\\s*\\n(.*?)```", text, flags=re.S|re.I):
                                    code = m.group(1) or ""
                                    lines = code.splitlines()
                                    marker_idx = None
                                    for i, line in enumerate(lines):
                                        if not line.strip():
                                            continue
                                        if line.strip().upper() in ("# EXECUTE", "#EXECUTE", "# EXEC"):
                                            marker_idx = i
                                        break
                                    if marker_idx is None:
                                        continue
                                    run_code = "\n".join(lines[marker_idx + 1:]).strip()
                                    if run_code:
                                        blocks.append(run_code)
                                return blocks

                            exec_blocks = _extract_exec_blocks(full_res)
                            for b in exec_blocks:
                                result = safe_execute_python(b)
                                out_txt = f"\\n**Output:**\\n```\\n{result}\\n```\\n"
                                full_res += out_txt
                                pub("content", out_txt)
                                pub("python", {"id": f"gem_local_py_{int(time.time()*1000)}_{os.urandom(3).hex()}", "code": b, "output": result})
                        except Exception as e:
                            log_force(f"Gemini local python failed: {e}")

                    if not full_res.strip() and last_file_tool_error:
                        full_res = last_file_tool_error
                        pub("content", full_res)

            # --- 1.5 Grok Imagine Image Generation ---
            elif model_key in (
                "grok-imagine-image-2.0",
                "grok-imagine-image",
                "grok-imagine-image-pro",
                "grok-imagine-image-quality",
            ):
                log_force("Routing: Grok Imagine Branch")
                try:
                    pub("status", "画像を生成中...")
                    
                    aspect_ratio = options.get('grok_image_aspect') or "1:1"
                    resolution = str(options.get('grok_image_resolution') or "1k").lower().strip()
                    if resolution not in ("1k", "2k"):
                        resolution = "1k"
                    quality = str(options.get('grok_image_quality') or "medium").lower().strip()
                    if quality not in ("low", "medium"):
                        quality = "medium"
                    image_count = 1
                    try:
                        image_count = max(1, min(10, int(options.get('grok_image_count') or 1)))
                    except (TypeError, ValueError):
                        pass
                    img_response_format = str(options.get('grok_image_format') or "b64_json").lower().strip()
                    if img_response_format not in ("url", "b64_json"):
                        img_response_format = "b64_json"
                    grok_supports_resolution = model_key in (
                        "grok-imagine-image-2.0",
                        "grok-imagine-image-quality",
                    )
                    grok_supports_quality = model_key == "grok-imagine-image-2.0"
                    grok_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
                    
                    img_kwargs = {
                        "model": model_key,
                        "prompt": grok_prompt,
                        "n": image_count,
                        "response_format": img_response_format
                    }
                    # aspect_ratio / resolution / quality are xAI-specific; pass via extra_body
                    eb = {}
                    if aspect_ratio:
                        eb["aspect_ratio"] = aspect_ratio
                    if grok_supports_resolution:
                        eb["resolution"] = resolution
                    if grok_supports_quality:
                        eb["quality"] = quality

                    img_inputs = []
                    for fi in loaded_files:
                        if not fi.get('bytes') or not fi.get('mime', '').startswith('image/'):
                            continue
                        img_bytes = fi['bytes']
                        img_mime = fi['mime']
                        # xAI supports jpg/jpeg or png.
                        if img_mime not in ('image/png', 'image/jpeg'):
                            try:
                                im = Image.open(BytesIO(img_bytes))
                                if im.mode not in ('RGB', 'RGBA'):
                                    im = im.convert('RGB')
                                out = BytesIO()
                                im.save(out, format='PNG')
                                img_bytes = out.getvalue()
                                img_mime = 'image/png'
                            except Exception:
                                pass
                        img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"input_{len(img_inputs)}")
                        img_inputs.append((img_name, img_bytes, img_mime))
                    for hp in history_image_parts:
                        ref = hp.get("ref")
                        if any(existing[0] == os.path.basename(ref or "") for existing in img_inputs):
                            continue
                        img_inputs.append((hp['name'], hp['bytes'], hp['mime']))

                    img_data_b64 = None
                    if img_inputs:
                        # xAI image edits expect JSON (not multipart), so send base64.
                        # Multi-image editing accepts up to 3 references via `images`.
                        image_payloads = []
                        for img_entry in img_inputs[:3]:
                            img_bytes = img_entry[1]
                            img_mime = img_entry[2] if len(img_entry) > 2 else "image/png"
                            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                            image_payloads.append({
                                "url": f"data:{img_mime};base64,{img_b64}",
                                "type": "image_url",
                            })
                        endpoint = f"https://{_XAI_API_HOST}/v1/images/edits"
                        headers = {
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                        payload = {
                            "model": model_key,
                            "prompt": grok_prompt,
                            "n": image_count,
                            "response_format": img_response_format
                        }
                        if len(image_payloads) == 1:
                            payload["image"] = image_payloads[0]
                        else:
                            payload["images"] = image_payloads
                        if aspect_ratio:
                            payload["aspect_ratio"] = aspect_ratio
                        if grok_supports_resolution:
                            payload["resolution"] = resolution
                        if grok_supports_quality:
                            payload["quality"] = quality
                        _mark_provider_request_started()
                        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=120)
                        if resp.status_code >= 400:
                            try:
                                log_force(f"Grok Imagine edit error {resp.status_code}: {resp.text}")
                            except Exception:
                                pass
                        resp.raise_for_status()
                        resp_json = resp.json()
                        image_results = []
                        if isinstance(resp_json, dict):
                            image_results = [item or {} for item in (resp_json.get("data") or [])]
                            if image_results:
                                img_data_b64 = image_results[0].get("b64_json")
                            if not img_data_b64 and resp_json.get("image"):
                                img_data_b64 = resp_json.get("image")
                    else:
                        _mark_provider_request_started()
                        resp = o_client.images.generate(**img_kwargs, extra_body=eb)
                        image_results = []
                        for item in (getattr(resp, "data", None) or []):
                            image_results.append({
                                "b64_json": getattr(item, "b64_json", None),
                                "url": getattr(item, "url", None),
                            })
                        if image_results:
                            img_data_b64 = image_results[0].get("b64_json")

                    if img_data_b64 and not image_results:
                        image_results = [{"b64_json": img_data_b64}]
                    saved_any = False
                    for image_index, image_result in enumerate(image_results[:image_count]):
                        image_b64 = image_result.get("b64_json")
                        image_url = image_result.get("url")
                        if image_b64:
                            img_bytes = _decode_base64_limited(image_b64, 50 * 1024 * 1024)
                        elif image_url:
                            img_bytes = _download_public_https_bytes(image_url, 50 * 1024 * 1024, timeout=120.0)
                        else:
                            continue
                        ext = "png"
                        fn2 = f"gen_grok_{int(time.time())}_{len(generated_images)}_{image_index}.{ext}"
                        _save_user_generated_bytes(
                            user_id, img_bytes, fn2, user_config.get('enable_e2ee')
                        )
                            
                        generated_images.append(f"{user_id}/{fn2}")
                        pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                        full_res += f"Generated Image {image_index + 1} for: {final_message_text}\n"
                        saved_any = True
                    if not saved_any:
                        pub("error", "Grok Image Gen Error: No data returned.")
                except Exception as e:
                    logger.exception("Grok Imagine Error")
                    err_body = ""
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        err_body = e.response.text
                    elif hasattr(e, 'body'): # OpenAI SDK errors
                        err_body = str(e.body)
                    
                    err_msg = str(e)
                    if "content moderation" in err_msg.lower() or "content moderation" in err_body.lower():
                        err_msg = "不適切な内容が含まれている可能性があるため、xAIの安全フィルタにより画像生成が拒否されました。プロンプトをより一般的な表現に変更して、再度お試しください。"
                    elif err_body:
                        err_msg = f"{err_msg} - {err_body}"
                    
                    pub("error", f"Grok Imagine Error: {err_msg}")

            # --- 1.6 Grok Imagine Video Generation ---
            elif model_key in ("grok-imagine-video", "grok-imagine-video-1.5"):
                log_force("Routing: Grok Video Branch")
                try:
                    pub("content", "**Generating Video (Grok)...**\n")
                    
                    # Prepare params
                    duration = None
                    try:
                        duration = int(options.get('grok_video_duration') or 5)
                    except: duration = 5
                    
                    aspect_ratio = options.get('grok_video_aspect') or "16:9"
                    resolution = options.get('grok_video_resolution') or "720p"
                    if resolution not in ("480p", "720p", "1080p"):
                        resolution = "720p"
                    if resolution == "1080p" and model_key != "grok-imagine-video-1.5":
                        resolution = "720p"
                    
                    api_key = key # Decrypted XAI API Key
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Determine endpoint and payload
                    endpoint = f"https://{_XAI_API_HOST}/v1/videos/generations"
                    payload = {
                        "model": model_key,
                        "prompt": final_message_text,
                        "duration": duration,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution
                    }
                    
                    # Check for image or video inputs
                    img_urls = []
                    vid_urls = []
                    for fi in loaded_files:
                        if fi.get('bytes'):
                            # For simplicity, if we have local bytes, we might need to upload them to a public URL 
                            # or use data URIs if supported. The docs say:
                            # "Note: The input video URL must be a direct, publicly accessible link to the video file."
                            # This is a limitation for local files.
                            # However, for Image-to-Video, the docs show:
                            # image: { url: '<url of the image>' }
                            # But also curl example shows "image": {"url": "<url of the image>"}
                            # Wait, can we use base64? 
                            # image-gen docs showed base64 support for Image.
                            # Let's try base64 for image-to-video.
                            mime = fi.get('mime', 'image/png')
                            if mime.startswith('image/'):
                                b64 = base64.b64encode(fi['bytes']).decode('utf-8')
                                payload["image"] = {"url": f"data:{mime};base64,{b64}"}
                                try:
                                    im = Image.open(BytesIO(fi['bytes']))
                                    inferred = _closest_aspect_ratio(im.width, im.height, {"16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3"})
                                    if inferred:
                                        payload["aspect_ratio"] = inferred
                                except Exception:
                                    pass
                            elif mime.startswith('video/'):
                                # Video edit requires a public URL. Local files won't work easily here.
                                # But we'll try to provide it if we had a public URL.
                                pass

                    # Send request
                    _mark_provider_request_started()
                    resp = httpx.post(endpoint, headers=headers, json=payload, timeout=60.0)
                    if resp.status_code != 200:
                        raise RuntimeError(f"xAI API Error: {resp.status_code} - {resp.text}")
                    
                    data = resp.json()
                    request_id = data.get("request_id")
                    if not request_id:
                        raise RuntimeError(f"No request_id returned: {data}")
                    
                    pub("content", f"Request ID: `{request_id}`. Polling for result...\n")
                    
                    # Polling
                    poll_url = f"https://{_XAI_API_HOST}/v1/videos/{request_id}"
                    max_polls = 300 # 10 minutes if 2s interval
                    video_url = None
                    for i in range(max_polls):
                        if check_stop(): break
                        time.sleep(2)
                        _mark_provider_request_started()
                        p_resp = httpx.get(poll_url, headers=headers, timeout=30.0)
                        if p_resp.status_code == 200:
                            p_data = p_resp.json()
                            status = p_data.get("status")
                            # xAI Video API might return URL nested inside "video" object
                            video_url = p_data.get("url")
                            if not video_url and isinstance(p_data.get("video"), dict):
                                video_url = p_data["video"].get("url")
                            
                            if status == "completed" or video_url:
                                break
                            elif status == "failed":
                                raise RuntimeError(f"Video generation failed: {p_data.get('error')}")
                            else:
                                if i % 5 == 0: # Log every 10s
                                    log_force(f"Polling video {request_id}: status={status}, has_url={bool(video_url)}")
                        elif p_resp.status_code != 200:
                            log_force(f"Polling error {p_resp.status_code}: {p_resp.text}")
                    
                    if video_url:
                        # Download and save the video locally
                        _mark_provider_request_started()
                        try:
                            video_bytes = _download_public_https_bytes(video_url, 128 * 1024 * 1024, timeout=60.0)
                        except Exception as download_error:
                            video_bytes = None
                            log_force(f"Grok video download rejected: {download_error}")
                        if video_bytes:
                            fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
                            _save_user_generated_bytes(
                                user_id, video_bytes, fn2, user_config.get('enable_e2ee')
                            )
                                
                            generated_images.append(f"{user_id}/{fn2}")
                            vid_tag = f'\n<video controls playsinline preload="metadata" src="/files/{user_id}/{fn2}" class="w-full mt-2"></video>\n'
                            pub("content", vid_tag)
                            full_res += f"Generated Video for: {final_message_text}\n"
                        else:
                            pub("error", "Failed to download generated video safely.")
                    else:
                        pub("error", "Video generation timed out or was canceled.")
                        
                except Exception as e:
                    logger.exception("Grok Imagine Video Error")
                    pub("error", f"Grok Imagine Video Error: {str(e)}")

            # --- 1.5 Anthropic Claude ---
            elif is_claude:
                log_force("Routing: Claude Branch")
                try:
                    claude_messages = []
                    # Convert history
                    for m in history:
                        role = 'assistant' if m['role'] == 'assistant' else 'user'
                        content = m['content'] or ""
                        
                        msg_parts = []
                        if content:
                            msg_parts.append({"type": "text", "text": content})
                        
                        if m.get('image_url'):
                            msg_images, _ = _load_message_history_images(m.get('image_url'), include_only_images=True)
                            for img in msg_images:
                                msg_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": img['mime'],
                                        "data": base64.b64encode(img['bytes']).decode('utf-8')
                                    }
                                })
                        if msg_parts:
                            claude_messages.append({"role": role, "content": msg_parts})

                    # Current message
                    curr_parts = []
                    if final_message_text and str(final_message_text).strip():
                        curr_parts.append({"type": "text", "text": final_message_text})
                    for fi in loaded_files:
                        if fi.get('bytes'):
                            if fi.get('mime', '').startswith('image/'):
                                curr_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": fi['mime'],
                                        "data": base64.b64encode(fi['bytes']).decode('utf-8')
                                    }
                                })
                            elif fi.get('is_pdf') or fi.get('mime') == 'application/pdf':
                                curr_parts.append({
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": base64.b64encode(fi['bytes']).decode('utf-8')
                                    }
                                })
                    claude_messages.append({"role": "user", "content": curr_parts})

                    # Claude Parameters
                    claude_kwargs = {
                        "model": model_key,
                        "messages": claude_messages,
                        "max_tokens": 8192,
                    }
                    sys_prompt_claude = options.get('system_prompt')
                    if options.get('enable_prompt_caching'):
                        # Automatic prompt caching (Claude): top-level cache_control
                        claude_kwargs["cache_control"] = {"type": "ephemeral"}
                        if sys_prompt_claude:
                            claude_kwargs["system"] = [{
                                "type": "text",
                                "text": sys_prompt_claude,
                                "cache_control": {"type": "ephemeral"},
                            }]
                        log_force("Claude Prompt Caching enabled")
                    elif sys_prompt_claude:
                        claude_kwargs["system"] = sys_prompt_claude

                    if options.get('enable_thinking'):
                        budget = 0
                        try:
                            budget = int(options.get('thinking_budget') or 4096)
                        except: budget = 4096
                        if budget < 1024: budget = 1024
                        claude_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                        claude_kwargs["max_tokens"] = budget + 4096

                    # MCP外部ツール（Claude）の付与
                    claude_mcp_env = None
                    claude_mcp_tools = []
                    try:
                        claude_mcp_env = _ensure_mcp_env()
                    except Exception:
                        claude_mcp_env = None
                    if claude_mcp_env is not None:
                        try:
                            claude_mcp_tools = claude_mcp_env.serialize_anthropic()
                        except Exception:
                            claude_mcp_tools = []
                    if claude_mcp_tools:
                        claude_kwargs["tools"] = claude_mcp_tools
                        log_force(f"Claude MCP tools attached: {len(claude_mcp_tools)}")

                    claude_max_rounds = 20 if claude_mcp_tools else 1
                    for _claude_round in range(claude_max_rounds):
                        _mark_provider_request_started()
                        _stopped_claude = False
                        with c_client.messages.stream(**claude_kwargs) as stream:
                            for event in stream:
                                if check_stop():
                                    _stopped_claude = True
                                    break
                                if event.type == "content_block_start":
                                    if event.content_block.type == "thinking":
                                        pub("thought_start", "")
                                elif event.type == "content_block_delta":
                                    if event.delta.type == "thinking_delta":
                                        thought = event.delta.thinking
                                        thought_accumulated += thought
                                        pub("thought", thought)
                                    elif event.delta.type == "text_delta":
                                        txt = event.delta.text
                                        full_res += txt
                                        pub("content", txt)
                                elif event.type == "content_block_stop":
                                    pass
                            # After stream, finalize thoughts if any
                            if thought_accumulated:
                                pub("thought_stop", "")
                            final_message = None
                            if claude_mcp_tools and not _stopped_claude:
                                try:
                                    final_message = stream.get_final_message()
                                except Exception:
                                    final_message = None
                        if _stopped_claude or not claude_mcp_tools or final_message is None:
                            break
                        stop_reason = getattr(final_message, "stop_reason", None)
                        if stop_reason != "tool_use":
                            break
                        content_blocks = list(getattr(final_message, "content", None) or [])
                        tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]
                        if not tool_uses:
                            break
                        # アシスタント履歴へ追加（thinkingブロックはAPIへ送らない）
                        history_content = []
                        for b in content_blocks:
                            bt = getattr(b, "type", None)
                            if bt == "text":
                                history_content.append({"type": "text", "text": getattr(b, "text", "") or ""})
                            elif bt == "tool_use":
                                history_content.append({
                                    "type": "tool_use",
                                    "id": getattr(b, "id", "") or "",
                                    "name": getattr(b, "name", "") or "",
                                    "input": getattr(b, "input", None) or {},
                                })
                        claude_messages.append({"role": "assistant", "content": history_content})
                        tool_result_blocks = []
                        for tu in tool_uses:
                            tu_id = getattr(tu, "id", "") or ""
                            tu_name = getattr(tu, "name", "") or ""
                            if not tu_id or not str(tu_name).startswith("mcp__"):
                                tool_result_blocks.append({
                                    "type": "tool_result", "tool_use_id": tu_id,
                                    "content": "Error: unknown MCP tool",
                                    "is_error": True,
                                })
                                continue
                            try:
                                _args_in = getattr(tu, "input", None) or {}
                                mcp_txt, mcp_out = claude_mcp_env.execute(tu_name, _args_in)
                                if mcp_out.get("ok"):
                                    _md = f"\n\n> 🔧 **MCPツール実行:** `{tu_name}` を実行しました。\n"
                                    full_res += _md
                                    pub("content", _md)
                                elif mcp_out.get("rejected"):
                                    _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{tu_name}`\n"
                                    full_res += _md
                                    pub("content", _md)
                                _result_content = mcp_txt if str(mcp_txt).strip() else "Tool executed."
                                tool_result_blocks.append({
                                    "type": "tool_result", "tool_use_id": tu_id,
                                    "content": _result_content,
                                })
                            except Exception as _te:
                                tool_result_blocks.append({
                                    "type": "tool_result", "tool_use_id": tu_id,
                                    "content": f"Error executing MCP tool: {_te}",
                                    "is_error": True,
                                })
                        claude_messages.append({"role": "user", "content": tool_result_blocks})
                        claude_kwargs["messages"] = claude_messages

                    if not full_res and not thought_accumulated:
                        pub("error", "Claude returned empty response.")

                except Exception as e:
                    logger.exception("Claude Branch Error")
                    pub("error", f"Claude Error: {str(e)}")

            # --- 2. xAI Grok (Native SDK) ---
            elif is_grok and x_client and not options.get('enable_python') and _ensure_mcp_env() is None:
                log_force("Routing: Grok Branch (Native SDK)")
                if options.get('enable_thinking') and not grok_reasoning_supported:
                    # Grok non-reasoning models should not emit thought events (avoids UI thought box).
                    log_force("Grok non-reasoning: skip thought stream")
                search_params = None
                tools = []
                include = []
                if grok_enable_search:
                    try:
                        tools = [x_web_search(), x_x_search()]
                        include = ["verbose_streaming", "inline_citations"]
                        log_force("Enabled Grok Search Tools (Web + X)")
                    except Exception as e:
                        log_force(f"Grok Search Tools Config Error: {e}")
                        try:
                            search_params = SearchParameters(
                                sources=[web_source(), x_source()],
                                mode="on",
                                return_citations=True
                            )
                            log_force("Enabled Grok Search (Legacy SearchParameters)")
                        except Exception as e2:
                            log_force(f"Grok Search Config Error (Legacy): {e2}")

                create_kwargs = {"model": model_key}
                if search_params: create_kwargs["search_parameters"] = search_params
                if tools: create_kwargs["tools"] = tools
                if include: create_kwargs["include"] = include
                forced_grok_reasoning = ("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l)
                if (options.get('enable_thinking') or forced_grok_reasoning) and grok_reasoning_effort_supported:
                    grok_effort = _grok_reasoning_effort()
                    if grok_effort:
                        create_kwargs["reasoning_effort"] = grok_effort
                elif (options.get('enable_thinking') or forced_grok_reasoning) and grok_reasoning_supported:
                    log_force("Grok reasoning_effort not supported for this model; skipping parameter")
                def _optional_float(option_key, minimum, maximum):
                    raw_value = options.get(option_key)
                    if raw_value is None or str(raw_value).strip() == "":
                        return None
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        return None
                    return max(minimum, min(maximum, value))
                def _optional_int(option_key, minimum=None, maximum=None):
                    raw_value = options.get(option_key)
                    if raw_value is None or str(raw_value).strip() == "":
                        return None
                    try:
                        value = int(raw_value)
                    except (TypeError, ValueError):
                        return None
                    if minimum is not None: value = max(minimum, value)
                    if maximum is not None: value = min(maximum, value)
                    return value
                xai_sampling = {
                    "temperature": _optional_float("xai_temperature", 0.0, 2.0),
                    "top_p": _optional_float("xai_top_p", 0.0, 1.0),
                    "frequency_penalty": _optional_float("xai_frequency_penalty", -2.0, 2.0),
                    "presence_penalty": _optional_float("xai_presence_penalty", -2.0, 2.0),
                    "seed": _optional_int("xai_seed"),
                    # xAI's current Chat API calls this max_completion_tokens; the
                    # installed native SDK serializes the equivalent max_tokens field.
                    "max_tokens": _optional_int("xai_max_completion_tokens", 1),
                }
                for option_key, option_value in xai_sampling.items():
                    if option_value is not None:
                        create_kwargs[option_key] = option_value
                stop_raw = str(options.get("xai_stop") or "").strip()
                if stop_raw:
                    create_kwargs["stop"] = [part.strip() for part in stop_raw.split(",") if part.strip()][:4]
                response_format = str(options.get("xai_response_format") or "").strip()
                if response_format in ("text", "json_object") and response_format != "text":
                    create_kwargs["response_format"] = response_format
                tool_choice = str(options.get("xai_tool_choice") or "auto").strip().lower()
                if tool_choice in ("auto", "none", "required") and (tools or tool_choice != "required"):
                    create_kwargs["tool_choice"] = tool_choice
                if tools:
                    create_kwargs["parallel_tool_calls"] = bool(options.get("xai_parallel_tool_calls", True))
                if not model_key_l.startswith("grok-4.20"):
                    if options.get("xai_logprobs"):
                        create_kwargs["logprobs"] = True
                    top_logprobs = _optional_int("xai_top_logprobs", 0, 8)
                    if top_logprobs is not None and options.get("xai_logprobs"):
                        create_kwargs["top_logprobs"] = top_logprobs
                create_kwargs["use_encrypted_content"] = True # Request encrypted reasoning if available
                if options.get('enable_python') and XAI_SDK_AVAILABLE:
                    create_kwargs["tools"] = [x_code_execution()]
                if options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
                    create_kwargs["conversation_id"] = options.get('prompt_cache_key')
                    log_force(f"Grok Prompt Caching conversation_id={options.get('prompt_cache_key')}")

                _mark_provider_request_started()
                chat_session = x_client.chat.create(**create_kwargs)

                grok_sys = _grok_system_prompt(options.get('system_prompt'), grok_enable_search)
                if grok_sys: chat_session.append(x_system(grok_sys))
                
                history_img_seen = set()
                history_img_bytes = 0
                for m in history:
                    if m['role'] in ('user', 'assistant'):
                        content_parts = [m['content']]
                        if m.get('image_url'):
                            try:
                                msg_images, history_img_bytes = _load_message_history_images(
                                    m.get('image_url'),
                                    seen=history_img_seen,
                                    total_bytes=history_img_bytes,
                                    include_only_images=True
                                )
                                for msg_img in msg_images:
                                    d_uri = f"data:{msg_img['mime']};base64,{base64.b64encode(msg_img['bytes']).decode('utf-8')}"
                                    content_parts.append(x_image(d_uri))
                            except: pass
                        if m['role'] == 'user':
                            chat_session.append(x_user(*content_parts))
                        else:
                            chat_session.append(x_assistant(*content_parts))
                    else:
                        chat_session.append(x_user(m['content']) if m['role'] == 'user' else x_assistant(m['content']))
                
                curr_user_content = []
                if final_message_text and str(final_message_text).strip():
                    curr_user_content.append(final_message_text)
                current_image_names = []
                for fi in loaded_files:
                    if fi.get('text'):
                        file_text_block = f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
                        if curr_user_content:
                            curr_user_content[0] += file_text_block
                        else:
                            curr_user_content.append(file_text_block.strip())
                    elif fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                        d_uri = f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"
                        curr_user_content.append(x_image(d_uri))
                        img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
                        current_image_names.append(os.path.basename(str(img_label)))
                name_block = _build_attachment_name_block(current_image_names)
                if name_block:
                    if curr_user_content:
                        curr_user_content[0] += f"\n\n{name_block}"
                    else:
                        curr_user_content.append(name_block)
                
                chat_session.append(x_user(*curr_user_content))
                
                _mark_provider_request_started()
                stream = chat_session.stream()
                search_reported = False
                last_response = None
                if grok_reasoning_supported:
                    # Ensure thought box is created even if Grok doesn't stream reasoning text.
                    pub("thought", " ")
                for resp, chunk in stream:
                    _latency_mark_once(job_id, "provider_first_chunk_ms")
                    last_response = resp
                    if check_stop(): break
                    tool_calls = getattr(chunk, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_type = getattr(tc, 'type', None)
                            tc_fn = getattr(getattr(tc, 'function', None), 'name', None)
                            tc_type_str = str(tc_type) if tc_type is not None else ""
                            if (tc_fn and "search" in tc_fn.lower()) or ("SEARCH" in tc_type_str):
                                if not search_reported:
                                    pub("search_status", "searching")
                                    search_reported = True
                                break
                    r_content = getattr(chunk, 'reasoning_content', None)
                    if r_content:
                        thought_accumulated += r_content
                        if grok_reasoning_supported:
                            pub("thought", r_content)
                    
                    # Also log encrypted content presence for debugging
                    if getattr(chunk, 'encrypted_content', None):
                         log_force("Received encrypted reasoning content")

                    c_content = getattr(chunk, 'content', None)
                    if c_content:
                        full_res += c_content
                        pub("content", c_content)
                if search_reported:
                    pub("search_status", "done")
                if last_response and getattr(last_response, 'citations', None):
                    citations_text = "\n\n**Sources:**\n"
                    inline_citations = getattr(last_response, 'inline_citations', None)
                    if inline_citations:
                        for c in inline_citations:
                            cid = getattr(c, 'id', None)
                            web_cit = getattr(c, 'web_citation', None)
                            url = None
                            title = None
                            if web_cit:
                                url = getattr(web_cit, 'url', None)
                                title = getattr(web_cit, 'title', None)
                            if url:
                                label = title or url
                                if cid is not None:
                                    citations_text += f"- [{cid}] {label} ({url})\n"
                                else:
                                    citations_text += f"- {label} ({url})\n"
                    else:
                        for c in last_response.citations:
                            if hasattr(c, 'url'):
                                url = c.url
                                title = getattr(c, 'title', None)
                            else:
                                url = str(c)
                                title = None
                            label = title or url
                            citations_text += f"- {label}\n"
                    full_res += citations_text
                    pub("content", citations_text)

            # --- 2.5 TTS Branch ---
            elif 'tts' in model_key:
                log_force("Routing: TTS Branch")
                try:
                    pub("content", "**Processing Audio Generation...**\n")
                    
                    speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.mp3"
                    audio_content = None

                    if 'google-tts' in model_key:
                        # Google Cloud TTS (requires Google Cloud API key, not Gemini API key)
                        g_key = model_api_key_override or decrypt_val(user.google_api_key)
                        if not g_key and _admin_env_fallback_enabled(user):
                            g_key = os.getenv('GOOGLE_API_KEY')
                        if not g_key:
                            raise RuntimeError("Google API Key is not configured for Google TTS.")
                        g_project = decrypt_val(user.google_cloud_project)
                        if not g_project and _admin_env_fallback_enabled(user):
                            g_project = os.getenv('GOOGLE_CLOUD_PROJECT')
                        opts = {"api_key": g_key}
                        if g_project: opts["quota_project_id"] = g_project
                        client_tts = texttospeech.TextToSpeechClient(
                            client_options=ClientOptions(**opts)
                        )
                        synthesis_input = texttospeech.SynthesisInput(text=final_message_text)
                        tts_lang = (options.get('tts_language') or "ja-JP").strip() or "ja-JP"
                        tts_voice_custom = (options.get('tts_voice_custom') or "").strip()
                        
                        # Selection logic
                        if tts_voice_custom:
                            voice = texttospeech.VoiceSelectionParams(language_code=tts_lang, name=tts_voice_custom)
                        elif 'studio' in model_key:
                            voice = pick_tts_voice(client_tts, tts_lang, "studio")
                        else:
                            voice = pick_tts_voice(client_tts, tts_lang, "neural")
                        
                        speed_val = clamp_float(options.get('tts_speed'), 0.25, 2.0)
                        audio_kwargs = {"audio_encoding": texttospeech.AudioEncoding.MP3}
                        if speed_val is not None:
                            audio_kwargs["speaking_rate"] = speed_val
                        audio_config = texttospeech.AudioConfig(**audio_kwargs)
                        response_tts = client_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                        audio_content = response_tts.audio_content

                    elif 'grok-tts' in model_key or 'xai-tts' in model_key:
                        # xAI standalone TTS (単独モデル化)
                        xai_key = model_api_key_override or decrypt_val(user.xai_api_key)
                        if not xai_key and _admin_env_fallback_enabled(user):
                            xai_key = os.getenv('XAI_API_KEY')
                        if not xai_key:
                            raise RuntimeError("xAI API Key is not configured for Grok TTS.")

                        raw_voice = (options.get('tts_voice') or "eve").strip()
                        # xAI voices are case-insensitive; use title case as per docs
                        xai_voice = raw_voice.capitalize()
                        if xai_voice not in ("Eve", "Ara", "Rex", "Sal", "Leo"):
                            xai_voice = "Eve"

                        tts_lang = (options.get('tts_language') or "ja").strip() or "ja"
                        speed_val = clamp_float(options.get('tts_speed'), 0.7, 1.5)

                        payload = {
                            "text": final_message_text,
                            "voice_id": xai_voice,
                            "language": tts_lang,
                        }
                        if speed_val is not None:
                            payload["speed"] = speed_val

                        headers = {
                            "Authorization": f"Bearer {xai_key}",
                            "Content-Type": "application/json"
                        }
                        tts_url = f"https://{_XAI_API_HOST}/v1/tts"
                        resp = requests.post(tts_url, headers=headers, json=payload, timeout=180, stream=True)
                        resp.raise_for_status()
                        audio_buf = bytearray()
                        for audio_chunk in resp.iter_content(chunk_size=64 * 1024):
                            if not audio_chunk:
                                continue
                            audio_buf.extend(audio_chunk)
                            if len(audio_buf) > _AUDIO_INPUT_MAX_BYTES:
                                raise ValueError("Generated audio is too large")
                        audio_content = bytes(audio_buf)

                    else:
                        # OpenAI TTS (default for generic tts models)
                        tts_voice = (options.get('tts_voice') or "alloy").strip().lower() or "alloy"
                        speed_val = clamp_float(options.get('tts_speed'), 0.25, 4.0)
                        tts_kwargs = {
                            "model": model_key,
                            "voice": tts_voice,
                            "input": final_message_text
                        }
                        if speed_val is not None:
                            tts_kwargs["speed"] = speed_val

                        _mark_provider_request_started()
                        with o_client.audio.speech.with_streaming_response.create(**tts_kwargs) as response:
                            audio_buf = bytearray()
                            for audio_chunk in response.iter_bytes(chunk_size=64 * 1024):
                                audio_buf.extend(audio_chunk)
                                if len(audio_buf) > _AUDIO_INPUT_MAX_BYTES:
                                    raise ValueError("Generated audio is too large")
                            audio_content = bytes(audio_buf)

                    _save_user_generated_bytes(
                        user_id, audio_content, speech_file_name, user_config.get('enable_e2ee')
                    )
                    
                    audio_url = f"/files/{user_id}/{speech_file_name}"
                    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
                    
                    full_res += audio_tag
                    pub("content", audio_tag)
                    generated_images.append(f"{user_id}/{speech_file_name}")

                except Exception as e:
                    pub("error", f"TTS Error: {str(e)}")

            # --- 3. GPT Image Branch ---
            elif 'gpt-image' in model_key:
                log_force("Routing: GPT Image Branch")
                try:
                    pub("status", "画像生成の準備中...")
                    # GPT Image models always return base64; response_format is not supported for them.
                    # Use a dedicated timeout/retry so image generation can be slower without timing out.
                    img_client = o_client.with_options(
                        timeout=_OPENAI_IMAGE_TIMEOUT_SECONDS,
                        max_retries=_OPENAI_IMAGE_MAX_RETRIES
                    )
                    def _pick_image_opt(val, allowed):
                        if val is None:
                            return None
                        v = str(val).strip()
                        return v if v in allowed else None
                    size_opt = _pick_image_opt(options.get('image_size'), {"auto", "1024x1024", "1536x1024", "1024x1536"}) or _OPENAI_IMAGE_DEFAULT_SIZE
                    quality_opt = _pick_image_opt(options.get('image_quality'), {"auto", "low", "medium", "high"}) or _OPENAI_IMAGE_DEFAULT_QUALITY
                    format_opt = _pick_image_opt(options.get('image_format'), {"png", "jpeg", "webp"}) or _OPENAI_IMAGE_OUTPUT_FORMAT
                    comp_opt = None
                    try:
                        comp_opt = int(options.get('image_compression')) if options.get('image_compression') is not None else None
                    except Exception:
                        comp_opt = None
                    if comp_opt is not None and (comp_opt < 0 or comp_opt > 100):
                        comp_opt = None
                    
                    pub("status", "プロンプトとコンテキストを構成中...")
                    img_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
                    if options.get('system_prompt'):
                        img_prompt = f"{options.get('system_prompt')}\n\n{img_prompt}"
                    img_kwargs = {"model": model_key, "prompt": img_prompt}
                    if size_opt:
                        img_kwargs["size"] = size_opt
                    if quality_opt:
                        img_kwargs["quality"] = quality_opt
                    if format_opt:
                        img_kwargs["output_format"] = format_opt
                        if format_opt in {"jpeg", "webp"}:
                            img_kwargs["output_compression"] = comp_opt if comp_opt is not None else _OPENAI_IMAGE_OUTPUT_COMPRESSION
                    
                    pub("status", "入力画像を処理中...")
                    img_inputs = []
                    for fi in loaded_files:
                        if not fi.get('bytes') or not fi.get('mime', '').startswith('image/'):
                            continue
                        img_bytes = fi['bytes']
                        img_mime = fi['mime']
                        if img_mime not in ('image/png', 'image/jpeg', 'image/webp'):
                            try:
                                im = Image.open(BytesIO(img_bytes))
                                if im.mode not in ('RGB', 'RGBA'):
                                    im = im.convert('RGB')
                                out = BytesIO()
                                im.save(out, format='PNG')
                                img_bytes = out.getvalue()
                                img_mime = 'image/png'
                            except Exception:
                                pass
                        img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"input_{len(img_inputs)}")
                        img_inputs.append((img_name, img_bytes, img_mime))
                    existing_input_names = {item[0] for item in img_inputs}
                    for hp in history_image_parts:
                        if hp['name'] in existing_input_names:
                            continue
                        img_inputs.append((hp['name'], hp['bytes'], hp['mime']))
                        existing_input_names.add(hp['name'])
                    mask_file = None
                    mask_name = options.get('image_mask')
                    if mask_name:
                        pub("status", "マスク画像を処理中...")
                        if not img_inputs:
                            raise RuntimeError("Mask requires at least one input image.")
                        norm = os.path.normpath(mask_name)
                        if norm.startswith("..") or os.path.isabs(norm) or not norm.startswith(f"{user_id}/"):
                            raise RuntimeError("Invalid mask path.")
                        mp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
                        me = mp + '.enc'
                        mbytes = None
                        if os.path.exists(mp):
                            with open(mp, 'rb') as f: mbytes = f.read()
                        elif os.path.exists(me):
                            with open(me, 'rb') as f: mbytes = decrypt_bytes(f.read())
                        if not mbytes:
                            raise RuntimeError("Mask file not found.")
                        try:
                            base_img = Image.open(BytesIO(img_inputs[0][1]))
                            mask_img = Image.open(BytesIO(mbytes)).convert('RGBA')
                            if base_img.size != mask_img.size:
                                raise RuntimeError("Mask must match input image size.")
                            out = BytesIO()
                            mask_img.save(out, format='PNG')
                            mbytes = out.getvalue()
                            if len(mbytes) > 4 * 1024 * 1024:
                                raise RuntimeError("Mask must be less than 4MB.")
                            mask_file = ("mask.png", mbytes, "image/png")
                        except RuntimeError:
                            raise
                        except Exception:
                            raise RuntimeError("Failed to process mask file.")
                    
                    if img_inputs:
                        pub("status", "OpenAI API (Edit) を呼び出し中...")
                    else:
                        pub("status", "OpenAI API (Generations) を呼び出し中... (これには時間がかかる場合があります)")
                    
                    _mark_provider_request_started()
                    
                    # Build tools and input for Responses API
                    tools = [
                        {
                            "type": "image_generation",
                            "model": model_key,
                            "size": size_opt,
                            "quality": quality_opt,
                            "output_format": format_opt,
                        }
                    ]
                    if comp_opt is not None:
                        tools[0]["output_compression"] = comp_opt
                    
                    input_content = [{"type": "input_text", "text": img_prompt}]
                    for name, bits, mime in img_inputs:
                        b64 = base64.b64encode(bits).decode()
                        input_content.append({
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{b64}"
                        })
                    
                    if mask_file:
                        mask_b64 = base64.b64encode(mask_file[1]).decode()
                        tools[0]["input_image_mask"] = {"image_url": f"data:image/png;base64,{mask_b64}"}
                        tools[0]["action"] = "edit"
                    elif img_inputs:
                        tools[0]["action"] = "edit"
                    else:
                        tools[0]["action"] = "generate"

                    # Use gpt-4o-mini as the driver for image generation tool
                    # background=True allows cancellation via the API
                    resp_obj = img_client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user", "content": input_content}],
                        tools=tools,
                        background=True
                    )
                    
                    # Polling loop to check for completion and cancellation
                    while resp_obj.status in {"queued", "in_progress"}:
                        if check_stop():
                            try:
                                img_client.responses.cancel(resp_obj.id)
                                log_force(f"GPT Image Gen Job {job_id} cancelled via Responses API.")
                            except Exception as ce:
                                log_force(f"Failed to cancel GPT Image Gen: {ce}")
                            raise RuntimeError("Generation stopped by user.")
                        time.sleep(2)
                        resp_obj = img_client.responses.retrieve(resp_obj.id)
                    
                    if resp_obj.status == "failed":
                        err_msg = "Unknown error"
                        if hasattr(resp_obj, "error") and resp_obj.error:
                            err_msg = resp_obj.error.message
                        raise RuntimeError(f"OpenAI Responses API failed: {err_msg}")
                    
                    if resp_obj.status == "cancelled":
                        raise RuntimeError("Generation was cancelled.")

                    # Extract the generated image from the tool output
                    image_data_b64 = None
                    for out_item in (resp_obj.output or []):
                        # Some versions of the SDK might use dict, others objects
                        if isinstance(out_item, dict):
                            if out_item.get("type") == "image_generation_call":
                                image_data_b64 = out_item.get("result")
                                break
                        else:
                            if getattr(out_item, "type", None) == "image_generation_call":
                                image_data_b64 = getattr(out_item, "result", None)
                                break
                    
                    if not image_data_b64:
                        raise RuntimeError("No image data found in the response.")
                    
                    img_bytes = _decode_base64_limited(image_data_b64, 50 * 1024 * 1024)
                    ext = "png"
                    if format_opt == "jpeg":
                        ext = "jpg"
                    elif format_opt == "webp":
                        ext = "webp"
                    fn2 = f"gen_gpt_{int(time.time())}_{len(generated_images)}.{ext}"
                    pub("status", "画像を保存して暗号化を適用中...")
                    _save_user_generated_bytes(
                        user_id, img_bytes, fn2, user_config.get('enable_e2ee')
                    )
                    generated_images.append(f"{user_id}/{fn2}")
                    pub("status", "完了")
                    pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                    full_res += f"Generated Image for: {final_message_text}\n"
                except APITimeoutError:
                    pub("error", "GPT Image Gen Timeout: Upstream is slow. Please retry.")
                except (APIConnectionError, RateLimitError) as e:
                    pub("error", f"GPT Image Gen Error: {str(e)}")
                except APIError as e:
                    pub("error", f"GPT Image Gen Error: {str(e)}")
                except Exception as e:
                    pub("error", f"GPT Image Gen Error: {str(e)}")

            # --- 3.5 OpenAI Search API (Chat Completions) ---
            elif is_openai_search_model:
                log_force("Routing: OpenAI Search API Branch (Chat Completions)")
                try:
                    if any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files):
                        pub("error", "gpt-5-search-api does not support image inputs. Please remove images and retry.")
                        return
                    if check_stop():
                        return
                    pub("search_status", "searching")
                    client = o_client
                    sys_prompt = _openai_system_prompt(options.get('system_prompt'), True)
                    messages = []
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    
                    history_img_seen = set()
                    history_img_bytes = 0

                    for m in history:
                        if m.get('image_url'):
                            try:
                                content_parts = [{"type": "text", "text": m['content']}]
                                msg_images, history_img_bytes = _load_message_history_images(
                                    m.get('image_url'),
                                    seen=history_img_seen,
                                    total_bytes=history_img_bytes,
                                    include_only_images=True
                                )
                                for msg_img in msg_images:
                                    b64 = base64.b64encode(msg_img['bytes']).decode('utf-8')
                                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{msg_img['mime']};base64,{b64}"}})
                                messages.append({"role": m['role'], "content": content_parts})
                            except Exception as e:
                                log_force(f"Error processing history image in search branch: {e}")
                                messages.append({"role": m['role'], "content": m['content']})
                        else:
                            messages.append({"role": m['role'], "content": m['content']})

                    user_text = ""
                    if quote_text:
                        user_text += f"User Quote:\n{quote_text}\n---\n"
                    user_text += message_text
                    file_parts = []
                    file_attach_errors = []
                    file_inline_limit = 20 * 1024 * 1024  # 20MiB inline limit for file inputs
                    for fi in loaded_files:
                        if (fi.get('is_pdf') or fi.get('is_docx') or fi.get('is_text')) and fi.get('bytes'):
                            attached = False
                            try:
                                f_bytes = fi['bytes']
                                f_name = os.path.basename(fi.get('send_name') or fi.get('name') or ('document.pdf' if fi.get('is_pdf') else 'document.docx' if fi.get('is_docx') else 'document.txt'))
                                if len(f_bytes) <= file_inline_limit:
                                    b64 = base64.b64encode(f_bytes).decode('utf-8')
                                    file_parts.append({"type": "file", "file": {"file_data": b64, "filename": f_name}})
                                else:
                                    rel_path = fi.get('path') or fi.get('name') or f_name
                                    cache = _get_file_cache(user_id, rel_path, "openai")
                                    file_id = None
                                    if _openai_cache_fresh(cache, fi.get('size'), fi.get('mtime'), fi.get('mime')):
                                        file_id = cache.file_id
                                        _upsert_file_cache(
                                            user_id,
                                            rel_path,
                                            "openai",
                                            state="ACTIVE",
                                            last_error=None,
                                            last_checked_at=datetime.utcnow()
                                        )
                                        safe_db_commit()
                                    if not file_id:
                                        suffix = os.path.splitext(f_name)[1] or ('.pdf' if fi.get('is_pdf') else '.docx' if fi.get('is_docx') else '.txt')
                                        file_id, up_err = _openai_upload_with_retry(
                                            client,
                                            f_bytes,
                                            suffix,
                                            rel_path,
                                            mime=fi.get('mime'),
                                            size=fi.get('size'),
                                            mtime=fi.get('mtime')
                                        )
                                        if not file_id:
                                            raise RuntimeError(up_err or "file upload failed")
                                    file_parts.append({"type": "file", "file": {"file_id": file_id, "filename": f_name}})
                                attached = True
                            except Exception as e:
                                log_force(f"OpenAI Search file attach failed: {e}")
                                file_attach_errors.append(f"{f_name}({str(e)[:120]})")
                            if attached:
                                continue
                        if fi.get('text'):
                            user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
                    if file_attach_errors:
                        parts = file_attach_errors[:5]
                        if len(file_attach_errors) > 5:
                            parts.append(f"...他{len(file_attach_errors)-5}件")
                        pub("error", "ファイル添付に失敗しました: " + " / ".join(parts))
                        return
                    if file_parts:
                        user_parts = [{"type": "text", "text": user_text}]
                        user_parts.extend(file_parts)
                        messages.append({"role": "user", "content": user_parts})
                    else:
                        messages.append({"role": "user", "content": user_text})

                    _mark_provider_request_started()
                    resp = client.chat.completions.create(
                        model=model_key,
                        messages=messages,
                        web_search_options={"search_context_size": "medium"}
                    )
                    if not resp or not getattr(resp, "choices", None):
                        pub("error", "Search API Error: Empty response.")
                        return

                    msg = resp.choices[0].message
                    text_parts = []
                    citations = []
                    seen_urls = set()

                    def _add_citation(title, url):
                        if not url or url in seen_urls:
                            return
                        seen_urls.add(url)
                        citations.append((title or url, url))

                    def _handle_annotations(ann_list):
                        for ann in ann_list or []:
                            if isinstance(ann, dict):
                                a_type = ann.get("type")
                                a_url = ann.get("url") or ann.get("source") or ann.get("link")
                                a_title = ann.get("title") or a_url
                            else:
                                a_type = getattr(ann, "type", None)
                                a_url = getattr(ann, "url", None)
                                a_title = getattr(ann, "title", None) or a_url
                            if a_type and "citation" in str(a_type).lower() and a_url:
                                _add_citation(a_title, a_url)

                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                p_type = part.get("type")
                                p_text = part.get("text")
                                p_anns = part.get("annotations")
                            else:
                                p_type = getattr(part, "type", None)
                                p_text = getattr(part, "text", None)
                                p_anns = getattr(part, "annotations", None)
                            if p_type in (None, "text", "output_text") and p_text:
                                text_parts.append(p_text)
                            if p_anns:
                                _handle_annotations(p_anns)
                    elif isinstance(content, str):
                        if content:
                            text_parts.append(content)
                    elif content is not None:
                        text_parts.append(str(content))

                    _handle_annotations(getattr(msg, "annotations", None))

                    final_text = "".join(text_parts).strip()
                    if final_text:
                        full_res += final_text
                        pub("content", final_text)

                    if citations:
                        citations_text = "\n\n**Sources:**\n"
                        for title, url in citations:
                            citations_text += f"- [{title}]({url})\n"
                        full_res += citations_text
                        pub("content", citations_text)
                except Exception as e:
                    pub("error", f"Search API Error: {str(e)}")
                finally:
                    pub("search_status", "done")

            elif is_deepseek:
                log_force("Routing: DeepSeek V4 Branch (Chat Completions)")
                try:
                    if check_stop():
                        return
                    client = o_client
                    messages = []
                    sys_prompt = options.get('system_prompt') or ""
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})

                    # DeepSeek V4 Flash Vision Exp accepts images natively via OpenAI-compatible
                    # image_url content blocks (official Vision guide). Other DeepSeek V4
                    # models remain text-only and use the vision-analysis fallback below.
                    deepseek_native_vision = "vision-exp" in model_key_l
                    ds_history_img_seen = set()
                    ds_history_img_bytes = 0

                    for m in history:
                        saved_tool_context = m.get("deepseek_tool_context")
                        if m.get("role") == "assistant" and isinstance(saved_tool_context, list):
                            for saved_message in saved_tool_context:
                                if isinstance(saved_message, dict) and saved_message.get("role") in {"assistant", "tool"}:
                                    messages.append(saved_message)
                        elif deepseek_native_vision and m.get('image_url'):
                            try:
                                ds_content_parts = [{"type": "text", "text": m['content'] or ""}]
                                ds_msg_images, ds_history_img_bytes = _load_message_history_images(
                                    m.get('image_url'),
                                    seen=ds_history_img_seen,
                                    total_bytes=ds_history_img_bytes,
                                    include_only_images=True
                                )
                                for ds_msg_img in ds_msg_images:
                                    ds_b64 = base64.b64encode(ds_msg_img['bytes']).decode('utf-8')
                                    ds_content_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{ds_msg_img['mime']};base64,{ds_b64}"}
                                    })
                                messages.append({"role": m['role'], "content": ds_content_parts})
                            except Exception as ds_hist_err:
                                log_force(f"Error processing history image in DeepSeek branch: {ds_hist_err}")
                                messages.append({"role": m['role'], "content": m['content']})
                        else:
                            messages.append({"role": m['role'], "content": m['content']})

                    user_text = ""
                    if quote_text:
                        user_text += f"User Quote:\n{quote_text}\n---\n"
                    user_text += message_text

                    # Separate image files from text files
                    image_files = [fi for fi in loaded_files if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')]
                    for fi in loaded_files:
                        if fi.get('text') and not (fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')):
                            user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"

                    # If images are present, analyze them with a vision model unless the selected
                    # DeepSeek model accepts image inputs natively (DeepSeek V4 Flash Vision Exp).
                    if image_files and not deepseek_native_vision:
                        vision_model = (options.get('image_vision_model') or "").strip()
                        analysis_prompt = _auto_notice_text("image_analysis") or DEFAULT_IMAGE_ANALYSIS_PROMPT
                        if not vision_model:
                            pub("error", "DeepSeek V4 does not support images. Please select a Vision Model in Settings > Default Vision Model to enable automatic image analysis.")
                            return
                        pub("status", "画像を vision model で解析中...")
                        analysis_texts = []
                        for idx, fi in enumerate(image_files):
                            pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析中...")
                            if check_stop():
                                return
                            img_data = fi.get('bytes')
                            img_mime = str(fi.get('mime', 'image/png'))
                            analysis_result = _analyze_image_with_vision_model(
                                vision_model, img_data, img_mime, analysis_prompt,
                                api_keys
                            )
                            if analysis_result:
                                analysis_texts.append(f"--- Image {idx+1} ---\n{analysis_result}")
                                pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析完了")
                            else:
                                pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析失敗")
                            if check_stop():
                                return
                        if analysis_texts:
                            analysis_block = "The user attached the following image(s). Detailed descriptions are provided below:\n\n" + "\n\n".join(analysis_texts)
                            if user_text.strip():
                                messages.append({"role": "system", "content": analysis_block})
                            else:
                                # Image-only send: use the vision analysis as the user turn.
                                user_text = analysis_block
                            pub("image_analysis", f"全{len(analysis_texts)}枚の画像解析完了。DeepSeek で応答生成中...")
                        else:
                            pub("error", "画像の解析に失敗しました。Vision Model の API 設定を確認してください。")
                            return

                    deepseek_user_content = None
                    if image_files and deepseek_native_vision:
                        # Native vision path: attach images inline as OpenAI-compatible
                        # base64 data URL blocks alongside the text turn.
                        deepseek_user_content = []
                        if user_text.strip():
                            deepseek_user_content.append({"type": "text", "text": user_text})
                        for ds_idx, ds_file in enumerate(image_files):
                            ds_img_b64 = base64.b64encode(ds_file.get('bytes') or b"").decode("utf-8")
                            ds_img_mime = str(ds_file.get('mime', 'image/png')) or "image/png"
                            deepseek_user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{ds_img_mime};base64,{ds_img_b64}"}
                            })
                            pub("image_analysis", f"画像 {ds_idx+1}/{len(image_files)} を送信対象に追加")

                    if not user_text.strip() and deepseek_user_content is None:
                        pub("error", "DeepSeek request is empty.")
                        return

                    if deepseek_user_content is not None:
                        messages.append({"role": "user", "content": deepseek_user_content})
                    else:
                        messages.append({"role": "user", "content": user_text})
                    enable_reasoning = (
                        req_reasoning_effort != "none"
                        and (bool(options.get('enable_thinking')) or bool(req_reasoning_effort))
                    )
                    python_tools = []
                    if options.get("enable_python"):
                        python_tools.append({
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "description": (
                                    "Execute Python code for calculations, verification, or data analysis. "
                                    "The isolated environment has no network access. "
                                    "The code must print every result that should be returned to you."
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": (
                                                "Python source code to execute. Use print(...) for every value "
                                                "you need in the tool result."
                                            ),
                                        }
                                    },
                                    "required": ["code"],
                                    "additionalProperties": False,
                                },
                            },
                        })
                    if options.get("enable_file_creation"):
                        python_tools.append(_build_create_file_tool_schema())
                        python_tools.append(_build_edit_file_tool_schema())

                    # MCP外部ツールの定義を追加（ユーザーが有効化＆認証済みのサーバーのみ）
                    ds_mcp_env = None
                    try:
                        ds_mcp_env = _ensure_mcp_env()
                    except Exception:
                        ds_mcp_env = None
                    if ds_mcp_env is not None:
                        try:
                            python_tools.extend(ds_mcp_env.serialize_chat_completions())
                        except Exception as _mcp_e:
                            log_force(f"DeepSeek MCP tool attach failed: {_mcp_e}")

                    deepseek_usage_totals = {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    }
                    saw_deepseek_usage = False
                    max_tool_rounds = 8
                    for tool_round in range(max_tool_rounds):
                        deepseek_kwargs = {
                            "model": _deepseek_api_model_id(model_key),
                            "messages": messages,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        }
                        if python_tools:
                            deepseek_kwargs["tools"] = python_tools
                            deepseek_kwargs["tool_choice"] = "auto"
                        if enable_reasoning:
                            deepseek_kwargs["reasoning_effort"] = _deepseek_reasoning_effort()
                            deepseek_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                        else:
                            deepseek_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                        # Official per-user isolation for safety, scheduling, and KV cache.
                        deepseek_kwargs["extra_body"]["user_id"] = f"app_user_{user_id}"

                        _mark_provider_request_started()
                        stream = client.chat.completions.create(**deepseek_kwargs)
                        chunk_count = 0
                        round_reasoning = ""
                        round_content = ""
                        streamed_tool_calls = {}

                        for chunk in stream:
                            _latency_mark_once(job_id, "provider_first_chunk_ms")
                            if check_stop():
                                return

                            # Refresh pending_job TTL periodically to prevent expiry during long thinking
                            chunk_count += 1
                            if chunk_count % 20 == 0:
                                _refresh_pending_job()

                            usage = getattr(chunk, 'usage', None)
                            if usage:
                                saw_deepseek_usage = True
                                for usage_key in ("completion_tokens", "prompt_tokens", "total_tokens"):
                                    usage_value = (
                                        usage.get(usage_key)
                                        if isinstance(usage, dict)
                                        else getattr(usage, usage_key, None)
                                    )
                                    if usage_value is not None:
                                        deepseek_usage_totals[usage_key] += int(usage_value or 0)
                                usage_details = (
                                    usage.get("completion_tokens_details")
                                    if isinstance(usage, dict)
                                    else getattr(usage, "completion_tokens_details", None)
                                )
                                reasoning_tokens = (
                                    usage_details.get("reasoning_tokens")
                                    if isinstance(usage_details, dict)
                                    else getattr(usage_details, "reasoning_tokens", None)
                                    if usage_details is not None
                                    else None
                                )
                                if reasoning_tokens is not None:
                                    deepseek_usage_totals["completion_tokens_details"]["reasoning_tokens"] += int(reasoning_tokens or 0)

                            choices = getattr(chunk, 'choices', None)
                            if not choices:
                                continue

                            delta = choices[0].delta

                            # DeepSeek exposes reasoning_content as an OpenAI-compatible extension.
                            r_content = getattr(delta, "reasoning_content", None)
                            if not r_content:
                                extra = getattr(delta, '__pydantic_extra__', None) or {}
                                r_content = extra.get('reasoning_content')
                            if r_content:
                                round_reasoning += r_content
                                thought_accumulated += r_content
                                pub("thought", r_content)

                            c_content = getattr(delta, 'content', None)
                            if c_content:
                                round_content += c_content
                                full_res += c_content
                                pub("content", c_content)

                            accumulate_deepseek_tool_call_deltas(
                                streamed_tool_calls,
                                getattr(delta, "tool_calls", None) or [],
                            )

                        if not streamed_tool_calls:
                            if deepseek_tool_context:
                                final_assistant_message = {
                                    "role": "assistant",
                                    "content": round_content,
                                }
                                if enable_reasoning:
                                    final_assistant_message["reasoning_content"] = round_reasoning
                                deepseek_tool_context.append(final_assistant_message)
                            break

                        assistant_tool_calls = []
                        for tool_index in sorted(streamed_tool_calls):
                            tool_state = streamed_tool_calls[tool_index]
                            assistant_tool_calls.append({
                                "id": tool_state["id"] or f"deepseek_tool_{tool_round}_{tool_index}",
                                "type": "function",
                                "function": {
                                    "name": tool_state["name"],
                                    "arguments": tool_state["arguments"],
                                },
                            })

                        assistant_tool_message = {
                            "role": "assistant",
                            "content": round_content,
                            "tool_calls": assistant_tool_calls,
                        }
                        if enable_reasoning:
                            # Required by DeepSeek for every subsequent request after a thinking tool call.
                            assistant_tool_message["reasoning_content"] = round_reasoning
                        messages.append(assistant_tool_message)
                        deepseek_tool_context.append(assistant_tool_message)

                        for tool_call in assistant_tool_calls:
                            call_id = tool_call["id"]
                            function_data = tool_call["function"]
                            call_name = function_data.get("name")
                            call_arguments = function_data.get("arguments") or ""
                            code = ""
                            if call_name == "create_file":
                                try:
                                    parsed_arguments = json.loads(call_arguments)
                                    if not isinstance(parsed_arguments, dict):
                                        raise ValueError("arguments must be a JSON object")
                                    create_result = _execute_create_file_tool(
                                        user_id, parsed_arguments, user_config.get('enable_e2ee')
                                    )
                                    result = _create_file_tool_result_text(create_result)
                                    if create_result.get("ok"):
                                        created_file_rel = f"{user_id}/{create_result['filename']}"
                                        if created_file_rel not in generated_images:
                                            generated_images.append(created_file_rel)
                                        file_link_md = (
                                            f"\n📄 **ファイルを作成しました:** [{create_result.get('display_name')}]({create_result.get('url')})\n"
                                        )
                                        full_res += file_link_md
                                        pub("content", file_link_md)
                                except Exception as tool_exc:
                                    result = f"Error: Invalid create_file arguments: {tool_exc}"
                            elif call_name == "edit_file":
                                try:
                                    parsed_arguments = json.loads(call_arguments)
                                    if not isinstance(parsed_arguments, dict):
                                        raise ValueError("arguments must be a JSON object")
                                    edit_result = _execute_edit_file_tool(
                                        user_id,
                                        parsed_arguments,
                                        user_config.get('enable_e2ee'),
                                        loaded_files=loaded_files,
                                        history=history,
                                        thread_id=thread_id,
                                    )
                                    result = _create_file_tool_result_text(edit_result, "edit_file", "編集")
                                    if edit_result.get("ok"):
                                        edited_file_rel = f"{user_id}/{edit_result['filename']}"
                                        if edited_file_rel not in generated_images:
                                            generated_images.append(edited_file_rel)
                                        file_link_md = (
                                            f"\n📄 **ファイルを編集しました:** [{edit_result.get('display_name')}]({edit_result.get('url')})\n"
                                        )
                                        full_res += file_link_md
                                        pub("content", file_link_md)
                                except Exception as tool_exc:
                                    result = f"Error: Invalid edit_file arguments: {tool_exc}"
                            elif str(call_name or "").startswith("mcp__"):
                                try:
                                    parsed_arguments = json.loads(call_arguments)
                                    if not isinstance(parsed_arguments, dict):
                                        raise ValueError("arguments must be a JSON object")
                                    if ds_mcp_env is None:
                                        try:
                                            ds_mcp_env = _ensure_mcp_env()
                                        except Exception:
                                            ds_mcp_env = None
                                    if ds_mcp_env is None:
                                        result = f"Error: MCP tools are not available: {call_name}"
                                    else:
                                        mcp_text, mcp_out = ds_mcp_env.execute(call_name, parsed_arguments)
                                        result = mcp_text
                                        if mcp_out.get("ok"):
                                            _md = f"\n\n> 🔧 **MCPツール実行:** `{call_name}` を実行しました。\n"
                                            full_res += _md
                                            pub("content", _md)
                                        elif mcp_out.get("rejected"):
                                            _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{call_name}`\n"
                                            full_res += _md
                                            pub("content", _md)
                                except Exception as tool_exc:
                                    result = f"Error: Invalid MCP tool arguments: {tool_exc}"
                            elif call_name != "execute_python":
                                result = f"Error: Unsupported tool: {call_name or '(missing name)'}"
                            else:
                                try:
                                    parsed_arguments = json.loads(call_arguments)
                                    if not isinstance(parsed_arguments, dict):
                                        raise ValueError("arguments must be a JSON object")
                                    code = parsed_arguments.get("code")
                                    if not isinstance(code, str) or not code.strip():
                                        raise ValueError("code must be a non-empty string")
                                    result = safe_execute_python(code)
                                except Exception as tool_exc:
                                    result = f"Error: Invalid execute_python arguments: {tool_exc}"

                            if code:
                                code_fence = _markdown_fence_for_code(code)
                                output_fence = _markdown_fence_for_code(result)
                                visible_tool_output = (
                                    f"\n{code_fence}python\n{code}\n{code_fence}\n"
                                    f"\n**Output:**\n{output_fence}\n{result}\n{output_fence}\n"
                                )
                                full_res += visible_tool_output
                                pub("content", visible_tool_output)
                                full_res += (
                                    f"\n```pyexec\n"
                                    f"{json.dumps({'code': code, 'output': result}, ensure_ascii=False)}"
                                    f"\n```\n"
                                )
                                pub("python", {"id": call_id, "code": code, "output": result})

                            tool_message = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": result,
                            }
                            messages.append(tool_message)
                            deepseek_tool_context.append(tool_message)
                        continue
                    else:
                        raise RuntimeError(f"DeepSeek Python tool exceeded {max_tool_rounds} rounds.")

                    if saw_deepseek_usage:
                        final_openai_usage = deepseek_usage_totals
                except Exception as e:
                    pub("error", f"DeepSeek Error: {str(e)}")

            elif is_kimi:
                log_force("Routing: Kimi K3 Branch (Chat Completions)")
                try:
                    if check_stop():
                        return
                    client = o_client
                    messages = []
                    sys_prompt = options.get('system_prompt') or ""
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})

                    for m in history:
                        messages.append({"role": m['role'], "content": m['content']})

                    user_text = ""
                    if quote_text:
                        user_text += f"User Quote:\n{quote_text}\n---\n"
                    user_text += message_text

                    # Separate image files from text files
                    image_files = [fi for fi in loaded_files if fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')]
                    for fi in loaded_files:
                        if fi.get('text') and not (fi.get('bytes') and str(fi.get('mime', '')).startswith('image/')):
                            user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"

                    # If images are present, analyze them with a vision model (Kimi K3 supports images directly)
                    if image_files:
                        vision_model = (options.get('image_vision_model') or "").strip()
                        analysis_prompt = _auto_notice_text("image_analysis") or DEFAULT_IMAGE_ANALYSIS_PROMPT
                        if not vision_model:
                            pub("error", "Kimi K3 supports images directly, but no Vision Model is configured for automatic image analysis. Please select a Vision Model in Settings > Default Vision Model.")
                            return
                        pub("status", "画像を vision model で解析中...")
                        analysis_texts = []
                        for idx, fi in enumerate(image_files):
                            pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析中...")
                            if check_stop():
                                return
                            img_data = fi.get('bytes')
                            img_mime = str(fi.get('mime', 'image/png'))
                            analysis_result = _analyze_image_with_vision_model(
                                vision_model, img_data, img_mime, analysis_prompt,
                                api_keys
                            )
                            if analysis_result:
                                analysis_texts.append(f"--- Image {idx+1} ---\n{analysis_result}")
                                pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析完了")
                            else:
                                pub("image_analysis", f"画像 {idx+1}/{len(image_files)} 解析失敗")
                            if check_stop():
                                return
                        if analysis_texts:
                            analysis_block = "The user attached the following image(s). Detailed descriptions are provided below:\n\n" + "\n\n".join(analysis_texts)
                            if user_text.strip():
                                messages.append({"role": "system", "content": analysis_block})
                            else:
                                # Image-only send: use the vision analysis as the user turn.
                                user_text = analysis_block
                            pub("image_analysis", f"全{len(analysis_texts)}枚の画像解析完了。Kimi K3 で応答生成中...")
                        else:
                            pub("error", "画像の解析に失敗しました。Vision Model の API 設定を確認してください。")
                            return

                    if not user_text.strip():
                        pub("error", "Kimi K3 request is empty.")
                        return

                    messages.append({"role": "user", "content": user_text})
                    kimi_kwargs = {
                        "model": model_key,
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    # Kimi K3 always thinks; reasoning_effort is top-level (low/high/max, default max)
                    enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
                    if enable_reasoning:
                        kimi_kwargs["reasoning_effort"] = _kimi_reasoning_effort()
                    if options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
                        kimi_kwargs["extra_body"] = dict(kimi_kwargs.get("extra_body") or {})
                        kimi_kwargs["extra_body"]["prompt_cache_key"] = options.get('prompt_cache_key')
                        log_force(f"Kimi Prompt Caching key={options.get('prompt_cache_key')}")

                    _mark_provider_request_started()
                    final_openai_usage = None
                    stream = client.chat.completions.create(**kimi_kwargs)
                    chunk_count = 0
                    for chunk in stream:
                        _latency_mark_once(job_id, "provider_first_chunk_ms")
                        if check_stop():
                            break

                        chunk_count += 1
                        if chunk_count % 20 == 0:
                            _refresh_pending_job()

                        usage = getattr(chunk, 'usage', None)
                        if usage:
                            final_openai_usage = usage

                        choices = getattr(chunk, 'choices', None)
                        if not choices:
                            continue

                        delta = choices[0].delta

                        # Extract reasoning_content
                        r_content = None
                        try:
                            r_content = delta.reasoning_content
                        except AttributeError:
                            pass
                        if not r_content:
                            try:
                                extra = getattr(delta, '__pydantic_extra__', None) or {}
                                r_content = extra.get('reasoning_content')
                            except Exception:
                                pass
                        if r_content:
                            thought_accumulated += r_content
                            pub("thought", r_content)

                        c_content = getattr(delta, 'content', None)
                        if c_content:
                            full_res += c_content
                            pub("content", c_content)
                except Exception as e:
                    pub("error", f"Kimi Error: {str(e)}")

            # --- 4. OpenAI Responses API (or Grok Fallback) ---
            else:
                log_force("Routing: Responses API Branch")
                client = o_client
                input_data = []
                sys_prompt = _grok_system_prompt(options.get('system_prompt'), grok_enable_search) if is_grok else _openai_system_prompt(options.get('system_prompt'), auto_enable_search)
                if sys_prompt: input_data.append({"role": "system", "content": sys_prompt})
                
                history_img_seen = set()
                history_img_bytes = 0
                text_type = "input_text"
                image_type = "input_image"

                for m in history:
                    if m.get('image_url'):
                        try:
                            content_parts = [{"type": text_type, "text": m['content']}]
                            msg_images, history_img_bytes = _load_message_history_images(
                                m.get('image_url'),
                                seen=history_img_seen,
                                total_bytes=history_img_bytes,
                                include_only_images=True
                            )
                            for msg_img in msg_images:
                                b64 = base64.b64encode(msg_img['bytes']).decode('utf-8')
                                content_parts.append({"type": image_type, "image_url": f"data:{msg_img['mime']};base64,{b64}"})
                            input_data.append({"role": m['role'], "content": content_parts})
                        except Exception as e:
                            log_force(f"Error processing history image: {e}")
                            input_data.append({"role": m['role'], "content": m['content']})
                    else:
                        input_data.append({"role": m['role'], "content": m['content']})

                curr_content = []
                if quote_text: curr_content.append({"type": text_type, "text": f"User Quote:\n{quote_text}\n---"})
                if message_text and str(message_text).strip():
                    curr_content.append({"type": text_type, "text": message_text})
                file_inline_limit = 20 * 1024 * 1024  # 20MiB inline limit for file inputs
                file_attach_errors = []
                current_image_names = []

                for fi in loaded_files:
                    if (fi.get('is_pdf') or fi.get('is_docx') or fi.get('is_text')) and fi.get('bytes') and not is_grok:
                        attached = False
                        try:
                            f_bytes = fi['bytes']
                            f_name = os.path.basename(fi.get('send_name') or fi.get('name') or ('document.pdf' if fi.get('is_pdf') else 'document.docx' if fi.get('is_docx') else 'document.txt'))
                            if len(f_bytes) <= file_inline_limit:
                                b64 = base64.b64encode(f_bytes).decode('utf-8')
                                curr_content.append({"type": "input_file", "file_data": b64, "filename": f_name})
                            else:
                                rel_path = fi.get('path') or fi.get('name') or f_name
                                cache = _get_file_cache(user_id, rel_path, "openai")
                                file_id = None
                                if _openai_cache_fresh(cache, fi.get('size'), fi.get('mtime'), fi.get('mime')):
                                    file_id = cache.file_id
                                    _upsert_file_cache(
                                        user_id,
                                        rel_path,
                                        "openai",
                                        state="ACTIVE",
                                        last_error=None,
                                        last_checked_at=datetime.utcnow()
                                    )
                                    safe_db_commit()
                                if not file_id:
                                    suffix = os.path.splitext(f_name)[1] or ('.pdf' if fi.get('is_pdf') else '.docx' if fi.get('is_docx') else '.txt')
                                    file_id, up_err = _openai_upload_with_retry(
                                        client,
                                        f_bytes,
                                        suffix,
                                        rel_path,
                                        mime=fi.get('mime'),
                                        size=fi.get('size'),
                                        mtime=fi.get('mtime')
                                    )
                                    if not file_id:
                                        raise RuntimeError(up_err or "file upload failed")
                                curr_content.append({"type": "input_file", "file_id": file_id, "filename": f_name})
                            attached = True
                        except Exception as e:
                            log_force(f"OpenAI file attach failed: {e}")
                            file_attach_errors.append(f"{f_name}({str(e)[:120]})")
                        if attached:
                            continue
                    if fi.get('text'):
                        for part in reversed(curr_content):
                            if part.get('type') == text_type:
                                part['text'] += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
                                break
                    elif fi.get('bytes') and fi['mime'].startswith('image/'):
                        img_bytes = fi['bytes']
                        img_mime = fi['mime']
                        if is_grok and img_mime not in ('image/jpeg', 'image/png'):
                            try:
                                im = Image.open(BytesIO(img_bytes))
                                if im.mode not in ('RGB', 'RGBA'):
                                    im = im.convert('RGB')
                                out = BytesIO()
                                im.save(out, format='PNG')
                                img_bytes = out.getvalue()
                                img_mime = 'image/png'
                            except Exception:
                                pass
                        b64 = base64.b64encode(img_bytes).decode('utf-8')
                        curr_content.append({"type": image_type, "image_url": f"data:{img_mime};base64,{b64}"})
                        img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
                        current_image_names.append(os.path.basename(str(img_label)))
                name_block = _build_attachment_name_block(current_image_names)
                if name_block:
                    for part in reversed(curr_content):
                        if part.get('type') == text_type:
                            part['text'] += f"\n\n{name_block}"
                            break
                if file_attach_errors:
                    parts = file_attach_errors[:5]
                    if len(file_attach_errors) > 5:
                        parts.append(f"...他{len(file_attach_errors)-5}件")
                    pub("error", "ファイル添付に失敗しました: " + " / ".join(parts))
                    return

                input_data.append({"role": "user", "content": curr_content})
                
                # OpenAI/xAI Responses API
                has_image_inputs = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files)
                # xAI docs: image understanding requests should avoid server-side storage.
                store_flag = False if (is_grok and has_image_inputs) else True
                kwargs = {
                    "model": model_key,
                    "input": input_data,
                    "stream": True,
                    "store": store_flag,
                }
                if options.get('enable_prompt_caching') and options.get('prompt_cache_key'):
                    kwargs["prompt_cache_key"] = options.get('prompt_cache_key')
                    log_force(f"Responses API Prompt Caching key={options.get('prompt_cache_key')}")

                if is_grok and grok_enable_search:
                    kwargs['tools'] = [{"type": "web_search"}, {"type": "x_search"}]
                    kwargs.setdefault("include", [])
                    if "inline_citations" not in kwargs["include"]:
                        kwargs["include"].append("inline_citations")
                    log_force("Enabled Web + X Search Tools (Responses API)")
                elif auto_enable_search:
                    kwargs['tools'] = [{"type": "web_search"}]
                    kwargs.setdefault("include", [])
                    if "web_search_call.action.sources" not in kwargs["include"]:
                        kwargs["include"].append("web_search_call.action.sources")
                    log_force("Enabled Web Search Tool (Responses API)")

                if options.get('enable_python'):
                    if 'tools' not in kwargs: kwargs['tools'] = []
                    kwargs['tools'].append({
                        "type": "function",
                        "name": "execute_python",
                        "description": "Execute Python code for calculations or data analysis. Isolated environment, no internet access.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Python code to run."}
                            },
                            "required": ["code"]
                        }
                    })

                if options.get('enable_file_creation'):
                    if 'tools' not in kwargs: kwargs['tools'] = []
                    kwargs['tools'].append(_build_create_file_tool_schema())
                    kwargs['tools'].append(_build_edit_file_tool_schema())

                # MCP外部ツール（Responses API の function tool として追加）
                resp_mcp_env = None
                try:
                    resp_mcp_env = _ensure_mcp_env()
                except Exception:
                    resp_mcp_env = None
                if resp_mcp_env is not None:
                    try:
                        if 'tools' not in kwargs: kwargs['tools'] = []
                        kwargs['tools'].extend(resp_mcp_env.serialize_openai())
                    except Exception as _mcp_e:
                        log_force(f"Responses MCP tool attach failed: {_mcp_e}")

                if is_grok and options.get('enable_thinking') and not grok_reasoning_supported:
                    pub("thought", "APIの仕様により表示されません")
                is_reasoning_model = (not is_grok) and any(x in model_key.lower() for x in ['o1', 'o3', 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'reasoning'])
                req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
                enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")

                def _normalize_reasoning_effort(model_key_l, effort):
                    if not effort:
                        return effort
                    effort = effort.lower().strip()
                    # Smaller GPT-5 tiers do not accept "none"; use minimal instead.
                    if any(x in model_key_l for x in ("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5-mini", "gpt-5.5-nano")) and effort == "none":
                        return "minimal"
                    return effort
                if is_grok and enable_reasoning and grok_reasoning_effort_supported:
                    grok_effort = _grok_reasoning_effort()
                    if grok_effort:
                        kwargs['reasoning'] = {"effort": grok_effort}
                        log_force(f"Grok reasoning config: {kwargs['reasoning']}")
                elif is_grok and enable_reasoning and grok_reasoning_supported:
                    log_force("Grok reasoning_effort not supported for this model; skipping reasoning param")
                elif is_reasoning_model and enable_reasoning:
                    effort = req_reasoning_effort
                    if not effort:
                        lvl = (options.get('thinking_level') or "medium").lower()
                        effort = "low" if lvl == "low" else "high" if lvl == "high" else "medium"
                    effort = _normalize_reasoning_effort(model_key_l, effort)
                    kwargs['reasoning'] = {"effort": effort}
                    kwargs['reasoning']["summary"] = "auto"
                    log_force(f"Reasoning config: {kwargs['reasoning']}")

                log_force(f"Responses API Params: {kwargs.keys()}")
                pub("status", "APIへ送信完了。モデルが応答を生成中です...")
                _mark_provider_request_started()
                stream = client.responses.create(**kwargs)
                search_reported = False
                saw_reasoning_summary_delta = False
                response_id = None
                pending_response_mcp_calls = []
                collected_sources = []
                seen_source_urls = set()
                sources_emitted = False
                final_openai_usage = None

                def _queue_response_mcp_call(call_id, call_name, call_args):
                    """元ストリームを読み切った後に実行するMCP function callを積む。"""
                    if not str(call_name or "").startswith("mcp__"):
                        return
                    key = (str(call_id or ""), str(call_name or ""))
                    if any((c["id"], c["name"]) == key for c in pending_response_mcp_calls):
                        return
                    pending_response_mcp_calls.append({
                        "id": call_id,
                        "name": call_name,
                        "arguments": call_args,
                    })

                def _response_item_fields(item):
                    if isinstance(item, dict):
                        return (
                            item.get("type"),
                            item.get("call_id") or item.get("id"),
                            item.get("name"),
                            item.get("arguments"),
                        )
                    return (
                        getattr(item, "type", None),
                        getattr(item, "call_id", None) or getattr(item, "id", None),
                        getattr(item, "name", None),
                        getattr(item, "arguments", None),
                    )

                def _consume_response_mcp_followup(followup_stream):
                    """MCP結果を渡したResponses APIの続きのストリームを読む。"""
                    nonlocal response_id, full_res, thought_accumulated
                    next_calls = []

                    def _queue_next_call(call_id, call_name, call_args):
                        if not str(call_name or "").startswith("mcp__"):
                            return
                        key = (str(call_id or ""), str(call_name or ""))
                        if any((c["id"], c["name"]) == key for c in next_calls):
                            return
                        next_calls.append({
                            "id": call_id,
                            "name": call_name,
                            "arguments": call_args,
                        })

                    for followup_chunk in followup_stream:
                        _latency_mark_once(job_id, "provider_first_chunk_ms")
                        if check_stop():
                            break
                        if isinstance(followup_chunk, dict):
                            followup_type = followup_chunk.get("type")
                            followup_response = followup_chunk.get("response")
                            followup_item = followup_chunk.get("item")
                            followup_delta = followup_chunk.get("delta")
                        else:
                            followup_type = getattr(followup_chunk, "type", None)
                            followup_response = getattr(followup_chunk, "response", None)
                            followup_item = getattr(followup_chunk, "item", None)
                            followup_delta = getattr(followup_chunk, "delta", None)

                        if followup_type == "response.created":
                            response_id = (
                                (followup_response.get("id") if isinstance(followup_response, dict) else getattr(followup_response, "id", None))
                                or response_id
                            )
                        elif followup_type == "response.output_text.delta" and followup_delta:
                            full_res += followup_delta
                            pub("content", followup_delta)
                        elif followup_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta") and followup_delta:
                            thought_accumulated += followup_delta
                            pub("thought", followup_delta)
                        elif followup_type == "response.output_item.done":
                            item_type, item_call_id, item_name, item_args = _response_item_fields(followup_item)
                            if item_type == "function_call":
                                _queue_next_call(item_call_id, item_name, item_args)
                        elif followup_type == "response.completed":
                            response_id = (
                                (followup_response.get("id") if isinstance(followup_response, dict) else getattr(followup_response, "id", None))
                                or response_id
                            )
                            output_items = (
                                followup_response.get("output")
                                if isinstance(followup_response, dict)
                                else getattr(followup_response, "output", None)
                            )
                            for output_item in output_items or []:
                                item_type, item_call_id, item_name, item_args = _response_item_fields(output_item)
                                if item_type == "function_call":
                                    _queue_next_call(item_call_id, item_name, item_args)
                    return next_calls

                def _execute_response_mcp_call(call):
                    """MCP callを実行し、Responses APIへ返すfunction_call_outputを作る。"""
                    nonlocal full_res
                    call_id = call.get("id")
                    call_name = call.get("name")
                    raw_args = call.get("arguments")
                    try:
                        args_json = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
                        if not isinstance(args_json, dict):
                            raise ValueError("arguments must be a JSON object")
                        if resp_mcp_env is None:
                            result = f"Error: MCP tools are not available: {call_name}"
                        else:
                            mcp_text, mcp_out = resp_mcp_env.execute(call_name, args_json)
                            result = mcp_text
                            if mcp_out.get("ok"):
                                _md = f"\n\n> 🔧 **MCPツール実行:** `{call_name}` を実行しました。\n"
                                full_res += _md
                                pub("content", _md)
                            elif mcp_out.get("rejected"):
                                _md = f"\n\n> 🚫 **MCPツール実行はユーザーにより拒否されました:** `{call_name}`\n"
                                full_res += _md
                                pub("content", _md)
                        return {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": str(result or "Tool executed."),
                        }
                    except Exception as exc:
                        return {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": f"Error executing MCP tool: {exc}",
                        }

                def _add_source(title, url):
                    if not url or url in seen_source_urls:
                        return
                    seen_source_urls.add(url)
                    collected_sources.append((title or url, url))

                def _collect_sources_from_annotations(ann_list):
                    for ann in ann_list or []:
                        if isinstance(ann, dict):
                            a_type = ann.get('type')
                            a_url = ann.get('url') or ann.get('source') or ann.get('link')
                            a_title = ann.get('title') or a_url
                        else:
                            a_type = getattr(ann, 'type', None)
                            a_url = getattr(ann, 'url', None)
                            a_title = getattr(ann, 'title', None) or a_url
                        if a_url and (a_type is None or "citation" in str(a_type).lower() or "annotation" in str(a_type).lower()):
                            _add_source(a_title, a_url)

                def _collect_sources_from_web_search_call(item):
                    action = item.get('action') if isinstance(item, dict) else getattr(item, 'action', None)
                    sources = None
                    if isinstance(action, dict):
                        sources = action.get('sources')
                    else:
                        sources = getattr(action, 'sources', None)
                    for src in sources or []:
                        if isinstance(src, dict):
                            _add_source(src.get('title') or src.get('name'), src.get('url'))
                        else:
                            _add_source(getattr(src, 'title', None) or getattr(src, 'name', None), getattr(src, 'url', None))

                def _emit_sources_once():
                    nonlocal sources_emitted
                    if sources_emitted or not collected_sources:
                        return
                    sources_emitted = True
                    sources_text = "\n\n**Sources:**\n"
                    for title, url in collected_sources:
                        sources_text += f"- [{title}]({url})\n"
                    full_res_add = sources_text
                    pub("content", sources_text)
                    return full_res_add

                for chunk in stream:
                    _latency_mark_once(job_id, "provider_first_chunk_ms")
                    if check_stop(): break
                    if isinstance(chunk, dict):
                        event_type = chunk.get('type')
                        usage = chunk.get('usage')
                    else:
                        event_type = getattr(chunk, 'type', None)
                        usage = getattr(chunk, 'usage', None)
                    
                    if usage:
                        final_openai_usage = usage

                    if response_id is None:
                        if isinstance(chunk, dict):
                            response_id = chunk.get('response_id') or response_id
                        else:
                            response_id = getattr(chunk, 'response_id', None) or response_id

                    if event_type == "response.created":
                        resp = chunk.get('response') if isinstance(chunk, dict) else getattr(chunk, 'response', None)
                        if isinstance(resp, dict):
                            response_id = resp.get('id') or response_id
                        else:
                            response_id = getattr(resp, 'id', None) or response_id
                        continue

                    if event_type in ("response.web_search_call.in_progress", "response.web_search_call.searching"):
                        if not search_reported:
                            pub("search_status", "searching")
                            search_reported = True
                    elif event_type == "response.web_search_call.completed":
                        if search_reported:
                            pub("search_status", "done")
                            search_reported = False
                    elif event_type == "response.output_text.delta":
                        text_delta = chunk.get('delta') if isinstance(chunk, dict) else getattr(chunk, 'delta', None)
                        if text_delta:
                            if search_reported:
                                pub("search_status", "done")
                                search_reported = False
                            full_res += text_delta
                            pub("content", text_delta)
                    elif event_type == "response.output_text.annotation.added":
                        ann = chunk.get('annotation') if isinstance(chunk, dict) else getattr(chunk, 'annotation', None)
                        if ann:
                            _collect_sources_from_annotations([ann])
                    elif event_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                        reasoning_delta = chunk.get('delta') if isinstance(chunk, dict) else getattr(chunk, 'delta', None)
                        if reasoning_delta:
                            log_force(f"Reasoning Delta: {reasoning_delta[:50]}...")
                            if event_type == "response.reasoning_summary_text.delta":
                                saw_reasoning_summary_delta = True
                            thought_accumulated += reasoning_delta
                            pub("thought", reasoning_delta)
                    elif event_type in ("response.reasoning_text.done", "response.reasoning_summary_text.done"):
                        reasoning_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)
                        if reasoning_text:
                            if event_type == "response.reasoning_summary_text.done":
                                saw_reasoning_summary_delta = True
                            thought_accumulated += reasoning_text
                            pub("thought", reasoning_text)
                    elif event_type in ("response.reasoning_summary_part.added", "response.reasoning_summary_part.done"):
                        part = chunk.get('part') if isinstance(chunk, dict) else getattr(chunk, 'part', None)
                        if isinstance(part, dict):
                            part_type = part.get('type')
                            part_text = part.get('text')
                        else:
                            part_type = getattr(part, 'type', None) if part else None
                            part_text = getattr(part, 'text', None) if part else None
                        if part_type == "summary_text" and part_text:
                            if not saw_reasoning_summary_delta:
                                thought_accumulated += part_text
                                pub("thought", part_text)
                    elif event_type in ("response.content_part.added", "response.content_part.done"):
                        part = chunk.get('part') if isinstance(chunk, dict) else getattr(chunk, 'part', None)
                        if isinstance(part, dict):
                            part_type = part.get('type')
                            part_text = part.get('text')
                        else:
                            part_type = getattr(part, 'type', None) if part else None
                            part_text = getattr(part, 'text', None) if part else None
                        if part_type in ("summary_text", "reasoning_text") and part_text:
                            thought_accumulated += part_text
                            pub("thought", part_text)
                    elif event_type == "response.output_item.added":
                        item = chunk.get('item') if isinstance(chunk, dict) else getattr(chunk, 'item', None)
                        if item:
                            if isinstance(item, dict):
                                i_type = item.get('type')
                                i_name = item.get('name')
                            else:
                                i_type = getattr(item, 'type', None)
                                i_name = getattr(item, 'name', None)
                            
                            if i_type in ("function_call", "tool_call") or (i_name and "search" in i_name.lower()):
                                if not search_reported:
                                    pub("search_status", "searching")
                                    search_reported = True
                            if i_type == "reasoning":
                                summary_parts = item.get('summary') if isinstance(item, dict) else getattr(item, 'summary', None)
                                if summary_parts:
                                    for part in summary_parts:
                                        if isinstance(part, dict):
                                            p_type = part.get('type')
                                            p_text = part.get('text')
                                        else:
                                            p_type = getattr(part, 'type', None)
                                            p_text = getattr(part, 'text', None)
                                        if p_type == "summary_text" and p_text:
                                            saw_reasoning_summary_delta = True
                                            thought_accumulated += p_text
                                            pub("thought", p_text)

                    elif event_type == "response.output_item.done":
                        item = chunk.get('item') if isinstance(chunk, dict) else getattr(chunk, 'item', None)
                        if isinstance(item, dict):
                            item_type = item.get('type')
                            summary_parts = item.get('summary')
                            tool_call_id = item.get('call_id') or item.get('id')
                            call_name = item.get('name')
                            call_args = item.get('arguments')
                        else:
                            item_type = getattr(item, 'type', None)
                            summary_parts = getattr(item, 'summary', None)
                            tool_call_id = getattr(item, 'call_id', None) or getattr(item, 'id', None)
                            call_name = getattr(item, 'name', None)
                            call_args = getattr(item, 'arguments', None)

                        # MCPの継続リクエストは、元のResponsesストリームを読み切って
                        # response.completed を受け取ってから送る。ストリーム途中で
                        # responses.create を再入すると、ツール後の回答が欠落する。
                        if item_type == "function_call" and str(call_name or "").startswith("mcp__"):
                            _queue_response_mcp_call(tool_call_id, call_name, call_args)
                            continue

                        if item_type == "function_call" and call_name in ("execute_python", "create_file", "edit_file"):
                            try:
                                args_json = json.loads(call_args or "{}")
                                if call_name == "create_file":
                                    create_result = _execute_create_file_tool(
                                        user_id, args_json, user_config.get('enable_e2ee')
                                    )
                                    result = _create_file_tool_result_text(create_result)
                                    if create_result.get("ok"):
                                        created_file_rel = f"{user_id}/{create_result['filename']}"
                                        if created_file_rel not in generated_images:
                                            generated_images.append(created_file_rel)
                                        file_link_md = (
                                            f"\n📄 **ファイルを作成しました:** [{create_result.get('display_name')}]({create_result.get('url')})\n"
                                        )
                                        full_res += file_link_md
                                        pub("content", file_link_md)
                                elif call_name == "edit_file":
                                    edit_result = _execute_edit_file_tool(
                                        user_id,
                                        args_json,
                                        user_config.get('enable_e2ee'),
                                        loaded_files=loaded_files,
                                        history=history,
                                        thread_id=thread_id,
                                    )
                                    result = _create_file_tool_result_text(edit_result, "edit_file", "編集")
                                    if edit_result.get("ok"):
                                        edited_file_rel = f"{user_id}/{edit_result['filename']}"
                                        if edited_file_rel not in generated_images:
                                            generated_images.append(edited_file_rel)
                                        file_link_md = (
                                            f"\n📄 **ファイルを編集しました:** [{edit_result.get('display_name')}]({edit_result.get('url')})\n"
                                        )
                                        full_res += file_link_md
                                        pub("content", file_link_md)
                                else:
                                    code = args_json.get('code', '')
                                    if code:
                                        pub("content", f"\n```python\n{code}\n```\n")
                                        result = safe_execute_python(code)
                                        pub("content", f"\n**Output:**\n```\n{result}\n```\n")
                                        full_res += f"\n```python\n{code}\n```\n\n**Output:**\n```\n{result}\n```\n"
                                        full_res += f"\n```pyexec\n{json.dumps({'code': code, 'output': result})}\n```\n"
                                        pub("python", {"id": tool_call_id or f"py_{int(time.time()*1000)}_{os.urandom(3).hex()}", "code": code, "output": result})
                                    else:
                                        result = "Error: No code provided."
                                if response_id and tool_call_id:
                                    _mark_provider_request_started()
                                    tool_stream = client.responses.create(
                                        model=model_key,
                                        previous_response_id=response_id,
                                        input=[{
                                            "type": "function_call_output",
                                            "call_id": tool_call_id,
                                            "output": result
                                        }],
                                        stream=True
                                    )
                                    for tchunk in tool_stream:
                                        _latency_mark_once(job_id, "provider_first_chunk_ms")
                                        if check_stop(): break
                                        if isinstance(tchunk, dict):
                                            t_event = tchunk.get('type')
                                        else:
                                            t_event = getattr(tchunk, 'type', None)
                                        if t_event == "response.output_text.delta":
                                            t_delta = tchunk.get('delta') if isinstance(tchunk, dict) else getattr(tchunk, 'delta', None)
                                            if t_delta:
                                                full_res += t_delta
                                                pub("content", t_delta)
                                        elif t_event in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                                            t_reason = tchunk.get('delta') if isinstance(tchunk, dict) else getattr(tchunk, 'delta', None)
                                            if t_reason:
                                                thought_accumulated += t_reason
                                                pub("thought", t_reason)
                            except Exception as e:
                                pub("error", f"Tool Error: {e}")

                        if item_type == "reasoning" and summary_parts:
                            for part in summary_parts:
                                if isinstance(part, dict):
                                    part_type = part.get('type')
                                    part_text = part.get('text')
                                else:
                                    part_type = getattr(part, 'type', None)
                                    part_text = getattr(part, 'text', None)
                                if part_type == "summary_text" and part_text:
                                    saw_reasoning_summary_delta = True
                                    thought_accumulated += part_text
                                    pub("thought", part_text)
                        if item_type == "reasoning":
                            content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                            if content_parts:
                                for part in content_parts:
                                    if isinstance(part, dict):
                                        p_type = part.get('type')
                                        p_text = part.get('text')
                                    else:
                                        p_type = getattr(part, 'type', None)
                                        p_text = getattr(part, 'text', None)
                                    if p_type == "reasoning_text" and p_text:
                                        thought_accumulated += p_text
                                        pub("thought", p_text)
                    else:
                        if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
                            if search_reported:
                                pub("search_status", "done")
                                search_reported = False
                            full_res += chunk.output_text_delta
                            pub("content", chunk.output_text_delta)

                        if hasattr(chunk, 'citations') and chunk.citations:
                            citations_text = "\n\n**Sources:**\n"
                            for c in chunk.citations:
                                title = getattr(c, 'title', 'Source')
                                url = getattr(c, 'url', '#')
                                citations_text += f"- [{title}]({url})\n"
                            full_res += citations_text
                            pub("content", citations_text)

                        reasoning_delta = getattr(chunk, 'output_reasoning_text_delta', None)
                        if reasoning_delta:
                            thought_accumulated += reasoning_delta
                            pub("thought", reasoning_delta)
                    if event_type == "response.completed":
                        resp = chunk.get('response') if isinstance(chunk, dict) else getattr(chunk, 'response', None)
                        if isinstance(resp, dict):
                            response_id = resp.get('id') or response_id
                            resp_usage = resp.get('usage')
                            output_items = resp.get('output')
                        else:
                            response_id = getattr(resp, 'id', None) or response_id
                            resp_usage = getattr(resp, 'usage', None) if resp else None
                            output_items = getattr(resp, 'output', None) if resp else None
                        if resp_usage:
                            final_openai_usage = resp_usage
                        if output_items:
                            for item in output_items:
                                item_type, item_call_id, item_name, item_args = _response_item_fields(item)
                                content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                                if item_type == "function_call":
                                    _queue_response_mcp_call(item_call_id, item_name, item_args)
                                if item_type == "web_search_call":
                                    _collect_sources_from_web_search_call(item)
                                if content_parts:
                                    for part in content_parts:
                                        if isinstance(part, dict):
                                            p_type = part.get('type')
                                            p_anns = part.get('annotations')
                                        else:
                                            p_type = getattr(part, 'type', None)
                                            p_anns = getattr(part, 'annotations', None)
                                        if p_type in ("output_text", "text") and p_anns:
                                            _collect_sources_from_annotations(p_anns)
                        if output_items and not saw_reasoning_summary_delta:
                            for item in output_items:
                                if isinstance(item, dict):
                                    item_type = item.get('type')
                                    summary_parts = item.get('summary')
                                else:
                                    item_type = getattr(item, 'type', None)
                                    summary_parts = getattr(item, 'summary', None)
                                if item_type == "reasoning":
                                    if summary_parts:
                                        for part in summary_parts:
                                            if isinstance(part, dict):
                                                text = part.get('text')
                                            else:
                                                text = getattr(part, 'text', None)
                                            if text:
                                                thought_accumulated += text
                                                pub("thought", text)
                                    content_parts = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
                                    if content_parts:
                                        for part in content_parts:
                                            if isinstance(part, dict):
                                                p_type = part.get('type')
                                                p_text = part.get('text')
                                            else:
                                                p_type = getattr(part, 'type', None)
                                                p_text = getattr(part, 'text', None)
                                            if p_type == "reasoning_text" and p_text:
                                                thought_accumulated += p_text
                                                pub("thought", p_text)
                        if collected_sources:
                            appended = _emit_sources_once()
                            if appended:
                                full_res += appended

                # MCP function callは元のストリーム完了後に結果を返し、回答生成を
                # 必要な回数だけ継続する。1回だけの継続では、ツール結果を受けた
                # モデルが別のMCPツールを続けて呼ぶケースで再び無言終了する。
                for _mcp_round in range(8):
                    if not pending_response_mcp_calls:
                        break
                    if not response_id:
                        pub("error", "MCPツールの実行後にResponses APIの応答IDを取得できませんでした。")
                        break
                    response_tool_outputs = [
                        _execute_response_mcp_call(call)
                        for call in pending_response_mcp_calls
                    ]
                    pending_response_mcp_calls = []
                    try:
                        _mark_provider_request_started()
                        followup_stream = client.responses.create(
                            model=model_key,
                            previous_response_id=response_id,
                            input=response_tool_outputs,
                            stream=True,
                            store=store_flag,
                        )
                        pending_response_mcp_calls = _consume_response_mcp_followup(followup_stream)
                    except Exception as exc:
                        pub("error", f"MCPツール実行後の回答生成に失敗しました: {exc}")
                        break
                else:
                    if pending_response_mcp_calls:
                        pub("error", "MCPツール呼び出しが上限回数に達しました。")

                # Fallback: retrieve full response if no reasoning summary surfaced in stream
                if enable_reasoning and not thought_accumulated and response_id:
                    try:
                        _mark_provider_request_started()
                        resp_full = client.responses.retrieve(response_id)
                        resp_usage = getattr(resp_full, 'usage', None)
                        if resp_usage:
                            final_openai_usage = resp_usage
                        output_items = getattr(resp_full, 'output', None)
                        if output_items:
                            for item in output_items:
                                if isinstance(item, dict):
                                    item_type = item.get('type')
                                    summary_parts = item.get('summary')
                                    content_parts = item.get('content')
                                else:
                                    item_type = getattr(item, 'type', None)
                                    summary_parts = getattr(item, 'summary', None)
                                    content_parts = getattr(item, 'content', None)
                                if item_type == "reasoning":
                                    if summary_parts:
                                        for part in summary_parts:
                                            if isinstance(part, dict):
                                                text = part.get('text')
                                            else:
                                                text = getattr(part, 'text', None)
                                            if text:
                                                thought_accumulated += text
                                                pub("thought", text)
                                    if content_parts:
                                        for part in content_parts:
                                            if isinstance(part, dict):
                                                p_type = part.get('type')
                                                p_text = part.get('text')
                                            else:
                                                p_type = getattr(part, 'type', None)
                                                p_text = getattr(part, 'text', None)
                                            if p_type == "reasoning_text" and p_text:
                                                thought_accumulated += p_text
                                                pub("thought", p_text)
                                if item_type == "web_search_call":
                                    _collect_sources_from_web_search_call(item)
                                if content_parts:
                                    for part in content_parts:
                                        if isinstance(part, dict):
                                            p_type = part.get('type')
                                            p_anns = part.get('annotations')
                                        else:
                                            p_type = getattr(part, 'type', None)
                                            p_anns = getattr(part, 'annotations', None)
                                        if p_type in ("output_text", "text") and p_anns:
                                            _collect_sources_from_annotations(p_anns)
                            if collected_sources and not sources_emitted:
                                appended = _emit_sources_once()
                                if appended:
                                    full_res += appended
                    except Exception as e:
                        log_force(f"Reasoning retrieve fallback failed: {e}")
                elif enable_reasoning and not thought_accumulated:
                    log_force("Reasoning summary missing after stream and retrieve fallback.")

            if is_grok and grok_reasoning_supported and not thought_accumulated:
                thought_accumulated = " "
            final_content = full_res
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

            thought_payload = {}
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

            gem_uuid_val = options.get('gem_uuid')
            gem_name_val = None
            if gem_uuid_val:
                gem = Gem.query.filter_by(uuid=gem_uuid_val).first()
                if gem:
                    gem_name_val = gem.name
            msg_entry = Message(
                thread_id=thread_id, role='assistant', content=final_content, 
                model=model_key, image_url=json.dumps(generated_images) if generated_images else None, 
                thought_data=final_thought, tokens_out=assistant_tokens_out, tokens=sum_token_counts(None, assistant_tokens_out), 
                is_encrypted=is_enc, thought_signature=final_signature,
                parent_id=message_id,
                tokens_thought=tokens_thought_val,
                gem_uuid=gem_uuid_val,
                gem_name=gem_name_val
            )
            db.session.add(msg_entry)
            th = Thread.query.get(thread_id)
            if th:
                th.updated_at = datetime.utcnow()
                th.last_model = model_key
                th.last_gem_uuid = options.get('gem_uuid')
            safe_db_commit()
            pub("done", "OK")

        except Exception as e:
            logger.exception("Worker Error")
            log_force(f"Worker Exception: {e}")
            err_msg = str(e)
            try:
                if is_gem:
                    err_msg = _format_gemini_runtime_error(e, gemini_backend_mode)
            except Exception:
                pass
            pub("error", err_msg)
        finally:
            _latency_mark_once(job_id, "worker_done_ms")
            r.delete(f"stop_job:{job_id}")
            try:
                r.delete(f"pending_job:{user_id}:{thread_id}")
            except Exception:
                pass
            try:
                r.delete(f"stream_acc:{job_id}:content")
                r.delete(f"stream_acc:{job_id}:thought")
                r.delete(f"stream_acc:{job_id}:search")
                r.delete(f"stream_acc:{job_id}:status")
                r.delete(f"stream_acc:{job_id}:final")
                r.delete(f"stream_acc:{job_id}:python")
                r.delete(f"stream_acc:{job_id}:coding_diff")
                r.delete(f"stream_acc:{job_id}:mcp")
                r.delete(f"mcp_decision:{job_id}")
            except Exception:
                pass
