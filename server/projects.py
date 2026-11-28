"""Project shared-file catalog, ``read_project_file`` tool schema and execution.

Mirrors ``server/skills.py``: only the list of shared files (name, type, size,
optional note) is shown to the model in the system prompt; the model calls
``read_project_file`` on demand to read file contents in chunks.  Paths that
cannot run tools (Batch API, OpenAI Search models, etc.) receive extracted text
excerpts embedded in the catalog instead.

For image files, the project's ``use_vision_for_images`` toggle governs whether
the configured Vision model is invoked; when OFF, image contents are never read
or billed and only safe file metadata plus a brief notice are returned.
"""

PROJECT_FILE_TOOL_NAME = "read_project_file"
PROJECT_NAME_MAX_CHARS = 100
PROJECT_DESCRIPTION_MAX_CHARS = 1000
PROJECT_INSTRUCTION_MAX_CHARS = 100_000
PROJECT_FILE_NOTE_MAX_CHARS = 300
PROJECT_MAX_FILES = 50
PROJECT_FILE_READ_DEFAULT_LIMIT = 20_000
PROJECT_FILE_READ_MAX_LIMIT = 30_000
PROJECT_FILE_CATALOG_TOTAL_LIMIT = 60_000
PROJECT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"}

PROJECT_FILE_TOOL_DESCRIPTION = (
    "Read the contents of one of the current project's shared files. "
    "Call this when the user asks about a project file or when a listed file "
    "appears relevant to the request. Use the exact file name from the project "
    "files list. For long files, pass offset to read subsequent chunks. "
    "Never invent a file name that is not listed."
)

PROJECT_FILE_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Exact file name from the project shared files list.",
        },
        "offset": {
            "type": "integer",
            "description": "Optional zero-based character offset to resume reading a long file.",
        },
        "limit": {
            "type": "integer",
            "description": "Optional max number of characters to return (default 20000, max 30000).",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


def resolve_project_for_user(project_ref, user_id):
    """Resolve a project by UUID or integer ID scoped to ``user_id``."""
    if not project_ref or not user_id:
        return None
    ident = str(project_ref).strip()
    if not ident:
        return None
    proj = Project.query.filter_by(uuid=ident, user_id=user_id).first()
    if proj:
        return proj
    if ident.isdigit():
        return Project.query.filter_by(id=int(ident), user_id=user_id).first()
    return None


def resolve_project_vision_model(project, user):
    """Return the Vision model ID to use when project image analysis is enabled."""
    chosen = str(getattr(project, "vision_model", None) or "").strip() if project else ""
    if chosen and chosen in ALL_VALID_MODEL_IDS:
        return chosen
    user_default = str(getattr(user, "default_vision_model", None) or "").strip() if user else ""
    if user_default and user_default in ALL_VALID_MODEL_IDS:
        return user_default
    return "gemini-3-flash-preview"


def _is_project_image_path(rel_path, mime=""):
    ext = os.path.splitext(str(rel_path or ""))[1].lower()
    if ext in PROJECT_IMAGE_EXTS:
        return True
    return str(mime or "").lower().startswith("image/")


def _format_project_file_size(size_bytes):
    size = int(size_bytes or 0)
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def get_project_files_for_prompt(project, user_id):
    """Return normalized metadata dicts for existing shared files of ``project``."""
    if not project:
        return []
    rows = (
        ProjectFile.query
        .filter_by(project_id=project.id, user_id=user_id)
        .order_by(ProjectFile.created_at.asc(), ProjectFile.id.asc())
        .all()
    )
    if not rows:
        return []
    label_map = _get_user_file_label_map(user_id)
    entries = []
    seen_names = {}
    for row in rows:
        rel_path = _normalize_upload_ref(row.rel_path)
        if not rel_path or not rel_path.startswith(f"{user_id}/"):
            continue
        info = _get_file_disk_info(rel_path)
        if not info.get("exists"):
            continue
        base_name = os.path.basename(rel_path)
        raw_name = str(row.display_name or label_map.get(rel_path) or base_name).strip() or base_name
        # Disambiguate duplicate display names deterministically so the model can address each file.
        lower_key = raw_name.casefold()
        count = seen_names.get(lower_key, 0) + 1
        seen_names[lower_key] = count
        catalog_name = raw_name if count == 1 else f"{raw_name} ({count})"
        mime_guess = mimetypes.guess_type(rel_path)[0] or "application/octet-stream"
        mime = _normalize_media_mime(rel_path, mime_guess)
        is_image = _is_project_image_path(rel_path, mime)
        entries.append({
            "id": row.id,
            "rel_path": rel_path,
            "name": catalog_name,
            "display_name": raw_name,
            "original_filename": base_name,
            "note": str(row.note or "").replace("\r\n", "\n").strip()[:PROJECT_FILE_NOTE_MAX_CHARS],
            "mime": mime,
            "size": int(info.get("size") or 0),
            "is_image": is_image,
            "mtime": info.get("mtime"),
        })
    return entries


def _extract_project_file_text(rel_path, mime, data):
    """Extract readable text from supported project file types, or return (None, kind)."""
    lower_path = str(rel_path or "").lower()
    ext = os.path.splitext(lower_path)[1]
    if _is_project_image_path(lower_path, mime):
        return None, "image"
    if lower_path.endswith(".pdf") or mime == "application/pdf":
        return (_extract_text_from_pdf(data) or ""), "pdf"
    if lower_path.endswith(".docx"):
        return (_extract_docx_as_numbered(data) or ""), "docx"
    if lower_path.endswith(".pptx"):
        return (_extract_text_from_pptx(data) or ""), "pptx"
    if lower_path.endswith((".xlsx", ".xlsm")):
        return (_extract_xlsx_as_tsv(data, include_column_headers=True) or ""), "xlsx"
    if str(mime or "").startswith("text/") or ext in _TEXT_LIKE_UPLOAD_EXTS:
        return (_decode_text_bytes_for_prompt(data) or ""), "text"
    return None, "binary"


def build_project_files_catalog(files, include_contents=False, user_id=None, use_vision_for_images=False):
    """Render the shared files list shown to the model."""
    rows = list(files or [])
    if not rows:
        return ""
    blocks = []
    used_chars = 0
    for item in rows:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        size_label = _format_project_file_size(item.get("size"))
        kind_label = "image" if item.get("is_image") else (item.get("mime") or "file")
        if item.get("is_image"):
            vision_tag = "Vision: ON" if use_vision_for_images else "Vision: OFF (metadata only)"
            meta_str = f"{kind_label}, {size_label}, {vision_tag}"
        else:
            meta_str = f"{kind_label}, {size_label}"
        note = str(item.get("note") or "").strip()
        summary = f"- {name} ({meta_str})" + (f": {note}" if note else "")
        if include_contents and user_id and not item.get("is_image"):
            info = _get_file_disk_info(item["rel_path"])
            raw = _load_user_file_bytes(item["rel_path"], info) if info.get("exists") else None
            extracted = None
            if raw:
                extracted, _kind = _extract_project_file_text(item["rel_path"], item.get("mime"), raw)
            if extracted:
                remaining = PROJECT_FILE_CATALOG_TOTAL_LIMIT - used_chars
                if remaining <= 200:
                    blocks.append("...(additional project file contents omitted)")
                    break
                excerpt = extracted[: min(PROJECT_FILE_READ_DEFAULT_LIMIT, remaining)]
                truncated = len(extracted) > len(excerpt)
                block = f"### {name} ({meta_str})\n{excerpt}" + (
                    "\n[Content truncated]" if truncated else ""
                )
                used_chars += len(block)
                blocks.append(block)
                continue
        blocks.append(summary)
    return "\n".join(blocks)


def render_project_files_notice(template_text, project, files, include_contents=False, user_id=None):
    """Expand ``{{project_name}}`` / ``{{project_files}}`` / ``{{project_file_count}}``."""
    rows = list(files or [])
    if not rows:
        return ""
    use_vision = bool(getattr(project, "use_vision_for_images", False)) if project else False
    catalog = build_project_files_catalog(
        rows,
        include_contents=include_contents,
        user_id=user_id,
        use_vision_for_images=use_vision,
    )
    proj_name = str(getattr(project, "name", "") or "").strip()
    rendered = str(template_text or "").replace("\r\n", "\n").strip()
    replaced_files = False
    for token in ("{{project_files}}", "{project_files}"):
        if token in rendered:
            rendered = rendered.replace(token, catalog)
            replaced_files = True
    for token in ("{{project_file_count}}", "{project_file_count}"):
        if token in rendered:
            rendered = rendered.replace(token, str(len(rows)))
    for token in ("{{project_name}}", "{project_name}"):
        if token in rendered:
            rendered = rendered.replace(token, proj_name)
    if catalog and not replaced_files:
        rendered = f"{rendered}\n\n{catalog}".strip() if rendered else catalog
    return rendered.strip()


def build_project_file_tool_definition():
    return {
        "name": PROJECT_FILE_TOOL_NAME,
        "description": PROJECT_FILE_TOOL_DESCRIPTION,
        "parameters": dict(PROJECT_FILE_TOOL_PARAMETERS),
    }


def build_project_file_tool_openai_chat():
    return {"type": "function", "function": build_project_file_tool_definition()}


def build_project_file_tool_openai_responses():
    definition = build_project_file_tool_definition()
    return {
        "type": "function",
        "name": definition["name"],
        "description": definition["description"],
        "parameters": definition["parameters"],
    }


def build_project_file_tool_anthropic():
    return {
        "name": PROJECT_FILE_TOOL_NAME,
        "description": PROJECT_FILE_TOOL_DESCRIPTION,
        "input_schema": dict(PROJECT_FILE_TOOL_PARAMETERS),
    }


def execute_project_file_tool(
    user_id,
    project,
    project_files,
    name,
    offset=0,
    limit=PROJECT_FILE_READ_DEFAULT_LIMIT,
    vision_fn=None,
):
    """Return ``(text_for_model, meta)`` for one ``read_project_file`` call.

    File contents are never logged by this module.  ``meta`` contains only safe
    status flags, canonical name, file kind, and character counts.
    """
    rows = list(project_files or [])
    query = str(name or "").strip()
    target = next((f for f in rows if str(f.get("name") or "") == query), None)
    if target is None and query:
        lowered = query.casefold()
        target = next(
            (f for f in rows if str(f.get("name") or "").casefold() == lowered),
            None,
        )
    if target is None and query:
        lowered = query.casefold()
        target = next(
            (f for f in rows if str(f.get("display_name") or "").casefold() == lowered),
            None,
        )
    if target is None:
        available = ", ".join(str(f.get("name") or "") for f in rows[:50] if f.get("name"))
        msg = f"Error: no project file named {query!r}."
        if available:
            msg += f" Available project files: {available}"
        return msg, {"ok": False, "found": False, "name": query}

    try:
        offset_i = max(0, int(offset or 0))
    except Exception:
        offset_i = 0
    try:
        limit_i = int(limit) if limit is not None else PROJECT_FILE_READ_DEFAULT_LIMIT
    except Exception:
        limit_i = PROJECT_FILE_READ_DEFAULT_LIMIT
    limit_i = max(500, min(limit_i, PROJECT_FILE_READ_MAX_LIMIT))

    rel_path = target.get("rel_path") or ""
    if not rel_path or not rel_path.startswith(f"{user_id}/"):
        return "Error: access denied.", {"ok": False, "found": True, "name": target.get("name")}
    info = _get_file_disk_info(rel_path)
    if not info.get("exists"):
        return f"Error: project file {target.get('name')!r} was not found on disk.", {
            "ok": False, "found": False, "name": target.get("name")
        }

    mime = target.get("mime") or "application/octet-stream"
    size_label = _format_project_file_size(info.get("size") or target.get("size"))

    # Image files obey the project's explicit Vision model toggle.
    if target.get("is_image") or _is_project_image_path(rel_path, mime):
        use_vision = bool(getattr(project, "use_vision_for_images", False)) if project else False
        if not use_vision:
            meta_reply = (
                f"Project image file: {target['name']} (mime={mime}, size={size_label}).\n"
                "Vision model analysis is disabled for this project, so the image contents were not read. "
                "Ask the user to enable Vision model analysis in the project settings or attach the image directly to the chat."
            )
            return meta_reply, {
                "ok": True,
                "found": True,
                "name": target["name"],
                "kind": "image_metadata_only",
                "vision_used": False,
                "chars": len(meta_reply),
            }
        data = _load_user_file_bytes(rel_path, info)
        if not data:
            return f"Error: failed to load image file {target['name']!r}.", {
                "ok": False, "found": True, "name": target["name"], "kind": "image"
            }
        if not callable(vision_fn):
            return (
                f"Project image file: {target['name']} (mime={mime}, size={size_label}).\n"
                "Vision model analysis is enabled, but no Vision model runner is available on this route."
            ), {
                "ok": False, "found": True, "name": target["name"], "kind": "image", "vision_used": False
            }
        vision_text, vision_err = vision_fn(data, mime, target["name"])
        if vision_err or not vision_text:
            err_detail = vision_err or "Vision analysis returned no text."
            return (
                f"Project image file: {target['name']} (mime={mime}, size={size_label}).\n"
                f"Vision model analysis failed: {err_detail}"
            ), {
                "ok": False, "found": True, "name": target["name"], "kind": "image", "vision_used": True
            }
        full_text = str(vision_text)
        total_chars = len(full_text)
        chunk = full_text[offset_i: offset_i + limit_i]
        has_more = (offset_i + len(chunk)) < total_chars
        header = (
            f"Project image file (Vision analysis): {target['name']} "
            f"[chars {offset_i}-{offset_i + len(chunk)} of {total_chars}]"
        )
        if has_more:
            header += f" (more available; call read_project_file with offset={offset_i + len(chunk)})"
        return f"{header}\n\n{chunk}", {
            "ok": True,
            "found": True,
            "name": target["name"],
            "kind": "image_vision",
            "vision_used": True,
            "chars": len(chunk),
            "total_chars": total_chars,
            "has_more": has_more,
        }

    data = _load_user_file_bytes(rel_path, info)
    if data is None:
        return f"Error: failed to read project file {target['name']!r}.", {
            "ok": False, "found": True, "name": target["name"]
        }
    extracted, kind = _extract_project_file_text(rel_path, mime, data)
    if extracted is None:
        return (
            f"Project file: {target['name']} (mime={mime}, size={size_label}).\n"
            "This file format cannot be extracted as text via read_project_file."
        ), {
            "ok": False, "found": True, "name": target["name"], "kind": kind
        }
    total_chars = len(extracted)
    if total_chars == 0:
        return f"Project file: {target['name']} ({kind}, {size_label}) is empty or contains no extractable text.", {
            "ok": True, "found": True, "name": target["name"], "kind": kind, "chars": 0, "total_chars": 0, "has_more": False
        }
    chunk = extracted[offset_i: offset_i + limit_i]
    has_more = (offset_i + len(chunk)) < total_chars
    header = (
        f"Project file: {target['name']} ({kind}, {size_label}) "
        f"[chars {offset_i}-{offset_i + len(chunk)} of {total_chars}]"
    )
    if has_more:
        header += f" (more available; call read_project_file with offset={offset_i + len(chunk)})"
    return f"{header}\n\n{chunk}", {
        "ok": True,
        "found": True,
        "name": target["name"],
        "kind": kind,
        "chars": len(chunk),
        "total_chars": total_chars,
        "has_more": has_more,
    }


def build_gemini_project_file_tool(executor):
    """Build the google-genai callable used by Automatic Function Calling."""

    def read_project_file(name: str, offset: int = 0, limit: int = PROJECT_FILE_READ_DEFAULT_LIMIT) -> str:
        return executor(name, offset=offset, limit=limit)

    read_project_file.__name__ = PROJECT_FILE_TOOL_NAME
    read_project_file.__doc__ = PROJECT_FILE_TOOL_DESCRIPTION
    read_project_file.__annotations__ = {"name": str, "offset": int, "limit": int, "return": str}
    return read_project_file


def project_file_tool_supported(model_key, batch_mode=False, gemini_local_python=False):
    """Whether the current route can run ``read_project_file`` (mirrors ``skill_tool_supported``)."""
    return skill_tool_supported(
        model_key,
        batch_mode=batch_mode,
        gemini_local_python=gemini_local_python,
    )


def resolve_project_prompt_for_thread(th, user_id):
    """Return ``(project_obj, project_sys_prompt_or_None)`` for ``background_chat_task``."""
    if not th or not getattr(th, "project_id", None):
        return None, None
    project_obj = Project.query.filter_by(id=th.project_id, user_id=user_id).first()
    include_project = (
        th.include_project_instruction
        if getattr(th, "include_project_instruction", None) is not None
        else True
    )
    if project_obj is not None and include_project:
        raw_inst = getattr(project_obj, "instruction", None)
        if raw_inst and str(raw_inst).strip():
            return project_obj, str(raw_inst).strip()
    return project_obj, None


def combine_system_prompt_layers(
    forced_prompt, global_prompt, user_prompt, project_sys_prompt, local_sys_prompt, options
):
    """Assemble Global -> User -> [Project Instructions] -> [Chat Specific Instructions] -> mode prompts."""
    combined = "\n\n".join(str(p).strip() for p in (forced_prompt, global_prompt, user_prompt) if p and str(p).strip())
    if project_sys_prompt:
        combined = f"{combined}\n\n[Project Instructions]:\n{project_sys_prompt}" if combined else f"[Project Instructions]:\n{project_sys_prompt}"
    if local_sys_prompt:
        combined = f"{combined}\n\n[Chat Specific Instructions]:\n{local_sys_prompt}" if combined else local_sys_prompt
    if options.get("coding_mode"):
        combined = f"{combined}\n\n{CODING_MODE_SYSTEM_PROMPT}" if combined else CODING_MODE_SYSTEM_PROMPT
    if options.get("math_steps"):
        math_steps_prompt = build_math_steps_system_prompt(
            normalize_math_steps_config(options.get("math_steps_config"))
        )
        combined = f"{combined}\n\n{math_steps_prompt}" if combined else math_steps_prompt
    return combined


def make_background_project_file_executor(user, get_project_obj, get_project_files, pub_fn, log_fn):
    """Return the ``_execute_project_file_tool`` callable used by provider stream branches."""
    def _exec(file_name, offset=0, limit=PROJECT_FILE_READ_DEFAULT_LIMIT):
        proj = get_project_obj()
        vision_fn = (
            _make_project_vision_runner(user, proj, pub_fn=pub_fn)
            if (proj is not None and bool(getattr(proj, "use_vision_for_images", False)))
            else None
        )
        text, meta = execute_project_file_tool(
            user.id,
            proj,
            get_project_files(),
            name=file_name,
            offset=offset,
            limit=limit,
            vision_fn=vision_fn,
        )
        log_fn(
            f"Project file tool call: ok={meta.get('ok')} kind={meta.get('kind')} "
            f"chars={meta.get('chars')} vision={meta.get('vision_used')}"
        )
        return text
    return _exec


def inject_project_files_into_system_prompt(
    options, project_obj, user_id, model_key, batch_mode, gemini_local_python, notice_template, log_fn
):
    """Inject project shared files catalog into ``options['system_prompt']`` and return ``(files, tool_enabled)``."""
    try:
        files = get_project_files_for_prompt(project_obj, user_id)
        if not files:
            return [], False
        tool_enabled = project_file_tool_supported(
            model_key,
            batch_mode=batch_mode,
            gemini_local_python=gemini_local_python,
        )
        notice = render_project_files_notice(
            notice_template,
            project_obj,
            files,
            include_contents=not tool_enabled,
            user_id=user_id,
        )
        if notice:
            curr_p = options.get("system_prompt") or ""
            if "Project shared files:" not in str(curr_p):
                options["system_prompt"] = (
                    f"{notice}\n\n{curr_p}".strip() if str(curr_p).strip() else notice
                )
        log_fn(f"Project file catalog injected: count={len(files)} tool={tool_enabled}")
        return files, tool_enabled
    except Exception as exc:
        log_fn(f"Project file prompt inject failed: {exc}")
        return [], False
