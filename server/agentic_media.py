
_AGENTIC_IMAGE_MAX_BYTES = 50 * 1024 * 1024
_AGENTIC_SVG_MAX_BYTES = 10 * 1024 * 1024
_AGENTIC_SVG_MAX_DIMENSION = 4096
_AGENTIC_SVG_MAX_PIXELS = 16_000_000
_SVG_FORBIDDEN_ELEMENTS = {"script", "foreignobject", "iframe", "object", "embed"}


def _xml_local_name(value):
    return str(value or "").rsplit("}", 1)[-1].lower()


def _svg_dimension(value):
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*$", str(value or ""), re.I)
    if not match:
        return None
    dimension = float(match.group(1))
    return dimension if dimension > 0 else None


def _sanitize_and_rasterize_agentic_svg(data):
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("Generated SVG is empty")
    if len(data) > _AGENTIC_SVG_MAX_BYTES:
        raise ValueError("Generated SVG is too large")

    root = ET.fromstring(bytes(data))
    if _xml_local_name(root.tag) != "svg":
        raise ValueError("Generated SVG has an invalid root element")

    for element in root.iter():
        if _xml_local_name(element.tag) in _SVG_FORBIDDEN_ELEMENTS:
            raise ValueError("Generated SVG contains an unsafe element")
        if element.text and (
            re.search(r"url\s*\(\s*(?![\"']?#)", element.text, re.I)
            or re.search(r"@import\b", element.text, re.I)
        ):
            raise ValueError("Generated SVG contains an external resource")
        for raw_name, raw_value in element.attrib.items():
            name = _xml_local_name(raw_name)
            value = str(raw_value or "").strip()
            if name.startswith("on"):
                raise ValueError("Generated SVG contains an event handler")
            if name == "base":
                raise ValueError("Generated SVG contains an external base URL")
            if name in {"href", "src"} and value and not value.startswith("#"):
                raise ValueError("Generated SVG contains an external resource")
            if (
                re.search(r"url\s*\(\s*(?![\"']?#)", value, re.I)
                or re.search(r"@import\b", value, re.I)
            ):
                raise ValueError("Generated SVG contains an external resource")

    width = _svg_dimension(root.attrib.get("width"))
    height = _svg_dimension(root.attrib.get("height"))
    view_box = str(root.attrib.get("viewBox") or root.attrib.get("viewbox") or "").split()
    if len(view_box) == 4:
        try:
            view_width = float(view_box[2])
            view_height = float(view_box[3])
            width = width or (view_width if view_width > 0 else None)
            height = height or (view_height if view_height > 0 else None)
        except (TypeError, ValueError):
            pass
    width = width or 300.0
    height = height or 150.0
    scale = min(
        1.0,
        _AGENTIC_SVG_MAX_DIMENSION / max(width, height),
        math.sqrt(_AGENTIC_SVG_MAX_PIXELS / (width * height)),
    )
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    sanitized_svg = ET.tostring(root, encoding="utf-8")
    png_data = _rasterize_svg_png(sanitized_svg, output_width, output_height)
    if not png_data or len(png_data) > _AGENTIC_IMAGE_MAX_BYTES:
        raise ValueError("Rasterized generated image is invalid")
    return png_data


def _prepare_agentic_image_bytes(data, declared_mime=None):
    if isinstance(data, str):
        data = _decode_base64_limited(data, _AGENTIC_IMAGE_MAX_BYTES)
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("Generated image is empty")
    data = bytes(data)
    if len(data) > _AGENTIC_IMAGE_MAX_BYTES:
        raise ValueError("Generated image is too large")

    mime = str(declared_mime or "").split(";", 1)[0].strip().lower()
    is_svg = mime == "image/svg+xml" or bool(
        re.match(br"\s*(?:<\?xml[^>]*>\s*)?<svg(?:\s|>)", data, re.I)
    )
    if is_svg:
        return _sanitize_and_rasterize_agentic_svg(data), "png"

    info = _probe_image(data)
    if not info or not _validate_image_structure(data, info):
        raise ValueError("Gemini inline data is not a supported image")
    image_format = info["format"]

    extension_by_format = {
        "PNG": "png",
        "JPEG": "jpg",
        "WEBP": "webp",
        "GIF": "gif",
    }
    extension = extension_by_format.get(image_format)
    if not extension:
        raise ValueError(f"Unsupported generated image format: {image_format or 'unknown'}")
    return data, extension


_SANDBOX_IMG_REF_RE = re.compile(
    r"!\[([^\]]*)\]\(\s*((?:sandbox:/mnt/data/|/mnt/data/)?[\w.\-]+\.(?:png|jpe?g|webp|gif|bmp))[^)\s]*\)",
    re.I,
)

_SANDBOX_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "gif", "bmp"})

_PY_SANDBOX_OUTPUT_IMAGE_RE = re.compile(
    r"""\.(?:save|savefig|imsave|imwrite|tofile|export)\s*\(\s*["']([^"']+\.(?:png|jpe?g|webp|gif|bmp))["']""",
    re.I,
)


def _extract_sandbox_image_filenames(code):
    """Return output image filenames written by Python sandbox code.

    Gemini code-execution names the files it produces inside the code it runs
    (e.g. ``img.save("result.png")``) while the produced bytes are streamed
    back separately as inline_data.  Matching the save-name to the streamed
    bytes lets a bare ``![alt](result.png)`` reference in the model's final
    answer resolve to the locally saved /files/... URL.
    """
    if not code:
        return []
    found = []
    for match in _PY_SANDBOX_OUTPUT_IMAGE_RE.finditer(str(code)):
        fname = os.path.basename(match.group(1).strip())
        if fname and fname not in found:
            found.append(fname)
    return found


def _sandbox_ref_basename(url):
    """Return the image basename of a sandbox-style reference URL, else None."""
    base = os.path.basename(str(url or "").strip())
    if "." in base and base.rsplit(".", 1)[-1].lower() in _SANDBOX_IMAGE_EXTENSIONS:
        return base
    return None


def _rewrite_sandbox_image_refs(text, saved_urls, consumed_urls, filename_url_map=None):
    """
    Rewrite Gemini code-execution sandbox image references (e.g.
    ![alt](sandbox:/mnt/data/name.png), ![alt](/mnt/data/name.png) or the bare
    ![alt](name.png) the model writes for its final result) to locally saved
    /files/... URLs.

    saved_urls is the live queue of agentic image URLs captured from
    inline_data during this request; references consume them in order.
    filename_url_map maps sandbox basenames (e.g. "result.png") to saved
    /files/ URLs so a bare reference to the model's output can be matched
    without relying on order.  Unresolved sandbox:/mnt/data references are
    replaced with a short note so the browser never renders an unloadable URL;
    unresolved bare filenames are left as-is because they may be a legitimate
    relative link.
    """
    if not text or "![" not in text:
        return text

    def _repl(match):
        alt = match.group(1) or ""
        url = match.group(2)
        basename = _sandbox_ref_basename(url)
        if filename_url_map and basename:
            mapped = filename_url_map.get(basename.lower())
            if mapped:
                consumed_urls.append(mapped)
                return f"![{alt}]({mapped})"
        if saved_urls:
            resolved = saved_urls.pop(0)
            consumed_urls.append(resolved)
            return f"![{alt}]({resolved})"
        if url.startswith("sandbox:") or url.startswith("/mnt/data/"):
            return f"（※画像データを取得できませんでした: {alt}）"
        return match.group(0)

    return _SANDBOX_IMG_REF_RE.sub(_repl, text)


def _rewrite_streamed_sandbox_refs(delta, buffer_state, saved_urls, consumed_urls, filename_url_map=None):
    """
    Process a streamed text delta, rewriting Gemini code-execution sandbox image
    references (e.g. ![alt](sandbox:/mnt/data/name.png)) to saved /files/... URLs.

    buffer_state is a single-element list holding a pending tail so a reference
    split across multiple streamed parts is still rewritten once completed.
    When no saved URL is available yet (the image bytes may still arrive later
    in the stream), the reference is left untouched so the final full_res pass
    can resolve it after the whole response has been received.
    Returns the text to publish (already appended to the caller's full_res).
    """
    pending = buffer_state[0] + delta
    out = []
    while True:
        match = _SANDBOX_IMG_REF_RE.search(pending)
        if not match:
            break
        out.append(pending[:match.start()])
        alt = match.group(1) or ""
        url = match.group(2)
        resolved = None
        basename = _sandbox_ref_basename(url)
        if filename_url_map and basename:
            resolved = filename_url_map.get(basename.lower())
        if resolved is None and saved_urls:
            resolved = saved_urls.pop(0)
        if resolved is not None:
            consumed_urls.append(resolved)
            out.append(f"![{alt}]({resolved})")
        else:
            out.append(match.group(0))
        pending = pending[match.end():]
    idx = pending.rfind("![")
    if idx >= 0 and ")" not in pending[idx:]:
        out.append(pending[:idx])
        buffer_state[0] = pending[idx:]
        return "".join(out)
    buffer_state[0] = ""
    out.append(pending)
    return "".join(out)


_PYEXEC_BLOCK_RE = re.compile(r"```pyexec\n(.*?)\n```", re.S)
_STORED_AGENTIC_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\((/files/\d+/agentic_(\d+)_[0-9a-f]+\.[A-Za-z0-9]+)\)"
)
_PRIOR_SANDBOX_SCAN_LIMIT = 200


def _collect_sandbox_image_names(text, names):
    """
    Record which saved /files/ image a stored Gemini code-execution answer
    produced for each sandbox file name (e.g. "result.png").

    The stored answer keeps every executed block as ```pyexec {"code": ...}```
    followed by the ![Agentic View](/files/...) placeholder of each image that
    run returned, so the names a block saves pair with the placeholders after
    it, mirroring the streaming pass.  A placeholder consumed by a reference in
    the same answer was moved to that reference; leftover names then pair with
    those images in the order they were saved.  names maps the lower-cased
    name to its URL, or to None when the code wrote the file but no image came
    back.  Later answers override earlier ones.
    """
    if not text or "```pyexec" not in text:
        return
    events = [(m.start(), "code", m) for m in _PYEXEC_BLOCK_RE.finditer(text)]
    events += [(m.start(), "image", m) for m in _STORED_AGENTIC_IMAGE_RE.finditer(text)]
    events.sort(key=lambda item: item[0])
    pending = []
    unpaired_names = []
    moved_images = []
    found = {}
    for _, kind, match in events:
        if kind == "code":
            unpaired_names.extend(pending)
            try:
                code = json.loads(match.group(1)).get("code")
            except Exception:
                code = None
            pending = _extract_sandbox_image_filenames(code)
            continue
        url = match.group(2)
        if match.group(1) != "Agentic View":
            moved_images.append((int(match.group(3)), url))
        elif pending:
            found[pending.pop(0).lower()] = url
    unpaired_names.extend(pending)
    moved_urls = []
    earlier_urls = set(url for url in names.values() if url)
    for _, url in sorted(moved_images):
        # An image an earlier answer produced may be cited again here.
        if url not in moved_urls and url not in found.values() and url not in earlier_urls:
            moved_urls.append(url)
    for name in unpaired_names:
        key = name.lower()
        if key in found:
            continue
        found[key] = moved_urls.pop(0) if moved_urls else None
    for key, url in found.items():
        if url or not names.get(key):
            names[key] = url


def _resolve_prior_sandbox_image_refs(text, names):
    """
    Rewrite bare ![alt](name.png) references to an image an earlier answer
    produced in the code-execution sandbox (names from
    _collect_sandbox_image_names).  A name the earlier code wrote without
    returning an image becomes a short note instead of an unloadable image;
    names no earlier answer produced are left unchanged.
    """
    if not text or not names or "![" not in text:
        return text

    def _repl(match):
        basename = _sandbox_ref_basename(match.group(2))
        key = basename.lower() if basename else None
        if not key or key not in names:
            return match.group(0)
        alt = match.group(1) or ""
        url = names[key]
        if url:
            return f"![{alt}]({url})"
        return f"（※画像データを取得できませんでした: {alt}）"

    return _SANDBOX_IMG_REF_RE.sub(_repl, text)


def _has_bare_sandbox_image_ref(text):
    return bool(text) and "![" in text and _SANDBOX_IMG_REF_RE.search(text) is not None


def _sandbox_history_rows(thread_id, before_id=None):
    """Earlier answers of the thread that may hold code-execution output, oldest first."""
    try:
        query = Message.query.filter(
            Message.thread_id == thread_id,
            Message.role == "assistant",
            db.or_(Message.is_encrypted.is_(True), Message.content.like("%```pyexec%")),
        )
        if before_id is not None:
            query = query.filter(Message.id < before_id)
        rows = query.order_by(Message.id.desc()).limit(_PRIOR_SANDBOX_SCAN_LIMIT).all()
    except Exception as exc:
        log_force(f"Sandbox image history lookup failed for thread {thread_id}: {exc}")
        return []
    rows.sort(key=lambda row: row.id)
    return rows


def _collect_sandbox_row_names(row, names):
    try:
        content = decrypt_val(row.content) if row.is_encrypted else row.content
    except Exception:
        return
    if isinstance(content, str):
        _collect_sandbox_image_names(content, names)


def _prior_sandbox_image_names(thread_id):
    """Sandbox file names -> saved /files/ URLs from every stored answer of the thread."""
    names = {}
    for row in _sandbox_history_rows(thread_id):
        _collect_sandbox_row_names(row, names)
    return names


def _resolve_thread_sandbox_image_refs(thread_id, texts):
    """
    Resolve bare sandbox image references in stored answers of one thread
    against the code-execution output of the answers before each of them.

    texts maps a message id to its decrypted content.  Returns
    {message_id: rewritten text} only for the texts that changed.  The caller
    must already have checked that the thread belongs to the user.
    """
    targets = {mid: text for mid, text in (texts or {}).items() if _has_bare_sandbox_image_ref(text)}
    if not targets:
        return {}
    rows = _sandbox_history_rows(thread_id, before_id=max(targets))
    names = {}
    changed = {}
    index = 0
    for mid in sorted(targets):
        while index < len(rows) and rows[index].id < mid:
            _collect_sandbox_row_names(rows[index], names)
            index += 1
        rewritten = _resolve_prior_sandbox_image_refs(targets[mid], names)
        if rewritten != targets[mid]:
            changed[mid] = rewritten
    return changed


def _apply_thread_sandbox_image_refs(thread_id, items):
    """Rewrite, in place, the content of serialized assistant messages ({"id", "role", "content"})."""
    try:
        changed = _resolve_thread_sandbox_image_refs(thread_id, {
            item["id"]: item["content"]
            for item in items
            if item.get("role") == "assistant" and isinstance(item.get("content"), str)
        })
    except Exception as exc:
        log_force(f"Sandbox image reference resolution failed for thread {thread_id}: {exc}")
        return
    for item in items:
        if item.get("id") in changed:
            item["content"] = changed[item["id"]]


def _save_user_audio(user_id, data, suffix, encrypt):
    fname = f"audio_{int(time.time())}_{os.urandom(4).hex()}{suffix}"
    fpath = _save_user_generated_bytes(user_id, data, fname, encrypt)
    return fname, fpath

MIC_TRANSCRIBE_MODES = {"stt_api", "llm"}

# Valid values for enum-constrained AI settings fields
VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
VALID_SAFETY_SETTINGS = {"default", "none"}
VALID_STT_MODELS = {
    "gpt-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize",
    "whisper-1",
    "grok-voice-transcribe-2.0",
    "grok-voice-transcribe-1.0",
}
XAI_STT_MODELS = {"grok-voice-transcribe-2.0", "grok-voice-transcribe-1.0"}
# Model-list entry for the non-live Grok Voice Transcribe 2.0 (recorded clip sent to /sts,
# transcribed through the batch /v1/stt endpoint) -> the model name xAI expects.
XAI_STT_FILE_STS_MODELS = {"grok-voice-transcribe-2.0-file": "grok-voice-transcribe-2.0"}

DEFAULT_LLM_TRANSCRIBE_PROMPT = (
    "この音声を正確に文字起こししてください。"
    "出力は文字起こし本文のみ。説明・要約・補足は不要です。"
)
LLM_TRANSCRIBE_PROMPT_MAX_CHARS = 4000

DEFAULT_IMAGE_ANALYSIS_PROMPT = (
    "Describe this image in extreme detail, covering every single element from corner to corner. "
    "Include: all visible text (transcribed verbatim), objects, people (count, appearance, expressions, clothing), "
    "colors, lighting, spatial layout, background/foreground relationships, any actions or interactions, "
    "signs, symbols, logos, diagrams, charts (with exact values if readable), "
    "and any subtle details that might be important. "
    "Do not summarize or omit anything. Be exhaustive and precise."
)

