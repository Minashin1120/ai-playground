
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
    png_data = cairosvg.svg2png(
        bytestring=sanitized_svg,
        output_width=output_width,
        output_height=output_height,
        unsafe=False,
    )
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

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
            image_format = str(image.format or "").upper()
    except Exception as exc:
        raise ValueError("Gemini inline data is not a supported image") from exc

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
}

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

