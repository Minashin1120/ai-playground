IDEOGRAM_API_BASE = "https://api.ideogram.ai"
IDEOGRAM_TIMEOUT_SECONDS = 300.0
IDEOGRAM_MAX_IMAGE_BYTES = 25 * 1024 * 1024
IDEOGRAM_MAX_OUTPUT_BYTES = 50 * 1024 * 1024
IDEOGRAM_MAX_IMAGES = 8

# App model id -> (API path segment, request family)
#   v45: multipart, generate + precise edit, `quality` / `size`
#   v4:  JSON, `resolution` / `rendering_speed`
#   v3:  multipart, `aspect_ratio` / `rendering_speed` / style controls
#   v2:  JSON, `aspect_ratio` / `rendering_speed` / `style_type`
IDEOGRAM_MODELS = {
    "ideogram-4.5": ("ideogram-4-5", "v45"),
    "ideogram-4.0": ("ideogram-4", "v4"),
    "ideogram-3.0": ("ideogram-3", "v3"),
    "ideogram-2a": ("ideogram-2a", "v2"),
    "ideogram-2.0": ("ideogram-2", "v2"),
}
IDEOGRAM_PRECISE_EDIT_PATH = "ideogram-4-5"

IDEOGRAM_ASPECTS = ("auto", "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "10:16", "16:10", "1:2", "2:1", "1:3", "3:1")
# Aspect ratio -> (1K tier, 2K tier) exact sizes accepted by Ideogram 4.x.
IDEOGRAM_SIZE_PRESETS = {
    "1:1": ("1024x1024", "2048x2048"),
    "4:5": ("896x1120", "1792x2240"),
    "5:4": ("1120x896", "2240x1792"),
    "3:4": ("864x1152", "1728x2304"),
    "4:3": ("1152x864", "2304x1728"),
    "2:3": ("832x1248", "1664x2496"),
    "3:2": ("1248x832", "2496x1664"),
    "9:16": ("720x1280", "1440x2560"),
    "16:9": ("1280x720", "2560x1440"),
    "10:16": ("800x1280", "1600x2560"),
    "16:10": ("1280x800", "2560x1600"),
    "1:2": ("720x1440", "1440x2880"),
    "2:1": ("1440x720", "2880x1440"),
    "1:3": ("512x1536", "1024x3072"),
    "3:1": ("1536x512", "3072x1024"),
}
# Aspect ratios accepted as `aspect_ratio` by Ideogram 3.0 / 2a / 2.0.
IDEOGRAM_LEGACY_ASPECTS = {
    "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "10:16", "16:10", "1:2", "2:1", "1:3", "3:1",
}
IDEOGRAM_QUALITIES = ("very_low", "low", "medium", "high")
IDEOGRAM_SPEEDS = ("turbo", "default", "quality")
IDEOGRAM_MAGIC_PROMPTS = ("auto", "on", "off")
IDEOGRAM_STYLE_TYPES_V3 = ("auto", "general", "realistic", "design", "fiction", "stylized")
IDEOGRAM_STYLE_TYPES_V2 = ("auto", "general", "realistic", "design", "render_3d", "anime")
IDEOGRAM_SOURCE_MIMES = ("image/png", "image/jpeg", "image/webp")


def is_ideogram_model_key(model_key):
    return str(model_key or "").strip().lower() in IDEOGRAM_MODELS


def ideogram_supports_edit(model_key):
    return str(model_key or "").strip().lower() == "ideogram-4.5"


def _ideogram_pick(value, allowed):
    v = str(value or "").strip().lower()
    return v if v in allowed else None


def _ideogram_int(value, low, high):
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return max(low, min(high, n))


def _ideogram_common_fields(options, n_images):
    """Fields shared by every request family, as plain strings/ints."""
    fields = {"num_images": n_images}
    seed = _ideogram_int(options.get("ideogram_seed"), 0, 2147483647)
    if seed is not None:
        fields["seed"] = seed
    magic = _ideogram_pick(options.get("ideogram_magic_prompt"), IDEOGRAM_MAGIC_PROMPTS)
    if magic:
        fields["magic_prompt"] = magic
    return fields


def _ideogram_aspect_value(options):
    return _ideogram_pick(options.get("ideogram_aspect"), IDEOGRAM_ASPECTS) or "auto"


def _ideogram_tier_index(options):
    return 1 if str(options.get("ideogram_resolution") or "").strip().lower() == "2k" else 0


def ideogram_build_request(model_key, prompt, options, source_images=None):
    """Return (url, request kwargs for httpx.post, n_images, edit_mode).

    ``source_images`` is a list of (name, bytes, mime) already limited to PNG/JPEG/WEBP.
    Only Ideogram 4.5 accepts source images (Precise Edit).
    """
    mk = str(model_key or "").strip().lower()
    spec = IDEOGRAM_MODELS.get(mk)
    if not spec:
        raise ValueError("Unsupported Ideogram model")
    path, family = spec
    options = options or {}
    source_images = list(source_images or [])
    if source_images and family != "v45":
        raise ValueError("画像の編集に対応しているのは Ideogram 4.5 だけです。Ideogram 4.5 を選択するか、添付画像を外してください。")
    n_images = _ideogram_int(options.get("ideogram_count"), 1, IDEOGRAM_MAX_IMAGES) or 1
    fields = _ideogram_common_fields(options, n_images)
    aspect = _ideogram_aspect_value(options)
    tier = _ideogram_tier_index(options)

    if family == "v45":
        quality = _ideogram_pick(options.get("ideogram_quality"), IDEOGRAM_QUALITIES)
        data = {"prompt": prompt, "num_images": str(n_images)}
        if "seed" in fields:
            data["seed"] = str(fields["seed"])
        files = []
        if source_images:
            # Precise Edit keeps the source's own size, so no size/aspect is sent.
            url = f"{IDEOGRAM_API_BASE}/v2/image/precise-edit/{IDEOGRAM_PRECISE_EDIT_PATH}"
            if quality:
                data["quality"] = quality
            files.append(("image", (source_images[0][0], source_images[0][1], source_images[0][2])))
            for name, blob, mime in source_images[1:5]:
                files.append(("reference_images", (name, blob, mime)))
            return url, {"data": data, "files": files}, n_images, True
        url = f"{IDEOGRAM_API_BASE}/v2/image/generate/{path}"
        if "magic_prompt" in fields:
            data["magic_prompt"] = fields["magic_prompt"]
        if quality and quality != "very_low":
            data["quality"] = quality
        if aspect != "auto" and aspect in IDEOGRAM_SIZE_PRESETS:
            data["size"] = IDEOGRAM_SIZE_PRESETS[aspect][tier]
        # httpx only sends multipart when `files` is non-empty; a text-only multipart
        # body is expressed with (None, value) tuples.
        multipart = [(k, (None, str(v))) for k, v in data.items()]
        return url, {"files": multipart}, n_images, False

    url = f"{IDEOGRAM_API_BASE}/v2/image/generate/{path}"
    if family == "v4":
        body = {"prompt": prompt, **fields}
        speed = _ideogram_pick(options.get("ideogram_speed"), IDEOGRAM_SPEEDS)
        if speed:
            body["rendering_speed"] = speed
        if aspect != "auto" and aspect in IDEOGRAM_SIZE_PRESETS:
            body["resolution"] = IDEOGRAM_SIZE_PRESETS[aspect][tier]
        return url, {"json": body}, n_images, False

    legacy_aspect = aspect.replace(":", "x") if aspect in IDEOGRAM_LEGACY_ASPECTS else "auto"
    speed = _ideogram_pick(options.get("ideogram_speed"), IDEOGRAM_SPEEDS)
    negative = str(options.get("ideogram_negative_prompt") or "").strip()
    if family == "v3":
        style_type = _ideogram_pick(options.get("ideogram_style_type"), IDEOGRAM_STYLE_TYPES_V3)
        data = {"prompt": prompt, "aspect_ratio": legacy_aspect}
        for key, value in fields.items():
            data[key] = str(value)
        if speed:
            data["rendering_speed"] = speed
        if style_type:
            data["style_type"] = style_type
        if negative:
            data["negative_prompt"] = negative[:2000]
        multipart = [(k, (None, str(v))) for k, v in data.items()]
        return url, {"files": multipart}, n_images, False

    body = {"prompt": prompt, "aspect_ratio": legacy_aspect, **fields}
    if speed:
        body["rendering_speed"] = speed
    style_type = _ideogram_pick(options.get("ideogram_style_type"), IDEOGRAM_STYLE_TYPES_V2)
    if style_type:
        body["style_type"] = style_type
    if negative and mk == "ideogram-2.0":  # 2a has no negative_prompt field
        body["negative_prompt"] = negative[:2000]
    return url, {"json": body}, n_images, False


def ideogram_error_message(status_code, body_text):
    detail = ""
    try:
        parsed = json.loads(body_text or "")
        if isinstance(parsed, dict):
            detail = str(parsed.get("error") or parsed.get("detail") or parsed.get("message") or "").strip()
            reason = str(parsed.get("reject_reason") or "").strip()
            if reason and reason not in detail:
                detail = f"{detail} ({reason})".strip()
    except Exception:
        detail = str(body_text or "").strip()
    detail = detail[:400]
    if status_code == 401:
        return "Ideogram のAPIキーが無効です。設定で Ideogram API Key を確認してください。"
    if status_code == 402:
        return f"Ideogram のクレジットまたは利用枠が不足しています。{detail}".strip()
    if status_code == 422:
        return "安全性チェックにより Ideogram が生成を拒否しました。プロンプトや入力画像の内容を変更して再度お試しください。"
    if status_code == 429:
        return f"Ideogram のリクエスト制限に達しました。しばらく待ってから再試行してください。{detail}".strip()
    return f"HTTP {status_code}: {detail or 'Ideogram API request failed'}"


def ideogram_request(api_key, model_key, prompt, options, source_images=None):
    """Run one synchronous Ideogram request and return the list of image dicts."""
    url, kwargs, _n_images, edit_mode = ideogram_build_request(model_key, prompt, options, source_images)
    headers = {"Api-Key": api_key, "Accept": "application/json"}
    resp = httpx.post(url, headers=headers, timeout=IDEOGRAM_TIMEOUT_SECONDS, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(ideogram_error_message(resp.status_code, resp.text))
    payload = resp.json()
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Ideogram から画像データが返されませんでした。")
    return [item for item in items if isinstance(item, dict)], edit_mode
