def is_sts_model(model_key):
    return model_key in STS_MODELS

def get_sts_provider(model_key):
    meta = STS_MODELS.get(model_key)
    return meta.get("provider") if meta else None

def is_gemini_model_key(model_key):
    mk = str(model_key or "").lower()
    return "gemini" in mk or is_gemini_video_model_key(mk) or is_gemini_music_model_key(mk) or is_gemini_agent_model_key(mk)

def is_anthropic_model_key(model_key):
    mk = str(model_key or "").lower()
    return "claude" in mk

def is_mistral_ocr_model_key(model_key):
    mk = str(model_key or "").lower().strip()
    if not mk:
        return False
    if mk in ("mistral-ocr-4-0", "mistral-ocr-latest"):
        return True
    return mk.startswith("mistral-ocr")

def get_model_api_provider(model_key):
    """Return the API provider id for a model (used by prompt-cache lock)."""
    mk = str(model_key or "").lower().strip()
    if not mk:
        return None
    if "claude" in mk:
        return "anthropic"
    if "deepseek" in mk:
        return "deepseek"
    if "grok" in mk and "gpt" not in mk:
        return "xai"
    if "google-tts" in mk:
        return "google"
    if is_gemini_model_key(mk):
        return "gemini"
    if "kimi" in mk:
        return "kimi"
    if is_mistral_ocr_model_key(mk) or mk.startswith("mistral"):
        return "mistral"
    return "openai"

_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "anthropic": "Anthropic (Claude)",
    "xai": "xAI (Grok)",
    "deepseek": "DeepSeek",
    "kimi": "Kimi (Moonshot)",
    "google": "Google Cloud",
    "mistral": "Mistral",
}

MISTRAL_API_BASE = "https://api.mistral.ai/v1"
MISTRAL_OCR_MODEL_ID = "mistral-ocr-4-0"
MISTRAL_OCR_MAX_FILE_BYTES = 512 * 1024 * 1024
MISTRAL_OCR_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".avif"}
MISTRAL_OCR_DOC_EXTS = {".pdf", ".docx", ".pptx"}
MISTRAL_OCR_PUBLIC_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

def _mistral_auth_headers(api_key, json_body=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers

def _mistral_error_message(resp):
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("message") or payload.get("detail") or payload.get("error")
        if isinstance(err, dict):
            err = err.get("message") or err.get("detail") or err.get("type")
        if err:
            return str(err)
    text = (resp.text or "").strip()
    if text:
        return text[:400]
    return f"HTTP {resp.status_code}"

def _mistral_raise_for_status(resp, action="Mistral API"):
    if resp.status_code == 401:
        raise RuntimeError("Mistral APIキーが無効です。")
    if resp.status_code == 402:
        raise RuntimeError("Mistral の利用上限または支払いに問題があります。")
    if resp.status_code == 429:
        raise RuntimeError("Mistral OCR のレート制限に達しました。しばらく待って再試行してください。")
    if resp.status_code >= 400:
        raise RuntimeError(f"{action}に失敗しました: {_mistral_error_message(resp)}")

def _mistral_upload_ocr_file(api_key, filename, data, mime):
    if not data or len(data) > MISTRAL_OCR_MAX_FILE_BYTES:
        raise RuntimeError("Mistral OCR のファイルサイズ上限（512MB）を超えています。")
    safe_name = os.path.basename(str(filename or "document")) or "document"
    files = {
        "file": (safe_name, data, mime or "application/octet-stream"),
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{MISTRAL_API_BASE}/files",
            headers=_mistral_auth_headers(api_key),
            files=files,
            data={"purpose": "ocr"},
        )
    _mistral_raise_for_status(resp, "Mistral Files API へのアップロード")
    payload = resp.json() if resp.content else {}
    file_id = payload.get("id") if isinstance(payload, dict) else None
    if not file_id:
        raise RuntimeError("Mistral Files API が file_id を返しませんでした。")
    return str(file_id)

def _mistral_delete_file(api_key, file_id):
    if not file_id:
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            client.delete(
                f"{MISTRAL_API_BASE}/files/{file_id}",
                headers=_mistral_auth_headers(api_key),
            )
    except Exception:
        logger.warning("Failed to delete Mistral OCR upload %s", file_id)

def _mistral_signed_file_url(api_key, file_id):
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{MISTRAL_API_BASE}/files/{file_id}/url",
            headers=_mistral_auth_headers(api_key),
            params={"expiry": 60},
        )
    _mistral_raise_for_status(resp, "Mistral 署名付きURLの取得")
    payload = resp.json() if resp.content else {}
    url = None
    if isinstance(payload, dict):
        url = payload.get("url") or payload.get("signed_url")
    if not url:
        raise RuntimeError("Mistral 署名付きURLを取得できませんでした。")
    return str(url)

def _mistral_ocr_request(api_key, document, extra):
    payload = {"model": MISTRAL_OCR_MODEL_ID, "document": document}
    if extra:
        payload.update(extra)
    with httpx.Client(timeout=httpx.Timeout(480.0, connect=30.0)) as client:
        resp = client.post(
            f"{MISTRAL_API_BASE}/ocr",
            headers=_mistral_auth_headers(api_key, json_body=True),
            json=payload,
        )
    _mistral_raise_for_status(resp, "Mistral OCR")
    return resp.json() if resp.content else {}

def _mistral_ocr_process_document(api_key, document, extra):
    try:
        return _mistral_ocr_request(api_key, document, extra)
    except RuntimeError as exc:
        # Some accounts accept file_id, others need a signed document URL.
        if (
            isinstance(document, dict)
            and document.get("type") == "file"
            and document.get("file_id")
            and "file" in str(exc).lower()
        ):
            signed = _mistral_signed_file_url(api_key, document.get("file_id"))
            return _mistral_ocr_request(
                api_key,
                {"type": "document_url", "document_url": signed},
                extra,
            )
        raise

def _mistral_data_uri(mime, data):
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"

def _mistral_guess_url_kind(url):
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in MISTRAL_OCR_IMAGE_EXTS:
        return "image"
    return "document"

def _extract_mistral_ocr_urls(text):
    urls = []
    seen = set()
    for match in MISTRAL_OCR_PUBLIC_URL_RE.findall(str(text or "")):
        cleaned = match.rstrip(").,];'\"")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        urls.append(cleaned)
    return urls

def _decode_mistral_image_base64(raw):
    if not raw:
        return None, None
    value = str(raw).strip()
    mime = "image/jpeg"
    if value.startswith("data:"):
        header, _, encoded = value.partition(",")
        if ";base64" not in header:
            return None, None
        mime_part = header[5:].split(";", 1)[0].strip()
        if mime_part:
            mime = mime_part
        value = encoded
    try:
        data = _decode_base64_limited(value, 20 * 1024 * 1024)
    except Exception:
        return None, None
    return data, mime

def _mistral_ext_from_mime(mime, fallback="jpg"):
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/avif": "avif",
    }
    return mapping.get((mime or "").lower(), fallback)

def _build_mistral_ocr_markdown(ocr_json, image_url_by_id, include_blocks=False):
    pages = (ocr_json or {}).get("pages") or []
    sections = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_no = int(page.get("index", 0)) + 1
        except Exception:
            page_no = 1
        markdown = str(page.get("markdown") or "")
        for img in page.get("images") or []:
            if not isinstance(img, dict):
                continue
            img_id = str(img.get("id") or img.get("image_id") or "").strip()
            local_url = image_url_by_id.get(img_id) if img_id else None
            if img_id and local_url:
                markdown = markdown.replace(f"]({img_id})", f"]({local_url})")
        for table in page.get("tables") or []:
            if not isinstance(table, dict):
                continue
            table_id = str(table.get("id") or table.get("table_id") or "").strip()
            table_body = table.get("html") or table.get("markdown") or table.get("content")
            if table_id and table_body:
                placeholder = f"[tbl-{table_id}]({table_id})"
                if placeholder in markdown:
                    markdown = markdown.replace(placeholder, str(table_body))
                else:
                    markdown += f"\n\n{table_body}"
        header = page.get("header")
        footer = page.get("footer")
        block_lines = [f"### ページ {page_no}"]
        if header:
            block_lines.append(f"*ヘッダー:* {header}")
        if markdown.strip():
            block_lines.append(markdown.strip())
        if footer:
            block_lines.append(f"*フッター:* {footer}")
        scores = page.get("confidence_scores")
        if isinstance(scores, dict):
            avg = scores.get("average_content_confidence_score")
            if avg is not None:
                block_lines.append(f"*ページ信頼度:* {avg}")
        if include_blocks and page.get("blocks"):
            try:
                blocks_json = json.dumps(page.get("blocks"), ensure_ascii=False, indent=2)
            except Exception:
                blocks_json = str(page.get("blocks"))
            if len(blocks_json) > 60000:
                blocks_json = blocks_json[:60000] + "\n..."
            block_lines.append("```json\n" + blocks_json + "\n```")
        sections.append("\n\n".join(block_lines))
    usage = (ocr_json or {}).get("usage_info") or {}
    pages_processed = usage.get("pages_processed")
    footer_bits = []
    if pages_processed is not None:
        footer_bits.append(f"処理ページ数: {pages_processed}")
    if footer_bits:
        sections.append("—" + " / ".join(footer_bits))
    return "\n\n".join(sections).strip() or "（抽出テキストはありません）"

def is_deepseek_model_key(model_key):
    return _is_deepseek_model_key(model_key)

def is_gemini_image_model_key(model_key):
    mk = str(model_key or "").lower()
    return "gemini" in mk and any(x in mk for x in ("image", "nano"))

def is_gemini_video_model_key(model_key):
    mk = str(model_key or "").lower().strip()
    return mk.startswith("veo-") or "omni-flash" in mk or "omni-1.1-flash" in mk

def is_gemini_music_model_key(model_key):
    mk = str(model_key or "").lower().strip()
    return mk.startswith("lyria-")

def is_gemini_embedding_model_key(model_key):
    mk = str(model_key or "").lower().strip()
    return mk.startswith("gemini-embedding")

def is_gemini_agent_model_key(model_key):
    mk = str(model_key or "").lower().strip()
    return mk.startswith("deep-research-") or mk.startswith("antigravity-")

def is_gemini_transcribe_model_key(model_key):
    """True for the unary (audio-file) Gemini Transcribe model. The Live variant
    (gemini-3.5-transcribe-live) is routed through STS_MODELS / Live API instead."""
    mk = str(model_key or "").lower().strip()
    return mk == "gemini-3.5-transcribe"

def _extract_interaction_text(interaction):
    """Extract the full text from a Gemini Interactions API response.

    The Interactions API returns content in ``outputs`` (older v1beta shape) or
    ``steps[].content[]`` (newer shape); both are handled defensively because the
    installed google-genai SDK is older than the Transcribe model documentation.
    """
    if interaction is None:
        return ""
    outputs = getattr(interaction, "outputs", None) or []
    texts = []
    for out in outputs or []:
        if out is None:
            continue
        if isinstance(out, dict):
            if out.get("type") == "text":
                t = out.get("text") or ""
                if t:
                    texts.append(str(t))
            continue
        t = getattr(out, "text", None)
        if t:
            texts.append(str(t))
    if texts:
        return "".join(texts)
    # Newer shape: steps[].content[].text
    steps = getattr(interaction, "steps", None) or []
    for step in steps or []:
        contents = step.get("content") if isinstance(step, dict) else getattr(step, "content", None)
        for c in contents or []:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    t = c.get("text") or ""
                    if t:
                        texts.append(str(t))
            else:
                t = getattr(c, "text", None)
                if t:
                    texts.append(str(t))
    return "".join(texts)


def _gemini_transcribe_rest(api_key, file_uri, mime_type, transcription_config, timeout=600):
    """Call the Gemini Interactions API directly (REST) for gemini-3.5-transcribe.

    The installed google-genai SDK (1.x) serializes the legacy Interactions schema,
    which the API removed on 2026-06-08. Calling /v1beta/interactions directly with
    the documented body works with the current steps schema. ``store=False`` keeps
    the request stateless (no project storage setting required) and still returns
    the transcript in ``steps``.
    """
    if not api_key:
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    payload = {
        "model": "gemini-3.5-transcribe",
        "store": False,
        "input": [
            {"type": "audio", "uri": file_uri, "mime_type": mime_type}
        ],
        "generation_config": {
            "transcription_config": transcription_config or {"language_codes": []}
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"Gemini Transcribe request failed: {exc}") from exc
    if resp.status_code >= 400:
        try:
            err = resp.json()
            msg = (err.get("error") or {}).get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Gemini Transcribe API error ({resp.status_code}): {str(msg)[:500]}")
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Gemini Transcribe returned non-JSON response")
    if data.get("status") not in (None, "completed", "done"):
        raise RuntimeError(f"Gemini Transcribe status: {data.get('status')}")
    texts = []
    for step in data.get("steps") or []:
        contents = step.get("content") if isinstance(step, dict) else getattr(step, "content", None)
        for c in contents or []:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    t = c.get("text") or ""
                    if t:
                        texts.append(str(t))
            else:
                t = getattr(c, "text", None)
                if t:
                    texts.append(str(t))
    if not texts:
        for out in data.get("outputs") or []:
            if isinstance(out, dict):
                if out.get("type") == "text":
                    t = out.get("text") or ""
                    if t:
                        texts.append(str(t))
            else:
                t = getattr(out, "text", None)
                if t:
                    texts.append(str(t))
    return "".join(texts)

def _chunk_bytes(data, chunk_size=32000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def _convert_audio_to_pcm(audio_bytes, src_suffix=".webm", rate=24000):
    cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-t", "300",
        "-threads", "1",
        "-ac", "1",
        "-ar", str(rate),
        "-f", "s16le",
        "pipe:1"
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=audio_bytes, timeout=10)
        if proc.returncode != 0:
            logger.error(f"FFmpeg failed (code {proc.returncode}): {stderr.decode()}")
            raise Exception("Audio conversion failed")
        return stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        raise Exception("Audio conversion timed out")
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        raise e

def _pcm_to_wav_bytes(pcm_bytes, rate=24000):
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def _pcm_audio_metrics_mono_s16le(pcm_bytes, rate=24000):
    try:
        if not pcm_bytes:
            return {"duration_sec": 0.0, "rms": 0, "peak": 0}
        width = 2
        frame_count = len(pcm_bytes) // width
        duration_sec = (frame_count / float(rate)) if rate else 0.0
        rms = int(audioop.rms(pcm_bytes, width)) if frame_count > 0 else 0
        peak = int(audioop.max(pcm_bytes, width)) if frame_count > 0 else 0
        return {"duration_sec": duration_sec, "rms": rms, "peak": peak}
    except Exception:
        return {"duration_sec": 0.0, "rms": 0, "peak": 0}

def _llm_transcript_is_no_speech(text, token="[[NO_SPEECH]]"):
    t = (text or "").strip()
    if not t:
        return False
    if t == token:
        return True
    if token in t and len(t) <= (len(token) + 12):
        return True
    return False

def _save_user_generated_bytes(user_id, data, filename, encrypt):
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("Generated file is empty")
    filename = secure_filename(str(filename or ''))
    if not filename:
        raise ValueError("Invalid generated filename")
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, mode=0o700, exist_ok=True)
    try:
        user = db.session.get(User, user_id)
    except Exception as exc:
        raise StorageLimitError("Unable to validate storage capacity") from exc
    if user is None:
        raise StorageLimitError("Unable to validate storage capacity")
    ok, used, limit = _check_storage_capacity(user, len(data))
    if not ok:
        used_mb = _bytes_to_mb_str(used)
        limit_mb = _bytes_to_mb_str(limit)
        raise StorageLimitError(f"Storage limit exceeded ({used_mb} / {limit_mb})", used=used, limit=limit)
    os.chmod(user_dir, 0o700)
    fpath = os.path.join(user_dir, filename)
    if encrypt:
        with open(fpath + '.enc', 'xb') as f:
            f.write(encrypt_bytes(bytes(data)))
    else:
        with open(fpath, 'xb') as f:
            f.write(data)
    return fpath


def _save_user_generated_bytes_verified(user_id, data, make_filename, encrypt, attempts=2):
    """Save generated bytes and confirm the file actually landed on disk.

    A silent save loss (the write reports success but no file exists afterwards,
    or an external race removes it moments later) must never leak a dead
    /files/ URL into chat content.  Every attempt uses a fresh filename and is
    followed by an existence check covering both the plain and .enc variants.
    Returns (filename, "/files/<user_id>/<filename>") or raises when every
    attempt fails, letting callers fall back to a user-visible note instead of
    a broken image reference.
    """
    last_error = None
    for _ in range(max(1, attempts)):
        fname = make_filename()
        try:
            _save_user_generated_bytes(user_id, data, fname, encrypt)
        except Exception as exc:
            last_error = exc
        else:
            fpath = os.path.join(
                app.config['UPLOAD_FOLDER'], str(user_id), secure_filename(fname)
            )
            if os.path.exists(fpath) or os.path.exists(fpath + '.enc'):
                return fname, f"/files/{user_id}/{fname}"
            last_error = RuntimeError("saved file failed on-disk verification")
        time.sleep(0.05)
    raise last_error if last_error else RuntimeError("generated file could not be saved")

