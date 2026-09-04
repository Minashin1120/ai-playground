class StorageLimitError(Exception):
    def __init__(self, message, used=None, limit=None):
        super().__init__(message)
        self.used = used
        self.limit = limit

def _bytes_to_mb_str(val):
    try:
        return f"{float(val) / (1024 * 1024):.1f}MB"
    except Exception:
        return "0MB"

def _get_primary_admin_username():
    name = app.config.get('PRIMARY_ADMIN_USERNAME')
    if not name:
        return None
    return str(name).strip()

def _is_primary_admin_username(username):
    name = _get_primary_admin_username()
    if not name or not username:
        return False
    return str(username) == name

def _is_primary_admin_user(user):
    if not user:
        return False
    return _is_primary_admin_username(getattr(user, "username", None))

def _get_user_storage_limit_bytes(user):
    try:
        if not user:
            return None
        if _is_primary_admin_user(user):
            return None
        limit_mb = int(app.config.get('USER_STORAGE_LIMIT_MB') or 0)
        if limit_mb <= 0:
            return None
        return limit_mb * 1024 * 1024
    except Exception:
        return None

def _get_user_storage_usage_bytes(user_id):
    # Upload validation can ask for the same value in both before_request and the
    # route handler. Keep one authoritative filesystem scan per request; the
    # cache is request-local, so it cannot leak between users or become stale
    # across separate upload requests.
    cache_key = str(user_id)
    try:
        request_cache = getattr(g, '_storage_usage_bytes', None)
        if isinstance(request_cache, dict) and cache_key in request_cache:
            return request_cache[cache_key]
    except RuntimeError:
        request_cache = None
    total = 0
    try:
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        if os.path.isdir(user_dir):
            for root, _, files in os.walk(user_dir):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except Exception:
                        pass
    except Exception:
        pass
    # Incomplete chunk uploads count against the same quota as completed files.
    try:
        chunk_dir = os.path.join(app.config['UPLOAD_FOLDER'], '.chunks', str(user_id))
        if os.path.isdir(chunk_dir):
            for root, _, files in os.walk(chunk_dir):
                for name in files:
                    if name != 'data.part':
                        continue
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        if not isinstance(request_cache, dict):
            request_cache = {}
            g._storage_usage_bytes = request_cache
        request_cache[cache_key] = total
    except RuntimeError:
        pass
    return total

def _get_filestorage_size(fs):
    if not fs:
        return None
    try:
        stream = fs.stream
        pos = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(pos)
        return size
    except Exception:
        try:
            data = fs.read()
            fs.stream.seek(0)
            return len(data)
        except Exception:
            return None

def _sanitize_upload_filename(raw_name):
    """Return (safe_base, ext) for a user-supplied upload filename.

    ``secure_filename()`` strips non-ASCII characters, so Japanese names such
    as ``売上データ.xlsx`` become ``xlsx`` (the leading dot is also removed),
    which makes every extension check fail with an empty extension.  The
    extension must therefore be read from the *original* filename (after
    taking only its basename), while ``safe_base`` stays the sanitized name
    used for display / storage.
    """
    raw = str(raw_name or "").strip().replace("\\", "/")
    base_raw = raw.rsplit("/", 1)[-1].strip()
    ext = os.path.splitext(base_raw)[1].lower()
    safe_base = secure_filename(base_raw) or ""
    return safe_base, ext

def _normalize_upload_ref(ref):
    if not ref:
        return None
    val = None
    if isinstance(ref, dict):
        for k in ("path", "filepath", "file", "url", "name"):
            if ref.get(k):
                val = ref.get(k)
                break
        if val is None:
            return None
    else:
        val = ref
    try:
        val = str(val).strip()
    except Exception:
        return None
    if not val:
        return None
    if "://" in val:
        try:
            val = urlparse(val).path or ""
        except Exception:
            pass
    if "?" in val:
        val = val.split("?", 1)[0]
    if "#" in val:
        val = val.split("#", 1)[0]
    val = val.lstrip("/")
    if val.startswith("files/"):
        val = val[len("files/"):]
    try:
        val = unquote(val)
    except Exception:
        pass
    norm = os.path.normpath(val)
    if norm.startswith("..") or os.path.isabs(norm):
        return None
    return norm

def _path_is_within(base_dir, candidate):
    try:
        base_real = os.path.realpath(base_dir)
        candidate_real = os.path.realpath(candidate)
        return os.path.commonpath((base_real, candidate_real)) == base_real
    except Exception:
        return False

def _resolve_user_upload_rel_path(filename, user_id):
    norm = _normalize_upload_ref(filename)
    if not norm:
        return None
    parts = norm.split(os.sep)
    if len(parts) > 1 and parts[0].isdigit():
        if int(parts[0]) != int(user_id):
            return None
        return norm
    return os.path.join(str(user_id), norm)

def _normalize_attachment_list(raw_list, user_id=None):
    if not raw_list:
        return []
    normalized = []
    seen = set()
    for item in raw_list:
        ref = _normalize_upload_ref(item)
        if not ref:
            continue
        
        ref_str = str(ref)
        # Security: If user_id is provided, ensure the ref belongs to them.
        if user_id is not None:
            prefix = f"{user_id}/"
            if not ref_str.startswith(prefix):
                # If it doesn't start with any user ID prefix, prepend current user's.
                parts = ref_str.split('/')
                if not (len(parts) > 1 and parts[0].isdigit()):
                    ref_str = prefix + ref_str
                else:
                    # Belongs to another user or invalid format.
                    continue
        
        if ref_str in seen:
            continue
        seen.add(ref_str)
        normalized.append(ref_str)
    return normalized

_AUDIO_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm"
}
_VIDEO_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm"
}
_IMAGE_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif", ".heic", ".heif"}

def _normalize_media_mime(filename, mime_guess):
    ext = os.path.splitext(filename or "")[1].lower()
    mg = (mime_guess or "").lower()
    if ext in _VIDEO_MIME_BY_EXT:
        if (not mg) or (not mg.startswith("video/")) or ("text" in mg):
            return _VIDEO_MIME_BY_EXT[ext]
    if ext in _AUDIO_MIME_BY_EXT:
        if (not mg) or (not mg.startswith("audio/")) or ("text" in mg):
            return _AUDIO_MIME_BY_EXT[ext]
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if ext == ".txt":
        return "text/plain"
    return mime_guess or "application/octet-stream"

def _extract_docx_paragraphs(data):
    """Return every paragraph's text (document order) from a docx file.

    Unlike _extract_text_from_docx, empty paragraphs are included so their
    positions (used by edit_file's paragraph_edits) stay stable.  Returns
    ``None`` when the bytes cannot be parsed safely.
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            info = zf.getinfo('word/document.xml')
            if info.file_size > 8 * 1024 * 1024:
                return None
            if info.compress_size > 0 and info.file_size / info.compress_size > 100:
                return None
            xml_content = zf.read(info)
        tree = ET.fromstring(xml_content)
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = []
        for paragraph in tree.findall('.//w:p', namespace):
            texts = [node.text for node in paragraph.findall('.//w:t', namespace) if node.text]
            paragraphs.append("".join(texts))
        return paragraphs
    except Exception:
        return None


def _extract_text_from_docx(data):
    paragraphs = _extract_docx_paragraphs(data)
    if paragraphs is None:
        return None
    return "\n".join(p for p in paragraphs if p)


def _extract_docx_as_numbered(data, max_paragraphs=3000):
    """Render a docx as numbered paragraphs so the model can reference them.

    Emits ``[1] text`` lines where the numbers are the same 1-based positions
    used by edit_file's paragraph_edits.  Returns ``None`` on parse failure.
    """
    paragraphs = _extract_docx_paragraphs(data)
    if paragraphs is None:
        return None
    lines = []
    for i, text in enumerate(paragraphs[:max_paragraphs], start=1):
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)

def _extract_text_from_pdf(data, max_pages=300, max_chars=2_000_000):
    if not data or len(data) > 100 * 1024 * 1024:
        return None
    try:
        reader = pypdf.PdfReader(BytesIO(data), strict=False)
        if len(reader.pages) > max_pages:
            return None
        parts = []
        total_chars = 0
        for page in reader.pages:
            text_value = page.extract_text() or ""
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            parts.append(text_value[:remaining])
            total_chars += min(len(text_value), remaining)
        return "\n".join(parts)
    except Exception:
        return None

def _get_file_disk_info(rel_path):
    if not rel_path:
        return {"exists": False}
    base = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
    try:
        if not _path_is_within(app.config['UPLOAD_FOLDER'], base):
            return {"exists": False}
    except Exception:
        return {"exists": False}
    enc = base + '.enc'
    path = None
    is_encrypted = False
    if os.path.exists(base):
        path = base
    elif os.path.exists(enc):
        path = enc
        is_encrypted = True
    if not path:
        return {"exists": False}
    size = None
    mtime = None
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    try:
        mtime = int(os.path.getmtime(path))
    except Exception:
        mtime = None
    return {
        "exists": True,
        "path": base,
        "enc_path": enc,
        "disk_path": path,
        "is_encrypted": is_encrypted,
        "size": size,
        "mtime": mtime
    }

def _is_low_latency_image_attachment_set(rel_paths):
    """Identify small local image sets that do not need the heavy chat queue."""
    refs = list(rel_paths or [])
    if not refs or len(refs) > _LOW_LATENCY_IMAGE_MAX_ITEMS:
        return False
    total_bytes = 0
    for rel_path in refs:
        if os.path.splitext(str(rel_path or ""))[1].lower() not in _IMAGE_THUMB_EXTS:
            return False
        info = _get_file_disk_info(rel_path)
        size = info.get("size") if info.get("exists") else None
        if size is None or size <= 0:
            return False
        total_bytes += int(size)
        if total_bytes > _LOW_LATENCY_IMAGE_MAX_BYTES:
            return False
    return True

_MEDIA_BYTES_CACHE_LOCK = threading.Lock()
_MEDIA_BYTES_CACHE = OrderedDict()
_MEDIA_BYTES_CACHE_SIZE = 0
_THUMBNAIL_BYTES_CACHE_LOCK = threading.Lock()
_THUMBNAIL_BYTES_CACHE = OrderedDict()
_THUMBNAIL_BYTES_CACHE_SIZE = 0
_TOKEN_FILE_TOKENS_CACHE_MAX = max(0, _env_int("TOKEN_FILE_TOKENS_CACHE_MAX", 512))
_TOKEN_FILE_TOKENS_CACHE_LOCK = threading.Lock()
_TOKEN_FILE_TOKENS_CACHE = OrderedDict()

def _ordered_lru_cache_get(cache, lock, key, enabled=True):
    if not enabled:
        return None
    with lock:
        hit = cache.get(key)
        if hit is None:
            return None
        cache.move_to_end(key)
        return hit

def _ordered_lru_bytes_cache_put(cache, lock, key, data, cache_max, item_max, current_size):
    if cache_max <= 0 or data is None:
        return current_size
    size = len(data)
    if size <= 0:
        return current_size
    if item_max and size > item_max:
        return current_size
    if size > cache_max:
        return current_size
    with lock:
        prev = cache.pop(key, None)
        if prev is not None:
            current_size -= len(prev)
        cache[key] = data
        current_size += size
        while current_size > cache_max and cache:
            _, ev = cache.popitem(last=False)
            current_size -= len(ev)
    return current_size

def _ordered_lru_bytes_cache_evict_path(cache, lock, rel_path, current_size):
    if not rel_path:
        return current_size
    with lock:
        to_del = [k for k in cache.keys() if isinstance(k, tuple) and len(k) > 0 and k[0] == rel_path]
        for k in to_del:
            prev = cache.pop(k, None)
            if prev is not None:
                current_size -= len(prev)
    return current_size

def _ordered_lru_cache_put_count_limited(cache, lock, key, value, max_items):
    if max_items <= 0:
        return
    if key is None or value is None:
        return
    with lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

def _media_bytes_cache_get(key):
    return _ordered_lru_cache_get(
        _MEDIA_BYTES_CACHE,
        _MEDIA_BYTES_CACHE_LOCK,
        key,
        enabled=_MEDIA_BYTES_CACHE_MAX > 0,
    )

def _media_bytes_cache_put(key, data):
    global _MEDIA_BYTES_CACHE_SIZE
    _MEDIA_BYTES_CACHE_SIZE = _ordered_lru_bytes_cache_put(
        _MEDIA_BYTES_CACHE,
        _MEDIA_BYTES_CACHE_LOCK,
        key,
        data,
        _MEDIA_BYTES_CACHE_MAX,
        _MEDIA_BYTES_CACHE_ITEM_MAX,
        _MEDIA_BYTES_CACHE_SIZE,
    )

def _media_bytes_cache_evict_path(rel_path):
    global _MEDIA_BYTES_CACHE_SIZE
    _MEDIA_BYTES_CACHE_SIZE = _ordered_lru_bytes_cache_evict_path(
        _MEDIA_BYTES_CACHE,
        _MEDIA_BYTES_CACHE_LOCK,
        rel_path,
        _MEDIA_BYTES_CACHE_SIZE,
    )

def _thumbnail_bytes_cache_get(key):
    return _ordered_lru_cache_get(
        _THUMBNAIL_BYTES_CACHE,
        _THUMBNAIL_BYTES_CACHE_LOCK,
        key,
        enabled=_THUMBNAIL_CACHE_MAX > 0,
    )

def _thumbnail_bytes_cache_put(key, data):
    global _THUMBNAIL_BYTES_CACHE_SIZE
    _THUMBNAIL_BYTES_CACHE_SIZE = _ordered_lru_bytes_cache_put(
        _THUMBNAIL_BYTES_CACHE,
        _THUMBNAIL_BYTES_CACHE_LOCK,
        key,
        data,
        _THUMBNAIL_CACHE_MAX,
        _THUMBNAIL_CACHE_ITEM_MAX,
        _THUMBNAIL_BYTES_CACHE_SIZE,
    )

def _thumbnail_bytes_cache_evict_path(rel_path):
    global _THUMBNAIL_BYTES_CACHE_SIZE
    _THUMBNAIL_BYTES_CACHE_SIZE = _ordered_lru_bytes_cache_evict_path(
        _THUMBNAIL_BYTES_CACHE,
        _THUMBNAIL_BYTES_CACHE_LOCK,
        rel_path,
        _THUMBNAIL_BYTES_CACHE_SIZE,
    )


def _token_file_tokens_cache_get(key):
    return _ordered_lru_cache_get(
        _TOKEN_FILE_TOKENS_CACHE,
        _TOKEN_FILE_TOKENS_CACHE_LOCK,
        key,
        enabled=_TOKEN_FILE_TOKENS_CACHE_MAX > 0,
    )


def _token_file_tokens_cache_put(key, value):
    _ordered_lru_cache_put_count_limited(
        _TOKEN_FILE_TOKENS_CACHE,
        _TOKEN_FILE_TOKENS_CACHE_LOCK,
        key,
        value,
        _TOKEN_FILE_TOKENS_CACHE_MAX,
    )

def _load_user_file_bytes(rel_path, info=None):
    if not rel_path:
        return None
    if info is None:
        info = _get_file_disk_info(rel_path)
    if not info or not info.get("exists"):
        return None
    key = (
        rel_path,
        info.get("mtime"),
        info.get("size"),
        1 if info.get("is_encrypted") else 0
    )
    cached = _media_bytes_cache_get(key)
    if cached is not None:
        return cached
    data = None
    try:
        with open(info["disk_path"], 'rb') as f:
            raw = f.read()
        if info.get("is_encrypted"):
            data = decrypt_bytes(raw)
        else:
            data = raw
    except Exception:
        data = None
    if data is None:
        return None
    _media_bytes_cache_put(key, data)
    return data


def _decode_text_bytes_for_prompt(raw):
    if not raw:
        return None
    sample = raw[:2 * 1024 * 1024]  # Avoid large decode for huge text files
    try:
        from charset_normalizer import from_bytes
        match = from_bytes(sample).best()
        if match and match.output():
            try:
                return match.output().decode('utf-8', errors='replace')
            except Exception:
                pass
    except Exception:
        pass
    try:
        return sample.decode('utf-8')
    except Exception:
        return sample.decode('utf-8', errors='replace')


def _estimate_attachment_prompt_tokens(rel_path, model_key=None):
    info = _get_file_disk_info(rel_path)
    if not info.get("exists"):
        return {"tokens": 0, "countable": False, "reason": "missing"}
    tokenizer_name = _select_tokenizer_name(model_key)
    cache_key = (
        rel_path,
        info.get("mtime"),
        info.get("size"),
        1 if info.get("is_encrypted") else 0,
        tokenizer_name
    )
    cached = _token_file_tokens_cache_get(cache_key)
    if cached is not None:
        return cached

    data = _load_user_file_bytes(rel_path, info)
    if data is None:
        result = {"tokens": 0, "countable": False, "reason": "read_error"}
        _token_file_tokens_cache_put(cache_key, result)
        return result

    clean_fn = os.path.basename(rel_path or "")
    mime_guess = mimetypes.guess_type(clean_fn)[0]
    mime = _normalize_media_mime(clean_fn, mime_guess)
    is_pdf = clean_fn.lower().endswith('.pdf')
    is_docx = clean_fn.lower().endswith('.docx')
    clean_ext = os.path.splitext(clean_fn)[1].lower()
    is_text = (mime or '').startswith('text/') or clean_ext in _TEXT_LIKE_UPLOAD_EXTS

    extracted = None
    if is_pdf:
        extracted = _extract_text_from_pdf(data)
    elif is_docx:
        extracted = _extract_text_from_docx(data)
    elif is_text:
        extracted = _decode_text_bytes_for_prompt(data)

    if extracted:
        result = {"tokens": count_tokens(extracted, model_key), "countable": True, "reason": "ok"}
    elif is_pdf or is_docx or is_text:
        result = {"tokens": 0, "countable": False, "reason": "no_text"}
    else:
        result = {"tokens": 0, "countable": False, "reason": "non_text"}

    _token_file_tokens_cache_put(cache_key, result)
    return result

def _get_file_cache(user_id, rel_path, provider):
    if not user_id or not rel_path or not provider:
        return None
    try:
        return FileCache.query.filter_by(user_id=user_id, rel_path=rel_path, provider=provider).order_by(FileCache.id.desc()).first()
    except Exception:
        return None

def _upsert_file_cache(user_id, rel_path, provider, **fields):
    if not user_id or not rel_path or not provider:
        return None
    cache = _get_file_cache(user_id, rel_path, provider)
    if not cache:
        cache = FileCache(user_id=user_id, rel_path=rel_path, provider=provider)
        db.session.add(cache)
    for k, v in fields.items():
        try:
            setattr(cache, k, v)
        except Exception:
            pass
    cache.updated_at = datetime.utcnow()
    return cache

def _delete_file_cache_for_path(user_id, rel_path):
    if not user_id or not rel_path:
        return
    try:
        FileCache.query.filter_by(user_id=user_id, rel_path=rel_path).delete()
    except Exception:
        pass
    try:
        _media_bytes_cache_evict_path(rel_path)
    except Exception:
        pass
    try:
        _thumbnail_bytes_cache_evict_path(rel_path)
    except Exception:
        pass

def _sanitize_file_display_name(raw_name):
    if raw_name is None:
        return None
    try:
        name = str(raw_name).strip()
    except Exception:
        return None
    if not name:
        return None
    name = name.replace("\x00", "")
    name = name.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    name = name.split("/")[-1].split("\\")[-1].strip()
    name = re.sub(r"\s{2,}", " ", name)
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    if not name or name in {".", ".."}:
        return None
    if len(name) > 180:
        name = name[:180].rstrip()
    return name or None

def _normalize_thread_title(value):
    title = str(value or "").replace("\x00", "").strip()
    title = title.replace("<", "").replace(">", "")
    title = re.sub(r"[\r\n\t]+", " ", title)
    title = re.sub(r"\s{2,}", " ", title)
    return (title or "Untitled")[:200]

def _normalize_gem_payload(data, existing=None):
    if not isinstance(data, dict):
        raise ValueError("invalid_payload")

    def value_for(key, default):
        if key in data:
            return data.get(key)
        return getattr(existing, key, default) if existing is not None else default

    name = str(value_for('name', 'My Gem') or '').replace('\x00', '').replace('<', '').replace('>', '').strip()
    description = str(value_for('description', '') or '').replace('\x00', '').strip()
    instruction = str(value_for('instruction', '') or '').replace('\x00', '').strip()
    if not name or len(name) > 100:
        raise ValueError("invalid_name")
    if len(description) > 4000 or len(instruction) > 100_000:
        raise ValueError("payload_too_large")

    fixed_raw = value_for('fixed_prompts_json', None)
    if 'fixed_prompts' in data:
        fixed_raw = data.get('fixed_prompts')
    fixed_prompts_json = None
    if fixed_raw:
        if isinstance(fixed_raw, str) and len(fixed_raw) > 200_000:
            raise ValueError("payload_too_large")
        try:
            prompts = json.loads(fixed_raw) if isinstance(fixed_raw, str) else fixed_raw
        except Exception as exc:
            raise ValueError("invalid_fixed_prompts") from exc
        if not isinstance(prompts, list) or len(prompts) > 50:
            raise ValueError("invalid_fixed_prompts")
        normalized_prompts = []
        for item in prompts:
            if not isinstance(item, dict):
                raise ValueError("invalid_fixed_prompts")
            prompt_name = str(item.get('name') or '').replace('\x00', '').replace('<', '').replace('>', '').strip()
            prompt_content = str(item.get('content') or '').replace('\x00', '').strip()
            if not prompt_name or not prompt_content or len(prompt_name) > 100 or len(prompt_content) > 20_000:
                raise ValueError("invalid_fixed_prompts")
            normalized_prompts.append({'name': prompt_name, 'content': prompt_content})
        fixed_prompts_json = json.dumps(normalized_prompts, ensure_ascii=False, separators=(',', ':'))

    default_model = value_for('default_model', None)
    default_model = str(default_model or '').strip() or None
    if default_model and default_model not in ALL_VALID_MODEL_IDS:
        raise ValueError("invalid_default_model")
    return {
        'name': name,
        'description': description,
        'instruction': instruction,
        'fixed_prompts_json': fixed_prompts_json,
        'default_model': default_model,
    }

def _normalize_display_name_for_path(rel_path, raw_name):
    base = os.path.basename(rel_path or "")
    if not base:
        return None
    safe = _sanitize_file_display_name(raw_name)
    if not safe:
        return None
    stem, ext = os.path.splitext(base)
    cand_stem, cand_ext = os.path.splitext(safe)
    if ext:
        if not cand_ext:
            safe = safe + ext
        elif cand_ext.lower() != ext.lower():
            safe = (cand_stem or safe) + ext
    if len(safe) > 180:
        base_stem, base_ext = os.path.splitext(safe)
        keep = max(1, 180 - len(base_ext))
        safe = base_stem[:keep].rstrip() + base_ext
    return safe or None

def _get_user_file_label_map(user_id):
    labels = {}
    if not user_id:
        return labels
    try:
        rows = FileCache.query.filter_by(user_id=user_id, provider="label").all()
    except Exception:
        rows = []
    for row in rows:
        try:
            rel = (row.rel_path or "").strip()
            name = _sanitize_file_display_name(row.file_uri or "")
            if rel and name:
                labels[rel] = name
        except Exception:
            continue
    return labels

def _check_storage_capacity(user, additional_bytes):
    limit = _get_user_storage_limit_bytes(user)
    if not limit:
        return True, None, None
    used = _get_user_storage_usage_bytes(user.id)
    if used + additional_bytes > limit:
        return False, used, limit
    return True, used + additional_bytes, limit

def _chunk_root_dir():
    return os.path.join(app.config['UPLOAD_FOLDER'], '.chunks')

def _chunk_user_dir(user_id):
    return os.path.join(_chunk_root_dir(), str(user_id))

_CHUNK_UPLOAD_ID_RE = re.compile(r"^up_[0-9]{10}_[0-9a-f]{8}$")
_CHUNK_SIZE_BYTES = 10 * 1024 * 1024
_CHUNK_UPLOAD_MAX_ACTIVE = 10
_CHUNK_UPLOAD_MAX_AGE_SECONDS = max(60 * 60, _env_int("CHUNK_UPLOAD_MAX_AGE_SECONDS", 6 * 60 * 60))
_CHUNK_SWEEP_INTERVAL_SECONDS = max(5 * 60, _env_int("CHUNK_SWEEP_INTERVAL_SECONDS", 15 * 60))
_CHUNK_SWEEP_REDIS_KEY = "chunk_upload:stale_sweep"

def _is_valid_chunk_upload_id(upload_id):
    return bool(_CHUNK_UPLOAD_ID_RE.fullmatch(str(upload_id or "")))

def _chunk_session_dir(user_id, upload_id):
    if not _is_valid_chunk_upload_id(upload_id):
        return None
    user_dir = os.path.realpath(_chunk_user_dir(user_id))
    candidate = os.path.realpath(os.path.join(user_dir, str(upload_id)))
    try:
        if os.path.commonpath((user_dir, candidate)) != user_dir:
            return None
    except Exception:
        return None
    return candidate

@contextmanager
def _chunk_upload_lock(session_dir):
    lock_file = None
    try:
        lock_path = os.path.join(session_dir, '.lock')
        lock_file = open(lock_path, 'a+b')
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        if lock_file is not None:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            lock_file.close()

def _cleanup_stale_chunk_uploads(user_id, now=None):
    now = int(now or time.time())
    user_dir = _chunk_user_dir(user_id)
    active = 0
    if not os.path.isdir(user_dir):
        return active
    for entry in os.scandir(user_dir):
        if not entry.is_dir(follow_symlinks=False) or not _is_valid_chunk_upload_id(entry.name):
            continue
        meta_path = os.path.join(entry.path, 'meta.json')
        meta = _load_chunk_meta(meta_path) or {}
        created = int(meta.get('created') or 0)
        updated = int(meta.get('updated') or 0)
        try:
            meta_mtime = int(os.path.getmtime(meta_path))
        except Exception:
            meta_mtime = 0
        last_activity = max(created, updated, meta_mtime)
        if last_activity <= 0 or now - last_activity > _CHUNK_UPLOAD_MAX_AGE_SECONDS:
            # Chunk completion already removes data.part with unlink. Use the
            # same lightweight deletion for abandoned transient data instead of
            # overwriting potentially multi-GB files with random bytes on the
            # request thread.
            try:
                shutil.rmtree(entry.path)
            except Exception:
                pass
            continue
        active += 1
    return active

def _cleanup_all_stale_chunk_uploads():
    root = _chunk_root_dir()
    if not os.path.isdir(root):
        return
    try:
        entries = list(os.scandir(root))
    except Exception:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            int(entry.name)
        except (TypeError, ValueError):
            continue
        _cleanup_stale_chunk_uploads(entry.name)

def _maybe_sweep_stale_chunk_uploads():
    try:
        acquired = redis_conn.set(
            _CHUNK_SWEEP_REDIS_KEY,
            str(int(time.time())),
            nx=True,
            ex=_CHUNK_SWEEP_INTERVAL_SECONDS,
        )
    except Exception:
        return
    if acquired:
        _cleanup_all_stale_chunk_uploads()

def _load_chunk_meta(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _save_chunk_meta(path, meta):
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix='.meta-', suffix='.json', dir=os.path.dirname(path))
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return True
    except Exception:
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

KEY_FILE = os.path.join(os.path.dirname(__file__), 'secret.key')
# Key ring (newest first).  index 0 is the ACTIVE key used for encryption; the
# rest are historical keys (secret.key.rotated.*) retained so that data still
# encrypted with an older key remains readable.  This is what makes a live key
# rotation safe: during a transition both the old and the new key decrypt.
_KEY_RING = []


