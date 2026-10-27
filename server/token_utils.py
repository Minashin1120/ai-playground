_LOCAL_RATE_LIMITS = {}

def _local_rate_limit(key, limit, window_seconds):
    now = time.monotonic()
    with _LOCAL_RATE_LIMIT_LOCK:
        count, expires_at = _LOCAL_RATE_LIMITS.get(key, (0, now + window_seconds))
        if now >= expires_at:
            count, expires_at = 0, now + window_seconds
        count += 1
        _LOCAL_RATE_LIMITS[key] = (count, expires_at)
        if len(_LOCAL_RATE_LIMITS) > 10_000:
            expired = [entry_key for entry_key, (_, expiry) in _LOCAL_RATE_LIMITS.items() if now >= expiry]
            for entry_key in expired:
                _LOCAL_RATE_LIMITS.pop(entry_key, None)
        return count <= limit

def rate_limit(key, limit, window_seconds):
    try:
        cur = redis_conn.incr(key)
        if cur == 1:
            redis_conn.expire(key, window_seconds)
        return cur <= limit
    except Exception:
        return _local_rate_limit(key, limit, window_seconds)

# Failed sign-in proofs per account, shared by Web and Android (login, 2FA and
# re-authentication). The per-IP/per-request limits above reset every few
# minutes; these stop slow, distributed guessing against one account.
AUTH_FAILURE_LIMITS = {'password': (20, 3600), 'totp': (10, 3600)}
TOTP_REPLAY_TTL = 120

def _auth_failure_key(kind, user_id):
    return f"authfail:{kind}:{user_id}"

def auth_failures_exceeded(kind, user_id):
    limit, _ = AUTH_FAILURE_LIMITS[kind]
    try:
        return int(redis_conn.get(_auth_failure_key(kind, user_id)) or 0) >= limit
    except Exception:
        return False

def record_auth_failure(kind, user_id):
    _, window = AUTH_FAILURE_LIMITS[kind]
    key = _auth_failure_key(kind, user_id)
    try:
        if redis_conn.incr(key) == 1:
            redis_conn.expire(key, window)
    except Exception:
        pass

def clear_auth_failures(kind, user_id):
    try:
        redis_conn.delete(_auth_failure_key(kind, user_id))
    except Exception:
        pass

def verify_totp_code(user_id, secret, code):
    """Check a TOTP code once: an accepted code cannot be replayed for the same account."""
    code = re.sub(r'\s+', '', str(code or ''))
    if not secret or not re.fullmatch(r'\d{6,8}', code):
        return False
    try:
        if not pyotp.TOTP(secret).verify(code):
            return False
    except Exception:
        return False
    try:
        return bool(redis_conn.set(f"totp:used:{user_id}:{code}", '1', ex=TOTP_REPLAY_TTL, nx=True))
    except Exception:
        return True

_TOKEN_ENCODER_BY_NAME = {}
_TOKEN_ENCODER_BY_MODEL = {}
_TOKEN_ENCODER_LOCK = threading.Lock()

def _select_tokenizer_name(model_key):
    mk = str(model_key or "").strip().lower()
    if not mk:
        return "o200k_base"
    if "grok" in mk or "gemini" in mk or "deepseek" in mk:
        return "o200k_base"
    if any(x in mk for x in ("gpt-4o", "gpt-4.1", "gpt-5", "gpt-6", "o1", "o3", "o4")):
        return "o200k_base"
    return "cl100k_base"

def _get_token_encoder(model_key=""):
    model_key = str(model_key or "").strip().lower()
    with _TOKEN_ENCODER_LOCK:
        enc = _TOKEN_ENCODER_BY_MODEL.get(model_key)
        if enc is not None:
            return enc
    chosen_name = None
    enc = None
    if model_key:
        try:
            enc = tiktoken.encoding_for_model(model_key)
            chosen_name = getattr(enc, "name", None)
        except Exception:
            enc = None
    if enc is None:
        chosen_name = _select_tokenizer_name(model_key)
        with _TOKEN_ENCODER_LOCK:
            enc = _TOKEN_ENCODER_BY_NAME.get(chosen_name)
        if enc is None:
            enc = tiktoken.get_encoding(chosen_name)
            with _TOKEN_ENCODER_LOCK:
                _TOKEN_ENCODER_BY_NAME[chosen_name] = enc
    with _TOKEN_ENCODER_LOCK:
        _TOKEN_ENCODER_BY_MODEL[model_key] = enc
        if chosen_name:
            _TOKEN_ENCODER_BY_NAME[chosen_name] = enc
    return enc

def count_tokens(text, model="gpt-4"):
    raw = text or ""
    if not raw:
        return 0
    for model_hint in (model, "gpt-4o", "gpt-4"):
        try:
            enc = _get_token_encoder(model_hint)
            c = len(enc.encode(raw, disallowed_special=()))
            if c == 0 and raw.strip():
                log_force("Token count 0 for non-empty text")
                continue
            return c
        except Exception:
            continue
    # Ultimate fallback to avoid 0 count for large text if all tokenizers fail
    return max(1, len(raw) // 4) if raw.strip() else 0

NON_COUNTABLE_TOKEN_MARKERS = (
    "transcribe",
    "whisper",
    "stt",
    "realtime",
    "native-audio",
    "voice-agent",
    "audio",
)

LLM_TOKEN_MARKERS = (
    "gpt",
    "gemini",
    "grok",
    "claude",
    "deepseek",
    "glm-",
    "kimi",
    "mistral",
    "o1",
    "o3",
    "o4",
)

PROMPT_TOKEN_MARKERS = (
    "gpt-image",
    "ideogram",
    "imagine",
    "image",
    "video",
    "tts",
)

def is_token_countable_model(model_key):
    if not model_key:
        return False
    if is_sts_model(model_key):
        return False
    mk = model_key.lower()
    for marker in NON_COUNTABLE_TOKEN_MARKERS:
        if marker in mk:
            return False
    # Prompt-based non-LLM models (image/video/tts) are still countable.
    for marker in PROMPT_TOKEN_MARKERS:
        if marker in mk:
            return True
    for marker in LLM_TOKEN_MARKERS:
        if marker in mk:
            return True
    return False

def should_count_tokens_for_display(model_key):
    return is_token_countable_model(model_key)

# Gemini converts attached media to tokens at fixed rates instead of tokenizing
# text. Gemini 3 and later use the media_resolution defaults (unspecified);
# earlier generations use the legacy per-image tile / per-second rates.
# https://ai.google.dev/gemini-api/docs/media-resolution
# https://ai.google.dev/gemini-api/docs/tokens
GEMINI_MEDIA_TOKEN_RATES = {
    "gemini3": {"image": 1120, "pdf_page": 560, "video_frame": 70, "audio_second": 25},
    "legacy": {"image_tile": 258, "pdf_page": 258, "video_second": 263, "audio_second": 32},
}

def gemini_media_token_profile(model_key):
    """Media token profile for a Gemini model, or None for other providers."""
    mk = str(model_key or "").strip().lower()
    if "gemini" not in mk:
        return None
    m = re.search(r"gemini-(\d+)(?:[.-]|$)", mk)
    if m and int(m.group(1)) < 3:
        return "legacy"
    return "gemini3"

def gemini_image_tokens(profile, width=None, height=None):
    if profile == "gemini3":
        return GEMINI_MEDIA_TOKEN_RATES["gemini3"]["image"]
    per_tile = GEMINI_MEDIA_TOKEN_RATES["legacy"]["image_tile"]
    try:
        width, height = int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width <= 384 and height <= 384:
        return per_tile
    return per_tile * math.ceil(width / 768) * math.ceil(height / 768)

def gemini_pdf_tokens(profile, pages):
    # Gemini 3 does not charge the native text extracted from PDFs.
    return GEMINI_MEDIA_TOKEN_RATES[profile]["pdf_page"] * max(0, int(pages or 0))

def gemini_audio_tokens(profile, seconds):
    return math.ceil(max(0.0, float(seconds or 0)) * GEMINI_MEDIA_TOKEN_RATES[profile]["audio_second"])

def gemini_video_tokens(profile, seconds, has_audio=True):
    seconds = max(0.0, float(seconds or 0))
    if profile == "legacy":
        return math.ceil(seconds * GEMINI_MEDIA_TOKEN_RATES["legacy"]["video_second"])
    # Frames are sampled at 1 fps; the audio track is counted separately.
    frames = max(1, math.ceil(seconds))
    tokens = frames * GEMINI_MEDIA_TOKEN_RATES["gemini3"]["video_frame"]
    if has_audio:
        tokens += gemini_audio_tokens(profile, seconds)
    return tokens

def extract_reasoning_text(thought_data):
    if not thought_data:
        return ""
    if isinstance(thought_data, dict):
        return (thought_data.get("text") or "").strip()
    if isinstance(thought_data, str):
        try:
            parsed = json.loads(thought_data)
        except Exception:
            return thought_data.strip()
        if isinstance(parsed, dict):
            return (parsed.get("text") or "").strip()
        if isinstance(parsed, list):
            parts = []
            for item in parsed:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
    return ""

def count_tokens_for_display(text, model_key, thought_text=None):
    if not should_count_tokens_for_display(model_key):
        return None
    total = 0
    if text:
        total += count_tokens(text, model_key)
    if thought_text:
        total += count_tokens(thought_text, model_key)
    return total

def sum_token_counts(tokens_in, tokens_out):
    if tokens_in is None and tokens_out is None:
        return None
    total = 0
    if tokens_in is not None:
        total += tokens_in
    if tokens_out is not None:
        total += tokens_out
    return total

def build_message_token_details(role, content, thought_text, model_key, tokens_in=None, tokens_out=None):
    if not should_count_tokens_for_display(model_key):
        return {
            "tokens_in": None,
            "tokens_out": None,
            "tokens_total": None,
            "tokens_content": None,
            "tokens_thought": None,
        }
    if role == "user":
        if tokens_in is None:
            tokens_in = count_tokens_for_display(content, model_key)
        tokens_content = count_tokens(content or "", model_key)
        return {
            "tokens_in": tokens_in,
            "tokens_out": None,
            "tokens_total": sum_token_counts(tokens_in, None),
            "tokens_content": tokens_content,
            "tokens_thought": None,
        }
    tokens_content = count_tokens(content or "", model_key)
    tokens_thought = count_tokens(thought_text or "", model_key) if thought_text else 0
    if tokens_out is None:
        tokens_out = sum_token_counts(tokens_content, tokens_thought)
    return {
        "tokens_in": None,
        "tokens_out": tokens_out,
        "tokens_total": sum_token_counts(None, tokens_out),
        "tokens_content": tokens_content,
        "tokens_thought": tokens_thought,
    }

def readable_provider_error(value):
    """Do not expose proxy HTML as a chat error; preserve the actionable HTTP status."""
    text = str(value or "Unknown error")
    if re.search(r"<\s*(?:html|head|body|!doctype)\b", text, re.I):
        if re.search(r"\b413\b", text):
            return "API Error (413): 画像や会話の送信サイズが上限を超えました。画像を圧縮するか、新しいチャットで再試行してください。"
        status = re.search(r"\b([45]\d{2})\b", text)
        return f"API Error ({status.group(1)}): 接続先からエラーが返されました。時間をおいて再試行してください。" if status else "API Error: 接続先から読み取れない応答が返されました。"
    return text


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(exc.SQLAlchemyError))
def format_chat_error_content(error_text, partial_content=""):
    """Build assistant content that renders as a persistent chat error bubble.

    Uses a fenced ```chat_error block so the client can restyle it on history
    reload the same way live stream errors are shown.
    """
    err_body = str(error_text or "Unknown error").strip() or "Unknown error"
    if len(err_body) > 50_000:
        err_body = err_body[:50_000] + "…"
    # Keep the fence well-formed even if the error text contains backticks.
    err_body = err_body.replace("```", "'''")
    fence = f"```chat_error\n{err_body}\n```"
    partial = str(partial_content or "").rstrip()
    if partial:
        return partial + "\n\n" + fence
    return fence


def safe_db_commit():
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

def evaluate_bot_score(payload):
    try:
        window_ms = int(payload.get('window_ms') or 0)
        clicks = int(payload.get('clicks') or 0)
        keys = int(payload.get('keys') or 0)
        fast_clicks = int(payload.get('fast_clicks') or 0)
        fast_keys = int(payload.get('fast_keys') or 0)
        click_burst = int(payload.get('click_burst') or 0)
        key_burst = int(payload.get('key_burst') or 0)
        event_rate = float(payload.get('event_rate') or 0)
        avg_click_ms = float(payload.get('avg_click_ms') or 9999)
        click_cv = float(payload.get('click_cv') or 1.0)
        pointer_speed_max = float(payload.get('pointer_speed_max') or 0)
    except Exception:
        return 0, []
    score = 0
    reasons = []
    if click_burst >= 12:
        score += 3
        reasons.append('click_burst')
    elif clicks >= 18 and window_ms <= 3000:
        score += 2
        reasons.append('click_rate')
    if fast_clicks >= 6:
        score += 2
        reasons.append('fast_clicks')
    if fast_keys >= 14:
        score += 2
        reasons.append('fast_keys')
    if key_burst >= 18:
        score += 2
        reasons.append('key_burst')
    if event_rate >= 25:
        score += 1
        reasons.append('high_event_rate')
    if avg_click_ms < 140 and click_cv < 0.08:
        score += 2
        reasons.append('robotic_clicks')
    if pointer_speed_max >= 6000:
        score += 1
        reasons.append('pointer_speed')
    return score, reasons

