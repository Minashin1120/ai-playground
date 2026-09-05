def _load_fernet_key_file(path):
    if os.path.islink(path):
        raise RuntimeError('Encryption key must not be a symbolic link')
    os.chmod(path, 0o600)
    with open(path, 'rb') as kf:
        return Fernet(kf.read().strip())


def _initialize_key_ring():
    global cipher
    ring = []
    if os.path.exists(KEY_FILE):
        ring.append(_load_fernet_key_file(KEY_FILE))
    else:
        key = Fernet.generate_key()
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as kf:
            kf.write(key)
        ring.append(Fernet(key))
    # Retained keys from earlier rotations, newest filename first, appended
    # after the active key so decryption tries the most recent key first.
    for path in sorted(glob.glob(KEY_FILE + '.rotated.*'), reverse=True):
        try:
            ring.append(_load_fernet_key_file(path))
        except Exception:
            continue
    _KEY_RING[:] = ring
    cipher = _KEY_RING[0]  # active key (backward-compatible module global)


# `cipher` stays defined on module load for backward compatibility; readers use
# decrypt_val / decrypt_bytes which fall back over the whole key ring.
try:
    _initialize_key_ring()
except Exception as e:
    raise RuntimeError(f'Encryption setup failed: {e}') from e

_DECRYPT_CACHE_LOCK = threading.Lock()
_DECRYPT_CACHE = OrderedDict()
_DECRYPT_CACHE_MISS = object()


def _ring_decrypt_bytes(data):
    """Decrypt bytes, trying the active key first then each retained key."""
    last = None
    for cipher_obj in _KEY_RING:
        try:
            return cipher_obj.decrypt(data)
        except InvalidToken:
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last = exc
            continue
    if last is not None:
        raise last
    raise InvalidToken()


def encrypt_val(val):
    if not val:
        return val
    if not cipher:
        raise RuntimeError("Encryption is unavailable")
    if not isinstance(val, str):
        val = str(val)
    return cipher.encrypt(val.encode()).decode()

def decrypt_val(val):
    if not val or not cipher: return val
    if _DECRYPT_TEXT_CACHE_MAX > 0 and isinstance(val, str):
        with _DECRYPT_CACHE_LOCK:
            hit = _DECRYPT_CACHE.get(val, _DECRYPT_CACHE_MISS)
            if hit is not _DECRYPT_CACHE_MISS:
                _DECRYPT_CACHE.move_to_end(val)
                return hit
    try:
        plain = _ring_decrypt_bytes(val.encode()).decode()
        if _DECRYPT_TEXT_CACHE_MAX > 0 and isinstance(val, str):
            with _DECRYPT_CACHE_LOCK:
                _DECRYPT_CACHE[val] = plain
                _DECRYPT_CACHE.move_to_end(val)
                while len(_DECRYPT_CACHE) > _DECRYPT_TEXT_CACHE_MAX:
                    _DECRYPT_CACHE.popitem(last=False)
        return plain
    except:
        return val

def encrypt_bytes(data):
    if not cipher:
        raise RuntimeError("Encryption is unavailable")
    return cipher.encrypt(data)

def decrypt_bytes(data):
    if not cipher or not data: return data
    return _ring_decrypt_bytes(data)

def pick_tts_voice(client, language, prefer_tier):
    try:
        voices = client.list_voices().voices
        lang_voices = [v for v in voices if language in v.language_codes]
        def find_by(substr):
            for v in lang_voices:
                if substr in v.name:
                    return v
            return None
        if prefer_tier == 'studio':
            v = find_by('Studio') or find_by('Neural2')
        elif prefer_tier == 'neural':
            v = find_by('Neural2')
        else:
            v = None
        if not v and lang_voices:
            v = lang_voices[0]
        if v:
            return texttospeech.VoiceSelectionParams(language_code=language, name=v.name)
    except Exception:
        pass
    return texttospeech.VoiceSelectionParams(
        language_code=language,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

def clamp_float(val, min_v, max_v):
    try:
        v = float(val)
    except Exception:
        return None
    if v < min_v: v = min_v
    if v > max_v: v = max_v
    return v

_AUDIO_INPUT_MAX_BYTES = 25 * 1024 * 1024

def _decode_base64_limited(value, max_bytes):
    encoded = str(value or '').strip()
    if not encoded or len(encoded) > ((max_bytes + 2) // 3) * 4 + 8:
        raise ValueError("Invalid or oversized base64 payload")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 payload") from exc
    if len(decoded) > max_bytes:
        raise ValueError("Decoded payload is too large")
    return decoded

GEMINI_TTS_VOICES = {
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe",
    "Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar",
    "Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"
}

def secure_delete(path):
    if os.path.lexists(path):
        try:
            if os.path.islink(path):
                os.unlink(path)
                return
            size = os.path.getsize(path)
            # Do not allocate a buffer as large as the file.  Export archives can
            # be hundreds of MB (or more), and cleanup must not create another
            # memory spike after the download has finished or disconnected.
            with open(path, "r+b") as f:
                remaining = size
                while remaining:
                    chunk_size = min(1024 * 1024, remaining)
                    f.write(os.urandom(chunk_size))
                    remaining -= chunk_size
                f.flush()
                os.fsync(f.fileno())
            os.remove(path)
        except: pass

# Speech-to-Speech (STS) model registry
STS_MODELS = {
    "gpt-transcribe": {"provider": "openai", "mode": "transcription", "rate_in": 24000, "rate_out": 24000},
    "gpt-live-transcribe": {"provider": "openai", "mode": "transcription", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-2": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-translate": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-whisper": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-1.5": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-mini": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gemini-2.5-flash-native-audio-preview-12-2025": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
    "gemini-3.1-flash-live-preview": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
    "gemini-3.5-live-translate-preview": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
    "gemini-3.5-transcribe-live": {"provider": "google", "mode": "transcription", "rate_in": 16000, "rate_out": 16000},
    "grok-voice-think-fast-2.0": {"provider": "xai", "rate_in": 24000, "rate_out": 24000},
    "grok-voice-think-fast-1.0": {"provider": "xai", "rate_in": 24000, "rate_out": 24000},
    "grok-voice-fast-1.0": {"provider": "xai", "rate_in": 24000, "rate_out": 24000},
    "grok-voice-agent": {"provider": "xai", "rate_in": 24000, "rate_out": 24000},
}
XAI_STS_MODEL_ALIASES = {
    "grok-voice-latest": "grok-voice-think-fast-2.0",
}
OPENAI_STS_VOICES = {
    "alloy","ash","ballad","coral","echo","sage","shimmer","verse","marin","cedar"
}
XAI_STS_VOICES = {"ara", "rex", "sal", "eve", "leo"}
GEMINI_STS_VOICES = {
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe",
    "Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar",
    "Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"
}
XAI_PCM_RATES = {8000, 16000, 22050, 24000, 32000, 44100, 48000}

# Canonical list of all valid model IDs (mirrors MODELS in chat_core.v4.*.js)
# Used to validate AI-suggested model IDs in settings updates.
# Includes deprecated models since existing threads may still reference them.
ALL_VALID_MODEL_IDS = {
    # Gemini 3.7 / 3.6 / 3.5
    "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
    # Gemini 3.1 / previous Gemini 3.x
    "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview", "gemini-3-pro-preview",
    # Gemini 2.5
    "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash",
    # Gemini Image
    "gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-image", "gemini-3-pro-image", "gemini-3-pro-image-preview",
    # Gemini Video Generation
    "gemini-omni-1.1-flash", "gemini-omni-flash", "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-preview",
    # Gemini Music Generation
    "lyria-3.5", "lyria-3-pro-preview", "lyria-3-clip-preview", "lyria-realtime-exp",
    # Gemini Agent / Specialized
    "gemini-robotics-er-2-preview", "deep-research-preview-04-2026", "deep-research-max-preview-04-2026",
    "antigravity-preview-05-2026", "gemini-2.5-computer-use-preview-10-2025", "gemini-embedding-2",
    # OpenAI Image Gen
    "gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini",
    # OpenAI GPT
    "gpt-4o", "gpt-4o-mini",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.5-mini", "gpt-5.5-nano", "gpt-5.5-pro",
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
    "gpt-5.2", "gpt-5-search-api", "gpt-5.1", "gpt-5-mini",
    # DeepSeek V4
    "deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-pro",
    # Anthropic Claude
    "claude-opus-4-6", "claude-sonnet-4-6",
    # Audio (TTS)
    "gemini-3.1-flash-tts-preview", "gpt-4o-mini-tts", "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts",
    "google-tts-studio", "google-tts-neural", "grok-tts",
    # Realtime Audio (STS)
    "gpt-transcribe", "gpt-live-transcribe",
    "gpt-realtime-2", "gpt-realtime-translate", "gpt-realtime-whisper", "gpt-realtime-1.5",
    "gpt-realtime", "gpt-realtime-mini",
    "gemini-2.5-flash-native-audio-preview-12-2025", "gemini-3.1-flash-live-preview",
    "gemini-3.5-transcribe", "gemini-3.5-transcribe-live",
    "grok-voice-latest", "grok-voice-think-fast-2.0", "grok-voice-think-fast-1.0", "grok-voice-fast-1.0", "grok-voice-agent",
    # Grok Imagine
    "grok-imagine-image-2.0", "grok-imagine-image-quality", "grok-imagine-image", "grok-imagine-image-pro", "grok-imagine-video-1.5", "grok-imagine-video",
    # xAI Grok
    "grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1",
    "grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning", "grok-4.20-multi-agent-0309",
    "grok-4.20-reasoning", "grok-4.20-non-reasoning", "grok-4.20-multi-agent",
    "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning",
    "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
    # Kimi K3
    "kimi-k3",
    # Mistral Document OCR
    "mistral-ocr-4-0",
}
