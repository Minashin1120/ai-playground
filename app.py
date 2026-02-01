import os
import sys
import json
import time
import logging
import base64
import mimetypes
import secrets
import re
import redis
import shutil
import glob
import requests
import tiktoken
import subprocess
import random
import pyotp
import qrcode
import wave
import asyncio
import tempfile
import threading
import hashlib
import httpx
from webauthn import (
    generate_registration_options, verify_registration_response,
    generate_authentication_options, verify_authentication_response
)
from webauthn.helpers import generate_challenge, base64url_to_bytes, options_to_json
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria, UserVerificationRequirement,
    PublicKeyCredentialCreationOptions, PublicKeyCredentialRequestOptions,
    PublicKeyCredentialDescriptor, AuthenticatorTransport
)
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from rq import Queue
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash, send_file, abort, session, g
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, exc, text
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError, RateLimitError
from google import genai
from google.genai import types
from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions
import websockets
import pypdf
from cryptography.fernet import Fernet
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as x_user, assistant as x_assistant, system as x_system, image as x_image
    from xai_sdk.search import SearchParameters, web_source, x_source
    from xai_sdk.tools import code_execution as x_code_execution, web_search as x_web_search, x_search as x_x_search
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAIClient = None
    XAI_SDK_AVAILABLE = False

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_force(msg):
    """Force log to stdout/journalctl"""
    try:
        print(f"[AI-CHAT-DEBUG] {msg}", file=sys.stdout, flush=True)
        logger.info(msg)
    except:
        pass

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
if not os.getenv('FLASK_SECRET_KEY'):
    raise RuntimeError("FLASK_SECRET_KEY is required")

def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _env_bool(name, default=False):
    val = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return val in ("1", "true", "yes", "on")

def _env_choice(name, default, allowed):
    val = (os.getenv(name) or "").strip()
    return val if val in allowed else default

def _key_sig(key, extra=""):
    if not key:
        return None
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h}:{extra}" if extra else h

def _closest_aspect_ratio(width, height, allowed):
    try:
        if not width or not height:
            return None
        ratio = float(width) / float(height)
    except Exception:
        return None
    best = None
    best_diff = None
    for a in allowed:
        try:
            parts = a.split(":")
            if len(parts) != 2:
                continue
            ar = float(parts[0]) / float(parts[1])
            diff = abs(ratio - ar)
            if best is None or diff < best_diff:
                best = a
                best_diff = diff
        except Exception:
            continue
    return best

_HTTP_MAX_CONNECTIONS = _env_int("AI_CHAT_HTTP_MAX_CONNECTIONS", 100)
_HTTP_MAX_KEEPALIVE = _env_int("AI_CHAT_HTTP_MAX_KEEPALIVE", 20)
_HTTP_KEEPALIVE_EXPIRY = _env_float("AI_CHAT_HTTP_KEEPALIVE_EXPIRY", 30.0)
_HTTP2_ENABLED = _env_bool("AI_CHAT_HTTP2", True)
if _HTTP2_ENABLED:
    try:
        import h2  # noqa: F401
    except Exception:
        _HTTP2_ENABLED = False

_OPENAI_CONNECT_TIMEOUT = _env_float("OPENAI_CONNECT_TIMEOUT_SECONDS", 5.0)
_OPENAI_READ_TIMEOUT = _env_float("OPENAI_READ_TIMEOUT_SECONDS", 120.0)
_OPENAI_WRITE_TIMEOUT = _env_float("OPENAI_WRITE_TIMEOUT_SECONDS", 30.0)
_OPENAI_POOL_TIMEOUT = _env_float("OPENAI_POOL_TIMEOUT_SECONDS", 5.0)
_OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 1)
_OPENAI_IMAGE_TIMEOUT_SECONDS = _env_float("OPENAI_IMAGE_TIMEOUT_SECONDS", 120.0)
_OPENAI_IMAGE_MAX_RETRIES = _env_int("OPENAI_IMAGE_MAX_RETRIES", 1)
_OPENAI_IMAGE_DEFAULT_SIZE = _env_choice(
    "OPENAI_IMAGE_DEFAULT_SIZE",
    "1024x1024",
    {"auto", "1024x1024", "1536x1024", "1024x1536"}
)
_OPENAI_IMAGE_DEFAULT_QUALITY = _env_choice(
    "OPENAI_IMAGE_DEFAULT_QUALITY",
    "medium",
    {"auto", "low", "medium", "high"}
)
_OPENAI_IMAGE_OUTPUT_FORMAT = _env_choice(
    "OPENAI_IMAGE_OUTPUT_FORMAT",
    "jpeg",
    {"png", "jpeg", "webp"}
)
_OPENAI_IMAGE_OUTPUT_COMPRESSION = _env_int("OPENAI_IMAGE_OUTPUT_COMPRESSION", 85)
if _OPENAI_IMAGE_OUTPUT_COMPRESSION < 0 or _OPENAI_IMAGE_OUTPUT_COMPRESSION > 100:
    _OPENAI_IMAGE_OUTPUT_COMPRESSION = 85
RUN_SCHEMA_MIGRATIONS = _env_bool("RUN_SCHEMA_MIGRATIONS", False)

_XAI_API_HOST = os.getenv("XAI_API_HOST", "api.x.ai").strip() or "api.x.ai"
_XAI_TIMEOUT_SECONDS = _env_float("XAI_TIMEOUT_SECONDS", 120.0)

_GEMINI_TIMEOUT_MS = _env_int("GEMINI_TIMEOUT_MS", 120000)

_HTTPX_LIMITS = httpx.Limits(
    max_connections=_HTTP_MAX_CONNECTIONS,
    max_keepalive_connections=_HTTP_MAX_KEEPALIVE,
    keepalive_expiry=_HTTP_KEEPALIVE_EXPIRY
)

_OPENAI_HTTPX_TIMEOUT = httpx.Timeout(
    connect=_OPENAI_CONNECT_TIMEOUT,
    read=_OPENAI_READ_TIMEOUT,
    write=_OPENAI_WRITE_TIMEOUT,
    pool=_OPENAI_POOL_TIMEOUT
)

_OPENAI_HTTPX_CLIENT = httpx.Client(http2=_HTTP2_ENABLED, limits=_HTTPX_LIMITS, timeout=_OPENAI_HTTPX_TIMEOUT)
_GEMINI_HTTPX_CLIENT = httpx.Client(http2=_HTTP2_ENABLED, limits=_HTTPX_LIMITS)

_CLIENT_CACHE_LOCK = threading.Lock()
_OPENAI_CLIENT_CACHE = {}
_GEMINI_CLIENT_CACHE = {}
_XAI_CLIENT_CACHE = {}

def _get_openai_client(api_key, base_url=None):
    sig = _key_sig(api_key, base_url or "openai")
    if not sig:
        return None
    with _CLIENT_CACHE_LOCK:
        client = _OPENAI_CLIENT_CACHE.get(sig)
        if client:
            return client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_OPENAI_HTTPX_TIMEOUT,
            max_retries=_OPENAI_MAX_RETRIES,
            http_client=_OPENAI_HTTPX_CLIENT
        )
        _OPENAI_CLIENT_CACHE[sig] = client
        return client

def _get_gemini_client(api_key):
    sig = _key_sig(api_key, "gemini")
    if not sig:
        return None
    with _CLIENT_CACHE_LOCK:
        client = _GEMINI_CLIENT_CACHE.get(sig)
        if client:
            return client
        http_options = types.HttpOptions(
            api_version='v1beta',
            timeout=_GEMINI_TIMEOUT_MS,
            httpx_client=_GEMINI_HTTPX_CLIENT
        )
        client = genai.Client(api_key=api_key, http_options=http_options)
        _GEMINI_CLIENT_CACHE[sig] = client
        return client

def _get_xai_client(api_key):
    if not XAI_SDK_AVAILABLE:
        return None
    sig = _key_sig(api_key, _XAI_API_HOST)
    if not sig:
        return None
    with _CLIENT_CACHE_LOCK:
        client = _XAI_CLIENT_CACHE.get(sig)
        if client:
            return client
        client = XAIClient(
            api_key=api_key,
            api_host=_XAI_API_HOST,
            timeout=_XAI_TIMEOUT_SECONDS
        )
        _XAI_CLIENT_CACHE[sig] = client
    return client

app = Flask(__name__)
app.config['APP_VERSION'] = os.getenv('APP_VERSION', '2026-01-16-001')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 280}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'instance/uploads')
app.config['CHANGELOG_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/changelogs')
_upload_max_mb = int(os.getenv('UPLOAD_MAX_MB', '512') or '512')
app.config['MAX_CONTENT_LENGTH'] = _upload_max_mb * 1024 * 1024
_user_storage_limit_mb = int(os.getenv('USER_STORAGE_LIMIT_MB', '100') or '100')
app.config['USER_STORAGE_LIMIT_MB'] = _user_storage_limit_mb
_primary_admin_username = (os.getenv('PRIMARY_ADMIN_USERNAME') or '').strip()
app.config['PRIMARY_ADMIN_USERNAME'] = _primary_admin_username or None
app.config['MAINTENANCE_MODE'] = os.path.exists(os.path.join(os.path.dirname(__file__), 'maintenance.lock'))

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.before_request
def _apply_per_user_upload_limits():
    if request.endpoint not in ('upload', 'upload_chunk'):
        return
    try:
        if current_user.is_authenticated and _is_primary_admin_user(current_user):
            request.max_content_length = None
        else:
            limit = _get_user_storage_limit_bytes(current_user) if current_user.is_authenticated else None
            if limit:
                if request.content_length and request.content_length > limit:
                    limit_mb = _bytes_to_mb_str(limit)
                    return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
                if request.endpoint == 'upload_chunk':
                    hard_cap = app.config.get('MAX_CONTENT_LENGTH') or limit
                    request.max_content_length = min(hard_cap, limit)
                else:
                    used = _get_user_storage_usage_bytes(current_user.id)
                    remaining = max(0, limit - used)
                    hard_cap = app.config.get('MAX_CONTENT_LENGTH') or remaining
                    request.max_content_length = min(hard_cap, remaining if remaining > 0 else 1)
            else:
                request.max_content_length = app.config.get('MAX_CONTENT_LENGTH')
    except Exception:
        request.max_content_length = app.config.get('MAX_CONTENT_LENGTH')

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

def _chunk_session_dir(user_id, upload_id):
    return os.path.join(_chunk_user_dir(user_id), upload_id)

def _load_chunk_meta(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _save_chunk_meta(path, meta):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f)
        return True
    except Exception:
        return False

KEY_FILE = os.path.join(os.path.dirname(__file__), 'secret.key')
cipher = None
try:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as kf: cipher = Fernet(kf.read().strip())
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as kf: kf.write(key)
        cipher = Fernet(key)
except Exception as e: logger.error(f'Encryption setup failed: {e}')

def encrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.encrypt(val.encode()).decode()
    except: return val

def decrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.decrypt(val.encode()).decode()
    except: return val

def encrypt_bytes(data):
    if not cipher: return data
    return cipher.encrypt(data)

def decrypt_bytes(data):
    if not cipher: return data
    return cipher.decrypt(data)

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

GEMINI_TTS_VOICES = {
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe",
    "Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar",
    "Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"
}

def secure_delete(path):
    if os.path.exists(path):
        try:
            size = os.path.getsize(path)
            with open(path, "wb") as f: f.write(os.urandom(size))
            os.remove(path)
        except: pass

# Speech-to-Speech (STS) model registry
STS_MODELS = {
    "gpt-realtime": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-mini": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gemini-2.5-flash-native-audio-preview-12-2025": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
    "grok-voice-agent": {"provider": "xai", "rate_in": 24000, "rate_out": 24000},
}
OPENAI_STS_VOICES = {
    "alloy","ash","ballad","coral","echo","sage","shimmer","verse","marin","cedar"
}
XAI_STS_VOICES = {"Ara","Rex","Sal","Eve","Leo"}
GEMINI_STS_VOICES = {
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe",
    "Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar",
    "Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"
}
XAI_PCM_RATES = {8000,16000,21050,24000,32000,44100,48000}

def is_sts_model(model_key):
    return model_key in STS_MODELS

def get_sts_provider(model_key):
    meta = STS_MODELS.get(model_key)
    return meta.get("provider") if meta else None

def _chunk_bytes(data, chunk_size=32000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def _convert_audio_to_pcm(audio_bytes, src_suffix=".webm", rate=24000):
    suffix = src_suffix if src_suffix.startswith('.') else f".{src_suffix}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as in_f, tempfile.NamedTemporaryFile(suffix=".pcm") as out_f:
        in_f.write(audio_bytes)
        in_f.flush()
        cmd = [
            "ffmpeg", "-y",
            "-i", in_f.name,
            "-ac", "1",
            "-ar", str(rate),
            "-f", "s16le",
            out_f.name
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_f.seek(0)
        return out_f.read()

def _pcm_to_wav_bytes(pcm_bytes, rate=24000):
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def _save_user_audio(user_id, data, suffix, encrypt):
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
    user = None
    try:
        user = User.query.get(user_id)
    except Exception:
        user = None
    if user:
        ok, used, limit = _check_storage_capacity(user, len(data) if data else 0)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            raise StorageLimitError(f"Storage limit exceeded ({used_mb} / {limit_mb})", used=used, limit=limit)
    fname = f"audio_{int(time.time())}_{os.urandom(4).hex()}{suffix}"
    fpath = os.path.join(user_dir, fname)
    if encrypt:
        with open(fpath + '.enc', 'wb') as f: f.write(encrypt_bytes(data))
    else:
        with open(fpath, 'wb') as f: f.write(data)
    return fname, fpath

async def _openai_sts_realtime(pcm_bytes, api_key, model_key, voice="alloy", speed=None, rate=24000):
    # OpenAI Realtime currently supports 24kHz PCM audio for output; keep session aligned.
    rate = 24000
    url = f"wss://api.openai.com/v1/realtime?model={model_key}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    audio_out = bytearray()
    transcript_out = ""
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": model_key,
                "output_modalities": ["audio"],
                "voice": voice,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "turn_detection": None
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": rate}
                    }
                }
            }
        }
        if speed is not None:
            session_update["session"]["speed"] = speed
        await ws.send(json.dumps(session_update))
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        for chunk in _chunk_bytes(pcm_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode('utf-8')
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        resp_cfg = {"voice": voice}
        if speed is not None:
            resp_cfg["speed"] = speed
        await ws.send(json.dumps({"type": "response.create", "response": resp_cfg}))
        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            if mtype == "error":
                logger.error(f"OpenAI STS error event: {msg}")
            elif mtype and mtype.startswith("response."):
                logger.debug(f"OpenAI STS event: {mtype}")
            if mtype in ("response.output_audio.delta", "response.audio.delta"):
                delta = msg.get("delta")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype in ("response.output_audio", "response.audio"):
                delta = msg.get("audio") or msg.get("data")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    transcript_out += delta
            elif mtype in ("response.output_audio.done", "response.done"):
                break
    return bytes(audio_out), transcript_out

async def _xai_sts_realtime(pcm_bytes, api_key, model_key="grok-voice-agent", voice="Ara", rate_in=24000, rate_out=24000):
    url = f"wss://{_XAI_API_HOST}/v1/realtime?model={model_key}"
    headers = {"Authorization": f"Bearer {api_key}"}
    audio_out = bytearray()
    transcript_out = ""
    async with websockets.connect(url, ssl=True, additional_headers=headers, max_size=None) as ws:
        session_update = {
            "type": "session.update",
            "session": {
                "voice": voice,
                "turn_detection": {"type": None},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": rate_in}},
                    "output": {"format": {"type": "audio/pcm", "rate": rate_out}}
                }
            }
        }
        await ws.send(json.dumps(session_update))
        for chunk in _chunk_bytes(pcm_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode('utf-8')
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait for commit confirmation when using client-side VAD
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("type") == "input_audio_buffer.committed":
                    break
        except Exception:
            pass

        await ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio", "text"]}}))
        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            if mtype == "error":
                logger.error(f"xAI STS error event: {msg}")
            elif mtype and mtype.startswith("response."):
                logger.debug(f"xAI STS event: {mtype}")
            if mtype == "response.output_audio.delta":
                delta = msg.get("delta")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio":
                delta = msg.get("audio")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    transcript_out += delta
            elif mtype in ("response.output_audio.done", "response.done"):
                break
    return bytes(audio_out), transcript_out

async def _google_sts_live(pcm_bytes, api_key, model_key, rate=16000, voice="Kore"):
    client = _get_gemini_client(api_key)
    audio_out = bytearray()
    transcript_out = ""
    input_transcript = ""
    live_conf = {"response_modalities": ["AUDIO"]}
    if voice and voice in GEMINI_STS_VOICES:
        live_conf["speech_config"] = {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": voice}
            }
        }
    async with client.aio.live.connect(
        model=model_key,
        config=live_conf,
    ) as session:
        await session.send_realtime_input(
            media=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={rate}")
        )
        await session.send_realtime_input(audio_stream_end=True)
        async for msg in session.receive():
            if msg.data:
                audio_out += msg.data
            sc = getattr(msg, "server_content", None)
            if sc:
                if getattr(sc, "output_transcription", None) and sc.output_transcription.text:
                    transcript_out += sc.output_transcription.text
                if getattr(sc, "input_transcription", None) and sc.input_transcription.text:
                    input_transcript = sc.input_transcription.text
                if sc.turn_complete:
                    break
    return bytes(audio_out), transcript_out, input_transcript

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    is_admin = db.Column(db.Boolean, default=False)
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)
    google_api_key = db.Column(db.Text, nullable=True)
    google_cloud_project = db.Column(db.Text, nullable=True)
    stt_model = db.Column(db.String(64), default="gpt-4o-mini-transcribe")
    enter_to_send = db.Column(db.Boolean, default=False)
    use_sw_cache = db.Column(db.Boolean, default=False)
    theme_color = db.Column(db.String(16), default="")
    auto_search_on_links = db.Column(db.Boolean, default=True)
    use_last_chat_settings = db.Column(db.Boolean, default=False)
    default_enable_search = db.Column(db.Boolean, default=False)
    default_enable_python = db.Column(db.Boolean, default=True)
    default_enable_thinking = db.Column(db.Boolean, default=False)
    default_thinking_level = db.Column(db.String(16), default="high")
    default_thinking_budget = db.Column(db.Integer, default=4096)
    default_reasoning_effort = db.Column(db.String(16), default="medium")
    default_enable_system_prompt = db.Column(db.Boolean, default=False)
    default_safety_setting = db.Column(db.String(16), default="default")
    last_enable_search = db.Column(db.Boolean, default=False)
    last_enable_python = db.Column(db.Boolean, default=True)
    last_enable_thinking = db.Column(db.Boolean, default=False)
    last_thinking_level = db.Column(db.String(16), default="high")
    last_thinking_budget = db.Column(db.Integer, default=4096)
    last_reasoning_effort = db.Column(db.String(16), default="medium")
    last_enable_system_prompt = db.Column(db.Boolean, default=False)
    last_safety_setting = db.Column(db.String(16), default="default")
    easy_login_hash = db.Column(db.Text, nullable=True)
    easy_login_expires_at = db.Column(db.DateTime, nullable=True)
    is_setup_completed = db.Column(db.Boolean, default=False)
    enable_e2ee = db.Column(db.Boolean, default=False)
    # 2FA Fields
    is_2fa_enabled = db.Column(db.Boolean, default=False)
    totp_secret = db.Column(db.String(32), nullable=True) # Encrypted
    webauthn_credentials = db.Column(db.Text, nullable=True) # JSON list
    passkey_only_login = db.Column(db.Boolean, default=False)
    bot_detection_enabled = db.Column(db.Boolean, default=True)
    is_bot_banned = db.Column(db.Boolean, default=False)
    bot_banned_at = db.Column(db.DateTime, nullable=True)
    bot_ban_reason = db.Column(db.Text, nullable=True)
    bot_unbanned_at = db.Column(db.DateTime, nullable=True)
    bot_unban_notice = db.Column(db.Boolean, default=False)
    appeal_blocked = db.Column(db.Boolean, default=False)
    appeal_block_reason = db.Column(db.Text, nullable=True)
    appeal_blocked_at = db.Column(db.DateTime, nullable=True)
    threads = db.relationship('Thread', backref='user', lazy=True, cascade="all, delete-orphan")
    gems = db.relationship('Gem', backref='user', lazy=True, cascade="all, delete-orphan")
    sessions = db.relationship('UserSession', backref='user', lazy=True, cascade="all, delete-orphan")
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(128), unique=True, index=True, nullable=False)
    user_agent = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_revoked = db.Column(db.Boolean, default=False)
    revoked_at = db.Column(db.DateTime, nullable=True)

class UserClientToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(128), index=True, nullable=False)
    ip_address = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)

class BannedIdentifier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(16), index=True, nullable=False)  # ip / cookie
    value = db.Column(db.String(255), index=True, nullable=False)
    reason = db.Column(db.Text, nullable=True)
    source_user_id = db.Column(db.Integer, nullable=True)
    source_username = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Thread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(64), unique=True, index=True, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    is_bookmarked = db.Column(db.Boolean, default=False)
    bookmarked_at = db.Column(db.DateTime, nullable=True)
    custom_instruction = db.Column(db.Text, nullable=True)
    include_global_instruction = db.Column(db.Boolean, default=True)
    last_model = db.Column(db.String(64), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='thread', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('thread.id'), nullable=False)
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    model = db.Column(db.String(50))
    image_url = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    thought_data = db.Column(db.Text)
    quote_text = db.Column(db.Text)
    is_encrypted = db.Column(db.Boolean, default=False)
    thought_signature = db.Column(db.Text, nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=True)
    children = db.relationship('Message', backref=db.backref('parent', remote_side=[id]), lazy=True)

class Gem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    instruction = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="")
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="new")  # new, in_review, replied, rejected, resolved
    admin_reply = db.Column(db.Text, nullable=True)
    handled_by = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class BanAppeal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="new")  # new, in_review, replied, resolved, rejected
    admin_note = db.Column(db.Text, nullable=True)
    admin_reply = db.Column(db.Text, nullable=True)
    admin_read_at = db.Column(db.DateTime, nullable=True)
    replied_at = db.Column(db.DateTime, nullable=True)
    handled_at = db.Column(db.DateTime, nullable=True)
    handled_by = db.Column(db.String(80), nullable=True)
    ban_reason = db.Column(db.Text, nullable=True)
    ban_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class AppSetting(db.Model):
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

def get_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

def get_client_ip():
    fwd = request.headers.get('X-Forwarded-For', '')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.remote_addr

CLIENT_TOKEN_COOKIE = "ai_client_token"

def _is_secure_request():
    if request.is_secure:
        return True
    proto = request.headers.get('X-Forwarded-Proto', '')
    return proto.lower() == 'https'

def get_client_token():
    token = request.cookies.get(CLIENT_TOKEN_COOKIE)
    if not token:
        token = secrets.token_urlsafe(24)
        g.new_client_token = token
    g.client_token = token
    return token

def is_request_banned_identifier():
    ip = get_client_ip()
    token = request.cookies.get(CLIENT_TOKEN_COOKIE)
    clauses = []
    if ip:
        clauses.append((BannedIdentifier.kind == 'ip') & (BannedIdentifier.value == ip))
    if token:
        clauses.append((BannedIdentifier.kind == 'cookie') & (BannedIdentifier.value == token))
    if not clauses:
        return False
    try:
        return BannedIdentifier.query.filter(or_(*clauses)).first() is not None
    except Exception:
        return False

def _is_admin_exempt(user):
    return bool(getattr(user, "is_admin", False)) or _is_primary_admin_user(user)

def record_user_client_token(user):
    if not user:
        return
    token = get_client_token()
    if not token:
        return
    now = datetime.utcnow()
    ip = get_client_ip()
    row = UserClientToken.query.filter_by(user_id=user.id, token=token).first()
    if row:
        if not row.last_seen_at or (now - row.last_seen_at) > timedelta(minutes=5):
            row.last_seen_at = now
            if ip:
                row.ip_address = ip
            safe_db_commit()
        return
    row = UserClientToken(
        user_id=user.id,
        token=token,
        ip_address=ip,
        created_at=now,
        last_seen_at=now
    )
    db.session.add(row)
    safe_db_commit()

def _ensure_banned_identifier(kind, value, reason, source_user):
    if not value:
        return
    existing = BannedIdentifier.query.filter_by(kind=kind, value=value).first()
    if existing:
        return
    entry = BannedIdentifier(
        kind=kind,
        value=value,
        reason=reason,
        source_user_id=getattr(source_user, "id", None),
        source_username=getattr(source_user, "username", None)
    )
    db.session.add(entry)

def ban_related_accounts(user, reason):
    if not user or _is_admin_exempt(user):
        return
    ban_reason = reason or "Linked ban"
    ips = set()
    tokens = set()
    try:
        if current_user.is_authenticated and current_user.id == user.id:
            ip_now = get_client_ip()
            if ip_now:
                ips.add(ip_now)
            token_now = get_client_token()
            if token_now:
                tokens.add(token_now)
    except Exception:
        pass
    for s in UserSession.query.filter_by(user_id=user.id).all():
        if s.ip_address:
            ips.add(s.ip_address)
    for t in UserClientToken.query.filter_by(user_id=user.id).all():
        if t.token:
            tokens.add(t.token)

    for ip in ips:
        _ensure_banned_identifier("ip", ip, ban_reason, user)
    for token in tokens:
        _ensure_banned_identifier("cookie", token, ban_reason, user)

    user_ids = set()
    if ips:
        for s in UserSession.query.filter(UserSession.ip_address.in_(list(ips))).all():
            if s.user_id:
                user_ids.add(s.user_id)
    if tokens:
        for t in UserClientToken.query.filter(UserClientToken.token.in_(list(tokens))).all():
            if t.user_id:
                user_ids.add(t.user_id)
    now = datetime.utcnow()
    for uid in user_ids:
        u = User.query.get(uid)
        if not u or _is_admin_exempt(u):
            continue
        if not u.is_bot_banned:
            u.is_bot_banned = True
            u.bot_banned_at = now
        if not u.bot_ban_reason:
            u.bot_ban_reason = ban_reason
    safe_db_commit()

def _get_user_identifiers(user):
    ips = set()
    tokens = set()
    if not user:
        return ips, tokens
    for s in UserSession.query.filter_by(user_id=user.id).all():
        if s.ip_address:
            ips.add(s.ip_address)
    for t in UserClientToken.query.filter_by(user_id=user.id).all():
        if t.token:
            tokens.add(t.token)
    if current_user.is_authenticated and current_user.id == user.id:
        try:
            ip_now = get_client_ip()
            if ip_now:
                ips.add(ip_now)
        except Exception:
            pass
        try:
            token_now = get_client_token()
            if token_now:
                tokens.add(token_now)
        except Exception:
            pass
    return ips, tokens

def _unban_user(user):
    if not user:
        return
    user.is_bot_banned = False
    user.bot_ban_reason = None
    user.bot_banned_at = None
    user.bot_unbanned_at = datetime.utcnow()
    user.bot_unban_notice = True

def _unblock_identifiers(ips, tokens):
    if not ips and not tokens:
        return
    clauses = []
    if ips:
        clauses.append((BannedIdentifier.kind == 'ip') & (BannedIdentifier.value.in_(list(ips))))
    if tokens:
        clauses.append((BannedIdentifier.kind == 'cookie') & (BannedIdentifier.value.in_(list(tokens))))
    if not clauses:
        return
    BannedIdentifier.query.filter(or_(*clauses)).delete(synchronize_session=False)

def unban_single_account(user):
    if not user or _is_admin_exempt(user):
        return
    ips, tokens = _get_user_identifiers(user)
    _unban_user(user)
    _unblock_identifiers(ips, tokens)
    safe_db_commit()

def unban_linked_accounts(user):
    if not user or _is_admin_exempt(user):
        return
    ips, tokens = _get_user_identifiers(user)
    user_ids = set()
    if ips:
        for s in UserSession.query.filter(UserSession.ip_address.in_(list(ips))).all():
            if s.user_id:
                user_ids.add(s.user_id)
    if tokens:
        for t in UserClientToken.query.filter(UserClientToken.token.in_(list(tokens))).all():
            if t.user_id:
                user_ids.add(t.user_id)
    for uid in user_ids:
        u = User.query.get(uid)
        if not u or _is_admin_exempt(u):
            continue
        _unban_user(u)
    _unblock_identifiers(ips, tokens)
    safe_db_commit()

def generate_thread_public_id():
    for _ in range(8):
        candidate = secrets.token_urlsafe(32)
        if not Thread.query.filter_by(public_id=candidate).first():
            return candidate
    return secrets.token_urlsafe(32)

def resolve_thread_for_user(identifier, user_id):
    if identifier is None:
        return None
    ident_str = str(identifier).strip()
    if not ident_str:
        return None
    t = None
    if ident_str.isdigit():
        t = Thread.query.get(int(ident_str))
        if t and t.user_id == user_id and not t.public_id:
            return t
    t = Thread.query.filter_by(public_id=ident_str).first()
    if t and t.user_id == user_id:
        return t
    return None

def create_user_session(user):
    sid = secrets.token_urlsafe(32)
    session['session_id'] = sid
    user_sess = UserSession(
        user_id=user.id,
        session_id=sid,
        user_agent=request.headers.get('User-Agent', ''),
        ip_address=get_client_ip()
    )
    db.session.add(user_sess)
    safe_db_commit()
    return user_sess

def revoke_user_sessions(user_id, exclude_session_id=None):
    q = UserSession.query.filter_by(user_id=user_id, is_revoked=False)
    if exclude_session_id:
        q = q.filter(UserSession.session_id != exclude_session_id)
    now = datetime.utcnow()
    for s in q.all():
        s.is_revoked = True
        s.revoked_at = now
    safe_db_commit()

@app.context_processor
def inject_csrf():
    is_admin = current_user.is_authenticated and bool(getattr(current_user, "is_admin", False))
    return {'csrf_token': get_csrf_token(), 'app_version': app.config.get('APP_VERSION'), 'is_admin': is_admin}

def validate_csrf():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    return token and token == session.get('csrf_token')

def get_app_setting(key, default=None):
    try:
        row = AppSetting.query.get(key)
        if row is None:
            return default
        return row.value
    except Exception:
        return default

def set_app_setting(key, value):
    row = AppSetting.query.get(key)
    if row is None:
        row = AppSetting(key=key, value=str(value))
        db.session.add(row)
    else:
        row.value = str(value)
    row.updated_at = datetime.utcnow()
    safe_db_commit()

def ensure_app_setting(key, default):
    try:
        row = AppSetting.query.get(key)
        if row is None:
            db.session.add(AppSetting(key=key, value=str(default)))
            safe_db_commit()
    except Exception:
        pass

def try_alter(sql):
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SET SESSION lock_wait_timeout=1"))
            conn.execute(text(sql))
    except Exception:
        pass

def ensure_thread_last_model_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='last_model'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN last_model VARCHAR(64)"))
    except Exception:
        pass

def get_bool_app_setting(key, default=False):
    val = get_app_setting(key, None)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def get_bot_detection_global_enabled():
    return get_bool_app_setting("bot_detection_global_enabled", True)

@app.before_request
def ensure_client_token():
    try:
        get_client_token()
    except Exception:
        pass

@app.after_request
def set_client_token_cookie(response):
    token = getattr(g, "new_client_token", None)
    if token:
        response.set_cookie(
            CLIENT_TOKEN_COOKIE,
            token,
            max_age=60 * 60 * 24 * 365 * 2,
            httponly=True,
            samesite="Lax",
            secure=_is_secure_request()
        )
    return response

@app.before_request
def check_maintenance():
    if app.config.get('MAINTENANCE_MODE'):
        if request.endpoint in ['static', 'login', 'logout', 'toggle_maintenance', 'login_passkey_options', 'login_passkey_verify']: return
        if current_user.is_authenticated and getattr(current_user, "is_admin", False): return
        return render_template('maintenance.html'), 503
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        if request.endpoint not in ['static']:
            if not validate_csrf():
                return jsonify({'error': 'CSRF token missing/invalid'}), 403

@app.before_request
def check_bot_ban():
    if not current_user.is_authenticated:
        return
    if getattr(current_user, "is_admin", False):
        return
    if current_user.bot_unban_notice:
        flash("ボット検出によるBANが解除されました。")
        current_user.bot_unban_notice = False
        safe_db_commit()
    if current_user.is_bot_banned:
        if request.endpoint in ['logout', 'static', 'banned', 'submit_ban_appeal', 'api_ban_appeal_status']:
            return
        if request.endpoint in ['api_ban_appeal', 'api_ban_appeals_summary']:
            return
        if request.path.startswith('/api/'):
            return jsonify({'error': 'banned'}), 403
        return redirect(url_for('banned'))

@app.before_request
def ensure_active_session():
    if not current_user.is_authenticated:
        return
    if request.endpoint == 'static':
        return
    sid = session.get('session_id')
    if not sid:
        try:
            create_user_session(current_user)
        except Exception:
            pass
        return
    user_sess = UserSession.query.filter_by(user_id=current_user.id, session_id=sid).first()
    if not user_sess or user_sess.is_revoked:
        session.pop('session_id', None)
        logout_user()
        if request.path.startswith('/api/'):
            return jsonify({'error': 'session_revoked'}), 401
        return redirect(url_for('login'))
    now = datetime.utcnow()
    if not user_sess.last_seen_at or (now - user_sess.last_seen_at) > timedelta(seconds=30):
        user_sess.last_seen_at = now
        user_sess.ip_address = get_client_ip() or user_sess.ip_address
        ua = request.headers.get('User-Agent', '')
        if ua:
            user_sess.user_agent = ua
        try:
            safe_db_commit()
        except Exception:
            pass
    try:
        record_user_client_token(current_user)
    except Exception:
        pass

def verify_turnstile(token):
    secret = os.getenv('TURNSTILE_SECRET_KEY')
    if not secret: return False
    if not token: return False
    try: return requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify', data={'secret': secret, 'response': token}, timeout=5).json().get('success', False)
    except: return False

def rate_limit(key, limit, window_seconds):
    try:
        cur = redis_conn.incr(key)
        if cur == 1:
            redis_conn.expire(key, window_seconds)
        return cur <= limit
    except Exception:
        return True

def count_tokens(text, model="gpt-4"):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except:
        return len(text or "") // 4

def should_count_tokens_for_display(model_key):
    return not (model_key and 'grok' in model_key.lower())

def count_tokens_for_display(text, model_key):
    if not should_count_tokens_for_display(model_key):
        return None
    return count_tokens(text)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(exc.SQLAlchemyError))
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

# --- Background Tasks ---

def migrate_e2ee_task(user_id, target_enable):
    with app.app_context():
        db.engine.dispose()
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

def safe_execute_python(code):
    """Executes Python code in a restricted environment using bubblewrap."""
    import subprocess
    import tempfile
    import os
    import shutil

    py_path = shutil.which("python3")
    if not py_path:
        return "Error: python3 not found."

    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "Error: Python execution disabled (sandbox not available)."

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
            "--dir", "/tmp",
            "--chdir", "/tmp",
        ]
        for b in binds:
            cmd.extend(list(b))
        cmd.extend(["--bind", td, "/work", py_path, "/work/code.py"])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = (result.stdout or "") + (result.stderr or "")
            return out if out.strip() else "Success (No output)"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (30s limit)"
        except Exception as e:
            return f"Error: {str(e)}"

def background_chat_task(job_id, thread_id, model_key, message_id, options, user_id, user_config):
    with app.app_context():
        db.engine.dispose()
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def pub(dt, d): r.publish(channel, json.dumps({"type": dt, "content": d}))
        
        def check_stop():
            if r.get(f"stop_job:{job_id}"):
                log_force(f"Job {job_id} stopped by user.")
                return True
            return False

        try:
            log_force(f"Task Start: model={model_key}, user={user_id}")
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
            
            # Global prompt (only if enabled via checkbox or if it's a Gem forced prompt)
            global_prompt = None
            if options.get('enable_system_prompt'):
                if base_sys_prompt:
                    global_prompt = base_sys_prompt
                elif user.system_prompt:
                    sp = user.system_prompt
                    if user.enable_e2ee: sp = decrypt_val(sp)
                    global_prompt = sp
            else:
                # If master checkbox is OFF, we might still have a Gem prompt in base_sys_prompt
                # But gems usually force enable_system_prompt=True, so this is just a safety.
                global_prompt = base_sys_prompt

            # Thread specific prompt
            th = Thread.query.get(thread_id)
            local_sys_prompt = th.custom_instruction if (th and th.custom_instruction and th.custom_instruction.strip()) else None
            
            final_sys_prompt = ""
            if local_sys_prompt:
                if global_prompt:
                    final_sys_prompt = f"{global_prompt}\n\n[Chat Specific Instructions]:\n{local_sys_prompt}"
                else:
                    final_sys_prompt = local_sys_prompt
            else:
                final_sys_prompt = global_prompt or ""
            
            options['system_prompt'] = final_sys_prompt

            if options.get('enable_python'):
                python_notice = "Python execution is available; you can run Python code when needed."
                curr_p = options.get('system_prompt')
                if curr_p and str(curr_p).strip():
                    if python_notice.lower() not in str(curr_p).lower():
                        options['system_prompt'] = f"{python_notice}\n\n{curr_p}"
                else:
                    options['system_prompt'] = python_notice
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
            
            history = []
            total_history_tokens = 0
            MAX_CONTEXT_TOKENS = 60000
            
            current_node = msg.parent # Start from the parent of the current message
            if current_node and (current_node.thread_id != thread_id or current_node.thread.user_id != user_id):
                current_node = None
            while current_node:
                cnt = decrypt_val(current_node.content) if current_node.is_encrypted else current_node.content
                t_len = count_tokens(cnt)
                
                if total_history_tokens + t_len <= MAX_CONTEXT_TOKENS:
                    history.insert(0, {
                        'role': current_node.role, 
                        'content': cnt, 
                        'image_url': current_node.image_url, 
                        'signature': current_node.thought_signature
                    })
                    total_history_tokens += t_len
                else:
                    break
                
                current_node = current_node.parent

            model_key = model_key.strip()
            is_gem = 'gemini' in model_key or 'nano' in model_key
            is_grok = 'grok' in model_key.lower() and 'gpt' not in model_key.lower()
            grok_reasoning_supported = "grok-3-mini" in model_key.lower()

            def _grok_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if raw in ("low", "high"):
                    return raw
                lvl = (options.get('thinking_level') or "low").lower()
                return "high" if lvl == "high" else "low"

            def _grok_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                notice = "You can access external links (including X posts) via the web_search and x_search tools. Use them when the user asks to read URLs or posts."
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice

            def _openai_system_prompt(base_prompt, enable_search):
                if not enable_search:
                    return base_prompt
                notice = "You can access external links via the web_search tool. If a URL cannot be accessed, say so clearly."
                if base_prompt and str(base_prompt).strip():
                    return f"{notice}\n\n{base_prompt}"
                return notice
            
            def get_k(db_val, env_key):
                k = decrypt_val(db_val)
                if k and str(k).strip():
                    return k
                if user and getattr(user, 'is_admin', False):
                    return os.getenv(env_key)
                return None

            api_keys = {
                'openai': get_k(user.openai_api_key, 'OPENAI_API_KEY'),
                'gemini': get_k(user.gemini_api_key, 'GEMINI_API_KEY'),
                'xai': get_k(user.xai_api_key, 'XAI_API_KEY')
            }

            key = None
            if is_gem: key = api_keys.get('gemini')
            elif is_grok: key = api_keys.get('xai')
            else: key = api_keys.get('openai') 

            if not key:
                pub("error", "API Key missing")
                return

            g_client = None; o_client = None; x_client = None
            if is_gem: g_client = _get_gemini_client(key)
            elif is_grok:
                x_client = _get_xai_client(key)
                o_client = _get_openai_client(key, base_url=f"https://{_XAI_API_HOST}/v1")
            else: o_client = _get_openai_client(key, base_url=None)

            loaded_files = []
            for fn in img_list:
                bp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
                ep = bp + '.enc'
                data = None
                mime = mimetypes.guess_type(bp)[0] or 'application/octet-stream'
                try:
                    if os.path.exists(bp):
                        with open(bp, 'rb') as f: data = f.read()
                    elif os.path.exists(ep):
                        with open(ep, 'rb') as f: data = decrypt_bytes(f.read())
                    if data:
                        if fn.lower().endswith('.pdf'):
                            reader = pypdf.PdfReader(BytesIO(data))
                            extracted = "".join([p.extract_text() + "\n" for p in reader.pages])
                            loaded_files.append({'name': fn, 'text': extracted[:50000], 'bytes': None, 'mime': 'application/pdf'})
                        else: loaded_files.append({'name': fn, 'text': None, 'bytes': data, 'mime': mime})
                except: pass

            full_res, thought_accumulated, generated_images = "", "", []
            signature_parts = []

            final_message_text = message_text
            if quote_text:
                final_message_text = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"

            auto_enable_search = options.get('enable_search')
            grok_enable_search = auto_enable_search
            user_auto_search = True
            try:
                user_auto_search = bool(getattr(user, "auto_search_on_links", True))
            except Exception:
                user_auto_search = True
            disable_auto = bool(options.get('disable_auto_search'))
            if is_grok and not grok_enable_search and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        grok_enable_search = True
                        auto_enable_search = True
                        log_force("Auto-enabled Grok search for URL/X post access")
                except Exception:
                    pass
            if not is_grok and not auto_enable_search and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {quote_text or ''}"
                    if re.search(r'https?://', check_text) or "x.com/" in check_text or "twitter.com/" in check_text:
                        auto_enable_search = True
                        log_force("Auto-enabled Web search for URL/X post access")
                except Exception:
                    pass

            # --- 1. GEMINI & GEMINI IMAGE ---
            if is_gem:
                log_force("Routing: Gemini Branch")
                
                # Gemini TTS (Preview)
                if "tts" in model_key:
                    try:
                        voice_name = (options.get('tts_voice') or "Kore").strip()
                        if voice_name not in GEMINI_TTS_VOICES:
                            voice_name = "Kore"
                        tts_lang = (options.get('tts_language') or "").strip() or None
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

                            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                            if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
                            speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.wav"
                            speech_file_path = os.path.join(user_dir, speech_file_name)

                            if user_config.get('enable_e2ee'):
                                with open(speech_file_path + '.enc', 'wb') as f: f.write(encrypt_bytes(wav_bytes))
                            else:
                                with open(speech_file_path, 'wb') as f: f.write(wav_bytes)

                            audio_url = f"/files/{user_id}/{speech_file_name}"
                            audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
                            full_res += audio_tag
                            pub("content", audio_tag)
                            generated_images.append(f"{user_id}/{speech_file_name}")
                    except Exception as e:
                        logger.exception("Gemini TTS Error")
                        pub("error", f"Gemini TTS Error: {str(e)}")

                # Image Generation
                elif "nano" in model_key or "image" in model_key:
                    try:
                        # [FIX] Apply System Prompt to Image Prompts if available
                        img_prompt = final_message_text
                        if options.get('system_prompt'):
                            img_prompt = f"{options.get('system_prompt')}\n\n{final_message_text}"

                        img_model = "gemini-2.5-flash-image" if "2.5" in model_key else "gemini-3-pro-image-preview"
                        aspect_allowed = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "auto"}
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
                        if size_val and "gemini-3-pro-image-preview" in img_model:
                            image_cfg_kwargs["image_size"] = size_val
                        config_kwargs = {
                            "temperature": 0.7,
                            "safety_settings": [
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                            ]
                        }
                        if image_cfg_kwargs:
                            config_kwargs["image_config"] = types.ImageConfig(**image_cfg_kwargs)

                        resp = g_client.models.generate_content(
                            model=img_model,
                            contents=[
                                *[
                                    types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime'])
                                    for fi in loaded_files
                                    if fi.get('bytes') and fi.get('mime', '').startswith('image/')
                                ],
                                types.Part(text=img_prompt)
                            ],
                            config=types.GenerateContentConfig(**config_kwargs)
                        )
                        
                        if resp.candidates:
                            cand0 = resp.candidates[0]
                            parts0 = getattr(getattr(cand0, 'content', None), 'parts', None) or []
                            for part in parts0:
                                if hasattr(part, 'thought_signature') and part.thought_signature:
                                    signature_parts.append(base64.b64encode(part.thought_signature).decode('utf-8'))

                                if hasattr(part, 'inline_data') and part.inline_data:
                                    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                                    os.makedirs(ud, exist_ok=True)
                                    mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                                    ext_map = {
                                        "image/png": "png",
                                        "image/jpeg": "jpg",
                                        "image/webp": "webp"
                                    }
                                    ext = ext_map.get(mime, "png")
                                    fn2 = f"gen_{int(time.time())}_{len(generated_images)}.{ext}"
                                    fp2 = os.path.join(ud, fn2)
                                    if user_config.get('enable_e2ee'):
                                        img_data = part.inline_data.data
                                        if isinstance(img_data, str):
                                            img_data = base64.b64decode(img_data)
                                        with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(img_data))
                                    else:
                                        img_data = part.inline_data.data
                                        if isinstance(img_data, str):
                                            img_data = base64.b64decode(img_data)
                                        with open(fp2, 'wb') as f: f.write(img_data)
                                    generated_images.append(f"{user_id}/{fn2}")
                                    pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                                    full_res += f"Generated Image for: {img_prompt}\n"
                        else:
                             pub("error", "No image candidates returned.")
                    except Exception as e:
                        logger.exception("Gemini Image Gen Error")
                        pub("error", f"Gemini Image Gen Error: {str(e)}")

                else:
                    # Text/Chat generation mode
                    rm = model_key
                    if "gemini-3-flash" in model_key or "gemini-3.0-flash" in model_key:
                        rm = "gemini-3-flash-preview"
                    elif "gemini-3-pro" in model_key or "gemini-3.0-pro" in model_key:
                        rm = "gemini-3-pro-preview"
                    elif "gemini-2.5-flash-lite" in model_key:
                        rm = model_key
                    elif "gemini-2.5" in model_key:
                        rm = "gemini-2.5-flash"

                    conf = {'temperature': 0.7}
                    is_gemini_3 = "gemini-3" in model_key
                    if is_gemini_3:
                        # Gemini 3 does not support fully disabling thinking; force enabled.
                        options['enable_thinking'] = True
                    if options.get('enable_thinking'):
                        raw_lvl = (options.get('thinking_level') or 'high').lower()
                        lvl = raw_lvl if raw_lvl in ("minimal", "low", "medium", "high") else "high"
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
                    # For Gemini 3, if thinking_level is not specified, the model defaults to "high".
                    # Avoid forcing "minimal" when users disable thinking, because Gemini 3 does not
                    # support fully turning thinking off and defaults are higher per docs.

                    if options.get('enable_search'):
                        conf['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('enable_python'):
                        if 'tools' not in conf: conf['tools'] = []
                        conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))
                    if options.get('system_prompt'):
                        conf['system_instruction'] = options.get('system_prompt')
                    
                    contents = []
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
                        if m['content']: parts.append(types.Part(text=m['content']))
                        if m['image_url']:
                            try:
                                for h_img in json.loads(m['image_url']):
                                    bp2 = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                    ep2 = bp2 + '.enc'
                                    d2 = None
                                    if os.path.exists(bp2):
                                        with open(bp2, 'rb') as f: d2 = f.read()
                                    elif os.path.exists(ep2):
                                        with open(ep2, 'rb') as f: d2 = decrypt_bytes(f.read())
                                    if d2:
                                        mime2 = mimetypes.guess_type(bp2)[0] or 'application/octet-stream'
                                        if mime2.startswith('image/'):
                                            parts.append(types.Part.from_bytes(data=d2, mime_type=mime2))
                            except: pass
                        if parts: contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

                    curr_parts = [types.Part(text=final_message_text)]
                    use_raw_parts = False
                    audio_inline_limit = 20 * 1024 * 1024  # 20MiB limit for inline audio

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

                    for fi in loaded_files:
                        if fi['text']:
                            curr_parts.append(types.Part(text=f"\nFile: {fi['name']}\n{fi['text']}"))
                            continue
                        if not fi.get('bytes'):
                            continue
                        mime = (fi.get('mime') or 'application/octet-stream').lower()
                        if mime.startswith('image/'):
                            curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                            continue
                        if mime.startswith('audio/'):
                            try:
                                audio_bytes, audio_mime, audio_name = _normalize_gemini_audio(fi['bytes'], fi.get('mime') or mime, fi.get('name') or "")
                                if len(audio_bytes) <= audio_inline_limit:
                                    curr_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime))
                                else:
                                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_name or '')[1] or '.bin') as tmp:
                                        tmp.write(audio_bytes)
                                        tmp.flush()
                                        up = g_client.files.upload(file=tmp.name, config={"mimeType": audio_mime})
                                    uri = getattr(up, "uri", None) or getattr(up, "name", None) or (up.get("uri") if isinstance(up, dict) else None)
                                    up_mime = getattr(up, "mime_type", None) or getattr(up, "mimeType", None) or audio_mime
                                    if uri and hasattr(types.Part, "from_uri"):
                                        curr_parts.append(types.Part.from_uri(uri=uri, mime_type=up_mime))
                                    else:
                                        use_raw_parts = True
                                        curr_parts.append(up)
                            except Exception as e:
                                log_force(f"Gemini audio upload failed: {e}")
                            continue
                        # Skip unsupported binary inputs for Gemini text models
                        pass

                    if use_raw_parts:
                        contents.extend(curr_parts)
                    else:
                        contents.append(types.Content(role='user', parts=curr_parts))

                    stream = g_client.models.generate_content_stream(model=rm, contents=contents, config=types.GenerateContentConfig(**conf))
                    current_py_id = None
                    current_py_code = None
                    for chunk in stream:
                        if check_stop(): break
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for cand in chunk.candidates:
                                gm = getattr(cand, 'grounding_metadata', None)
                                g_chunks = getattr(gm, 'grounding_chunks', None) or []
                                if g_chunks:
                                    sources_text = "\n\n**Sources:**\n"
                                    found = False
                                    for g_chunk in g_chunks:
                                        if hasattr(g_chunk, 'web') and g_chunk.web:
                                            sources_text += f"- [{g_chunk.web.title}]({g_chunk.web.uri})\n"
                                            found = True
                                    if found: pub("content", sources_text)

                                parts = getattr(getattr(cand, 'content', None), 'parts', None) or []
                                for part in parts:
                                    if hasattr(part, 'thought_signature') and part.thought_signature:
                                        signature_parts.append(base64.b64encode(part.thought_signature).decode('utf-8'))
                                    if hasattr(part, 'thought') and part.thought:
                                        thought_text = part.text or ""
                                        if thought_text:
                                            thought_accumulated += thought_text
                                            pub("thought", thought_text)
                                        continue
                                    if hasattr(part, 'executable_code') and part.executable_code:
                                        c_txt = f"\n```python\n{part.executable_code.code}\n```\n"
                                        full_res += c_txt
                                        pub("content", c_txt)
                                        current_py_id = f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                                        current_py_code = part.executable_code.code
                                        pub("python", {"id": current_py_id, "code": part.executable_code.code})
                                    if hasattr(part, 'code_execution_result') and part.code_execution_result:
                                        r_txt = f"\n**Output:**\n```\n{part.code_execution_result.output}\n```\n"
                                        full_res += r_txt
                                        pub("content", r_txt)
                                        py_id = current_py_id or f"gem_py_{int(time.time()*1000)}_{os.urandom(3).hex()}"
                                        pub("python", {"id": py_id, "output": part.code_execution_result.output})
                                        py_payload = {"code": current_py_code or "", "output": part.code_execution_result.output}
                                        full_res += f"\n```pyexec\n{json.dumps(py_payload)}\n```\n"

                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                                            os.makedirs(ud, exist_ok=True)
                                            mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                                            ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
                                            ext = ext_map.get(mime, "png")
                                            fn2 = f"agentic_{int(time.time())}_{len(generated_images)}.{ext}"
                                            fp2 = os.path.join(ud, fn2)
                                            
                                            img_data = part.inline_data.data
                                            if isinstance(img_data, str):
                                                img_data = base64.b64decode(img_data)
                                            
                                            if user_config.get('enable_e2ee'):
                                                with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(img_data))
                                            else:
                                                with open(fp2, 'wb') as f: f.write(img_data)
                                                
                                            generated_images.append(f"{user_id}/{fn2}")
                                            img_md = f"\n![Agentic View](/files/{user_id}/{fn2})\n"
                                            full_res += img_md
                                            pub("content", img_md)
                                        except Exception as e:
                                            log_force(f"Agentic Vision Image Error: {e}")

                                    if hasattr(part, 'text') and part.text:
                                        full_res += part.text
                                        pub("content", part.text)

            # --- 1.5 Grok Imagine Image Generation ---
            elif model_key == "grok-imagine-image":
                log_force("Routing: Grok Imagine Branch")
                try:
                    pub("content", "**Generating Image (Grok)...**\n")
                    
                    aspect_ratio = options.get('grok_image_aspect') or "1:1"
                    
                    img_response_format = "b64_json"
                    img_kwargs = {
                        "model": "grok-imagine-image",
                        "prompt": final_message_text,
                        "n": 1,
                        "response_format": img_response_format
                    }
                    # aspect_ratio is an xAI-specific parameter; pass via extra_body for generate
                    eb = {}
                    if aspect_ratio:
                        eb["aspect_ratio"] = aspect_ratio

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
                        img_inputs.append((f"input_{len(img_inputs)}", img_bytes, img_mime))

                    img_data_b64 = None
                    if img_inputs:
                        # Use first image for editing as per docs.
                        # xAI image edits expect JSON (not multipart), so send base64.
                        img_bytes = img_inputs[0][1]
                        img_mime = img_inputs[0][2] if len(img_inputs[0]) > 2 else "image/png"
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        img_data_url = f"data:{img_mime};base64,{img_b64}"
                        endpoint = f"https://{_XAI_API_HOST}/v1/images/edits"
                        headers = {
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                        payload = {
                            "model": "grok-imagine-image",
                            "prompt": final_message_text,
                            "image": {"url": img_data_url},
                            "response_format": img_response_format
                        }
                        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=120)
                        if resp.status_code >= 400:
                            try:
                                log_force(f"Grok Imagine edit error {resp.status_code}: {resp.text}")
                            except Exception:
                                pass
                        resp.raise_for_status()
                        resp_json = resp.json()
                        if isinstance(resp_json, dict):
                            if resp_json.get("data") and resp_json["data"]:
                                img_data_b64 = (resp_json["data"][0] or {}).get("b64_json")
                            if not img_data_b64 and resp_json.get("image"):
                                img_data_b64 = resp_json.get("image")
                    else:
                        resp = o_client.images.generate(**img_kwargs, extra_body=eb)
                        if resp.data:
                            img_data_b64 = resp.data[0].b64_json
                    
                    if img_data_b64:
                        img_bytes = base64.b64decode(img_data_b64)
                        
                        ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                        if not os.path.exists(ud): os.makedirs(ud, exist_ok=True)
                        
                        ext = "png"
                        fn2 = f"gen_grok_{int(time.time())}_{len(generated_images)}.{ext}"
                        fp2 = os.path.join(ud, fn2)
                        
                        if user_config.get('enable_e2ee'):
                            with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(img_bytes))
                        else:
                            with open(fp2, 'wb') as f: f.write(img_bytes)
                            
                        generated_images.append(f"{user_id}/{fn2}")
                        pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                        full_res += f"Generated Image for: {final_message_text}\n"
                    else:
                        pub("error", "Grok Image Gen Error: No data returned.")
                except Exception as e:
                    logger.exception("Grok Imagine Error")
                    pub("error", f"Grok Imagine Error: {str(e)}")

            # --- 1.6 Grok Imagine Video Generation ---
            elif model_key == "grok-imagine-video":
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
                    
                    api_key = key # Decrypted XAI API Key
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Determine endpoint and payload
                    endpoint = f"https://{_XAI_API_HOST}/v1/videos/generations"
                    payload = {
                        "model": "grok-imagine-video",
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
                        v_resp = httpx.get(video_url, timeout=60.0)
                        if v_resp.status_code == 200:
                            ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                            if not os.path.exists(ud): os.makedirs(ud, exist_ok=True)
                            
                            fn2 = f"gen_video_{int(time.time())}_{os.urandom(4).hex()}.mp4"
                            fp2 = os.path.join(ud, fn2)
                            
                            if user_config.get('enable_e2ee'):
                                with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(v_resp.content))
                            else:
                                with open(fp2, 'wb') as f: f.write(v_resp.content)
                                
                            generated_images.append(f"{user_id}/{fn2}")
                            vid_tag = f'\n<video controls playsinline preload="metadata" src="/files/{user_id}/{fn2}" class="w-full mt-2"></video>\n'
                            pub("content", vid_tag)
                            full_res += f"Generated Video for: {final_message_text}\n"
                        else:
                            pub("error", f"Failed to download generated video: {v_resp.status_code}")
                    else:
                        pub("error", "Video generation timed out or was canceled.")
                        
                except Exception as e:
                    logger.exception("Grok Imagine Video Error")
                    pub("error", f"Grok Imagine Video Error: {str(e)}")

            # --- 2. xAI Grok (Native SDK) ---
            elif is_grok and x_client and not options.get('enable_python'):
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
                if options.get('enable_thinking') and grok_reasoning_supported:
                    create_kwargs["reasoning_effort"] = _grok_reasoning_effort()
                create_kwargs["use_encrypted_content"] = True # Request encrypted reasoning if available
                if options.get('enable_python') and XAI_SDK_AVAILABLE:
                    create_kwargs["tools"] = [x_code_execution()]

                chat_session = x_client.chat.create(**create_kwargs)

                grok_sys = _grok_system_prompt(options.get('system_prompt'), grok_enable_search)
                if grok_sys: chat_session.append(x_system(grok_sys))
                
                for m in history:
                    if m['role'] == 'user':
                        content_parts = [m['content']]
                        if m['image_url']:
                            try:
                                for h_img in json.loads(m['image_url']):
                                    bp2 = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                    ep2 = bp2 + '.enc'
                                    d2 = None
                                    if os.path.exists(bp2):
                                        with open(bp2, 'rb') as f: d2 = f.read()
                                    elif os.path.exists(ep2):
                                        with open(ep2, 'rb') as f: d2 = decrypt_bytes(f.read())
                                    if d2:
                                        mime = mimetypes.guess_type(bp2)[0] or 'image/webp'
                                        d_uri = f"data:{mime};base64,{base64.b64encode(d2).decode('utf-8')}"
                                        content_parts.append(x_image(d_uri))
                            except: pass
                        chat_session.append(x_user(*content_parts))
                    else: chat_session.append(x_assistant(m['content']))
                
                curr_user_content = [final_message_text]
                for fi in loaded_files:
                    if fi.get('text'): 
                        curr_user_content[0] += f"\n\n[File: {fi['name']}]\n{fi['text']}"
                    elif fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                        d_uri = f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"
                        curr_user_content.append(x_image(d_uri))
                
                chat_session.append(x_user(*curr_user_content))
                
                stream = chat_session.stream()
                search_reported = False
                last_response = None
                for resp, chunk in stream:
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
                    for c in last_response.citations:
                        if hasattr(c, 'url'): url = c.url
                        else: url = str(c)
                        citations_text += f"- {url}\n"
                    full_res += citations_text
                    pub("content", citations_text)

            # --- 2.5 TTS Branch ---
            elif 'tts' in model_key:
                log_force("Routing: TTS Branch")
                try:
                    pub("content", "**Processing Audio Generation...**\n")
                    
                    speech_file_name = f"speech_{int(time.time())}_{os.urandom(4).hex()}.mp3"
                    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                    if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
                    speech_file_path = os.path.join(user_dir, speech_file_name)

                    if 'google-tts' in model_key:
                        # Google Cloud TTS (requires Google Cloud API key, not Gemini API key)
                        g_key = decrypt_val(current_user.google_api_key)
                        if not g_key and getattr(current_user, 'is_admin', False):
                            g_key = os.getenv('GOOGLE_API_KEY')
                        if not g_key:
                            raise RuntimeError("Google API Key is not configured for Google TTS.")
                        g_project = decrypt_val(current_user.google_cloud_project)
                        if not g_project and getattr(current_user, 'is_admin', False):
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
                        with open(speech_file_path, 'wb') as f: f.write(audio_content)
                    else:
                        # OpenAI TTS
                        tts_voice = (options.get('tts_voice') or "alloy").strip().lower() or "alloy"
                        speed_val = clamp_float(options.get('tts_speed'), 0.25, 4.0)
                        tts_kwargs = {
                            "model": model_key,
                            "voice": tts_voice,
                            "input": final_message_text
                        }
                        if speed_val is not None:
                            tts_kwargs["speed"] = speed_val
                        with o_client.audio.speech.with_streaming_response.create(**tts_kwargs) as response:
                            response.stream_to_file(speech_file_path)

                    # Encryption if enabled
                    if user_config.get('enable_e2ee'):
                        with open(speech_file_path, 'rb') as f: data = f.read()
                        with open(speech_file_path + '.enc', 'wb') as f: f.write(encrypt_bytes(data))
                        secure_delete(speech_file_path) # Delete original
                    
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
                    pub("content", "**Generating Image (OpenAI)...**\n")
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
                    img_kwargs = {"model": model_key, "prompt": final_message_text}
                    if size_opt:
                        img_kwargs["size"] = size_opt
                    if quality_opt:
                        img_kwargs["quality"] = quality_opt
                    if format_opt:
                        img_kwargs["output_format"] = format_opt
                        if format_opt in {"jpeg", "webp"}:
                            img_kwargs["output_compression"] = comp_opt if comp_opt is not None else _OPENAI_IMAGE_OUTPUT_COMPRESSION
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
                        img_inputs.append((f"input_{len(img_inputs)}", img_bytes, img_mime))
                    mask_file = None
                    mask_name = options.get('image_mask')
                    if mask_name:
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
                        if mask_file:
                            resp = img_client.images.edit(image=img_inputs, mask=mask_file, **img_kwargs)
                        else:
                            resp = img_client.images.edit(image=img_inputs, **img_kwargs)
                    else:
                        resp = img_client.images.generate(**img_kwargs)
                    if resp.data:
                        img_bytes = base64.b64decode(resp.data[0].b64_json)
                        ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                        if not os.path.exists(ud): os.makedirs(ud, exist_ok=True)
                        ext = "png"
                        if format_opt == "jpeg":
                            ext = "jpg"
                        elif format_opt == "webp":
                            ext = "webp"
                        fn2 = f"gen_gpt_{int(time.time())}_{len(generated_images)}.{ext}"
                        fp2 = os.path.join(ud, fn2)
                        if user_config.get('enable_e2ee'):
                            with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(img_bytes))
                        else:
                            with open(fp2, 'wb') as f: f.write(img_bytes)
                        generated_images.append(f"{user_id}/{fn2}")
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

            # --- 4. OpenAI Responses API (or Grok Fallback) ---
            else:
                log_force("Routing: Responses API Branch")
                client = o_client
                input_data = []
                sys_prompt = _grok_system_prompt(options.get('system_prompt'), grok_enable_search) if is_grok else _openai_system_prompt(options.get('system_prompt'), auto_enable_search)
                if sys_prompt: input_data.append({"role": "system", "content": sys_prompt})
                
                for m in history:
                    content_block = m['content']
                    input_data.append({"role": m['role'], "content": content_block})

                curr_content = []
                text_type = "input_text"
                image_type = "input_image"
                if quote_text: curr_content.append({"type": text_type, "text": f"User Quote:\n{quote_text}\n---"})
                curr_content.append({"type": text_type, "text": message_text})
                
                for fi in loaded_files:
                    if fi['text']:
                        for part in reversed(curr_content):
                            if part.get('type') == text_type:
                                part['text'] += f"\n\n[File: {fi['name']}]\n{fi['text']}"
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
                
                input_data.append({"role": "user", "content": curr_content})
                
                # OpenAI/xAI Responses API
                has_image_inputs = any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files)
                # xAI docs: image understanding requests should avoid server-side storage.
                store_flag = False if (is_grok and has_image_inputs) else True
                kwargs = {"model": model_key, "input": input_data, "stream": True, "store": store_flag}

                if is_grok and grok_enable_search:
                    kwargs['tools'] = [{"type": "web_search"}, {"type": "x_search"}]
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

                if is_grok and options.get('enable_thinking') and not grok_reasoning_supported:
                    pub("thought", "APIの仕様により表示されません")
                is_reasoning_model = (not is_grok) and any(x in model_key.lower() for x in ['o1', 'o3', 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'reasoning'])
                req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
                enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
                if is_grok and enable_reasoning and grok_reasoning_supported:
                    kwargs['reasoning'] = {"effort": _grok_reasoning_effort()}
                    log_force(f"Grok reasoning config: {kwargs['reasoning']}")
                elif is_reasoning_model and enable_reasoning:
                    effort = req_reasoning_effort
                    if not effort:
                        lvl = (options.get('thinking_level') or "medium").lower()
                        effort = "low" if lvl == "low" else "high" if lvl == "high" else "medium"
                    kwargs['reasoning'] = {"effort": effort}
                    kwargs['reasoning']["summary"] = "auto"
                    log_force(f"Reasoning config: {kwargs['reasoning']}")

                log_force(f"Responses API Params: {kwargs.keys()}")
                stream = client.responses.create(**kwargs)
                search_reported = False
                saw_reasoning_summary_delta = False
                response_id = None

                for chunk in stream:
                    if check_stop(): break
                        # log_force(f"Responses Chunk: {chunk}") # Temporarily disabled to avoid log flooding
                    # Capture response_id from any event type (some streams may skip response.created)
                    if response_id is None:
                        if isinstance(chunk, dict):
                            response_id = chunk.get('response_id') or response_id
                        else:
                            response_id = getattr(chunk, 'response_id', None) or response_id
                    if isinstance(chunk, dict):
                        event_type = chunk.get('type')
                    else:
                        event_type = getattr(chunk, 'type', None)

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

                        if item_type == "function_call" and call_name == "execute_python":
                            try:
                                args_json = json.loads(call_args or "{}")
                                code = args_json.get('code', '')
                                if code:
                                    pub("content", f"\n```python\n{code}\n```\n")
                                    result = safe_execute_python(code)
                                    pub("content", f"\n**Output:**\n```\n{result}\n```\n")
                                    full_res += f"\n```python\n{code}\n```\n\n**Output:**\n```\n{result}\n```\n"
                                    full_res += f"\n```pyexec\n{json.dumps({'code': code, 'output': result})}\n```\n"
                                    pub("python", {"id": tool_call_id or f"py_{int(time.time()*1000)}_{os.urandom(3).hex()}", "code": code, "output": result})
                                    if response_id and tool_call_id:
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
                                pub("error", f"Python Tool Error: {e}")

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
                            output_items = resp.get('output')
                        else:
                            response_id = getattr(resp, 'id', None) or response_id
                            output_items = getattr(resp, 'output', None) if resp else None
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

                # Fallback: retrieve full response if no reasoning summary surfaced in stream
                if enable_reasoning and not thought_accumulated and response_id:
                    try:
                        resp_full = client.responses.retrieve(response_id)
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
                    except Exception as e:
                        log_force(f"Reasoning retrieve fallback failed: {e}")
                elif enable_reasoning and not thought_accumulated:
                    log_force("Reasoning summary missing after stream and retrieve fallback.")

            final_content = full_res
            final_signature = json.dumps(signature_parts) if signature_parts else None
            final_thought = json.dumps({'text': thought_accumulated}) if thought_accumulated else None
            is_enc = user_config.get('enable_e2ee', False)
            if is_enc:
                final_content = encrypt_val(final_content)
                if final_thought: final_thought = encrypt_val(final_thought)
            
            msg_entry = Message(
                thread_id=thread_id, role='assistant', content=final_content, 
                model=model_key, image_url=json.dumps(generated_images) if generated_images else None, 
                thought_data=final_thought, tokens=count_tokens_for_display(full_res, model_key), 
                is_encrypted=is_enc, thought_signature=final_signature,
                parent_id=message_id
            )
            db.session.add(msg_entry)
            th = Thread.query.get(thread_id)
            if th:
                th.updated_at = datetime.utcnow()
                th.last_model = model_key
            safe_db_commit()
            pub("done", "OK")

        except Exception as e:
            logger.exception("Worker Error")
            log_force(f"Worker Exception: {e}")
            pub("error", str(e))
        finally:
            r.delete(f"stop_job:{job_id}")

@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        easy_login_used = bool(session.pop('easy_login_used', False))
        bot_config = {
            "username": current_user.username,
            "isAdmin": bool(getattr(current_user, "is_admin", False)),
            "globalEnabled": get_bot_detection_global_enabled(),
            "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
            "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or ""
        }
        return render_template('chat.html', easy_login_used=easy_login_used, bot_config=bot_config)
    return render_template('landing.html')

@app.route('/settings')
@login_required
def settings_page():
    if not current_user.is_setup_completed:
        return redirect(url_for('setup'))
    easy_login_used = bool(session.pop('easy_login_used', False))
    bot_config = {
        "username": current_user.username,
        "isAdmin": bool(getattr(current_user, "is_admin", False)),
        "globalEnabled": get_bot_detection_global_enabled(),
        "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
        "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or ""
    }
    return render_template('chat.html', easy_login_used=easy_login_used, bot_config=bot_config)

@app.route('/c/<thread_id>')
@login_required
def chat_permalink(thread_id):
    thread = resolve_thread_for_user(thread_id, current_user.id)
    if not thread:
        return render_template('404.html', message="指定されたチャットは存在しません。"), 404
    easy_login_used = bool(session.pop('easy_login_used', False))
    bot_config = {
        "username": current_user.username,
        "isAdmin": bool(getattr(current_user, "is_admin", False)),
        "globalEnabled": get_bot_detection_global_enabled(),
        "accountEnabled": current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
        "turnstileSiteKey": os.getenv('TURNSTILE_SITE_KEY') or ""
    }
    initial_thread_id = thread.public_id or thread.id
    return render_template('chat.html', initial_thread_id=initial_thread_id, easy_login_used=easy_login_used, bot_config=bot_config)

@app.route('/changelog')
def changelog():
    log_dir = app.config['CHANGELOG_FOLDER']
    logs = []
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.md'))
        def _changelog_meta(path):
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r'^(\\d{4}-\\d{2}-\\d{2})_v(.+)$', base)
            if not m:
                m = re.match(r'^(\\d{8})_v(.+)$', base)
            if m:
                date_raw, version = m.group(1), m.group(2)
                if len(date_raw) == 8:
                    date_fmt = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date_fmt = date_raw
                date_key = int(date_fmt.replace('-', ''))
                ver_nums = tuple(int(x) for x in re.findall(r'\\d+', version)) or (0,)
                title = f"V{version} ({date_fmt})"
                return date_key, ver_nums, title
            return 0, (0,), base
        files.sort(key=lambda p: _changelog_meta(p)[:2], reverse=True)
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
            title = None
            if not content.lstrip().startswith('#'):
                _, _, title = _changelog_meta(f)
            logs.append({'content': content, 'title': title})
    return render_template('changelog.html', logs=logs)

@app.route('/banned')
@login_required
def banned():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for('index'))
    if not current_user.is_bot_banned:
        return redirect(url_for('index'))
    latest_appeal = None
    try:
        latest_appeal = BanAppeal.query.filter_by(user_id=current_user.id).order_by(BanAppeal.created_at.desc()).first()
    except Exception:
        latest_appeal = None
    return render_template(
        'banned.html',
        reason=current_user.bot_ban_reason,
        banned_at=current_user.bot_banned_at,
        latest_appeal=latest_appeal,
        appeal_submitted=session.pop('appeal_submitted', False),
        appeal_error=session.pop('appeal_error', None),
        appeal_blocked=bool(getattr(current_user, "appeal_blocked", False)),
        appeal_block_reason=getattr(current_user, "appeal_block_reason", None)
    )

@app.route('/ban/appeal', methods=['POST'])
@login_required
def submit_ban_appeal():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for('index'))
    if not current_user.is_bot_banned:
        return redirect(url_for('index'))
    if getattr(current_user, "appeal_blocked", False):
        session['appeal_error'] = current_user.appeal_block_reason or "異議申し立てはブロックされています。"
        return redirect(url_for('banned'))
    message = (request.form.get('message') or '').strip()
    if not message or len(message) < 10:
        session['appeal_error'] = "内容は10文字以上で入力してください。"
        return redirect(url_for('banned'))
    if len(message) > 3000:
        session['appeal_error'] = "内容は3000文字以内で入力してください。"
        return redirect(url_for('banned'))
    appeal = BanAppeal(
        user_id=current_user.id,
        username=current_user.username,
        message=message,
        ban_reason=current_user.bot_ban_reason,
        ban_at=current_user.bot_banned_at,
        ip_address=get_client_ip(),
        user_agent=request.headers.get('User-Agent', '')
    )
    db.session.add(appeal)
    safe_db_commit()
    session['appeal_submitted'] = True
    return redirect(url_for('banned'))

@app.route('/api/ban/appeal/status')
@login_required
def api_ban_appeal_status():
    if getattr(current_user, 'is_admin', False):
        return jsonify({'error': 'admin_not_allowed'}), 403
    latest = BanAppeal.query.filter_by(user_id=current_user.id).order_by(BanAppeal.created_at.desc()).first()
    if not latest:
        return jsonify({'has_appeal': False})
    return jsonify({
        'has_appeal': True,
        'status': latest.status,
        'created_at': latest.created_at.isoformat() + "Z" if latest.created_at else None
    })

@app.route('/api/version')
def api_version():
    resp = jsonify({'version': app.config.get('APP_VERSION', '')})
    resp.headers['Cache-Control'] = 'no-store'
    return resp

# -----------------------------------------------------------
# Auth Routes
# -----------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        if not rate_limit(f"rl:login:ip:{request.remote_addr}", 20, 300):
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
        if not verify_turnstile(request.form.get('cf-turnstile-response')): return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
        username = (request.form.get('username') or '').strip()
        user = User.query.filter_by(username=username).first()
        # Allow login even if IP/Cookie is banned; ban screen will handle after login.
        if user:
            if not rate_limit(f"rl:login:user:{user.id}", 10, 300):
                return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
            pw = request.form.get('password') or ""
            now = datetime.utcnow()
            easy_ok = False
            try:
                if user.easy_login_hash and user.easy_login_expires_at and now <= user.easy_login_expires_at:
                    easy_ok = check_password_hash(user.easy_login_hash, pw)
            except Exception:
                easy_ok = False
            if easy_ok:
                # One-time easy login: disable after first successful use
                user.easy_login_hash = None
                user.easy_login_expires_at = None
                safe_db_commit()
                session['easy_login_used'] = True
                remember = bool(request.form.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                return redirect(url_for('index'))
            if user.easy_login_hash and user.easy_login_expires_at and now > user.easy_login_expires_at:
                user.easy_login_hash = None
                user.easy_login_expires_at = None
                safe_db_commit()
            if user.check_password(pw):
                if user.is_2fa_enabled:
                    session['remember_me'] = bool(request.form.get('remember'))
                    session['pre_2fa_user_id'] = user.id
                    return redirect(url_for('verify_2fa'))
                remember = bool(request.form.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                return redirect(url_for('index'))
        return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Invalid credentials")
    return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'))

@app.route('/login/passkey/options', methods=['POST'])
def login_passkey_options():
    if current_user.is_authenticated:
        return jsonify({'error': 'already_authenticated'}), 400
    if not rate_limit(f"rl:login:ip:{request.remote_addr}", 20, 300):
        return jsonify({'error': 'Too many attempts. Try again later.'}), 429
    data = request.json or {}
    if not verify_turnstile(data.get('turnstile')):
        return jsonify({'error': 'Auth Error'}), 401
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': 'Username required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not getattr(user, "passkey_only_login", False):
        return jsonify({'error': 'Invalid credentials'}), 400
    # Allow passkey login even if IP/Cookie is banned; ban screen will handle after login.
    if not rate_limit(f"rl:login:user:{user.id}", 10, 300):
        return jsonify({'error': 'Too many attempts. Try again later.'}), 429
    creds = []
    if user.webauthn_credentials:
        try:
            creds = json.loads(user.webauthn_credentials)
        except Exception:
            creds = []
    if not creds:
        return jsonify({'error': 'No credentials'}), 400
    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.PREFERRED
    )
    session['passkey_login_user_id'] = user.id
    session['webauthn_login_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    session['passkey_login_remember'] = bool(data.get('remember'))
    return options_to_json(options)

@app.route('/login/passkey/verify', methods=['POST'])
def login_passkey_verify():
    if current_user.is_authenticated:
        return jsonify({'error': 'already_authenticated'}), 400
    user_id = session.get('passkey_login_user_id')
    if not user_id:
        return jsonify({'error': 'Session expired'}), 401
    if not rate_limit(f"rl:webauthn:user:{user_id}", 8, 300):
        return jsonify({'error': 'Too many attempts'}), 429
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'Invalid user'}), 400
    try:
        data = request.json
        challenge = session.get('webauthn_login_challenge')
        if not challenge:
            return jsonify({'error': 'Challenge missing'}), 400
        creds = json.loads(user.webauthn_credentials) if user.webauthn_credentials else []
        current_cred = next((c for c in creds if c['id'] == data['id']), None)
        if not current_cred:
            return jsonify({'error': 'Credential not found'}), 400
        verification = verify_authentication_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=False
        )
        current_cred['sign_count'] = verification.new_sign_count
        user.webauthn_credentials = json.dumps(creds)
        db.session.commit()
        session.pop('passkey_login_user_id', None)
        session.pop('webauthn_login_challenge', None)
        remember = bool(session.pop('passkey_login_remember', False))
        login_user(user, remember=remember)
        create_user_session(user)
        record_user_client_token(user)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Passkey Login Verify Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    if current_user.is_authenticated: return redirect(url_for('index'))
    user_id = session.get('pre_2fa_user_id')
    if not user_id: return redirect(url_for('login'))
    
    user = User.query.get(user_id)
    if not user: return redirect(url_for('login'))

    if request.method == 'POST':
        if not rate_limit(f"rl:2fa:user:{user.id}", 8, 300):
            return render_template('verify_2fa.html', error="Too many attempts. Try again later.")
        code = request.form.get('totp_code')
        if code:
            secret = decrypt_val(user.totp_secret)
            if secret and pyotp.TOTP(secret).verify(code):
                session.pop('pre_2fa_user_id', None)
                remember = bool(session.pop('remember_me', False))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                return redirect(url_for('index'))
            return render_template('verify_2fa.html', error="Invalid Code")
            
    return render_template('verify_2fa.html')

@app.route('/verify-2fa/webauthn/options', methods=['POST'])
def verify_2fa_webauthn_options():
    user_id = session.get('pre_2fa_user_id')
    logger.info(f"WebAuthn Options Req: user_id={user_id}, session={session.keys()}")
    if not user_id: return jsonify({'error': 'Session expired'}), 401
    user = User.query.get(user_id)
    
    creds = []
    if user.webauthn_credentials:
        try: creds = json.loads(user.webauthn_credentials)
        except Exception as e: logger.error(f"JSON Parse Error: {e}")
    
    logger.info(f"User Creds Count: {len(creds)}")
    if not creds: return jsonify({'error': 'No credentials'}), 400

    options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in creds
        ],
        user_verification=UserVerificationRequirement.PREFERRED
    )
    
    session['webauthn_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    return options_to_json(options)

@app.route('/verify-2fa/webauthn/verify', methods=['POST'])
def verify_2fa_webauthn_verify():
    user_id = session.get('pre_2fa_user_id')
    if not user_id: return jsonify({'error': 'Session expired'}), 401
    user = User.query.get(user_id)
    if not rate_limit(f"rl:webauthn:user:{user_id}", 8, 300):
        return jsonify({'error': 'Too many attempts'}), 429
    
    try:
        data = request.json
        challenge = session.get('webauthn_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        creds = json.loads(user.webauthn_credentials) if user.webauthn_credentials else []
        current_cred = next((c for c in creds if c['id'] == data['id']), None)
        if not current_cred: return jsonify({'error': 'Credential not found'}), 400

        verification = verify_authentication_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            credential_public_key=base64url_to_bytes(current_cred['public_key']),
            credential_current_sign_count=current_cred['sign_count'],
            require_user_verification=False # Depends on device
        )
        
        current_cred['sign_count'] = verification.new_sign_count
        user.webauthn_credentials = json.dumps(creds)
        db.session.commit()
        
        session.pop('pre_2fa_user_id', None)
        remember = bool(session.pop('remember_me', False))
        login_user(user, remember=remember)
        create_user_session(user)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"WebAuthn Verify Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        if not rate_limit(f"rl:signup:ip:{request.remote_addr}", 10, 3600):
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
        if not verify_turnstile(request.form.get('cf-turnstile-response')): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
        if is_request_banned_identifier():
            return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Signup blocked.")
        if _is_primary_admin_username(request.form.get('username')): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username taken")
        if User.query.filter_by(username=request.form.get('username')).first(): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username taken")
        new_user = User(username=request.form.get('username'), is_setup_completed=False)
        new_user.set_password(request.form.get('password'))
        db.session.add(new_user)
        safe_db_commit()
        login_user(new_user)
        record_user_client_token(new_user)
        return redirect(url_for('setup'))
    return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'))

@app.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if current_user.is_setup_completed: return redirect(url_for('index'))
    if request.method == 'POST':
        current_user.openai_api_key = encrypt_val(request.form.get('openai_key'))
        current_user.gemini_api_key = encrypt_val(request.form.get('gemini_key'))
        current_user.xai_api_key = encrypt_val(request.form.get('xai_key'))
        current_user.google_api_key = encrypt_val(request.form.get('google_key'))
        current_user.google_cloud_project = encrypt_val(request.form.get('google_project'))
        current_user.enable_e2ee = (request.form.get('enable_e2ee') == 'on')
        current_user.is_setup_completed = True
        safe_db_commit()
        return redirect(url_for('index'))
    return render_template('setup.html')

@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        sid = session.get('session_id')
        if sid:
            user_sess = UserSession.query.filter_by(user_id=current_user.id, session_id=sid, is_revoked=False).first()
            if user_sess:
                user_sess.is_revoked = True
                user_sess.revoked_at = datetime.utcnow()
                try:
                    safe_db_commit()
                except Exception:
                    pass
    logout_user()
    session.pop('session_id', None)
    return redirect(url_for('index'))

# -----------------------------------------------------------
# API Routes
# -----------------------------------------------------------

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    data = request.json
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}"

    thread_id = data.get('thread_id')
    if not thread_id:
        return jsonify({'error': 'thread_id required'}), 400
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    thread_id = t.id
    
    user_msg = None
    try:
        raw_msg_content = data.get('message')
        msg_content = raw_msg_content
        if user_config['enable_e2ee']: msg_content = encrypt_val(msg_content)
        
        parent_id = data.get('parent_id')
        if parent_id:
            try:
                pm = Message.query.get(int(parent_id))
                if not pm or pm.thread_id != thread_id or pm.thread.user_id != current_user.id:
                    parent_id = None
                else:
                    parent_id = pm.id
            except Exception:
                parent_id = None
        if not parent_id:
            # Default to the last message in the thread
            last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
            if last_msg:
                parent_id = last_msg.id

        user_msg = Message(
            thread_id=thread_id,
            role='user',
            content=msg_content,
            image_url=json.dumps(data.get('image_urls', [])) if data.get('image_urls') else None,
            quote_text=data.get('quote_text'),
            is_encrypted=user_config['enable_e2ee'],
            parent_id=parent_id,
            tokens=count_tokens_for_display(raw_msg_content, data.get('model'))
        )
        db.session.add(user_msg)
        safe_db_commit()
    except Exception as e:
        logger.error(f"Failed to save user msg: {e}")
        return jsonify({'error': 'Failed to save message'}), 500

    quote_text = data.get('quote_text')
    if quote_text:
        try:
            redis_conn.setex(f"quote:{job_id}", 600, quote_text)
        except: pass

    sys_prompt = data.get('system_prompt')
    if sys_prompt:
        try:
            redis_conn.setex(f"sys:{job_id}", 600, sys_prompt)
        except: pass

    options = {
        'system_prompt': None,
        'enable_search': data.get('enable_search'),
        'disable_auto_search': data.get('disable_auto_search'),
        'enable_python': data.get('enable_python'),
        'enable_thinking': data.get('enable_thinking'),
        'thinking_level': data.get('thinking_level'),
        'thinking_budget': data.get('thinking_budget'),
        'reasoning_effort': data.get('reasoning_effort'),
        'enable_system_prompt': data.get('enable_system_prompt'),
        'safety_setting': data.get('safety_setting'),
        'tts_voice': data.get('tts_voice'),
        'tts_voice_custom': data.get('tts_voice_custom'),
        'tts_language': data.get('tts_language'),
        'tts_speed': data.get('tts_speed'),
        'image_size': data.get('image_size'),
        'image_quality': data.get('image_quality'),
        'image_format': data.get('image_format'),
        'image_compression': data.get('image_compression'),
        'image_mask': data.get('image_mask'),
        'gemini_image_aspect': data.get('gemini_image_aspect'),
        'gemini_image_size': data.get('gemini_image_size'),
        'grok_image_aspect': data.get('grok_image_aspect'),
        'grok_image_format': data.get('grok_image_format'),
        'grok_video_duration': data.get('grok_video_duration'),
        'grok_video_aspect': data.get('grok_video_aspect'),
        'grok_video_resolution': data.get('grok_video_resolution'),
    }

    if current_user.use_last_chat_settings:
        current_user.last_enable_search = bool(data.get('enable_search'))
        current_user.last_enable_python = bool(data.get('enable_python'))
        current_user.last_enable_thinking = bool(data.get('enable_thinking'))
        current_user.last_thinking_level = (data.get('thinking_level') or current_user.last_thinking_level or "high")
        tb = data.get('thinking_budget')
        try:
            if tb is not None and str(tb).strip() != "":
                current_user.last_thinking_budget = int(tb)
        except Exception:
            pass
        current_user.last_reasoning_effort = (data.get('reasoning_effort') or current_user.last_reasoning_effort or "medium")
        current_user.last_enable_system_prompt = bool(data.get('enable_system_prompt'))
        current_user.last_safety_setting = (data.get('safety_setting') or current_user.last_safety_setting or "default")
        safe_db_commit()

    task_queue.enqueue(background_chat_task, job_id, thread_id, data.get('model'), user_msg.id, options, current_user.id, user_config, job_timeout=600)
    
    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        yield json.dumps({"type": "job_id", "data": job_id}) + "\n"
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    yield json.dumps(data) + "\n"
                    if data['type'] in ['done', 'error']: break
        finally: pubsub.unsubscribe()
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/api/stop_chat', methods=['POST'])
@login_required
def stop_chat():
    job_id = request.json.get('job_id')
    if job_id:
        redis_conn.set(f"stop_job:{job_id}", "1", ex=300)
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'no job_id'}), 400

@app.route('/api/generate_title', methods=['POST'])
@login_required
def generate_title_api():
    """Auto-generate chat title with multi-model fallback"""
    try:
        data = request.json
        thread_id = data.get('thread_id')
        thread = resolve_thread_for_user(thread_id, current_user.id)
        if not thread:
            return jsonify({'error': 'Unauthorized'}), 403

        first_msg = Message.query.filter_by(thread_id=thread.id, role='user').order_by(Message.timestamp).first()
        if not first_msg: return jsonify({'status': 'skipped'})
        
        content = decrypt_val(first_msg.content) if first_msg.is_encrypted else first_msg.content
        
        # [FIX] Multi-model fallback logic
        title = "New Chat"
        o_key = decrypt_val(current_user.openai_api_key) or (os.getenv('OPENAI_API_KEY') if getattr(current_user, 'is_admin', False) else None)
        g_key = decrypt_val(current_user.gemini_api_key) or (os.getenv('GEMINI_API_KEY') if getattr(current_user, 'is_admin', False) else None)
        x_key = decrypt_val(current_user.xai_api_key) or (os.getenv('XAI_API_KEY') if getattr(current_user, 'is_admin', False) else None)

        # Try OpenAI (gpt-4o-mini)
        if o_key:
            try:
                client = _get_openai_client(o_key, base_url=None)
                resp = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output JSON: {\"title\": \"...\"}"},
                        {"role": "user", "content": content[:500]}
                    ]
                )
                raw = None
                if isinstance(resp, dict):
                    raw = resp.get("output_text")
                else:
                    raw = getattr(resp, "output_text", None)
                if not raw and hasattr(resp, "output"):
                    try:
                        raw = resp.output[0].content[0].text
                    except Exception:
                        raw = None
                if raw:
                    title = json.loads(raw).get('title', 'New Chat')
            except: pass
        
        # Try Gemini (flash)
        elif g_key and title == "New Chat":
            try:
                g_client = _get_gemini_client(g_key)
                resp = g_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. JSON: {{'title': '...'}}\n\nChat: {content[:500]}")],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                title = json.loads(resp.text).get('title', 'New Chat')
            except: pass

        # Try xAI (grok-fast)
        elif x_key and XAI_SDK_AVAILABLE and title == "New Chat":
            try:
                x_client = XAIClient(api_key=x_key)
                chat = x_client.chat.create(model="grok-4-1-fast-non-reasoning")
                chat.append(x_system("Generate a short, descriptive title (max 6 words) for this chat conversation. Output only the title text without any quotes or JSON."))
                chat.append(x_user(content[:500]))
                resp = chat.sample()
                if resp and resp.message and resp.message.content:
                    title = resp.message.content.strip()
            except: pass
            
        thread.title = title
        safe_db_commit()
        return jsonify({'status': 'ok', 'title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/<path:filename>')
@login_required
def serve_file(filename):
    norm = os.path.normpath(filename)
    if norm.startswith("..") or os.path.isabs(norm): abort(403)
    if not norm.startswith(f"{current_user.id}/"): abort(403)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], norm)
    if not os.path.realpath(file_path).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])): abort(403)
    enc_path = file_path + '.enc'
    mtype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    if os.path.exists(file_path):
        resp = send_file(file_path, mimetype=mtype, conditional=True)
        resp.headers.setdefault("Accept-Ranges", "bytes")
        return resp
    elif os.path.exists(enc_path):
        with open(enc_path, 'rb') as f:
            data = decrypt_bytes(f.read())
        range_header = request.headers.get('Range')
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                size = len(data)
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
                chunk = data[start:end + 1]
                resp = Response(chunk, status=206, mimetype=mtype, direct_passthrough=True)
                resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Content-Length"] = str(end - start + 1)
                return resp
        return send_file(BytesIO(data), download_name=os.path.basename(filename), as_attachment=False, mimetype=mtype)
    else:
        abort(404)

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = 20
        query = Thread.query.filter_by(user_id=current_user.id)
        if q: 
            if current_user.enable_e2ee: query = query.filter(Thread.title.contains(q))
            else: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
        
        pagination = query.order_by(Thread.is_bookmarked.desc(), Thread.bookmarked_at.desc(), Thread.updated_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        threads = [{
            'id': t.public_id or t.id,
            'title': t.title,
            'is_bookmarked': bool(t.is_bookmarked),
            'last_model': t.last_model
        } for t in pagination.items]
        return jsonify({
            'threads': threads,
            'has_next': pagination.has_next,
            'next_page': pagination.next_num
        })
    t = Thread(user_id=current_user.id, public_id=generate_thread_public_id())
    db.session.add(t)
    safe_db_commit()
    return jsonify({'id': t.public_id, 'title': t.title})

@app.route('/api/threads/<thread_id>', methods=['GET', 'DELETE'])
@login_required
def handle_thread_item(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        # Ensure stable ordering even when timestamps collide (e.g., rapid edit/regenerate).
        ms = Message.query.filter_by(thread_id=t.id).order_by(Message.timestamp, Message.id).all()
        res = []
        for m in ms:
            cnt = decrypt_val(m.content) if m.is_encrypted else m.content
            tht = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
            token_count = None
            if should_count_tokens_for_display(m.model):
                if m.tokens is not None and m.tokens > 0:
                    token_count = m.tokens
                else:
                    token_count = count_tokens(cnt)
            res.append({
                'id': m.id, 
                'role': m.role, 
                'content': cnt, 
                'image_url': m.image_url, 
                'model': m.model, 
                'thought_data': tht,
                'tokens': token_count,
                'quote_text': m.quote_text,
                'parent_id': m.parent_id
            })
        return jsonify({
            'messages': res,
            'custom_instruction': t.custom_instruction,
            'include_global_instruction': t.include_global_instruction if t.include_global_instruction is not None else True,
            'last_model': t.last_model
        })
    
    for m in t.messages:
        if m.image_url:
            try:
                paths = json.loads(m.image_url)
                if not isinstance(paths, list): paths = [paths]
                for p in paths:
                    norm = os.path.normpath(p)
                    if norm.startswith("..") or os.path.isabs(norm): continue
                    if not norm.startswith(f"{current_user.id}/"): continue
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
                    if not os.path.realpath(fp).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])): continue
                    secure_delete(fp)
                    secure_delete(fp + '.enc')
            except: pass

    db.session.delete(t)
    safe_db_commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/threads/<thread_id>/settings', methods=['PUT'])
@login_required
def update_thread_settings(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    d = request.json
    if 'custom_instruction' in d:
        t.custom_instruction = d['custom_instruction']
    if 'include_global_instruction' in d:
        t.include_global_instruction = bool(d['include_global_instruction'])
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/threads/<thread_id>/title', methods=['PUT'])
@login_required
def update_title(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/threads/<thread_id>/bookmark', methods=['POST'])
@login_required
def toggle_bookmark(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    t.is_bookmarked = not bool(t.is_bookmarked)
    t.bookmarked_at = datetime.utcnow() if t.is_bookmarked else None
    safe_db_commit()
    return jsonify({'status': 'ok', 'is_bookmarked': t.is_bookmarked})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    if msg.thread.user_id != current_user.id: return jsonify({'error': '403'}), 403
    
    msgs_to_delete = Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).all()
    for m in msgs_to_delete:
        if m.image_url:
            try:
                paths = json.loads(m.image_url)
                if not isinstance(paths, list): paths = [paths]
                for p in paths:
                    norm = os.path.normpath(p)
                    if norm.startswith("..") or os.path.isabs(norm): continue
                    if not norm.startswith(f"{current_user.id}/"): continue
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
                    if not os.path.realpath(fp).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])): continue
                    secure_delete(fp)
                    secure_delete(fp + '.enc')
            except: pass

    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    try:
        msgs = Message.query.join(Thread).filter(Thread.user_id == current_user.id, Message.image_url != None).order_by(Message.timestamp.desc()).all()
        files = []
        seen = set()
        for m in msgs:
            if not m.image_url: continue
            try:
                l = json.loads(m.image_url)
                if not isinstance(l, list): l = [m.image_url]
            except: l = [m.image_url]
            for p in l:
                if p and p not in seen:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], p)
                    if os.path.exists(fp) or os.path.exists(fp + '.enc'):
                        seen.add(p)
                        ext = os.path.splitext(p)[1].lower().replace('.', '')
                        files.append({'filename': os.path.basename(p), 'filepath': p, 'url': url_for('serve_file', filename=p), 'type': 'image' if ext in ['png','jpg','webp'] else 'file', 'ext': ext})
        # Include uploaded files that are not yet attached to any message
        ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.isdir(ud):
            for entry in os.scandir(ud):
                if not entry.is_file():
                    continue
                name = entry.name
                is_enc = name.endswith('.enc')
                base_name = name[:-4] if is_enc else name
                if not base_name:
                    continue
                rel_path = f"{current_user.id}/{base_name}"
                if rel_path in seen:
                    continue
                ext = os.path.splitext(base_name)[1].lower().replace('.', '')
                if not ext:
                    continue
                seen.add(rel_path)
                files.append({
                    'filename': os.path.basename(rel_path),
                    'filepath': rel_path,
                    'url': url_for('serve_file', filename=rel_path),
                    'type': 'image' if ext in ['png','jpg','webp','jpeg','gif'] else 'file',
                    'ext': ext
                })
        return jsonify(files)
    except: return jsonify([])

@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    for f in request.json.get('filenames', []):
        norm = os.path.normpath(f)
        if norm.startswith("..") or os.path.isabs(norm): continue
        if norm.startswith(f"{current_user.id}/"):
            fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
            if not os.path.realpath(fp).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])): continue
            secure_delete(fp)
            secure_delete(fp + '.enc')
    return jsonify({'status': 'ok'})

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    try:
        # Remove feedback records first (no cascade)
        Feedback.query.filter_by(user_id=current_user.id).delete()
        BanAppeal.query.filter_by(user_id=current_user.id).delete()

        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.exists(user_dir):
            for root, dirs, files in os.walk(user_dir, topdown=False):
                for name in files: secure_delete(os.path.join(root, name))
                for name in dirs: os.rmdir(os.path.join(root, name))
            os.rmdir(user_dir)
        # Clear migration status/progress
        redis_conn.delete(f"migration_status:{current_user.id}")
        redis_conn.delete(f"migration_progress:{current_user.id}")
        db.session.delete(current_user)
        safe_db_commit()
        logout_user()
        return jsonify({'status': 'ok'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        data = request.json or {}
        title = (data.get('title') or "").strip()[:200]
        message = (data.get('message') or "").strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        fb = Feedback(user_id=current_user.id, title=title, message=message)
        db.session.add(fb)
        safe_db_commit()
        return jsonify({'status': 'ok'})

    # GET
    is_admin = bool(getattr(current_user, "is_admin", False))
    if is_admin and request.args.get('all') == '1':
        items = Feedback.query.order_by(Feedback.created_at.desc()).all()
    else:
        items = Feedback.query.filter_by(user_id=current_user.id).order_by(Feedback.created_at.desc()).all()

    res = []
    for f in items:
        res.append({
            'id': f.id,
            'user_id': f.user_id,
            'title': f.title or "",
            'message': f.message,
            'status': f.status,
            'admin_reply': f.admin_reply or "",
            'handled_by': f.handled_by or "",
            'created_at': f.created_at.isoformat(),
            'updated_at': f.updated_at.isoformat() if f.updated_at else None
        })
    return jsonify({'items': res, 'is_admin': is_admin})

@app.route('/api/easy_login', methods=['POST'])
@login_required
def create_easy_login():
    try:
        data = request.json or {}
        if data.get('cancel'):
            current_user.easy_login_hash = None
            current_user.easy_login_expires_at = None
            safe_db_commit()
            return jsonify({'status': 'ok', 'cancelled': True})
        minutes = data.get('minutes', 5)
        try:
            minutes = int(minutes)
        except Exception:
            minutes = 5
        if minutes < 1: minutes = 1
        if minutes > 120: minutes = 120

        temp_pw = secrets.token_urlsafe(6)[:10]
        current_user.easy_login_hash = generate_password_hash(temp_pw)
        current_user.easy_login_expires_at = datetime.utcnow() + timedelta(minutes=minutes)
        safe_db_commit()
        return jsonify({
            'status': 'ok',
            'temp_password': temp_pw,
            'expires_at': current_user.easy_login_expires_at.isoformat() + "Z",
            'minutes': minutes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/<int:fid>/update', methods=['POST'])
@login_required
def feedback_update(fid):
    if not getattr(current_user, "is_admin", False): return jsonify({'error': '403'}), 403
    fb = Feedback.query.get_or_404(fid)
    data = request.json or {}
    status = data.get('status')
    reply = data.get('admin_reply')
    if status:
        fb.status = status
    if reply is not None:
        fb.admin_reply = reply
    fb.handled_by = current_user.username
    fb.updated_at = datetime.utcnow()
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/ban/appeals/summary', methods=['GET'])
@login_required
def api_ban_appeals_summary():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    unread = BanAppeal.query.filter(BanAppeal.admin_read_at.is_(None)).count()
    return jsonify({'unread_count': unread})

@app.route('/api/ban/appeals', methods=['GET'])
@login_required
def api_ban_appeals():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    limit = request.args.get('limit') or '50'
    try:
        limit = max(1, min(200, int(limit)))
    except Exception:
        limit = 50
    items = BanAppeal.query.order_by(BanAppeal.created_at.desc()).limit(limit).all()
    res = []
    for a in items:
        res.append({
            'id': a.id,
            'user_id': a.user_id,
            'username': a.username,
            'message': a.message,
            'status': a.status,
            'admin_note': a.admin_note or "",
            'admin_reply': a.admin_reply or "",
            'admin_read_at': a.admin_read_at.isoformat() + "Z" if a.admin_read_at else None,
            'replied_at': a.replied_at.isoformat() + "Z" if a.replied_at else None,
            'handled_at': a.handled_at.isoformat() + "Z" if a.handled_at else None,
            'handled_by': a.handled_by or "",
            'ban_reason': a.ban_reason or "",
            'ban_at': a.ban_at.isoformat() + "Z" if a.ban_at else None,
            'ip_address': a.ip_address or "",
            'user_agent': a.user_agent or "",
            'created_at': a.created_at.isoformat() + "Z" if a.created_at else None,
            'updated_at': a.updated_at.isoformat() + "Z" if a.updated_at else None
        })
    return jsonify({'items': res})

@app.route('/api/ban/appeals/mark_read', methods=['POST'])
@login_required
def api_ban_appeals_mark_read():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    ids = data.get('ids') or []
    mark_all = bool(data.get('all'))
    now = datetime.utcnow()
    if mark_all:
        BanAppeal.query.filter(BanAppeal.admin_read_at.is_(None)).update({'admin_read_at': now, 'updated_at': now})
        safe_db_commit()
        return jsonify({'status': 'ok', 'all': True})
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': 'ids_required'}), 400
    items = BanAppeal.query.filter(BanAppeal.id.in_(ids)).all()
    for a in items:
        if not a.admin_read_at:
            a.admin_read_at = now
        a.updated_at = now
    safe_db_commit()
    return jsonify({'status': 'ok', 'count': len(items)})

@app.route('/api/ban/appeals/update', methods=['POST'])
@login_required
def api_ban_appeals_update():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    appeal_id = data.get('id')
    if not appeal_id:
        return jsonify({'error': 'id_required'}), 400
    appeal = BanAppeal.query.get_or_404(int(appeal_id))
    status = data.get('status')
    admin_note = data.get('admin_note')
    admin_reply = data.get('admin_reply')
    block_user = bool(data.get('block_user')) if 'block_user' in data else False
    unblock_user = bool(data.get('unblock_user')) if 'unblock_user' in data else False
    block_reason = (data.get('block_reason') or '').strip()
    now = datetime.utcnow()
    if status in ['new', 'in_review', 'replied', 'resolved', 'rejected']:
        appeal.status = status
        if status in ['resolved', 'rejected']:
            appeal.handled_at = now
            appeal.handled_by = current_user.username
    if admin_note is not None:
        appeal.admin_note = admin_note.strip()[:2000]
    if admin_reply is not None:
        reply_text = admin_reply.strip()
        appeal.admin_reply = reply_text[:3000]
        appeal.replied_at = now if reply_text else None
        if not status:
            appeal.status = 'replied' if reply_text else appeal.status
    if not appeal.admin_read_at:
        appeal.admin_read_at = now
    appeal.updated_at = now
    if block_user or unblock_user:
        target_user = User.query.get(appeal.user_id)
        if target_user:
            if unblock_user:
                target_user.appeal_blocked = False
                target_user.appeal_block_reason = None
                target_user.appeal_blocked_at = None
            else:
                target_user.appeal_blocked = True
                target_user.appeal_block_reason = block_reason or "異議申し立てはブロックされています。"
                target_user.appeal_blocked_at = now
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/bot-telemetry', methods=['POST'])
@login_required
def bot_telemetry():
    if getattr(current_user, 'is_admin', False):
        return jsonify({'status': 'ok', 'skipped': True})
    if not get_bot_detection_global_enabled() or not current_user.bot_detection_enabled:
        return jsonify({'status': 'disabled'})
    if current_user.is_bot_banned:
        return jsonify({'error': 'banned'}), 403
    data = request.json or {}
    if not verify_turnstile(data.get('turnstile_token')):
        return jsonify({'error': 'turnstile_failed'}), 403
    score, reasons = evaluate_bot_score(data)
    if score <= 0:
        return jsonify({'status': 'ok', 'score': 0})
    key = f"bot:score:{current_user.id}"
    try:
        new_score = redis_conn.incrbyfloat(key, float(score))
        redis_conn.expire(key, 300)
    except Exception:
        new_score = float(score)
    if new_score >= 8:
        current_user.is_bot_banned = True
        current_user.bot_banned_at = datetime.utcnow()
        current_user.bot_ban_reason = "Automated behavior detected (fast clicks/inputs)"
        ban_related_accounts(current_user, current_user.bot_ban_reason)
        return jsonify({'error': 'banned', 'score': new_score, 'reasons': reasons}), 403
    return jsonify({'status': 'ok', 'score': new_score, 'reasons': reasons})

@app.route('/api/bot/unban', methods=['POST'])
@login_required
def bot_unban():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': 'username_required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'not_found'}), 404
    mode = (data.get('mode') or 'single').strip().lower()
    if mode == 'linked':
        unban_linked_accounts(user)
    else:
        unban_single_account(user)
    return jsonify({'status': 'ok', 'username': username, 'mode': mode})

@app.route('/api/bot/users', methods=['GET'])
@login_required
def bot_users():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    q = (request.args.get('q') or '').strip()
    limit = request.args.get('limit') or '100'
    try:
        limit = max(1, min(500, int(limit)))
    except Exception:
        limit = 100
    query = User.query
    if q:
        query = query.filter(User.username.like(f"%{q}%"))
    users = query.order_by(User.id.desc()).limit(limit).all()
    res = []
    for u in users:
        res.append({
            'username': u.username,
            'bot_detection_enabled': u.bot_detection_enabled if u.bot_detection_enabled is not None else True,
            'is_bot_banned': bool(u.is_bot_banned),
            'bot_ban_reason': u.bot_ban_reason,
            'bot_banned_at': u.bot_banned_at.isoformat() + "Z" if u.bot_banned_at else None
        })
    return jsonify({'users': res})

@app.route('/api/bot/update', methods=['POST'])
@login_required
def bot_update():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    data = request.json or {}
    username = (data.get('username') or '').strip()
    action = (data.get('action') or '').strip()
    if not username or not action:
        return jsonify({'error': 'bad_request'}), 400
    if _is_primary_admin_username(username):
        return jsonify({'error': 'protected'}), 400
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'not_found'}), 404
    if action == 'toggle_detection':
        enabled = bool(data.get('enabled'))
        user.bot_detection_enabled = enabled
    elif action == 'ban':
        user.is_bot_banned = True
        user.bot_banned_at = datetime.utcnow()
        user.bot_ban_reason = data.get('reason') or "Manual ban"
        user.bot_unban_notice = False
        ban_related_accounts(user, user.bot_ban_reason)
    elif action == 'unban':
        unban_single_account(user)
    elif action == 'unban_linked':
        unban_linked_accounts(user)
    else:
        return jsonify({'error': 'bad_action'}), 400
    safe_db_commit()
    return jsonify({'status': 'ok', 'username': username, 'action': action})

def normalize_theme_color(value):
    if not value:
        return ""
    v = str(value).strip()
    if not v:
        return ""
    if not v.startswith('#'):
        v = f"#{v}"
    if len(v) == 4:
        v = f"#{v[1]}{v[1]}{v[2]}{v[2]}{v[3]}{v[3]}"
    if len(v) != 7:
        return ""
    if any(c not in "0123456789abcdefABCDEF" for c in v[1:]):
        return ""
    return v.lower()

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        status = redis_conn.get(f"migration_status:{current_user.id}")
        mig_status = status.decode() if status else "idle"
        prog = redis_conn.get(f"migration_progress:{current_user.id}")
        mig_progress = prog.decode() if prog else ""
        sp = current_user.system_prompt
        if current_user.enable_e2ee and sp: sp = decrypt_val(sp)
        
        # 2FA Status
        has_totp = bool(current_user.totp_secret)
        has_webauthn = bool(current_user.webauthn_credentials and json.loads(current_user.webauthn_credentials))
        
        return jsonify({
            'system_prompt': sp or "",
            'username': current_user.username, 
            'openai_key': decrypt_val(current_user.openai_api_key) or "", 
            'gemini_key': decrypt_val(current_user.gemini_api_key) or "", 
            'xai_key': decrypt_val(current_user.xai_api_key) or "",
            'google_key': decrypt_val(current_user.google_api_key) or "",
            'google_project': decrypt_val(current_user.google_cloud_project) or "",
            'stt_model': current_user.stt_model or "gpt-4o-mini-transcribe",
            'enter_to_send': current_user.enter_to_send,
            'use_sw_cache': current_user.use_sw_cache,
            'theme_color': current_user.theme_color or "",
            'auto_search_on_links': current_user.auto_search_on_links,
            'use_last_chat_settings': current_user.use_last_chat_settings,
            'default_enable_search': current_user.default_enable_search,
            'default_enable_python': current_user.default_enable_python,
            'default_enable_thinking': current_user.default_enable_thinking,
            'default_thinking_level': current_user.default_thinking_level or "high",
            'default_thinking_budget': current_user.default_thinking_budget if current_user.default_thinking_budget is not None else 4096,
            'default_reasoning_effort': current_user.default_reasoning_effort or "medium",
            'default_enable_system_prompt': current_user.default_enable_system_prompt,
            'default_safety_setting': current_user.default_safety_setting or "default",
            'last_enable_search': current_user.last_enable_search,
            'last_enable_python': current_user.last_enable_python,
            'last_enable_thinking': current_user.last_enable_thinking,
            'last_thinking_level': current_user.last_thinking_level or "high",
            'last_thinking_budget': current_user.last_thinking_budget if current_user.last_thinking_budget is not None else 4096,
            'last_reasoning_effort': current_user.last_reasoning_effort or "medium",
            'last_enable_system_prompt': current_user.last_enable_system_prompt,
            'last_safety_setting': current_user.last_safety_setting or "default",
            'enable_e2ee': current_user.enable_e2ee,
            'migration_status': mig_status,
            'migration_progress': mig_progress,
            'is_2fa_enabled': current_user.is_2fa_enabled,
            'has_totp': has_totp,
            'has_webauthn': has_webauthn,
            'passkey_only_login': current_user.passkey_only_login,
            'bot_detection_enabled': current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
            'bot_detection_global_enabled': get_bot_detection_global_enabled(),
            'is_bot_banned': current_user.is_bot_banned,
            'bot_ban_reason': current_user.bot_ban_reason
        })
    d = request.json
    if 'system_prompt' in d: 
        if current_user.enable_e2ee: current_user.system_prompt = encrypt_val(d['system_prompt'])
        else: current_user.system_prompt = d['system_prompt']
    if 'openai_key' in d: current_user.openai_api_key = encrypt_val(d['openai_key'])
    if 'gemini_key' in d: current_user.gemini_api_key = encrypt_val(d['gemini_key'])
    if 'xai_key' in d: current_user.xai_api_key = encrypt_val(d['xai_key'])
    if 'google_key' in d: current_user.google_api_key = encrypt_val(d['google_key'])
    if 'google_project' in d: current_user.google_cloud_project = encrypt_val(d['google_project'])
    if 'stt_model' in d: current_user.stt_model = d['stt_model']
    if 'enter_to_send' in d: current_user.enter_to_send = bool(d['enter_to_send'])
    if 'use_sw_cache' in d: current_user.use_sw_cache = bool(d['use_sw_cache'])
    if 'theme_color' in d: current_user.theme_color = normalize_theme_color(d.get('theme_color'))
    if 'auto_search_on_links' in d: current_user.auto_search_on_links = bool(d['auto_search_on_links'])
    if 'use_last_chat_settings' in d: current_user.use_last_chat_settings = bool(d['use_last_chat_settings'])
    if 'default_enable_search' in d: current_user.default_enable_search = bool(d['default_enable_search'])
    if 'default_enable_python' in d: current_user.default_enable_python = bool(d['default_enable_python'])
    if 'default_enable_thinking' in d: current_user.default_enable_thinking = bool(d['default_enable_thinking'])
    if 'default_thinking_level' in d: current_user.default_thinking_level = d['default_thinking_level'] or "high"
    if 'default_thinking_budget' in d:
        try:
            current_user.default_thinking_budget = int(d['default_thinking_budget'])
        except Exception:
            pass
    if 'default_reasoning_effort' in d: current_user.default_reasoning_effort = d['default_reasoning_effort'] or "medium"
    if 'default_enable_system_prompt' in d: current_user.default_enable_system_prompt = bool(d['default_enable_system_prompt'])
    if 'default_safety_setting' in d: current_user.default_safety_setting = d['default_safety_setting'] or "default"
    if 'passkey_only_login' in d:
        target = bool(d['passkey_only_login'])
        if target:
            creds = []
            if current_user.webauthn_credentials:
                try:
                    creds = json.loads(current_user.webauthn_credentials)
                except Exception:
                    creds = []
            if not creds:
                return jsonify({'error': 'No passkey registered'}), 400
        current_user.passkey_only_login = target
    if 'bot_detection_enabled' in d and d['bot_detection_enabled'] is not None:
        current_user.bot_detection_enabled = bool(d['bot_detection_enabled'])
    if getattr(current_user, 'is_admin', False) and 'bot_detection_global_enabled' in d:
        set_app_setting("bot_detection_global_enabled", "1" if d['bot_detection_global_enabled'] else "0")
    if d.get('new_password'): current_user.set_password(d['new_password'])
    if d.get('new_username') and d['new_username'] != current_user.username:
        if _is_primary_admin_username(d['new_username']) and not getattr(current_user, "is_admin", False):
            pass
        elif not User.query.filter_by(username=d['new_username']).first(): current_user.username = d['new_username']
    if 'enable_e2ee' in d and d['enable_e2ee'] != current_user.enable_e2ee:
        target_enable = d['enable_e2ee']
        task_queue.enqueue(migrate_e2ee_task, current_user.id, target_enable)
        flash("暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。")
    if 'disable_2fa' in d and d['disable_2fa']:
        current_user.is_2fa_enabled = False
        flash("2FAを無効化しました。")
    else:
        safe_db_commit()
        flash("設定を保存しました")
    return jsonify({'status': 'ok'})

# --- Session Management ---

@app.route('/api/sessions', methods=['GET'])
@login_required
def list_sessions():
    sid = session.get('session_id')
    rows = UserSession.query.filter_by(user_id=current_user.id).order_by(UserSession.last_seen_at.desc()).limit(50).all()
    return jsonify({
        'sessions': [
            {
                'id': s.id,
                'created_at': s.created_at.isoformat(),
                'last_seen_at': s.last_seen_at.isoformat() if s.last_seen_at else None,
                'ip_address': s.ip_address,
                'user_agent': s.user_agent,
                'is_current': s.session_id == sid,
                'is_revoked': s.is_revoked
            } for s in rows
        ]
    })

@app.route('/api/sessions/revoke', methods=['POST'])
@login_required
def revoke_session():
    data = request.json or {}
    sess_id = data.get('id')
    if not sess_id:
        return jsonify({'error': 'id_required'}), 400
    user_sess = UserSession.query.filter_by(id=sess_id, user_id=current_user.id).first()
    if not user_sess:
        return jsonify({'error': 'not_found'}), 404
    if not user_sess.is_revoked:
        user_sess.is_revoked = True
        user_sess.revoked_at = datetime.utcnow()
        safe_db_commit()
    logged_out = False
    if user_sess.session_id == session.get('session_id'):
        session.pop('session_id', None)
        logout_user()
        logged_out = True
    return jsonify({'status': 'ok', 'logged_out': logged_out})

@app.route('/api/sessions/revoke_others', methods=['POST'])
@login_required
def revoke_other_sessions():
    sid = session.get('session_id')
    revoke_user_sessions(current_user.id, exclude_session_id=sid)
    return jsonify({'status': 'ok'})

@app.route('/api/sessions/revoke_all', methods=['POST'])
@login_required
def revoke_all_sessions():
    revoke_user_sessions(current_user.id, exclude_session_id=None)
    session.pop('session_id', None)
    logout_user()
    return jsonify({'status': 'ok', 'logged_out': True})

# --- 2FA Settings Routes ---

@app.route('/api/2fa/totp/setup', methods=['POST'])
@login_required
def totp_setup():
    secret = pyotp.random_base32()
    # Save temporarily encrypted or just send back? Ideally verify first.
    # We will send back the secret and QR, but not enable it until verified.
    session['temp_totp_secret'] = secret
    
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.username, issuer_name="AI Chat Playground")
    img = qrcode.make(uri)
    buf = BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    return jsonify({'secret': secret, 'qr_image': f"data:image/png;base64,{b64}"})

@app.route('/api/2fa/totp/enable', methods=['POST'])
@login_required
def totp_enable():
    code = request.json.get('code')
    secret = session.get('temp_totp_secret')
    if not secret: return jsonify({'error': 'Setup session expired'}), 400
    
    if pyotp.TOTP(secret).verify(code):
        current_user.totp_secret = encrypt_val(secret)
        current_user.is_2fa_enabled = True
        session.pop('temp_totp_secret', None)
        safe_db_commit()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Invalid code'}), 400

@app.route('/api/2fa/webauthn/register/options', methods=['POST'])
@login_required
def webauthn_reg_options():
    options = generate_registration_options(
        rp_name="AI Chat Playground",
        rp_id=request.host.split(':')[0],
        user_id=str(current_user.id).encode(),
        user_name=current_user.username,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED
        )
    )
    session['webauthn_reg_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    return options_to_json(options)

@app.route('/api/2fa/webauthn/register/verify', methods=['POST'])
@login_required
def webauthn_reg_verify():
    try:
        data = request.json
        challenge = session.get('webauthn_reg_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        verification = verify_registration_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            require_user_verification=False 
        )
        
        creds = []
        if current_user.webauthn_credentials:
            try: creds = json.loads(current_user.webauthn_credentials)
            except: pass
            
        creds.append({
            'id': base64.b64encode(verification.credential_id).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('='),
            'public_key': base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('='),
            'sign_count': verification.sign_count,
            'name': data.get('name', 'Security Key')
        })
        
        current_user.webauthn_credentials = json.dumps(creds)
        current_user.is_2fa_enabled = True
        safe_db_commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"WebAuthn Reg Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/gems', methods=['GET', 'POST'])
@login_required
def handle_gems():
    if request.method == 'GET':
        gems = Gem.query.filter_by(user_id=current_user.id).order_by(Gem.created_at.desc()).all()
        return jsonify([{'id': g.id, 'name': g.name, 'description': g.description, 'instruction': g.instruction} for g in gems])
    d = request.json
    gem = Gem(user_id=current_user.id, name=d.get('name', 'My Gem'), description=d.get('description', ''), instruction=d.get('instruction', ''))
    db.session.add(gem)
    safe_db_commit()
    return jsonify({'id': gem.id, 'name': gem.name})

@app.route('/api/gems/<int:gid>', methods=['DELETE'])
@login_required
def delete_gem(gid):
    gem = Gem.query.get_or_404(gid)
    if gem.user_id != current_user.id: return jsonify({'error': '403'}), 403
    db.session.delete(gem)
    safe_db_commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/debug/log', methods=['GET'])
@login_required
def debug_log():
    if not getattr(current_user, "is_admin", False): return abort(403)
    def generate():
        process = subprocess.Popen(['sudo', 'journalctl', '-u', 'ai-chat', '-n', '50', '--no-pager'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        yield stdout
    return Response(generate(), mimetype='text/plain')

@app.route('/api/maintenance', methods=['POST'])
@login_required
def toggle_maintenance():
    if not getattr(current_user, "is_admin", False): return abort(403)
    lock_file = os.path.join(os.path.dirname(__file__), 'maintenance.lock')
    if request.json.get('enabled'):
        with open(lock_file, 'w') as f: f.write('locked')
        app.config['MAINTENANCE_MODE'] = True
    else:
        if os.path.exists(lock_file): os.remove(lock_file)
        app.config['MAINTENANCE_MODE'] = False
    return jsonify({'status': 'ok', 'mode': app.config['MAINTENANCE_MODE']})

@app.route('/synthesize', methods=['POST'])
@login_required
def synthesize():
    data = request.json
    text_content = data.get('text')
    voice_type = data.get('voice_type', 'neural') # studio, neural, standard
    language = data.get('language', 'ja-JP')
    
    if not text_content: return jsonify({'error': 'No text provided'}), 400
    
    try:
        g_key = decrypt_val(current_user.google_api_key)
        if not g_key and getattr(current_user, 'is_admin', False):
            g_key = os.getenv('GOOGLE_API_KEY')
        if not g_key:
            return jsonify({'error': 'Google API Key not configured (Google Cloud API key required)'}), 400
        
        g_project = decrypt_val(current_user.google_cloud_project)
        if not g_project and getattr(current_user, 'is_admin', False):
            g_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        opts = {"api_key": g_key}
        if g_project: opts["quota_project_id"] = g_project
        client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(**opts)
        )
        
        synthesis_input = texttospeech.SynthesisInput(text=text_content)
        
        # Voice selection
        if voice_type == 'studio':
            voice = pick_tts_voice(client, language, "studio")
        elif voice_type == 'neural':
            voice = pick_tts_voice(client, language, "neural")
        else:
            voice = pick_tts_voice(client, language, "standard")

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save audio file
        try:
            ok, used, limit = _check_storage_capacity(current_user, len(response.audio_content) if response.audio_content else 0)
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
        except Exception:
            pass
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
        
        fname = f"tts_{int(time.time())}_{os.urandom(4).hex()}.mp3"
        fpath = os.path.join(user_dir, fname)
        
        if current_user.enable_e2ee:
            with open(fpath + '.enc', 'wb') as f: f.write(encrypt_bytes(response.audio_content))
        else:
            with open(fpath, 'wb') as f: f.write(response.audio_content)
            
        return jsonify({'url': f"/files/{current_user.id}/{fname}", 'filename': f"{current_user.id}/{fname}"})
    except Exception as e:
        logger.error(f"TTS Synthesis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe():
    audio_content = None
    fname = None

    if request.files and 'file' in request.files:
        f = request.files['file']
        if not f or not f.filename:
            return jsonify({'error': 'No file'}), 400
        fname = secure_filename(f.filename)
        audio_content = f.read()
    else:
        data = request.json or {}
        filename = data.get('filename')
        if not filename: return jsonify({'error': 'No filename'}), 400
        
        # Path handling (same as /files route)
        parts = filename.split('/')
        if len(parts) != 2: return jsonify({'error': 'Invalid path'}), 400
        uid, fname = parts
        if uid != str(current_user.id): return jsonify({'error': 'Unauthorized'}), 403
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uid, fname)
        if current_user.enable_e2ee:
            if not os.path.exists(file_path + '.enc'):
                return jsonify({'error': 'File not found'}), 404
            with open(file_path + '.enc', 'rb') as f:
                audio_content = decrypt_bytes(f.read())
        else:
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            with open(file_path, 'rb') as f:
                audio_content = f.read()

    if not audio_content:
        return jsonify({'error': 'Empty audio'}), 400

    # OpenAI STT Implementation
    try:
        key = decrypt_val(current_user.openai_api_key)
        if not key and getattr(current_user, 'is_admin', False):
            key = os.getenv('OPENAI_API_KEY')
        if not key:
            return jsonify({'error': 'OpenAI API Key not configured'}), 400

        allowed_models = {
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "gpt-4o-transcribe-diarize",
            "whisper-1"
        }
        model = (current_user.stt_model or "").strip()
        if model not in allowed_models:
            model = "gpt-4o-mini-transcribe"

        client = _get_openai_client(key, base_url=None)
        audio_file = BytesIO(audio_content)
        audio_file.name = fname

        kwargs = {"model": model, "file": audio_file}
        if model == "gpt-4o-transcribe-diarize":
            kwargs["response_format"] = "diarized_json"
            kwargs["chunking_strategy"] = "auto"

        transcription = client.audio.transcriptions.create(**kwargs)

        transcript = ""
        segments = None
        if isinstance(transcription, dict):
            transcript = transcription.get("text") or ""
            segments = transcription.get("segments")
        else:
            transcript = getattr(transcription, "text", "") or ""
            segments = getattr(transcription, "segments", None)

        if model == "gpt-4o-transcribe-diarize" and segments:
            lines = []
            for seg in segments:
                if isinstance(seg, dict):
                    speaker = seg.get("speaker") or "Speaker"
                    text = seg.get("text") or ""
                else:
                    speaker = getattr(seg, "speaker", None) or "Speaker"
                    text = getattr(seg, "text", "") or ""
                if text:
                    lines.append(f"{speaker}: {text}")
            if lines:
                transcript = "\n".join(lines)

        return jsonify({'transcript': transcript})
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sts', methods=['POST'])
@login_required
def speech_to_speech():
    if not request.files or 'file' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'No file'}), 400

    model_key = (request.form.get('model') or "").strip()
    if not is_sts_model(model_key):
        return jsonify({'error': 'Invalid STS model'}), 400

    thread_id = request.form.get('thread_id')
    if not thread_id:
        return jsonify({'error': 'thread_id required'}), 400
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    thread_id = t.id

    audio_bytes = f.read()
    if not audio_bytes:
        return jsonify({'error': 'Empty audio'}), 400

    provider = get_sts_provider(model_key)
    meta = STS_MODELS.get(model_key, {})
    rate_in = meta.get("rate_in", 24000)
    rate_out = meta.get("rate_out", 24000)
    sts_voice = (request.form.get('sts_voice') or "").strip()
    sts_speed_raw = request.form.get('sts_speed')
    sts_rate_in_raw = request.form.get('sts_rate_in')
    sts_rate_out_raw = request.form.get('sts_rate_out')
    sts_speed = None

    if provider == "openai":
        v = sts_voice.lower() if sts_voice else "alloy"
        if v not in OPENAI_STS_VOICES:
            v = "alloy"
        sts_voice = v
        sts_speed = clamp_float(sts_speed_raw, 0.25, 1.5)
    elif provider == "xai":
        if sts_voice not in XAI_STS_VOICES:
            sts_voice = "Ara"
        try:
            ri = int(sts_rate_in_raw) if sts_rate_in_raw is not None and str(sts_rate_in_raw).strip() != "" else None
            ro = int(sts_rate_out_raw) if sts_rate_out_raw is not None and str(sts_rate_out_raw).strip() != "" else None
            if ri in XAI_PCM_RATES: rate_in = ri
            if ro in XAI_PCM_RATES: rate_out = ro
        except Exception:
            pass
    elif provider == "google":
        if sts_voice not in GEMINI_STS_VOICES:
            sts_voice = "Kore"

    try:
        src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
        pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)
    except Exception as e:
        logger.error(f"Audio convert failed: {e}")
        return jsonify({'error': 'Audio conversion failed'}), 500

    assistant_audio = b""
    assistant_text = ""
    input_text = ""
    try:
        if provider == "openai":
            key = decrypt_val(current_user.openai_api_key)
            if not key and getattr(current_user, 'is_admin', False):
                key = os.getenv('OPENAI_API_KEY')
            if not key:
                return jsonify({'error': 'OpenAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _openai_sts_realtime(pcm_bytes, key, model_key, voice=sts_voice, speed=sts_speed, rate=rate_out)
            )
        elif provider == "xai":
            key = decrypt_val(current_user.xai_api_key)
            if not key and getattr(current_user, 'is_admin', False):
                key = os.getenv('XAI_API_KEY')
            if not key:
                return jsonify({'error': 'xAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _xai_sts_realtime(pcm_bytes, key, model_key=model_key, voice=sts_voice, rate_in=rate_in, rate_out=rate_out)
            )
        elif provider == "google":
            key = decrypt_val(current_user.gemini_api_key)
            if not key and getattr(current_user, 'is_admin', False):
                key = os.getenv('GEMINI_API_KEY')
            if not key:
                return jsonify({'error': 'Gemini API Key not configured'}), 400
            assistant_audio, assistant_text, input_text = asyncio.run(
                _google_sts_live(pcm_bytes, key, model_key, rate=rate_in, voice=sts_voice)
            )
        else:
            return jsonify({'error': 'Unsupported provider'}), 400
    except Exception as e:
        logger.error(f"STS failed: {e}")
        return jsonify({'error': str(e)}), 500

    if not assistant_audio:
        return jsonify({'error': 'No audio response'}), 500

    wav_bytes = _pcm_to_wav_bytes(assistant_audio, rate=rate_out)
    try:
        incoming_size = len(wav_bytes) + (len(audio_bytes) if audio_bytes else 0)
        ok, used, limit = _check_storage_capacity(current_user, incoming_size)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
    except Exception:
        pass
    out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
    audio_url = f"/files/{current_user.id}/{out_fname}"

    in_fname = None
    try:
        in_suffix = src_ext if src_ext.startswith('.') else f".{src_ext}"
        in_fname, _ = _save_user_audio(current_user.id, audio_bytes, in_suffix, current_user.enable_e2ee)
    except Exception:
        in_fname = None

    parent_id = None
    try:
        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
        if last_msg:
            parent_id = last_msg.id
    except Exception:
        parent_id = None

    user_text = (input_text or "Voice message").strip()
    assistant_text_clean = (assistant_text or "").strip()
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
    assistant_content = (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_msg = Message(
            thread_id=thread_id,
            role='user',
            content=u_content,
            image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
            is_encrypted=current_user.enable_e2ee,
            parent_id=parent_id,
            tokens=count_tokens_for_display(user_text, model_key)
        )
        db.session.add(user_msg)
        safe_db_commit()

        assistant_msg = Message(
            thread_id=thread_id,
            role='assistant',
            content=a_content,
            model=model_key,
            is_encrypted=current_user.enable_e2ee,
            parent_id=user_msg.id,
            tokens=count_tokens_for_display(assistant_text_clean, model_key)
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as e:
        logger.error(f"STS message save failed: {e}")

    return jsonify({
        'audio_url': audio_url,
        'transcript': assistant_text_clean,
        'input_transcript': user_text,
        'filename': f"{current_user.id}/{out_fname}"
    })

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mov', '.mkv', '.avi', '.m4v'}
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    try:
        if not _is_primary_admin_user(current_user):
            hard_limit = _get_user_storage_limit_bytes(current_user)
            if hard_limit:
                for f in files:
                    size = _get_filestorage_size(f)
                    if size is not None and size > hard_limit:
                        limit_mb = _bytes_to_mb_str(hard_limit)
                        return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        total_incoming = 0
        for f in files:
            size = _get_filestorage_size(f)
            if size is None:
                continue
            total_incoming += size
        if not _is_primary_admin_user(current_user):
            ok, used, limit = _check_storage_capacity(current_user, total_incoming)
            if not ok:
                used_mb = _bytes_to_mb_str(used)
                limit_mb = _bytes_to_mb_str(limit)
                return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413
    except Exception:
        pass
    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass
    res = []
    for f in files:
        if f.filename:
            orig_name = secure_filename(f.filename)
            ext = os.path.splitext(orig_name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({'error': f'File type {ext} not allowed'}), 400
            
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            save_path = os.path.join(ud, fname)
            if current_user.enable_e2ee:
                is_image = ext in ['.jpg', '.jpeg', '.png']
                if is_image and not orig_name.endswith('.webp'):
                    try:
                        buf = BytesIO()
                        Image.open(f).convert('RGB').save(buf, 'WEBP', quality=80)
                        enc_data = encrypt_bytes(buf.getvalue())
                        fname = f"{fname_base}.webp"
                        with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(enc_data)
                    except:
                        f.seek(0)
                        with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(encrypt_bytes(f.read()))
                else:
                    with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(encrypt_bytes(f.read()))
            else:
                is_image = ext in ['.jpg', '.jpeg', '.png']
                if is_image and not orig_name.endswith('.webp'):
                    try:
                        Image.open(f).convert('RGB').save(os.path.join(ud, f"{fname_base}.webp"), 'WEBP', quality=80)
                        fname = f"{fname_base}.webp"
                    except:
                        f.seek(0)
                        f.save(save_path)
                else: f.save(save_path)
            res.append(f"{current_user.id}/{fname}")
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/upload/init', methods=['POST'])
@login_required
def upload_init():
    data = request.json or {}
    filename = secure_filename((data.get('filename') or '').strip())
    total_size = int(data.get('size') or 0)
    if not filename or total_size <= 0:
        return jsonify({'error': 'Invalid upload'}), 400

    allowed = {'.txt', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mov', '.mkv', '.avi', '.m4v'}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'File type {ext} not allowed'}), 400

    if not _is_primary_admin_user(current_user):
        hard_limit = _get_user_storage_limit_bytes(current_user)
        if hard_limit and total_size > hard_limit:
            limit_mb = _bytes_to_mb_str(hard_limit)
            return jsonify({'error': f'File too large. Max {limit_mb}'}), 413
        ok, used, limit = _check_storage_capacity(current_user, total_size)
        if not ok:
            used_mb = _bytes_to_mb_str(used)
            limit_mb = _bytes_to_mb_str(limit)
            return jsonify({'error': f'Storage limit exceeded ({used_mb} / {limit_mb})'}), 413

    upload_id = f"up_{int(time.time())}_{os.urandom(4).hex()}"
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    os.chmod(session_dir, 0o700)
    meta = {
        "filename": filename,
        "size": total_size,
        "received": 0,
        "created": int(time.time()),
        "ext": ext
    }
    if not _save_chunk_meta(os.path.join(session_dir, 'meta.json'), meta):
        return jsonify({'error': 'Init failed'}), 500
    chunk_size = 10 * 1024 * 1024
    return jsonify({'upload_id': upload_id, 'chunk_size': chunk_size})

@app.route('/upload/chunk', methods=['POST'])
@login_required
def upload_chunk():
    upload_id = (request.form.get('upload_id') or '').strip()
    index = request.form.get('index')
    total = request.form.get('total')
    f = request.files.get('chunk')
    if not upload_id or f is None:
        return jsonify({'error': 'Invalid chunk'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    meta_path = os.path.join(session_dir, 'meta.json')
    meta = _load_chunk_meta(meta_path)
    if not meta:
        return jsonify({'error': 'Upload not found'}), 404
    try:
        index = int(index) if index is not None else 0
        total = int(total) if total is not None else 0
    except Exception:
        return jsonify({'error': 'Invalid chunk index'}), 400

    part_path = os.path.join(session_dir, 'data.part')
    try:
        with open(part_path, 'ab') as out:
            chunk_data = f.read()
            out.write(chunk_data)
        meta['received'] = int(meta.get('received') or 0) + len(chunk_data)
        _save_chunk_meta(meta_path, meta)
    except Exception:
        return jsonify({'error': 'Chunk write failed'}), 500

    return jsonify({'received': meta['received'], 'total': meta.get('size', 0), 'index': index, 'chunks': total})

@app.route('/upload/complete', methods=['POST'])
@login_required
def upload_complete():
    data = request.json or {}
    upload_id = (data.get('upload_id') or '').strip()
    if not upload_id:
        return jsonify({'error': 'Invalid upload'}), 400
    session_dir = _chunk_session_dir(current_user.id, upload_id)
    meta_path = os.path.join(session_dir, 'meta.json')
    meta = _load_chunk_meta(meta_path)
    if not meta:
        return jsonify({'error': 'Upload not found'}), 404

    part_path = os.path.join(session_dir, 'data.part')
    if not os.path.exists(part_path):
        return jsonify({'error': 'Upload missing'}), 400
    if int(meta.get('received') or 0) != int(meta.get('size') or 0):
        return jsonify({'error': 'Upload incomplete'}), 400

    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o700)
    else:
        try: os.chmod(ud, 0o700)
        except: pass

    orig_name = meta.get('filename') or 'file'
    ext = os.path.splitext(orig_name)[1].lower()
    fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
    fname = f"{fname_base}{ext}"
    save_path = os.path.join(ud, fname)
    res = []
    try:
        if current_user.enable_e2ee:
            is_image = ext in ['.jpg', '.jpeg', '.png']
            if is_image and not orig_name.endswith('.webp'):
                try:
                    buf = BytesIO()
                    Image.open(part_path).convert('RGB').save(buf, 'WEBP', quality=80)
                    enc_data = encrypt_bytes(buf.getvalue())
                    fname = f"{fname_base}.webp"
                    with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(enc_data)
                except Exception:
                    with open(part_path, 'rb') as rf:
                        with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                            ef.write(encrypt_bytes(rf.read()))
            else:
                with open(part_path, 'rb') as rf:
                    with open(os.path.join(ud, fname + '.enc'), 'wb') as ef:
                        ef.write(encrypt_bytes(rf.read()))
        else:
            is_image = ext in ['.jpg', '.jpeg', '.png']
            if is_image and not orig_name.endswith('.webp'):
                try:
                    Image.open(part_path).convert('RGB').save(os.path.join(ud, f"{fname_base}.webp"), 'WEBP', quality=80)
                    fname = f"{fname_base}.webp"
                except Exception:
                    os.replace(part_path, save_path)
            else:
                os.replace(part_path, save_path)
        res.append(f"{current_user.id}/{fname}")
    except Exception as e:
        logger.error(f"Chunk finalize failed: {e}")
        return jsonify({'error': 'Finalize failed'}), 500
    finally:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except Exception:
            pass
        try:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            os.rmdir(session_dir)
        except Exception:
            pass
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/api/storage', methods=['GET'])
@login_required
def get_storage_usage():
    limit = _get_user_storage_limit_bytes(current_user)
    used = _get_user_storage_usage_bytes(current_user.id)
    if limit is None:
        limit = 0
    return jsonify({
        'used_bytes': used,
        'limit_bytes': limit,
        'used_mb': _bytes_to_mb_str(used),
        'limit_mb': _bytes_to_mb_str(limit) if limit else 'unlimited',
        'is_unlimited': limit == 0
    })

@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(e):
    limit = getattr(request, 'max_content_length', None) or app.config.get('MAX_CONTENT_LENGTH')
    if not limit:
        return jsonify({'error': 'File too large. The server rejected the upload.'}), 413
    limit_mb = limit // (1024 * 1024)
    return jsonify({'error': f'File too large. Max {limit_mb}MB'}), 413

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        try:
            logger.error(f"db.create_all failed: {e}")
        except Exception:
            pass
    try:
        ensure_thread_last_model_column()
    except Exception:
        pass
    try:
        ensure_app_setting("bot_detection_global_enabled", "1")
    except Exception:
        pass
    try:
        admin_user = None
        primary_admin = _get_primary_admin_username()
        if primary_admin:
            admin_user = User.query.filter_by(username=primary_admin).first()
        if admin_user and not getattr(admin_user, "is_admin", False):
            admin_user.is_admin = True
            safe_db_commit()
    except Exception:
        pass
    if RUN_SCHEMA_MIGRATIONS:
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN thought_signature TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_e2ee BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN is_encrypted BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN xai_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_2fa_enabled BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN totp_secret VARCHAR(255)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN webauthn_credentials TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN passkey_only_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enter_to_send BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_sw_cache BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN theme_color VARCHAR(16)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_search_on_links BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_last_chat_settings BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_cloud_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_hash TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_expires_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_detection_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_bot_banned BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_banned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_ban_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unbanned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unban_notice BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_block_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN admin_reply TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN replied_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN last_seen_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN ip_address VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_bookmarked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN bookmarked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN public_id VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN last_model VARCHAR(64)")
        except: pass

@app.errorhandler(403)
def handle_forbidden(_error):
    return render_template("403.html"), 403

@app.errorhandler(404)
def handle_not_found(_error):
    return render_template("404.html"), 404

if __name__ == '__main__':
    app.run(debug=True)
