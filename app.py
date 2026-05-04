import os
import sys
import json
import time
import logging
import gzip
import base64
import html
import mimetypes
import secrets
import re
import audioop
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
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote
import threading
import hashlib
from collections import OrderedDict
import httpx
from webauthn import (
    generate_registration_options, verify_registration_response,
    generate_authentication_options, verify_authentication_response
)
from webauthn.helpers import generate_challenge, base64url_to_bytes, options_to_json
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria, UserVerificationRequirement,
    PublicKeyCredentialCreationOptions, PublicKeyCredentialRequestOptions,
    PublicKeyCredentialDescriptor, AuthenticatorTransport, ResidentKeyRequirement
)
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from rq import Queue
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash, send_file, send_from_directory, abort, session, g
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, exc, text, func
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError, RateLimitError
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions
import websockets
import pypdf
from cryptography.fernet import Fernet
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from authlib.integrations.flask_client import OAuth

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
VERBOSE_DEBUG_LOGS = str(os.getenv("VERBOSE_DEBUG_LOGS", "0")).strip().lower() in ("1", "true", "yes", "on")

def log_force(msg):
    """Force log to file and journalctl"""
    try:
        if isinstance(msg, str) and msg.startswith("DEBUG:") and not VERBOSE_DEBUG_LOGS:
            return
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{t}] [AI-CHAT-DEBUG] {msg}"
        with open("/home/ai-chat-minashin1120/app/debug.log", "a") as f:
            f.write(log_msg + "\n")
        print(log_msg, file=sys.stdout, flush=True)
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

ENABLE_HTTP_GZIP = _env_bool("ENABLE_HTTP_GZIP", True)
HTTP_GZIP_MIN_BYTES = max(512, _env_int("HTTP_GZIP_MIN_BYTES", 1024))

def _coerce_bool_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return False
    return None

def _coerce_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None

_DECRYPT_TEXT_CACHE_MAX = max(0, _env_int("DECRYPT_TEXT_CACHE_MAX", 4096))
_MEDIA_BYTES_CACHE_MAX = max(0, _env_int("MEDIA_BYTES_CACHE_MAX_MB", 128)) * 1024 * 1024
_MEDIA_BYTES_CACHE_ITEM_MAX = max(0, _env_int("MEDIA_BYTES_CACHE_ITEM_MAX_MB", 12)) * 1024 * 1024
_HISTORY_IMAGE_MAX_ITEMS = max(0, _env_int("HISTORY_IMAGE_MAX_ITEMS", 0))
_HISTORY_IMAGE_MAX_BYTES = max(0, _env_int("HISTORY_IMAGE_MAX_MB", 0)) * 1024 * 1024
_THUMBNAIL_CACHE_MAX = max(0, _env_int("THUMBNAIL_CACHE_MAX_MB", 48)) * 1024 * 1024
_THUMBNAIL_CACHE_ITEM_MAX = max(0, _env_int("THUMBNAIL_CACHE_ITEM_MAX_MB", 2)) * 1024 * 1024
_THUMBNAIL_SIZE = max(64, _env_int("THUMBNAIL_SIZE", 320))
_THUMBNAIL_QUALITY = min(95, max(50, _env_int("THUMBNAIL_QUALITY", 78)))

def _key_sig(key, extra=""):
    if not key:
        return None
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h}:{extra}" if extra else h

def _normalize_gemini_backend(value):
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in ("vertex_ai", "vertex", "vertexai"):
        return "vertex_ai"
    return "gemini_api"

def _is_deepseek_model_key(model_key):
    mk = str(model_key or "").lower()
    return "deepseek" in mk

def _normalize_admin_api_key_mode(value):
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in ("user_only", "user", "settings", "user_settings"):
        return "user_only"
    return "env_fallback"

def _admin_env_fallback_enabled(user):
    if not user or not getattr(user, "is_admin", False):
        return False
    return _normalize_admin_api_key_mode(getattr(user, "admin_api_key_mode", None)) == "env_fallback"

def _normalize_gemini_vertex_location(value):
    v = str(value or "").strip()
    return v or "global"

def _normalize_gemini_vertex_credentials_json(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        info = json.loads(raw)
    except Exception:
        raise ValueError("Vertex AI サービスアカウント JSON の形式が不正です。")
    if not isinstance(info, dict):
        raise ValueError("Vertex AI サービスアカウント JSON はオブジェクト形式で入力してください。")
    if str(info.get("type") or "").strip() != "service_account":
        raise ValueError("Vertex AI サービスアカウント JSON の type は service_account である必要があります。")
    if not str(info.get("client_email") or "").strip() or not str(info.get("private_key") or "").strip():
        raise ValueError("Vertex AI サービスアカウント JSON に client_email / private_key がありません。")
    try:
        return json.dumps(info, sort_keys=True, separators=(",", ":"))
    except Exception:
        raise ValueError("Vertex AI サービスアカウント JSON の正規化に失敗しました。")

def _load_gemini_vertex_credentials(vertex_credentials_json):
    normalized = _normalize_gemini_vertex_credentials_json(vertex_credentials_json)
    if not normalized:
        return None, None
    try:
        info = json.loads(normalized)
        creds = service_account.Credentials.from_service_account_info(info)
    except Exception:
        raise ValueError("Vertex AI サービスアカウント JSON から認証情報を読み込めませんでした。")
    return creds, _key_sig(normalized, "gemini_vertex_sa")

def _is_missing_google_adc_error(err):
    msg = str(err or "")
    if not msg:
        return False
    lower_msg = msg.lower()
    if "application default credentials" in lower_msg and ("not found" in lower_msg or "not available" in lower_msg):
        return True
    if "default credentials were not found" in lower_msg:
        return True
    if "set up application default credentials" in lower_msg:
        return True
    if "google_application_credentials" in lower_msg and "credential" in lower_msg:
        return True
    return False

def _gemini_vertex_auth_error_message():
    return (
        "Vertex AI の認証情報 (Application Default Credentials) が見つかりません。"
        "設定画面で Vertex AI サービスアカウント JSON を入力するか、"
        "サーバーで gcloud auth application-default login を実行するか、"
        "GOOGLE_APPLICATION_CREDENTIALS を設定してください。"
        "Gemini API を使う場合は設定画面で Gemini Backend を Gemini API に変更してください。"
    )

def _format_gemini_runtime_error(err, backend="gemini_api"):
    if _normalize_gemini_backend(backend) == "vertex_ai" and _is_missing_google_adc_error(err):
        return _gemini_vertex_auth_error_message()
    return str(err)

def _resolve_gemini_runtime(user):
    backend_raw = getattr(user, "gemini_backend", None) if user else None
    backend = _normalize_gemini_backend(backend_raw)
    api_key = decrypt_val(getattr(user, "gemini_api_key", None)) if user else None
    vertex_project = decrypt_val(getattr(user, "gemini_vertex_project", None)) if user else None
    vertex_location_raw = getattr(user, "gemini_vertex_location", None) if user else None
    vertex_credentials_json = decrypt_val(getattr(user, "gemini_vertex_credentials_json", None)) if user else None
    vertex_location = _normalize_gemini_vertex_location(vertex_location_raw)
    if _admin_env_fallback_enabled(user):
        env_backend = os.getenv("GEMINI_BACKEND")
        if env_backend and (not backend_raw or not str(backend_raw).strip()):
            backend = _normalize_gemini_backend(env_backend)
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not vertex_project:
            vertex_project = (
                os.getenv("GEMINI_VERTEX_PROJECT")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or ""
            ).strip() or None
        if not vertex_location_raw or not str(vertex_location_raw).strip():
            env_loc = os.getenv("GEMINI_VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION")
            if env_loc:
                vertex_location = _normalize_gemini_vertex_location(env_loc)
        if not vertex_credentials_json:
            env_vertex_json = os.getenv("GEMINI_VERTEX_SERVICE_ACCOUNT_JSON")
            if env_vertex_json and str(env_vertex_json).strip():
                vertex_credentials_json = str(env_vertex_json).strip()
    if vertex_project and str(vertex_project).strip():
        vertex_project = str(vertex_project).strip()
    else:
        vertex_project = None
    if api_key and not str(api_key).strip():
        api_key = None
    if vertex_credentials_json and not str(vertex_credentials_json).strip():
        vertex_credentials_json = None
    return {
        "backend": backend,
        "api_key": api_key,
        "vertex_project": vertex_project,
        "vertex_location": vertex_location,
        "vertex_credentials_json": vertex_credentials_json,
    }

_MODEL_API_KEY_MAX_ENTRIES = 256
_MODEL_API_KEY_MODEL_MAX_LEN = 128
_MODEL_API_KEY_VALUE_MAX_LEN = 1024

def _normalize_model_api_key_map(raw):
    if raw is None:
        return {}
    parsed = raw
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return {}
        try:
            parsed = json.loads(txt)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for k, v in parsed.items():
        mk = str(k or "").strip()
        if not mk:
            continue
        if len(mk) > _MODEL_API_KEY_MODEL_MAX_LEN:
            mk = mk[:_MODEL_API_KEY_MODEL_MAX_LEN]
        mv = str(v or "").strip()
        if not mv:
            continue
        if len(mv) > _MODEL_API_KEY_VALUE_MAX_LEN:
            mv = mv[:_MODEL_API_KEY_VALUE_MAX_LEN]
        out[mk] = mv
        if len(out) >= _MODEL_API_KEY_MAX_ENTRIES:
            break
    return out

def _load_user_model_api_key_map(user):
    if not user:
        return {}
    raw = decrypt_val(getattr(user, "model_api_keys", None))
    return _normalize_model_api_key_map(raw)

def _save_user_model_api_key_map(user, raw):
    if not user:
        return {}
    normalized = _normalize_model_api_key_map(raw)
    if not normalized:
        user.model_api_keys = None
        return {}
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    user.model_api_keys = encrypt_val(payload)
    return normalized

def _get_model_specific_api_key(user, model_key):
    mk = str(model_key or "").strip()
    if not user or not mk:
        return None
    key_map = _load_user_model_api_key_map(user)
    hit = key_map.get(mk)
    if hit and str(hit).strip():
        return str(hit).strip()
    mk_l = mk.lower()
    for k, v in key_map.items():
        if str(k or "").strip().lower() == mk_l:
            val = str(v or "").strip()
            return val or None
    return None

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
_GEMINI_HTTPX_CLIENT = httpx.Client(http2=False, limits=_HTTPX_LIMITS)

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

def _get_gemini_client(
    api_key=None,
    backend="gemini_api",
    vertex_project=None,
    vertex_location=None,
    vertex_credentials_json=None,
):
    backend = _normalize_gemini_backend(backend)
    vertex_project = (vertex_project or "").strip() if vertex_project else ""
    vertex_location = _normalize_gemini_vertex_location(vertex_location)
    vertex_creds = None
    vertex_creds_sig = "adc"
    if backend == "vertex_ai":
        if not vertex_project:
            return None
        if vertex_credentials_json and str(vertex_credentials_json).strip():
            vertex_creds, vertex_creds_sig = _load_gemini_vertex_credentials(vertex_credentials_json)
        sig = f"vertex:{vertex_project}:{vertex_location}:{vertex_creds_sig}"
    else:
        sig = _key_sig(api_key, "gemini_api")
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
        if backend == "vertex_ai":
            kwargs = {
                "vertexai": True,
                "project": vertex_project,
                "location": vertex_location,
                "http_options": http_options,
            }
            if vertex_creds is not None:
                kwargs["credentials"] = vertex_creds
            client = genai.Client(**kwargs)
        else:
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
app.config['APP_VERSION'] = os.getenv('APP_VERSION', '2026-05-04-006')
app.config['SYSTEM_VERSION'] = 'V4.8.484'
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
try:
    _attachment_max_files = int(os.getenv('ATTACHMENT_MAX_FILES', '30') or '30')
except Exception:
    _attachment_max_files = 30
app.config['ATTACHMENT_MAX_FILES'] = max(1, _attachment_max_files)
try:
    _upload_concurrency = int(os.getenv('UPLOAD_CONCURRENCY', '3') or '3')
except Exception:
    _upload_concurrency = 3
app.config['UPLOAD_CONCURRENCY'] = max(1, min(8, _upload_concurrency))
_user_storage_limit_mb = int(os.getenv('USER_STORAGE_LIMIT_MB', '100') or '100')
app.config['USER_STORAGE_LIMIT_MB'] = _user_storage_limit_mb
_primary_admin_username = (os.getenv('PRIMARY_ADMIN_USERNAME') or '').strip()
app.config['PRIMARY_ADMIN_USERNAME'] = _primary_admin_username or None
app.config['MAINTENANCE_MODE'] = os.path.exists(os.path.join(os.path.dirname(__file__), 'maintenance.lock'))

oauth = OAuth(app)
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)
_CHAT_FAST_QUEUE_NAME = os.getenv("AI_CHAT_FAST_QUEUE", "ai_chat_fast_queue")
_CHAT_HEAVY_QUEUE_NAME = os.getenv("AI_CHAT_HEAVY_QUEUE", "ai_chat_heavy_queue")
chat_fast_queue = Queue(_CHAT_FAST_QUEUE_NAME, connection=redis_conn)
chat_heavy_queue = Queue(_CHAT_HEAVY_QUEUE_NAME, connection=redis_conn)
_LATENCY_TRACE_PREFIX = "latency_trace:"
_LATENCY_TRACE_TTL_SECONDS = max(300, _env_int("LATENCY_TRACE_TTL_SECONDS", 86400))
_DIRECT_FIRST_TURN_ENABLED = _env_bool("CHAT_STREAM_DIRECT_FIRST_TURN", True)

_TEMP_CHAT_TIMEOUT_MIN_SECONDS = max(10, _env_int("TEMP_CHAT_TIMEOUT_MIN_SECONDS", 30))
_TEMP_CHAT_TIMEOUT_MAX_SECONDS = max(_TEMP_CHAT_TIMEOUT_MIN_SECONDS, _env_int("TEMP_CHAT_TIMEOUT_MAX_SECONDS", 3600))
_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = _env_int("TEMP_CHAT_STALE_SECONDS", 90)
if _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS < _TEMP_CHAT_TIMEOUT_MIN_SECONDS:
    _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = _TEMP_CHAT_TIMEOUT_MIN_SECONDS
if _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS > _TEMP_CHAT_TIMEOUT_MAX_SECONDS:
    _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = _TEMP_CHAT_TIMEOUT_MAX_SECONDS
_TEMP_CHAT_MONITOR_INTERVAL = max(5, _env_int("TEMP_CHAT_MONITOR_INTERVAL_SECONDS", 15))
_TEMP_CHAT_TRACK_TTL_SECONDS = max(_TEMP_CHAT_TIMEOUT_MAX_SECONDS * 20, 3600)
_TEMP_CHAT_LAST_SEEN_ZSET = "temp_chat:last_seen"
_TEMP_CHAT_STATE_PREFIX = "temp_chat:state:"
_TEMP_CHAT_UPLOADS_PREFIX = "temp_chat:uploads:"
_TEMP_CHAT_MONITOR_LEADER_KEY = "temp_chat:monitor:leader"
_TEMP_CHAT_MONITOR_LEASE_SECONDS = max(15, _TEMP_CHAT_MONITOR_INTERVAL * 3)

_TEMP_CHAT_MONITOR_LOCK = threading.Lock()
_TEMP_CHAT_MONITOR_THREAD = None
_TEMP_CHAT_MONITOR_PID = None
_TEMP_CHAT_MONITOR_TOKEN = None

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
    if ext == ".txt":
        return "text/plain"
    return mime_guess or "application/octet-stream"

def _extract_text_from_docx(data):
    try:
        from io import BytesIO
        with zipfile.ZipFile(BytesIO(data)) as zf:
            xml_content = zf.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = []
        for paragraph in tree.findall('.//w:p', namespace):
            texts = [node.text for node in paragraph.findall('.//w:t', namespace) if node.text]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs)
    except Exception:
        return None

def _get_file_disk_info(rel_path):
    if not rel_path:
        return {"exists": False}
    base = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
    try:
        if not os.path.realpath(base).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])):
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
    is_text = (mime or '').startswith('text/') or clean_fn.lower().endswith('.txt')

    extracted = None
    if is_pdf:
        try:
            reader = pypdf.PdfReader(BytesIO(data))
            extracted = "".join([(p.extract_text() or "") + "\n" for p in reader.pages])
        except Exception:
            extracted = None
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

_DECRYPT_CACHE_LOCK = threading.Lock()
_DECRYPT_CACHE = OrderedDict()
_DECRYPT_CACHE_MISS = object()

def encrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.encrypt(val.encode()).decode()
    except: return val

def decrypt_val(val):
    if not val or not cipher: return val
    if _DECRYPT_TEXT_CACHE_MAX > 0 and isinstance(val, str):
        with _DECRYPT_CACHE_LOCK:
            hit = _DECRYPT_CACHE.get(val, _DECRYPT_CACHE_MISS)
            if hit is not _DECRYPT_CACHE_MISS:
                _DECRYPT_CACHE.move_to_end(val)
                return hit
    try:
        plain = cipher.decrypt(val.encode()).decode()
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
    "gpt-realtime-1.5": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gpt-realtime-mini": {"provider": "openai", "rate_in": 24000, "rate_out": 24000},
    "gemini-2.5-flash-native-audio-preview-12-2025": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
    "gemini-3.1-flash-live-preview": {"provider": "google", "rate_in": 16000, "rate_out": 24000},
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

def is_gemini_model_key(model_key):
    mk = str(model_key or "").lower()
    return "gemini" in mk

def is_deepseek_model_key(model_key):
    return _is_deepseek_model_key(model_key)

def is_gemini_image_model_key(model_key):
    mk = str(model_key or "").lower()
    return "gemini" in mk and any(x in mk for x in ("image", "nano"))

def _chunk_bytes(data, chunk_size=32000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def _convert_audio_to_pcm(audio_bytes, src_suffix=".webm", rate=24000):
    cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
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

MIC_TRANSCRIBE_MODES = {"stt_api", "llm"}
DEFAULT_LLM_TRANSCRIBE_PROMPT = (
    "この音声を正確に文字起こししてください。"
    "出力は文字起こし本文のみ。説明・要約・補足は不要です。"
)
LLM_TRANSCRIBE_PROMPT_MAX_CHARS = 4000

def _normalize_mic_transcribe_mode(value):
    v = str(value or "").strip().lower()
    return v if v in MIC_TRANSCRIBE_MODES else "stt_api"

def _normalize_llm_transcribe_prompt(raw_text):
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    if len(text) > LLM_TRANSCRIBE_PROMPT_MAX_CHARS:
        text = text[:LLM_TRANSCRIBE_PROMPT_MAX_CHARS]
    return text

def get_user_llm_transcribe_prompt(user):
    try:
        raw = getattr(user, "llm_transcribe_prompt", None)
    except Exception:
        raw = None
    return _normalize_llm_transcribe_prompt(raw) or DEFAULT_LLM_TRANSCRIBE_PROMPT

def _extract_openai_response_text(resp):
    try:
        if isinstance(resp, dict):
            raw = resp.get("output_text")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        else:
            raw = getattr(resp, "output_text", None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
        texts = []
        for item in output or []:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            for part in content or []:
                if isinstance(part, dict):
                    p_type = part.get("type")
                    if p_type in ("output_text", "text") and part.get("text"):
                        texts.append(str(part.get("text")))
                else:
                    p_type = getattr(part, "type", None)
                    p_text = getattr(part, "text", None)
                    if p_type in ("output_text", "text") and p_text:
                        texts.append(str(p_text))
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        return ""
    return ""

def _transcribe_audio_with_llm(audio_content, fname, llm_model_key, user):
    no_speech_token = "[[NO_SPEECH]]"
    base_transcription_prompt = get_user_llm_transcribe_prompt(user)
    transcription_prompt = (
        f"{base_transcription_prompt}\n"
        f"聞き取れない、無音、音声が極端に小さい場合は推測せず {no_speech_token} のみを返してください。"
    )
    model_key = (llm_model_key or "").strip()
    model_key_l = model_key.lower()
    is_gem = is_gemini_model_key(model_key_l)
    is_deepseek = is_deepseek_model_key(model_key_l)
    is_grok = ("grok" in model_key_l) and ("gpt" not in model_key_l)
    if is_grok:
        raise ValueError("現在の xAI/Grok モデルのLLM文字起こしは未対応です。OpenAI/Gemini対応モデルに切り替えるか、STT APIを使用してください。")
    if is_deepseek:
        raise ValueError("DeepSeek モデルのLLM文字起こしは未対応です。OpenAI/Gemini対応モデルに切り替えるか、STT APIを使用してください。")
    if not model_key:
        model_key = "gpt-4o-mini"
        model_key_l = model_key.lower()

    src_ext = os.path.splitext(fname or "")[1].lower() or ".webm"
    if src_ext not in (".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"):
        src_ext = ".webm"
    try:
        target_rate = 16000 if is_gem else 24000
        pcm = _convert_audio_to_pcm(audio_content, src_suffix=src_ext, rate=target_rate)
        wav_bytes = _pcm_to_wav_bytes(pcm, rate=target_rate)
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed (ffmpeg): {e}") from e

    metrics = _pcm_audio_metrics_mono_s16le(pcm, rate=target_rate)
    # Guard against near-silent capture in LLM mode to prevent hallucinated transcripts.
    # Relaxed thresholds to allow quieter valid inputs.
    if metrics["duration_sec"] >= 0.35 and metrics["rms"] < 30 and metrics["peak"] < 250:
        logger.warning(
            "LLM transcription rejected due to near-silent audio "
            f"(model={model_key}, dur={metrics['duration_sec']:.2f}s, rms={metrics['rms']}, peak={metrics['peak']})"
        )
        raise ValueError("録音音声が極端に小さい/無音です。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")

    if is_gem:
        gemini_runtime = _resolve_gemini_runtime(user)
        g_key = _get_model_specific_api_key(user, model_key) or gemini_runtime.get("api_key")
        backend = gemini_runtime.get("backend")
        if backend == "vertex_ai":
            if not gemini_runtime.get("vertex_project"):
                raise ValueError("Vertex AI Project ID が未設定です。Gemini設定を確認してください。")
        elif not g_key:
            raise ValueError("Gemini API Key not configured")
        g_client = _get_gemini_client(
            api_key=g_key,
            backend=backend,
            vertex_project=gemini_runtime.get("vertex_project"),
            vertex_location=gemini_runtime.get("vertex_location"),
            vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
        )
        if not g_client:
            raise RuntimeError("Gemini client initialization failed")
        resp = g_client.models.generate_content(
            model=model_key,
            contents=[
                types.Part(text=transcription_prompt),
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
        text_out = (getattr(resp, "text", None) or "").strip()
        if _llm_transcript_is_no_speech(text_out, no_speech_token):
            raise ValueError("音声を検出できませんでした。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")
        return text_out

    o_key = _get_model_specific_api_key(user, model_key) or decrypt_val(user.openai_api_key)
    if not o_key and _admin_env_fallback_enabled(user):
        o_key = os.getenv('OPENAI_API_KEY')
    if not o_key:
        raise ValueError("OpenAI API Key not configured")
    client = _get_openai_client(o_key, base_url=None)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    resp = client.responses.create(
        model=model_key,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": transcription_prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ]
            }
        ]
    )
    text_out = _extract_openai_response_text(resp).strip()
    if _llm_transcript_is_no_speech(text_out, no_speech_token):
        raise ValueError("音声を検出できませんでした。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")
    return text_out

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

async def _google_sts_live(
    pcm_bytes,
    model_key,
    gemini_api_key=None,
    gemini_backend="gemini_api",
    gemini_vertex_project=None,
    gemini_vertex_location=None,
    gemini_vertex_credentials_json=None,
    rate=16000,
    voice="Kore",
    thinking_level=None,
    include_thoughts=False,
):
    client = _get_gemini_client(
        api_key=gemini_api_key,
        backend=gemini_backend,
        vertex_project=gemini_vertex_project,
        vertex_location=gemini_vertex_location,
        vertex_credentials_json=gemini_vertex_credentials_json,
    )
    if not client:
        raise ValueError("Gemini client not configured")

    live_conf = {"response_modalities": ["AUDIO"]}
    if voice and voice in GEMINI_STS_VOICES:
        live_conf["speech_config"] = {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": voice}
            }
        }
    if thinking_level:
        live_conf["thinking_config"] = {
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts
        }

    async with client.aio.live.connect(
        model=model_key,
        config=live_conf,
    ) as session:
        # Send audio in small chunks to the Live API
        for chunk in _chunk_bytes(pcm_bytes, 4096):
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={rate}")
            )
        await session.send_realtime_input(audio_stream_end=True)
        
        total_audio_len = 0
        async for msg in session.receive():
            if total_audio_len > 10 * 1024 * 1024:
                break

            chunk_audio = bytearray()
            chunk_transcript = ""
            chunk_thought = ""
            chunk_input_transcript = ""
            turn_complete = False

            sc = getattr(msg, "server_content", None)
            if sc:
                model_turn = getattr(sc, "model_turn", None)
                if model_turn:
                    for part in model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            chunk_audio.extend(part.inline_data.data)
                        if part.text:
                            if getattr(part, "thought", False):
                                chunk_thought += part.text
                            else:
                                chunk_transcript += part.text

                if getattr(sc, "output_transcription", None) and sc.output_transcription.text:
                    chunk_transcript += sc.output_transcription.text
                if getattr(sc, "input_transcription", None) and sc.input_transcription.text:
                    chunk_input_transcript = sc.input_transcription.text
                
                if sc.turn_complete:
                    turn_complete = True
            elif msg.data:
                chunk_audio.extend(msg.data)

            if chunk_audio:
                total_audio_len += len(chunk_audio)
            
            if chunk_audio or chunk_transcript or chunk_thought or chunk_input_transcript or turn_complete:
                yield bytes(chunk_audio), chunk_transcript, chunk_input_transcript, chunk_thought, turn_complete
                if turn_complete:
                    break

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    google_id = db.Column(db.String(128), unique=True, nullable=True, index=True)
    google_email = db.Column(db.String(128), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    admin_api_key_mode = db.Column(db.String(24), default="env_fallback")
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    system_prompt_enabled = db.Column(db.Boolean, default=True)
    apply_global_system_prompt = db.Column(db.Boolean, default=True)
    apply_auto_system_prompt_notices = db.Column(db.Boolean, default=True)
    auto_system_prompt_notices_config = db.Column(db.Text, nullable=True)
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    deepseek_api_key = db.Column(db.Text, nullable=True)
    model_api_keys = db.Column(db.Text, nullable=True)
    gemini_backend = db.Column(db.String(24), default="gemini_api")
    gemini_vertex_project = db.Column(db.Text, nullable=True)
    gemini_vertex_location = db.Column(db.String(64), default="global")
    gemini_vertex_credentials_json = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)
    google_api_key = db.Column(db.Text, nullable=True)
    google_cloud_project = db.Column(db.Text, nullable=True)
    mic_transcribe_mode = db.Column(db.String(16), default="stt_api")
    stt_model = db.Column(db.String(64), default="gpt-4o-mini-transcribe")
    llm_transcribe_prompt = db.Column(db.Text, nullable=True)
    enter_to_send = db.Column(db.Boolean, default=False)
    use_sw_cache = db.Column(db.Boolean, default=False)
    theme_color = db.Column(db.String(16), default="")
    auto_search_on_links = db.Column(db.Boolean, default=True)
    compact_prompt_mode = db.Column(db.Boolean, default=False)
    use_last_chat_settings = db.Column(db.Boolean, default=False)
    temp_chat_timeout_seconds = db.Column(db.Integer, default=_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS)
    default_model = db.Column(db.String(64), default="gemini-3.1-flash-lite-preview")
    default_enable_search = db.Column(db.Boolean, default=False)
    default_enable_url_context = db.Column(db.Boolean, default=False)
    default_enable_maps = db.Column(db.Boolean, default=False)
    default_enable_python = db.Column(db.Boolean, default=True)
    default_enable_thinking = db.Column(db.Boolean, default=False)
    default_thinking_level = db.Column(db.String(16), default="high")
    default_thinking_budget = db.Column(db.Integer, default=4096)
    default_reasoning_effort = db.Column(db.String(16), default="medium")
    default_enable_system_prompt = db.Column(db.Boolean, default=False)
    default_safety_setting = db.Column(db.String(16), default="default")
    rich_paste_prompt_default = db.Column(db.Text, nullable=True)
    rich_paste_prompt_use_custom_default = db.Column(db.Boolean, default=False)
    last_model = db.Column(db.String(64), nullable=True)
    last_enable_search = db.Column(db.Boolean, default=False)
    last_enable_url_context = db.Column(db.Boolean, default=False)
    last_enable_maps = db.Column(db.Boolean, default=False)
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
    skip_2fa_on_google_login = db.Column(db.Boolean, default=False)
    default_2fa_method = db.Column(db.String(16), default='totp')
    bot_detection_enabled = db.Column(db.Boolean, default=True)
    is_bot_banned = db.Column(db.Boolean, default=False)
    bot_banned_at = db.Column(db.DateTime, nullable=True)
    bot_ban_reason = db.Column(db.Text, nullable=True)
    bot_unbanned_at = db.Column(db.DateTime, nullable=True)
    bot_unban_notice = db.Column(db.Boolean, default=False)
    appeal_blocked = db.Column(db.Boolean, default=False)
    appeal_block_reason = db.Column(db.Text, nullable=True)
    appeal_blocked_at = db.Column(db.DateTime, nullable=True)
    enable_latency_metrics = db.Column(db.Boolean, default=False)
    enable_client_debug_log = db.Column(db.Boolean, default=False)
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
    is_temporary = db.Column(db.Boolean, default=False)
    custom_instruction = db.Column(db.Text, nullable=True)
    include_global_instruction = db.Column(db.Boolean, default=True)
    last_model = db.Column(db.String(64), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='thread', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('thread.id'), nullable=False, index=True)
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    model = db.Column(db.String(50))
    image_url = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    tokens_in = db.Column(db.Integer, default=0)
    tokens_out = db.Column(db.Integer, default=0)
    tokens_thought = db.Column(db.Integer, default=0)
    thought_data = db.Column(db.Text)
    quote_text = db.Column(db.Text)
    is_encrypted = db.Column(db.Boolean, default=False)
    thought_signature = db.Column(db.Text, nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=True)
    children = db.relationship('Message', backref=db.backref('parent', remote_side=[id]), lazy=True)

class FileCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    rel_path = db.Column(db.Text, nullable=False, index=True)
    provider = db.Column(db.String(32), nullable=False, index=True)
    size_bytes = db.Column(db.Integer, nullable=True)
    mtime = db.Column(db.Integer, nullable=True)
    mime_type = db.Column(db.String(128), nullable=True)
    file_id = db.Column(db.String(256), nullable=True)
    file_uri = db.Column(db.Text, nullable=True)
    state = db.Column(db.String(32), default="unknown")
    last_error = db.Column(db.Text, nullable=True)
    retries = db.Column(db.Integer, default=0)
    last_checked_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Gem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    instruction = db.Column(db.Text, nullable=False)
    fixed_prompts_json = db.Column(db.Text, nullable=True)
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

class FirstTokenLatencyMetric(db.Model):
    __tablename__ = 'first_token_latency_metric'
    __table_args__ = (
        db.Index('idx_ft_latency_user_created', 'user_id', 'created_at'),
        db.Index('idx_ft_latency_user_event_created', 'user_id', 'first_event_type', 'created_at'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    thread_public_id = db.Column(db.String(64), nullable=True, index=True)
    job_id = db.Column(db.String(64), nullable=True, index=True)
    model = db.Column(db.String(80), nullable=True)
    first_event_type = db.Column(db.String(32), nullable=True)
    latency_seconds = db.Column(db.Float, nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False)
    client_sent_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class ChatLatencyTrace(db.Model):
    __tablename__ = 'chat_latency_trace'
    __table_args__ = (
        db.UniqueConstraint('job_id', name='uq_chat_latency_trace_job_id'),
        db.Index('idx_chat_latency_trace_user_created', 'user_id', 'created_at'),
        db.Index('idx_chat_latency_trace_thread_created', 'thread_public_id', 'created_at'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    thread_public_id = db.Column(db.String(64), nullable=True, index=True)
    job_id = db.Column(db.String(64), nullable=False, index=True)
    model = db.Column(db.String(80), nullable=True)
    execution_path = db.Column(db.String(24), nullable=True)
    client_sent_at = db.Column(db.DateTime, nullable=True)
    client_first_event_type = db.Column(db.String(32), nullable=True)
    client_first_latency_ms = db.Column(db.Integer, nullable=True)
    route_received_at = db.Column(db.DateTime, nullable=True)
    route_dispatch_at = db.Column(db.DateTime, nullable=True)
    route_stream_open_at = db.Column(db.DateTime, nullable=True)
    worker_started_at = db.Column(db.DateTime, nullable=True)
    provider_request_started_at = db.Column(db.DateTime, nullable=True)
    provider_first_chunk_at = db.Column(db.DateTime, nullable=True)
    provider_first_status_at = db.Column(db.DateTime, nullable=True)
    provider_first_thought_at = db.Column(db.DateTime, nullable=True)
    provider_first_content_at = db.Column(db.DateTime, nullable=True)
    stream_first_pubsub_at = db.Column(db.DateTime, nullable=True)
    stream_first_status_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_first_thought_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_first_content_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_done_at = db.Column(db.DateTime, nullable=True)
    worker_done_at = db.Column(db.DateTime, nullable=True)
    client_done_at = db.Column(db.DateTime, nullable=True)
    client_total_latency_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

_LATENCY_PHASE_TO_FIELD = {
    "route_received_ms": "route_received_at",
    "route_dispatch_ms": "route_dispatch_at",
    "route_stream_open_ms": "route_stream_open_at",
    "worker_started_ms": "worker_started_at",
    "provider_request_started_ms": "provider_request_started_at",
    "provider_first_chunk_ms": "provider_first_chunk_at",
    "provider_first_status_ms": "provider_first_status_at",
    "provider_first_thought_ms": "provider_first_thought_at",
    "provider_first_content_ms": "provider_first_content_at",
    "stream_first_pubsub_ms": "stream_first_pubsub_at",
    "stream_first_status_to_client_ms": "stream_first_status_to_client_at",
    "stream_first_thought_to_client_ms": "stream_first_thought_to_client_at",
    "stream_first_content_to_client_ms": "stream_first_content_to_client_at",
    "stream_done_ms": "stream_done_at",
    "worker_done_ms": "worker_done_at",
}

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

def _now_epoch_ms():
    return int(time.time() * 1000)

def _latency_trace_key(job_id):
    return f"{_LATENCY_TRACE_PREFIX}{job_id}"

def _latency_mark(job_id, phase, ts_ms=None, only_if_missing=False):
    if not job_id or not phase:
        return None
    try:
        v = int(ts_ms if ts_ms is not None else _now_epoch_ms())
    except Exception:
        v = _now_epoch_ms()
    try:
        k = _latency_trace_key(job_id)
        if only_if_missing:
            redis_conn.hsetnx(k, phase, v)
        else:
            redis_conn.hset(k, phase, v)
        redis_conn.expire(k, _LATENCY_TRACE_TTL_SECONDS)
    except Exception:
        pass
    return v

def _latency_mark_once(job_id, phase, ts_ms=None):
    return _latency_mark(job_id, phase, ts_ms=ts_ms, only_if_missing=True)

def _latency_read(job_id):
    if not job_id:
        return {}
    out = {}
    try:
        raw = redis_conn.hgetall(_latency_trace_key(job_id)) or {}
    except Exception:
        raw = {}
    for k, v in raw.items():
        try:
            ks = k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)
            vs = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
            out[ks] = int(vs)
        except Exception:
            continue
    return out

def _epoch_ms_to_utc_datetime(ms):
    try:
        ms_i = int(ms)
    except Exception:
        return None
    if ms_i < 946684800000 or ms_i > 4102444800000:
        return None
    try:
        return datetime.utcfromtimestamp(ms_i / 1000.0)
    except Exception:
        return None

def _upsert_chat_latency_trace(job_id, user_id, thread_public_id=None, model=None, execution_path=None, client_sent_at_ms=None, client_first_event_type=None, client_first_latency_ms=None, client_done_at_ms=None, client_total_latency_ms=None):
    if not job_id or not user_id:
        return None
    try:
        trace = ChatLatencyTrace.query.filter_by(job_id=job_id).first()
        if not trace:
            trace = ChatLatencyTrace(job_id=job_id, user_id=user_id)
        if thread_public_id:
            trace.thread_public_id = str(thread_public_id)[:64]
        if model:
            trace.model = str(model)[:80]
        if execution_path:
            trace.execution_path = str(execution_path)[:24]
        if client_sent_at_ms is not None:
            dt = _epoch_ms_to_utc_datetime(client_sent_at_ms)
            if dt:
                trace.client_sent_at = dt
        if client_first_event_type:
            trace.client_first_event_type = str(client_first_event_type)[:32]
        if client_first_latency_ms is not None:
            try:
                c_ms = max(0, int(client_first_latency_ms))
            except Exception:
                c_ms = None
            if c_ms is not None and (trace.client_first_latency_ms is None or c_ms < trace.client_first_latency_ms):
                trace.client_first_latency_ms = c_ms
        
        if client_done_at_ms is not None:
            dt = _epoch_ms_to_utc_datetime(client_done_at_ms)
            if dt:
                trace.client_done_at = dt
        if client_total_latency_ms is not None:
            try:
                t_ms = max(0, int(client_total_latency_ms))
            except Exception:
                t_ms = None
            if t_ms is not None:
                trace.client_total_latency_ms = t_ms

        phases = _latency_read(job_id)
        for phase_key, field_name in _LATENCY_PHASE_TO_FIELD.items():
            if getattr(trace, field_name, None) is not None:
                continue
            dt = _epoch_ms_to_utc_datetime(phases.get(phase_key))
            if dt:
                setattr(trace, field_name, dt)
        db.session.add(trace)
        safe_db_commit()
        return trace
    except Exception:
        db.session.rollback()
        return None

def _trace_delta_ms(trace, start_field, end_field):
    if not trace:
        return None
    a = getattr(trace, start_field, None)
    b = getattr(trace, end_field, None)
    if not a or not b:
        return None
    try:
        return int((b - a).total_seconds() * 1000)
    except Exception:
        return None

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

def _secure_delete_tree(path):
    if not path or not os.path.exists(path):
        return
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                secure_delete(os.path.join(root, name))
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass
    except Exception:
        pass

def _delete_user_account_immediately(user):
    if not user:
        raise ValueError("user_required")
    user_id = int(user.id)

    try:
        ips, tokens = _get_user_identifiers(user)
    except Exception:
        ips, tokens = set(), set()

    Feedback.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    BanAppeal.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    UserClientToken.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    UserSession.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    FileCache.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    BannedIdentifier.query.filter_by(source_user_id=user_id).delete(synchronize_session=False)
    _unblock_identifiers(ips, tokens)

    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    _secure_delete_tree(user_dir)
    _secure_delete_tree(_chunk_user_dir(user_id))

    try:
        redis_conn.delete(f"migration_status:{user_id}")
        redis_conn.delete(f"migration_progress:{user_id}")
        redis_conn.delete(f"bot:score:{user_id}")
    except Exception:
        pass

    db.session.delete(user)
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
    
    # Try public_id first
    t = Thread.query.filter_by(public_id=ident_str, user_id=user_id).first()
    if t:
        return t
        
    # Try numerical id
    if ident_str.isdigit():
        t = Thread.query.get(int(ident_str))
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
    initial_theme_color = normalize_theme_color(getattr(current_user, 'theme_color', '')) if current_user.is_authenticated else ""
    return {
        'csrf_token': get_csrf_token(),
        'app_version': app.config.get('APP_VERSION'),
        'system_version': app.config.get('SYSTEM_VERSION'),
        'is_admin': is_admin,
        'attachment_max_files': app.config.get('ATTACHMENT_MAX_FILES', 30),
        'upload_concurrency': app.config.get('UPLOAD_CONCURRENCY', 3),
        'initial_theme_color': initial_theme_color,
        'initial_theme_css': build_theme_css_vars(initial_theme_color),
    }

def validate_csrf():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    session_token = session.get('csrf_token')
    res = bool(token and token == session_token)
    log_force(f"DEBUG: validate_csrf header_token={token}, session_token={session_token}, result={res}")
    return res

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
    """
    Executes a raw SQL command.
    WARNING: This function executes raw SQL and is potentially vulnerable to SQL injection if used with untrusted input.
    It should ONLY be used for internal schema migrations with hardcoded SQL strings.
    """
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

def ensure_thread_temporary_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='is_temporary'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN is_temporary BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def ensure_message_token_io_columns():
    try:
        with db.engine.connect() as conn:
            res_in = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_in'"
            )).scalar()
            if not res_in:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_in INTEGER DEFAULT 0"))
            res_out = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_out'"
            )).scalar()
            if not res_out:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_out INTEGER DEFAULT 0"))
            res_thought = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_thought'"
            )).scalar()
            if not res_thought:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_thought INTEGER DEFAULT 0"))
    except Exception:
        pass

def ensure_user_system_prompt_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("system_prompt_enabled", "ALTER TABLE user ADD COLUMN system_prompt_enabled BOOLEAN DEFAULT 1"),
                ("apply_global_system_prompt", "ALTER TABLE user ADD COLUMN apply_global_system_prompt BOOLEAN DEFAULT 1"),
                ("apply_auto_system_prompt_notices", "ALTER TABLE user ADD COLUMN apply_auto_system_prompt_notices BOOLEAN DEFAULT 1"),
                ("auto_system_prompt_notices_config", "ALTER TABLE user ADD COLUMN auto_system_prompt_notices_config TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_gemini_backend_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("gemini_backend", "ALTER TABLE user ADD COLUMN gemini_backend VARCHAR(24) DEFAULT 'gemini_api'"),
                ("gemini_vertex_project", "ALTER TABLE user ADD COLUMN gemini_vertex_project TEXT"),
                ("gemini_vertex_location", "ALTER TABLE user ADD COLUMN gemini_vertex_location VARCHAR(64) DEFAULT 'global'"),
                ("gemini_vertex_credentials_json", "ALTER TABLE user ADD COLUMN gemini_vertex_credentials_json TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_deepseek_api_key_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='deepseek_api_key'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN deepseek_api_key TEXT"))
    except Exception:
        pass

def ensure_user_2fa_default_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("skip_2fa_on_google_login", "ALTER TABLE user ADD COLUMN skip_2fa_on_google_login BOOLEAN DEFAULT 0"),
                ("default_2fa_method", "ALTER TABLE user ADD COLUMN default_2fa_method VARCHAR(16) DEFAULT 'totp'"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_admin_api_key_mode_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='admin_api_key_mode'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN admin_api_key_mode VARCHAR(24) DEFAULT 'env_fallback'"))
    except Exception:
        pass

def ensure_user_model_api_keys_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='model_api_keys'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN model_api_keys TEXT"))
    except Exception:
        pass

def ensure_user_temp_chat_timeout_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='temp_chat_timeout_seconds'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text(
                    f"ALTER TABLE user ADD COLUMN temp_chat_timeout_seconds INTEGER DEFAULT {_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS}"
                ))
    except Exception:
        pass

def ensure_user_compact_prompt_mode_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='compact_prompt_mode'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN compact_prompt_mode BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def ensure_gem_fixed_prompts_column():
    try:
        from sqlalchemy import text
        db.session.execute(text("ALTER TABLE gem ADD COLUMN fixed_prompts_json TEXT"))
        db.session.commit()
        logger.info("Column fixed_prompts_json added to gem table.")
    except Exception:
        db.session.rollback()

def ensure_chat_latency_trace_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("client_done_at", "ALTER TABLE chat_latency_trace ADD COLUMN client_done_at DATETIME"),
                ("client_total_latency_ms", "ALTER TABLE chat_latency_trace ADD COLUMN client_total_latency_ms INTEGER"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='chat_latency_trace' "
                    "AND COLUMN_NAME=:col"
                ), {"col": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_stt_settings_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("mic_transcribe_mode", "ALTER TABLE user ADD COLUMN mic_transcribe_mode VARCHAR(16) DEFAULT 'stt_api'"),
                ("stt_model", "ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)"),
                ("llm_transcribe_prompt", "ALTER TABLE user ADD COLUMN llm_transcribe_prompt TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def cleanup_user_temp_system_prompt_columns():
    try:
        with db.engine.connect() as conn:
            for column_name in ("temp_system_prompt", "temp_system_prompt_enabled"):
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(f"ALTER TABLE user DROP COLUMN {column_name}"))
    except Exception:
        pass

def ensure_user_debug_settings_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='enable_client_debug_log'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN enable_client_debug_log BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception:
        pass

def ensure_user_default_model_columns():
    try:
        with db.engine.connect() as conn:
            # check default_model
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_model'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_model VARCHAR(64) DEFAULT 'gemini-3.1-flash-lite-preview'"))
                conn.commit()
            # check last_model
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_model'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_model VARCHAR(64)"))
                conn.commit()
            # check default_enable_url_context / default_enable_maps
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_enable_url_context'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_enable_maps'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_enable_maps BOOLEAN DEFAULT 0"))
                conn.commit()
            # check last_enable_url_context / last_enable_maps
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_enable_url_context'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_enable_maps'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_enable_maps BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='rich_paste_prompt_default'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN rich_paste_prompt_default TEXT"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='rich_paste_prompt_use_custom_default'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN rich_paste_prompt_use_custom_default BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to ensure user default model columns: {e}")

def ensure_user_google_columns():
    try:
        with db.engine.connect() as conn:
            # google_id
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='google_id'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN google_id VARCHAR(128) UNIQUE"))
                conn.execute(text("ALTER TABLE user ADD INDEX idx_user_google_id (google_id)"))
                conn.commit()
            # google_email
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='google_email'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN google_email VARCHAR(128)"))
                conn.commit()
    except Exception:
        pass

def ensure_db_index(table_name, index_name, ddl):
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME=:table_name "
                "AND INDEX_NAME=:index_name"
            ), {
                "table_name": table_name,
                "index_name": index_name
            }).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text(ddl))
    except Exception:
        pass

def ensure_performance_indexes():
    ensure_db_index(
        "thread",
        "idx_thread_public_id",
        "CREATE INDEX idx_thread_public_id ON thread (public_id)"
    )
    ensure_db_index(
        "thread",
        "idx_thread_user_bookmark_updated",
        "CREATE INDEX idx_thread_user_bookmark_updated ON thread (user_id, is_bookmarked, bookmarked_at, updated_at)"
    )
    ensure_db_index(
        "message",
        "idx_message_thread_ts_id",
        "CREATE INDEX idx_message_thread_ts_id ON message (thread_id, timestamp, id)"
    )
    ensure_db_index(
        "message",
        "idx_message_thread_id",
        "CREATE INDEX idx_message_thread_id ON message (thread_id, id)"
    )

def _normalize_attachment_source(value):
    raw = str(value or "").strip().lower()
    if raw in ("library", "lib", "library_attach"):
        return "library"
    if raw in ("upload", "uploaded", "new_upload"):
        return "upload"
    return "unknown"

def _iter_message_attachment_refs(raw_value):
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except Exception:
        parsed = [raw_value]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return parsed

def _delete_user_upload_ref(user_id, ref):
    norm = _normalize_upload_ref(ref)
    if not norm:
        return False
    if norm.startswith("..") or os.path.isabs(norm):
        return False
    if not norm.startswith(f"{user_id}/"):
        return False
    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
    if not os.path.realpath(fp).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])):
        return False
    secure_delete(fp)
    secure_delete(fp + '.enc')
    _delete_file_cache_for_path(user_id, norm)
    return True

def _temp_chat_member(thread):
    if not thread:
        return None
    return str(thread.public_id or thread.id)

def _temp_chat_state_key(member):
    return f"{_TEMP_CHAT_STATE_PREFIX}{member}"

def _temp_chat_uploads_key(member):
    return f"{_TEMP_CHAT_UPLOADS_PREFIX}{member}"

def _decode_redis_value(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return str(v) if v is not None else None

def _resolve_temp_chat_thread(member):
    if not member:
        return None
    ident = str(member).strip()
    if not ident:
        return None
    t = Thread.query.filter_by(public_id=ident).first()
    if t:
        return t
    if ident.isdigit():
        return Thread.query.get(int(ident))
    return None

def _clear_temp_chat_tracking(member):
    if not member:
        return
    try:
        pipe = redis_conn.pipeline()
        pipe.zrem(_TEMP_CHAT_LAST_SEEN_ZSET, member)
        pipe.delete(_temp_chat_state_key(member))
        pipe.delete(_temp_chat_uploads_key(member))
        pipe.execute()
    except Exception:
        pass

def _clear_temp_chat_tracking_for_thread(thread):
    member = _temp_chat_member(thread)
    if member:
        _clear_temp_chat_tracking(member)

def _normalize_temp_chat_timeout_seconds(value, fallback=None):
    base = _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS if fallback is None else fallback
    sec = _coerce_int_or_none(value)
    if sec is None:
        sec = base
    if sec < _TEMP_CHAT_TIMEOUT_MIN_SECONDS:
        sec = _TEMP_CHAT_TIMEOUT_MIN_SECONDS
    if sec > _TEMP_CHAT_TIMEOUT_MAX_SECONDS:
        sec = _TEMP_CHAT_TIMEOUT_MAX_SECONDS
    return int(sec)

def _get_user_temp_chat_timeout_seconds(user):
    if not user:
        return _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS
    return _normalize_temp_chat_timeout_seconds(getattr(user, "temp_chat_timeout_seconds", None))

def _resolve_temp_chat_timeout_seconds(thread=None, user_id=None, timeout_seconds=None):
    if timeout_seconds is not None:
        return _normalize_temp_chat_timeout_seconds(timeout_seconds)
    if thread is not None:
        try:
            if getattr(thread, "user", None) is not None:
                return _get_user_temp_chat_timeout_seconds(thread.user)
        except Exception:
            pass
        if user_id is None:
            user_id = thread.user_id
    uid = _coerce_int_or_none(user_id)
    if uid is not None:
        try:
            u = User.query.get(int(uid))
        except Exception:
            u = None
        if u:
            return _get_user_temp_chat_timeout_seconds(u)
    return _TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS

def _resolve_temp_chat_member_timeout_seconds(member, thread=None):
    raw = None
    try:
        raw = redis_conn.hget(_temp_chat_state_key(member), "timeout_seconds")
    except Exception:
        raw = None
    parsed = _coerce_int_or_none(_decode_redis_value(raw))
    if parsed is not None:
        return _normalize_temp_chat_timeout_seconds(parsed)
    uid = thread.user_id if thread is not None else None
    return _resolve_temp_chat_timeout_seconds(thread=thread, user_id=uid)

def _mark_temp_chat_presence(thread, user_id=None, timeout_seconds=None):
    if not thread:
        return
    member = _temp_chat_member(thread)
    if not member:
        return
    uid = user_id if user_id is not None else thread.user_id
    timeout_val = _resolve_temp_chat_timeout_seconds(thread=thread, user_id=uid, timeout_seconds=timeout_seconds)
    now_ts = int(time.time())
    expires_at = now_ts + int(timeout_val)
    try:
        pipe = redis_conn.pipeline()
        pipe.zadd(_TEMP_CHAT_LAST_SEEN_ZSET, {member: now_ts})
        pipe.hset(_temp_chat_state_key(member), mapping={
            "user_id": str(uid or ""),
            "thread_id": str(thread.id or ""),
            "thread_public_id": str(thread.public_id or ""),
            "last_seen": str(now_ts),
            "timeout_seconds": str(timeout_val),
            "expires_at": str(expires_at),
        })
        pipe.expire(_temp_chat_state_key(member), _TEMP_CHAT_TRACK_TTL_SECONDS)
        pipe.execute()
    except Exception:
        pass

def _get_temp_chat_runtime_meta(thread, user=None):
    timeout_seconds = _resolve_temp_chat_timeout_seconds(
        thread=thread,
        user_id=(thread.user_id if thread is not None else None),
        timeout_seconds=(_get_user_temp_chat_timeout_seconds(user) if user is not None else None),
    )
    meta = {
        "timeout_seconds": int(timeout_seconds),
        "temp_chat_expires_at": None,
        "temp_chat_remaining_seconds": None,
    }
    if not thread or not bool(getattr(thread, "is_temporary", False)):
        return meta
    member = _temp_chat_member(thread)
    if not member:
        return meta
    now_ts = int(time.time())
    try:
        score = redis_conn.zscore(_TEMP_CHAT_LAST_SEEN_ZSET, member)
    except Exception:
        score = None
    if score is None:
        meta["temp_chat_expires_at"] = now_ts + int(timeout_seconds)
        meta["temp_chat_remaining_seconds"] = int(timeout_seconds)
        return meta
    try:
        timeout_seconds = _resolve_temp_chat_member_timeout_seconds(member, thread=thread)
    except Exception:
        timeout_seconds = _resolve_temp_chat_timeout_seconds(thread=thread, user_id=thread.user_id)
    expires_at = int(float(score) + float(timeout_seconds))
    remaining = max(0, int(expires_at - now_ts))
    meta["timeout_seconds"] = int(timeout_seconds)
    meta["temp_chat_expires_at"] = expires_at
    meta["temp_chat_remaining_seconds"] = remaining
    return meta

def _track_temp_chat_uploaded_refs(thread, user_id, refs):
    if not thread or not bool(getattr(thread, "is_temporary", False)):
        return
    if not refs:
        return
    member = _temp_chat_member(thread)
    if not member:
        return
    normalized = _normalize_attachment_list(refs, user_id)
    if not normalized:
        return
    try:
        pipe = redis_conn.pipeline()
        pipe.sadd(_temp_chat_uploads_key(member), *normalized)
        pipe.expire(_temp_chat_uploads_key(member), _TEMP_CHAT_TRACK_TTL_SECONDS)
        pipe.execute()
    except Exception:
        pass

def _cleanup_temp_chat_member(member, now_ts=None, force=False):
    if not member:
        return False
    member = str(member).strip()
    if not member:
        return False

    thread = _resolve_temp_chat_thread(member)
    if not thread:
        _clear_temp_chat_tracking(member)
        return False
    if not bool(getattr(thread, "is_temporary", False)) and not force:
        _clear_temp_chat_tracking(member)
        return False
    if now_ts is not None and not force:
        try:
            score = redis_conn.zscore(_TEMP_CHAT_LAST_SEEN_ZSET, member)
        except Exception:
            score = None
        if score is None:
            return False
        timeout_seconds = _resolve_temp_chat_member_timeout_seconds(member, thread=thread)
        expires_at = float(score) + float(timeout_seconds)
        if expires_at > float(now_ts):
            return False

    user_id = thread.user_id
    uploaded_paths = []
    try:
        raw_paths = redis_conn.smembers(_temp_chat_uploads_key(member)) or set()
        for raw in raw_paths:
            val = _decode_redis_value(raw)
            if val:
                uploaded_paths.append(val)
    except Exception:
        uploaded_paths = []

    for p in uploaded_paths:
        _delete_user_upload_ref(user_id, p)

    try:
        redis_conn.delete(f"pending_job:{user_id}:{thread.id}")
    except Exception:
        pass

    try:
        db.session.delete(thread)
        safe_db_commit()
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        logger.error(f"Temporary chat cleanup failed ({member}): {e}")
        return False

    _clear_temp_chat_tracking(member)
    log_force(
        f"Temporary chat auto-deleted: member={member}, user_id={user_id}, "
        f"deleted_uploads={len(uploaded_paths)}"
    )
    return True

def _cleanup_stale_temp_chats():
    while True:
        now_ts = int(time.time())
        cutoff = now_ts - _TEMP_CHAT_TIMEOUT_MIN_SECONDS
        try:
            stale_members = redis_conn.zrangebyscore(
                _TEMP_CHAT_LAST_SEEN_ZSET,
                "-inf",
                cutoff,
                start=0,
                num=50,
            )
        except Exception:
            return
        if not stale_members:
            return
        for raw_member in stale_members:
            member = _decode_redis_value(raw_member)
            if member:
                _cleanup_temp_chat_member(member, now_ts=now_ts)

def _temp_chat_monitor_has_lead():
    global _TEMP_CHAT_MONITOR_TOKEN
    if not _TEMP_CHAT_MONITOR_TOKEN:
        _TEMP_CHAT_MONITOR_TOKEN = f"{os.getpid()}:{threading.get_ident()}"
    token = _TEMP_CHAT_MONITOR_TOKEN
    try:
        if redis_conn.set(_TEMP_CHAT_MONITOR_LEADER_KEY, token, nx=True, ex=_TEMP_CHAT_MONITOR_LEASE_SECONDS):
            return True
        cur = redis_conn.get(_TEMP_CHAT_MONITOR_LEADER_KEY)
        cur_val = _decode_redis_value(cur)
        if cur_val == token:
            redis_conn.expire(_TEMP_CHAT_MONITOR_LEADER_KEY, _TEMP_CHAT_MONITOR_LEASE_SECONDS)
            return True
        return False
    except Exception:
        return True

def _temp_chat_monitor_loop():
    while True:
        try:
            if _temp_chat_monitor_has_lead():
                with app.app_context():
                    _cleanup_stale_temp_chats()
        except Exception as e:
            logger.error(f"Temporary chat monitor error: {e}")
        time.sleep(_TEMP_CHAT_MONITOR_INTERVAL)

def _ensure_temp_chat_monitor_running():
    global _TEMP_CHAT_MONITOR_THREAD, _TEMP_CHAT_MONITOR_PID
    pid = os.getpid()
    with _TEMP_CHAT_MONITOR_LOCK:
        if (
            _TEMP_CHAT_MONITOR_THREAD
            and _TEMP_CHAT_MONITOR_THREAD.is_alive()
            and _TEMP_CHAT_MONITOR_PID == pid
        ):
            return
        th = threading.Thread(
            target=_temp_chat_monitor_loop,
            name=f"temp-chat-monitor-{pid}",
            daemon=True,
        )
        _TEMP_CHAT_MONITOR_THREAD = th
        _TEMP_CHAT_MONITOR_PID = pid
        th.start()

def get_bool_app_setting(key, default=False):
    val = get_app_setting(key, None)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def get_bot_detection_global_enabled():
    return get_bool_app_setting("bot_detection_global_enabled", True)

AUTO_SYSTEM_PROMPT_NOTICE_PYTHON = "Python execution is available; you can run Python code when needed."
AUTO_SYSTEM_PROMPT_NOTICE_GEMINI_LOCAL_PYTHON = (
    "Python execution is available locally. To run code, include a python fenced block "
    "that starts with '# EXECUTE' on the first line."
)
AUTO_SYSTEM_PROMPT_NOTICE_GROK_SEARCH = (
    "You can access external links (including X posts) via the web_search and x_search tools. "
    "Use them when the user asks to read URLs or posts."
)
AUTO_SYSTEM_PROMPT_NOTICE_OPENAI_SEARCH = (
    "You can access external links via the web_search tool. If a URL cannot be accessed, say so clearly."
)
AUTO_SYSTEM_PROMPT_NOTICE_MARKER = "編集済みの画像を見てください。"
AUTO_SYSTEM_PROMPT_NOTICE_ATTACHMENT_NAMES = "添付ファイル名:\n{{attachment_names}}"

AUTO_SYSTEM_PROMPT_NOTICE_MATHJAX = (
    "Mathematical formulas should be output in MathJax (LaTeX) format. "
    "Inline formulas should be enclosed in \\( ... \\) and block formulas in $$ ... $$."
)

AUTO_SYSTEM_PROMPT_NOTICE_KEYS = (
    "python",
    "gemini_local_python",
    "grok_search",
    "openai_search",
    "marker",
    "attachment_names",
    "mathjax",
)

AUTO_SYSTEM_PROMPT_NOTICE_LABELS = {
    "python": "Python",
    "gemini_local_python": "Gemini 音声/動画 + Python (ローカル実行時)",
    "grok_search": "Search補助 (Grok)",
    "openai_search": "Search補助 (OpenAI/xAI Responses)",
    "marker": "Marker編集時",
    "attachment_names": "添付ファイル名 (LLM入力時)",
    "mathjax": "MathJax (LaTeX数式)",
}

AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS = {
    "python": AUTO_SYSTEM_PROMPT_NOTICE_PYTHON,
    "gemini_local_python": AUTO_SYSTEM_PROMPT_NOTICE_GEMINI_LOCAL_PYTHON,
    "grok_search": AUTO_SYSTEM_PROMPT_NOTICE_GROK_SEARCH,
    "openai_search": AUTO_SYSTEM_PROMPT_NOTICE_OPENAI_SEARCH,
    "marker": AUTO_SYSTEM_PROMPT_NOTICE_MARKER,
    "attachment_names": AUTO_SYSTEM_PROMPT_NOTICE_ATTACHMENT_NAMES,
    "mathjax": AUTO_SYSTEM_PROMPT_NOTICE_MATHJAX,
}

AUTO_SYSTEM_PROMPT_NOTICE_MAX_CHARS = 4000


def _coerce_bool_like(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _normalize_auto_notice_text(raw_text, default_text):
    if raw_text is None:
        return default_text
    text = str(raw_text).replace("\r\n", "\n").strip()
    if not text:
        return default_text
    if len(text) > AUTO_SYSTEM_PROMPT_NOTICE_MAX_CHARS:
        text = text[:AUTO_SYSTEM_PROMPT_NOTICE_MAX_CHARS]
    return text


def _render_attachment_names_notice(template_text, names):
    cleaned = []
    for name in names or []:
        v = os.path.basename(str(name or "").strip())
        if v:
            cleaned.append(v)
    if not cleaned:
        return ""

    # Keep the association explicit so the model sees each filename as a label,
    # not as a loose bulleted list that could be read as a general summary.
    names_block = "\n".join([f"画像{idx}: {n}" for idx, n in enumerate(cleaned, start=1)])
    rendered = str(template_text or "").replace("\r\n", "\n").strip()
    if not rendered:
        rendered = AUTO_SYSTEM_PROMPT_NOTICE_ATTACHMENT_NAMES

    replaced = False
    for token in ("{{attachment_names}}", "{attachment_names}", "{{attachment_list}}", "{attachment_list}"):
        if token in rendered:
            rendered = rendered.replace(token, names_block)
            replaced = True
    for token in ("{{attachment_count}}", "{attachment_count}"):
        if token in rendered:
            rendered = rendered.replace(token, str(len(cleaned)))
            replaced = True

    # Accept whitespace variants such as "{{ attachment_names }}".
    spaced_rendered = re.sub(r"\{\{\s*(attachment_names|attachment_list)\s*\}\}", names_block, rendered)
    if spaced_rendered != rendered:
        rendered = spaced_rendered
        replaced = True
    spaced_rendered = re.sub(r"\{\{\s*attachment_count\s*\}\}", str(len(cleaned)), rendered)
    if spaced_rendered != rendered:
        rendered = spaced_rendered
        replaced = True

    if not replaced:
        # Backward-compatible fallback for old one-line title texts.
        if rendered.endswith(":") or rendered.endswith("："):
            rendered = f"{rendered}\n{names_block}"
        elif "\n" in rendered:
            rendered = f"{rendered}\n{names_block}"
        else:
            rendered = f"{rendered}:\n{names_block}"

    return rendered.strip()


def _build_default_auto_system_prompt_notices_config():
    config = {}
    for key in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        default_text = AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS.get(key, "")
        config[key] = {
            "label": AUTO_SYSTEM_PROMPT_NOTICE_LABELS.get(key, key),
            "enabled": True,
            "text": default_text,
            "default_text": default_text,
        }
    return config


def get_user_auto_system_prompt_notices_config(user):
    config = _build_default_auto_system_prompt_notices_config()
    raw = None
    try:
        raw = getattr(user, "auto_system_prompt_notices_config", None)
    except Exception:
        raw = None
    if not raw:
        return config
    parsed = None
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
    if not isinstance(parsed, dict):
        return config
    for key in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        item = parsed.get(key)
        if not isinstance(item, dict):
            continue
        default_text = AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS.get(key, "")
        config[key]["enabled"] = _coerce_bool_like(item.get("enabled"), True)
        config[key]["text"] = _normalize_auto_notice_text(item.get("text"), default_text)
    return config


def set_user_auto_system_prompt_notices_config(user, new_config):
    current = get_user_auto_system_prompt_notices_config(user)
    if not isinstance(new_config, dict):
        new_config = {}
    for key in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        item = new_config.get(key)
        if not isinstance(item, dict):
            continue
        if "enabled" in item:
            current[key]["enabled"] = _coerce_bool_like(item.get("enabled"), True)
        if "text" in item:
            default_text = AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS.get(key, "")
            current[key]["text"] = _normalize_auto_notice_text(item.get("text"), default_text)
    stored = {
        key: {
            "enabled": bool(current[key]["enabled"]),
            "text": current[key]["text"],
        }
        for key in AUTO_SYSTEM_PROMPT_NOTICE_KEYS
    }
    user.auto_system_prompt_notices_config = json.dumps(stored, ensure_ascii=False)
    return current


def get_user_auto_system_prompt_notices_enabled(user):
    try:
        return getattr(user, "apply_auto_system_prompt_notices", None) is not False
    except Exception:
        return True


def get_user_auto_system_prompt_notice_enabled(user, notice_key, config=None):
    if notice_key not in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        return False
    if not get_user_auto_system_prompt_notices_enabled(user):
        return False
    if config is None:
        config = get_user_auto_system_prompt_notices_config(user)
    try:
        item = config.get(notice_key) or {}
        return bool(item.get("enabled", True))
    except Exception:
        return True


def get_user_auto_system_prompt_notice_text(user, notice_key, config=None):
    default_text = AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS.get(notice_key, "")
    if notice_key not in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        return default_text
    if config is None:
        config = get_user_auto_system_prompt_notices_config(user)
    try:
        item = config.get(notice_key) or {}
        text = str(item.get("text") or "").strip()
        return text if text else default_text
    except Exception:
        return default_text


def build_auto_system_prompt_notices_preview(user=None):
    if user is None:
        config = _build_default_auto_system_prompt_notices_config()
        global_enabled = True
    else:
        config = get_user_auto_system_prompt_notices_config(user)
        global_enabled = get_user_auto_system_prompt_notices_enabled(user)
    lines = []
    for key in AUTO_SYSTEM_PROMPT_NOTICE_KEYS:
        item = config.get(key) or {}
        label = AUTO_SYSTEM_PROMPT_NOTICE_LABELS.get(key, key)
        enabled = bool(global_enabled and item.get("enabled", True))
        text = str(item.get("text") or AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS.get(key, "")).strip()
        lines.append(f"[{label}] {'ON' if enabled else 'OFF'}")
        lines.append(text)
        lines.append("")
    if lines:
        lines.pop()
    return "\n".join(lines)


def build_global_system_prompt(now=None):
    if now is None:
        now = datetime.now().astimezone()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (UTC{now.strftime('%z')})"

@app.before_request
def ensure_client_token():
    try:
        get_client_token()
    except Exception:
        pass

@app.before_request
def ensure_temp_chat_monitor():
    try:
        _ensure_temp_chat_monitor_running()
    except Exception:
        pass

def _append_vary_header(resp, token):
    if not resp or not token:
        return
    existing = resp.headers.get("Vary", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    lowered = {p.lower() for p in parts}
    if token.lower() not in lowered:
        parts.append(token)
        resp.headers["Vary"] = ", ".join(parts)

def _looks_like_versioned_static(filename):
    name = str(filename or "")
    if not name:
        return False
    if re.search(r"\.v\d+\.\d+\.\d+\.", name):
        return True
    return False

def _apply_performance_cache_headers(response):
    try:
        if not response or request.method != "GET":
            return response
        endpoint = request.endpoint or ""
        view_args = request.view_args or {}
        version_query = (request.args.get("v") or "").strip()

        if endpoint in ("index", "settings_page", "chat_permalink"):
            response.headers.setdefault("Cache-Control", "private, no-cache, max-age=0, must-revalidate")
            _append_vary_header(response, "Cookie")
            return response

        if endpoint == "static":
            filename = view_args.get("filename") or ""
            if version_query or _looks_like_versioned_static(filename):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
    except Exception:
        return response

def _maybe_gzip_response(response):
    try:
        if not ENABLE_HTTP_GZIP:
            return response
        if not response:
            return response
        if request.method == "HEAD":
            return response
        if response.status_code in (204, 304) or response.status_code < 200:
            return response
        if response.is_streamed or response.direct_passthrough:
            return response
        if response.headers.get("Content-Encoding"):
            return response
        cache_control = (response.headers.get("Cache-Control") or "").lower()
        if "no-transform" in cache_control:
            return response
        accept_encoding = (request.headers.get("Accept-Encoding") or "").lower()
        if "gzip" not in accept_encoding:
            return response
        mimetype = (response.mimetype or "").lower()
        if mimetype not in (
            "text/html",
            "text/plain",
            "text/css",
            "application/json",
            "application/javascript",
            "text/javascript",
            "application/xml",
            "image/svg+xml",
        ):
            return response
        raw = response.get_data()
        if not raw or len(raw) < HTTP_GZIP_MIN_BYTES:
            return response
        compressed = gzip.compress(raw, compresslevel=5)
        if not compressed or len(compressed) >= len(raw):
            return response
        response.set_data(compressed)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed))
        _append_vary_header(response, "Accept-Encoding")
        return response
    except Exception:
        return response

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
    response = _apply_performance_cache_headers(response)
    response = _maybe_gzip_response(response)
    return response

@app.before_request
def check_maintenance():
    log_force(f"DEBUG: request reaching before_request: {request.method} {request.path}")
    if app.config.get('MAINTENANCE_MODE'):
        if request.endpoint in ['static', 'login', 'logout', 'toggle_maintenance', 'login_passkey_options', 'login_passkey_verify']: return
        if current_user.is_authenticated and getattr(current_user, "is_admin", False): return
        return render_template('maintenance.html'), 503
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        if request.endpoint not in ['static', 'receive_client_log', 'client_log']:
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

_TOKEN_ENCODER_BY_NAME = {}
_TOKEN_ENCODER_BY_MODEL = {}
_TOKEN_ENCODER_LOCK = threading.Lock()

def _select_tokenizer_name(model_key):
    mk = str(model_key or "").strip().lower()
    if not mk:
        return "o200k_base"
    if "grok" in mk or "gemini" in mk or "deepseek" in mk:
        return "o200k_base"
    if any(x in mk for x in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")):
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
                log_force(f"Token count 0 for non-empty text: {raw[:20]}...")
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
)

PROMPT_TOKEN_MARKERS = (
    "gpt-image",
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
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        _latency_mark_once(job_id, "worker_started_ms")
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
        def pub(dt, d):
            if dt == "status":
                _latency_mark_once(job_id, "provider_first_status_ms")
            elif dt == "thought":
                _latency_mark_once(job_id, "provider_first_thought_ms")
            elif dt == "content":
                _latency_mark_once(job_id, "provider_first_content_ms")
            elif dt in ("done", "error"):
                _latency_mark_once(job_id, "worker_done_ms")
            r.publish(channel, json.dumps({"type": dt, "content": d}))
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
                elif dt in ["error", "done"]:
                    r.setex(f"stream_acc:{job_id}:final", 600, dt)
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
                    history_rev.append({
                        'role': current_node.role, 
                        'content': cnt, 
                        'image_url': current_node.image_url, 
                        'signature': current_node.thought_signature
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
            is_deepseek = is_deepseek_model_key(model_key_l)
            is_grok = 'grok' in model_key_l and 'gpt' not in model_key_l
            gemini_backend_mode = "gemini_api"
            def _is_non_llm_model(m):
                mk = str(m or "").lower().strip()
                if not mk:
                    return False
                if "gpt-image" in mk:
                    return True
                if mk in ("grok-imagine-image", "grok-imagine-image-pro", "grok-imagine-video"):
                    return True
                if "tts" in mk:
                    return True
                if is_gemini_image_model_key(mk):
                    return True
                if "gemini" in mk and "native-audio" in mk:
                    return True
                return False
            is_llm_model = not _is_non_llm_model(model_key_l)
            grok_reasoning_supported = ("grok-3-mini" in model_key_l) or ("reasoning" in model_key_l and "non-reasoning" not in model_key_l) or ("multi-agent" in model_key_l)
            grok_reasoning_effort_supported = "grok-3-mini" in model_key_l
            req_reasoning_effort = (options.get('reasoning_effort') or "").lower().strip()
            reasoning_requested = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
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
                if raw in ("low", "high"):
                    return raw
                lvl = (options.get('thinking_level') or "low").lower()
                return "high" if lvl == "high" else "low"

            def _deepseek_reasoning_effort():
                raw = (options.get('reasoning_effort') or "").lower().strip()
                if raw in ("high", "max"):
                    return raw
                if raw == "xhigh":
                    return "max"
                if raw in ("low", "medium"):
                    return "high"
                return "high"

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

            gemini_runtime = _resolve_gemini_runtime(user)
            model_api_key_override = _get_model_specific_api_key(user, model_key)

            api_keys = {
                'openai': get_k(user.openai_api_key, 'OPENAI_API_KEY'),
                'gemini': gemini_runtime.get('api_key'),
                'xai': get_k(user.xai_api_key, 'XAI_API_KEY'),
                'deepseek': get_k(user.deepseek_api_key, 'DEEPSEEK_API_KEY')
            }

            key = None
            if is_gem: key = model_api_key_override or api_keys.get('gemini')
            elif is_grok: key = model_api_key_override or api_keys.get('xai')
            elif is_deepseek: key = model_api_key_override or api_keys.get('deepseek')
            else: key = model_api_key_override or api_keys.get('openai')

            if is_gem:
                if gemini_runtime.get("backend") == "vertex_ai":
                    if not gemini_runtime.get("vertex_project"):
                        pub("error", "Vertex AI Project ID が未設定です。設定で Gemini Backend を Vertex AI にした場合は Project ID を入力してください。")
                        return
                elif not key:
                    pub("error", "Gemini API Key missing")
                    return
            elif not key:
                pub("error", "API Key missing")
                return

            g_client = None; o_client = None; x_client = None
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
            elif is_grok:
                x_client = _get_xai_client(key)
                o_client = _get_openai_client(key, base_url=f"https://{_XAI_API_HOST}/v1")
            elif is_deepseek:
                o_client = _get_openai_client(key, base_url="https://api.deepseek.com")
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
                    mime_guess = mimetypes.guess_type(clean_fn)[0]
                    mime = _normalize_media_mime(clean_fn, mime_guess)
                    is_text = (mime or '').startswith('text/') or clean_fn.lower().endswith('.txt')
                    if is_pdf:
                        mime = 'application/pdf'
                    elif is_docx:
                        mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    elif is_text:
                        mime = 'text/plain'
                    send_name = _resolve_send_name(clean_fn, mime) if is_llm_model else None

                    if is_pdf:
                        extracted = None
                        try:
                            reader = pypdf.PdfReader(BytesIO(data))
                            extracted = "".join([p.extract_text() + "\n" for p in reader.pages])
                        except Exception:
                            extracted = None
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': extracted if extracted else None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': True,
                            'is_docx': False,
                            'is_text': False,
                            'send_name': send_name,
                            'size': len(data),
                            'mtime': info.get("mtime")
                        })
                    elif is_docx:
                        extracted = _extract_text_from_docx(data)
                        loaded_files.append({
                            'name': clean_fn,
                            'path': clean_fn,
                            'text': extracted if extracted else None,
                            'bytes': data,
                            'mime': mime,
                            'is_pdf': False,
                            'is_docx': True,
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
            gemini_local_python = False
            if is_gem and (has_audio or has_video) and options.get('enable_python'):
                # Gemini code_execution does not accept audio/video inputs; fall back to local exec.
                gemini_local_python = True
                log_force("Gemini: local python mode for audio/video inputs")
                if _auto_notice_enabled("gemini_local_python"):
                    local_py_notice = _auto_notice_text("gemini_local_python")
                    curr_p = options.get('system_prompt') or ""
                    if local_py_notice not in str(curr_p):
                        options['system_prompt'] = f"{local_py_notice}\n\n{curr_p}" if str(curr_p).strip() else local_py_notice
            if (has_audio and not supports_audio_inputs) or (has_video and not supports_video_inputs):
                pub("error", "This model does not support audio/video inputs. Please remove them and retry.")
                return

            full_res, thought_accumulated, generated_images = "", "", []
            signature_parts = []

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
            if is_gem and not auto_enable_url_context and user_auto_search and not disable_auto:
                try:
                    import re
                    check_text = f"{message_text} {quote_text or ''}"
                    if re.search(r'https?://', check_text):
                        auto_enable_url_context = True
                        log_force("Auto-enabled URL context for Gemini URL access")
                except Exception:
                    pass

            # --- 1. GEMINI & GEMINI IMAGE ---
            if is_gem:
                log_force("Routing: Gemini Branch")
                gemini_files_api_enabled = (gemini_backend_mode != "vertex_ai")
                
                # Gemini TTS (Preview)
                if "tts" in model_key:
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
                elif is_gemini_image_model_key(model_key):
                    try:
                        def _collect_gemini_image_output_parts(resp_obj, keep_only_last_image=False):
                            text_chunks = []
                            image_parts = []
                            seen_part_ids = set()

                            def _append_parts(parts_seq):
                                for _part in parts_seq or []:
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
                            ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                            os.makedirs(ud, exist_ok=True)
                            mime = getattr(part_obj.inline_data, "mime_type", None) or "image/png"
                            ext_map = {
                                "image/png": "png",
                                "image/jpeg": "jpg",
                                "image/webp": "webp"
                            }
                            ext = ext_map.get(mime, "png")
                            fn2 = f"gen_{int(time.time())}_{len(generated_images)}.{ext}"
                            fp2 = os.path.join(ud, fn2)
                            img_data = part_obj.inline_data.data
                            if isinstance(img_data, str):
                                img_data = base64.b64decode(img_data)
                            if user_config.get('enable_e2ee'):
                                with open(fp2 + '.enc', 'wb') as f:
                                    f.write(encrypt_bytes(img_data))
                            else:
                                with open(fp2, 'wb') as f:
                                    f.write(img_data)
                            generated_images.append(f"{user_id}/{fn2}")
                            return fn2

                        # [FIX] Apply System Prompt to Image Prompts if available
                        img_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
                        if options.get('system_prompt'):
                            img_prompt = f"{options.get('system_prompt')}\n\n{img_prompt}"

                        mk_lower = str(model_key or "").lower()
                        if "gemini-3.1-flash-image" in mk_lower:
                            img_model = "gemini-3.1-flash-image-preview"
                        elif "2.5" in mk_lower:
                            img_model = "gemini-2.5-flash-image"
                        else:
                            img_model = "gemini-3-pro-image-preview"
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
                            "candidate_count": 1,
                            "response_modalities": ["TEXT", "IMAGE"],
                            "safety_settings": [
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                            ]
                        }
                        if img_model == "gemini-3.1-flash-image-preview":
                            raw_lvl = str(options.get('thinking_level') or 'high').lower()
                            if raw_lvl in ("low", "minimal"):
                                nano_banana2_lvl = "minimal"
                            elif raw_lvl in ("medium", "high"):
                                nano_banana2_lvl = "high"
                            else:
                                nano_banana2_lvl = "high"
                            # Gemini 3.1 Flash Image supports only minimal/high thinking levels.
                            # The UI checkbox controls thought output visibility; internal thinking remains model-driven.
                            config_kwargs["thinking_config"] = types.ThinkingConfig(
                                include_thoughts=bool(options.get('enable_thinking')),
                                thinking_level=nano_banana2_lvl
                            )
                            if options.get('enable_search'):
                                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                        if image_cfg_kwargs:
                            config_kwargs["image_config"] = types.ImageConfig(**image_cfg_kwargs)

                        _mark_provider_request_started()
                        gemini_image_parts = []
                        history_image_refs_included = set()
                        for fi in loaded_files:
                            if fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                                gemini_image_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                        for hp in history_image_parts:
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
                            keep_only_last_image=(img_model == "gemini-3.1-flash-image-preview")
                        )

                        if not image_outputs and img_model == "gemini-3.1-flash-image-preview":
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
                    if "gemini-3.1-pro" in model_key:
                        rm = "gemini-3.1-pro-preview"
                    elif "gemini-3.1-flash-lite" in model_key:
                        rm = "gemini-3.1-flash-lite-preview"
                    elif "gemini-3-flash" in model_key or "gemini-3.0-flash" in model_key:
                        rm = "gemini-3-flash-preview"
                    elif "gemini-3-pro" in model_key or "gemini-3.0-pro" in model_key:
                        rm = "gemini-3-pro-preview"
                    elif "gemini-2.5-flash-lite" in model_key:
                        rm = model_key
                    elif "gemini-2.5" in model_key:
                        rm = "gemini-2.5-flash"

                    conf = {'temperature': 0.7}
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
                    # Avoid forcing "minimal" when users disable thinking, because Gemini 3 does not
                    # support fully turning thinking off and defaults are higher per docs.

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
                        conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))
                    if options.get('system_prompt'):
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

                    curr_parts = [types.Part(text=final_message_text)]
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

                    def _gemini_upload_with_retry(data, mime, suffix, rel_path, label=""):
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
                                    up = g_client.files.upload(file=tmp.name, config={"mimeType": mime})
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
                                            label=f"pdf:{pdf_name}"
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
                            curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
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
                                            label=f"audio:{audio_name}"
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
                                        label=f"video:{video_name}"
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

                    _mark_provider_request_started()
                    log_force(f"STREAM-TRACE: Gemini stream starting for {job_id} model={rm}")
                    stream = g_client.models.generate_content_stream(model=rm, contents=contents, config=types.GenerateContentConfig(**conf))
                    current_py_id = None
                    current_py_code = None
                    final_usage_metadata = None
                    log_force(f"STREAM-TRACE: Gemini stream loop start for {job_id}")
                    for chunk in stream:
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
                                                
                                            img_md = f"\n![Agentic View](/files/{user_id}/{fn2})\n"
                                            full_res += img_md
                                            pub("content", img_md)
                                        except Exception as e:
                                            log_force(f"Agentic Vision Image Error: {e}")
                                        continue

                                    if hasattr(part, 'text') and part.text:
                                        t_delta = part.text
                                        for char in t_delta:
                                            full_res += char
                                            pub("content", char)
                        
                        # Fallback to chunk.text if parts didn't cover it (unlikely but safe)
                        # but be careful not to double-publish.

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

            # --- 1.5 Grok Imagine Image Generation ---
            elif model_key in ("grok-imagine-image", "grok-imagine-image-pro"):
                log_force("Routing: Grok Imagine Branch")
                try:
                    pub("content", "**Generating Image (Grok)...**\n")
                    
                    aspect_ratio = options.get('grok_image_aspect') or "1:1"
                    grok_prompt, history_image_parts = _build_non_llm_image_context(final_message_text)
                    
                    img_response_format = "b64_json"
                    img_kwargs = {
                        "model": model_key,
                        "prompt": grok_prompt,
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
                        img_name = os.path.basename(fi.get('send_name') or fi.get('name') or f"input_{len(img_inputs)}")
                        img_inputs.append((img_name, img_bytes, img_mime))
                    for hp in history_image_parts:
                        ref = hp.get("ref")
                        if any(existing[0] == os.path.basename(ref or "") for existing in img_inputs):
                            continue
                        img_inputs.append((hp['name'], hp['bytes'], hp['mime']))

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
                            "model": model_key,
                            "prompt": grok_prompt,
                            "image": {"url": img_data_url},
                            "response_format": img_response_format
                        }
                        _mark_provider_request_started()
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
                        _mark_provider_request_started()
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
                if options.get('enable_thinking') and grok_reasoning_effort_supported:
                    create_kwargs["reasoning_effort"] = _grok_reasoning_effort()
                elif options.get('enable_thinking') and grok_reasoning_supported:
                    log_force("Grok reasoning_effort not supported for this model; skipping parameter")
                create_kwargs["use_encrypted_content"] = True # Request encrypted reasoning if available
                if options.get('enable_python') and XAI_SDK_AVAILABLE:
                    create_kwargs["tools"] = [x_code_execution()]

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
                
                curr_user_content = [final_message_text]
                current_image_names = []
                for fi in loaded_files:
                    if fi.get('text'): 
                        curr_user_content[0] += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"
                    elif fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                        d_uri = f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"
                        curr_user_content.append(x_image(d_uri))
                        img_label = fi.get('send_name') or fi.get('name') or f"画像{len(current_image_names) + 1}"
                        current_image_names.append(os.path.basename(str(img_label)))
                name_block = _build_attachment_name_block(current_image_names)
                if name_block:
                    curr_user_content[0] += f"\n\n{name_block}"
                
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
                    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                    if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
                    speech_file_path = os.path.join(user_dir, speech_file_name)

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
                    _mark_provider_request_started()
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
                    pub("status", "画像生成の準備中...")
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
                    
                    img_bytes = base64.b64decode(image_data_b64)
                    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                    if not os.path.exists(ud): os.makedirs(ud, exist_ok=True)
                    ext = "png"
                    if format_opt == "jpeg":
                        ext = "jpg"
                    elif format_opt == "webp":
                        ext = "webp"
                    fn2 = f"gen_gpt_{int(time.time())}_{len(generated_images)}.{ext}"
                    fp2 = os.path.join(ud, fn2)
                    
                    pub("status", "画像を保存して暗号化を適用中...")
                    if user_config.get('enable_e2ee'):
                        with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(img_bytes))
                    else:
                        with open(fp2, 'wb') as f: f.write(img_bytes)
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
                    if any(fi.get('bytes') and str(fi.get('mime', '')).startswith('image/') for fi in loaded_files):
                        pub("error", "DeepSeek V4 does not support image inputs. Please remove images and retry.")
                        return
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
                    for fi in loaded_files:
                        if fi.get('text'):
                            user_text += f"\n\n[File: {fi.get('send_name') or fi.get('name') or 'file'}]\n{fi['text']}"

                    if not user_text.strip():
                        pub("error", "DeepSeek request is empty.")
                        return

                    messages.append({"role": "user", "content": user_text})
                    deepseek_kwargs = {
                        "model": model_key,
                        "messages": messages,
                    }
                    enable_reasoning = bool(options.get('enable_thinking')) or (req_reasoning_effort and req_reasoning_effort != "none")
                    if enable_reasoning:
                        deepseek_kwargs["reasoning_effort"] = _deepseek_reasoning_effort()
                        deepseek_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                    else:
                        deepseek_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

                    _mark_provider_request_started()
                    resp = client.chat.completions.create(**deepseek_kwargs)
                    if not resp or not getattr(resp, "choices", None):
                        pub("error", "DeepSeek API Error: Empty response.")
                        return

                    msg = resp.choices[0].message
                    reasoning_text = getattr(msg, "reasoning_content", None)
                    if reasoning_text:
                        thought_accumulated += reasoning_text
                        pub("thought", reasoning_text)

                    content = getattr(msg, "content", None)
                    text_parts = []
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                p_type = part.get("type")
                                p_text = part.get("text")
                            else:
                                p_type = getattr(part, "type", None)
                                p_text = getattr(part, "text", None)
                            if p_type in (None, "text", "output_text") and p_text:
                                text_parts.append(p_text)
                    elif isinstance(content, str):
                        if content:
                            text_parts.append(content)
                    elif content is not None:
                        text_parts.append(str(content))

                    final_text = "".join(text_parts).strip()
                    if final_text:
                        full_res += final_text
                        pub("content", final_text)
                except Exception as e:
                    pub("error", f"DeepSeek Error: {str(e)}")

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
                    kwargs['reasoning'] = {"effort": _grok_reasoning_effort()}
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
                collected_sources = []
                seen_source_urls = set()
                sources_emitted = False
                final_openai_usage = None

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
                                if isinstance(item, dict):
                                    item_type = item.get('type')
                                    content_parts = item.get('content')
                                else:
                                    item_type = getattr(item, 'type', None)
                                    content_parts = getattr(item, 'content', None)
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

            final_thought = json.dumps({'text': thought_accumulated}) if thought_accumulated else None
            is_enc = user_config.get('enable_e2ee', False)
            if is_enc:
                final_content = encrypt_val(final_content)
                if final_thought: final_thought = encrypt_val(final_thought)
            
            assistant_tokens_out = count_tokens_for_display(full_res, model_key, thought_accumulated)
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

            msg_entry = Message(
                thread_id=thread_id, role='assistant', content=final_content, 
                model=model_key, image_url=json.dumps(generated_images) if generated_images else None, 
                thought_data=final_thought, tokens_out=assistant_tokens_out, tokens=sum_token_counts(None, assistant_tokens_out), 
                is_encrypted=is_enc, thought_signature=final_signature,
                parent_id=message_id,
                tokens_thought=tokens_thought_val
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
            except Exception:
                pass

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

def _get_changelogs(page=1, limit=10):
    log_dir = app.config.get('CHANGELOG_FOLDER', os.path.join(os.path.dirname(__file__), 'static/changelogs'))
    all_logs = []
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.md'))
        def _changelog_meta(path):
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r'^(\d{4}-\d{2}-\d{2})_v(.+)$', base)
            if not m:
                m = re.match(r'^(\d{8})_v(.+)$', base)
            if m:
                date_raw, version = m.group(1), m.group(2)
                if len(date_raw) == 8:
                    date_fmt = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date_fmt = date_raw
                date_key = int(date_fmt.replace('-', ''))
                ver_nums = tuple(int(x) for x in re.findall(r'\d+', version)) or (0,)
                title = f"V{version} ({date_fmt})"
                return date_key, ver_nums, title
            return 0, (0,), base
        
        files.sort(key=lambda p: _changelog_meta(p)[:2], reverse=True)
        
        start = (page - 1) * limit
        end = start + limit
        paginated_files = files[start:end]
        
        for f in paginated_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                title = None
                if not content.lstrip().startswith('#'):
                    _, _, title = _changelog_meta(f)
                all_logs.append({'content': content, 'title': title})
            except Exception as e:
                logger.error(f"Error reading changelog file {f}: {e}")
                
        return all_logs, len(files)
    return [], 0

@app.route('/changelog')
def changelog():
    logs, total = _get_changelogs(page=1, limit=10)
    return render_template('changelog.html', logs=logs, total=total, limit=10)

@app.route('/api/changelogs')
def api_changelogs():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    logs, total = _get_changelogs(page=page, limit=limit)
    return jsonify({'logs': logs, 'total': total, 'page': page, 'limit': limit})

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

@app.route('/api/assets/fonts/japanese.ttf')
def serve_japanese_font():
    font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
    if not os.path.exists(font_path):
        # Fallback to other possible locations
        alt_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
        ]
        for p in alt_paths:
            if os.path.exists(p):
                font_path = p
                break
    try:
        return send_file(font_path, mimetype='font/ttf', max_age=31536000)
    except Exception as e:
        log_force(f"Error serving font: {e}")
        abort(404)

@app.route('/sw.js')
def service_worker():
    resp = send_from_directory(app.static_folder, 'sw.js')
    resp.headers['Content-Type'] = 'application/javascript; charset=utf-8'
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['Service-Worker-Allowed'] = '/'
    return resp

# -----------------------------------------------------------
# Auth Routes
# -----------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                  'application/json' in request.headers.get('Accept', '')
        
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form

        if not rate_limit(f"rl:login:ip:{request.remote_addr}", 20, 300):
            if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")

        if not verify_turnstile(form_data.get('cf-turnstile-response')):
            if is_ajax: return jsonify({'error': "Auth Error"}), 401
            return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
            
        username = (form_data.get('username') or '').strip()
        user = User.query.filter_by(username=username).first()
        # Allow login even if IP/Cookie is banned; ban screen will handle after login.
        if user:
            if not rate_limit(f"rl:login:user:{user.id}", 10, 300):
                if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
                return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Too many attempts. Try again later.")
            
            pw = form_data.get('password') or ""
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
                remember = bool(form_data.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
                
            if user.easy_login_hash and user.easy_login_expires_at and now > user.easy_login_expires_at:
                user.easy_login_hash = None
                user.easy_login_expires_at = None
                safe_db_commit()
                
            if user.check_password(pw):
                if user.is_2fa_enabled:
                    session['remember_me'] = bool(form_data.get('remember'))
                    session['pre_2fa_user_id'] = user.id
                    if is_ajax: return jsonify({
                        'status': '2fa_required',
                        'default_method': user.default_2fa_method or 'totp'
                    })
                    return redirect(url_for('verify_2fa'))
                
                remember = bool(form_data.get('remember'))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
                
        if is_ajax: return jsonify({'error': "Invalid credentials"}), 401
        g_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
        if not g_client_id:
            log_force("DEBUG: GOOGLE_CLIENT_ID is missing in .env")
        return render_template('login.html', 
                               site_key=os.getenv('TURNSTILE_SITE_KEY'), 
                               google_client_id=g_client_id, 
                               error="Invalid credentials")
    
    g_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
    return render_template('login.html', 
                           site_key=os.getenv('TURNSTILE_SITE_KEY'), 
                           google_client_id=g_client_id)

@app.route('/login/google')
def login_google():
    if current_user.is_authenticated:
        # If already logged in, we are linking
        session['google_link_mode'] = True
    else:
        session.pop('google_link_mode', None)
    redirect_uri = url_for('login_google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def login_google_callback():
    link_mode = session.pop('google_link_mode', False)
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            flash("Google からユーザー情報を取得できませんでした。")
            return redirect(url_for('login' if not current_user.is_authenticated else 'index'))
        
        google_id = str(user_info.get('sub'))
        email = user_info.get('email')
        
        if current_user.is_authenticated:
            # Explicit linking from settings
            existing_with_id = User.query.filter_by(google_id=google_id).first()
            if existing_with_id and existing_with_id.id != current_user.id:
                flash("この Google アカウントは既に他のユーザーに紐付けられています。")
                return redirect(url_for('index'))
            
            current_user.google_id = google_id
            if not current_user.google_email:
                current_user.google_email = email
            safe_db_commit()
            flash("Google アカウントと連携しました。")
            return redirect(url_for('index'))

        # Login/Signup flow
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            # Try to link by email if user exists but google_id is not set
            user = User.query.filter(or_(User.google_email == email, User.username == email)).first()
            if user:
                user.google_id = google_id
                if not user.google_email:
                    user.google_email = email
                safe_db_commit()
            else:
                # Create new user
                user = User(
                    username=email,
                    google_id=google_id,
                    google_email=email,
                    is_setup_completed=False
                )
                db.session.add(user)
                safe_db_commit()

        if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
            session['pre_2fa_user_id'] = user.id
            session['remember_me'] = True
            return redirect(url_for('verify_2fa'))

        login_user(user, remember=True)
        create_user_session(user)
        record_user_client_token(user)
        
        if not user.is_setup_completed:
            return redirect(url_for('setup'))
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Google Login Callback Error: {e}")
        flash("Google 連携中にエラーが発生しました。")
        return redirect(url_for('login' if not current_user.is_authenticated else 'index'))

@app.route('/login/google/one-tap', methods=['POST'])
def login_google_one_tap():
    token = request.form.get('credential')
    if not token:
        return jsonify({'error': 'No credential provided'}), 400
    
    try:
        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), os.getenv('GOOGLE_CLIENT_ID'))

        google_id = str(idinfo['sub'])
        email = idinfo['email']
        
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            user = User.query.filter(or_(User.google_email == email, User.username == email)).first()
            if user:
                user.google_id = google_id
                if not user.google_email:
                    user.google_email = email
                safe_db_commit()
            else:
                user = User(
                    username=email,
                    google_id=google_id,
                    google_email=email,
                    is_setup_completed=False
                )
                db.session.add(user)
                safe_db_commit()

        if user.is_2fa_enabled and not user.skip_2fa_on_google_login:
            session['pre_2fa_user_id'] = user.id
            session['remember_me'] = True
            return jsonify({
                'status': '2fa_required', 
                'redirect': url_for('verify_2fa'),
                'default_method': user.default_2fa_method or 'totp'
            })

        login_user(user, remember=True)
        create_user_session(user)
        record_user_client_token(user)
        
        if not user.is_setup_completed:
            return jsonify({'status': 'ok', 'redirect': url_for('setup')})
        return jsonify({'status': 'ok', 'redirect': url_for('index')})

    except Exception as e:
        logger.error(f"Google One Tap Login Error: {e}")
        return jsonify({'error': 'Google One Tap 認証中にエラーが発生しました'}), 400

@app.route('/api/account/unlink_google', methods=['POST'])
@login_required
def unlink_google():
    if not current_user.google_id:
        return jsonify({'error': 'Not linked'}), 400
    
    # Optional: Prevent unlinking if no password or other login method exists
    # but here we allow it.
    current_user.google_id = None
    current_user.google_email = None
    safe_db_commit()
    return jsonify({'status': 'ok'})

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
    creds = _load_user_webauthn_credentials(user)
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
        data = request.json or {}
        challenge = session.get('webauthn_login_challenge')
        if not challenge:
            return jsonify({'error': 'Challenge missing'}), 400
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(data.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
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
        _save_user_webauthn_credentials(user, creds)
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
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                  'application/json' in request.headers.get('Accept', '')
        
        if not rate_limit(f"rl:2fa:user:{user.id}", 8, 300):
            if is_ajax: return jsonify({'error': "Too many attempts. Try again later."}), 429
            return render_template('verify_2fa.html', error="Too many attempts. Try again later.")
            
        code = None
        if is_ajax:
            data = request.json or {}
            code = data.get('totp_code')
        
        if not code:
            code = request.form.get('totp_code')
            
        if code:
            secret = decrypt_val(user.totp_secret)
            if secret and pyotp.TOTP(secret).verify(code):
                session.pop('pre_2fa_user_id', None)
                remember = bool(session.pop('remember_me', False))
                login_user(user, remember=remember)
                create_user_session(user)
                record_user_client_token(user)
                if is_ajax: return jsonify({'status': 'ok', 'redirect': url_for('index')})
                return redirect(url_for('index'))
            
            if is_ajax: return jsonify({'error': "Invalid Code"}), 400
            return render_template('verify_2fa.html', error="Invalid Code")
        
        if is_ajax: return jsonify({'error': "Code required"}), 400
            
    has_totp = bool(user.totp_secret)
    has_webauthn = bool(_load_user_webauthn_credentials(user))
    default_method = user.default_2fa_method or 'totp'
    
    # If the default method is not available, switch to the one that is
    if default_method == 'totp' and not has_totp and has_webauthn:
        default_method = 'webauthn'
    elif default_method == 'webauthn' and not has_webauthn and has_totp:
        default_method = 'totp'

    return render_template('verify_2fa.html', 
                           has_totp=has_totp, 
                           has_webauthn=has_webauthn, 
                           default_method=default_method)

@app.route('/verify-2fa/webauthn/options', methods=['POST'])
def verify_2fa_webauthn_options():
    user_id = session.get('pre_2fa_user_id')
    logger.info(f"WebAuthn Options Req: user_id={user_id}, session={session.keys()}")
    if not user_id: return jsonify({'error': 'Session expired'}), 401
    user = User.query.get(user_id)
    
    creds = _load_user_webauthn_credentials(user)
    
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
        data = request.json or {}
        challenge = session.get('webauthn_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        creds = _load_user_webauthn_credentials(user)
        credential_id = str(data.get('id') or '').strip()
        current_cred = next((c for c in creds if c['id'] == credential_id), None)
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
        _save_user_webauthn_credentials(user, creds)
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
        try:
            vertex_credentials_json = _normalize_gemini_vertex_credentials_json(request.form.get('gemini_vertex_credentials_json'))
        except ValueError as e:
            return render_template('setup.html', error=str(e))
        current_user.openai_api_key = encrypt_val(request.form.get('openai_key'))
        current_user.gemini_api_key = encrypt_val(request.form.get('gemini_key'))
        current_user.deepseek_api_key = encrypt_val(request.form.get('deepseek_key'))
        current_user.gemini_backend = _normalize_gemini_backend(request.form.get('gemini_backend'))
        current_user.gemini_vertex_project = encrypt_val(request.form.get('gemini_vertex_project'))
        current_user.gemini_vertex_location = _normalize_gemini_vertex_location(request.form.get('gemini_vertex_location'))
        current_user.gemini_vertex_credentials_json = encrypt_val(vertex_credentials_json)
        current_user.xai_api_key = encrypt_val(request.form.get('xai_key'))
        current_user.google_api_key = encrypt_val(request.form.get('google_key'))
        current_user.google_cloud_project = encrypt_val(request.form.get('google_project'))
        current_user.default_model = request.form.get('default_model') or "gemini-3.1-flash-lite-preview"
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
    session.pop('pre_2fa_user_id', None)
    return redirect(url_for('index'))

# -----------------------------------------------------------
# API Routes
# -----------------------------------------------------------

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    data = request.json or {}
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}"
    _latency_mark_once(job_id, "route_received_ms")

    temporary_requested = _coerce_bool_or_none(data.get('temporary_chat'))
    thread_ref = data.get('thread_id')
    thread_was_created = False
    if thread_ref:
        t = resolve_thread_for_user(thread_ref, current_user.id)
        if not t:
            return jsonify({'error': 'Invalid thread'}), 403
        if temporary_requested is not None:
            t.is_temporary = bool(temporary_requested)
    else:
        # Avoid a separate round trip on first send; create the thread in the same transaction.
        t = Thread(
            user_id=current_user.id,
            public_id=generate_thread_public_id(),
            is_temporary=bool(temporary_requested)
        )
        db.session.add(t)
        thread_was_created = True
    thread_id = t.id
    thread_stream_id = t.public_id if t and t.public_id else None
    
    user_msg = None
    attachment_name_map = {}
    try:
        raw_msg_content = data.get('message')
        msg_content = raw_msg_content
        if user_config['enable_e2ee']: msg_content = encrypt_val(msg_content)
        raw_image_urls = data.get('image_urls') or []
        if not isinstance(raw_image_urls, list):
            raw_image_urls = [raw_image_urls]
        norm_image_urls = _normalize_attachment_list(raw_image_urls, current_user.id)
        raw_image_items = data.get('image_items') or []
        if not isinstance(raw_image_items, list):
            raw_image_items = [raw_image_items]
        uploaded_image_refs = []
        for item in raw_image_items:
            if not isinstance(item, dict):
                continue
            ref = item.get('path') or item.get('filepath') or item.get('url') or item.get('file')
            norm_ref = _normalize_upload_ref(ref)
            if norm_ref and norm_ref.startswith(f"{current_user.id}/"):
                raw_name = item.get('name') or item.get('filename') or item.get('display_name')
                norm_name = _normalize_display_name_for_path(norm_ref, raw_name)
                if norm_name:
                    attachment_name_map[norm_ref] = norm_name
            source = _normalize_attachment_source(item.get('source'))
            if source == "upload" and ref:
                uploaded_image_refs.append(ref)
        explicit_uploaded_refs = data.get('uploaded_image_urls') or []
        if isinstance(explicit_uploaded_refs, list):
            uploaded_image_refs.extend(explicit_uploaded_refs)
        elif explicit_uploaded_refs:
            uploaded_image_refs.append(explicit_uploaded_refs)
        max_files = int(app.config.get('ATTACHMENT_MAX_FILES') or 30)
        if len(norm_image_urls) > max_files:
            return jsonify({'error': f'Too many attachments. Max {max_files} files per message.'}), 400
        
        parent_id = data.get('parent_id', None)
        parent_explicit = data.get('parent_id_explicit', False)
        if isinstance(parent_explicit, str):
            parent_explicit = parent_explicit.strip().lower() in ("1", "true", "yes", "on")
        else:
            parent_explicit = bool(parent_explicit)

        if parent_id is not None and parent_id != "":
            try:
                if isinstance(parent_id, str) and parent_id.strip().lower() in ("null", "none", "root"):
                    parent_id = None
                else:
                    parent_id_int = int(parent_id)
                    if parent_id_int <= 0:
                        parent_id = None
                    else:
                        pm = Message.query.get(parent_id_int)
                        if not pm or pm.thread_id != thread_id or pm.thread.user_id != current_user.id:
                            parent_id = None
                        else:
                            parent_id = pm.id
            except Exception:
                parent_id = None

        if parent_id is None and not parent_explicit and not thread_was_created and t.id is not None:
            # Default to the last message in the thread
            last_msg = Message.query.filter_by(thread_id=t.id).order_by(Message.id.desc()).first()
            if last_msg:
                parent_id = last_msg.id

        # Calculate user tokens on send to avoid worker re-counting.
        user_tokens_in = None
        if user_tokens_in is None:
            user_tokens_in = count_tokens(raw_msg_content or "", data.get('model'))
        user_msg = Message(
            thread=t,
            role='user',
            content=msg_content,
            model=data.get('model'),
            image_url=json.dumps(norm_image_urls) if norm_image_urls else None,
            quote_text=data.get('quote_text'),
            is_encrypted=user_config['enable_e2ee'],
            parent_id=parent_id,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None)
        )
        db.session.add(user_msg)
        if current_user.use_last_chat_settings:
            current_user.last_model = data.get('model')
            current_user.last_enable_search = bool(data.get('enable_search'))
            current_user.last_enable_url_context = bool(data.get('enable_url_context'))
            current_user.last_enable_maps = bool(data.get('enable_maps'))
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
        thread_id = t.id
        thread_stream_id = t.public_id if t and t.public_id else str(thread_id)
        if bool(getattr(t, "is_temporary", False)):
            _mark_temp_chat_presence(
                t,
                current_user.id,
                timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
            )
            _track_temp_chat_uploaded_refs(t, current_user.id, uploaded_image_refs)
        elif temporary_requested is False:
            _clear_temp_chat_tracking_for_thread(t)
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
        'enable_url_context': data.get('enable_url_context'),
        'disable_auto_search': data.get('disable_auto_search'),
        'enable_python': data.get('enable_python'),
        'enable_thinking': data.get('enable_thinking'),
        'thinking_level': data.get('thinking_level'),
        'thinking_budget': data.get('thinking_budget'),
        'reasoning_effort': data.get('reasoning_effort'),
        'enable_system_prompt': data.get('enable_system_prompt'),
        'marker_system_prompt': data.get('marker_system_prompt'),
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
            'attachment_name_map': attachment_name_map,
        }
    if 'thread_custom_instruction' in data:
        options['thread_custom_instruction'] = data.get('thread_custom_instruction')

    model_key = str(data.get('model') or '').strip()
    model_key_l = model_key.lower()
    is_gemini_3 = "gemini-3" in model_key_l or "gemini-3.1" in model_key_l
    no_attachments = not bool(norm_image_urls)
    no_special_tools = not bool(data.get('enable_search')) and not bool(data.get('enable_python')) and not (bool(data.get('enable_maps')) and is_gemini_3)
    no_quote = not bool(data.get('quote_text'))
    no_thread_custom_instruction = not bool((data.get('thread_custom_instruction') or '').strip())
    model_looks_heavy = any(x in model_key_l for x in ("image", "video", "tts", "audio", "native-audio"))
    supports_direct_first_turn = not model_looks_heavy
    # Enhanced Direct Execution: Allow direct path if thread is new OR message history is short (<15 msgs)
    # This prevents the queue-induced delay (Dispatch delay) for typical chat interactions.
    history_is_short = False
    try:
        if not thread_was_created:
            msg_count = Message.query.filter_by(thread_id=thread_id).count()
            history_is_short = (msg_count <= 15)
        else:
            history_is_short = True
    except:
        history_is_short = thread_was_created

    is_reasoning_minimal = (options.get('reasoning_effort') or "").lower() == "minimal"
    
    fast_queue_eligible = bool(
        not model_looks_heavy
        and no_attachments
        # Now allows tools in the fast queue to reduce Dispatch delay
    )
    queue_name = _CHAT_FAST_QUEUE_NAME if fast_queue_eligible else _CHAT_HEAVY_QUEUE_NAME
    
    first_turn_direct_eligible = bool(
        _DIRECT_FIRST_TURN_ENABLED
        and history_is_short
        and supports_direct_first_turn
        and no_attachments
        # Removed no_special_tools constraint: 
        # Allow tools in direct path for fast TTFB when history is short
        and no_thread_custom_instruction
    )
    execution_path = "direct" if first_turn_direct_eligible else "queued"
    try:
        redis_conn.hset(
            _latency_trace_key(job_id),
            mapping={
                "execution_path": execution_path,
                "queue_name": queue_name[:64],
                "model": model_key[:80],
                "thread_public_id": (thread_stream_id or "")[:64],
                "user_id": str(current_user.id),
            }
        )
        redis_conn.expire(_latency_trace_key(job_id), _LATENCY_TRACE_TTL_SECONDS)
    except Exception:
        pass

    if execution_path == "queued":
        enqueue_queue = chat_fast_queue if queue_name == _CHAT_FAST_QUEUE_NAME else chat_heavy_queue
        enqueue_queue.enqueue(
            background_chat_task,
            job_id,
            thread_id,
            data.get('model'),
            user_msg.id,
            options,
            current_user.id,
            user_config,
            job_timeout=600,
            at_front=(queue_name == _CHAT_FAST_QUEUE_NAME)
        )
        _latency_mark_once(job_id, "route_dispatch_ms")
    try:
        redis_conn.setex(
            f"pending_job:{current_user.id}:{thread_id}",
            600,
            json.dumps({
                "job_id": job_id,
                "message_id": user_msg.id,
                "created_at": int(time.time()),
                "model": data.get('model')
            })
        )
    except Exception:
        pass
    try:
        if execution_path == "direct":
            redis_conn.setex(f"stream_acc:{job_id}:status", 600, "高速経路で実行中です。モデル応答を待機しています...")
        else:
            if queue_name == _CHAT_FAST_QUEUE_NAME:
                redis_conn.setex(f"stream_acc:{job_id}:status", 600, "高速キューに投入しました。優先ワーカー待機中です...")
            else:
                redis_conn.setex(f"stream_acc:{job_id}:status", 600, "通常キューに投入しました。ワーカー待機中です...")
    except Exception:
        pass

    direct_worker_started = False
    direct_worker_lock = threading.Lock()
    def _start_direct_worker_once():
        nonlocal direct_worker_started
        if execution_path != "direct":
            return
        with direct_worker_lock:
            if direct_worker_started:
                return
            direct_worker_started = True
            _latency_mark_once(job_id, "route_dispatch_ms")
            th = threading.Thread(
                target=background_chat_task,
                args=(job_id, thread_id, data.get('model'), user_msg.id, options, current_user.id, user_config),
                daemon=True,
                name=f"direct-chat-{job_id}"
            )
            th.start()

    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        _latency_mark_once(job_id, "route_stream_open_ms")
        _start_direct_worker_once()
        if thread_stream_id:
            yield json.dumps({"type": "thread_id", "content": thread_stream_id}) + "\n"
        yield json.dumps({"type": "job_id", "content": job_id}) + "\n"
        try:
            cached_status = redis_conn.get(f"stream_acc:{job_id}:status")
            if cached_status:
                _latency_mark_once(job_id, "stream_first_status_to_client_ms")
                yield json.dumps({"type": "status", "content": cached_status.decode("utf-8", "ignore")}) + "\n"
        except Exception:
            pass
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                if message['type'] == 'message':
                    _latency_mark_once(job_id, "stream_first_pubsub_ms")
                    evt = json.loads(message['data'])
                    evt_type = str(evt.get('type') or '')
                    if evt_type == "status":
                        _latency_mark_once(job_id, "stream_first_status_to_client_ms")
                    elif evt_type == "thought":
                        _latency_mark_once(job_id, "stream_first_thought_to_client_ms")
                    elif evt_type == "content":
                        _latency_mark_once(job_id, "stream_first_content_to_client_ms")
                    yield json.dumps(evt) + "\n"
                    if evt_type in ['done', 'error']:
                        _latency_mark_once(job_id, "stream_done_ms")
                        break
        finally:
            _latency_mark_once(job_id, "stream_done_ms")
            pubsub.unsubscribe()
            _upsert_chat_latency_trace(
                job_id=job_id,
                user_id=current_user.id,
                thread_public_id=thread_stream_id,
                model=model_key,
                execution_path=execution_path
            )
    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/token_estimate', methods=['POST'])
@login_required
def estimate_prompt_tokens_api():
    data = request.get_json(silent=True) or {}
    model_key = str(data.get('model') or '')
    message_text = data.get('message')
    quote_text = data.get('quote_text')
    raw_image_urls = data.get('image_urls') or []
    if not isinstance(raw_image_urls, list):
        raw_image_urls = [raw_image_urls]

    if message_text is None:
        message_text = ''
    else:
        message_text = str(message_text)
    if quote_text is None:
        quote_text = ''
    else:
        quote_text = str(quote_text)

    norm_image_urls = _normalize_attachment_list(raw_image_urls, current_user.id)
    max_files = int(app.config.get('ATTACHMENT_MAX_FILES') or 30)
    if len(norm_image_urls) > max_files:
        norm_image_urls = norm_image_urls[:max_files]
    if quote_text:
        message_for_count = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"
    else:
        message_for_count = message_text

    if not should_count_tokens_for_display(model_key):
        return jsonify({
            'countable': False,
            'tokens_total': None,
            'tokens_prompt': None,
            'tokens_files': None,
            'files_total': len(norm_image_urls),
            'files_counted': 0,
            'files_non_text': 0,
            'files_missing': 0,
            'files_error': 0
        })

    prompt_tokens = count_tokens(message_for_count or "", model_key)
    file_tokens = 0
    files_counted = 0
    files_non_text = 0
    files_missing = 0
    files_error = 0

    for rel_path in norm_image_urls:
        est = _estimate_attachment_prompt_tokens(rel_path, model_key=model_key)
        tok = int(est.get("tokens") or 0)
        file_tokens += tok
        reason = est.get("reason")
        if est.get("countable"):
            files_counted += 1
        elif reason == "missing":
            files_missing += 1
        elif reason in ("non_text", "no_text"):
            files_non_text += 1
        else:
            files_error += 1

    return jsonify({
        'countable': True,
        'tokens_total': prompt_tokens + file_tokens,
        'tokens_prompt': prompt_tokens,
        'tokens_files': file_tokens,
        'files_total': len(norm_image_urls),
        'files_counted': files_counted,
        'files_non_text': files_non_text,
        'files_missing': files_missing,
        'files_error': files_error
    })


@app.route('/chat_stream_resume', methods=['POST'])
@login_required
def chat_stream_resume():
    data = request.json or {}
    job_id = data.get('job_id')
    thread_id = data.get('thread_id')
    if not job_id or not thread_id:
        return jsonify({'error': 'job_id and thread_id required'}), 400
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 403
    pending_raw = None
    try:
        pending_raw = redis_conn.get(f"pending_job:{current_user.id}:{t.id}")
    except Exception:
        pending_raw = None
    if not pending_raw:
        return jsonify({'error': 'no pending job'}), 404
    pending_job = None
    try:
        pending_job = json.loads(pending_raw)
    except Exception:
        try:
            pending_job = {"job_id": pending_raw.decode("utf-8", "ignore")}
        except Exception:
            pending_job = None
    if pending_job and pending_job.get('job_id') and pending_job.get('job_id') != job_id:
        return jsonify({'error': 'job mismatch'}), 404

    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        yield json.dumps({"type": "job_id", "content": job_id}) + "\n"
        try:
            cached_status = redis_conn.get(f"stream_acc:{job_id}:status")
            if cached_status:
                yield json.dumps({"type": "status", "content": cached_status.decode("utf-8", "ignore")}) + "\n"
            cached_thought = redis_conn.get(f"stream_acc:{job_id}:thought")
            if cached_thought:
                yield json.dumps({"type": "thought", "content": cached_thought.decode("utf-8", "ignore")}) + "\n"
            cached_content = redis_conn.get(f"stream_acc:{job_id}:content")
            if cached_content:
                yield json.dumps({"type": "content", "content": cached_content.decode("utf-8", "ignore")}) + "\n"
            cached_search = redis_conn.get(f"stream_acc:{job_id}:search")
            if cached_search:
                yield json.dumps({"type": "search_status", "content": cached_search.decode("utf-8", "ignore")}) + "\n"
            cached_py = redis_conn.hgetall(f"stream_acc:{job_id}:python")
            if cached_py:
                for _, raw in cached_py.items():
                    try:
                        py = json.loads(raw)
                        yield json.dumps({"type": "python", "content": py}) + "\n"
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    yield json.dumps(data) + "\n"
                    if data.get('type') in ['done', 'error']: break
        finally:
            pubsub.unsubscribe()
    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route('/api/stop_chat', methods=['POST'])
@login_required
def stop_chat():
    data = request.json or {}
    job_id = data.get('job_id')
    stop_source = 'job_id'
    if not job_id:
        thread_ref = data.get('thread_id')
        t = resolve_thread_for_user(thread_ref, current_user.id) if thread_ref else None
        if t:
            pending_raw = None
            try:
                pending_raw = redis_conn.get(f"pending_job:{current_user.id}:{t.id}")
            except Exception:
                pending_raw = None
            if pending_raw:
                try:
                    pending_obj = json.loads(pending_raw)
                    job_id = (pending_obj or {}).get('job_id')
                except Exception:
                    try:
                        job_id = pending_raw.decode("utf-8", "ignore")
                    except Exception:
                        job_id = None
                if job_id:
                    stop_source = 'thread_id'
    
    log_force(f"STREAM-STOP-SIGNAL: Received stop request for job_id={job_id} via {stop_source}")
    if job_id:
        redis_conn.set(f"stop_job:{job_id}", "1", ex=300)
        return jsonify({'status': 'stopped', 'job_id': job_id, 'source': stop_source})
    return jsonify({'error': 'no job_id', 'detail': 'pending job not found'}), 400

@app.route('/api/temporary_chat/heartbeat', methods=['POST'])
@login_required
def temporary_chat_heartbeat():
    data = request.json or {}
    thread_ref = data.get('thread_id')
    t = resolve_thread_for_user(thread_ref, current_user.id)
    if not t:
        return jsonify({'error': 'Invalid thread'}), 404
    active = _coerce_bool_or_none(data.get('active'))
    if active is None:
        active = True
    if active and bool(getattr(t, "is_temporary", False)):
        _mark_temp_chat_presence(
            t,
            current_user.id,
            timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
        )
    temp_meta = _get_temp_chat_runtime_meta(t, user=current_user)
    return jsonify({
        'status': 'ok',
        'thread_id': t.public_id or t.id,
        'is_temporary': bool(getattr(t, "is_temporary", False)),
        'timeout_seconds': temp_meta.get('timeout_seconds'),
        'temp_chat_expires_at': temp_meta.get('temp_chat_expires_at'),
        'temp_chat_remaining_seconds': temp_meta.get('temp_chat_remaining_seconds')
    })

@app.route('/api/generate_title', methods=['POST'])
@login_required
def generate_title_api():
    """Auto-generate chat title with multi-model fallback, prioritizing requested model"""
    try:
        data = request.json
        thread_id = data.get('thread_id')
        requested_model = data.get('model_id')  # Captured from frontend
        
        thread = resolve_thread_for_user(thread_id, current_user.id)
        if not thread:
            return jsonify({'error': 'Unauthorized'}), 403

        first_msg = Message.query.filter_by(thread_id=thread.id, role='user').order_by(Message.timestamp).first()
        if not first_msg: return jsonify({'status': 'skipped'})
        
        content = decrypt_val(first_msg.content) if first_msg.is_encrypted else first_msg.content
        title = "New Chat"

        # Determine target provider for requested_model
        primary_provider = None
        if requested_model:
            rml = requested_model.lower()
            if "gemini" in rml: primary_provider = "gemini"
            elif "grok" in rml: primary_provider = "xai"
            elif "deepseek" in rml: primary_provider = "deepseek"
            else: primary_provider = "openai"

        # Preparation
        o_key = (
            _get_model_specific_api_key(current_user, "gpt-4o-mini")
            or decrypt_val(current_user.openai_api_key)
            or (os.getenv('OPENAI_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )
        gemini_runtime = _resolve_gemini_runtime(current_user)
        g_key = _get_model_specific_api_key(current_user, "gemini-2.0-flash-lite-preview") or gemini_runtime.get("api_key")
        x_key = (
            _get_model_specific_api_key(current_user, "grok-beta")
            or decrypt_val(current_user.xai_api_key)
            or (os.getenv('XAI_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )
        d_key = (
            _get_model_specific_api_key(current_user, "deepseek-v4-flash")
            or decrypt_val(current_user.deepseek_api_key)
            or (os.getenv('DEEPSEEK_API_KEY') if _admin_env_fallback_enabled(current_user) else None)
        )

        # 1. Try Primary requested provider/model
        if primary_provider == "openai" and o_key:
            try:
                client = _get_openai_client(o_key, base_url=None)
                resp = client.chat.completions.create(
                    model=requested_model,
                    messages=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output only the title text."},
                        {"role": "user", "content": content[:500]}
                    ],
                    max_tokens=20
                )
                if resp.choices[0].message.content:
                    title = resp.choices[0].message.content.strip().replace('"', '')
            except: pass
        elif primary_provider == "gemini":
            try:
                backend = gemini_runtime.get("backend")
                if (backend == "gemini_api" and bool(g_key)) or (backend == "vertex_ai"):
                    g_client = _get_gemini_client(api_key=g_key, backend=backend, vertex_project=gemini_runtime.get("vertex_project"), vertex_location=gemini_runtime.get("vertex_location"), vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"))
                    if g_client:
                        resp = g_client.models.generate_content(
                            model=requested_model,
                            contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. Output only the title text.\n\nChat: {content[:500]}")]
                        )
                        if resp.text:
                            title = resp.text.strip().replace('"', '')
            except: pass
        elif primary_provider == "xai" and x_key and XAI_SDK_AVAILABLE:
            try:
                x_client = XAIClient(api_key=x_key)
                chat = x_client.chat.create(model=requested_model)
                chat.append(x_system("Generate a short, descriptive title (max 6 words) for this chat conversation. Output only the title text."))
                chat.append(x_user(content[:500]))
                resp = chat.sample()
                if resp and resp.message and resp.message.content:
                    title = resp.message.content.strip().replace('"', '')
            except: pass
        elif primary_provider == "deepseek" and d_key:
            try:
                client = _get_openai_client(d_key, base_url="https://api.deepseek.com")
                resp = client.chat.completions.create(
                    model=requested_model,
                    messages=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output only the title text."},
                        {"role": "user", "content": content[:500]}
                    ],
                    max_tokens=20,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    title = str(resp.choices[0].message.content).strip().replace('"', '')
            except: pass

        # 2. Fallbacks if still "New Chat"
        if title == "New Chat":
            # Try OpenAI (gpt-4o-mini)
            if o_key:
                try:
                    client = _get_openai_client(o_key, base_url=None)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output JSON: {\"title\": \"...\"}"},
                            {"role": "user", "content": content[:500]}
                        ],
                        response_format={"type": "json_object"}
                    )
                    title = json.loads(resp.choices[0].message.content).get('title', 'New Chat')
                except: pass
            
            # Try Gemini (flash)
            if title == "New Chat":
                try:
                    backend = gemini_runtime.get("backend")
                    if (backend == "gemini_api" and bool(g_key)) or (backend == "vertex_ai"):
                        g_client = _get_gemini_client(api_key=g_key, backend=backend, vertex_project=gemini_runtime.get("vertex_project"), vertex_location=gemini_runtime.get("vertex_location"), vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"))
                        if g_client:
                            resp = g_client.models.generate_content(
                                model="gemini-2.0-flash-lite-preview",
                                contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. JSON: {{'title': '...'}}\n\nChat: {content[:500]}")],
                                config=types.GenerateContentConfig(response_mime_type="application/json")
                            )
                            title = json.loads(resp.text).get('title', 'New Chat')
                except: pass

            # Try xAI (grok-beta)
            if title == "New Chat" and x_key and XAI_SDK_AVAILABLE:
                try:
                    x_client = XAIClient(api_key=x_key)
                    chat = x_client.chat.create(model="grok-beta")
                    chat.append(x_system("Generate a short, descriptive title (max 6 words) for this chat conversation. Output only the title text."))
                    chat.append(x_user(content[:500]))
                    resp = chat.sample()
                    if resp and resp.message and resp.message.content:
                        title = resp.message.content.strip()
                except: pass
            
        # 5. Final fallback if still "New Chat" or empty
        if title == "New Chat" or not title.strip():
            # Use content snippet as fallback title
            snippet = content[:50].strip().replace('\n', ' ')
            if snippet:
                title = snippet + ('...' if len(content) > 50 else '')
            else:
                title = "New Chat"

        thread.title = title
        safe_db_commit()
        return jsonify({'status': 'ok', 'title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/robots.txt')
def robots_txt():
    # Allow landing and auth pages, but disallow private/API paths
    lines = [
        "User-agent: *",
        "Disallow: /files/",
        "Disallow: /api/",
        "Disallow: /thread/",
        "Disallow: /chat/",
        "Allow: /login",
        "Allow: /signup",
        "Allow: /landing",
        "Allow: /"
    ]
    return Response("\n".join(lines), mimetype="text/plain")

def _add_file_privacy_headers(resp):
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    resp.headers["Vary"] = "Cookie"
    return resp

def _add_thumb_cache_headers(resp, etag=None):
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "private, max-age=86400, stale-while-revalidate=604800"
    resp.headers["Vary"] = "Cookie"
    if etag:
        resp.headers["ETag"] = f'"{etag}"'
    return resp

@app.route('/files/thumb/<path:filename>')
@login_required
def serve_file_thumb(filename):
    actual_rel_path = _resolve_user_upload_rel_path(filename, current_user.id)
    if not actual_rel_path:
        abort(403)
    info = _get_file_disk_info(actual_rel_path)
    if not info or not info.get("exists"):
        abort(404)

    ext = os.path.splitext(actual_rel_path)[1].lower()
    if ext not in _IMAGE_THUMB_EXTS:
        return redirect(url_for('serve_file', filename=filename))

    cache_key = (
        actual_rel_path,
        info.get("mtime"),
        info.get("size"),
        1 if info.get("is_encrypted") else 0,
        _THUMBNAIL_SIZE,
        _THUMBNAIL_QUALITY
    )
    etag = hashlib.sha256(repr(cache_key).encode()).hexdigest()
    request_etag = request.headers.get("If-None-Match") or ""
    if request_etag:
        req_tokens = [tok.strip().strip('"') for tok in request_etag.split(",") if tok.strip()]
        if "*" in req_tokens or etag in req_tokens:
            resp = Response(status=304)
            return _add_thumb_cache_headers(resp, etag=etag)

    thumb_bytes = _thumbnail_bytes_cache_get(cache_key)
    if thumb_bytes is None:
        data = _load_user_file_bytes(actual_rel_path, info)
        if data is None:
            abort(404)
        try:
            with Image.open(BytesIO(data)) as im:
                if hasattr(Image, "Resampling"):
                    resample_lanczos = Image.Resampling.LANCZOS
                else:
                    resample_lanczos = Image.LANCZOS
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                im.thumbnail((_THUMBNAIL_SIZE, _THUMBNAIL_SIZE), resample=resample_lanczos)
                buf = BytesIO()
                im.save(buf, format="WEBP", quality=_THUMBNAIL_QUALITY, method=4)
                thumb_bytes = buf.getvalue()
        except Exception as e:
            log_force(f"Thumbnail generation failed for {actual_rel_path}: {e}")
            return redirect(url_for('serve_file', filename=filename))
        _thumbnail_bytes_cache_put(cache_key, thumb_bytes)

    resp = send_file(BytesIO(thumb_bytes), mimetype="image/webp")
    return _add_thumb_cache_headers(resp, etag=etag)

@app.route('/files/<path:filename>')
@login_required
def serve_file(filename):
    actual_rel_path = _resolve_user_upload_rel_path(filename, current_user.id)
    if not actual_rel_path:
        abort(403)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], actual_rel_path)
    if not os.path.realpath(file_path).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])):
        abort(403)
    
    enc_path = file_path + '.enc'
    mtype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

    if os.path.exists(file_path):
        resp = send_file(file_path, mimetype=mtype, conditional=True)
        resp.headers.setdefault("Accept-Ranges", "bytes")
        return _add_file_privacy_headers(resp)
    elif os.path.exists(enc_path):
        info = _get_file_disk_info(actual_rel_path)
        data = _load_user_file_bytes(actual_rel_path, info)
        if data is None:
            abort(404)
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
                return _add_file_privacy_headers(resp)
        resp = send_file(BytesIO(data), download_name=os.path.basename(actual_rel_path), as_attachment=False, mimetype=mtype)
        return _add_file_privacy_headers(resp)
    else:
        abort(404)

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    log_force(f"DEBUG: handle_threads started, method={request.method}")
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = 20
        log_force(f"DEBUG: handle_threads query q={q}, page={page}")
        try:
            query = Thread.query.filter_by(user_id=current_user.id).filter(~Thread.title.startswith("[LIBRARY]"))
            if q: 
                if current_user.enable_e2ee: query = query.filter(Thread.title.contains(q))
                else: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
            
            pagination = query.order_by(Thread.is_bookmarked.desc(), Thread.bookmarked_at.desc(), Thread.updated_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
            threads = []
            for t in pagination.items:
                threads.append({
                    'id': t.public_id or t.id,
                    'title': t.title,
                    'is_bookmarked': bool(t.is_bookmarked),
                    'last_model': t.last_model,
                    'is_temporary': bool(getattr(t, "is_temporary", False))
                })
            log_force(f"DEBUG: handle_threads returning {len(threads)} threads")
            return jsonify({
                'threads': threads,
                'has_next': pagination.has_next,
                'next_page': pagination.next_num
            })
        except Exception as e:
            log_force(f"DEBUG: handle_threads GET failed: {e}")
            raise e
    
    # POST logic
    try:
        payload = request.get_json(silent=True) or {}
        requested_temp = _coerce_bool_or_none(payload.get('is_temporary'))
        log_force(f"DEBUG: handle_threads creating new thread, is_temporary={requested_temp}")
        t = Thread(
            user_id=current_user.id,
            public_id=generate_thread_public_id(),
            is_temporary=bool(requested_temp)
        )
        db.session.add(t)
        safe_db_commit()
        log_force(f"DEBUG: handle_threads thread created, public_id={t.public_id}")
        if bool(t.is_temporary):
            _mark_temp_chat_presence(
                t,
                current_user.id,
                timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
            )
        return jsonify({
            'id': t.public_id,
            'title': t.title,
            'is_temporary': bool(t.is_temporary),
            **_get_temp_chat_runtime_meta(t, user=current_user)
        })
    except Exception as e:
        log_force(f"DEBUG: handle_threads POST failed: {e}")
        raise e

@app.route('/api/threads/<thread_id>', methods=['GET', 'DELETE'])
@login_required
def handle_thread_item(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        limit = request.args.get('limit', type=int)
        before_id = request.args.get('before_id', type=int)
        include_meta_raw = str(request.args.get('include_meta', '1')).strip().lower()
        include_meta = include_meta_raw not in ('0', 'false', 'no', 'off')
        if limit is not None:
            limit = max(1, min(limit, 200))
        has_older_messages = False
        total_messages = None
        oldest_loaded_id = None

        q = Message.query.filter_by(thread_id=t.id)
        if before_id:
            q = q.filter(Message.id < before_id)

        if limit:
            # Page by message ID (cursor=before_id) and then restore stable presentation order.
            rows_desc = q.order_by(Message.id.desc()).limit(limit + 1).all()
            if len(rows_desc) > limit:
                has_older_messages = True
                rows_desc = rows_desc[:limit]
            ms = sorted(rows_desc, key=lambda m: ((m.timestamp or datetime.min), m.id))
            if ms:
                oldest_loaded_id = ms[0].id
            if before_id is None:
                try:
                    total_messages = Message.query.filter_by(thread_id=t.id).count()
                except Exception:
                    total_messages = None
        else:
            # Ensure stable ordering even when timestamps collide (e.g., rapid edit/regenerate).
            ms = Message.query.filter_by(thread_id=t.id).order_by(Message.timestamp, Message.id).all()
            if ms:
                oldest_loaded_id = ms[0].id
        res = []
        for m in ms:
            cnt = decrypt_val(m.content) if m.is_encrypted else m.content
            tht = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
            thought_text = extract_reasoning_text(tht)
            token_in = None
            token_out = None
            token_total = None
            tokens_content = None
            tokens_thought = None
            legacy_token_total = None
            legacy_token_in = None
            legacy_token_out = None
            if (m.tokens_in and m.tokens_in > 0) or (m.tokens_out and m.tokens_out > 0):
                token_in = m.tokens_in if m.tokens_in and m.tokens_in > 0 else None
                token_out = m.tokens_out if m.tokens_out and m.tokens_out > 0 else None
                token_total = sum_token_counts(token_in, token_out)
                stored_tokens_thought = getattr(m, 'tokens_thought', None)
                if stored_tokens_thought is not None and stored_tokens_thought > 0:
                    tokens_thought = stored_tokens_thought
            elif m.tokens is not None and m.tokens > 0 and (should_count_tokens_for_display(m.model) or not m.model):
                if m.role == 'user':
                    legacy_token_in = m.tokens
                else:
                    legacy_token_out = m.tokens
                legacy_token_total = m.tokens
            if token_total is None and should_count_tokens_for_display(m.model):
                details = build_message_token_details(m.role, cnt, thought_text, m.model, token_in, token_out)
                token_in = details["tokens_in"] if details["tokens_in"] is not None else token_in
                token_out = details["tokens_out"] if details["tokens_out"] is not None else token_out
                token_total = details["tokens_total"] if details["tokens_total"] is not None else token_total
                tokens_content = details["tokens_content"]
                tokens_thought = details["tokens_thought"]
            if token_total is None and legacy_token_total is not None:
                token_in = token_in if token_in is not None else legacy_token_in
                token_out = token_out if token_out is not None else legacy_token_out
                token_total = legacy_token_total
            res.append({
                'id': m.id, 
                'role': m.role, 
                'content': cnt, 
                'image_url': m.image_url, 
                'model': m.model, 
                'thought_data': tht,
                'tokens': token_total,
                'tokens_in': token_in,
                'tokens_out': token_out,
                'tokens_content': tokens_content,
                'tokens_thought': tokens_thought,
                'is_encrypted': bool(m.is_encrypted),
                'quote_text': m.quote_text,
                'parent_id': m.parent_id
            })
        payload = {
            'messages': res,
            'has_older_messages': bool(has_older_messages),
            'oldest_loaded_id': oldest_loaded_id,
            'loaded_count': len(res),
            'total_messages': total_messages
        }
        if include_meta:
            pending_job = None
            try:
                pending_raw = redis_conn.get(f"pending_job:{current_user.id}:{t.id}")
                if pending_raw:
                    try:
                        pending_job = json.loads(pending_raw)
                    except Exception:
                        pending_job = {"job_id": pending_raw.decode("utf-8", "ignore")}
            except Exception:
                pending_job = None
            payload.update({
                'title': t.title,
                'custom_instruction': t.custom_instruction,
                'include_global_instruction': t.include_global_instruction if t.include_global_instruction is not None else True,
                'last_model': t.last_model,
                'is_temporary': bool(getattr(t, "is_temporary", False)),
                'pending_job': pending_job
            })
            payload.update(_get_temp_chat_runtime_meta(t, user=current_user))
        return jsonify(payload)

    temp_member = _temp_chat_member(t)
    for m in t.messages:
        if m.image_url:
            try:
                for p in _iter_message_attachment_refs(m.image_url):
                    _delete_user_upload_ref(current_user.id, p)
            except Exception:
                pass

    db.session.delete(t)
    safe_db_commit()
    _clear_temp_chat_tracking(temp_member)
    try:
        redis_conn.delete(f"pending_job:{current_user.id}:{t.id}")
    except Exception:
        pass
    return jsonify({'status': 'deleted'})

def _serialize_message_attachment_for_pdf(raw_ref):
    source = "unknown"
    ref = raw_ref
    if isinstance(raw_ref, dict):
        source = _normalize_attachment_source(raw_ref.get("source"))
        ref = raw_ref.get("filepath") or raw_ref.get("path") or raw_ref.get("url") or raw_ref.get("file") or ""
    norm = _normalize_upload_ref(ref)
    if not norm:
        return None
    filename = os.path.basename(norm)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    is_image = ext in {e.lstrip(".") for e in _IMAGE_THUMB_EXTS}
    preview_endpoint = 'serve_file_thumb' if is_image else 'serve_file'
    return {
        "path": norm,
        "filename": filename,
        "source": source,
        "is_image": is_image,
        "url": url_for('serve_file', filename=norm),
        "preview_url": url_for(preview_endpoint, filename=norm)
    }

def _build_thread_pdf_payload(thread, leaf_id=None):
    messages = Message.query.filter_by(thread_id=thread.id).order_by(Message.timestamp, Message.id).all()
    if not messages:
        return {
            "thread": {
                "id": thread.id,
                "public_id": thread.public_id,
                "title": thread.title or "AI Chat"
            },
            "messages": [],
            "leaf_id": None,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

    msg_map = {m.id: m for m in messages}
    leaf = msg_map.get(leaf_id) if leaf_id else None
    if leaf is None:
        leaf = messages[-1]

    path = []
    seen = set()
    curr = leaf
    while curr and curr.id not in seen:
        seen.add(curr.id)
        path.append(curr)
        parent_id = curr.parent_id
        curr = msg_map.get(parent_id) if parent_id else None
    path.reverse()

    serialized = []
    for m in path:
        content = decrypt_val(m.content) if m.is_encrypted else m.content
        thought_raw = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
        thought_text = extract_reasoning_text(thought_raw)
        token_in = None
        token_out = None
        token_total = None
        tokens_content = None
        tokens_thought = None
        legacy_token_total = None
        legacy_token_in = None
        legacy_token_out = None
        if (m.tokens_in and m.tokens_in > 0) or (m.tokens_out and m.tokens_out > 0):
            token_in = m.tokens_in if m.tokens_in and m.tokens_in > 0 else None
            token_out = m.tokens_out if m.tokens_out and m.tokens_out > 0 else None
            token_total = sum_token_counts(token_in, token_out)
            stored_tokens_thought = getattr(m, 'tokens_thought', None)
            if stored_tokens_thought is not None and stored_tokens_thought > 0:
                tokens_thought = stored_tokens_thought
        elif m.tokens is not None and m.tokens > 0 and (should_count_tokens_for_display(m.model) or not m.model):
            if m.role == 'user':
                legacy_token_in = m.tokens
            else:
                legacy_token_out = m.tokens
            legacy_token_total = m.tokens
        if token_total is None and should_count_tokens_for_display(m.model):
            details = build_message_token_details(m.role, content, thought_text, m.model, token_in, token_out)
            token_in = details["tokens_in"] if details["tokens_in"] is not None else token_in
            token_out = details["tokens_out"] if details["tokens_out"] is not None else token_out
            token_total = details["tokens_total"] if details["tokens_total"] is not None else token_total
            tokens_content = details["tokens_content"]
            tokens_thought = details["tokens_thought"]
        if token_total is None and legacy_token_total is not None:
            token_in = token_in if token_in is not None else legacy_token_in
            token_out = token_out if token_out is not None else legacy_token_out
            token_total = legacy_token_total

        attachments = []
        for raw_ref in _iter_message_attachment_refs(m.image_url):
            item = _serialize_message_attachment_for_pdf(raw_ref)
            if item:
                attachments.append(item)

        serialized.append({
            "id": m.id,
            "role": m.role,
            "content": content,
            "image_url": m.image_url,
            "attachments": attachments,
            "model": m.model,
            "thought_data": thought_raw,
            "thought_text": thought_text,
            "tokens": token_total,
            "tokens_in": token_in,
            "tokens_out": token_out,
            "tokens_content": tokens_content,
            "tokens_thought": tokens_thought,
            "is_encrypted": bool(m.is_encrypted),
            "quote_text": m.quote_text,
            "parent_id": m.parent_id,
            "timestamp": m.timestamp.isoformat() if m.timestamp else None
        })

    return {
        "thread": {
            "id": thread.id,
            "public_id": thread.public_id,
            "title": thread.title or "AI Chat",
            "last_model": thread.last_model,
            "custom_instruction": thread.custom_instruction,
            "include_global_instruction": thread.include_global_instruction if thread.include_global_instruction is not None else True
        },
        "messages": serialized,
        "leaf_id": leaf.id,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

_RICH_PASTE_PDF_FONT_STATE = {
    "ready": False,
    "base": "Helvetica",
    "base_bold": "Helvetica-Bold",
    "mono": "Courier",
}


def _ensure_rich_paste_pdf_fonts():
    if _RICH_PASTE_PDF_FONT_STATE.get("ready"):
        return _RICH_PASTE_PDF_FONT_STATE
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _RICH_PASTE_PDF_FONT_STATE["ready"] = True
        return _RICH_PASTE_PDF_FONT_STATE

    font_candidates = [
        ("IPAGothic", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
        ("IPAPGothic", "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"),
        ("NotoSansMono", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"),
    ]
    for font_name, font_path in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception:
            continue

    if "IPAGothic" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["base"] = "IPAGothic"
    if "IPAPGothic" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["base_bold"] = "IPAPGothic"
    elif _RICH_PASTE_PDF_FONT_STATE["base"] == "IPAGothic":
        _RICH_PASTE_PDF_FONT_STATE["base_bold"] = "IPAGothic"
    if "NotoSansMono" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["mono"] = "NotoSansMono"
    _RICH_PASTE_PDF_FONT_STATE["ready"] = True
    return _RICH_PASTE_PDF_FONT_STATE


def _css_color_to_hex(value):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower in {"inherit", "initial", "unset", "transparent", "none", "currentcolor"}:
        return None
    if lower.startswith("#"):
        if len(lower) == 4:
            return "#" + "".join(ch * 2 for ch in lower[1:])
        if len(lower) >= 7:
            return lower[:7]
    m = re.match(r"rgba?\(([^)]+)\)", lower)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            nums = []
            for part in parts[:3]:
                if part.endswith("%"):
                    try:
                        num = int(round(float(part[:-1]) * 2.55))
                    except Exception:
                        return None
                else:
                    try:
                        num = int(float(part))
                    except Exception:
                        return None
                nums.append(max(0, min(255, num)))
            return "#%02x%02x%02x" % tuple(nums)
    return None


def _parse_inline_style(style_text):
    styles = {}
    if not style_text:
        return styles
    for decl in str(style_text).split(";"):
        if ":" not in decl:
            continue
        prop, value = decl.split(":", 1)
        prop = prop.strip().lower()
        value = value.strip()
        if prop in {"color", "background-color", "font-weight", "font-style", "text-decoration"}:
            styles[prop] = value
    return styles


def _rich_paste_pdf_filename(title):
    slug = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]+", "_", str(title or "").strip()).strip("_")
    if not slug:
        slug = "clipboard_rich"
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{ts}.pdf"


def _build_rich_paste_pdf_bytes(title, content_html, created_at=None):
    from io import BytesIO
    from bs4 import BeautifulSoup, NavigableString, Tag
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import HRFlowable, Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

    rl_config.defaultPageSize = A4
    fonts = _ensure_rich_paste_pdf_fonts()
    base_font = fonts.get("base", "Helvetica")
    bold_font = fonts.get("base_bold", "Helvetica-Bold")
    mono_font = fonts.get("mono", "Courier")

    def esc(value):
        return html.escape("" if value is None else str(value), quote=False)

    def normalize_text(value, preserve_newlines=False):
        txt = "" if value is None else str(value)
        txt = txt.replace("\u00a0", " ")
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        if preserve_newlines:
            return txt.strip("\n")
        txt = re.sub(r"[ \t\f\v]+", " ", txt)
        txt = re.sub(r"\n[ \t]+", "\n", txt)
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    def get_attr_style(tag):
        return _parse_inline_style(tag.get("style"))

    def apply_inline_styles(text, tag=None):
        markup = text
        style = get_attr_style(tag) if tag is not None and hasattr(tag, "get") else {}
        if tag is not None and getattr(tag, "name", "").lower() == "code":
            markup = f'<font face="{mono_font}" backColor="#eeeeee">{markup}</font>'
        if "font-weight" in style and "bold" in style["font-weight"].lower():
            markup = f"<b>{markup}</b>"
        if "font-style" in style and "italic" in style["font-style"].lower():
            markup = f"<i>{markup}</i>"
        if "text-decoration" in style and "underline" in style["text-decoration"].lower():
            markup = f"<u>{markup}</u>"
        color_hex = _css_color_to_hex(style.get("color"))
        if color_hex:
            markup = f'<font color="{color_hex}">{markup}</font>'
        back_hex = _css_color_to_hex(style.get("background-color"))
        if back_hex:
            markup = f'<font backColor="{back_hex}">{markup}</font>'
        return markup

    def inline_markup(node):
        if node is None:
            return ""
        if isinstance(node, NavigableString):
            if type(node).__name__ in ('Doctype', 'Comment', 'Declaration', 'CData', 'ProcessingInstruction'):
                return ""
            return esc(str(node)).replace("\n", "<br/>")
        if not isinstance(node, Tag):
            return ""
        tag_name = (node.name or "").lower()
        if tag_name in {"script", "style", "noscript", "meta", "link", "head", "title", "base", "canvas", "svg", "object", "embed"}:
            return ""
        if tag_name == "br":
            return "<br/>"
        rendered = "".join(inline_markup(child) for child in node.children)
        rendered = rendered or ""
        if not rendered:
            return ""
        if tag_name in {"strong", "b"}:
            rendered = f"<b>{rendered}</b>"
        elif tag_name in {"em", "i"}:
            rendered = f"<i>{rendered}</i>"
        elif tag_name in {"u"}:
            rendered = f"<u>{rendered}</u>"
        elif tag_name in {"s", "strike", "del"}:
            rendered = f"<strike>{rendered}</strike>"
        elif tag_name == "code":
            rendered = f'<font face="{mono_font}" backColor="#eeeeee">{rendered}</font>'
        elif tag_name == "a":
            href = str(node.get("href") or "").strip()
            if href:
                rendered = f'<a href="{esc(href)}">{rendered}</a>'
        rendered = apply_inline_styles(rendered, node)
        return rendered

    def paragraph_style(name, font_size=10.5, leading=15, bold=False, italic=False, color="#111827", space_after=6, left_indent=0, first_line_indent=0):
        return ParagraphStyle(
            name,
            parent=styles["BodyText"],
            fontName=bold_font if bold else base_font,
            fontSize=font_size,
            leading=leading,
            textColor=colors.HexColor(color),
            alignment=TA_LEFT,
            spaceAfter=space_after,
            leftIndent=left_indent,
            firstLineIndent=first_line_indent,
            wordWrap="CJK",
            splitLongWords=1,
        )

    styles = getSampleStyleSheet()
    title_style = paragraph_style("RichPasteTitle", font_size=18, leading=23, bold=True, color="#0f172a", space_after=10)
    meta_style = paragraph_style("RichPasteMeta", font_size=9, leading=12, color="#64748b", space_after=12)
    body_style = paragraph_style("RichPasteBody", font_size=10.5, leading=15, color="#111827", space_after=7)
    heading_styles = {
        1: paragraph_style("RichPasteH1", font_size=16, leading=20, bold=True, color="#0f172a", space_after=8),
        2: paragraph_style("RichPasteH2", font_size=14, leading=18, bold=True, color="#0f172a", space_after=8),
        3: paragraph_style("RichPasteH3", font_size=12.5, leading=16, bold=True, color="#0f172a", space_after=7),
        4: paragraph_style("RichPasteH4", font_size=11.5, leading=15, bold=True, color="#0f172a", space_after=6),
        5: paragraph_style("RichPasteH5", font_size=10.8, leading=14, bold=True, color="#0f172a", space_after=6),
        6: paragraph_style("RichPasteH6", font_size=10.5, leading=14, bold=True, color="#0f172a", space_after=6),
    }
    quote_style = paragraph_style("RichPasteQuote", font_size=10.2, leading=15, color="#334155", space_after=0, left_indent=4)
    code_style = ParagraphStyle(
        "RichPasteCode",
        parent=styles["Code"],
        fontName=mono_font,
        fontSize=9.2,
        leading=12.2,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=10,
        spaceBefore=10,
        leftIndent=0,
        rightIndent=0,
        wordWrap="CJK",
        splitLongWords=1,
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        borderRadius=2,
    )
    list_style = paragraph_style("RichPasteList", font_size=10.5, leading=15, color="#111827", space_after=3)
    table_cell_style = paragraph_style("RichPasteTableCell", font_size=9.3, leading=12.5, color="#111827", space_after=0)
    note_style = paragraph_style("RichPasteNote", font_size=9.2, leading=12.2, color="#64748b", space_after=4)

    story = []
    doc_buffer = BytesIO()
    
    # Ensure title and created_at are safe for ReportLab
    safe_title = normalize_text(title) or "Clipboard Export"
    safe_created_at = normalize_text(created_at) or datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    doc = SimpleDocTemplate(
        doc_buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=str(safe_title),
        author="AI Playground",
    )
    available_width = A4[0] - doc.leftMargin - doc.rightMargin

    def add_paragraph(text, style):
        clean = normalize_text(text)
        if not clean:
            return
        try:
            story.append(Paragraph(clean, style))
        except Exception:
            try:
                # Try escaping the whole thing if markup was invalid
                story.append(Paragraph(esc(clean), style))
            except Exception:
                # Last resort: just add as plain text if still failing
                pass

    def add_blockquote(node):
        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            return
        try:
            para = Paragraph(esc(text), quote_style)
            # If the blockquote is very long, don't use Table as it may fail page splitting
            if len(text) > 1200:
                story.append(para)
                return
            box = Table([[para]], colWidths=[available_width])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9eb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor("#f59e0b")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fde68a")),
            ]))
            story.append(box)
        except Exception:
            add_paragraph(esc(text), quote_style)

    def add_hr():
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=6, spaceAfter=6))

    def add_image(node):
        src = str(node.get("src") or "").strip()
        if not src:
            return
        image_bytes = None
        try:
            if src.startswith("data:"):
                header, encoded = src.split(",", 1)
                if ";base64" in header:
                    image_bytes = base64.b64decode(encoded)
                else:
                    image_bytes = unquote(encoded).encode("utf-8")
            elif src.startswith("http://") or src.startswith("https://"):
                response = requests.get(src, timeout=10)
                if response.ok:
                    image_bytes = response.content
            elif src.startswith("/"):
                local_path = os.path.join(app.root_path, src.lstrip("/"))
                if os.path.exists(local_path):
                    with open(local_path, "rb") as fh:
                        image_bytes = fh.read()
        except Exception:
            image_bytes = None
        if not image_bytes:
            alt = normalize_text(node.get("alt") or "image")
            if alt:
                story.append(Paragraph(f"[Image: {esc(alt)}]", note_style))
            return
        try:
            image = Image(BytesIO(image_bytes))
            image.hAlign = "CENTER"
            max_width = available_width
            max_height = 150 * mm
            try:
                image._restrictSize(max_width, max_height)
            except Exception:
                pass
            story.append(image)
            caption = normalize_text(node.get("alt") or "")
            if caption:
                story.append(Paragraph(esc(caption), note_style))
        except Exception:
            alt = normalize_text(node.get("alt") or "image")
            if alt:
                story.append(Paragraph(f"[Image: {esc(alt)}]", note_style))

    def list_item_children_text(li_node):
        inline_parts = []
        nested_lists = []
        for child in li_node.children:
            if isinstance(child, Tag) and (child.name or "").lower() in {"ul", "ol"}:
                nested_lists.append(child)
                continue
            inline_parts.append(inline_markup(child))
        return "".join(inline_parts), nested_lists

    def add_list(list_node, level=0, ordered=False):
        items = list_node.find_all("li", recursive=False)
        for idx, li in enumerate(items, start=1):
            item_markup, nested_lists = list_item_children_text(li)
            # Remove any trailing <br/> or whitespace before normalization
            item_markup = re.sub(r'(<br\s*/?>\s*)+$', '', item_markup.strip())
            item_text = normalize_text(item_markup, preserve_newlines=True)
            bullet = f"{idx}." if ordered else "-"
            item_style = paragraph_style(
                f"RichPasteList{level}_{idx}",
                font_size=10.3,
                leading=14.8,
                color="#111827",
                space_after=3,
                left_indent=max(0, (level + 1) * 12),
                first_line_indent=-12,
            )
            if item_text:
                try:
                    story.append(Paragraph(item_text, item_style, bulletText=bullet))
                except Exception:
                    story.append(Paragraph(esc(item_text), item_style, bulletText=bullet))
            elif not nested_lists:
                # Add empty item if no text and no nested lists
                story.append(Paragraph("&nbsp;", item_style, bulletText=bullet))

            for nested in nested_lists:
                add_list(nested, level=level + 1, ordered=(nested.name or "").lower() == "ol")

    def add_table(table_node):
        rows = []
        header_rows = 0
        tbody = table_node.find("tbody")
        tr_nodes = tbody.find_all("tr", recursive=False) if tbody else table_node.find_all("tr", recursive=False)
        if not tr_nodes:
            tr_nodes = table_node.find_all("tr")
        for row_index, tr in enumerate(tr_nodes):
            cells = []
            cell_nodes = tr.find_all(["th", "td"], recursive=False)
            if not cell_nodes:
                continue
            if any((cell.name or "").lower() == "th" for cell in cell_nodes) and header_rows == 0:
                header_rows = 1
            for cell in cell_nodes:
                cell_markup = normalize_text(inline_markup(cell), preserve_newlines=True)
                if not cell_markup:
                    cell_markup = "&nbsp;"
                cell_style = ParagraphStyle(
                    f"RichPasteTableCell{row_index}_{len(cells)}",
                    parent=table_cell_style,
                    wordWrap="CJK",
                    splitLongWords=1,
                )
                try:
                    p = Paragraph(cell_markup, cell_style)
                except Exception:
                    p = Paragraph(esc(cell_markup), cell_style)
                cells.append(p)
            if cells:
                rows.append(cells)
        if not rows:
            return
        
        col_count = max(len(r) for r in rows)
        if col_count == 0:
            return
        for row in rows:
            while len(row) < col_count:
                row.append(Paragraph("&nbsp;", table_cell_style))
        
        # Split very large tables into chunks to avoid layout issues
        CHUNK_SIZE = 50
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i : i + CHUNK_SIZE]
            is_first_chunk = (i == 0)
            
            # If not first chunk, we might want to repeat header, but SimpleDocTemplate 
            # might handle it if repeatRows is set. However, splitting manual is safer for build.
            current_header_rows = header_rows if is_first_chunk else 0
            
            table = Table(chunk, repeatRows=current_header_rows, hAlign="LEFT", 
                          colWidths=[available_width / col_count] * col_count,
                          splitByRow=1)
            
            style_cmds = [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            if current_header_rows:
                style_cmds.extend([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ])
            table.setStyle(TableStyle(style_cmds))
            story.append(table)
            if i + CHUNK_SIZE < len(rows):
                story.append(Spacer(1, 4))

    def add_pre(node):
        text = normalize_text(node.get_text("\n"), preserve_newlines=True)
        if not text:
            return
        try:
            # XPreformatted inherits from Paragraph and supports splitting across pages.
            # It also interprets XML-like tags, so we MUST escape the content.
            story.append(XPreformatted(esc(text), code_style, dedent=0))
        except Exception:
            try:
                # Fallback to Paragraph with manual line breaks
                story.append(Paragraph(esc(text).replace("\n", "<br/>"), code_style))
            except Exception:
                pass

    def render_node(node):
        if node is None:
            return
        if isinstance(node, NavigableString):
            if type(node).__name__ in ('Doctype', 'Comment', 'Declaration', 'CData', 'ProcessingInstruction'):
                return
            text = normalize_text(str(node))
            if text:
                # NavigableString must be escaped before Paragraph
                add_paragraph(esc(text).replace("\n", "<br/>"), body_style)
            return
        if not isinstance(node, Tag):
            return
        tag_name = (node.name or "").lower()
        if tag_name in {"script", "style", "noscript", "meta", "link", "iframe", "canvas", "svg", "object", "embed", "head", "title"}:
            return
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag_name[1])
            add_paragraph(inline_markup(node), heading_styles.get(level, body_style))
            return
        if tag_name == "p":
            add_paragraph(inline_markup(node), body_style)
            return
        if tag_name in {"div", "section", "article", "main", "header", "footer", "aside", "body", "html"}:
            # Optimization: if it contains only inline elements, process as one paragraph
            inline_only = True
            for child in node.children:
                if isinstance(child, Tag):
                    child_tag = (child.name or "").lower()
                    if child_tag not in {"span", "b", "strong", "i", "em", "u", "s", "a", "code", "font", "br"}:
                        inline_only = False
                        break
            if inline_only:
                markup = inline_markup(node)
                if markup:
                    add_paragraph(markup, body_style)
                return
            for child in node.children:
                render_node(child)
            return
        if tag_name == "blockquote":
            add_blockquote(node)
            return
        if tag_name == "pre":
            add_pre(node)
            return
        if tag_name == "hr":
            add_hr()
            return
        if tag_name == "br":
            story.append(Spacer(1, 3))
            return
        if tag_name == "img":
            add_image(node)
            return
        if tag_name == "figure":
            img = node.find("img")
            if img:
                add_image(img)
            caption = node.find("figcaption")
            if caption:
                add_paragraph(inline_markup(caption), note_style)
            return
        if tag_name in {"ul", "ol"}:
            add_list(node, level=0, ordered=(tag_name == "ol"))
            return
        if tag_name == "table":
            add_table(node)
            return
        text_markup = inline_markup(node)
        if text_markup:
            add_paragraph(text_markup, body_style)

    soup = BeautifulSoup(content_html or "", "html.parser")
    container = soup.body or soup
    story.append(Paragraph(esc(str(safe_title)), title_style))
    story.append(Paragraph(f"Created at: {esc(safe_created_at)}", meta_style))
    story.append(Spacer(1, 6))

    try:
        for child in container.children:
            render_node(child)
    except Exception as e:
        logger.exception("Error during rich paste node rendering")
        story.append(Paragraph(f"[Content rendering partially failed: {esc(str(e))}]", note_style))

    if not any(isinstance(item, (Paragraph, Table, XPreformatted)) for item in story if not isinstance(item, Spacer)):
        story.append(Paragraph("内容がありません。", body_style))

    def draw_page(canvas, doc_obj):
        try:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#dbe3ee"))
            canvas.setLineWidth(0.6)
            canvas.line(doc.leftMargin, doc.height + doc.topMargin + 2, A4[0] - doc.rightMargin, doc.height + doc.topMargin + 2)
            canvas.setFont(base_font, 8.5)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc_obj.page}")
            canvas.restoreState()
        except Exception:
            pass

    try:
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    except Exception as e:
        logger.exception("ReportLab doc.build failed")
        # Fallback build with simple story if it failed due to complex layout
        doc_buffer.seek(0)
        doc_buffer.truncate()
        fallback_story = [Paragraph(f"PDF生成エラーが発生しました: {esc(str(e))}", body_style)]
        doc.build(fallback_story)

    return doc_buffer.getvalue()


@app.route('/api/rich-paste/pdf', methods=['POST'])
@login_required
def rich_paste_pdf():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    
    log_force(
        "[DEBUG] rich_paste_pdf start "
        f"content_type={request.content_type} "
        f"content_length={request.content_length} "
        f"is_json={request.is_json}"
    )

    d = None
    if request.is_json:
        try:
            d = request.get_json(silent=True)
            if d:
                log_force(f"[DEBUG] rich_paste_pdf get_json success keys={list(d.keys())}")
        except Exception as e:
            log_force(f"[DEBUG] rich_paste_pdf get_json exception: {e}")

    if not isinstance(d, dict) or not d:
        if request.form:
            d = request.form.to_dict(flat=True)
            log_force("[DEBUG] rich_paste_pdf used request.form")
        else:
            try:
                # Crucial: Use cache=True to allow multiple reads if needed
                raw_body = request.get_data(cache=True, as_text=True)
                if raw_body and raw_body.strip():
                    log_force(f"[DEBUG] rich_paste_pdf raw_body_len={len(raw_body)}")
                    try:
                        d = json.loads(raw_body)
                        log_force("[DEBUG] rich_paste_pdf json.loads(raw_body) success")
                    except Exception:
                        # Fallback for some clients that might send raw HTML as body
                        if "<html>" in raw_body.lower() or "<div" in raw_body.lower():
                            d = {"html": raw_body}
                            log_force("[DEBUG] rich_paste_pdf treated raw_body as html")
            except Exception as e:
                log_force(f"[DEBUG] rich_paste_pdf get_data exception: {e}")
                d = {}

    if not isinstance(d, dict):
        d = {}

    content_html = str(d.get('html') or '').strip()
    title = str(d.get('title') or 'Clipboard Export').strip() or 'Clipboard Export'
    created_at = str(d.get('created_at') or datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')).strip()

    log_force(
        "[DEBUG] rich_paste_pdf payload info "
        f"title_len={len(title)} "
        f"html_len={len(content_html)} "
        f"keys={sorted(list(d.keys())) if d else 'None'}"
    )

    if not content_html:
        log_force("[DEBUG] rich_paste_pdf error: missing_html")
        return jsonify({'error': 'missing_html'}), 400

    log_force(f"[DEBUG] rich_paste_pdf starting _build_rich_paste_pdf_bytes for title={title}")
    try:
        pdf_bytes = _build_rich_paste_pdf_bytes(title, content_html, created_at=created_at)
        log_force(f"[DEBUG] rich_paste_pdf _build_rich_paste_pdf_bytes finished, size={len(pdf_bytes)}")
    except Exception as e:
        logger.exception("Server-side rich paste PDF generation failed")
        log_force(f"[DEBUG] rich_paste_pdf generation exception: {type(e).__name__}: {e}")
        return jsonify({'error': 'pdf_generation_failed', 'message': str(e)}), 500

    try:
        filename = _rich_paste_pdf_filename(title)
        resp = send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        resp.headers['X-Rich-Paste-Filename'] = filename
        resp.headers['Cache-Control'] = 'no-store'
        log_force(f"[DEBUG] rich_paste_pdf success filename={filename} bytes={len(pdf_bytes)}")
        return resp
    except Exception as e:
        logger.exception("Server-side rich paste PDF response failed")
        log_force(f"[DEBUG] rich_paste_pdf response exception: {e}")
        return jsonify({'error': 'response_failed', 'message': str(e)}), 500


@app.route('/c/<thread_id>/pdf')
@login_required
def export_thread_pdf(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': '403'}), 403
    leaf_id = request.args.get('leaf_id', type=int)
    payload = _build_thread_pdf_payload(t, leaf_id=leaf_id)
    return jsonify(payload)

@app.route('/api/encryption_scan', methods=['GET'])
@login_required
def encryption_scan():
    thread_id = request.args.get('thread_id')
    q = Message.query.join(Thread, Message.thread_id == Thread.id).filter(Thread.user_id == current_user.id)
    target_thread = None
    if thread_id:
        target_thread = resolve_thread_for_user(thread_id, current_user.id)
        if not target_thread:
            return jsonify({'error': 'Invalid thread'}), 403
        q = q.filter(Message.thread_id == target_thread.id)
    try:
        total = q.count()
        encrypted = q.filter(Message.is_encrypted.is_(True)).count()
        unencrypted = q.filter((Message.is_encrypted.is_(False)) | (Message.is_encrypted.is_(None))).order_by(Message.timestamp.desc()).limit(100).all()
        unenc_list = []
        for m in unencrypted:
            unenc_list.append({
                "id": m.id,
                "thread_id": m.thread.public_id if m.thread else None,
                "role": m.role,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            })
        return jsonify({
            "thread_id": target_thread.public_id if target_thread else None,
            "total": total,
            "encrypted": encrypted,
            "unencrypted": total - encrypted,
            "samples": unenc_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threads/<thread_id>/settings', methods=['GET', 'PUT'])
@login_required
def update_thread_settings(thread_id):
    log_force(f"DEBUG: update_thread_settings started for {thread_id}, method={request.method}")
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        log_force(f"DEBUG: thread not found for {thread_id}")
        return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        log_force(f"DEBUG: update_thread_settings GET returning data for {thread_id}")
        return jsonify({
            'custom_instruction': t.custom_instruction,
            'include_global_instruction': t.include_global_instruction if t.include_global_instruction is not None else True,
            'is_temporary': bool(getattr(t, "is_temporary", False)),
            **_get_temp_chat_runtime_meta(t, user=current_user)
        })
    d = request.json or {}
    log_force(f"DEBUG: update_thread_settings received json: {d}")
    if 'custom_instruction' in d:
        t.custom_instruction = d['custom_instruction']
    if 'include_global_instruction' in d:
        t.include_global_instruction = bool(d['include_global_instruction'])
    if 'is_temporary' in d:
        requested_temp = _coerce_bool_or_none(d.get('is_temporary'))
        t.is_temporary = bool(requested_temp)
    
    t.updated_at = datetime.utcnow()
    db.session.add(t)
    log_force(f"DEBUG: calling safe_db_commit")
    safe_db_commit()
    log_force(f"DEBUG: safe_db_commit finished")
    if bool(getattr(t, "is_temporary", False)):
        _mark_temp_chat_presence(
            t,
            current_user.id,
            timeout_seconds=_get_user_temp_chat_timeout_seconds(current_user)
        )
    else:
        _clear_temp_chat_tracking_for_thread(t)
    log_force(f"DEBUG: update_thread_settings returning ok")
    temp_meta = _get_temp_chat_runtime_meta(t, user=current_user)
    return jsonify({
        'status': 'ok',
        'is_temporary': bool(getattr(t, "is_temporary", False)),
        'timeout_seconds': temp_meta.get('timeout_seconds'),
        'temp_chat_expires_at': temp_meta.get('temp_chat_expires_at'),
        'temp_chat_remaining_seconds': temp_meta.get('temp_chat_remaining_seconds')
    })

@app.route('/api/threads/<thread_id>/title', methods=['PUT'])
@login_required
def update_title(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t: return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    safe_db_commit()
    return jsonify({'status': 'ok', 'title': t.title})

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
                for p in _iter_message_attachment_refs(m.image_url):
                    _delete_user_upload_ref(current_user.id, p)
            except Exception:
                pass

    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    try:
        label_map = _get_user_file_label_map(current_user.id)
        msgs = Message.query.join(Thread).filter(Thread.user_id == current_user.id, Message.image_url != None).order_by(Message.timestamp.desc()).all()
        files = []
        seen = set()
        image_exts = {'png','jpg','jpeg','webp','gif'}
        for m in msgs:
            if not m.image_url: continue
            try:
                l = json.loads(m.image_url)
                if not isinstance(l, list): l = [m.image_url]
            except: l = [m.image_url]
            msg_ts = None
            try:
                msg_ts = int(m.timestamp.timestamp())
            except:
                msg_ts = None
            for p in l:
                norm = _normalize_upload_ref(p)
                if norm and norm not in seen:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
                    if os.path.exists(fp) or os.path.exists(fp + '.enc'):
                        seen.add(norm)
                        ext = os.path.splitext(norm)[1].lower().replace('.', '')
                        base_name = os.path.basename(norm)
                        display_name = label_map.get(norm) or base_name
                        files.append({
                            'filename': display_name,
                            'original_filename': base_name,
                            'filepath': norm,
                            'url': url_for('serve_file', filename=norm),
                            'thumbnail_url': url_for('serve_file_thumb', filename=norm) if ext in image_exts else None,
                            'type': 'image' if ext in image_exts else 'file',
                            'ext': ext,
                            'ts': msg_ts
                        })
        # Include uploaded files that are not yet attached to any message
        ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.isdir(ud):
            for entry in os.scandir(ud):
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith('.'):
                    continue
                is_enc = name.endswith('.enc')
                base_name = name[:-4] if is_enc else name
                if not base_name:
                    continue
                rel_path = f"{current_user.id}/{base_name}"
                if rel_path in seen:
                    continue
                ext = os.path.splitext(base_name)[1].lower().replace('.', '')
                seen.add(rel_path)
                ts = None
                try:
                    ts = int(entry.stat().st_mtime)
                except:
                    ts = None
                display_name = label_map.get(rel_path) or os.path.basename(rel_path)
                files.append({
                    'filename': display_name,
                    'original_filename': os.path.basename(rel_path),
                    'filepath': rel_path,
                    'url': url_for('serve_file', filename=rel_path),
                    'thumbnail_url': url_for('serve_file_thumb', filename=rel_path) if ext in image_exts else None,
                    'type': 'image' if ext in image_exts else 'file',
                    'ext': ext,
                    'ts': ts
                })
        return jsonify(files)
    except: return jsonify([])


@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    for f in request.json.get('filenames', []):
        norm = _normalize_upload_ref(f)
        if not norm:
            continue
        if norm.startswith("..") or os.path.isabs(norm): continue
        if norm.startswith(f"{current_user.id}/"):
            fp = os.path.join(app.config['UPLOAD_FOLDER'], norm)
            if not os.path.realpath(fp).startswith(os.path.realpath(app.config['UPLOAD_FOLDER'])): continue
            secure_delete(fp)
            secure_delete(fp + '.enc')
            _delete_file_cache_for_path(current_user.id, norm)
    return jsonify({'status': 'ok'})

@app.route('/api/files/rename', methods=['POST'])
@login_required
def rename_library_file():
    data = request.json or {}
    rel_path = _normalize_upload_ref(data.get('filepath') or data.get('path'))
    if not rel_path:
        return jsonify({'error': 'invalid filepath'}), 400
    if rel_path.startswith("..") or os.path.isabs(rel_path) or not rel_path.startswith(f"{current_user.id}/"):
        return jsonify({'error': 'forbidden'}), 403
    info = _get_file_disk_info(rel_path)
    if not info or not info.get("exists"):
        return jsonify({'error': 'file not found'}), 404
    base_name = os.path.basename(rel_path)
    display_name = _normalize_display_name_for_path(rel_path, data.get('filename') or data.get('name'))
    if not display_name:
        return jsonify({'error': 'invalid filename'}), 400
    try:
        if display_name == base_name:
            FileCache.query.filter_by(user_id=current_user.id, rel_path=rel_path, provider="label").delete()
        else:
            _upsert_file_cache(
                current_user.id,
                rel_path,
                "label",
                file_uri=display_name,
                state="ready",
                last_error=None
            )
        safe_db_commit()
        return jsonify({'status': 'ok', 'filepath': rel_path, 'filename': display_name, 'original_filename': base_name})
    except Exception:
        return jsonify({'error': 'rename failed'}), 500

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    try:
        _delete_user_account_immediately(current_user)
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

def _deny_unless_primary_admin_for_speedtest():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    if not _is_primary_admin_user(current_user):
        return jsonify({'error': 'primary_admin_only'}), 403
    return None

def _mark_speedtest_no_store(resp):
    try:
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp

@app.route('/api/speedtest/ping', methods=['GET'])
@login_required
def speedtest_ping():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    resp = jsonify({
        'status': 'ok',
        'server_time_ms': int(time.time() * 1000)
    })
    return _mark_speedtest_no_store(resp)

@app.route('/api/speedtest/download', methods=['GET'])
@login_required
def speedtest_download():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    try:
        size = int(request.args.get('bytes') or 0)
    except Exception:
        size = 0
    size = max(64 * 1024, min(32 * 1024 * 1024, size or (8 * 1024 * 1024)))
    chunk = (b'0123456789abcdef' * 4096)  # 64KB deterministic payload
    chunk_len = len(chunk)

    def generate():
        remaining = size
        while remaining > 0:
            n = min(chunk_len, remaining)
            yield chunk[:n]
            remaining -= n

    resp = Response(stream_with_context(generate()), mimetype='application/octet-stream')
    resp.headers['Content-Length'] = str(size)
    resp.headers['X-Speedtest-Bytes'] = str(size)
    return _mark_speedtest_no_store(resp)

@app.route('/api/speedtest/upload', methods=['POST'])
@login_required
def speedtest_upload():
    denied = _deny_unless_primary_admin_for_speedtest()
    if denied:
        return denied
    start = time.perf_counter()
    total = 0
    max_bytes = 32 * 1024 * 1024
    try:
        while True:
            chunk = request.stream.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return jsonify({'error': 'payload_too_large', 'max_bytes': max_bytes}), 413
    except RequestEntityTooLarge:
        return jsonify({'error': 'payload_too_large', 'max_bytes': max_bytes}), 413
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    resp = jsonify({
        'status': 'ok',
        'bytes_received': total,
        'server_elapsed_ms': elapsed_ms
    })
    return _mark_speedtest_no_store(resp)

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
    elif action == 'delete_account':
        _delete_user_account_immediately(user)
        return jsonify({'status': 'ok', 'username': username, 'action': action})
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

def build_theme_css_vars(value):
    hex_value = normalize_theme_color(value)
    if not hex_value:
        return ""

    r = int(hex_value[1:3], 16)
    g = int(hex_value[3:5], 16)
    b = int(hex_value[5:7], 16)

    def mix(channel, target, pct):
        return round(channel + (target - channel) * pct)

    def to_hex(red, green, blue):
        return f"#{red:02x}{green:02x}{blue:02x}"

    light = to_hex(mix(r, 255, 0.45), mix(g, 255, 0.45), mix(b, 255, 0.45))
    lighter = to_hex(mix(r, 255, 0.7), mix(g, 255, 0.7), mix(b, 255, 0.7))
    dark = to_hex(mix(r, 0, 0.18), mix(g, 0, 0.18), mix(b, 0, 0.18))
    darker = to_hex(mix(r, 0, 0.32), mix(g, 0, 0.32), mix(b, 0, 0.32))
    rgb = f"{r}, {g}, {b}"
    return (
        ":root{"
        f"--theme-500:{hex_value};"
        f"--theme-600:{dark};"
        f"--theme-700:{darker};"
        f"--theme-300:{light};"
        f"--theme-200:{lighter};"
        f"--theme-rgb:{rgb};"
        "}"
    )

def _normalize_webauthn_credentials(raw):
    if not raw:
        return []
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    creds = []
    seen_ids = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        cred_id = str(item.get('id') or '').strip()
        public_key = str(item.get('public_key') or '').strip()
        if not cred_id or not public_key or cred_id in seen_ids:
            continue
        try:
            base64url_to_bytes(cred_id)
            base64url_to_bytes(public_key)
        except Exception:
            continue
        try:
            sign_count = int(item.get('sign_count', 0) or 0)
        except Exception:
            sign_count = 0
        if sign_count < 0:
            sign_count = 0
        name = str(item.get('name') or '').strip() or 'Security Key'
        created_at = item.get('created_at')
        created_at_val = None
        if created_at is not None:
            created_at_val = str(created_at).strip() or None
        creds.append({
            'id': cred_id,
            'public_key': public_key,
            'sign_count': sign_count,
            'name': name,
            'created_at': created_at_val
        })
        seen_ids.add(cred_id)
    return creds

def _load_user_webauthn_credentials(user):
    if not user:
        return []
    return _normalize_webauthn_credentials(getattr(user, "webauthn_credentials", None))

def _save_user_webauthn_credentials(user, creds):
    normalized = _normalize_webauthn_credentials(creds)
    if not normalized:
        user.webauthn_credentials = None
        return []
    user.webauthn_credentials = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return normalized

def _serialize_public_webauthn_credentials(creds):
    rows = []
    for c in _normalize_webauthn_credentials(creds):
        rows.append({
            'id': c['id'],
            'name': c.get('name') or 'Security Key',
            'created_at': c.get('created_at')
        })
    return rows

def _refresh_user_2fa_state(user):
    has_totp = bool(getattr(user, 'totp_secret', None))
    has_webauthn = bool(_load_user_webauthn_credentials(user))
    if getattr(user, 'passkey_only_login', False) and not has_webauthn:
        user.passkey_only_login = False
    if not has_totp and not has_webauthn:
        user.is_2fa_enabled = False

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        # Ensure we have the latest data from DB
        db.session.refresh(current_user)
        status = redis_conn.get(f"migration_status:{current_user.id}")
        mig_status = status.decode() if status else "idle"
        prog = redis_conn.get(f"migration_progress:{current_user.id}")
        mig_progress = prog.decode() if prog else ""
        sp = current_user.system_prompt
        if current_user.enable_e2ee and sp: sp = decrypt_val(sp)
        # 2FA Status
        has_totp = bool(current_user.totp_secret)
        passkeys = _load_user_webauthn_credentials(current_user)
        has_webauthn = bool(passkeys)
        
        global_prompt_value = get_app_setting("global_system_prompt", "") or ""
        global_prompt_enabled = get_bool_app_setting("global_system_prompt_enabled", True)
        global_prompt_uses_time_fallback = bool(global_prompt_enabled and not str(global_prompt_value).strip())
        global_prompt_effective = ""
        if global_prompt_enabled:
            if str(global_prompt_value).strip():
                global_prompt_effective = str(global_prompt_value)
            else:
                global_prompt_effective = build_global_system_prompt()
        
        auto_notices_config = get_user_auto_system_prompt_notices_config(current_user)

        payload = {
            'system_prompt': sp or "",
            'system_prompt_enabled': current_user.system_prompt_enabled if current_user.system_prompt_enabled is not None else True,
            'apply_global_system_prompt': current_user.apply_global_system_prompt if current_user.apply_global_system_prompt is not None else True,
            'apply_auto_system_prompt_notices': get_user_auto_system_prompt_notices_enabled(current_user),
            'auto_system_prompt_notices_preview': build_auto_system_prompt_notices_preview(current_user),
            'auto_system_prompt_notices_config': auto_notices_config,
            'global_system_prompt': global_prompt_value,
            'global_system_prompt_enabled': global_prompt_enabled,
            'global_system_prompt_effective': global_prompt_effective,
            'global_system_prompt_uses_time_fallback': global_prompt_uses_time_fallback,
            'username': current_user.username, 
            'openai_key': decrypt_val(current_user.openai_api_key) or "", 
            'gemini_key': decrypt_val(current_user.gemini_api_key) or "", 
            'deepseek_key': decrypt_val(current_user.deepseek_api_key) or "",
            'model_api_keys': _load_user_model_api_key_map(current_user),
            'gemini_backend': _normalize_gemini_backend(current_user.gemini_backend),
            'gemini_vertex_project': decrypt_val(current_user.gemini_vertex_project) or "",
            'gemini_vertex_location': _normalize_gemini_vertex_location(current_user.gemini_vertex_location),
            'gemini_vertex_credentials_json': decrypt_val(current_user.gemini_vertex_credentials_json) or "",
            'xai_key': decrypt_val(current_user.xai_api_key) or "",
            'google_key': decrypt_val(current_user.google_api_key) or "",
            'google_project': decrypt_val(current_user.google_cloud_project) or "",
            'mic_transcribe_mode': _normalize_mic_transcribe_mode(getattr(current_user, 'mic_transcribe_mode', None)),
            'stt_model': current_user.stt_model or "gpt-4o-mini-transcribe",
            'llm_transcribe_prompt': _normalize_llm_transcribe_prompt(getattr(current_user, 'llm_transcribe_prompt', None)) or "",
            'llm_transcribe_prompt_default': DEFAULT_LLM_TRANSCRIBE_PROMPT,
            'enter_to_send': current_user.enter_to_send,
            'use_sw_cache': current_user.use_sw_cache,
            'theme_color': current_user.theme_color or "",
            'auto_search_on_links': current_user.auto_search_on_links,
            'compact_prompt_mode': current_user.compact_prompt_mode if current_user.compact_prompt_mode is not None else False,
            'use_last_chat_settings': current_user.use_last_chat_settings,
            'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(current_user),
            'default_model': current_user.default_model or "gemini-3.1-flash-lite-preview",
            'default_enable_search': current_user.default_enable_search,
            'default_enable_url_context': current_user.default_enable_url_context,
            'default_enable_maps': current_user.default_enable_maps,
            'default_enable_python': current_user.default_enable_python,
            'default_enable_thinking': current_user.default_enable_thinking,
            'default_thinking_level': current_user.default_thinking_level or "high",
            'default_thinking_budget': current_user.default_thinking_budget if current_user.default_thinking_budget is not None else 4096,
            'default_reasoning_effort': current_user.default_reasoning_effort or "medium",
            'default_enable_system_prompt': current_user.default_enable_system_prompt,
            'default_safety_setting': current_user.default_safety_setting or "default",
            'rich_paste_prompt_default': current_user.rich_paste_prompt_default or "",
            'rich_paste_prompt_use_custom_default': current_user.rich_paste_prompt_use_custom_default if current_user.rich_paste_prompt_use_custom_default is not None else False,
            'last_model': current_user.last_model or "gemini-3.1-flash-lite-preview",
            'last_enable_search': current_user.last_enable_search,
            'last_enable_url_context': current_user.last_enable_url_context,
            'last_enable_maps': current_user.last_enable_maps,
            'last_enable_python': current_user.last_enable_python,
            'last_enable_thinking': current_user.last_enable_thinking,
            'last_thinking_level': current_user.last_thinking_level or "high",
            'last_thinking_budget': current_user.last_thinking_budget if current_user.last_thinking_budget is not None else 4096,
            'last_reasoning_effort': current_user.last_reasoning_effort or "medium",
            'google_id': current_user.google_id,
            'google_email': current_user.google_email,
            'last_enable_system_prompt': current_user.last_enable_system_prompt,
            'last_safety_setting': current_user.last_safety_setting or "default",
            'enable_e2ee': current_user.enable_e2ee,
            'migration_status': mig_status,
            'migration_progress': mig_progress,
            'is_2fa_enabled': current_user.is_2fa_enabled,
            'has_totp': has_totp,
            'has_webauthn': has_webauthn,
            'passkey_credentials': _serialize_public_webauthn_credentials(passkeys),
            'passkey_count': len(passkeys),
            'passkey_only_login': current_user.passkey_only_login,
            'skip_2fa_on_google_login': current_user.skip_2fa_on_google_login,
            'default_2fa_method': current_user.default_2fa_method or 'totp',
            'bot_detection_enabled': current_user.bot_detection_enabled if current_user.bot_detection_enabled is not None else True,
            'bot_detection_global_enabled': get_bot_detection_global_enabled(),
            'is_bot_banned': current_user.is_bot_banned,
            'bot_ban_reason': current_user.bot_ban_reason,
            'enable_latency_metrics': current_user.enable_latency_metrics if current_user.enable_latency_metrics is not None else False,
            'enable_client_debug_log': current_user.enable_client_debug_log if current_user.enable_client_debug_log is not None else False
        }
        if getattr(current_user, 'is_admin', False):
            payload['admin_api_key_mode'] = _normalize_admin_api_key_mode(current_user.admin_api_key_mode)
        return jsonify(payload)
    d = request.json
    if 'system_prompt' in d: 
        if current_user.enable_e2ee: current_user.system_prompt = encrypt_val(d['system_prompt'])
        else: current_user.system_prompt = d['system_prompt']
    if 'system_prompt_enabled' in d:
        current_user.system_prompt_enabled = bool(d['system_prompt_enabled'])
    if 'apply_global_system_prompt' in d:
        current_user.apply_global_system_prompt = bool(d['apply_global_system_prompt'])
    if 'apply_auto_system_prompt_notices' in d:
        current_user.apply_auto_system_prompt_notices = bool(d['apply_auto_system_prompt_notices'])
    if 'auto_system_prompt_notices_config' in d:
        set_user_auto_system_prompt_notices_config(current_user, d.get('auto_system_prompt_notices_config'))
    if 'openai_key' in d: current_user.openai_api_key = encrypt_val(d['openai_key'])
    if 'gemini_key' in d: current_user.gemini_api_key = encrypt_val(d['gemini_key'])
    if 'deepseek_key' in d: current_user.deepseek_api_key = encrypt_val(d['deepseek_key'])
    if 'model_api_keys' in d: _save_user_model_api_key_map(current_user, d.get('model_api_keys'))
    if 'gemini_backend' in d: current_user.gemini_backend = _normalize_gemini_backend(d['gemini_backend'])
    if 'gemini_vertex_project' in d: current_user.gemini_vertex_project = encrypt_val(d['gemini_vertex_project'])
    if 'gemini_vertex_location' in d: current_user.gemini_vertex_location = _normalize_gemini_vertex_location(d['gemini_vertex_location'])
    if 'gemini_vertex_credentials_json' in d:
        try:
            normalized_vertex_json = _normalize_gemini_vertex_credentials_json(d['gemini_vertex_credentials_json'])
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        current_user.gemini_vertex_credentials_json = encrypt_val(normalized_vertex_json)
    if 'xai_key' in d: current_user.xai_api_key = encrypt_val(d['xai_key'])
    if 'google_key' in d: current_user.google_api_key = encrypt_val(d['google_key'])
    if 'google_project' in d: current_user.google_cloud_project = encrypt_val(d['google_project'])
    if 'mic_transcribe_mode' in d:
        current_user.mic_transcribe_mode = _normalize_mic_transcribe_mode(d['mic_transcribe_mode'])
    if 'stt_model' in d: current_user.stt_model = d['stt_model']
    if 'llm_transcribe_prompt' in d:
        current_user.llm_transcribe_prompt = _normalize_llm_transcribe_prompt(d.get('llm_transcribe_prompt'))
    if 'enter_to_send' in d: current_user.enter_to_send = bool(d['enter_to_send'])
    if 'use_sw_cache' in d: current_user.use_sw_cache = bool(d['use_sw_cache'])
    if 'compact_prompt_mode' in d: current_user.compact_prompt_mode = bool(d['compact_prompt_mode'])
    if 'theme_color' in d: current_user.theme_color = normalize_theme_color(d.get('theme_color'))
    if 'auto_search_on_links' in d: current_user.auto_search_on_links = bool(d['auto_search_on_links'])
    if 'use_last_chat_settings' in d: current_user.use_last_chat_settings = bool(d['use_last_chat_settings'])
    if 'default_model' in d: current_user.default_model = d['default_model']
    if 'temp_chat_timeout_seconds' in d:
        current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(
            d.get('temp_chat_timeout_seconds')
        )
    if 'default_enable_search' in d: current_user.default_enable_search = bool(d['default_enable_search'])
    if 'default_enable_url_context' in d: current_user.default_enable_url_context = bool(d['default_enable_url_context'])
    if 'default_enable_maps' in d: current_user.default_enable_maps = bool(d['default_enable_maps'])
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
    if 'rich_paste_prompt_default' in d: current_user.rich_paste_prompt_default = d['rich_paste_prompt_default'] or ""
    if 'rich_paste_prompt_use_custom_default' in d: current_user.rich_paste_prompt_use_custom_default = bool(d['rich_paste_prompt_use_custom_default'])
    if 'passkey_only_login' in d:
        target = bool(d['passkey_only_login'])
        if target:
            creds = _load_user_webauthn_credentials(current_user)
            if not creds:
                return jsonify({'error': 'No passkey registered'}), 400
        current_user.passkey_only_login = target
    if 'skip_2fa_on_google_login' in d:
        current_user.skip_2fa_on_google_login = bool(d['skip_2fa_on_google_login'])
    if 'default_2fa_method' in d:
        current_user.default_2fa_method = str(d['default_2fa_method'])
    if 'bot_detection_enabled' in d and d['bot_detection_enabled'] is not None:
        current_user.bot_detection_enabled = bool(d['bot_detection_enabled'])
    if 'enable_latency_metrics' in d:
        current_user.enable_latency_metrics = bool(d['enable_latency_metrics'])
    if 'enable_client_debug_log' in d:
        current_user.enable_client_debug_log = bool(d['enable_client_debug_log'])
        log_force(f"SETTINGS-UPDATE: user={current_user.id} enable_client_debug_log={current_user.enable_client_debug_log}")
    if getattr(current_user, 'is_admin', False) and 'admin_api_key_mode' in d:
        current_user.admin_api_key_mode = _normalize_admin_api_key_mode(d['admin_api_key_mode'])
    if getattr(current_user, 'is_admin', False) and 'bot_detection_global_enabled' in d:
        set_app_setting("bot_detection_global_enabled", "1" if d['bot_detection_global_enabled'] else "0")
    
    log_force(f"DEBUG: handle_settings processing extra fields, d={d}")
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
        log_force("DEBUG: handle_settings calling _refresh_user_2fa_state")
        _refresh_user_2fa_state(current_user)
        log_force("DEBUG: handle_settings calling safe_db_commit")
        safe_db_commit()
        log_force("DEBUG: handle_settings safe_db_commit finished")
        flash("設定を保存しました")
    log_force("DEBUG: handle_settings returning ok")
    return jsonify({'status': 'ok'})

@app.route('/api/debug/client_log', methods=['POST'])
@login_required
def receive_client_log():
    if not getattr(current_user, 'enable_client_debug_log', False):
        return jsonify({'status': 'ignored', 'reason': 'disabled'}), 200
    try:
        d = request.get_json(silent=True) or {}
        level = str(d.get('level') or 'info').upper()
        msg = str(d.get('message') or '')
        if not msg:
            return jsonify({'status': 'ignored', 'reason': 'empty'}), 200
        log_force(f"CLIENT-DEBUG [{level}]: {msg}")
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_force(f"CLIENT-DEBUG-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 400

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
    existing = _load_user_webauthn_credentials(current_user)
    options_kwargs = {
        "rp_name": "AI Chat Playground",
        "rp_id": request.host.split(':')[0],
        "user_id": str(current_user.id).encode(),
        "user_name": current_user.username,
        "authenticator_selection": AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED
        )
    }
    if existing:
        options_kwargs["exclude_credentials"] = [
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c['id'])) for c in existing
        ]
    options = generate_registration_options(**options_kwargs)
    session['webauthn_reg_challenge'] = base64.b64encode(options.challenge).decode('utf-8')
    return options_to_json(options)

@app.route('/api/2fa/webauthn/register/verify', methods=['POST'])
@login_required
def webauthn_reg_verify():
    try:
        data = request.json or {}
        challenge = session.get('webauthn_reg_challenge')
        if not challenge: return jsonify({'error': 'Challenge missing'}), 400
        
        verification = verify_registration_response(
            credential=data,
            expected_challenge=base64.b64decode(challenge),
            expected_rp_id=request.host.split(':')[0],
            expected_origin=request.url_root.rstrip('/'),
            require_user_verification=False 
        )
        
        creds = _load_user_webauthn_credentials(current_user)
        cred_id = base64.b64encode(verification.credential_id).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
        cred_name = str(data.get('name') or '').strip()
        if not cred_name:
            cred_name = f"Security Key {len(creds) + 1}"
        existing = next((c for c in creds if c['id'] == cred_id), None)
        if existing:
            existing['public_key'] = base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('=')
            existing['sign_count'] = verification.sign_count
            existing['name'] = cred_name
        else:
            creds.append({
                'id': cred_id,
                'public_key': base64.b64encode(verification.credential_public_key).decode('utf-8').replace('+', '-').replace('/', '_').rstrip('='),
                'sign_count': verification.sign_count,
                'name': cred_name,
                'created_at': datetime.utcnow().isoformat() + "Z"
            })
        _save_user_webauthn_credentials(current_user, creds)
        current_user.is_2fa_enabled = True
        session.pop('webauthn_reg_challenge', None)
        safe_db_commit()
        return jsonify({'status': 'ok', 'passkey_credentials': _serialize_public_webauthn_credentials(creds)})
    except Exception as e:
        logger.error(f"WebAuthn Reg Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/2fa/webauthn/remove', methods=['POST'])
@login_required
def webauthn_remove():
    data = request.json or {}
    cred_id = str(data.get('id') or '').strip()
    if not cred_id:
        return jsonify({'error': 'id_required'}), 400
    creds = _load_user_webauthn_credentials(current_user)
    filtered = [c for c in creds if c['id'] != cred_id]
    if len(filtered) == len(creds):
        return jsonify({'error': 'not_found'}), 404
    _save_user_webauthn_credentials(current_user, filtered)
    _refresh_user_2fa_state(current_user)
    safe_db_commit()
    return jsonify({
        'status': 'ok',
        'has_webauthn': bool(filtered),
        'passkey_only_login': bool(current_user.passkey_only_login),
        'is_2fa_enabled': bool(current_user.is_2fa_enabled),
        'passkey_count': len(filtered)
    })

@app.route('/api/gems', methods=['GET', 'POST'])
@login_required
def handle_gems():
    if request.method == 'GET':
        gems = Gem.query.filter_by(user_id=current_user.id).order_by(Gem.created_at.desc()).all()
        return jsonify([{'id': g.id, 'name': g.name, 'description': g.description, 'instruction': g.instruction, 'fixed_prompts': g.fixed_prompts_json} for g in gems])
    d = request.json
    gem = Gem(user_id=current_user.id, name=d.get('name', 'My Gem'), description=d.get('description', ''), instruction=d.get('instruction', ''), fixed_prompts_json=d.get('fixed_prompts'))
    db.session.add(gem)
    safe_db_commit()
    return jsonify({'id': gem.id, 'name': gem.name})

@app.route('/api/gems/<int:gid>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def handle_gem_item(gid):
    gem = Gem.query.get_or_404(gid)
    if gem.user_id != current_user.id: return jsonify({'error': '403'}), 403

    if request.method == 'GET':
        return jsonify({'id': gem.id, 'name': gem.name, 'description': gem.description, 'instruction': gem.instruction, 'fixed_prompts': gem.fixed_prompts_json})

    if request.method == 'PUT':
        d = request.json
        gem.name = d.get('name', gem.name)
        gem.description = d.get('description', gem.description)
        gem.instruction = d.get('instruction', gem.instruction)
        gem.fixed_prompts_json = d.get('fixed_prompts', gem.fixed_prompts_json)
        safe_db_commit()
        return jsonify({'id': gem.id, 'name': gem.name})
    if request.method == 'DELETE':
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
        if not g_key and _admin_env_fallback_enabled(current_user):
            g_key = os.getenv('GOOGLE_API_KEY')
        if not g_key:
            return jsonify({'error': 'Google API Key not configured (Google Cloud API key required)'}), 400
        
        g_project = decrypt_val(current_user.google_cloud_project)
        if not g_project and _admin_env_fallback_enabled(current_user):
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

    try:
        transcribe_mode = _normalize_mic_transcribe_mode(getattr(current_user, 'mic_transcribe_mode', None))
        if transcribe_mode == "llm":
            req_data = request.form if request.form else (request.json or {})
            llm_model_key = (req_data.get('llm_model') or req_data.get('model') or "").strip()
            transcript = _transcribe_audio_with_llm(audio_content, fname, llm_model_key, current_user)
            return jsonify({'transcript': transcript, 'mode': 'llm'})

        allowed_models = {
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "gpt-4o-transcribe-diarize",
            "whisper-1"
        }
        model = (current_user.stt_model or "").strip()
        if model not in allowed_models:
            model = "gpt-4o-mini-transcribe"
        key = _get_model_specific_api_key(current_user, model) or decrypt_val(current_user.openai_api_key)
        if not key and _admin_env_fallback_enabled(current_user):
            key = os.getenv('OPENAI_API_KEY')
        if not key:
            return jsonify({'error': 'OpenAI API Key not configured'}), 400

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

        return jsonify({'transcript': transcript, 'mode': 'stt_api'})
    except ValueError as e:
        logger.warning(f"Transcription validation failed: {e}")
        return jsonify({'error': str(e)}), 400
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
    sts_thinking_level = request.form.get('sts_thinking_level')
    sts_include_thoughts = request.form.get('sts_include_thoughts') == 'true'
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

    model_specific_key = _get_model_specific_api_key(current_user, model_key)

    if provider == "google":
        gemini_runtime = _resolve_gemini_runtime(current_user)
        key = model_specific_key or gemini_runtime.get("api_key")
        if gemini_runtime.get("backend") == "vertex_ai":
            if not gemini_runtime.get("vertex_project"):
                return jsonify({'error': 'Vertex AI Project ID not configured'}), 400
        elif not key:
            return jsonify({'error': 'Gemini API Key not configured'}), 400

        def generate_sts_stream():
            assistant_audio = bytearray()
            assistant_text = ""
            assistant_thought = ""
            input_text = ""
            try:
                # Yield a processing status immediately
                yield json.dumps({'status': 'processing'}) + "\n"

                # Move conversion inside the stream for faster response start
                src_ext = os.path.splitext(secure_filename(f.filename))[1].lower() or ".webm"
                pcm_bytes = _convert_audio_to_pcm(audio_bytes, src_ext, rate=rate_in)

                # Use a small buffer for audio chunks to send to client
                audio_buffer = bytearray()
                
                # Consume the generator
                gen = _google_sts_live(
                    pcm_bytes,
                    model_key,
                    gemini_api_key=key,
                    gemini_backend=gemini_runtime.get("backend"),
                    gemini_vertex_project=gemini_runtime.get("vertex_project"),
                    gemini_vertex_location=gemini_runtime.get("vertex_location"),
                    gemini_vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
                    rate=rate_in,
                    voice=sts_voice,
                    thinking_level=sts_thinking_level,
                    include_thoughts=sts_include_thoughts
                )
                
                # Iterate over the async generator using an event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def run_gen():
                    nonlocal assistant_text, assistant_thought, input_text
                    async for audio_chunk, transcript_delta, input_delta, thought_delta, turn_complete in gen:
                        if audio_chunk:
                            assistant_audio.extend(audio_chunk)
                            audio_buffer.extend(audio_chunk)
                        if transcript_delta: assistant_text += transcript_delta
                        if input_delta: input_text += input_delta
                        if thought_delta: assistant_thought += thought_delta
                        
                        # Send chunks if we have enough audio or text updates
                        if len(audio_buffer) >= 1000 or transcript_delta or thought_delta or turn_complete:
                            payload = {}
                            if audio_buffer:
                                payload['audio_delta'] = base64.b64encode(audio_buffer).decode('utf-8')
                                audio_buffer.clear()
                            if transcript_delta: payload['transcript_delta'] = transcript_delta
                            if thought_delta: payload['thought_delta'] = thought_delta
                            if input_delta: payload['input_delta'] = input_delta
                            if turn_complete: payload['turn_complete'] = True
                            
                            if payload:
                                yield json.dumps(payload) + "\n"

                # Convert async generator to sync generator for Flask
                it = run_gen().__aiter__()
                while True:
                    try:
                        yield loop.run_until_complete(it.__anext__())
                    except StopAsyncIteration:
                        break
                
                # After stream ends, save to DB
                if assistant_audio:
                    wav_bytes = _pcm_to_wav_bytes(bytes(assistant_audio), rate=rate_out)
                    out_fname, _ = _save_user_audio(current_user.id, wav_bytes, ".wav", current_user.enable_e2ee)
                    audio_url = f"/files/{current_user.id}/{out_fname}"
                    
                    in_fname = None
                    try:
                        in_suffix = src_ext if src_ext.startswith('.') else f".{src_ext}"
                        in_fname, _ = _save_user_audio(current_user.id, audio_bytes, in_suffix, current_user.enable_e2ee)
                    except Exception: pass

                    user_text = (input_text or "Voice message").strip()
                    assistant_text_clean = (assistant_text or "").strip()
                    assistant_thought_clean = (assistant_thought or "").strip()
                    
                    thought_tag = f"<thought>\n{assistant_thought_clean}\n</thought>\n" if assistant_thought_clean else ""
                    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
                    assistant_content = thought_tag + (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

                    # DB Save Logic
                    try:
                        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
                        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
                        user_tokens_in = count_tokens_for_display(user_text, model_key)
                        assistant_tokens_out = count_tokens_for_display(assistant_text_clean, model_key)
                        if assistant_thought_clean:
                            assistant_tokens_out += count_tokens_for_display(assistant_thought_clean, model_key)
                        
                        parent_id = None
                        last_msg = Message.query.filter_by(thread_id=thread_id).order_by(Message.id.desc()).first()
                        if last_msg: parent_id = last_msg.id

                        user_msg = Message(
                            thread_id=thread_id,
                            role='user',
                            content=u_content,
                            image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
                            is_encrypted=current_user.enable_e2ee,
                            parent_id=parent_id,
                            model=model_key,
                            tokens_in=user_tokens_in,
                            tokens=sum_token_counts(user_tokens_in, None)
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
                            tokens_out=assistant_tokens_out,
                            tokens=sum_token_counts(None, assistant_tokens_out)
                        )
                        db.session.add(assistant_msg)
                        safe_db_commit()
                        
                        # Send final metadata
                        yield json.dumps({
                            'final': True,
                            'audio_url': audio_url,
                            'transcript': assistant_text_clean,
                            'thought': assistant_thought_clean,
                            'input_transcript': user_text
                        }) + "\n"
                    except Exception as e:
                        logger.error(f"STS stream message save failed: {e}")
            except Exception as e:
                logger.error(f"STS stream failed: {e}")
                yield json.dumps({'error': str(e)}) + "\n"
            finally:
                loop.close()

        resp = Response(stream_with_context(generate_sts_stream()), content_type='application/x-ndjson')
        resp.headers['X-Accel-Buffering'] = 'no'
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    # Original sync logic for OpenAI/xAI
    assistant_audio = b""
    assistant_text = ""
    assistant_thought = ""
    input_text = ""
    try:
        if provider == "openai":
            key = model_specific_key or decrypt_val(current_user.openai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('OPENAI_API_KEY')
            if not key:
                return jsonify({'error': 'OpenAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _openai_sts_realtime(pcm_bytes, key, model_key, voice=sts_voice, speed=sts_speed, rate=rate_out)
            )
        elif provider == "xai":
            key = model_specific_key or decrypt_val(current_user.xai_api_key)
            if not key and _admin_env_fallback_enabled(current_user):
                key = os.getenv('XAI_API_KEY')
            if not key:
                return jsonify({'error': 'xAI API Key not configured'}), 400
            assistant_audio, assistant_text = asyncio.run(
                _xai_sts_realtime(pcm_bytes, key, model_key=model_key, voice=sts_voice, rate_in=rate_in, rate_out=rate_out)
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
    assistant_thought_clean = (assistant_thought or "").strip()
    
    thought_tag = f"<thought>\n{assistant_thought_clean}\n</thought>\n" if assistant_thought_clean else ""
    audio_tag = f'\n<audio controls src="{audio_url}" class="w-full mt-2"></audio>\n'
    assistant_content = thought_tag + (assistant_text_clean + "\n" if assistant_text_clean else "") + audio_tag

    try:
        u_content = encrypt_val(user_text) if current_user.enable_e2ee else user_text
        a_content = encrypt_val(assistant_content) if current_user.enable_e2ee else assistant_content
        user_tokens_in = count_tokens_for_display(user_text, model_key)
        assistant_tokens_out = count_tokens_for_display(assistant_text_clean, model_key)
        # Add thought tokens to assistant tokens out if possible
        if assistant_thought_clean:
            assistant_tokens_out += count_tokens_for_display(assistant_thought_clean, model_key)
        
        user_msg = Message(
            thread_id=thread_id,
            role='user',
            content=u_content,
            image_url=json.dumps([f"{current_user.id}/{in_fname}"]) if in_fname else None,
            is_encrypted=current_user.enable_e2ee,
            parent_id=parent_id,
            model=model_key,
            tokens_in=user_tokens_in,
            tokens=sum_token_counts(user_tokens_in, None)
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
            tokens_out=assistant_tokens_out,
            tokens=sum_token_counts(None, assistant_tokens_out)
        )
        db.session.add(assistant_msg)
        safe_db_commit()
    except Exception as e:
        logger.error(f"STS message save failed: {e}")

    return jsonify({
        'audio_url': audio_url,
        'transcript': assistant_text_clean,
        'thought': assistant_thought_clean,
        'input_transcript': user_text,
        'filename': f"{current_user.id}/{out_fname}"
    })

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mov', '.mkv', '.avi', '.m4v'}
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
    cache_updated = False
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
            rel_path = f"{current_user.id}/{fname}"
            res.append(rel_path)
            try:
                disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
                size = None
                mtime = None
                try:
                    size = os.path.getsize(disk_path)
                except Exception:
                    size = None
                try:
                    mtime = int(os.path.getmtime(disk_path))
                except Exception:
                    mtime = None
                mime_guess = mimetypes.guess_type(fname)[0]
                mime = _normalize_media_mime(fname, mime_guess)
                _upsert_file_cache(
                    current_user.id,
                    rel_path,
                    "local",
                    size_bytes=size,
                    mtime=mtime,
                    mime_type=mime,
                    state="stored",
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
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/upload/init', methods=['POST'])
@login_required
def upload_init():
    data = request.json or {}
    filename = secure_filename((data.get('filename') or '').strip())
    total_size = int(data.get('size') or 0)
    if not filename or total_size <= 0:
        return jsonify({'error': 'Invalid upload'}), 400

    allowed = {'.txt', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mov', '.mkv', '.avi', '.m4v'}
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
    cache_updated = False
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
        rel_path = f"{current_user.id}/{fname}"
        try:
            disk_path = os.path.join(ud, fname + '.enc') if current_user.enable_e2ee else os.path.join(ud, fname)
            size = None
            mtime = None
            try:
                size = os.path.getsize(disk_path)
            except Exception:
                size = None
            try:
                mtime = int(os.path.getmtime(disk_path))
            except Exception:
                mtime = None
            mime_guess = mimetypes.guess_type(fname)[0]
            mime = _normalize_media_mime(fname, mime_guess)
            _upsert_file_cache(
                current_user.id,
                rel_path,
                "local",
                size_bytes=size,
                mtime=mtime,
                mime_type=mime,
                state="stored",
                last_error=None
            )
            cache_updated = True
        except Exception:
            pass
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
    if cache_updated:
        try:
            safe_db_commit()
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
        ensure_thread_temporary_column()
    except Exception:
        pass
    try:
        ensure_message_token_io_columns()
    except Exception:
        pass
    try:
        ensure_user_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_user_gemini_backend_columns()
    except Exception:
        pass
    try:
        ensure_user_deepseek_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_admin_api_key_mode_column()
    except Exception:
        pass
    try:
        ensure_user_2fa_default_columns()
    except Exception:
        pass
    try:
        ensure_user_model_api_keys_column()
    except Exception:
        pass
    try:
        ensure_user_temp_chat_timeout_column()
    except Exception:
        pass
    try:
        ensure_user_compact_prompt_mode_column()
    except Exception:
        pass
    try:
        ensure_gem_fixed_prompts_column()
    except Exception:
        pass
    try:
        ensure_user_stt_settings_columns()
    except Exception:
        pass
    try:
        ensure_chat_latency_trace_columns()
    except Exception:
        pass
    try:
        ensure_user_debug_settings_columns()
    except Exception:
        pass
    try:
        ensure_user_default_model_columns()
    except Exception:
        pass
    try:
        ensure_user_google_columns()
    except Exception:
        pass
    try:
        cleanup_user_temp_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_performance_indexes()
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
            try_alter("ALTER TABLE message ADD COLUMN tokens_in INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_out INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_thought INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_e2ee BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN is_encrypted BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_client_debug_log BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_done_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_total_latency_ms INTEGER")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN xai_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_backend VARCHAR(24) DEFAULT 'gemini_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_location VARCHAR(64) DEFAULT 'global'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_credentials_json TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN deepseek_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN admin_api_key_mode VARCHAR(24) DEFAULT 'env_fallback'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN model_api_keys TEXT")
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
            try_alter("ALTER TABLE user ADD COLUMN skip_2fa_on_google_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_2fa_method VARCHAR(16) DEFAULT 'totp'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN mic_transcribe_mode VARCHAR(16) DEFAULT 'stt_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN llm_transcribe_prompt TEXT")
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
            try_alter("ALTER TABLE user ADD COLUMN compact_prompt_mode BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_last_chat_settings BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter(f"ALTER TABLE user ADD COLUMN temp_chat_timeout_seconds INTEGER DEFAULT {_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS}")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_maps BOOLEAN DEFAULT 0")
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
            try_alter("ALTER TABLE user ADD COLUMN system_prompt_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_global_system_prompt BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_auto_system_prompt_notices BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_system_prompt_notices_config TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_default TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_use_custom_default BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_maps BOOLEAN DEFAULT 0")
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
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_temporary BOOLEAN DEFAULT 0")
        except: pass

@app.route('/api/metrics/first_token', methods=['POST'])
@login_required
def first_token_metric():
    if not rate_limit(f"rl:first_token_metric:user:{current_user.id}", 240, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}

        try:
            latency_seconds = float(d.get('latency_seconds'))
        except Exception:
            return jsonify({'error': 'latency_seconds is required'}), 400
        if latency_seconds != latency_seconds or latency_seconds < 0 or latency_seconds > 600:
            return jsonify({'error': 'latency_seconds out of range'}), 400

        latency_ms = _coerce_int_or_none(d.get('latency_ms'))
        if latency_ms is None:
            latency_ms = int(round(latency_seconds * 1000))
        if latency_ms < 0:
            latency_ms = 0

        thread_public_id = str(d.get('thread_id') or '').strip()[:64] or None
        job_id = str(d.get('job_id') or '').strip()[:64] or None
        model = str(d.get('model') or '').strip()[:80] or None
        first_event_type = str(d.get('first_event_type') or '').strip()[:32] or None

        client_sent_at = None
        client_sent_at_ms = _coerce_int_or_none(d.get('client_sent_at_ms'))
        if client_sent_at_ms is not None and 946684800000 <= client_sent_at_ms <= 4102444800000:
            client_sent_at = datetime.utcfromtimestamp(client_sent_at_ms / 1000.0)

        is_total = bool(d.get('is_total'))
        client_done_at_ms = _coerce_int_or_none(d.get('client_done_at_ms'))

        if is_total:
            log_force(f"RECEIVED-TOTAL-REPORT: job={job_id} model={model} latency_ms={latency_ms}")
            # If it's a total latency report, we primarily update the trace
            _upsert_chat_latency_trace(
                job_id=job_id,
                user_id=current_user.id,
                thread_public_id=thread_public_id,
                model=model,
                client_sent_at_ms=client_sent_at_ms,
                client_done_at_ms=client_done_at_ms,
                client_total_latency_ms=latency_ms
            )
            log_force(
                "TOTAL-LATENCY-METRIC: "
                f"user={current_user.id} "
                f"model={model or '-'} "
                f"thread={thread_public_id or '-'} "
                f"job={job_id or '-'} "
                f"total_seconds={latency_seconds:.3f} "
                f"client_done_at={client_done_at_ms or '-'}"
            )
            return jsonify({'status': 'ok', 'type': 'total'})

        row = FirstTokenLatencyMetric(
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            job_id=job_id,
            model=model,
            first_event_type=first_event_type,
            latency_seconds=round(latency_seconds, 6),
            latency_ms=latency_ms,
            client_sent_at=client_sent_at,
            ip_address=get_client_ip(),
            user_agent=request.headers.get('User-Agent', '')
        )
        db.session.add(row)
        safe_db_commit()
        trace = _upsert_chat_latency_trace(
            job_id=job_id,
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            model=model,
            client_sent_at_ms=client_sent_at_ms,
            client_first_event_type=first_event_type,
            client_first_latency_ms=latency_ms
        )

        window_start = datetime.utcnow() - timedelta(hours=24)
        stats = db.session.query(
            func.count(FirstTokenLatencyMetric.id),
            func.avg(FirstTokenLatencyMetric.latency_seconds),
            func.min(FirstTokenLatencyMetric.latency_seconds),
            func.max(FirstTokenLatencyMetric.latency_seconds)
        ).filter(
            FirstTokenLatencyMetric.user_id == current_user.id,
            FirstTokenLatencyMetric.created_at >= window_start
        ).first()
        stats_evt = None
        if first_event_type:
            stats_evt = db.session.query(
                func.count(FirstTokenLatencyMetric.id),
                func.avg(FirstTokenLatencyMetric.latency_seconds),
                func.min(FirstTokenLatencyMetric.latency_seconds),
                func.max(FirstTokenLatencyMetric.latency_seconds)
            ).filter(
                FirstTokenLatencyMetric.user_id == current_user.id,
                FirstTokenLatencyMetric.first_event_type == first_event_type,
                FirstTokenLatencyMetric.created_at >= window_start
            ).first()

        cnt = int((stats[0] or 0)) if stats else 0
        avg_s = float(stats[1]) if stats and stats[1] is not None else latency_seconds
        min_s = float(stats[2]) if stats and stats[2] is not None else latency_seconds
        max_s = float(stats[3]) if stats and stats[3] is not None else latency_seconds
        evt_cnt = int((stats_evt[0] or 0)) if stats_evt else 0
        evt_avg_s = float(stats_evt[1]) if stats_evt and stats_evt[1] is not None else latency_seconds
        evt_min_s = float(stats_evt[2]) if stats_evt and stats_evt[2] is not None else latency_seconds
        evt_max_s = float(stats_evt[3]) if stats_evt and stats_evt[3] is not None else latency_seconds
        phase_parts = []
        if trace:
            phase_candidates = {
                "client_to_route_ms": _trace_delta_ms(trace, "client_sent_at", "route_received_at"),
                "route_to_dispatch_ms": _trace_delta_ms(trace, "route_received_at", "route_dispatch_at"),
                "dispatch_to_worker_ms": _trace_delta_ms(trace, "route_dispatch_at", "worker_started_at"),
                "worker_to_provider_req_ms": _trace_delta_ms(trace, "worker_started_at", "provider_request_started_at"),
                "provider_req_to_first_chunk_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_chunk_at"),
                "provider_req_to_first_content_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_content_at"),
                "provider_content_to_client_ms": _trace_delta_ms(trace, "provider_first_content_at", "stream_first_content_to_client_at"),
                "route_to_client_content_ms": _trace_delta_ms(trace, "route_received_at", "stream_first_content_to_client_at"),
            }
            for key, val in phase_candidates.items():
                if val is not None:
                    phase_parts.append(f"{key}={val}")
        log_force(
            "FIRST-TOKEN-METRIC: "
            f"user={current_user.id} "
            f"model={model or '-'} "
            f"thread={thread_public_id or '-'} "
            f"job={job_id or '-'} "
            f"event={first_event_type or '-'} "
            f"seconds={latency_seconds:.3f} "
            f"window24h(count={cnt},avg={avg_s:.3f},min={min_s:.3f},max={max_s:.3f}) "
            f"event24h(count={evt_cnt},avg={evt_avg_s:.3f},min={evt_min_s:.3f},max={evt_max_s:.3f}) "
            f"path={getattr(trace, 'execution_path', '-') or '-'} "
            f"phases({','.join(phase_parts)})"
        )

        return jsonify({
            'status': 'ok',
            'latency_seconds': round(latency_seconds, 3),
            'window24h': {
                'count': cnt,
                'avg_seconds': round(avg_s, 3),
                'min_seconds': round(min_s, 3),
                'max_seconds': round(max_s, 3),
            },
            'event24h': {
                'event': first_event_type or None,
                'count': evt_cnt,
                'avg_seconds': round(evt_avg_s, 3),
                'min_seconds': round(evt_min_s, 3),
                'max_seconds': round(evt_max_s, 3),
            },
            'execution_path': getattr(trace, 'execution_path', None) if trace else None
        })
    except Exception as e:
        db.session.rollback()
        log_force(f"FIRST-TOKEN-METRIC-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 500

@app.route('/api/client_log', methods=['POST'])
@login_required
def client_log():
    if not getattr(current_user, 'enable_client_debug_log', False):
        return jsonify({'status': 'ignored', 'reason': 'disabled'}), 200
    if not rate_limit(f"rl:client_log:user:{current_user.id}", 60, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}
        level = str(d.get('level') or 'info').upper()
        msg = str(d.get('message') or '')
        if not msg:
            return jsonify({'status': 'ignored', 'reason': 'empty'}), 200
        log_force(f"CLIENT-DEBUG [LEGACY {level}]: {msg}")
        return jsonify({'status': 'ok'})
    except Exception:
        return jsonify({'status': 'error'}), 500

@app.errorhandler(403)
def handle_forbidden(_error):
    return render_template("403.html"), 403

@app.errorhandler(404)
def handle_not_found(_error):
    return render_template("404.html"), 404

if __name__ == '__main__':
    log_force("DEBUG: App starting in main")
    app.run(debug=True)
else:
    log_force("DEBUG: App imported/starting in worker or gunicorn")

# Pre-warm common token encoders to avoid first-call latency in workers
try:
    for m in ("gpt-4o", "gemini-1.5-pro"):
        _get_token_encoder(m)
except:
    pass
