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
import math
import pyotp
import qrcode
import wave
import asyncio
import tempfile
import zipfile
import warnings
import cairosvg
from lxml import etree as _LXML
from defusedxml import ElementTree as ET
from urllib.parse import urlparse, unquote, quote, urlencode
import threading
import queue as _queue
import hashlib
import socket
import difflib
import itertools
from typing import Optional
from contextlib import contextmanager
from ipaddress import ip_address
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
from flask.sessions import SecureCookieSessionInterface
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, exc, text, func, inspect
from sqlalchemy.dialects.mysql import LONGTEXT
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
from cryptography.fernet import Fernet, InvalidToken
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from authlib.integrations.flask_client import OAuth

Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

try:
    from anthropic import Anthropic, APIError as AnthropicAPIError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    AnthropicAPIError = None
    ANTHROPIC_AVAILABLE = False

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
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.WARNING)
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
_LOW_LATENCY_IMAGE_MAX_ITEMS = max(1, _env_int("LOW_LATENCY_IMAGE_MAX_ITEMS", 4))
_LOW_LATENCY_IMAGE_MAX_BYTES = max(1, _env_int("LOW_LATENCY_IMAGE_MAX_MB", 12)) * 1024 * 1024
BROWSER_FAST_HISTORY_IMAGE_MAX_ITEMS = 4
BROWSER_FAST_HISTORY_IMAGE_MAX_BYTES = 12 * 1024 * 1024
_GEMINI_INLINE_IMAGE_MAX_BYTES = max(1, _env_int("GEMINI_INLINE_IMAGE_MAX_MB", 12)) * 1024 * 1024
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

def _deepseek_api_model_id(model_key):
    """Map app-facing DeepSeek release IDs to the stable official API alias."""
    mk = str(model_key or "").strip()
    if mk.lower() == "deepseek-v4-flash-0731":
        return "deepseek-v4-flash"
    return mk

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
    try:
        if "DEADLINE_EXCEEDED" in str(err) or "504" in str(err):
            return (
                "Gemini APIの応答が制限時間を超えました（504 DEADLINE_EXCEEDED）。"
                "Pythonコード実行時は、実行環境の制限時間（最大約30秒）を超えるとこのエラーが発生します。"
                "処理を簡潔にするか、画像を小さくしてから再試行してください。"
                "改善しない場合はPythonを無効にして再度お試しください。"
            )
    except Exception:
        pass
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
_SECRET_MASK = "********"

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

def _merge_masked_model_api_key_map(user, raw):
    submitted = _normalize_model_api_key_map(raw)
    existing = _load_user_model_api_key_map(user)
    merged = {}
    for model_key, value in submitted.items():
        if value == _SECRET_MASK:
            value = existing.get(model_key)
        if value:
            merged[model_key] = value
    return _save_user_model_api_key_map(user, merged)

def _masked_secret(value):
    return _SECRET_MASK if value else ""

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
    # Carry an existing per-model key forward when the stable Flash alias is
    # represented by the dated app-facing release ID.
    if mk_l == "deepseek-v4-flash-0731":
        legacy_value = key_map.get("deepseek-v4-flash")
        if legacy_value and str(legacy_value).strip():
            return str(legacy_value).strip()
    return None

def _resolve_chat_model_auth(user, model_key):
    """Resolve chat credentials exactly as the worker does, without calling a provider."""
    mk = str(model_key or "").strip()
    mk_l = mk.lower()
    model_key_override = _get_model_specific_api_key(user, mk)

    def user_or_admin_env(field_name, env_name):
        value = decrypt_val(getattr(user, field_name, None)) if user else None
        if value and str(value).strip():
            return str(value).strip()
        if _admin_env_fallback_enabled(user):
            env_value = os.getenv(env_name)
            if env_value and str(env_value).strip():
                return str(env_value).strip()
        return None

    provider = "openai"
    api_key = None
    gemini_runtime = None
    error_code = None
    error_message = None

    if "google-tts" in mk_l:
        provider = "google"
        api_key = model_key_override or user_or_admin_env("google_api_key", "GOOGLE_API_KEY")
    elif is_gemini_model_key(mk_l):
        provider = "gemini"
        gemini_runtime = _resolve_gemini_runtime(user)
        api_key = model_key_override or gemini_runtime.get("api_key")
        if gemini_runtime.get("backend") == "vertex_ai":
            if not gemini_runtime.get("vertex_project"):
                error_code = "provider_configuration_missing"
                error_message = (
                    "Vertex AI Project ID が未設定です。設定で Gemini Backend を "
                    "Vertex AI にした場合は Project ID を入力してください。"
                )
        elif not api_key:
            error_code = "api_key_missing"
            error_message = "Gemini APIキーが設定されていません。"
    elif is_anthropic_model_key(mk_l):
        provider = "anthropic"
        api_key = model_key_override or user_or_admin_env("anthropic_api_key", "ANTHROPIC_API_KEY")
    elif is_deepseek_model_key(mk_l):
        provider = "deepseek"
        api_key = model_key_override or user_or_admin_env("deepseek_api_key", "DEEPSEEK_API_KEY")
    elif "kimi" in mk_l:
        provider = "kimi"
        api_key = model_key_override or user_or_admin_env("kimi_api_key", "MOONSHOT_API_KEY")
    elif "grok" in mk_l and "gpt" not in mk_l:
        provider = "xai"
        api_key = model_key_override or user_or_admin_env("xai_api_key", "XAI_API_KEY")
    elif is_mistral_ocr_model_key(mk_l) or mk_l.startswith("mistral"):
        provider = "mistral"
        api_key = model_key_override or user_or_admin_env("mistral_api_key", "MISTRAL_API_KEY")
    else:
        api_key = model_key_override or user_or_admin_env("openai_api_key", "OPENAI_API_KEY")

    if not error_code and not api_key and not (
        provider == "gemini"
        and gemini_runtime
        and gemini_runtime.get("backend") == "vertex_ai"
    ):
        error_code = "api_key_missing"
        provider_labels = {
            "openai": "OpenAI",
            "gemini": "Gemini",
            "anthropic": "Anthropic",
            "deepseek": "DeepSeek",
            "kimi": "Kimi",
            "xai": "xAI",
            "google": "Google",
            "mistral": "Mistral",
        }
        error_message = f"{provider_labels.get(provider, provider)} APIキーが設定されていません。"

    return {
        "provider": provider,
        "api_key": api_key,
        "gemini_runtime": gemini_runtime,
        "error_code": error_code,
        "error": error_message,
    }

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
# Agentic View / code execution requests run server-side Python and can take
# several minutes. The client timeout is sent to Google as X-Server-Timeout,
# so a short value makes the API abort with 504 DEADLINE_EXCEEDED before the
# sandboxed Python finishes. Use a dedicated, longer deadline for these requests.
_GEMINI_AGENTIC_TIMEOUT_MS = _env_int("GEMINI_AGENTIC_TIMEOUT_MS", 600000)
# Google intermittently returns 504 DEADLINE_EXCEEDED on the initial response of
# code-execution requests before any content is generated. A plain retry usually
# succeeds, so re-pull the first streaming chunk this many times before failing.
_GEMINI_STREAM_DEADLINE_RETRIES = max(0, _env_int("GEMINI_STREAM_DEADLINE_RETRIES", 2))

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

def _is_public_https_url(url):
    try:
        parsed = urlparse(str(url or ''))
        if parsed.scheme.lower() != 'https' or not parsed.hostname or parsed.username or parsed.password:
            return False
        if parsed.port not in (None, 443):
            return False
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        if not addresses:
            return False
        for entry in addresses:
            address = entry[4][0]
            if not ip_address(address).is_global:
                return False
        return True
    except Exception:
        return False

def _download_public_https_bytes(url, max_bytes, timeout=60.0):
    if not _is_public_https_url(url):
        raise ValueError("Unsafe download URL")
    chunks = []
    total = 0
    with httpx.stream('GET', url, timeout=timeout, follow_redirects=False) as response:
        response.raise_for_status()
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_bytes:
            raise ValueError("Download is too large")
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Download is too large")
            chunks.append(chunk)
    return b''.join(chunks)

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
    api_version='v1beta'
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
        sig = f"vertex:{vertex_project}:{vertex_location}:{vertex_creds_sig}:{api_version}"
    else:
        sig = _key_sig(api_key, f"gemini_api:{api_version}")
        if not sig:
            return None
    with _CLIENT_CACHE_LOCK:
        client = _GEMINI_CLIENT_CACHE.get(sig)
        if client:
            return client
        http_options = types.HttpOptions(
            api_version=api_version,
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

class _StaticAssetSessionInterface(SecureCookieSessionInterface):
    def save_session(self, flask_app, session_obj, response):
        # Flask-Login checks the remember-cookie flag during every response,
        # which marks the session as accessed and makes Flask add Vary: Cookie
        # even for public static files. Static handlers never mutate session
        # state, so suppressing the no-op save keeps CDN responses shareable.
        if request.endpoint == 'static':
            return
        return super().save_session(flask_app, session_obj, response)

app.session_interface = _StaticAssetSessionInterface()
app.config['APP_VERSION'] = os.getenv('APP_VERSION', '2026-09-06-002')
app.config['SYSTEM_VERSION'] = 'V4.8.919'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['REMEMBER_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['TRUSTED_HOSTS'] = [
    host.strip() for host in os.getenv('TRUSTED_HOSTS', 'ai.minashin1120.com').split(',') if host.strip()
]
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 280}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'instance/uploads')
app.config['CHANGELOG_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/changelogs')
os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o700, exist_ok=True)
os.chmod(app.config['UPLOAD_FOLDER'], 0o700)
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

# Minashin 中央アカウントシステム（account.minashin1120.com）との
# OAuth 2.0 + PKCE 連携用の設定。client_id は連携サイト自身の Origin
# （= redirect_uri の Origin）で、事前登録は不要（Origin-Based 自動登録）。
MINASHIN_ACCOUNT_BASE_URL = (os.getenv('MINASHIN_ACCOUNT_BASE_URL') or 'https://account.minashin1120.com').rstrip('/')
MINASHIN_REQUEST_TIMEOUT = 10
_PKCE_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)
_CHAT_FAST_QUEUE_NAME = os.getenv("AI_CHAT_FAST_QUEUE", "ai_chat_fast_queue")
_CHAT_HEAVY_QUEUE_NAME = os.getenv("AI_CHAT_HEAVY_QUEUE", "ai_chat_heavy_queue")
chat_fast_queue = Queue(_CHAT_FAST_QUEUE_NAME, connection=redis_conn)
chat_heavy_queue = Queue(_CHAT_HEAVY_QUEUE_NAME, connection=redis_conn)
_LATENCY_TRACE_PREFIX = "latency_trace:"
_LATENCY_TRACE_TTL_SECONDS = max(300, _env_int("LATENCY_TRACE_TTL_SECONDS", 86400))

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
MESSAGE_PAYLOAD_TEXT = db.Text().with_variant(LONGTEXT(), "mysql", "mariadb")

def _sqlite_database_uri(uri):
    return (uri or "").strip().lower().startswith("sqlite:")

def schema_reset_is_allowed(uri=None):
    """True only for sqlite, or an explicit emergency override.

    Regression tests call db.drop_all().  If DATABASE_URL already pointed at
    the production MySQL database, os.environ.setdefault() in those tests was
    a no-op and drop_all wiped live chat/gem rows.  Non-sqlite URLs are
    therefore refused unless ALLOW_PRODUCTION_SCHEMA_RESET=1.
    """
    if os.getenv("ALLOW_PRODUCTION_SCHEMA_RESET") == "1":
        return True
    value = uri if uri is not None else (
        app.config.get("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL") or ""
    )
    return _sqlite_database_uri(value)

def _refuse_protected_schema_reset(op="drop_all"):
    if schema_reset_is_allowed():
        return
    uri = app.config.get("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL") or ""
    location = uri.split("@", 1)[-1] if "@" in uri else "non-sqlite"
    raise RuntimeError(
        f"refusing {op} on {location}. Tests must use a sqlite DATABASE_URL. "
        "Set ALLOW_PRODUCTION_SCHEMA_RESET=1 only for a deliberate emergency reset."
    )

def _install_schema_reset_guard():
    original_drop_all = db.drop_all
    original_metadata_drop_all = db.metadata.drop_all

    def guarded_drop_all(*args, **kwargs):
        _refuse_protected_schema_reset("db.drop_all")
        return original_drop_all(*args, **kwargs)

    def guarded_metadata_drop_all(*args, **kwargs):
        _refuse_protected_schema_reset("metadata.drop_all")
        return original_metadata_drop_all(*args, **kwargs)

    db.drop_all = guarded_drop_all
    db.metadata.drop_all = guarded_metadata_drop_all

_install_schema_reset_guard()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
# Flask-Login's default login_message ("Please log in to access this page.")
# is flashed by unauthorized() whenever an unauthenticated request hits a
# @login_required route.  The login page does not render flashed messages, so
# this English message is never shown where intended -- it only stays in the
# session and leaks onto the chat home screen (#flash-msg) after the user logs
# in, misleadingly saying they need to log in when they already have.
login_manager.login_message = None
login_manager.needs_refresh_message = None


def _exec_server_part(filename):
    """Load a server/*.py slice into this module's globals.

    The historical app.py was split for navigation.  Behavior stays the same
    because each part is exec()'d here, so names remain on the `app` module
    (`from app import User` and gunicorn `app:app` keep working).
    """
    part_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", filename)
    with open(part_path, encoding="utf-8") as handle:
        source = handle.read()
    exec(compile(source, part_path, "exec"), globals())


_SERVER_PARTS = [
    "request_hooks.py",
    "storage.py",
    "crypto.py",
    "providers.py",
    "create_file.py",
    "edit_file.py",
    "agentic_media.py",
    "settings_ai.py",
    "lyria.py",
    "realtime.py",
    "models.py",
    "account_transfer.py",
    "request_identity.py",
    "account_security.py",
    "chat_state.py",
    "app_settings_schema.py",
    "temp_chat.py",
    "request_security.py",
    "token_utils.py",
    "background.py",
    "routes_pages.py",
    "routes_auth.py",
    "routes_chat.py",
    "routes_realtime.py",
    "routes_files.py",
    "rich_paste_pdf.py",
    "routes_threads_library.py",
    "routes_account.py",
    "routes_admin.py",
    "routes_settings.py",
    "routes_media.py",
]
for _server_part in _SERVER_PARTS:
    _exec_server_part(_server_part)

# MCP外部連携（mcp_service）のBlueprint登録
try:
    from mcp_service.web import bp as mcp_service_bp
    app.register_blueprint(mcp_service_bp, url_prefix='/api/mcp')
except Exception as _mcp_bp_err:
    log_force(f"MCP service blueprint registration failed: {_mcp_bp_err}")

@app.errorhandler(403)
def handle_forbidden(_error):
    return render_template("403.html"), 403

@app.errorhandler(404)
def handle_not_found(_error):
    return render_template("404.html"), 404

if __name__ == '__main__':
    log_force("DEBUG: App starting in main")
    app.run(
        host=os.getenv('FLASK_RUN_HOST', '127.0.0.1'),
        port=_env_int('FLASK_RUN_PORT', 5000),
        debug=_env_bool('FLASK_DEBUG', False)
    )
else:
    log_force("DEBUG: App imported/starting in worker or gunicorn")

# Pre-warm common token encoders to avoid first-call latency in workers
try:
    for m in ("gpt-4o", "gemini-1.5-pro"):
        _get_token_encoder(m)
except:
    pass
