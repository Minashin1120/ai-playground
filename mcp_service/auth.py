"""MCP OAuth（認可コード + PKCE）とBearer認証フロー。

- OAuthクライアント情報（Client ID / Secret）はユーザー単位のDB設定
  （``mcp_user_oauth_clients``）から取得する（BYOK。実装プラン §6）。
- コールバックは固定パス ``/api/mcp/oauth/callback`` とし、state / iss を検証する。
- Redis に state と PKCE code_verifier を TTL 付きで保存する。
- タイムアウト・最大リダイレクトは ``config.py`` のコード既定値を使う。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
from datetime import datetime, timedelta

from . import config
from . import security
from .errors import MCPAuthRequiredError, MCPConnectionError, MCPInsufficientScopeError, MCPValidationError
from . import registry as mcp_registry

_STATE_PREFIX = "mcp_oauth_state:"

# Google Workspace MCP 向けの補助情報（discovery 失敗時フォールバック等）
_GOOGLE_DISCOVERY = {
    "issuer": "https://accounts.google.com",
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
}


def _redis():
    import app as _a
    try:
        return _a.redis_conn
    except Exception:
        import redis as _redis
        return _redis.from_url(_a.REDIS_URL)


def _http_get_json(url, *, headers=None, timeout=15.0, extra_query=None):
    import httpx
    security.validate_mcp_url(url, resolve=True)
    params = None
    if extra_query:
        params = extra_query
    try:
        resp = httpx.get(url, params=params, headers=headers or {"Accept": "application/json"},
                         timeout=timeout, follow_redirects=False)
    except Exception as e:
        raise MCPConnectionError(f"Discovery request failed: {e}")
    if resp.status_code >= 400:
        raise MCPConnectionError(f"Discovery request returned HTTP {resp.status_code} for {url}")
    try:
        return resp.json(), resp
    except Exception:
        raise MCPConnectionError(f"Discovery response is not JSON for {url}")


def _discover_protected_resource(server_url):
    """OAuth Protected Resource Metadata（RFC 9728）を発見する。

    失敗時は None を返す（呼び出し側でフォールバック可能）。
    """
    candidates = []
    parsed_base = server_url.rstrip("/")
    candidates.append(f"{parsed_base}/.well-known/oauth-protected-resource")
    try:
        data, resp = _http_get_json(
            candidates[0], timeout=config.MCP_CONNECT_TIMEOUT_SECONDS,
            extra_query={"resource": server_url},
        )
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # サーバー自身へのプローブ（401 の WWW-Authenticate に resource_metadata が載る場合）
    try:
        import httpx
        probe = httpx.get(
            server_url,
            headers={"Accept": "application/json, text/event-stream"},
            timeout=config.MCP_CONNECT_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
        if probe.status_code == 401:
            www = probe.headers.get("www-authenticate") or ""
            if 'resource_metadata="' in www or "resource_metadata=" in www:
                for part in www.split(","):
                    part = part.strip()
                    if part.startswith("resource_metadata="):
                        meta_url = part.split("=", 1)[1].strip().strip('"')
                        if meta_url:
                            try:
                                data, _ = _http_get_json(
                                    meta_url, timeout=config.MCP_CONNECT_TIMEOUT_SECONDS
                                )
                                if isinstance(data, dict):
                                    return data
                            except Exception:
                                return None
            if "Bearer" in www and "scope=" in www:
                # WWW-Authenticate に scope だけ載っているケース
                return {"resource": server_url, "scopes_supported": None,
                        "www_authenticate_scope": _extract_www_scope(www)}
        elif probe.status_code in (200, 202, 204) and probe.status_code < 300:
            try:
                ctype = (probe.headers.get("content-type") or "").lower()
                if "json" in ctype:
                    data = probe.json()
                    if isinstance(data, dict) and "authorization_servers" in data:
                        return data
            except Exception:
                pass
    except Exception:
        pass
    return None


def _extract_www_scope(www):
    import re
    m = re.search(r'scope="([^"]+)"', www)
    return m.group(1) if m else None


def _discover_authorization_server(issuer):
    """Authorization Server Metadata（RFC 8414 / OIDC Discovery）を取得する。"""
    base = str(issuer or "").rstrip("/")
    if not base:
        raise MCPValidationError("Authorization server issuer is missing.")
    security.validate_mcp_url(base, resolve=True)
    # RFC 8414
    for well_known in (
        f"{base}/.well-known/oauth-authorization-server",
        f"{base}/.well-known/openid-configuration",
    ):
        try:
            data, _ = _http_get_json(well_known, timeout=config.MCP_CONNECT_TIMEOUT_SECONDS)
            if isinstance(data, dict) and (
                data.get("authorization_endpoint") or data.get("token_endpoint")
            ):
                return data
        except Exception:
            continue
    raise MCPAuthRequiredError("Authorization server metadata could not be discovered.")


def _provider_extra(provider_key):
    return {"issuer": _GOOGLE_DISCOVERY["issuer"]} if provider_key == "google_workspace" else {}


def _as_metadata_for_server(server):
    """サーバー行から AS metadata を解決する。Google プリセットは既知の補助情報を併用。"""
    from . import registry as reg
    extra = _provider_extra(server.oauth_provider_key_safe)
    meta = None
    try:
        prm = _discover_protected_resource(server.url)
        as_list = []
        if prm and isinstance(prm.get("authorization_servers"), list):
            as_list = [x for x in prm["authorization_servers"] if isinstance(x, str) and x]
        if as_list:
            meta = _discover_authorization_server(as_list[0])
            meta["_protected_resource_metadata"] = prm
    except Exception:
        meta = None
    if meta is None and extra.get("issuer"):
        try:
            meta = _discover_authorization_server(extra["issuer"])
        except Exception:
            meta = None
    if meta is None:
        # 最後のフォールバック（google 固定エンドポイント）
        if extra.get("issuer"):
            meta = dict(_GOOGLE_DISCOVERY)
        else:
            raise MCPAuthRequiredError("OAuth authorization server could not be discovered for this MCP server.")
    if extra.get("issuer") and not meta.get("issuer"):
        meta["issuer"] = extra["issuer"]
    return meta


def _resolve_scope_candidates(server, meta):
    """要求スコープ候補を返す。優先: サーバー推奨 → PRM scopes_supported → 空。"""
    candidates = []
    rec = (server.recommended_scopes or "").strip()
    if rec:
        candidates = rec.split()
    else:
        prm = meta.get("_protected_resource_metadata") or {}
        scopes = prm.get("scopes_supported")
        if isinstance(scopes, list):
            candidates = [str(s) for s in scopes]
        elif isinstance(scopes, str) and scopes.strip():
            candidates = scopes.split()
        www_scope = prm.get("www_authenticate_scope")
        if www_scope and not candidates:
            candidates = www_scope.split()
    # offline_access は AS がサポートする場合のみ
    if config.MCP_OAUTH_OFFLINE_ACCESS:
        supported = meta.get("scopes_supported")
        supported_set = set(supported) if isinstance(supported, list) else set()
        if not supported_set or "offline_access" in supported_set:
            if "offline_access" not in candidates:
                candidates.append("offline_access")
    return candidates


def _existing_scope(user_id, server_id):
    cred = mcp_registry.get_credential(user_id, server_id)
    if cred is None or not cred.scope:
        return []
    return [s for s in str(cred.scope).split() if s]


def build_authorization_url(user_id, server, *, redirect_uri):
    """認可URLを組み立てる。state/code_verifier を Redis へ保存して URL を返す。"""
    oauth_client = mcp_registry.decrypt_oauth_client(user_id, server.oauth_provider_key_safe)
    if oauth_client is None or not oauth_client.get("client_id"):
        raise MCPValidationError(
            "OAuthクライアント情報が未登録です。設定の「MCP」タブから Client ID / Secret を登録してください。",
            detail={"requires_oauth_client": True},
        )
    meta = _as_metadata_for_server(server)
    auth_endpoint = meta.get("authorization_endpoint")
    token_endpoint = meta.get("token_endpoint")
    if not auth_endpoint or not token_endpoint:
        raise MCPAuthRequiredError("Authorization server endpoints could not be determined.")

    candidates = _resolve_scope_candidates(server, meta)
    existing = _existing_scope(user_id, server.id)
    # 既存スコープを失わないよう和集合で要求する
    scope = " ".join(dict.fromkeys(existing + candidates))
    if len(scope.split()) > config.MCP_OAUTH_MAX_SCOPES:
        scope = " ".join(scope.split()[: config.MCP_OAUTH_MAX_SCOPES])

    state = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(48)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")

    params = {
        "response_type": "code",
        "client_id": oauth_client["client_id"],
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if config.MCP_OAUTH_SEND_RESOURCE and server.url:
        params["resource"] = server.url
    # Google では offline アクセストークン（refresh_token）に access_type=offline が必要
    if server.oauth_provider_key_safe == "google_workspace":
        params["access_type"] = "offline"
        params.setdefault("prompt", "consent")

    state_data = {
        "user_id": user_id,
        "server_id": server.id,
        "code_verifier": code_verifier,
        "issuer": meta.get("issuer") or "",
        "scope": scope,
        "resource": server.url,
        "redirect_uri": redirect_uri,
        "token_endpoint": token_endpoint,
        "auth_endpoint": auth_endpoint,
        "authorization_response_iss_parameter_supported": bool(
            meta.get("authorization_response_iss_parameter_supported")
        ),
    }
    # 注意: client_id / client_secret は Redis の state へ保存しない。
    # コールバック時・トークン要求時に DB（mcp_user_oauth_clients）から再取得する。
    _redis().setex(_STATE_PREFIX + state, config.MCP_OAUTH_STATE_TTL_SECONDS, json.dumps(state_data))

    url = auth_endpoint + ("&" if "?" in auth_endpoint else "?") + urllib.parse.urlencode(params)
    return url, state


def _auth_header_for_oauth_client(client_id, client_secret):
    token = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _token_request(state_data, grant_params, use_basic=True):
    import httpx
    url = state_data.get("token_endpoint")
    if not url:
        raise MCPAuthRequiredError("Token endpoint is missing.")
    headers = {"Accept": "application/json"}
    if use_basic and state_data.get("client_secret"):
        headers.update(_auth_header_for_oauth_client(state_data["client_id"], state_data["client_secret"]))
    elif state_data.get("client_secret"):
        grant_params["client_secret"] = state_data["client_secret"]
    else:
        # 公開クライアント（secretなし）
        grant_params["client_id"] = state_data.get("client_id")
    try:
        resp = httpx.post(url, data=grant_params, headers=headers,
                          timeout=config.MCP_CONNECT_TIMEOUT_SECONDS, follow_redirects=False)
    except Exception as e:
        raise MCPConnectionError(f"Token request failed: {e}")
    if resp.status_code >= 400:
        err_text = resp.text[:500]
        if resp.status_code in (401, 403):
            raise MCPAuthRequiredError(f"OAuth token request failed (HTTP {resp.status_code}).")
        if "invalid_grant" in err_text:
            raise MCPAuthRequiredError("OAuth authorization code is invalid or expired.")
        raise MCPConnectionError(f"OAuth token request failed (HTTP {resp.status_code}).")
    try:
        data = resp.json()
    except Exception:
        raise MCPConnectionError("Token response is not JSON.")
    return data


def handle_oauth_callback(query_args):
    """OAuthコールバック処理。state検証・iss検証・トークン交換・保存。

    query_args: コールバックのクエリパラメータ dict。
    戻り値: (server_dict, message_dict) / エラー時 raise
    """
    from app import db  # noqa: F401
    state = query_args.get("state")
    if not state:
        raise MCPValidationError("OAuth callback is missing state.")
    raw = _redis().get(_STATE_PREFIX + state)
    if not raw:
        raise MCPValidationError("OAuth state is missing or expired. Please start authentication again.")
    try:
        state_data = json.loads(raw)
    except Exception:
        raise MCPValidationError("OAuth state is invalid.")
    _redis().delete(_STATE_PREFIX + state)
    user_id = state_data.get("user_id")
    server_id = state_data.get("server_id")

    error = query_args.get("error")
    if error:
        desc = query_args.get("error_description") or ""
        _set_error_state(user_id, server_id, f"OAuth error: {error} {desc}")
        raise MCPAuthRequiredError(f"OAuth authorization failed: {error} {desc}".strip())

    # iss 検証（RFC 9207）
    expected_issuer = state_data.get("issuer") or ""
    advertised = state_data.get("authorization_response_iss_parameter_supported") is True
    iss = query_args.get("iss")
    if advertised and not iss:
        raise MCPAuthRequiredError("OAuth authorization response is missing required iss parameter.")
    if iss:
        if not expected_issuer or iss != expected_issuer:
            raise MCPAuthRequiredError("OAuth issuer mismatch (possible authorization server mix-up).")
    code = query_args.get("code")
    if not code:
        raise MCPAuthRequiredError("OAuth callback is missing authorization code.")

    # Redisのstateへ client_id / secret を入れないため、ここでDBから再取得する
    srv_for_client = mcp_registry.get_server_for_user(user_id, server_id)
    if srv_for_client is None:
        raise MCPAuthRequiredError("MCP server not found for this OAuth state.")
    oauth_client = mcp_registry.decrypt_oauth_client(user_id, srv_for_client.oauth_provider_key_safe)
    if oauth_client is None or not oauth_client.get("client_id"):
        raise MCPValidationError(
            "OAuthクライアント情報が見つかりません。設定の「MCP」タブから Client ID / Secret を登録してください。"
        )
    state_data["client_id"] = oauth_client["client_id"]
    state_data["client_secret"] = oauth_client.get("client_secret") or ""

    grant_params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": state_data.get("redirect_uri") or "",
        "code_verifier": state_data.get("code_verifier") or "",
        "client_id": state_data.get("client_id") or "",
    }
    if config.MCP_OAUTH_SEND_RESOURCE and state_data.get("resource"):
        grant_params["resource"] = state_data["resource"]
    data = _token_request(state_data, grant_params, use_basic=True)

    access_token = data.get("access_token")
    if not access_token:
        raise MCPAuthRequiredError("OAuth token response is missing access_token.")
    refresh_token = data.get("refresh_token")
    scope = data.get("scope")
    if not scope and state_data.get("scope"):
        scope = state_data["scope"]
    expires_in = data.get("expires_in")
    expires_at = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = datetime.utcnow() + timedelta(seconds=int(expires_in))
    mcp_registry.save_oauth_tokens(
        user_id, server_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        scope=scope,
        issuer=expected_issuer or state_data.get("issuer"),
    )
    mcp_registry.set_connection_state(user_id, server_id, "connected", commit=True)
    try:
        # 認証完了したサーバーはモデルへ公開する（設定のトグルで後から無効化できる）
        mcp_registry.set_enabled(user_id, server_id, True)
    except Exception:
        pass
    srv = mcp_registry.get_server_for_user(user_id, server_id)
    # 認証後に tools/list を再取得してキャッシュへ載せる
    if srv is not None:
        try:
            from . import client as _client
            _headers = mcp_registry.headers_for_server(user_id, srv)
            _tools = _client.fetch_tools(
                srv.url, headers=_headers,
                read_timeout=config.MCP_READ_TIMEOUT_SECONDS,
                max_tools=config.MCP_MAX_TOOLS_PER_SERVER,
            )
            mcp_registry.set_cached_tools(user_id, server_id, _tools or [])
        except Exception:
            pass
    return {
        "server_id": server_id,
        "server_name": srv.name if srv else "MCP server",
        "user_id": user_id,
    }


def _set_error_state(user_id, server_id, message):
    mcp_registry.set_connection_state(user_id, server_id, "error", last_error=message, commit=True)


def refresh_access_token(user_id, server):
    """refresh_token によるアクセストークン更新。失敗時は MCPAuthRequiredError。"""
    from app import decrypt_val
    cred = mcp_registry.get_credential(user_id, server.id)
    if cred is None or not cred.refresh_token_enc:
        raise MCPAuthRequiredError("このサーバーには再認証用トークンがありません。再認証してください。")
    oauth_client = mcp_registry.decrypt_oauth_client(user_id, server.oauth_provider_key_safe)
    if oauth_client is None or not oauth_client.get("client_id"):
        raise MCPAuthRequiredError("OAuthクライアント情報が未登録です。")
    state_data = {
        "token_endpoint": _resolve_token_endpoint(user_id, server),
        "client_id": oauth_client["client_id"],
        "client_secret": oauth_client.get("client_secret") or "",
    }
    if not state_data["token_endpoint"]:
        raise MCPAuthRequiredError("Token endpoint could not be resolved.")
    refresh_token = decrypt_val(cred.refresh_token_enc)
    grant_params = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth_client["client_id"],
    }
    if config.MCP_OAUTH_SEND_RESOURCE and server.url:
        grant_params["resource"] = server.url
    data = _token_request(state_data, grant_params, use_basic=True)
    access_token = data.get("access_token")
    if not access_token:
        raise MCPAuthRequiredError("Token refresh failed: no access_token returned.")
    new_refresh = data.get("refresh_token") or refresh_token
    expires_in = data.get("expires_in")
    expires_at = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = datetime.utcnow() + timedelta(seconds=int(expires_in))
    scope = data.get("scope") or cred.scope or ""
    mcp_registry.save_oauth_tokens(
        user_id, server.id,
        access_token=access_token,
        refresh_token=new_refresh,
        expires_at=expires_at,
        scope=scope,
        issuer=cred.issuer or "",
    )
    return True


def _resolve_token_endpoint(user_id, server):
    # 既存 cred の issuer からメタデータを引く（キャッシュなし簡易版）
    cred = mcp_registry.get_credential(user_id, server.id)
    if cred and cred.issuer:
        try:
            meta = _discover_authorization_server(cred.issuer)
            if meta.get("token_endpoint"):
                return meta["token_endpoint"]
        except Exception:
            pass
    if server.oauth_provider_key_safe == "google_workspace":
        return _GOOGLE_DISCOVERY["token_endpoint"]
    try:
        meta = _as_metadata_for_server(server)
        return meta.get("token_endpoint") or ""
    except Exception:
        return ""


# Bearerトークン保存（registry側に委譲）
def save_bearer_token(user_id, server_id, token):
    mcp_registry.save_bearer_token(user_id, server_id, token)
