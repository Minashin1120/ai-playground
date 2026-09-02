"""MCP設定画面・OAuthコールバック・チャット判断受付のAPIルート（Blueprint）。

- プレフィックス: /api/mcp
- GET /servers ほか書き込み系は app.py のグローバルCSRF（before_request）で自動保護される。
- ブラウザJS・レスポンスへトークン・Client Secret は一切含めない。
"""
from __future__ import annotations

import re
import json
from urllib.parse import urlparse

from flask import Blueprint, current_app, jsonify, request, render_template
from flask_login import current_user, login_required

from . import registry as mcp_registry
from . import security as mcp_security
from . import client as mcp_client
from .errors import (
    MCPAuthRequiredError,
    MCPError,
    MCPInsufficientScopeError,
    MCPSecurityError,
    MCPTimeoutError,
    MCPConnectionError,
    MCPValidationError,
)

bp = Blueprint("mcp_service", __name__)

_CUSTOM_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _json_error(exc, status=None):
    if isinstance(exc, MCPError):
        return jsonify(exc.to_dict()), status or exc.http_status
    return jsonify({"error": str(exc), "code": "mcp_error"}), status or 400


def _ok(payload=None, status=200):
    if payload is None:
        payload = {"status": "ok"}
    return jsonify(payload), status


def _redirect_uri():
    """OAuthコールバックの絶対URL（固定パス）を返す。"""
    from . import config
    host_url = request.host_url.rstrip("/")
    return host_url + config.MCP_OAUTH_CALLBACK_PATH


def _probe_server(user_id, srv, *, max_tools=8):
    """接続確認＋tools/list 取得（ツールは実行しない）。"""
    headers = {}
    if srv.auth_type_safe != "none":
        cred = mcp_registry.get_credential(user_id, srv.id)
        if cred is None or not cred.access_token_enc:
            return {
                "ok": True,
                "needs_auth": True,
                "message": "認証後にツール一覧を取得します。",
                "tool_count": 0,
            }
        headers = mcp_registry.headers_for_server(user_id, srv)
    try:
        tools = mcp_client.fetch_tools(
            srv.url, headers=headers, read_timeout=30, max_tools=max_tools
        )
    except MCPAuthRequiredError as exc:
        return {"ok": True, "needs_auth": True,
                "message": "認証が必要です（401）。設定画面から認証してください。", "tool_count": 0}
    except MCPInsufficientScopeError as exc:
        return {"ok": True, "needs_auth": True,
                "message": "追加の権限が必要です（403 insufficient_scope）。再認証してください。",
                "tool_count": 0}
    except (MCPSecurityError, MCPTimeoutError, MCPConnectionError, MCPError) as exc:
        return {"ok": False, "message": str(exc), "tool_count": 0}
    except Exception as exc:
        return {"ok": False, "message": f"接続確認に失敗しました: {exc}", "tool_count": 0}
    if not tools:
        return {"ok": True, "message": "接続できました（ツールはありません）。", "tool_count": 0}
    mcp_registry.set_cached_tools(user_id, srv.id, tools)
    return {"ok": True, "message": f"接続できました（{len(tools)}個のツール）。",
            "tool_count": len(tools)}


# ---------------------------------------------------------------------------
# サーバー一覧・登録
# ---------------------------------------------------------------------------
@bp.route("/servers", methods=["GET"])
@login_required
def list_servers():
    try:
        servers = mcp_registry.list_servers_for_user(current_user.id)
    except Exception as exc:
        return _json_error(exc)
    return _ok({"servers": servers, "google_workspace_provider_key": "google_workspace"})


@bp.route("/servers", methods=["POST"])
@login_required
def add_custom_server():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    url = (data.get("url") or "").strip()
    auth_type = (data.get("auth_type") or "none").strip().lower()
    description = (data.get("description") or "").strip()
    bearer_token = data.get("bearer_token")
    if bearer_token is not None and str(bearer_token).strip() in ("", "********"):
        bearer_token = None
    try:
        mcp_security.validate_mcp_url(url, resolve=True)
    except MCPSecurityError as exc:
        return _json_error(exc)
    try:
        srv = mcp_registry.register_custom(current_user.id, {
            "name": name, "url": url, "auth_type": auth_type, "description": description,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    except Exception as exc:
        return _json_error(exc)
    # Bearerトークンは登録時に入力された場合のみ保存
    if auth_type == "bearer" and bearer_token:
        try:
            mcp_registry.save_bearer_token(current_user.id, srv.id, bearer_token)
        except Exception as exc:
            return _json_error(exc)
    probe = _probe_server(current_user.id, srv)
    mcp_registry.set_connection_state(
        current_user.id, srv.id,
        "needs_auth" if probe.get("needs_auth") else ("error" if not probe.get("ok") else "connected"),
        last_error=None if probe.get("ok") else probe.get("message"),
    )
    item = mcp_registry.server_to_api_dict(current_user.id, srv)
    item["tool_count"] = probe.get("tool_count") or 0
    return _ok({"server": item, "probe": probe}, status=201)


@bp.route("/servers/<int:server_id>", methods=["PUT"])
@login_required
def update_server(server_id):
    data = request.get_json(silent=True) or {}
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    # カスタム項目の更新
    if not srv.is_preset:
        try:
            mcp_registry.update_custom(current_user.id, server_id, data)
            srv = mcp_registry.get_server_for_user(current_user.id, server_id)
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    # 有効/無効
    if "enabled" in data and isinstance(data["enabled"], bool):
        try:
            mcp_registry.set_enabled(current_user.id, server_id, data["enabled"])
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    # Bearerトークン更新
    if srv.auth_type_safe == "bearer" and data.get("bearer_token"):
        token = str(data["bearer_token"]).strip()
        if token and token != "********":
            try:
                mcp_registry.save_bearer_token(current_user.id, server_id, token)
            except ValueError as exc:
                return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    item = mcp_registry.server_to_api_dict(current_user.id, srv)
    item["tool_count"] = mcp_registry.get_cached_tool_count(current_user.id, server_id)
    return _ok({"server": item})


@bp.route("/servers/<int:server_id>", methods=["DELETE"])
@login_required
def delete_server(server_id):
    try:
        mcp_registry.delete_custom(current_user.id, server_id)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    return _ok({"deleted": True})


@bp.route("/servers/<int:server_id>/test", methods=["POST"])
@login_required
def test_server(server_id):
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    probe = _probe_server(current_user.id, srv)
    mcp_registry.set_connection_state(
        current_user.id, server_id,
        "needs_auth" if probe.get("needs_auth") else ("error" if not probe.get("ok") else "connected"),
        last_error=None if probe.get("ok") else probe.get("message"),
    )
    item = mcp_registry.server_to_api_dict(current_user.id, srv)
    item["tool_count"] = probe.get("tool_count") or 0
    return _ok({"server": item, "probe": probe})


@bp.route("/servers/<int:server_id>/tools", methods=["GET"])
@login_required
def list_server_tools(server_id):
    """サーバーのツール一覧（キャッシュ優先・無ければ取得）を返す。"""
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    tools = mcp_registry.get_cached_tools(current_user.id, server_id)
    if not tools:
        probe = _probe_server(current_user.id, srv, max_tools=30)
        if not probe.get("ok") and probe.get("needs_auth"):
            return jsonify({"error": "このサーバーは認証が必要です。先に認証してください。",
                            "code": "mcp_auth_required"}), 400
        tools = mcp_registry.get_cached_tools(current_user.id, server_id) or []
    from . import tools as mcp_tools_mod
    rows = []
    for tool in tools:
        name = tool.get("name") or ""
        rows.append({
            "name": name,
            "title": tool.get("title") or "",
            "description": tool.get("description") or "",
            "read_only": mcp_tools_mod.classify_readonly(name, tool.get("description") or ""),
        })
    return _ok({"tools": rows})


# ---------------------------------------------------------------------------
# OAuthクライアント情報（BYOK）
# ---------------------------------------------------------------------------
@bp.route("/oauth-client", methods=["PUT"])
@login_required
def save_oauth_client():
    data = request.get_json(silent=True) or {}
    provider_key = (data.get("provider_key") or "").strip()
    if not provider_key:
        return jsonify({"error": "provider_key が必要です。", "code": "mcp_validation_error"}), 400
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    if client_id is not None and str(client_id).strip():
        client_id = str(client_id).strip()
    elif client_id == "********":
        client_id = "********"
    else:
        client_id = None
    try:
        mcp_registry.save_oauth_client(current_user.id, provider_key, client_id, client_secret)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    return _ok({"status": "ok", "provider_key": provider_key})


# ---------------------------------------------------------------------------
# OAuth フロー
# ---------------------------------------------------------------------------
@bp.route("/servers/<int:server_id>/auth/start", methods=["POST"])
@login_required
def start_oauth(server_id):
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    from . import auth as mcp_auth
    try:
        url, _state = mcp_auth.build_authorization_url(current_user.id, srv, redirect_uri=_redirect_uri())
    except MCPValidationError as exc:
        if (exc.detail or {}).get("requires_oauth_client"):
            return jsonify({"error": str(exc), "code": "mcp_requires_oauth_client",
                            "requires_oauth_client": True}), 400
        return _json_error(exc)
    except MCPError as exc:
        return _json_error(exc)
    return _ok({"url": url, "server_id": server_id})


@bp.route("/oauth/callback", methods=["GET"])
@login_required
def oauth_callback():
    from . import auth as mcp_auth
    args = {k: (v[0] if isinstance(v, list) else v) for k, v in request.args.items()}
    try:
        info = mcp_auth.handle_oauth_callback(args)
    except MCPError as exc:
        return render_template(
            "mcp_oauth_callback.html",
            ok=False,
            message=str(exc),
            server_name="MCP server",
        ), 200
    except Exception as exc:
        return render_template(
            "mcp_oauth_callback.html", ok=False,
            message=f"OAuth処理中にエラーが発生しました: {exc}",
            server_name="MCP server",
        ), 200
    return render_template(
        "mcp_oauth_callback.html",
        ok=True,
        message=f"「{info.get('server_name')}」の認証が完了しました。",
        server_name=info.get("server_name") or "",
    )


@bp.route("/servers/<int:server_id>/auth/disconnect", methods=["POST"])
@login_required
def disconnect_auth(server_id):
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    try:
        mcp_registry.clear_credentials(current_user.id, server_id)
    except Exception as exc:
        return _json_error(exc)
    return _ok({"disconnected": True})


# ---------------------------------------------------------------------------
# ツール権限・確認ポリシー
# ---------------------------------------------------------------------------
@bp.route("/servers/<int:server_id>/tools/<path:tool_name>/permission", methods=["POST"])
@login_required
def set_tool_permission(server_id, tool_name):
    srv = mcp_registry.get_server_for_user(current_user.id, server_id)
    if srv is None:
        return jsonify({"error": "MCPサーバーが見つかりません。", "code": "mcp_not_found"}), 404
    data = request.get_json(silent=True) or {}
    tool_name = tool_name or ""
    if not tool_name:
        return jsonify({"error": "tool_name が必要です。", "code": "mcp_validation_error"}), 400
    try:
        mcp_registry.set_tool_permission(
            current_user.id, server_id, tool_name,
            allow=data.get("allow"),
            confirm=data.get("confirm"),
            classified_read_only=data.get("classified_read_only"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "mcp_validation_error"}), 400
    return _ok({"status": "ok"})


# ---------------------------------------------------------------------------
# チャット中の変更操作判断
# ---------------------------------------------------------------------------
@bp.route("/chat/<job_id>/decision", methods=["POST"])
@login_required
def chat_decision(job_id):
    data = request.get_json(silent=True) or {}
    decision = data.get("decision")
    if decision not in ("allow", "deny"):
        return jsonify({"error": "decision must be allow or deny", "code": "mcp_validation_error"}), 400
    # ジョブが自分（ログイン中ユーザー）のものであることを確認
    if not _job_belongs_to_user(current_user.id, job_id):
        return jsonify({"error": "ジョブが見つかりません。", "code": "mcp_not_found"}), 404
    from .execution import submit_chat_decision
    try:
        submit_chat_decision(job_id, decision)
    except Exception as exc:
        return _json_error(exc)
    return _ok({"status": "ok", "decision": decision})


def _job_belongs_to_user(user_id, job_id):
    """pending_job を横断して job_id が user のものか確認する。"""
    import redis as _redis
    try:
        import app as _a
        r = _a.redis_conn
        ThreadModel = _a.Thread
    except Exception:
        return False
    import os
    try:
        threads = ThreadModel.query.filter_by(user_id=user_id).limit(50).all()
        for t in threads:
            raw = r.get(f"pending_job:{user_id}:{t.id}")
            if raw:
                try:
                    data = json.loads(raw)
                    if data.get("job_id") == job_id:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# 監査ログ（管理者のみ）
# ---------------------------------------------------------------------------
@bp.route("/audit", methods=["GET"])
@login_required
def audit_logs():
    if not getattr(current_user, "is_admin", False):
        return jsonify({"error": "Forbidden", "code": "forbidden"}), 403
    from mcp_service.models import MCPCallLog
    limit = min(200, max(1, request.args.get("limit", 50, type=int)))
    try:
        rows = MCPCallLog.query.filter_by(user_id=current_user.id).order_by(
            MCPCallLog.id.desc()
        ).limit(limit).all()
        if not rows:
            # 管理者は全ユーザー分も見られる
            rows = MCPCallLog.query.order_by(MCPCallLog.id.desc()).limit(limit).all()
    except Exception:
        rows = []
    out = []
    for row in rows:
        out.append({
            "id": row.id,
            "user_id": row.user_id,
            "server_name": row.server_name,
            "tool_name": row.tool_name,
            "status": row.status,
            "http_status": row.http_status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "duration_ms": row.duration_ms,
            "result_size_bytes": row.result_size_bytes,
            "error_code": row.error_code,
        })
    return _ok({"logs": out})


# ---------------------------------------------------------------------------
# エラーハンドラ（Blueprint内）
# ---------------------------------------------------------------------------
@bp.app_errorhandler(MCPError)
def handle_mcp_error(exc):
    return jsonify(exc.to_dict()), exc.http_status
