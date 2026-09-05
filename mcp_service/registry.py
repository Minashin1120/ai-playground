"""プリセット定義・MCPサーバー／接続／権限のCRUD。

``db`` や ``encrypt_val`` など app.py が持つ依存は、このモジュールの import 時ではなく
各関数の実行時に ``from app import ...`` で取得する（app.py との循環import防止・軽量import）。
"""
from __future__ import annotations

import json
import secrets

from sqlalchemy import exc as sa_exc

from . import config
from . import tools as mcp_tools

# ---------------------------------------------------------------------------
# Google Workspace MCP プリセット（8サービス）
# ※ エンドポイント・スコープは 2026-09-03 時点の Google 公式ドキュメント記載値。
#    Google Workspace MCP は Developer Preview のため変更される可能性がある。
# ---------------------------------------------------------------------------
_GMAIL_SCOPES = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose"
_DRIVE_SCOPES = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file"
_DOCS_SCOPES = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/documents.readonly https://www.googleapis.com/auth/documents"
_SHEETS_SCOPES = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/spreadsheets.readonly https://www.googleapis.com/auth/spreadsheets"
_SLIDES_SCOPES = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/presentations.readonly https://www.googleapis.com/auth/presentations"
_CALENDAR_SCOPES = "https://www.googleapis.com/auth/calendar.calendarlist.readonly https://www.googleapis.com/auth/calendar.events.freebusy https://www.googleapis.com/auth/calendar.events.readonly"
_CHAT_SCOPES = "https://www.googleapis.com/auth/chat.spaces.readonly https://www.googleapis.com/auth/chat.memberships.readonly https://www.googleapis.com/auth/chat.messages.readonly https://www.googleapis.com/auth/chat.messages.create https://www.googleapis.com/auth/chat.users.readstate"
_PEOPLE_SCOPES = "https://www.googleapis.com/auth/directory.readonly https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/contacts.readonly"


PRESET_SERVERS = [
    {
        "slug": "google_gmail",
        "preset_key": "google_gmail",
        "name": "Google Gmail",
        "url": "https://gmailmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _GMAIL_SCOPES,
        "description": "Gmail の検索・取得・下書き作成・送信を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_drive",
        "preset_key": "google_drive",
        "name": "Google Drive",
        "url": "https://drivemcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _DRIVE_SCOPES,
        "description": "Google Drive のファイル検索・取得・アップロードを提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_docs",
        "preset_key": "google_docs",
        "name": "Google Docs",
        "url": "https://docsmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _DOCS_SCOPES,
        "description": "Google Docs 文書の読み取り・編集を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_sheets",
        "preset_key": "google_sheets",
        "name": "Google Sheets",
        "url": "https://sheetsmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _SHEETS_SCOPES,
        "description": "Google Sheets スプレッドシートの読み取り・編集を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_slides",
        "preset_key": "google_slides",
        "name": "Google Slides",
        "url": "https://slidesmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _SLIDES_SCOPES,
        "description": "Google Slides プレゼンテーションの読み取り・編集を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_calendar",
        "preset_key": "google_calendar",
        "name": "Google Calendar",
        "url": "https://calendarmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _CALENDAR_SCOPES,
        "description": "Google Calendar の予定の読み取り・作成を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_chat",
        "preset_key": "google_chat",
        "name": "Google Chat",
        "url": "https://chatmcp.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _CHAT_SCOPES,
        "description": "Google Chat のスペース・メッセージの読み取り・送信を提供する Google 公式 MCP サーバー。",
    },
    {
        "slug": "google_people",
        "preset_key": "google_people",
        "name": "Google People",
        "url": "https://people.googleapis.com/mcp/v1",
        "auth_type": "oauth",
        "oauth_provider_key": "google_workspace",
        "recommended_scopes": _PEOPLE_SCOPES,
        "description": "Google People（連絡先・プロフィール）を提供する Google 公式 MCP サーバー。",
    },
]

PRESET_BY_SLUG = {p["slug"]: p for p in PRESET_SERVERS}


def _app_module():
    # app.py が完全ロード済みの状態でのみ呼ばれる
    import app as _app_mod
    return _app_mod


def _models():
    from mcp_service import models as m
    return m


# ---------------------------------------------------------------------------
# プリセット・行の確保
# ---------------------------------------------------------------------------
def get_or_create_presets():
    """プリセットサーバー行が無ければ作る（起動時・一覧取得前などに呼ぶ）。"""
    from app import db
    m = _models()
    created = False
    for p in PRESET_SERVERS:
        row = m.MCPServer.query.filter_by(preset_key=p["preset_key"]).first()
        if row is None:
            row = m.MCPServer(
                slug=p["slug"],
                name=p["name"],
                url=p["url"],
                transport="streamable_http",
                auth_type=p["auth_type"],
                oauth_provider_key=p.get("oauth_provider_key"),
                recommended_scopes=p.get("recommended_scopes"),
                is_preset=True,
                preset_key=p["preset_key"],
                description=p.get("description"),
            )
            db.session.add(row)
            created = True
        else:
            # URL等の定義が古い場合に追随させる
            changed = False
            for field, value in (
                ("name", p["name"]),
                ("url", p["url"]),
                ("auth_type", p["auth_type"]),
                ("oauth_provider_key", p.get("oauth_provider_key")),
                ("recommended_scopes", p.get("recommended_scopes")),
            ):
                if getattr(row, field, None) != value:
                    setattr(row, field, value)
                    changed = True
            if changed:
                db.session.add(row)
    if created:
        try:
            db.session.commit()
        except sa_exc.IntegrityError:
            db.session.rollback()
        except Exception:
            db.session.rollback()
    return True


def ensure_user_connection(user_id, server_id, commit=False):
    """ユーザー×サーバーの接続行を確保して返す。"""
    from app import db
    m = _models()
    conn = m.MCPUserConnection.query.filter_by(user_id=user_id, server_id=server_id).first()
    if conn is None:
        conn = m.MCPUserConnection(
            user_id=user_id, server_id=server_id, is_enabled=False, connection_state="none"
        )
        db.session.add(conn)
        try:
            db.session.commit()
        except sa_exc.IntegrityError:
            db.session.rollback()
            conn = m.MCPUserConnection.query.filter_by(user_id=user_id, server_id=server_id).first()
        except Exception:
            db.session.rollback()
    return conn


def ensure_user_rows(user_id):
    """プリセットと自分所有のカスタムについて接続行を確保する。"""
    get_or_create_presets()
    from app import db
    m = _models()
    servers = m.MCPServer.query.filter(
        (m.MCPServer.is_preset.is_(True)) | (m.MCPServer.owner_user_id == user_id)
    ).all()
    server_ids = [srv.id for srv in servers]
    if server_ids:
        existing = m.MCPUserConnection.query.filter(
            m.MCPUserConnection.user_id == user_id,
            m.MCPUserConnection.server_id.in_(server_ids),
        ).all()
        existing_ids = {row.server_id for row in existing}
        missing = [server_id for server_id in server_ids if server_id not in existing_ids]
        if missing:
            for server_id in missing:
                db.session.add(m.MCPUserConnection(
                    user_id=user_id,
                    server_id=server_id,
                    is_enabled=False,
                    connection_state="none",
                ))
            try:
                # A single commit is substantially faster than one commit per
                # preset, especially on remote databases during the first load.
                db.session.commit()
            except sa_exc.IntegrityError:
                # Another request may have created the same rows concurrently.
                db.session.rollback()
    db.session.expire_all()
    return True


def visible_servers_for_user(user_id):
    """ユーザーが見えるサーバー行（プリセット＋自分所有カスタム）を返す。"""
    m = _models()
    return (
        m.MCPServer.query.filter(
            (m.MCPServer.is_preset.is_(True)) | (m.MCPServer.owner_user_id == user_id)
        )
        .order_by(m.MCPServer.is_preset.desc(), m.MCPServer.id.asc())
        .all()
    )


# ---------------------------------------------------------------------------
# 一覧（Web API 用：秘密は含めない）
# ---------------------------------------------------------------------------
def server_to_api_dict(user_id, srv, conn=None, cred=None):
    """MCP設定画面へ返す1サーバー分の dict。トークン類は含めない。"""
    from app import decrypt_val
    m = _models()
    if conn is None:
        conn = m.MCPUserConnection.query.filter_by(user_id=user_id, server_id=srv.id).first()
    oauth_client = None
    if srv.auth_type_safe == "oauth" and srv.oauth_provider_key_safe:
        oauth_client = (
            m.MCPUserOAuthClient.query.filter_by(
                user_id=user_id, provider_key=srv.oauth_provider_key_safe
            ).first()
        )
    cred_state = None
    cred_expired = False
    if cred is None and srv.id:
        cred = m.MCPUserCredential.query.filter_by(user_id=user_id, server_id=srv.id).first()
    if cred is not None and cred.access_token_enc:
        cred_state = "connected"
        if cred.expires_at is not None:
            from datetime import datetime, timezone
            try:
                exp = cred.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                cred_expired = exp <= datetime.now(timezone.utc)
            except Exception:
                cred_expired = False
    else:
        cred_state = None

    state = "none"
    if conn is not None:
        state = conn.connection_state or "none"
    # 認証済み状態の導出
    auth_status = "none"
    if srv.auth_type_safe == "none":
        auth_status = "connected"
    elif srv.auth_type_safe == "bearer":
        auth_status = "connected" if cred_state == "connected" else "needs_auth"
    elif srv.auth_type_safe == "oauth":
        if cred_state == "connected":
            auth_status = "expired" if cred_expired else "connected"
        else:
            auth_status = "needs_auth"

    d = {
        "id": srv.id,
        "slug": srv.slug,
        "name": srv.name,
        "url": srv.url,
        "transport": srv.transport or "streamable_http",
        "auth_type": srv.auth_type_safe,
        "oauth_provider_key": srv.oauth_provider_key_safe,
        "is_preset": bool(srv.is_preset),
        "preset_key": srv.preset_key,
        "description": srv.description or "",
        "enabled": bool(conn and conn.is_enabled),
        "connection_state": state,
        "auth_status": auth_status,
        "auth_has_token": cred_state == "connected",
        "auth_expired": cred_expired,
        "auth_scope": _plain_scope(cred),
        "last_error": (conn.last_error if conn else None),
        "last_checked_at": (conn.last_checked_at.isoformat() if conn and conn.last_checked_at else None),
        "oauth_client_registered": bool(oauth_client and oauth_client.has_client_info),
        "oauth_client_id_masked": (
            _mask_client_id(decrypt_val(oauth_client.client_id_enc)) if oauth_client and oauth_client.client_id_enc else ""
        ),
    }
    return d


def _plain_scope(cred):
    if cred is None:
        return ""
    return cred.scope or ""


def _mask_client_id(client_id):
    if not client_id:
        return ""
    if len(client_id) <= 10:
        return client_id[:3] + "********"
    return f"{client_id[:8]}...{client_id[-4:]}"


def list_servers_for_user(user_id):
    """設定画面用のサーバー一覧（プリセット＋自分所有カスタム）。"""
    get_or_create_presets()
    ensure_user_rows(user_id)
    m = _models()
    out = []
    for srv in visible_servers_for_user(user_id):
        try:
            item = server_to_api_dict(user_id, srv)
            item["tool_count"] = get_cached_tool_count(user_id, srv.id)
            out.append(item)
        except Exception:
            continue
    return out


def get_server_for_user(user_id, server_id):
    m = _models()
    srv = m.MCPServer.query.get(int(server_id))
    if srv is None:
        return None
    if not (srv.is_preset or srv.owner_user_id == user_id):
        return None
    return srv


# ---------------------------------------------------------------------------
# カスタム登録
# ---------------------------------------------------------------------------
def _make_custom_slug():
    for _ in range(5):
        cand = "custom_" + secrets.token_hex(4)
        if not _models().MCPServer.query.filter_by(slug=cand).first():
            return cand
    return "custom_" + secrets.token_hex(10)


def register_custom(user_id, payload):
    """カスタムMCPを追加する。SSRFチェックは呼び出し側で実施済み前提。"""
    from app import db
    m = _models()
    name = (str(payload.get("name") or "")).strip()
    url = (str(payload.get("url") or "")).strip()
    auth_type = (str(payload.get("auth_type") or "none")).strip().lower()
    description = (str(payload.get("description") or "")).strip()
    if auth_type not in ("none", "bearer", "oauth"):
        auth_type = "none"
    if not name or len(name) > 150:
        raise ValueError("表示名は1〜150文字で入力してください。")
    if not url:
        raise ValueError("MCP URLを入力してください。")
    # 重複URL（自分所有カスタム）のチェック
    existing = (
        m.MCPServer.query.filter(
            m.MCPServer.owner_user_id == user_id,
            m.MCPServer.url == url,
            m.MCPServer.is_preset.is_(False),
        ).first()
    )
    if existing is not None:
        raise ValueError("同じURLのMCPサーバーがすでに登録されています。")
    srv = m.MCPServer(
        slug=_make_custom_slug(),
        name=name,
        url=url,
        transport="streamable_http",
        auth_type=auth_type,
        oauth_provider_key=None,
        recommended_scopes=None,
        is_preset=False,
        preset_key=None,
        description=description,
        owner_user_id=user_id,
    )
    if auth_type == "oauth":
        # カスタムOAuthサーバーはサーバー固有の provider_key を持つ
        srv.oauth_provider_key = srv.slug
    db.session.add(srv)
    try:
        db.session.flush()
    except Exception as e:
        db.session.rollback()
        raise ValueError(f"MCPサーバーの登録に失敗しました: {e}")
    ensure_user_connection(user_id, srv.id, commit=True)
    return srv


def update_custom(user_id, server_id, payload):
    from app import db
    srv = get_server_for_user(user_id, server_id)
    if srv is None or srv.is_preset:
        raise ValueError("更新できるのは自分で登録したカスタムMCPサーバーのみです。")
    name = (str(payload.get("name") or srv.name or "")).strip()
    url = (str(payload.get("url") or srv.url or "")).strip()
    auth_type = (str(payload.get("auth_type") or srv.auth_type_safe or "none")).strip().lower()
    description = payload.get("description")
    if description is not None:
        description = str(description).strip()
    if not name or not url:
        raise ValueError("表示名とURLは必須です。")
    if auth_type not in ("none", "bearer", "oauth"):
        auth_type = "none"
    auth_type_changed = srv.auth_type_safe != auth_type
    srv.name = name
    srv.url = url
    srv.auth_type = auth_type
    if auth_type == "oauth" and not srv.oauth_provider_key:
        srv.oauth_provider_key = srv.slug
    if auth_type_changed and auth_type != "oauth":
        # 認証方式の変更時のみ、以前の認証情報を無効化して再設定を促す
        clear_credentials(user_id, server_id)
    if description is not None:
        srv.description = description
    db.session.add(srv)
    db.session.commit()
    return srv


def delete_custom(user_id, server_id):
    """カスタムMCPを削除する（プリセットは削除不可）。"""
    from app import db
    m = _models()
    srv = get_server_for_user(user_id, server_id)
    if srv is None:
        return False
    if srv.is_preset:
        raise ValueError("プリセットMCPサーバーは削除できません。設定タブから無効化してください。")
    # 紐づく行を削除
    for model in (
        m.MCPUserConnection,
        m.MCPUserCredential,
        m.MCPToolPermission,
        m.MCPCallLog,
    ):
        try:
            model.query.filter_by(user_id=user_id, server_id=srv.id).delete(synchronize_session=False)
        except Exception:
            pass
    db.session.delete(srv)
    db.session.commit()
    return True


def delete_user_mcp_data(user_id):
    """アカウント削除時にMCP関連データを掃除する。"""
    from app import db
    m = _models()
    for model in (
        m.MCPUserOAuthClient,
        m.MCPUserCredential,
        m.MCPToolPermission,
        m.MCPCallLog,
        m.MCPUserConnection,
    ):
        try:
            model.query.filter_by(user_id=user_id).delete(synchronize_session=False)
        except Exception:
            pass
    try:
        for srv in m.MCPServer.query.filter_by(owner_user_id=user_id).all():
            for model in (m.MCPUserConnection, m.MCPUserCredential, m.MCPToolPermission, m.MCPCallLog):
                try:
                    model.query.filter_by(server_id=srv.id).delete(synchronize_session=False)
                except Exception:
                    pass
            db.session.delete(srv)
    except Exception:
        pass
    db.session.commit()


# ---------------------------------------------------------------------------
# 有効/無効・状態
# ---------------------------------------------------------------------------
def set_enabled(user_id, server_id, enabled):
    from app import db
    m = _models()
    srv = get_server_for_user(user_id, server_id)
    if srv is None:
        raise ValueError("MCPサーバーが見つかりません。")
    conn = ensure_user_connection(user_id, server_id)
    conn.is_enabled = bool(enabled)
    if enabled and conn.connection_state in ("none",):
        # 認証タイプに応じて初期状態
        if srv.auth_type_safe == "none":
            conn.connection_state = "connected"
        else:
            conn.connection_state = "needs_auth"
    db.session.add(conn)
    db.session.commit()
    _invalidate_tool_cache(user_id, server_id)
    return conn


def set_connection_state(user_id, server_id, state, last_error=None, commit=True):
    from app import db
    m = _models()
    conn = ensure_user_connection(user_id, server_id)
    conn.connection_state = state
    if last_error is not None:
        conn.last_error = last_error
    from datetime import datetime
    conn.last_checked_at = datetime.utcnow()
    db.session.add(conn)
    if commit:
        db.session.commit()
    return conn


# ---------------------------------------------------------------------------
# 資格情報（保存・取得）
# ---------------------------------------------------------------------------
def save_bearer_token(user_id, server_id, token):
    """Bearerトークンを暗号化保存する。"""
    from app import db, encrypt_val
    m = _models()
    if not token or str(token).strip() == "":
        raise ValueError("Bearerトークンを入力してください。")
    cred = m.MCPUserCredential.query.filter_by(user_id=user_id, server_id=server_id).first()
    if cred is None:
        cred = m.MCPUserCredential(user_id=user_id, server_id=server_id)
        db.session.add(cred)
    cred.access_token_enc = encrypt_val(str(token).strip())
    cred.refresh_token_enc = None
    cred.expires_at = None
    cred.scope = None
    cred.issuer = None
    db.session.commit()
    set_connection_state(user_id, server_id, "connected", commit=True)
    _invalidate_tool_cache(user_id, server_id)


def save_oauth_tokens(user_id, server_id, *, access_token, refresh_token=None,
                      expires_at=None, scope=None, issuer=None):
    from app import db, encrypt_val
    m = _models()
    cred = m.MCPUserCredential.query.filter_by(user_id=user_id, server_id=server_id).first()
    if cred is None:
        cred = m.MCPUserCredential(user_id=user_id, server_id=server_id)
        db.session.add(cred)
    cred.access_token_enc = encrypt_val(access_token) if access_token else cred.access_token_enc
    cred.refresh_token_enc = encrypt_val(refresh_token) if refresh_token else cred.refresh_token_enc
    cred.expires_at = expires_at
    cred.scope = scope or None
    cred.issuer = issuer or None
    db.session.commit()
    set_connection_state(user_id, server_id, "connected", commit=True)
    _invalidate_tool_cache(user_id, server_id)


def get_credential(user_id, server_id):
    m = _models()
    return m.MCPUserCredential.query.filter_by(user_id=user_id, server_id=server_id).first()


def clear_credentials(user_id, server_id):
    """認証解除（保存済みトークンを削除）。"""
    from app import db
    m = _models()
    cred = m.MCPUserCredential.query.filter_by(user_id=user_id, server_id=server_id).first()
    if cred is not None:
        db.session.delete(cred)
    conn = m.MCPUserConnection.query.filter_by(user_id=user_id, server_id=server_id).first()
    if conn is not None:
        conn.connection_state = "needs_auth"
        conn.last_error = None
        db.session.add(conn)
    db.session.commit()
    _invalidate_tool_cache(user_id, server_id)
    return True


def _decrypt_credential(cred):
    """資格情報行を復号して dict で返す（ワーカー・サーバー内のみで使用）。"""
    from app import decrypt_val
    if cred is None:
        return None
    return {
        "access_token": decrypt_val(cred.access_token_enc) if cred.access_token_enc else None,
        "refresh_token": decrypt_val(cred.refresh_token_enc) if cred.refresh_token_enc else None,
        "token_type": cred.token_type or "Bearer",
        "expires_at": cred.expires_at,
        "scope": cred.scope or "",
        "issuer": cred.issuer or "",
    }


def headers_for_server(user_id, srv):
    """認証タイプに応じた Authorization ヘッダーを返す（無ければ空 dict）。"""
    if srv.auth_type_safe == "none":
        return {}
    cred = get_credential(user_id, srv.id)
    if cred is None or not cred.access_token_enc:
        return {}
    data = _decrypt_credential(cred)
    token = data.get("access_token") or ""
    if not token:
        return {}
    return {"Authorization": f"{(data.get('token_type') or 'Bearer')} {token}".strip()}


# ---------------------------------------------------------------------------
# OAuthクライアント情報（BYOK）
# ---------------------------------------------------------------------------
def save_oauth_client(user_id, provider_key, client_id, client_secret):
    """ユーザーごとのOAuthクライアント情報を保存する（マスクは呼び出し側で除去済み）。"""
    from app import db, encrypt_val, _SECRET_MASK
    if not provider_key:
        raise ValueError("provider_key がありません。")
    m = _models()
    row = m.MCPUserOAuthClient.query.filter_by(user_id=user_id, provider_key=provider_key).first()
    if row is None:
        row = m.MCPUserOAuthClient(user_id=user_id, provider_key=provider_key)
        db.session.add(row)
    if client_id is not None and client_id != _SECRET_MASK:
        if not str(client_id).strip():
            raise ValueError("Client IDを入力してください。")
        row.client_id_enc = encrypt_val(str(client_id).strip())
    if client_secret is not None and client_secret != _SECRET_MASK:
        if not str(client_secret).strip():
            raise ValueError("Client Secretを入力してください。")
        row.client_secret_enc = encrypt_val(str(client_secret).strip())
    if not row.client_id_enc:
        raise ValueError("Client IDを入力してください。")
    db.session.commit()
    return row


def get_oauth_client(user_id, provider_key):
    m = _models()
    if not provider_key:
        return None
    return m.MCPUserOAuthClient.query.filter_by(user_id=user_id, provider_key=provider_key).first()


def decrypt_oauth_client(user_id, provider_key):
    """OAuthクライアント情報を復号して返す。無ければ None。"""
    from app import decrypt_val
    row = get_oauth_client(user_id, provider_key)
    if row is None:
        return None
    client_id = decrypt_val(row.client_id_enc) if row.client_id_enc else None
    client_secret = decrypt_val(row.client_secret_enc) if row.client_secret_enc else None
    if not client_id:
        return None
    return {"client_id": client_id, "client_secret": client_secret or ""}


# ---------------------------------------------------------------------------
# ツール権限
# ---------------------------------------------------------------------------
def get_tool_permission(user_id, server_id, tool_name):
    m = _models()
    return (
        m.MCPToolPermission.query.filter_by(
            user_id=user_id, server_id=server_id, tool_name=tool_name
        ).first()
    )


def set_tool_permission(user_id, server_id, tool_name, *, allow=None, confirm=None,
                        classified_read_only=None):
    from app import db
    m = _models()
    row = get_tool_permission(user_id, server_id, tool_name)
    if row is None:
        row = m.MCPToolPermission(
            user_id=user_id, server_id=server_id, tool_name=tool_name,
            allow=True, confirm="default",
        )
        db.session.add(row)
    if allow is not None:
        row.allow = bool(allow)
    if confirm is not None:
        if confirm not in ("default", "always", "never"):
            raise ValueError("確認ポリシーが不正です。")
        row.confirm = confirm
    if classified_read_only is not None:
        row.classified_read_only = bool(classified_read_only) if isinstance(classified_read_only, bool) else None
    db.session.commit()
    return row


def tool_decision(user_id, server_id, tool_name, auto_read_only=None):
    """ツールの実行方針を返す: (allow, needs_confirm, is_read_only)。"""
    perm = get_tool_permission(user_id, server_id, tool_name)
    read_only = auto_read_only
    if perm is not None and perm.classified_read_only is not None:
        read_only = perm.classified_read_only
    if perm is None:
        allow = True
        confirm_policy = "default"
    else:
        allow = perm.allow
        confirm_policy = perm.confirm or "default"
    if confirm_policy == "always":
        needs_confirm = True
    elif confirm_policy == "never":
        needs_confirm = False
    else:  # default: 変更操作は毎回確認
        needs_confirm = (read_only is False)
    return allow, needs_confirm, read_only


# ---------------------------------------------------------------------------
# ツール一覧キャッシュ（Redis。ユーザー間共有しない）
# ---------------------------------------------------------------------------
def _redis():
    import app as _a
    try:
        return _a.redis_conn
    except Exception:
        import redis as _redis
        return _redis.from_url(_a.REDIS_URL)


def _tool_cache_key(user_id, server_id):
    return f"mcp_tools:{server_id}:{user_id}"


def _invalidate_tool_cache(user_id, server_id):
    try:
        _redis().delete(_tool_cache_key(user_id, server_id))
    except Exception:
        pass


def get_cached_tools(user_id, server_id):
    try:
        raw = _redis().get(_tool_cache_key(user_id, server_id))
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def get_cached_tool_count(user_id, server_id):
    tools = get_cached_tools(user_id, server_id)
    return len(tools) if isinstance(tools, list) else 0


def set_cached_tools(user_id, server_id, tools, ttl=None):
    try:
        if ttl is None:
            ttl = config.MCP_TOOLS_CACHE_TTL_SECONDS
        _redis().setex(_tool_cache_key(user_id, server_id), ttl, json.dumps(tools))
    except Exception:
        pass
