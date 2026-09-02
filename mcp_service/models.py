"""MCP外部連携のDBモデル。

このモジュールは app.py 内で ``db`` 定義後・``db.create_all()`` 実行前に
import される前提（実装プラン §3 / §14-10）。SQLAlchemy モデル定義のみを
持ち、外部通信や認証処理は含めない。
"""
from __future__ import annotations

from datetime import datetime

from app import db


def _now():
    return datetime.utcnow()


class MCPServer(db.Model):
    """MCPサーバー定義（プリセット＋カスタム）。サーバー全体で共有。"""

    __tablename__ = "mcp_servers"

    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(80), unique=True, nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    url = db.Column(db.Text, nullable=False)
    transport = db.Column(db.String(32), default="streamable_http", nullable=False)
    auth_type = db.Column(db.String(24), default="none", nullable=False)  # none/bearer/oauth
    oauth_provider_key = db.Column(db.String(120), nullable=True)  # google_workspace 等
    recommended_scopes = db.Column(db.Text, nullable=True)  # スペース区切り
    is_preset = db.Column(db.Boolean, default=False, nullable=False)
    preset_key = db.Column(db.String(80), nullable=True, index=True)
    # カスタムサーバーの所有者（プリセットは NULL）。他ユーザーへは見せない
    owner_user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_now)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)

    @property
    def auth_type_safe(self):
        return self.auth_type or "none"

    @property
    def oauth_provider_key_safe(self):
        return self.oauth_provider_key or (self.slug if not self.is_preset else None)


class MCPUserConnection(db.Model):
    """ユーザーごとのMCPサーバー接続状態・有効/無効。"""

    __tablename__ = "mcp_user_connections"
    __table_args__ = (
        db.UniqueConstraint("user_id", "server_id", name="uq_mcp_user_conn"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    server_id = db.Column(db.Integer, db.ForeignKey("mcp_servers.id", ondelete="CASCADE"), nullable=False, index=True)
    is_enabled = db.Column(db.Boolean, default=False, nullable=False)
    connection_state = db.Column(db.String(24), default="none", nullable=False)
    # none / needs_auth / connected / error
    last_checked_at = db.Column(db.DateTime, nullable=True)
    last_error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_now)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)


class MCPUserCredential(db.Model):
    """ユーザーごとのOAuth/Bearer秘密情報（Fernet暗号化カラム）。"""

    __tablename__ = "mcp_user_credentials"
    __table_args__ = (
        db.UniqueConstraint("user_id", "server_id", name="uq_mcp_user_cred"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    server_id = db.Column(db.Integer, db.ForeignKey("mcp_servers.id", ondelete="CASCADE"), nullable=False, index=True)
    access_token_enc = db.Column(db.Text, nullable=True)
    refresh_token_enc = db.Column(db.Text, nullable=True)
    token_type = db.Column(db.String(32), default="Bearer", nullable=False)
    expires_at = db.Column(db.DateTime, nullable=True)
    scope = db.Column(db.Text, nullable=True)
    issuer = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_now)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)

    @property
    def has_token(self):
        return bool(self.access_token_enc)


class MCPUserOAuthClient(db.Model):
    """ユーザーごとのOAuthクライアント情報（BYOK。Client ID / Secret 暗号化）。"""

    __tablename__ = "mcp_user_oauth_clients"
    __table_args__ = (
        db.UniqueConstraint("user_id", "provider_key", name="uq_mcp_user_oauth_client"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    provider_key = db.Column(db.String(120), nullable=False)
    client_id_enc = db.Column(db.Text, nullable=True)
    client_secret_enc = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_now)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)

    @property
    def has_client_info(self):
        return bool(self.client_id_enc and self.client_secret_enc)


class MCPToolPermission(db.Model):
    """ツール単位の許可・確認ポリシー（ユーザー×サーバー×ツール）。"""

    __tablename__ = "mcp_tool_permissions"
    __table_args__ = (
        db.UniqueConstraint("user_id", "server_id", "tool_name", name="uq_mcp_tool_perm"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    server_id = db.Column(db.Integer, db.ForeignKey("mcp_servers.id", ondelete="CASCADE"), nullable=False, index=True)
    tool_name = db.Column(db.String(255), nullable=False)
    allow = db.Column(db.Boolean, default=True, nullable=False)
    # default / always / never（default = 変更操作は毎回確認）
    confirm = db.Column(db.String(16), default="default", nullable=False)
    # True/False = ユーザーによる明示上書き。None = 未設定（自動判定を使う）
    classified_read_only = db.Column(db.Boolean, nullable=True)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)


class MCPCallLog(db.Model):
    """MCPツール実行の監査ログ。本文・秘密は含めない。"""

    __tablename__ = "mcp_call_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    server_id = db.Column(db.Integer, nullable=True, index=True)
    server_name = db.Column(db.String(160), nullable=True)
    tool_name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(24), default="ok", nullable=False)  # ok/error/rejected/timeout/security
    http_status = db.Column(db.Integer, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True, index=True)
    duration_ms = db.Column(db.Integer, nullable=True)
    result_size_bytes = db.Column(db.Integer, nullable=True)
    error_code = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=_now)
