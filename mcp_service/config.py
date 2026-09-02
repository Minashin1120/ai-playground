"""動作既定値の一元管理（MCP外部連携）。

`.env` は原則読まない（実装プラン §4）。将来、管理者DB設定による上書きを
追加する場合もここを差し替え口にする。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 接続・タイムアウト
# ---------------------------------------------------------------------------
MCP_CONNECT_TIMEOUT_SECONDS = 15.0
MCP_READ_TIMEOUT_SECONDS = 60.0
MCP_WRITE_TIMEOUT_SECONDS = 30.0
MCP_POOL_TIMEOUT_SECONDS = 15.0
# tools/call の実行タイムアウト（SDK が内部的に read タイムアウトを使うため、
# HTTP層の read よりやや長めを指定して read タイムアウトに先に当たるようにする）
MCP_CALL_READ_TIMEOUT_SECONDS = 90.0
MCP_MAX_REDIRECTS = 3
MCP_MAX_CONNECTIONS_PER_HOST = 4
# 接続確認 / tools/list の試行あたり制限
MCP_DISCOVER_MAX_BYTES = 512 * 1024

# ツール結果のサイズ上限（モデルコンテキスト肥大・ストリーム遅延の防止）
MCP_TOOL_RESULT_MAX_CHARS = 60_000

# ---------------------------------------------------------------------------
# リダイレクト / 許可
# ---------------------------------------------------------------------------
# MCPツール一覧のRedisキャッシュTTL（秒）
MCP_TOOLS_CACHE_TTL_SECONDS = 900
# OAuth 用の一時 state / 判断待ちの Redis TTL（秒）
MCP_OAUTH_STATE_TTL_SECONDS = 600
MCP_DECISION_WAIT_TTL_SECONDS = 300
MCP_DECISION_WAIT_POLL_INTERVAL = 0.5
# チャット中のMCP実行ラウンド上限
MCP_MAX_TOOL_ROUNDS_PER_CHAT = 20

# 同時にモデルへ公開してよいサーバー数・ツール総数の上限
MCP_MAX_ENABLED_SERVERS = 6
MCP_MAX_TOTAL_TOOLS = 60
# 各サーバーあたりモデルへ公開するツール上限（サーバー側が多い場合は先頭から採用）
MCP_MAX_TOOLS_PER_SERVER = 30

# ---------------------------------------------------------------------------
# ツール名の名前空間
# ---------------------------------------------------------------------------
MCP_TOOL_PREFIX = "mcp__"
# Gemini / OpenAI 等は関数名に 64 文字の上限がある
MCP_FUNCTION_NAME_MAX_LEN = 64
MCP_INTERNAL_TOOL_NAME_MAX_LEN = 60

# 読み取り判定に使う語幹
MCP_READONLY_VERBS = (
    "search", "list", "get", "read", "view", "find", "query", "fetch",
    "retrieve", "lookup", "describe", "show", "check", "count", "download",
    "export", "preview", "peek", "load", "summarize", "info", "whoami",
    "locate", "resolve", "head",
)
MCP_WRITE_VERBS = (
    "create", "send", "update", "delete", "insert", "move", "copy", "write",
    "edit", "upload", "post", "put", "patch", "remove", "rename", "trash",
    "restore", "share", "unshare", "comment", "reply", "forward", "draft",
    "batchdelete", "set", "add", "approve", "reject", "cancel", "subscribe",
    "unsubscribe", "star", "unstar", "mark", "archive", "tag", "untag",
)

# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
MCP_OAUTH_CALLBACK_PATH = "/api/mcp/oauth/callback"
MCP_OAUTH_REDIRECT_URI = None  # app側で https://<host> を解決して設定する

# 許可するスコープ数
MCP_OAUTH_MAX_SCOPES = 40

# ---------------------------------------------------------------------------
# 監査ログ
# ---------------------------------------------------------------------------
MCP_AUDIT_LOG_ENABLED = True

# ---------------------------------------------------------------------------
# プリセットの既定設定
# ---------------------------------------------------------------------------
# 認証フロー開始時に Google の authorization server から offline_access を
# 取得できる場合は要求するか
MCP_OAUTH_OFFLINE_ACCESS = True

# 認可要求・トークン要求へ RFC 8707 の resource パラメータを含めるか。
# MCP Authorization 仕様（2026-07-28）は MUST としているが、一部の
# authorization server（カスタム/将来のGoogle実装）が未対応の場合は
# False へ変更できるようにコード定数で持つ。
MCP_OAUTH_SEND_RESOURCE = True
