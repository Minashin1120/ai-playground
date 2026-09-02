"""MCP外部連携の型付き例外。"""
from __future__ import annotations


class MCPError(Exception):
    """MCP連携の基底エラー。"""

    code = "mcp_error"
    http_status = 400

    def __init__(self, message="MCP error", *, detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self):
        d = {"error": self.message, "code": self.code}
        if self.detail is not None:
            d["detail"] = self.detail
        return d


class MCPValidationError(MCPError):
    code = "mcp_validation_error"
    http_status = 400


class MCPSecurityError(MCPError):
    """SSRF対策等で接続先を拒否したとき。"""

    code = "mcp_security_error"
    http_status = 400


class MCPAuthRequiredError(MCPError):
    """認証（OAuth/Bearer）が未完了・期限切れのとき。"""

    code = "mcp_auth_required"
    http_status = 401


class MCPInsufficientScopeError(MCPError):
    """403 insufficient_scope を受け取ったとき。"""

    code = "mcp_insufficient_scope"
    http_status = 403

    def __init__(self, message="Additional authorization is required", *, scope=None, detail=None):
        super().__init__(message, detail=detail)
        self.scope = scope


class MCPNotFoundError(MCPError):
    code = "mcp_not_found"
    http_status = 404


class MCPTimeoutError(MCPError):
    code = "mcp_timeout"
    http_status = 504


class MCPConnectionError(MCPError):
    code = "mcp_connection_error"
    http_status = 502


class MCPToolError(MCPError):
    """tools/call が isError を返したとき（チャット継続用）。"""

    code = "mcp_tool_error"
    http_status = 200

    def __init__(self, message, *, detail=None):
        super().__init__(message, detail=detail)


class MCPToolNotFoundError(MCPToolError):
    code = "mcp_tool_not_found"


class MCPDecisionDeniedError(MCPToolError):
    """ユーザーが変更操作の実行を拒否したとき。"""

    code = "mcp_decision_denied"
