"""チャット実行時（RQワーカー内）のMCPツール収集・実行・判断待ち。

- 有効かつ認証済みのMCPサーバーからツール一覧を取得（Redisキャッシュ付き）。
- 内部ツール名 ``mcp__{slug}__{tool}`` での呼び出しをサーバーへディスパッチする。
- 変更操作はユーザー確認（mcp_decision_request → 判断待ち → 結果）を挟む。
- プロバイダ別シリアライズ（OpenAI互換 / Anthropic / Gemini 用の素材）を提供する。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime

from . import config
from . import client as mcp_client
from . import registry as mcp_registry
from . import tools as mcp_tools
from .errors import (
    MCPAuthRequiredError,
    MCPDecisionDeniedError,
    MCPError,
    MCPInsufficientScopeError,
    MCPNotFoundError,
    MCPToolError,
    MCPToolNotFoundError,
)

logger = logging.getLogger(__name__)

_DECISION_PREFIX = "mcp_decision:"
_MCP_DISABLED_SENTINEL = object()

_decision_lock = threading.Lock()


class MCPToolMeta:
    """モデルへ公開する1ツール分のメタ情報。"""

    __slots__ = (
        "server", "server_id", "server_slug", "server_name", "url",
        "internal_name", "name", "title", "description", "input_schema",
        "is_read_only", "confirm_policy", "allow",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class McpRuntime:
    """1回のチャット生成で使うMCP実行環境。"""

    def __init__(self, user_id, *, job_id=None, pub=None, check_stop=None,
                 redis_client=None, log=None):
        self.user_id = user_id
        self.job_id = job_id
        self.pub = pub
        self.check_stop = check_stop
        self._redis_client = redis_client
        self._log = log or (lambda msg: None)
        self.servers = []
        self._by_internal = {}
        self._loaded = False
        self._fetch_error = None

    # ------------------------------------------------------------------ redis
    def _redis(self):
        if self._redis_client is not None:
            return self._redis_client
        import app as _a
        return _a.redis_conn

    def _dbg(self, msg):
        try:
            self._log(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ load
    def _visible_connections(self):
        """ユーザーが有効化したサーバーと接続情報のリスト。"""
        from mcp_service.models import MCPUserConnection
        from mcp_service.models import MCPServer
        rows = (
            MCPUserConnection.query.join(
                MCPServer, MCPUserConnection.server_id == MCPServer.id
            )
            .filter(
                MCPUserConnection.user_id == self.user_id,
                MCPUserConnection.is_enabled.is_(True),
            )
            .all()
        )
        servers = []
        for conn in rows:
            srv = MCPServer.query.get(conn.server_id)
            if srv is None:
                continue
            if not (srv.is_preset or srv.owner_user_id == self.user_id):
                continue
            servers.append(srv)
        return servers

    def _auth_ok(self, srv):
        if srv.auth_type_safe == "none":
            return True
        cred = mcp_registry.get_credential(self.user_id, srv.id)
        return bool(cred and cred.access_token_enc)

    def _fetch_server_tools(self, srv):
        cached = mcp_registry.get_cached_tools(self.user_id, srv.id)
        if cached is not None:
            return cached
        try:
            headers = mcp_registry.headers_for_server(self.user_id, srv)
            tools = mcp_client.fetch_tools(
                srv.url, headers=headers,
                read_timeout=config.MCP_READ_TIMEOUT_SECONDS,
                max_tools=config.MCP_MAX_TOOLS_PER_SERVER,
            )
        except Exception as exc:
            self._dbg(f"MCP fetch tools failed for {srv.slug}: {exc}")
            # 短時間だけ失敗をキャッシュし、毎チャット再試行しない
            try:
                mcp_registry.set_cached_tools(self.user_id, srv.id, [], ttl=120)
            except Exception:
                pass
            return None
        if not tools:
            return []
        mcp_registry.set_cached_tools(self.user_id, srv.id, tools)
        return tools

    def load(self, *, skip_errors=True):
        """有効かつ認証済みサーバーのツールを収集する。"""
        if self._loaded:
            return self.servers
        self._loaded = True
        conn_servers = self._visible_connections()
        if len(conn_servers) > config.MCP_MAX_ENABLED_SERVERS:
            conn_servers = conn_servers[: config.MCP_MAX_ENABLED_SERVERS]
        total_tools = 0
        used = set()
        for srv in conn_servers:
            if srv.id in used:
                continue
            used.add(srv.id)
            if not self._auth_ok(srv):
                continue
            try:
                raw_tools = self._fetch_server_tools(srv)
            except Exception:
                raw_tools = None
            if not raw_tools:
                continue
            metas = []
            for tool in raw_tools:
                name = tool.get("name")
                if not name:
                    continue
                if config.MCP_MAX_TOOLS_PER_SERVER and len(metas) >= config.MCP_MAX_TOOLS_PER_SERVER:
                    break
                auto_read = mcp_tools.classify_readonly(name, tool.get("description") or "")
                try:
                    perm = mcp_registry.get_tool_permission(self.user_id, srv.id, name)
                except Exception:
                    perm = None
                if perm is not None and not perm.allow:
                    continue
                read_only = auto_read
                if perm is not None and perm.classified_read_only is not None:
                    read_only = bool(perm.classified_read_only)
                confirm_policy = "default"
                if perm is not None and perm.confirm in ("always", "never"):
                    confirm_policy = perm.confirm
                needs_confirm = (
                    confirm_policy == "always"
                    or (confirm_policy == "default" and read_only is False)
                )
                internal_name = mcp_tools.make_internal_tool_name(srv.slug, name)
                meta = MCPToolMeta(
                    server=srv,
                    server_id=srv.id,
                    server_slug=srv.slug,
                    server_name=srv.name,
                    url=srv.url,
                    internal_name=internal_name,
                    name=name,
                    title=tool.get("title") or "",
                    description=tool.get("description") or "",
                    input_schema=tool.get("input_schema") or {"type": "object"},
                    is_read_only=read_only,
                    confirm_policy=confirm_policy,
                    allow=True,
                )
                metas.append(meta)
                self._by_internal[internal_name] = meta
            if metas:
                self.servers.append({"server": srv, "tools": metas})
                total_tools += len(metas)
                if total_tools >= config.MCP_MAX_TOTAL_TOOLS:
                    break
        return self.servers

    # ------------------------------------------------------------------ list
    def tool_metas(self):
        self.load()
        out = []
        for s in self.servers:
            out.extend(s["tools"])
        return out

    def empty(self):
        try:
            return not self.tool_metas()
        except Exception:
            return True

    def serialize_openai(self):
        """OpenAI Responses API / Chat Completions 用の function 定義リスト。"""
        return [
            mcp_tools.to_openai_function_schema(m.internal_name, {
                "name": m.name, "description": m.description, "input_schema": m.input_schema,
            })
            for m in self.tool_metas()
        ]

    def serialize_anthropic(self):
        return [
            mcp_tools.to_anthropic_tool(m.internal_name, {
                "name": m.name, "description": m.description, "input_schema": m.input_schema,
            })
            for m in self.tool_metas()
        ]

    # ------------------------------------------------------------------ exec
    def _meta_for(self, internal_name):
        if internal_name in self._by_internal:
            return self._by_internal[internal_name]
        # load() 前に呼ばれた場合の安全網
        self.load()
        return self._by_internal.get(internal_name)

    def _emit(self, payload):
        if self.pub:
            try:
                self.pub("mcp", payload)
            except Exception:
                pass

    def _emit_decision_request(self, meta, args_preview):
        payload = {
            "type": "decision_request",
            "id": self._call_id(meta),
            "server_name": meta.server_name,
            "server_slug": meta.server_slug,
            "tool_name": meta.name,
            "args_preview": args_preview,
        }
        if self.pub:
            try:
                self.pub("mcp_decision_request", payload)
            except Exception:
                pass
        return payload

    def _call_id(self, meta):
        return f"{meta.server_slug}_{int(time.time() * 1000)}_{self.user_id}"

    def _wait_decision(self, meta, payload):
        """ユーザー判断を待つ。デフォルトはタイムアウトで拒否。"""
        if not self.job_id:
            return "allow"  # ジョブ外（テスト接続等）では実行を許可しない呼び出し側が制御
        decision_key = _DECISION_PREFIX + self.job_id
        try:
            self._redis().setex(decision_key, config.MCP_DECISION_WAIT_TTL_SECONDS, "waiting")
        except Exception:
            return "deny"
        deadline = time.time() + config.MCP_DECISION_WAIT_TTL_SECONDS
        while time.time() < deadline:
            if self.check_stop is not None:
                try:
                    if self.check_stop():
                        return "deny"
                except Exception:
                    pass
            try:
                val = self._redis().get(decision_key)
                if val in (b"allow", "allow"):
                    return "allow"
                if val in (b"deny", "deny"):
                    return "deny"
            except Exception:
                pass
            time.sleep(config.MCP_DECISION_WAIT_POLL_INTERVAL)
        try:
            self._redis().delete(decision_key)
        except Exception:
            pass
        return "deny"

    def _record_decision(self, decision):
        if not self.job_id:
            return
        try:
            self._redis().setex(_DECISION_PREFIX + self.job_id, 60, decision)
        except Exception:
            pass

    def _audit(self, meta, status, started, duration_ms, result_size=None, http_status=None, error_code=None):
        if not config.MCP_AUDIT_LOG_ENABLED:
            return
        try:
            from app import db
            from mcp_service.models import MCPCallLog
            row = MCPCallLog(
                user_id=self.user_id,
                server_id=meta.server_id,
                server_name=meta.server_name,
                tool_name=meta.name,
                status=status,
                http_status=http_status,
                started_at=started,
                duration_ms=duration_ms,
                result_size_bytes=result_size,
                error_code=error_code,
            )
            db.session.add(row)
            db.session.commit()
        except Exception:
            try:
                from app import db
                db.session.rollback()
            except Exception:
                pass

    def execute(self, internal_name, arguments, *, allow_decision=True):
        """MCPツールを実行し、(model_text, meta_dict) を返す。失敗時もテキストを返す。"""
        meta = self._meta_for(internal_name)
        if meta is None:
            return "Error: Unknown MCP tool: " + str(internal_name), {"ok": False}
        started = datetime.utcnow()
        started_ms = int(time.time() * 1000)
        args = arguments if isinstance(arguments, dict) else {}
        args_preview = json.dumps(args, ensure_ascii=False)[:2000]

        # 実行前確認
        if meta.confirm_policy == "always":
            needs_confirm = True
        elif meta.confirm_policy == "never":
            needs_confirm = False
        else:
            needs_confirm = meta.is_read_only is False
        needs_confirm = needs_confirm and allow_decision
        if needs_confirm:
            payload = self._emit_decision_request(meta, args_preview)
            decision = self._wait_decision(meta, payload)
            self._emit({"type": "decision_resolved", "id": payload["id"], "decision": decision})
            if decision != "allow":
                text = "Tool execution was canceled because the user did not approve it."
                self._audit(meta, "rejected", started, int(time.time() * 1000) - started_ms)
                return text, {"ok": False, "rejected": True, "id": payload["id"]}

        call_id = self._call_id(meta)
        self._emit({
            "type": "start",
            "id": call_id,
            "server_name": meta.server_name,
            "server_slug": meta.server_slug,
            "tool_name": meta.name,
            "internal_name": internal_name,
            "read_only": meta.is_read_only,
        })
        try:
            headers = mcp_registry.headers_for_server(self.user_id, meta.server)
            result = mcp_client.call_tool(
                meta.url, headers=headers, tool_name=meta.name, arguments=args
            )
        except MCPAuthRequiredError as exc:
            self._audit(meta, "error", started, int(time.time() * 1000) - started_ms, error_code="auth_required")
            mcp_registry.set_connection_state(self.user_id, meta.server_id, "needs_auth",
                                              last_error="認証の有効期限切れまたは無効です。再認証してください。")
            text = ("MCPツールの実行に認証が必要です（トークンの期限切れ等）。"
                    "設定の「MCP」タブから再認証してください。")
            self._emit({"type": "error", "id": call_id, "message": text,
                        "server_name": meta.server_name, "tool_name": meta.name})
            return text, {"ok": False, "auth_required": True, "id": call_id}
        except MCPInsufficientScopeError as exc:
            self._audit(meta, "error", started, int(time.time() * 1000) - started_ms, error_code="insufficient_scope")
            text = ("MCPツールの実行に追加の権限が必要です（insufficient_scope）。"
                    "設定の「MCP」タブから再認証して権限を追加してください。")
            self._emit({"type": "error", "id": call_id, "message": text,
                        "server_name": meta.server_name, "tool_name": meta.name})
            return text, {"ok": False, "insufficient_scope": True, "id": call_id}
        except MCPToolError as exc:
            self._audit(meta, "error", started, int(time.time() * 1000) - started_ms, error_code=exc.code)
            self._emit({"type": "error", "id": call_id, "message": str(exc),
                        "server_name": meta.server_name, "tool_name": meta.name})
            return f"Error: {exc}", {"ok": False, "id": call_id}
        except MCPError as exc:
            self._audit(meta, "error", started, int(time.time() * 1000) - started_ms, error_code=exc.code)
            self._emit({"type": "error", "id": call_id, "message": str(exc),
                        "server_name": meta.server_name, "tool_name": meta.name})
            return f"Error: {exc}", {"ok": False, "id": call_id}
        except Exception as exc:
            logger.warning("MCP execute unexpected error: %s", exc)
            self._audit(meta, "error", started, int(time.time() * 1000) - started_ms, error_code="unexpected")
            text = f"Error executing MCP tool: {exc}"
            self._emit({"type": "error", "id": call_id, "message": text,
                        "server_name": meta.server_name, "tool_name": meta.name})
            return text, {"ok": False, "id": call_id}

        duration_ms = int(time.time() * 1000) - started_ms
        if result.get("is_error"):
            self._audit(meta, "error", started, duration_ms,
                        result_size=result.get("size_bytes"), error_code="tool_error")
            text = result.get("text") or "MCP tool returned an error."
            self._emit({"type": "error", "id": call_id, "message": text[:2000],
                        "server_name": meta.server_name, "tool_name": meta.name})
            return text, {"ok": False, "is_error": True, "id": call_id}

        self._audit(meta, "ok", started, duration_ms, result_size=result.get("size_bytes"))
        text = result.get("text") or ""
        self._emit({"type": "result", "id": call_id,
                    "server_name": meta.server_name, "tool_name": meta.name,
                    "summary": text[:1500]})
        return text, {"ok": True, "id": call_id}


# ---------------------------------------------------------------------------
# 判断の受け付け（Webプロセス→ワーカー）
# ---------------------------------------------------------------------------
def submit_chat_decision(job_id, decision):
    """チャット中のMCP変更操作の判断をRedisへ書き込む。"""
    import redis as _redis
    if decision not in ("allow", "deny"):
        raise ValueError("decision must be allow or deny")
    key = _DECISION_PREFIX + job_id
    r = None
    try:
        import app as _a
        r = _a.redis_conn
    except Exception:
        pass
    if r is None:
        from app import REDIS_URL
        r = _redis.from_url(REDIS_URL)
    r.setex(key, config.MCP_DECISION_WAIT_TTL_SECONDS, decision)
    return True
