"""MCP Python SDK v2 による接続・tools/list・tools/call（同期ラッパ）。

- SDK（`mcp`）はメモリ制約のためこのモジュールの関数内でのみ遅延 import する。
- 接続の都度 `security.validate_mcp_url` で宛先を再検査する（SSRF / DNS 再解決対策）。
- タイムアウト・証明書検証はコード既定値（`config.py`）のみを使用し、
  検証無効化オプションは提供しない。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

from . import config
from . import security
from . import tools as mcp_tools
from .errors import (
    MCPAuthRequiredError,
    MCPConnectionError,
    MCPInsufficientScopeError,
    MCPSecurityError,
    MCPTimeoutError,
    MCPToolError,
)


def _translate_sdk_error(exc, *, op="mcp_request"):
    """SDK / httpx2 の例外を型付きMCPエラーへ変換する。"""
    text = str(exc)
    lowered = text.lower()
    # 例外チェーンを辿ってHTTPステータスを探す
    status = None
    seen = exc
    chain = []
    while seen is not None and seen not in chain:
        chain.append(seen)
        try:
            resp = getattr(seen, "response", None)
            if resp is not None:
                status = getattr(resp, "status_code", None) or status
        except Exception:
            pass
        seen = getattr(seen, "__cause__", None) or getattr(seen, "__context__", None)
    if status is None:
        for token in ("401", "unauthorized", "authentication required", "invalid token", "access token"):
            if token in lowered:
                return MCPAuthRequiredError(f"MCP server returned an authentication error: {text}")
        if status is None:
            for token in ("403", "insufficient_scope", "forbidden", "permission"):
                if token in lowered:
                    return MCPInsufficientScopeError(f"MCP server denied access (insufficient scope?): {text}")
    if isinstance(exc, TimeoutError):
        return MCPTimeoutError(f"MCP server timed out during {op}.")
    for token in ("timeout", "timed out", "read timeout", "connect timeout", "deadline"):
        if token in lowered:
            return MCPTimeoutError(f"MCP server timed out during {op}: {text[:300]}")
    for token in ("connection refused", "connect error", "connection error", "connection reset", "network", "dns", "name or service not known", "unreachable", "ssl", "certificate", "tls"):
        if token in lowered:
            return MCPConnectionError(f"MCP connection failed during {op}: {text[:300]}")
    return MCPConnectionError(f"MCP request failed during {op}: {text[:300]}")


def _http_headers_from_kwargs(**kwargs):
    return kwargs.get("headers") or {}


def _build_httpx_client(headers, read_timeout, call=False):
    import httpx2

    connect_timeout = config.MCP_CONNECT_TIMEOUT_SECONDS
    if call:
        read_timeout = config.MCP_CALL_READ_TIMEOUT_SECONDS
    return httpx2.AsyncClient(
        headers=headers or None,
        timeout=httpx2.Timeout(
            connect_timeout,
            read=read_timeout,
            write=config.MCP_WRITE_TIMEOUT_SECONDS,
            pool=config.MCP_POOL_TIMEOUT_SECONDS,
        ),
        follow_redirects=False,
        verify=True,
    )


async def _open_client(url, headers, read_timeout, call=False):
    # 接続前に毎回URLを再検証（DNS再解決対策）
    security.validate_mcp_url(url, resolve=True)
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    http_client = _build_httpx_client(headers, read_timeout, call=call)
    transport = streamable_http_client(url, http_client=http_client)
    client = Client(transport)
    return client


def _run(coro):
    try:
        return asyncio.run(coro)
    except Exception as e:
        # asyncio.run でラップされることがある ExceptionGroup
        if hasattr(e, "exceptions"):
            inner = e.exceptions[0] if e.exceptions else e
            raise inner
        raise


def fetch_tools(url, headers=None, *, read_timeout=None, max_tools=None):
    """tools/list を取得して正規化リストを返す。"""
    from mcp import MCPError
    from mcp.types import Tool  # noqa: F401

    if headers is None:
        headers = {}
    read_timeout = read_timeout or config.MCP_READ_TIMEOUT_SECONDS
    security.validate_mcp_url(url, resolve=True)

    async def _list():
        client = await _open_client(url, headers, read_timeout)
        async with client:
            tools = []
            cursor = None
            while True:
                page = await client.list_tools(cursor=cursor)
                for tool in page.tools:
                    tools.append(mcp_tools.tool_spec_from_sdk_tool(tool))
                    if max_tools and len(tools) >= max_tools:
                        return tools
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            return tools

    try:
        return _run(_list())
    except MCPError:
        raise
    except Exception as exc:
        translated = _translate_sdk_error(exc, op="tools/list")
        if isinstance(translated, MCPAuthRequiredError) or isinstance(translated, MCPInsufficientScopeError):
            raise translated
        raise translated


def call_tool(url, headers=None, *, tool_name, arguments, read_timeout=None):
    """tools/call を実行して正規化結果 dict を返す。

    戻り値:
      {content: [...], text: str, is_error: bool, structured_content: ...,
       started_at: iso, duration_ms: int, size_bytes: int}
    """
    if headers is None:
        headers = {}
    security.validate_mcp_url(url, resolve=True)
    started = datetime.utcnow()
    started_ms = int(time.time() * 1000)

    async def _call():
        client = await _open_client(url, headers, read_timeout or config.MCP_CALL_READ_TIMEOUT_SECONDS, call=True)
        async with client:
            result = await client.call_tool(tool_name, arguments or {})
            return result

    try:
        result = _run(_call())
    except Exception as exc:
        translated = _translate_sdk_error(exc, op=f"tools/call:{tool_name}")
        raise translated

    duration_ms = int(time.time() * 1000) - started_ms
    content = []
    if result.content is not None:
        for block in result.content:
            try:
                if hasattr(block, "model_dump"):
                    bd = block.model_dump(exclude_none=True)
                elif isinstance(block, dict):
                    bd = block
                else:
                    bd = {}
            except Exception:
                bd = {}
            content.append(bd)
    text = mcp_tools.content_blocks_to_text({"content": content, "structured_content": getattr(result, "structured_content", None)})
    text = (text or "")[: config.MCP_TOOL_RESULT_MAX_CHARS]
    return {
        "content": content,
        "text": text,
        "is_error": bool(getattr(result, "is_error", False)),
        "structured_content": getattr(result, "structured_content", None),
        "started_at": started.isoformat(),
        "duration_ms": duration_ms,
        "size_bytes": len(text.encode("utf-8", errors="replace")),
    }
