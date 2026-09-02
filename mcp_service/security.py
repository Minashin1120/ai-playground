"""URL検証・SSRF対策・秘密情報ガード。

ユーザー入力のMCP URLは信用せず、スキーム・フラグメント・宛先IPを検査する。
DNS再解決（TOCTOU）対策として、接続時にも解決済みIPを再検査する入口を提供する。
"""
from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from urllib.parse import urlparse, urlunparse

from .errors import MCPSecurityError

_SCHEME_RE = re.compile(r"^https?$", re.IGNORECASE)
# 解決結果のキャッシュ（TTL短め）
_RESOLVE_CACHE = {}
_RESOLVE_CACHE_LOCK = threading.Lock()
_RESOLVE_CACHE_TTL = 10.0

_BLOCKED_IP_REASONS = (
    "loopback", "link-local", "private", "reserved", "multicast",
    "unspecified", "metadata",
)


def _classify_ip(ip):
    """IPアドレスを分類し、ブロック理由を返す（許可なら None）。"""
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return "invalid-ip"
    if ip_obj.is_loopback:
        return "loopback"
    if ip_obj.is_link_local:
        # 169.254.0.0/16 にはクラウドメタデータ 169.254.169.254 も含まれる
        return "link-local"
    if ip_obj.is_private:
        return "private"
    if ip_obj.is_reserved:
        return "reserved"
    if ip_obj.is_multicast:
        return "multicast"
    if ip_obj.is_unspecified:
        return "unspecified"
    return None


def _resolve_host_cached(host):
    """ホスト名を解決し、全IP（文字列）のリストを返す。"""
    now = time.monotonic()
    with _RESOLVE_CACHE_LOCK:
        hit = _RESOLVE_CACHE.get(host)
        if hit and now - hit[0] < _RESOLVE_CACHE_TTL:
            return hit[1]
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:
        infos = []
    ips = []
    for info in infos:
        addr = info[4][0]
        if addr not in ips:
            ips.append(addr)
    with _RESOLVE_CACHE_LOCK:
        _RESOLVE_CACHE[host] = (now, ips)
    return ips


def clear_resolve_cache():
    with _RESOLVE_CACHE_LOCK:
        _RESOLVE_CACHE.clear()


def validate_mcp_url(url, *, resolve=True):
    """MCPサーバーURLを検証する。

    - http/https スキームのみ
    - フラグメント禁止・ユーザー情報禁止
    - ポートはホスト名/IP必須
    - resolve=True の場合、ホスト名の解決IPがブロック対象なら拒否
    戻り値: (scheme, host, port, path) の正規化済みタプル
    """
    if not url or not isinstance(url, str):
        raise MCPSecurityError("MCP URL is required.")
    url = url.strip()
    if len(url) > 2048:
        raise MCPSecurityError("MCP URL is too long.")
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise MCPSecurityError(f"Invalid MCP URL: {e}")
    if parsed.scheme not in ("http", "https"):
        raise MCPSecurityError("MCP URL must use http or https scheme.")
    if parsed.fragment:
        raise MCPSecurityError("MCP URL must not contain a fragment.")
    if parsed.username or parsed.password:
        raise MCPSecurityError("MCP URL must not contain userinfo.")
    host = parsed.hostname
    if not host:
        raise MCPSecurityError("MCP URL must include a host.")
    if not re.match(r"^[A-Za-z0-9_.\-\[\]:]+$", host):
        raise MCPSecurityError("MCP URL host contains invalid characters.")
    # IPリテラルはDNS解決なしで常に検査する（resolve=False でも安全側）
    if _host_is_ip_literal(host):
        reason = _classify_ip(host)
        if reason:
            raise MCPSecurityError(f"MCP URL target is blocked ({reason}).")
    elif resolve:
        ips = _resolve_host_cached(host)
        if not ips:
            raise MCPSecurityError("Could not resolve MCP URL host.")
        for ip in ips:
            reason = _classify_ip(ip)
            if reason:
                raise MCPSecurityError(
                    f"MCP URL host resolves to a blocked address ({reason})."
                )
    return parsed.scheme, host, parsed.port, (parsed.path or "/")


def _host_is_ip_literal(host):
    """ホストがIPリテラル（v4 or v6）かどうか。"""
    h = host.strip("[]")
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def is_redirect_allowed(location, *, original_scheme=None):
    """リダイレクト先URLが安全か検査する（http/https・ブロックIP拒否）。"""
    if not location:
        return False
    try:
        validate_mcp_url(location, resolve=True)
    except MCPSecurityError:
        return False
    except Exception:
        return False
    if original_scheme and original_scheme == "https":
        parsed = urlparse(location)
        if parsed.scheme != "https":
            return False
    return True


def normalize_redirect_location(base_url, location):
    """Location ヘッダー値を絶対URLへ解決する。"""
    from urllib.parse import urljoin
    return urljoin(base_url, location)
