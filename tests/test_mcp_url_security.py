import os
import socket
import unittest
from unittest import mock

from mcp_service.errors import MCPSecurityError
from mcp_service import security


def _fake_resolve(ips):
    def _getaddrinfo(host, port=None, *a, **k):
        infos = []
        for ip in ips:
            infos.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)))
        return infos
    return _getaddrinfo


class McpUrlSecurityTests(unittest.TestCase):
    """MCP URLのSSRF対策（スキーム・IP・DNS解決）のテスト。"""

    def tearDown(self):
        security.clear_resolve_cache()

    def test_rejects_bad_schemes_and_fragments(self):
        for url in ("ftp://example.com/mcp", "file:///etc/passwd", "gopher://x", ""):
            with self.subTest(url=url):
                with self.assertRaises(MCPSecurityError):
                    security.validate_mcp_url(url, resolve=False)
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://example.com/mcp#frag", resolve=False)
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://user:pass@example.com/mcp", resolve=False)

    @mock.patch("mcp_service.security.socket.getaddrinfo")
    def test_blocks_loopback_and_private_resolutions(self, ga):
        ga.side_effect = _fake_resolve(["127.0.0.1"])
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://internal.example.com/mcp", resolve=True)
        ga.side_effect = _fake_resolve(["10.0.0.5"])
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://corp.example.com/mcp", resolve=True)
        ga.side_effect = _fake_resolve(["192.168.1.10"])
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://router.example.com/mcp", resolve=True)

    def test_blocks_private_ip_literals(self):
        for ip in ("127.0.0.1", "10.1.2.3", "172.16.5.5", "192.168.0.1", "169.254.169.254"):
            with self.subTest(ip=ip):
                with self.assertRaises(MCPSecurityError):
                    security.validate_mcp_url(f"https://{ip}/mcp", resolve=False)
        for ip in ("::1", "fd00::1", "fe80::1"):
            with self.subTest(ip=ip):
                with self.assertRaises(MCPSecurityError):
                    security.validate_mcp_url(f"https://[{ip}]/mcp", resolve=False)

    @mock.patch("mcp_service.security.socket.getaddrinfo")
    def test_metadata_ip_is_blocked_even_when_mixed(self, ga):
        # 解決結果に1つでもブロック対象が含まれれば拒否
        ga.side_effect = _fake_resolve(["169.254.169.254", "93.184.216.34"])
        with self.assertRaises(MCPSecurityError):
            security.validate_mcp_url("https://target.example.com/mcp", resolve=True)

    @mock.patch("mcp_service.security.socket.getaddrinfo")
    def test_public_url_passes(self, ga):
        ga.side_effect = _fake_resolve(["93.184.216.34"])
        scheme, host, port, path = security.validate_mcp_url("https://mcp.example.com/v1/mcp", resolve=True)
        self.assertEqual(scheme, "https")
        self.assertEqual(host, "mcp.example.com")
        self.assertEqual(path, "/v1/mcp")

    @mock.patch("mcp_service.security.socket.getaddrinfo")
    def test_redirect_validation(self, ga):
        ga.side_effect = _fake_resolve(["93.184.216.34"])
        self.assertTrue(security.is_redirect_allowed("https://public.example.com/mcp"))
        security.clear_resolve_cache()
        self.assertFalse(security.is_redirect_allowed("http://127.0.0.1/"))
        self.assertFalse(security.is_redirect_allowed("ftp://x/"))


if __name__ == "__main__":
    unittest.main()
