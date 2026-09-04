import os
import unittest


os.environ.setdefault("FLASK_SECRET_KEY", "mcp-release-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-mcp-release-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")


class McpReleaseAllowlistTests(unittest.TestCase):
    """公開用allowlistに mcp_service/ が含まれることを検証する。"""

    def _load_common(self):
        import importlib.util
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "_release_common_under_test", scripts / "_release_common.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_allowlist_includes_mcp_service(self):
        module = self._load_common()
        self.assertIn("mcp_service/", module.ALLOWED_GIT_PREFIXES)
        self.assertIn("server/", module.ALLOWED_GIT_PREFIXES)

    def test_mcp_service_files_would_be_allowed(self):
        module = self._load_common()
        for rel in ("mcp_service/__init__.py", "mcp_service/web.py", "mcp_service/client.py", "server/models.py", "server/README.md"):
            with self.subTest(rel=rel):
                self.assertTrue(module.is_allowed_git_path(rel), f"{rel} should be committable")

    def test_blocked_files_still_blocked(self):
        module = self._load_common()
        for rel in ("app.py.bak", "引き継ぎ資料.txt", ".env", "instance/x.txt", "mcp_service/secret.key"):
            with self.subTest(rel=rel):
                self.assertFalse(module.is_allowed_git_path(rel), f"{rel} must stay blocked")


if __name__ == "__main__":
    unittest.main()
