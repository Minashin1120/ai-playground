import os
import unittest
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "mcp-exec-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-mcp-exec-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target  # noqa: E402
from app import app as flask_app, db  # noqa: E402


class MemoryRedis:
    def __init__(self):
        self.data = {}

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)
        return 1


class McpExecutionTests(unittest.TestCase):
    """チャット実行時McpRuntimeの動作（判断待ち・拒否・結果整形）テスト。"""

    @classmethod
    def setUpClass(cls):
        with flask_app.app_context():
            db.create_all()
            from mcp_service.registry import get_or_create_presets
            get_or_create_presets()

    def test_denied_write_returns_rejected(self):
        from mcp_service.execution import McpRuntime, MCPToolMeta
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_gmail").first()
            meta = MCPToolMeta(
                server=srv, server_id=srv.id, server_slug=srv.slug, server_name=srv.name,
                url=srv.url, internal_name="mcp__google_gmail__messages_send",
                name="messages.send", description="Send an email",
                input_schema={"type": "object", "properties": {}},
                is_read_only=False, confirm_policy="default", allow=True,
            )
            rt = McpRuntime(1234, job_id="job_20260101_1", pub=None, check_stop=lambda: False,
                            redis_client=MemoryRedis())
            rt._by_internal[meta.internal_name] = meta
            with mock.patch.object(rt, "_wait_decision", return_value="deny") as wd:
                text, mout = rt.execute(meta.internal_name, {"to": "a@example.com"}, allow_decision=True)
            self.assertTrue(mout.get("rejected"))
            wd.assert_called_once()

    def test_allowed_write_runs_remote(self):
        from mcp_service.execution import McpRuntime, MCPToolMeta
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_gmail").first()
            meta = MCPToolMeta(
                server=srv, server_id=srv.id, server_slug=srv.slug, server_name=srv.name,
                url=srv.url, internal_name="mcp__google_gmail__messages_send",
                name="messages.send", description="Send an email",
                input_schema={"type": "object", "properties": {}},
                is_read_only=False, confirm_policy="default", allow=True,
            )
            rt = McpRuntime(1234, job_id="job_20260101_2", pub=None, check_stop=lambda: False,
                            redis_client=MemoryRedis())
            rt._by_internal[meta.internal_name] = meta
            with mock.patch.object(rt, "_wait_decision", return_value="allow"), \
                 mock.patch("mcp_service.client.call_tool") as call:
                call.return_value = {
                    "text": '{"id":"msg_1"}', "is_error": False,
                    "content": [{"type": "text", "text": '{"id":"msg_1"}'}],
                    "size_bytes": 20, "structured_content": None,
                }
                text, mout = rt.execute(meta.internal_name, {"to": "a@example.com"}, allow_decision=True)
            self.assertTrue(mout.get("ok"))
            self.assertIn("msg_1", text)

    def test_unknown_internal_name_is_error_text(self):
        from mcp_service.execution import McpRuntime
        rt = McpRuntime(1234, job_id="job_20260101_3", redis_client=MemoryRedis())
        rt._loaded = True
        text, mout = rt.execute("mcp__none__missing", {})
        self.assertFalse(mout.get("ok"))
        self.assertIn("Unknown MCP tool", text)

    def test_load_with_no_enabled_servers_is_empty(self):
        from mcp_service.execution import McpRuntime
        with flask_app.app_context():
            rt = McpRuntime(987654, job_id="job_20260101_4")
            self.assertTrue(rt.empty())


if __name__ == "__main__":
    unittest.main()
