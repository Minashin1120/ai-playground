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

    def test_guidance_and_serialize_tell_model_about_mcp(self):
        from mcp_service.execution import McpRuntime, MCPToolMeta
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_gmail").first()
            rt = McpRuntime(1234, job_id="job_20260101_5", redis_client=MemoryRedis())
            rt._loaded = True
            meta = MCPToolMeta(
                server=srv, server_id=srv.id, server_slug=srv.slug, server_name=srv.name,
                url=srv.url, internal_name="mcp__google_gmail__search_messages",
                name="search_messages", title="", description="Search Gmail messages",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                is_read_only=True, confirm_policy="never", allow=True,
            )
            rt.servers = [{"server": srv, "tools": [meta]}]
            rt._by_internal[meta.internal_name] = meta
            guidance = rt.guidance_text()
            self.assertIn("Model Context Protocol", guidance)
            self.assertIn("mcp__google_gmail__search_messages", guidance)
            self.assertIn(srv.name, guidance)
            openai = rt.serialize_openai()
            self.assertEqual(openai[0]["name"], "mcp__google_gmail__search_messages")
            self.assertIn("Model Context Protocol", openai[0]["description"])
            self.assertIn("Search Gmail messages", openai[0]["description"])
            cc = rt.serialize_chat_completions()
            self.assertIn("Model Context Protocol", cc[0]["function"]["description"])
            anth = rt.serialize_anthropic()
            self.assertIn("Model Context Protocol", anth[0]["description"])

    def test_guidance_text_uses_editable_preamble_and_expands_token(self):
        from mcp_service.execution import McpRuntime, MCPToolMeta
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_gmail").first()
            rt = McpRuntime(1234, job_id="job_20260101_6", redis_client=MemoryRedis())
            rt._loaded = True
            meta = MCPToolMeta(
                server=srv, server_id=srv.id, server_slug=srv.slug, server_name=srv.name,
                url=srv.url, internal_name="mcp__google_gmail__search_messages",
                name="search_messages", title="", description="Search Gmail messages",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                is_read_only=True, confirm_policy="never", allow=True,
            )
            rt.servers = [{"server": srv, "tools": [meta]}]
            rt._by_internal[meta.internal_name] = meta
            expanded = rt.guidance_text(preamble="カスタム案内:\n{{mcp_tools}}")
            self.assertIn("カスタム案内:", expanded)
            self.assertIn("mcp__google_gmail__search_messages", expanded)
            self.assertNotIn("{{mcp_tools}}", expanded)
            appended = rt.guidance_text(preamble="自由記述のみ")
            self.assertIn("自由記述のみ", appended)
            self.assertIn("Connected MCP tools:", appended)
            self.assertIn(srv.name, appended)


if __name__ == "__main__":
    unittest.main()
