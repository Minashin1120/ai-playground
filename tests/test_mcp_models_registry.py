import os
import unittest

os.environ.setdefault("FLASK_SECRET_KEY", "mcp-models-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-mcp-models-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target  # noqa: E402
from app import app as flask_app, db  # noqa: E402


class McpModelsRegistryTests(unittest.TestCase):
    """MCPテーブル・プリセット・レジストリCRUDのテスト。"""

    @classmethod
    def setUpClass(cls):
        with flask_app.app_context():
            db.create_all()
            from mcp_service.registry import get_or_create_presets
            get_or_create_presets()

    def _server_count(self):
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            return MCPServer.query.count()

    def test_mcp_tables_created(self):
        from mcp_service import models as m
        expected = {
            "mcp_servers",
            "mcp_user_connections",
            "mcp_user_credentials",
            "mcp_user_oauth_clients",
            "mcp_tool_permissions",
            "mcp_call_logs",
        }
        with flask_app.app_context():
            tables = {t.name for t in db.metadata.sorted_tables}
            for name in expected:
                with self.subTest(table=name):
                    self.assertIn(name, tables)
            self.assertIsNotNone(m.MCPServer.query.first())

    def test_presets_seeded(self):
        from mcp_service.models import MCPServer
        with flask_app.app_context():
            presets = MCPServer.query.filter_by(is_preset=True).all()
            self.assertGreaterEqual(len(presets), 8)
            slugs = {p.slug for p in presets}
            for slug in ("google_gmail", "google_drive", "google_docs", "google_sheets",
                         "google_slides", "google_calendar", "google_chat", "google_people"):
                self.assertIn(slug, slugs)
            gmail = MCPServer.query.filter_by(preset_key="google_gmail").first()
            self.assertEqual(gmail.auth_type, "oauth")
            self.assertEqual(gmail.oauth_provider_key, "google_workspace")
            self.assertIn("gmail.readonly", gmail.recommended_scopes or "")

    def test_register_and_list_custom_server(self):
        from mcp_service import registry as reg
        user_id = 424242
        with flask_app.app_context():
            try:
                srv = reg.register_custom(user_id, {
                    "name": "テストMCP", "url": "https://mcp.example.com/mcp",
                    "auth_type": "none", "description": "テスト用",
                })
                self.assertFalse(srv.is_preset)
                self.assertTrue(str(srv.slug).startswith("custom_"))
                # 同一URLは登録できない
                with self.assertRaises(ValueError):
                    reg.register_custom(user_id, {
                        "name": "重複", "url": "https://mcp.example.com/mcp", "auth_type": "none",
                    })
                listed = reg.list_servers_for_user(user_id)
                mine = [x for x in listed if str(x.get("slug")) == srv.slug]
                self.assertEqual(len(mine), 1)
                self.assertEqual(mine[0]["name"], "テストMCP")
                # 別ユーザーには見えない
                other = reg.list_servers_for_user(user_id + 1)
                self.assertNotIn(srv.slug, [x.get("slug") for x in other])
                reg.delete_custom(user_id, srv.id)
            finally:
                try:
                    db.session.rollback()
                except Exception:
                    pass

    def test_bearer_token_encryption_and_masking(self):
        from mcp_service import registry as reg
        from mcp_service.models import MCPUserCredential, MCPServer
        from app import decrypt_val
        user_id = 5151
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_drive").first()
            self.assertIsNotNone(srv)
            reg.save_bearer_token(user_id, srv.id, "sekret-token-123")
            cred = MCPUserCredential.query.filter_by(user_id=user_id, server_id=srv.id).first()
            self.assertIsNotNone(cred)
            self.assertIn("sekret", decrypt_val(cred.access_token_enc) or "")
            # DBには平文で保存されない
            self.assertNotIn("sekret", (cred.access_token_enc or ""))
            d = reg.server_to_api_dict(user_id, srv)
            self.assertTrue(d["auth_has_token"])
            self.assertNotIn("sekret", str(d))
            headers = reg.headers_for_server(user_id, srv)
            self.assertEqual(headers.get("Authorization"), "Bearer sekret-token-123")
            reg.clear_credentials(user_id, srv.id)
            self.assertIsNone(MCPUserCredential.query.filter_by(user_id=user_id, server_id=srv.id).first())

    def test_oauth_client_save_masked(self):
        from mcp_service import registry as reg
        from mcp_service.models import MCPUserOAuthClient
        user_id = 6161
        with flask_app.app_context():
            reg.save_oauth_client(user_id, "google_workspace", "cid-abc", "csecret-xyz")
            row = MCPUserOAuthClient.query.filter_by(user_id=user_id, provider_key="google_workspace").first()
            self.assertIsNotNone(row)
            self.assertTrue(row.has_client_info)
            self.assertNotIn("csecret", row.client_secret_enc or "")
            info = reg.decrypt_oauth_client(user_id, "google_workspace")
            self.assertEqual(info["client_id"], "cid-abc")
            self.assertEqual(info["client_secret"], "csecret-xyz")
            # マスクを送っても上書きされない
            reg.save_oauth_client(user_id, "google_workspace", "********", "********")
            info2 = reg.decrypt_oauth_client(user_id, "google_workspace")
            self.assertEqual(info2["client_id"], "cid-abc")
            self.assertEqual(info2["client_secret"], "csecret-xyz")
            row2 = MCPUserOAuthClient.query.filter_by(user_id=user_id, provider_key="google_workspace").first()
            db.session.delete(row2)
            db.session.commit()

    def test_tool_permission_and_decision(self):
        from mcp_service import registry as reg
        from mcp_service.models import MCPServer
        user_id = 7171
        with flask_app.app_context():
            srv = MCPServer.query.filter_by(preset_key="google_gmail").first()
            perm = reg.set_tool_permission(user_id, srv.id, "messages.send",
                                           allow=True, confirm="default", classified_read_only=False)
            self.assertEqual(perm.confirm, "default")
            allow, need, ro = reg.tool_decision(user_id, srv.id, "messages.send", auto_read_only=True)
            self.assertTrue(allow)
            # classified_read_only が明示False → 変更扱いで確認必要
            self.assertFalse(ro)
            self.assertTrue(need)
            reg.set_tool_permission(user_id, srv.id, "messages.send", confirm="never")
            _, need2, _ = reg.tool_decision(user_id, srv.id, "messages.send", auto_read_only=False)
            self.assertFalse(need2)

    def test_delete_user_mcp_data(self):
        from mcp_service import registry as reg
        user_id = 8181
        with flask_app.app_context():
            srv = reg.register_custom(user_id, {
                "name": "掃除用", "url": "https://cleanup.example.com/mcp", "auth_type": "bearer",
            })
            reg.save_bearer_token(user_id, srv.id, "tok")
            reg.delete_user_mcp_data(user_id)
            servers = reg.visible_servers_for_user(user_id)
            self.assertNotIn(srv.slug, [s.slug for s in servers])


if __name__ == "__main__":
    unittest.main()
