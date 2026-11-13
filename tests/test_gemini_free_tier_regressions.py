"""Gemini 有料枠／無料枠 APIキーの切り替え（プロンプトバーの「無料枠」）の回帰テスト。"""
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.app_source import read_app_source

os.environ.setdefault("FLASK_SECRET_KEY", "gemini-free-tier-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-free-tier-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class GeminiFreeTierRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username="gemini-free-tier", is_setup_completed=True)
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
        # These tests exercise credentials, not the API-level Turnstile gate.
        self._turnstile_gate_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self._turnstile_gate_patcher.start()
        self.addCleanup(self._turnstile_gate_patcher.stop)

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
        self.temp_dir.cleanup()

    def authenticated_client(self):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(self.user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def test_free_flag_uses_only_the_free_key_and_paid_stays_default(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.gemini_api_key = target.encrypt_val("paid-key")
            user.gemini_free_api_key = target.encrypt_val("free-key")
            target._save_user_model_api_key_map(user, {"gemini-3.6-flash": "model-key"})

            paid = target._resolve_chat_model_auth(user, "gemini-3.6-flash")
            self.assertIsNone(paid["error_code"])
            self.assertEqual(paid["api_key"], "model-key")

            paid_common = target._resolve_chat_model_auth(user, "gemini-2.5-pro")
            self.assertEqual(paid_common["api_key"], "paid-key")

            free = target._resolve_chat_model_auth(user, "gemini-3.6-flash", gemini_free_tier=True)
            self.assertIsNone(free["error_code"])
            self.assertEqual(free["api_key"], "free-key")
            self.assertEqual(free["gemini_runtime"]["api_tier"], "free")

    def test_free_flag_never_falls_back_to_paid_key_or_admin_environment(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            user.admin_api_key_mode = "env_fallback"
            user.gemini_api_key = target.encrypt_val("paid-key")
            user.gemini_free_api_key = None
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "admin-env-key"}):
                missing = target._resolve_chat_model_auth(user, "gemini-3.6-flash", gemini_free_tier=True)
            self.assertEqual(missing["error_code"], "api_key_missing")
            self.assertIsNone(missing["api_key"])
            self.assertEqual(missing["error"], "Gemini 無料枠のAPIキーが設定されていません。")

    def test_vertex_backend_ignores_the_free_flag(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.gemini_backend = "vertex_ai"
            user.gemini_vertex_project = target.encrypt_val("vertex-project")
            user.gemini_api_key = target.encrypt_val("paid-key")
            user.gemini_free_api_key = None

            resolved = target._resolve_chat_model_auth(user, "gemini-3.6-flash", gemini_free_tier=True)

            self.assertIsNone(resolved["error_code"])
            self.assertEqual(resolved["gemini_runtime"]["backend"], "vertex_ai")
            self.assertEqual(resolved["gemini_runtime"]["api_tier"], "paid")

    def test_non_gemini_models_ignore_the_free_flag(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.openai_api_key = target.encrypt_val("openai-key")
            user.gemini_free_api_key = None

            resolved = target._resolve_chat_model_auth(user, "gpt-5.4", gemini_free_tier=True)

            self.assertIsNone(resolved["error_code"])
            self.assertEqual(resolved["api_key"], "openai-key")

    def test_settings_round_trip_masks_both_keys_and_keeps_them_independent(self):
        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        response = client.post(
            "/api/settings",
            json={"gemini_key": "AIza-paid", "gemini_free_key": "AIza-free", "default_gemini_free_tier": True},
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        payload = client.get("/api/settings", base_url="https://localhost").get_json()
        self.assertEqual(payload["gemini_key"], "********")
        self.assertEqual(payload["gemini_free_key"], "********")
        self.assertIs(payload["default_gemini_free_tier"], True)

        # Posting the mask keeps the stored values, and clearing only the free key keeps the paid key.
        client.post(
            "/api/settings",
            json={"gemini_key": "********", "gemini_free_key": ""},
            headers=headers,
            base_url="https://localhost",
        )
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertEqual(target.decrypt_val(user.gemini_api_key), "AIza-paid")
            self.assertFalse(target.decrypt_val(user.gemini_free_api_key))

    def test_browser_fast_helper_returns_only_the_requested_tier_key(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.gemini_api_key = target.encrypt_val("paid-key")
            user.gemini_free_api_key = target.encrypt_val("free-key")
            target._save_user_model_api_key_map(user, {"gemini-2.5-flash": "model-key"})

            paid_key, paid_source = target._get_browser_fast_mode_user_key(user, "gemini-2.5-flash")
            self.assertEqual((paid_key, paid_source), ("model-key", "model_specific"))

            free_key, free_source = target._get_browser_fast_mode_user_key(
                user, "gemini-2.5-flash", use_free=True)
            self.assertEqual((free_key, free_source), ("free-key", "gemini_free"))

            user.gemini_free_api_key = None
            self.assertEqual(target._get_browser_fast_mode_user_key(
                user, "gemini-2.5-flash", use_free=True), (None, None))

    def test_prompt_options_record_keeps_the_tier_flag(self):
        record = target._build_prompt_options_record({"enable_search": True, "gemini_free_tier": True})
        self.assertEqual(json.loads(record)["gemini_free_tier"], True)
        self.assertEqual(json.loads(record)["enable_search"], True)

    def test_tier_log_does_not_include_the_key(self):
        source = read_app_source()
        self.assertIn(
            'logger.info( "gemini_api_tier user=%s model=%s tier=%s backend=%s"',
            " ".join(source.split()),
        )


if __name__ == "__main__":
    unittest.main()
