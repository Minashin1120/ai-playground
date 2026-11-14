"""Gemini Flex 推論（service_tier="flex"）の回帰テスト。"""
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.app_source import read_app_source

os.environ.setdefault("FLASK_SECRET_KEY", "gemini-flex-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-flex-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class GeminiFlexRegressionTests(unittest.TestCase):
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
            user = target.User(username="gemini-flex", is_setup_completed=True)
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
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

    def test_flex_supported_model_list_matches_documented_models(self):
        supported = {
            "gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen",
            "gemini-3.8-flash", "gemini-3.8-flash-cyber", "gemini-3.6-flash",
            "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
            "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.5-flash-lite", "gemini-3-pro-image", "gemini-2.5-flash-image",
        }
        self.assertEqual(set(target.GEMINI_FLEX_MODEL_KEYS), supported)
        for model in supported:
            self.assertTrue(target.is_gemini_flex_model_key(model), model)
        for model in ("gemini-3.7-flash", "gemini-3.5-flash", "gpt-5.4", "gemini-3.1-flash-image"):
            self.assertFalse(target.is_gemini_flex_model_key(model), model)

    def test_settings_round_trip_keeps_the_default_and_last_flex_flags(self):
        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        response = client.post(
            "/api/settings",
            json={"default_gemini_flex": True},
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        payload = client.get("/api/settings", base_url="https://localhost").get_json()
        self.assertIs(payload["default_gemini_flex"], True)

        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.last_gemini_flex = True
            target.db.session.commit()
        payload = client.get("/api/settings", base_url="https://localhost").get_json()
        self.assertIs(payload["last_gemini_flex"], True)

    def test_prompt_options_record_keeps_the_flex_flag(self):
        record = target._build_prompt_options_record({"gemini_flex": True, "enable_search": True})
        self.assertEqual(json.loads(record)["gemini_flex"], True)
        self.assertEqual(json.loads(record)["enable_search"], True)

    def test_stream_passes_flex_service_tier_and_a_long_timeout(self):
        source = " ".join(read_app_source().split())
        self.assertIn("conf['service_tier'] = types.ServiceTier.FLEX", source)
        self.assertIn("config_kwargs[\"service_tier\"] = types.ServiceTier.FLEX", source)
        self.assertIn("_GEMINI_FLEX_TIMEOUT_MS = _env_int(\"GEMINI_FLEX_TIMEOUT_MS\", 900000)", source)
        self.assertIn("and is_gemini_flex_model_key(rm)", source)
        self.assertIn("not batch_mode", source)
        self.assertIn("and not options.get('gemini_free_tier')", source)

    def test_flex_retries_shedable_errors_before_the_first_chunk(self):
        source = " ".join(read_app_source().split())
        self.assertIn("_GEMINI_FLEX_RETRIES = max(0, _env_int(\"GEMINI_FLEX_RETRIES\", 3))", source)
        self.assertIn("Gemini Flex busy before first chunk, retry", source)
        self.assertIn("Gemini Flex image busy, retry", source)
        self.assertIn('"503" in _exc_text or "429" in _exc_text', source)


if __name__ == "__main__":
    unittest.main()
