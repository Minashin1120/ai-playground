import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from tests.app_source import read_app_source
os.environ.setdefault("FLASK_SECRET_KEY", "settings-flash-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-settings-flash-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")


class SettingsFlashLeakRegressionTests(unittest.TestCase):
    """The settings endpoint is AJAX-only, so its result must be returned in the
    JSON response instead of flash().  flash() messages are only consumed on the
    next full page render, which used to leak a stale "設定を保存しました" toast
    onto the next reload -- even for background auto-saves (rich-paste prompts,
    Gem application, version-update cache pref, ...) when the user never opened
    the settings modal."""

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
            user = target.User(username="settings-flash-test", is_setup_completed=True)
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

    def test_settings_save_returns_message_and_does_not_leak_flash_to_next_page(self):
        client = self.authenticated_client()
        response = client.post(
            "/api/settings",
            json={"theme_color": "#123456"},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload.get("message"), "設定を保存しました")

        # A subsequent full page render must NOT show a stale flash toast.
        page = client.get("/", base_url="https://localhost")
        self.assertEqual(page.status_code, 200, page.get_data(as_text=True))
        self.assertNotIn('id="flash-msg"', page.get_data(as_text=True))

    def test_settings_handler_returns_message_via_json_not_flash(self):
        # The POST branch must return the result message in the JSON payload
        # instead of queueing flash() messages into the session.
        self.assertIn("'message': result_message", APP_SOURCE)
        for leaked in ("設定を保存しました", "暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。"):
            self.assertNotIn('flash("%s")' % leaked, APP_SOURCE)

    def test_stale_settings_save_flash_is_purged_before_page_render(self):
        # A session created while the settings endpoint still used flash() may
        # carry a stale "設定を保存しました" message.  Because the settings
        # modal is a client-side route (history.pushState), opening it does not
        # consume the flash; the next full page render (e.g. navigating to "/"
        # from the URL bar) used to render it as a misleading toast even though
        # nothing was saved.  Every request must purge these stale flashes so no
        # page render can show them.
        client = self.authenticated_client()
        with client.session_transaction() as sess:
            sess["_flashes"] = [
                ("message", "設定を保存しました"),
                ("message", "暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。"),
                ("message", "2FAを無効化しました。"),
                ("message", "ボット検出によるBANが解除されました。"),
            ]
        page = client.get("/", base_url="https://localhost")
        self.assertEqual(page.status_code, 200, page.get_data(as_text=True))
        html = page.get_data(as_text=True)
        # The stale settings-save flashes must not be rendered as a toast.
        for leaked in ("設定を保存しました", "暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。", "2FAを無効化しました。"):
            self.assertNotIn(leaked, html)
        # Unrelated flashes (e.g. the bot-unban notice) survive the purge.
        self.assertIn("ボット検出によるBANが解除されました。", html)

    def test_client_uses_server_message_for_save_toast(self):
        # The settings-modal save handler reads the server-provided message
        # (e.g. the E2EE migration notice) instead of always showing the generic
        # "設定を保存しました" text.
        handler_start = CHAT_JS.index("get('save-settings-btn').onclick")
        handler = CHAT_JS[handler_start:]
        self.assertIn("let saveMsg = \"設定を保存しました\"", handler)
        self.assertIn("if (d && d.message) saveMsg = d.message", handler)
        self.assertIn("showToast(saveMsg, \"success\")", handler)


if __name__ == "__main__":
    unittest.main()
