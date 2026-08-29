import os
import tempfile
import unittest
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "login-flash-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-login-flash-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class FlaskLoginFlashLeakRegressionTests(unittest.TestCase):
    """Flask-Login's default login_message ("Please log in to access this
    page.") used to be flashed by unauthorized() whenever an unauthenticated
    request hit a @login_required route.  The login page never rendered
    flashed messages, so the English text stayed in the session and leaked
    onto the chat home screen (#flash-msg) after the user logged in -- a
    misleading "you must log in" toast right after account creation."""

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
            user = target.User(username="login-flash-test", is_setup_completed=True)
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

    def test_login_manager_login_message_is_disabled(self):
        self.assertIsNone(target.login_manager.login_message)

    def test_unauthorized_access_no_longer_flashes_login_message(self):
        client = target.app.test_client()
        # Access a @login_required route while logged out.
        response = client.get("/setup", base_url="https://localhost")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        # The session must NOT contain the Flask-Login login flash.
        with client.session_transaction() as sess:
            flashes = sess.get("_flashes") or []
        leaked = [
            f for f in flashes
            if isinstance(f, (list, tuple)) and len(f) >= 2 and f[1] == "Please log in to access this page."
        ]
        self.assertEqual(leaked, [])

    def test_stale_login_flash_is_cleaned_up(self):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_flashes"] = [
                ("message", "Please log in to access this page."),
                ("message", "設定を保存しました"),
                ("message", "ボット検出によるBANが解除されました。"),
            ]
        client.get("/", base_url="https://localhost")
        with client.session_transaction() as sess:
            flashes = sess.get("_flashes") or []
        self.assertNotIn(
            "Please log in to access this page.",
            [f[1] for f in flashes if isinstance(f, (list, tuple)) and len(f) >= 2],
        )
        # Settings-save flashes are stale too (see _LEAKY_SETTINGS_FLASHES) and
        # must never leak onto a page render.
        self.assertNotIn(
            "設定を保存しました",
            [f[1] for f in flashes if isinstance(f, (list, tuple)) and len(f) >= 2],
        )
        # Truly unrelated flashes (e.g. the bot-unban notice) are preserved.
        self.assertIn(
            "ボット検出によるBANが解除されました。",
            [f[1] for f in flashes if isinstance(f, (list, tuple)) and len(f) >= 2],
        )


if __name__ == "__main__":
    unittest.main()
