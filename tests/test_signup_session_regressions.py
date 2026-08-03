import os
import unittest
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "signup-session-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-signup-session-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class SignupSessionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def test_password_signup_establishes_session_before_setup_redirect(self):
        client = target.app.test_client()
        with client.session_transaction() as session:
            session["csrf_token"] = "signup-csrf-token"

        with mock.patch.object(target, "verify_turnstile", return_value=True), \
             mock.patch.object(target, "rate_limit", return_value=True):
            response = client.post(
                "/signup",
                data={
                    "csrf_token": "signup-csrf-token",
                    "cf-turnstile-response": "test-token",
                    "username": "new-signup-user",
                    "password": "strong-password",
                },
                base_url="https://localhost",
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/setup")
        with client.session_transaction() as session:
            self.assertTrue(session.get("_user_id"))
            self.assertTrue(session.get("session_id"))
            session_id = session["session_id"]

        with target.app.app_context():
            user = target.User.query.filter_by(username="new-signup-user").one()
            self.assertFalse(user.is_setup_completed)
            active_session = target.UserSession.query.filter_by(
                user_id=user.id,
                session_id=session_id,
                is_revoked=False,
            ).one()
            self.assertIsNotNone(active_session)

        setup_response = client.get("/setup", base_url="https://localhost")
        self.assertEqual(setup_response.status_code, 200)
        self.assertIn("初期セットアップ", setup_response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
