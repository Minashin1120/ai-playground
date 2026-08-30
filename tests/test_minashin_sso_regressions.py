import os
import unittest
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "minashin-sso-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-minashin-sso-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


def _fake_token_response(**overrides):
    resp = mock.Mock()
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
        "scope": "openid profile email",
        **overrides,
    }
    return resp


def _fake_userinfo_response(**overrides):
    resp = mock.Mock()
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = {
        "sub": "user_minashin_test_1",
        "name": "Minashin 太郎",
        "nickname": "taro.minashin",
        "preferred_username": "taro.minashin",
        "email": "taro@minashin1120.com",
        "email_verified": True,
        "picture": "https://account.minashin1120.com/avatars/user_minashin_test_1.webp",
        "updated_at": 1717378800,
        **overrides,
    }
    return resp


def _fake_failure_response(status=400, error="invalid_grant", error_description="expired"):
    resp = mock.Mock()
    resp.ok = False
    resp.status_code = status
    resp.json.return_value = {
        "error": error,
        "error_description": error_description,
    }
    return resp


class MinashinSSORegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(
            TESTING=True,
            MAINTENANCE_MODE=False,
            TRUSTED_HOSTS=["localhost"],
            WTF_CSRF_ENABLED=False,
        )
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def _start_login(self, client, username="taro.minashin"):
        """GET /login/minashin を実行し、認可URLとセッションを検証する。"""
        with mock.patch.object(target, "rate_limit", return_value=True):
            response = client.get("/login/minashin", base_url="https://localhost")
        self.assertEqual(response.status_code, 302)
        location = response.headers["Location"]
        self.assertTrue(location.startswith("https://account.minashin1120.com/oauth/authorize?"))
        from urllib.parse import parse_qs, urlparse
        params = {k: v[0] for k, v in parse_qs(urlparse(location).query).items()}
        self.assertEqual(params["response_type"], "code")
        self.assertEqual(params["client_id"], "https://localhost")
        self.assertEqual(
            params["redirect_uri"],
            "https://localhost/auth/minashin/callback",
        )
        self.assertEqual(params["code_challenge_method"], "S256")
        self.assertIn("code_challenge", params)
        self.assertIn("state", params)
        self.assertEqual(params["scope"], "openid profile email")
        with client.session_transaction() as session:
            self.assertEqual(session["minashin_oauth_state"], params["state"])
            self.assertTrue(session["minashin_code_verifier"])
        return params

    def test_login_route_starts_pkce_flow(self):
        client = target.app.test_client()
        self._start_login(client)

    def test_callback_creates_new_account_on_first_login(self):
        client = target.app.test_client()
        params = self._start_login(client)
        with mock.patch.object(target.requests, "post", return_value=_fake_token_response()), \
             mock.patch.object(target.requests, "get", return_value=_fake_userinfo_response()):
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=" + params["state"],
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/setup")
        with target.app.app_context():
            user = target.User.query.filter_by(minashin_sub="user_minashin_test_1").one()
            self.assertEqual(user.minashin_email, "taro@minashin1120.com")
            self.assertFalse(user.is_setup_completed)

    def test_callback_logs_in_existing_linked_user(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(
                username="existing-linked",
                minashin_sub="user_minashin_test_1",
                minashin_email="taro@minashin1120.com",
                is_setup_completed=True,
            )
            target.db.session.add(user)
            target.db.session.commit()
        params = self._start_login(client)
        with mock.patch.object(target.requests, "post", return_value=_fake_token_response()), \
             mock.patch.object(target.requests, "get", return_value=_fake_userinfo_response()):
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=" + params["state"],
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        with client.session_transaction() as session:
            self.assertTrue(session.get("_user_id"))

    def test_callback_links_existing_user_by_matching_email(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(
                username="email-match-user",
                minashin_email="taro@minashin1120.com",
                is_setup_completed=True,
            )
            target.db.session.add(user)
            target.db.session.commit()
        params = self._start_login(client)
        with mock.patch.object(target.requests, "post", return_value=_fake_token_response()), \
             mock.patch.object(target.requests, "get", return_value=_fake_userinfo_response()):
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=" + params["state"],
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        with target.app.app_context():
            linked = target.User.query.filter_by(username="email-match-user").one()
            self.assertEqual(linked.minashin_sub, "user_minashin_test_1")

    def test_callback_link_mode_from_settings(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(username="logged-in-user", is_setup_completed=True)
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True

        with mock.patch.object(target, "rate_limit", return_value=True):
            response = client.get("/login/minashin", base_url="https://localhost")
        with client.session_transaction() as session:
            self.assertTrue(session.get("minashin_link_mode"))
        state = self._extract_state(response.headers["Location"])

        with mock.patch.object(target.requests, "post", return_value=_fake_token_response()), \
             mock.patch.object(target.requests, "get", return_value=_fake_userinfo_response()):
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=" + state,
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        with target.app.app_context():
            linked = target.User.query.filter_by(username="logged-in-user").one()
            self.assertEqual(linked.minashin_sub, "user_minashin_test_1")

    @staticmethod
    def _extract_state(location):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(location).query)["state"][0]

    def test_callback_rejects_state_mismatch(self):
        client = target.app.test_client()
        self._start_login(client)
        with mock.patch.object(target.requests, "post") as mock_post, \
             mock.patch.object(target.requests, "get") as mock_get:
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=WRONG_STATE",
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")
        mock_post.assert_not_called()
        mock_get.assert_not_called()

    def test_callback_handles_oauth_error(self):
        client = target.app.test_client()
        self._start_login(client)
        response = client.get(
            "/auth/minashin/callback?error=access_denied&error_description=denied",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

    def test_callback_handles_invalid_grant(self):
        client = target.app.test_client()
        params = self._start_login(client)
        with mock.patch.object(
            target.requests,
            "post",
            return_value=_fake_failure_response(status=400, error="invalid_grant"),
        ):
            response = client.get(
                "/auth/minashin/callback?code=expired-code&state=" + params["state"],
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

    def test_callback_ignores_unverified_email_for_linking(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(
                username="email-unverified-user",
                minashin_email="taro@minashin1120.com",
                is_setup_completed=True,
            )
            target.db.session.add(user)
            target.db.session.commit()
        params = self._start_login(client)
        unverified_info = _fake_userinfo_response(email_verified=False)
        with mock.patch.object(target.requests, "post", return_value=_fake_token_response()), \
             mock.patch.object(target.requests, "get", return_value=unverified_info):
            response = client.get(
                "/auth/minashin/callback?code=test-code&state=" + params["state"],
                base_url="https://localhost",
            )
        # 未確認メールでは既存ユーザーに吸収されず、新しいアカウントが作られる
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/setup")
        with target.app.app_context():
            existing = target.User.query.filter_by(username="email-unverified-user").one()
            self.assertIsNone(existing.minashin_sub)
            created = target.User.query.filter_by(minashin_sub="user_minashin_test_1").one()
            self.assertIsNone(created.minashin_email)

    def test_unlink_minashin_route(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(
                username="to-unlink",
                minashin_sub="user_minashin_test_1",
                minashin_email="taro@minashin1120.com",
                is_setup_completed=True,
                bot_detection_enabled=False,
            )
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True
            session["csrf_token"] = "minashin-csrf-token"
        with mock.patch.object(target, "rate_limit", return_value=True):
            response = client.post(
                "/api/account/unlink_minashin",
                headers={"X-CSRF-Token": "minashin-csrf-token"},
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 200)
        with target.app.app_context():
            unlinked = target.User.query.filter_by(username="to-unlink").one()
            self.assertIsNone(unlinked.minashin_sub)
            self.assertIsNone(unlinked.minashin_email)

    def test_settings_payload_contains_minashin_fields(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(
                username="settings-user",
                minashin_sub="user_minashin_test_1",
                minashin_email="taro@minashin1120.com",
                is_setup_completed=True,
            )
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True
        with mock.patch.object(target, "rate_limit", return_value=True):
            response = client.get("/api/settings", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["minashin_sub"], "user_minashin_test_1")
        self.assertEqual(data["minashin_email"], "taro@minashin1120.com")

    def test_login_page_and_signup_page_have_minashin_buttons(self):
        client = target.app.test_client()
        with mock.patch.object(target, "rate_limit", return_value=True):
            login_response = client.get("/login", base_url="https://localhost")
        login_html = login_response.get_data(as_text=True)
        self.assertIn("Minashin アカウントでログイン", login_html)
        self.assertIn('href="/login/minashin"', login_html)

        signup_response = client.get("/signup", base_url="https://localhost")
        signup_html = signup_response.get_data(as_text=True)
        self.assertIn("Minashin アカウントで登録", signup_html)
        self.assertIn('href="/login/minashin"', signup_html)


if __name__ == "__main__":
    unittest.main()
