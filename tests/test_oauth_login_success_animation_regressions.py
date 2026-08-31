import os
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("FLASK_SECRET_KEY", "oauth-anim-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-oauth-anim-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target

APP_ROOT = Path(__file__).resolve().parent.parent


class OAuthLoginSuccessAnimationRegressionTests(unittest.TestCase):
    """アカウント連携ログイン完了時に認証成功アニメーションを表示する回帰テスト。"""

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
        self.login_html = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def test_login_template_has_auth_success_trigger(self):
        # auth_success フラグがある場合に triggerSuccess を自動実行するスクリプトがある
        self.assertIn("{% if auth_success %}", self.login_html)
        self.assertIn("triggerSuccess({{ auth_success_redirect | tojson }});", self.login_html)
        self.assertIn('id="success-screen" class="success-overlay"', self.login_html)
        self.assertIn(".checkmark-svg", self.login_html)

    def test_login_template_hides_auth_container_on_auth_success(self):
        # auth_success フラグがあるときは初期状態でコンテナを非表示にしてアニメーションを滑らかにする
        self.assertIn("{% if auth_success %} opacity-0 pointer-events-none{% endif %}", self.login_html)

    def test_login_route_renders_template_with_auth_success_for_authenticated_user(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(username="oauth-anim-user", is_setup_completed=True)
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id

        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True

        with mock.patch.object(target, "rate_limit", return_value=True):
            # auth_success=1 付きでアクセスした場合、200 OK でログイン画面がレンダリングされる
            response = client.get("/login?auth_success=1&next=/", base_url="https://localhost")
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn('triggerSuccess("/")', html)
            self.assertIn("認証成功", html)

            # auth_success なしで通常アクセスした場合は index に 302 リダイレクトされる
            response_normal = client.get("/login", base_url="https://localhost")
            self.assertEqual(response_normal.status_code, 302)
            self.assertEqual(response_normal.headers["Location"], "/")

    def test_login_route_sanitizes_open_redirect_next_url(self):
        client = target.app.test_client()
        with target.app.app_context():
            user = target.User(username="oauth-anim-user", is_setup_completed=True)
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id

        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True

        with mock.patch.object(target, "rate_limit", return_value=True):
            # 外部URLへのオープンリダイレクトが / にサニタイズされること
            response = client.get("/login?auth_success=1&next=https://evil.com", base_url="https://localhost")
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn('triggerSuccess("/")', html)


if __name__ == "__main__":
    unittest.main()
