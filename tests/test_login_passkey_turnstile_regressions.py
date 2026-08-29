import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "login-passkey-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-login-passkey-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target

APP_ROOT = Path(__file__).resolve().parent.parent
CSRF_TOKEN = "csrf-test-token"


class LoginPasskeyTurnstileRegressionTests(unittest.TestCase):
    """パスキーログインとパスワードログインは同じ Turnstile ウィジェットを共有する。

    Turnstile のトークンは single-use のため、パスキー試行がサーバーで
    verify_turnstile を通ると（アカウント側で失敗しても）トークンは消費される。
    クライアントがウィジェットをリセットしないまま次のパスワードログインへ
    進むと、消費済みトークンを再送してしまい、初回が必ず失敗する。
    """

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
            user = target.User(username="passkey-test", is_setup_completed=True)
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

    def client_with_csrf(self):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = CSRF_TOKEN
        return client

    def post_passkey_options(self, client, username, turnstile_token):
        return client.post(
            "/login/passkey/options",
            json={"username": username, "turnstile": turnstile_token},
            headers={"X-CSRF-Token": CSRF_TOKEN},
            base_url="https://localhost",
        )

    def post_password_login(self, client, username, password, turnstile_token):
        return client.post(
            "/login",
            json={
                "username": username,
                "password": password,
                "cf-turnstile-response": turnstile_token,
            },
            headers={
                "X-CSRF-Token": CSRF_TOKEN,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            },
            base_url="https://localhost",
        )

    def test_passkey_options_rejects_non_passkey_only_account(self):
        # パスキー専用ログインが無効なアカウントへ passkey を試みると
        # 「Invalid credentials」で拒否される（期待どおりのエラー）。
        client = self.client_with_csrf()
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            res = self.post_passkey_options(client, "passkey-test", "valid-token")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json().get("error"), "Invalid credentials")

    def test_passkey_options_consumes_turnstile_before_account_check(self):
        # アカウントが passkey 専用でなくても、サーバーはアカウント判定より先に
        # verify_turnstile を呼ぶため single-use トークンは必ず消費される。
        client = self.client_with_csrf()
        with mock.patch.object(target, "verify_turnstile", return_value=True) as vt:
            res = self.post_passkey_options(client, "passkey-test", "valid-token")
        self.assertEqual(res.status_code, 400)
        vt.assert_called_once()

    def test_reused_turnstile_token_makes_first_password_login_fail(self):
        # 報告された症状を再現: パスキー試行で消費されたトークンをそのまま
        # パスワードログインが再利用すると、初回は必ず「認証エラー」になる。
        # 新しいトークンを使えば成功する（= クライアント側でウィジェットを
        # リセットすれば回避できる）。
        client = self.client_with_csrf()

        consumed = set()

        def fake_verify(token):
            # Turnstile の single-use 挙動を模倣: 同じトークンは2回目以降失敗する
            if token in consumed:
                return False
            consumed.add(token)
            return True

        with mock.patch.object(target, "verify_turnstile", side_effect=fake_verify):
            # 1) パスキー試行（パスキー専用でないため Invalid credentials で拒否、
            #    ただしトークンは消費される）
            res = self.post_passkey_options(client, "passkey-test", "used-token")
            self.assertEqual(res.status_code, 400)
            self.assertEqual(res.get_json().get("error"), "Invalid credentials")

            # 2) 同じ消費済みトークンでのパスワードログイン → 初回は認証エラー
            res = self.post_password_login(client, "passkey-test", "test-password", "used-token")
            self.assertEqual(res.status_code, 401)

            # 3) 新しいトークンでのパスワードログイン → 成功
            res = self.post_password_login(client, "passkey-test", "test-password", "fresh-token")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json().get("status"), "ok")

    def test_login_page_resets_turnstile_after_passkey_attempt(self):
        # クライアントはパスキー試行の各失敗経路（options 失敗 / verify 失敗 /
        # 例外）で Turnstile ウィジェットをリセットし、次のログイン試行が
        # 新しいトークンを使えるようにする。
        source = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")
        self.assertIn("const resetTurnstile = () => {", source)
        self.assertIn("window.turnstile.reset()", source)
        handler = source[source.index("document.getElementById('passkey-login-btn').onclick") :]
        handler = handler[: handler.index("const passwordInput")]
        self.assertGreaterEqual(handler.count("resetTurnstile();"), 3)

    def test_password_login_failure_resets_turnstile(self):
        # パスワードログイン失敗時は従来どおりウィジェットをリセットする
        # （2回目以降が成功するのはこのため）。
        source = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")
        self.assertIn("if (window.turnstile) window.turnstile.reset();", source)


if __name__ == "__main__":
    unittest.main()
