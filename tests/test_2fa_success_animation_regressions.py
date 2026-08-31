import os
import unittest
from pathlib import Path

os.environ.setdefault("FLASK_SECRET_KEY", "2fa-anim-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-2fa-anim-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

APP_ROOT = Path(__file__).resolve().parent.parent


class TwoFactorSuccessAnimationRegressionTests(unittest.TestCase):
    """2FA完了後の成功アニメーション（認証成功オーバーレイ）を保つ回帰テスト。

    ログイン画面内の2FA（TOTP / WebAuthn）は triggerSuccess() 経由でオーバーレイを
    表示する。スタンドアロンの verify_2fa.html（Google / Minashin SSO 経由）は
    従来 window.location.href で即遷移してアニメーションが無かったため、同じ
    triggerSuccess() でオーバーレイを表示するよう修正した。ここでは各2FA成功経路が
    オーバーレイを経由し、チェックマーク描画が視認できることを検証する。
    """

    def setUp(self):
        self.login = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")
        self.verify = (APP_ROOT / "templates" / "verify_2fa.html").read_text(encoding="utf-8")

    def test_verify_2fa_page_has_success_overlay(self):
        # スタンドアロン2FAページに成功オーバーレイとチェックマークがある
        self.assertIn('id="success-screen" class="success-overlay"', self.verify)
        self.assertIn("認証成功", self.verify)
        self.assertIn(".checkmark-svg", self.verify)

    def test_verify_2fa_totp_success_uses_trigger_success(self):
        # TOTP成功時に即遷移せず、オーバーレイを表示してから遷移する
        self.assertIn("triggerSuccess(data.redirect)", self.verify)
        self.assertNotIn(
            "window.location.href = data.redirect || \"{{ url_for('index') }}\"", self.verify
        )

    def test_verify_2fa_webauthn_success_uses_trigger_success(self):
        # WebAuthn成功時もオーバーレイを表示してから遷移する
        self.assertIn('triggerSuccess("{{ url_for(\'index\') }}")', self.verify)
        self.assertNotIn(
            'window.location.href = "{{ url_for(\'index\') }}"', self.verify
        )

    def test_verify_2fa_checkmark_animation_is_visible(self):
        # チェックマーク描画は0.8秒（視認できる長さ）
        self.assertIn(
            ".active .checkmark-svg { animation: checkmark 0.8s", self.verify
        )

    def test_verify_2fa_redirect_waits_for_draw(self):
        # 描画完了後もオーバーレイを表示してから遷移する（1200ms）
        trigger = self.verify[self.verify.index("const triggerSuccess") :]
        trigger = trigger[: trigger.index("function switchTab")]
        self.assertIn("}, 1200);", trigger)
        self.assertNotIn("}, 600);", trigger)

    def test_login_inline_2fa_uses_trigger_success(self):
        # ログイン画面内2FA（TOTP / WebAuthn）も triggerSuccess() 経由
        self.assertIn("if (res.ok && result.status === 'ok') {\n                    triggerSuccess(result.redirect);", self.login)
        self.assertIn("if(result.status === 'ok') {\n                    triggerSuccess(\"/\");", self.login)


if __name__ == "__main__":
    unittest.main()
