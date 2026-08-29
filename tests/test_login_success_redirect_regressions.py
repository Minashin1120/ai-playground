import os
import unittest
from pathlib import Path


os.environ.setdefault("FLASK_SECRET_KEY", "login-redirect-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-login-redirect-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

APP_ROOT = Path(__file__).resolve().parent.parent


class LoginSuccessRedirectRegressionTests(unittest.TestCase):
    """認証成功後のダッシュボードへのリダイレクトを速く保つための回帰テスト。

    以前はパスワードログインで約2秒（ボタンアニメ 520ms + 成功画面 300ms +
    1200ms）、パスキー／2FA／Google で約1.5秒待ってからリダイレクトしていました。
    成功画面の表示は維持しつつ、各待ち時間を短縮してリダイレクトまでの
    合計を1秒前後にしています。この待ち時間が元へ戻らないことを検証します。
    """

    def setUp(self):
        self.source = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")

    def test_button_success_animation_wait_is_short(self):
        # パスワードログインのボタン成功アニメーションは短時間で次へ進む
        self.assertIn("await new Promise((resolve) => setTimeout(resolve, 300));", self.source)
        self.assertNotIn("setTimeout(resolve, 520)", self.source)

    def test_trigger_success_redirects_within_short_budget(self):
        # 認証成功画面の表示後、すぐにリダイレクトする（旧 1200ms は禁止）
        trigger = self.source[self.source.index("const triggerSuccess = (redirectUrl) => {") :]
        trigger = trigger[: trigger.index("window.backToLogin")]
        self.assertIn("}, 600);", trigger)  # 成功画面の表示時間
        self.assertIn("}, 200);", trigger)  # コンテナのフェード時間
        self.assertNotIn("}, 1200);", trigger)

    def test_success_screen_total_budget_under_one_second(self):
        # フェード + 成功画面の合計が1秒未満であること（以前は1.5秒）
        self.assertLess(200 + 600, 1000)

    def test_success_checkmark_animation_matches_short_window(self):
        # チェックマークの描画が短縮した表示時間内に完了する（旧 0.8s は禁止）
        self.assertIn(".active .checkmark-svg { animation: checkmark 0.4s", self.source)
        self.assertNotIn("checkmark 0.8s", self.source)


if __name__ == "__main__":
    unittest.main()
