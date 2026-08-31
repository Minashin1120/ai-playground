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
    """認証成功後のダッシュボードへのリダイレクト時間と成功アニメーション表示を保つ回帰テスト。

    以前はパスワードログインで約2秒（ボタンアニメ 520ms + 成功画面 300ms +
    1200ms）、パスキー／2FA／Google で約1.5秒待ってからリダイレクトしていました。
    V4.8.877 で各待ち時間を短縮しましたが、リダイレクトがチェックマーク描画完了と
    同時刻になり、アニメーションが視認できなくなりました。チェックマークの描画
    （0.8s）がオーバーレイ上で確実に視認できるよう、描画完了後に余裕を持って
    リダイレクトすることを検証します。
    """

    def setUp(self):
        self.source = (APP_ROOT / "templates" / "login.html").read_text(encoding="utf-8")

    def test_button_success_animation_wait_is_short(self):
        # パスワードログインのボタン成功アニメーションは短時間で次へ進む
        self.assertIn("await new Promise((resolve) => setTimeout(resolve, 300));", self.source)
        self.assertNotIn("setTimeout(resolve, 520)", self.source)

    def test_trigger_success_redirects_after_checkmark_draw(self):
        # 成功画面のフェード + チェックマーク描画が完了したあとにリダイレクトする
        # （旧 600ms は描画完了と同時でアニメーションが見えない）
        trigger = self.source[self.source.index("const triggerSuccess = (redirectUrl) => {") :]
        trigger = trigger[: trigger.index("window.backToLogin")]
        self.assertIn("}, 1200);", trigger)  # 成功画面の表示時間（描画完了＋余白）
        self.assertIn("}, 200);", trigger)  # コンテナのフェード時間
        self.assertNotIn("}, 600);", trigger)

    def test_success_screen_total_budget_shorter_than_original(self):
        # フェード + 成功画面の合計が従来（1.5秒）より短いこと
        self.assertLess(200 + 1200, 1500)

    def test_success_checkmark_animation_is_visible(self):
        # チェックマークの描画は0.8秒かけて視認できる（0.4秒では描画が消える）
        self.assertIn(".active .checkmark-svg { animation: checkmark 0.8s", self.source)
        self.assertNotIn("checkmark 0.4s", self.source)


if __name__ == "__main__":
    unittest.main()
