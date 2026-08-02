import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "turnstile-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-turnstile-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6398/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class TurnstileBotDetectionRegressionTests(unittest.TestCase):
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
            user = target.User(username="turnstile-test", is_setup_completed=True)
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id

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

    def post_telemetry(self, client, payload):
        return client.post(
            "/api/bot-telemetry",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def test_turnstile_failed_report_is_scored_not_rejected(self):
        # Turnstile 不合格（トークンなし）でも turnstile_failed フラグがあれば
        # 403 で弾かず、スコア判定へ進める（検出の抜けを塞ぐ）。
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            client = self.authenticated_client()
            res = self.post_telemetry(client, {
                "turnstile_failed": True,
                "window_ms": 4000,
                "clicks": 5,
                "keys": 3,
                "moves": 2,
                "fast_clicks": 0,
                "fast_keys": 0,
                "click_burst": 2,
                "key_burst": 2,
                "event_rate": 2.5,
                "avg_click_ms": 9999,
                "click_cv": 1.0,
                "pointer_speed_max": 0,
            })
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        # ペナルティ（+2）が反映されたスコアが返る
        self.assertEqual(body.get("score"), 2)

    def test_report_without_token_and_flag_is_rejected(self):
        # フラグなし・トークンなしの報告は従来どおり 403 で拒否
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            client = self.authenticated_client()
            res = self.post_telemetry(client, {
                "window_ms": 4000,
                "clicks": 5,
                "keys": 3,
                "moves": 2,
            })
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("error"), "turnstile_failed")

    def test_turnstile_failed_adds_penalty_on_top_of_behavior_score(self):
        # 挙動スコア 0 でも turnstile_failed なら +2 が加算される
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            client = self.authenticated_client()
            res = self.post_telemetry(client, {
                "turnstile_token": "valid-token",
                "turnstile_failed": True,
                "window_ms": 4000,
                "clicks": 1,
                "keys": 1,
                "moves": 1,
                "fast_clicks": 0,
                "fast_keys": 0,
                "click_burst": 1,
                "key_burst": 1,
                "event_rate": 0.75,
                "avg_click_ms": 9999,
                "click_cv": 1.0,
                "pointer_speed_max": 0,
            })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("score"), 2)

    def test_valid_token_without_failed_flag_is_not_penalized(self):
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            client = self.authenticated_client()
            res = self.post_telemetry(client, {
                "turnstile_token": "valid-token",
                "window_ms": 4000,
                "clicks": 1,
                "keys": 1,
                "moves": 1,
                "fast_clicks": 0,
                "fast_keys": 0,
                "click_burst": 1,
                "key_burst": 1,
                "event_rate": 0.75,
                "avg_click_ms": 9999,
                "click_cv": 1.0,
                "pointer_speed_max": 0,
            })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("score"), 0)

    def test_high_score_report_bans_user_even_with_turnstile_failure(self):
        # スコア合計（挙動 + turnstile_failed ペナルティ）が 8 以上で BAN
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            client = self.authenticated_client()
            res = self.post_telemetry(client, {
                "turnstile_failed": True,
                "window_ms": 2000,
                "clicks": 30,
                "keys": 30,
                "moves": 10,
                "fast_clicks": 8,
                "fast_keys": 20,
                "click_burst": 15,
                "key_burst": 20,
                "event_rate": 35.0,
                "avg_click_ms": 80,
                "click_cv": 0.02,
                "pointer_speed_max": 9000,
            })
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)

    def test_js_uses_interaction_only_appearance(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        render_block = source[source.index("window.initTurnstileWidget = () =>") :]
        render_block = render_block[: render_block.index("async function getTurnstileToken()")]
        self.assertIn("appearance: 'interaction-only'", render_block)

    def test_js_blocks_send_when_token_unavailable(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("function isBotDetectionActive()", source)
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = get('prompt-input').value")]
        self.assertIn("const gateToken = await getTurnstileToken();", gate)
        self.assertIn("安全性の確認を完了できませんでした", gate)
        self.assertIn("botTelemetry.send(true);", gate)
        self.assertIn("return;", gate)

    def test_js_sends_turnstile_failed_report_instead_of_skipping(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("payload.turnstile_failed = true;", source)
        self.assertNotIn(
            "if (botConfig && botConfig.turnstileSiteKey && !payload.turnstile_token) return;",
            source,
        )


if __name__ == "__main__":
    unittest.main()
