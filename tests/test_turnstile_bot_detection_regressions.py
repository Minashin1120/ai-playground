import os
import tempfile
import time
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


class _FakeRedis:
    """Minimal dict-backed Redis stand-in for tests (keys with TTL)."""

    def __init__(self):
        self._d = {}

    def exists(self, key):
        entry = self._d.get(key)
        if entry is None:
            return 0
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._d.pop(key, None)
            return 0
        return 1

    def set(self, key, value, ex=None):
        self._d[key] = (value, time.time() + ex if ex else None)
        return True

    def get(self, key):
        entry = self._d.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._d.pop(key, None)
            return None
        return value

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self._d:
                del self._d[key]
                removed += 1
        return removed

    def expire(self, key, ttl):
        entry = self._d.get(key)
        if entry is None:
            return True
        self._d[key] = (entry[0], time.time() + ttl)
        return True

    def ttl(self, key):
        entry = self._d.get(key)
        if entry is None:
            return -2
        value, expires_at = entry
        if expires_at is None:
            return -1
        return int(expires_at - time.time())

    def incr(self, key):
        current = self.get(key)
        new_value = (int(current) + 1) if current is not None else 1
        self._d[key] = (new_value, None)
        return new_value

    def incrbyfloat(self, key, amount):
        current = self.get(key)
        new_value = (float(current) + float(amount)) if current is not None else float(amount)
        self._d[key] = (new_value, None)
        return new_value

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            raise NotImplementedError(name)
        return _missing


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

    def post_chat_stream(self, client, payload):
        return client.post(
            "/chat_stream",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def post_turnstile_verify(self, client, payload):
        return client.post(
            "/api/bot/turnstile-verify",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def post_bot_lock(self, client, payload=None):
        return client.post(
            "/api/bot/lock",
            data=target.json.dumps(payload or {}),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def turnstile_env(self):
        # Turnstile keys are only enabled for the duration of the API-level gate tests,
        # so other test modules are not affected by the gate.
        return mock.patch.dict(
            os.environ,
            {"TURNSTILE_SITE_KEY": "test-site-key", "TURNSTILE_SECRET_KEY": "test-secret-key"},
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

    def test_chat_stream_blocked_without_turnstile_verification(self):
        # ボット検出が有効なユーザーは、API から未検証で送信しようとすると
        # 403 turnstile_required でブロックされる（API レベルでのゲート）。
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_chat_stream(client, {"message": "hello"})
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("error"), "turnstile_required")

    def test_chat_stream_blocked_with_invalid_token(self):
        # 無効なトークンだけを添えて送信しても 403 turnstile_required
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_chat_stream(client, {"message": "hello", "turnstile_token": "invalid"})
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("error"), "turnstile_required")

    def test_chat_stream_allowed_with_valid_token_inline(self):
        # 本文に有効なトークンがあればゲートを通過（後続の入力検証 400 へ進む）
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_chat_stream(client, {"message": "hello", "turnstile_token": "valid"})
        self.assertNotEqual(res.status_code, 403)
        self.assertNotEqual(res.get_json().get("error"), "turnstile_required")

    def test_turnstile_verify_sets_marker_then_chat_stream_passes(self):
        # /api/bot/turnstile-verify でマーカーが立つと、トークンなしの送信も通過する
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            fake = _FakeRedis()
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    verify_res = self.post_turnstile_verify(client, {"turnstile_token": "valid"})
                    self.assertEqual(verify_res.status_code, 200)
                    res = self.post_chat_stream(client, {"message": "hello"})
        self.assertNotEqual(res.status_code, 403)
        self.assertNotEqual(res.get_json().get("error"), "turnstile_required")

    def test_turnstile_verify_rejects_invalid_token(self):
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_turnstile_verify(client, {"turnstile_token": "invalid"})
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("error"), "turnstile_failed")

    def test_gate_skipped_for_admin_user(self):
        # 管理者はゲート対象外（マーカー・トークンなしでも通過）
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            target.db.session.commit()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_chat_stream(client, {"message": "hello"})
        self.assertNotEqual(res.status_code, 403)
        self.assertNotEqual(res.get_json().get("error"), "turnstile_required")

    def test_chat_stream_resume_and_fast_save_are_gated(self):
        # 復帰・ブラウザ高速モード保存も未検証では 403 turnstile_required
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    resume_res = client.post(
                        "/chat_stream_resume",
                        data=target.json.dumps({"job_id": "job_1234567890_1_abc", "thread_id": "thread-1"}),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(resume_res.status_code, 403)
                    self.assertEqual(resume_res.get_json().get("error"), "turnstile_required")
                    save_res = client.post(
                        "/api/browser_fast_mode/save",
                        data=target.json.dumps({"message": "hi", "assistant_content": "hello"}),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(save_res.status_code, 403)
                    self.assertEqual(save_res.get_json().get("error"), "turnstile_required")

    def test_js_uses_interaction_only_appearance(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        render_block = source[source.index("window.initTurnstileWidget = () =>") :]
        render_block = render_block[: render_block.index("async function getTurnstileToken(")]
        self.assertIn("appearance: 'interaction-only'", render_block)

    def test_js_blocks_send_when_token_unavailable(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("function isBotDetectionActive()", source)
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = get('prompt-input').value")]
        self.assertIn("botTurnstileToken = await getTurnstileToken();", gate)
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

    def test_js_verifies_turnstile_on_server(self):
        # クライアントはトークン取得成功時に /api/bot/turnstile-verify で検証マーカーを立てる
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("async function verifyTurnstileOnServer(token, force = false, challenged = null)", source)
        self.assertIn("'/api/bot/turnstile-verify'", source)

    def test_js_send_attaches_turnstile_token_to_api_payload(self):
        # 送信ゲートで取得したトークンを /chat_stream 等のペイロードへ含める（APIレベルのゲートを通過するため）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("if (botTurnstileToken) p.turnstile_token = botTurnstileToken;", source)
        self.assertIn("turnstile_token: botTurnstileTokenForRequest()", source)
        self.assertIn("async function verifyTurnstileOnServer(token, force = false, challenged = null)", source)

    def test_js_handles_turnstile_required_from_api(self):
        # サーバーが 403 turnstile_required を返した場合は再検証して再送を案内する
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("e.serverCode === 'turnstile_required'", source)
        self.assertIn("もう一度送信してください", source)

    def test_server_defines_api_level_turnstile_gate(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("def _bot_turnstile_gate(token=None):", source)
        self.assertIn("def bot_turnstile_verify():", source)
        self.assertIn("'turnstile_required'", source)

    def test_turnstile_failures_below_limit_do_not_ban(self):
        # ダイアログ表示中（challenged）の Turnstile 失敗がリミット（5回）未満なら BAN されない
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(4):
                        res = self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                        self.assertEqual(res.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_turnstile_failures_reach_limit_ban(self):
        # ダイアログ表示中（challenged）の Turnstile チェックが一定回数（5回）ダメだったら BAN
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(5):
                        res = self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                    self.assertEqual(res.status_code, 403)
                    self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)
            self.assertIn("Turnstile", user.bot_ban_reason)

    def test_unchallenged_turnstile_failed_does_not_ban(self):
        # ダイアログ未表示（challenged なし）の turnstile_failed は BAN に積み上がらない
        # （ダイアログを出さずにいきなり BAN する誤判定の防止）
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(12):
                        res = self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                        self.assertEqual(res.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_verify_fail_without_challenged_does_not_ban(self):
        # ダイアログ未表示の verify 失敗は失敗カウントに加算されず BAN に至らない
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(12):
                        res = self.post_turnstile_verify(client, {"turnstile_token": "invalid"})
                        self.assertEqual(res.status_code, 403)
                        self.assertEqual(res.get_json().get("error"), "turnstile_failed")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_verify_fail_with_challenged_bans_after_limit(self):
        # ダイアログ表示中（challenged）の verify 失敗はカウントされ、5回で BAN
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(5):
                        res = self.post_turnstile_verify(client, {
                            "turnstile_token": "invalid",
                            "challenged": True,
                        })
                    self.assertEqual(res.status_code, 403)
                    self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)

    def test_turnstile_pass_resets_failure_count(self):
        # 一度成功すれば失敗カウントはリセットされ、BAN には至らない
        fake = _FakeRedis()
        with mock.patch.object(
            target, "verify_turnstile", side_effect=[False, False, False, False, True]
        ):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(4):
                        self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                    res = self.post_turnstile_verify(client, {"turnstile_token": "t"})
                    self.assertEqual(res.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_turnstile_pass_fail_cycling_bans(self):
        # ダイアログ表示中の成功→失敗が繰り返される（パス/フェイルのサイクリング）なら BAN
        fake = _FakeRedis()
        results = [False, True, False, True, False, True]
        with mock.patch.object(target, "verify_turnstile", side_effect=results):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = None
                    for r in results:
                        if r:
                            res = self.post_turnstile_verify(client, {"turnstile_token": "t"})
                        else:
                            res = self.post_telemetry(client, {
                                "turnstile_failed": True,
                                "challenged": True,
                                "window_ms": 4000,
                                "clicks": 2,
                                "keys": 2,
                                "moves": 2,
                            })
                    self.assertEqual(res.status_code, 403)
                    self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)
            self.assertIn("cycled", user.bot_ban_reason)

    def test_high_behavior_score_bans_without_turnstile_failure(self):
        # 挙動スコアだけで BAN しきい値に達した場合は Turnstile 失敗がなくても BAN
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_telemetry(client, {
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

    def test_js_counts_clicks_via_pointerdown_only(self):
        # クリック検知は pointerdown 主体とし、二重計上のもとになった touchstart
        # の重複リスナーを削除（click は PointerEvent 非対応環境のフォールバックのみ）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        telemetry = source[source.index("const botTelemetry = (() => {") :]
        telemetry = telemetry[: telemetry.index("return { start, refreshEnabled, send, looksSuspicious };")]
        self.assertIn("document.addEventListener('pointerdown', recordClick, true)", telemetry)
        self.assertNotIn("document.addEventListener('touchstart', recordClick, true)", telemetry)
        self.assertNotIn("document.addEventListener('mousedown', recordClick, true)", telemetry)
        self.assertNotIn("document.addEventListener('touchend', recordClick, true)", telemetry)

    def test_js_ignores_new_chat_control_clicks(self):
        # 新規チャットボタンのクリックは bot 判定の対象外にする
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("data-bot-ignore-click", source)
        self.assertIn("#new-chat-btn", source)
        html = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="new-chat-btn" data-bot-ignore-click="true"', html)
        self.assertIn('id="mobile-new-chat-btn" data-bot-ignore-click="true"', html)

    def test_js_defines_blocking_verification_gate(self):
        # 未検証ユーザーは画面をオーバーレイでブロックし、検証が終わるまで操作不可にする
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("function showBotDetectionOverlay", source)
        self.assertIn("function hideBotDetectionOverlay", source)
        self.assertIn("const runBotDetectionGate = () =>", source)
        self.assertIn("botDetectionVerified = true", source)
        # API からの turnstile_required を検知して再検証＋再送する
        self.assertIn("errBody.error === 'turnstile_required'", source)

    def test_unverified_post_blocked_until_turnstile_verified(self):
        # Turnstile 検証が終わるまで、そのユーザーのサーバー通信（POST）をブロックする
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = client.post(
                        "/api/threads",
                        data=target.json.dumps({"is_temporary": False}),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(res.status_code, 403)
                    self.assertEqual(res.get_json().get("error"), "turnstile_required")
                    # 検証が成功するとマーカーが立ち、POST が通るようになる
                    with mock.patch.object(target, "verify_turnstile", return_value=True):
                        vres = self.post_turnstile_verify(client, {"turnstile_token": "valid"})
                        self.assertEqual(vres.status_code, 200)
                    res2 = client.post(
                        "/api/threads",
                        data=target.json.dumps({"is_temporary": False}),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(res2.status_code, 200)

    def test_verify_endpoint_stays_reachable_while_unverified(self):
        # 検証エンドポイント自体は未検証でも到達できる（ホワイトリスト）
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_turnstile_verify(client, {"turnstile_token": "valid"})
                    self.assertEqual(res.status_code, 200)

    def test_ban_records_evidence_and_log(self):
        # BAN 時に不審な履歴のスナップショットとログが残る
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(5):
                        self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)
            self.assertTrue(user.bot_evidence)
            import json as _json
            snapshot = _json.loads(user.bot_evidence)
            self.assertIn("reason", snapshot)
            self.assertIn("recent_events", snapshot)
            logs = target.BotEvidenceLog.query.filter_by(user_id=self.user_id).all()
            self.assertGreaterEqual(len(logs), 1)
            self.assertIn("ban", [entry.event_type for entry in logs])

    def test_appeal_submission_stores_evidence(self):
        # 異議申し立て時に不審な履歴が添付される
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(5):
                        self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                    res = client.post(
                        "/ban/appeal",
                        data={"message": "これは誤判定です。確認してください。", "csrf_token": "csrf-test-token"},
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(res.status_code, 302)
        with target.app.app_context():
            appeal = target.BanAppeal.query.filter_by(user_id=self.user_id).first()
            self.assertIsNotNone(appeal)
            self.assertTrue(appeal.evidence)

    def test_js_skips_turnstile_failed_when_verified(self):
        # 検証済みユーザーは turnstile_failed を送信しない（誤BAN防止）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("payload.turnstile_failed = true;", source)
        self.assertIn("!botDetectionVerified", source)
        # オーバーレイは Turnstile ウィジェット読込前でも表示される
        self.assertIn("showBotDetectionOverlay();", source)
        self.assertIn("turnstileWidgetId === null", source)

    def test_js_gate_verifies_silently_before_showing_overlay(self):
        # 通常ユーザーには毎回ダイアログを出さない。ゲートは最初にサイレント検証を
        # 試み、怪しい（失敗が続く／挙動が怪しい）時だけオーバーレイを表示する
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        gate = source[source.index("const runBotDetectionGate = () => {") :]
        gate = gate[: gate.index("let turnstileServerVerifiedAt = 0;")]
        # サイレントフェーズ（オーバーレイ未表示）が存在する
        self.assertIn("!botDetectionOverlayShown", gate)
        # オーバーレイ表示前に getTurnstileToken による検証を試みる
        silent_idx = gate.index("if (!botDetectionOverlayShown)")
        overlay_idx = gate.index("showBotDetectionOverlay();")
        self.assertLess(silent_idx, overlay_idx, "Silent verification must come before the overlay")
        self.assertIn("silentAttempts", gate)

    def test_js_turnstile_failed_only_when_dialog_shown(self):
        # turnstile_failed はダイアログ表示中（botDetectionOverlayShown）だけ送る。
        # ダイアログを出さずに失敗が積んで BAN される誤判定を防ぐ
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        telemetry = source[source.index("const botTelemetry = (() => {") :]
        telemetry = telemetry[: telemetry.index("return { start, refreshEnabled, send, looksSuspicious };")]
        self.assertIn("payload.turnstile_failed = true;", telemetry)
        self.assertIn("botDetectionOverlayShown", telemetry)
        self.assertIn("payload.challenged = true;", telemetry)

    def test_js_dialog_renders_visible_turnstile_widget(self):
        # ダイアログ内に可視の Turnstile ウィジェットを描画する（ボックスが出ない問題の修正）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("function renderBotDetectionDialogWidget()", source)
        self.assertIn("bot-detection-widget-box", source)
        self.assertIn("window.turnstile.render", source)
        self.assertIn("botDetectionDialogWidgetId", source)
        # ダイアログ専用ウィジェットの callback は challenged=true で検証する
        dialog = source[source.index("function renderBotDetectionDialogWidget()") :]
        dialog = dialog[: dialog.index("function showBotDetectionOverlay")]
        self.assertIn("verifyTurnstileOnServer(token, true, true)", dialog)

    def test_js_server_verify_sends_challenged_flag(self):
        # /api/bot/turnstile-verify はダイアログ表示時 challenged を送る
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("challenged: !!challenged", source)
        self.assertIn("async function verifyTurnstileOnServer(token, force = false, challenged = null)", source)

    def test_js_verified_user_not_blocked_when_token_unavailable(self):
        # 検証済みユーザーは、連打時にトークンが一時的に取れなくても
        # 「安全性を確認できませんでした」で送信をブロックしない
        # （サーバー側は Redis マーカーで通過できるため）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = get('prompt-input').value")]
        # ブロック条件は「トークンなし かつ 未検証」のときだけ
        self.assertIn("if (!botTurnstileToken && !botDetectionVerified)", gate)
        self.assertIn("if (botTurnstileToken) await verifyTurnstileOnServer(botTurnstileToken);", gate)

    def test_js_init_removes_hidden_before_render(self):
        # Turnstile は display:none コンテナでは初期化できないため、
        # 描画前に hidden を外してから render する（ボックスが出ない問題の修正）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        init = source[source.index("window.initTurnstileWidget = () => {") :]
        init = init[: init.index("if (isBotDetectionActive()) runBotDetectionGate();")]
        self.assertIn("container.classList.remove('hidden')", init)
        self.assertNotIn("container.classList.add('hidden')", init)

    def test_js_dialog_widget_retries_when_api_not_ready(self):
        # ダイアログの Turnstile ウィジェットは API 未ロード時に再試行する
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        dialog = source[source.index("function renderBotDetectionDialogWidget()") :]
        dialog = dialog[: dialog.index("function showBotDetectionOverlay")]
        self.assertIn("setTimeout(renderBotDetectionDialogWidget, 250)", dialog)
        self.assertIn("window.turnstile.reset", dialog)

    def test_css_dialog_card_stretches_widget_box(self):
        # ダイアログのカードが align-items:stretch で #bot-detection-widget-box を
        # 幅いっぱいに広げ、Turnstile の size:flexible が 0 幅にならないようにする
        assets = sorted((APP_ROOT / "static" / "css").glob("chat.custom.v4.8.*.css"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat custom asset should remain")
        css = assets[0].read_text(encoding="utf-8")
        self.assertIn(".turnstile-box { position: fixed; left: -9999px", css)

    def test_bot_lock_blocks_posts_and_returns_reason(self):
        # 連打ロック中はほとんどのPOSTが 403 account_locked でブロックされ、
        # ロック理由が返る
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_bot_lock(client, {"reason": "連打検出"})
                    self.assertEqual(res.status_code, 200)
                    self.assertEqual(res.get_json().get("status"), "locked")
                    # ロック中は chat_stream のPOSTが account_locked で拒否される
                    blocked = self.post_chat_stream(client, {"message": "hello", "turnstile_token": "valid"})
                    self.assertEqual(blocked.status_code, 403)
                    body = blocked.get_json()
                    self.assertEqual(body.get("error"), "account_locked")
                    self.assertIn("remaining_seconds", body)
                    self.assertTrue(body.get("remaining_seconds") > 0)

    def test_bot_lock_escalates_to_ban_after_repeats(self):
        # ロックが繰り返されると（別イベントとして3回）BANにエスカレーションする
        fake = _FakeRedis()
        with mock.patch.object(target, "redis_conn", fake):
            with self.turnstile_env():
                client = self.authenticated_client()
                # 1回目・2回目はロック（TTL切れで解除しながら別イベントを積む）
                for _ in range(2):
                    res = self.post_bot_lock(client, {"reason": "連打"})
                    self.assertEqual(res.get_json().get("status"), "locked")
                    fake.delete(f"bot:lock:{self.user_id}")  # TTL切れを模擬
                # 3回目の別ロックでBAN
                res = self.post_bot_lock(client, {"reason": "連打"})
                self.assertEqual(res.status_code, 403)
                self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)

    def test_bot_lock_admin_skipped(self):
        # 管理者はロック対象外
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            target.db.session.commit()
        fake = _FakeRedis()
        with mock.patch.object(target, "redis_conn", fake):
            with self.turnstile_env():
                client = self.authenticated_client()
                res = self.post_bot_lock(client, {"reason": "連打"})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.get_json().get("status"), "skipped")

    def test_banned_page_does_not_show_evidence(self):
        # BAN画面では不審履歴（evidence）を表示しない（管理者画面専用）
        html = (APP_ROOT / "templates" / "banned.html").read_text(encoding="utf-8")
        self.assertNotIn("アカウントの不審な履歴", html)
        self.assertNotIn("evidence-box", html)

    def test_js_send_spam_triggers_bot_check_dialog(self):
        # 送信ボタンの連打（5回/2秒）でアカウントロック（10分）をかけ、
        # 理由を表示してサーバー通信をブロックする（サーバー負荷対策）。
        # ただし検証済みユーザーにはロックをかけない。
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = get('prompt-input').value")]
        self.assertIn("registerSendButtonSpam()", gate)
        self.assertIn("sendCount >= 5", gate)
        self.assertIn("runSendSpamVerification()", gate)
        self.assertIn("送信操作が速すぎるため", gate)
        self.assertIn("registerSendButtonSpam", source)
        self.assertIn("runSendSpamVerification", source)
        # 検証済みユーザーは連打ロック対象外
        self.assertIn("isBotDetectionActive() && !botDetectionVerified", gate)

    def test_js_send_spam_locks_account_via_server(self):
        # runSendSpamVerification は /api/bot/lock を呼んでサーバー側で
        # アカウントをロックし、理由付きロック画面を表示する
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        fn = source[source.index("async function runSendSpamVerification()") :]
        fn = fn[: fn.index("let turnstileServerVerifiedAt = 0;")]
        self.assertIn("applyBotLockFromServer(", fn)
        self.assertIn("return await applyBotLockFromServer(", fn)
        # /api/bot/lock は applyBotLockFromServer 内にある（ソース全体で確認）
        self.assertIn("'/api/bot/lock'", source)

    def test_js_lock_overlay_and_endpoint(self):
        # ロック画面は理由と残り時間を表示し、403 account_locked で表示される
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("function showBotLockOverlay(", source)
        self.assertIn("function hideBotLockOverlay(", source)
        self.assertIn("bot-lock-overlay", source)
        self.assertIn("アカウントが一時的にロックされました", source)
        self.assertIn("errBody.error === 'account_locked'", source)
        # ロック画面はAPIリクエスト時にサーバーから取得した残り時間でカウントダウン
        self.assertIn("applyBotLockFromServer", source)
        # サーバーにロック状態取得エンドポイントが定義されている
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("/api/bot/lock-status", app_source)


if __name__ == "__main__":
    unittest.main()
