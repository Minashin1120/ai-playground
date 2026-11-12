"""Turnstile verification, gate, and client-side challenge regressions."""
import unittest
from unittest import mock

from tests.bot_detection_test_support import (
    APP_ROOT,
    BotDetectionTestBase,
    _FakeRedis,
    read_app_source,
    read_chat_markup,
    target,
)


class TurnstileChallengeRegressionTests(BotDetectionTestBase):
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

    def test_delete_account_not_blocked_by_turnstile_gate(self):
        # アカウントの自己削除は認証済みの破壊的操作であり、Turnstile 未検証でも
        # gate にブロックされず実行できる（未検証ユーザーが自分のアカウントを
        # 削除できなくなる事象の回帰防止）。
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", _FakeRedis()):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = client.post(
                        "/api/account/delete",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("status"), "ok")
        with target.app.app_context():
            self.assertIsNone(target.db.session.get(target.User, self.user_id))

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
        gate = gate[: gate.index("const rawText = ")]
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
        source = read_app_source()
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
                        self.clear_fail_cooldown(fake)
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
                        self.clear_fail_cooldown(fake)
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
                        self.clear_fail_cooldown(fake)
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
                    for i in range(12):
                        self.clear_fail_cooldown(fake)
                        # Unique tokens: same token is deduped and not re-counted
                        res = self.post_turnstile_verify(client, {"turnstile_token": f"invalid-{i}"})
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
                    for i in range(5):
                        self.clear_fail_cooldown(fake)
                        res = self.post_turnstile_verify(client, {
                            "turnstile_token": f"invalid-{i}",
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
                        self.clear_fail_cooldown(fake)
                        self.post_telemetry(client, {
                            "turnstile_failed": True,
                            "challenged": True,
                            "window_ms": 4000,
                            "clicks": 2,
                            "keys": 2,
                            "moves": 2,
                        })
                    res = self.post_turnstile_verify(client, {"turnstile_token": "t-ok"})
                    self.assertEqual(res.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_turnstile_pass_fail_cycling_bans(self):
        # ダイアログ表示中の成功→失敗が繰り返される（パス/フェイルのサイクリング）なら BAN
        # 検証マーカー失効後の再チャレンジをシミュレートするため、失敗の前に verified を消す
        fake = _FakeRedis()
        results = [False, True, False, True, False, True]
        with mock.patch.object(target, "verify_turnstile", side_effect=results):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = None
                    for i, r in enumerate(results):
                        if r:
                            res = self.post_turnstile_verify(client, {"turnstile_token": f"ok-{i}"})
                        else:
                            self.clear_verified_marker(fake)
                            self.clear_fail_cooldown(fake)
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

    def test_same_token_concurrent_fails_after_success_do_not_ban(self):
        # 再現: アカウント作成直後、同一 Turnstile トークンが複数経路から同時送信され
        # verify_ok の直後に verify_fail が同一秒で積み上がって即 BAN されていた問題。
        # 成功後の同一トークン再送・検証済みユーザーの失敗は BAN に数えない。
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", side_effect=[True] + [False] * 8):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    ok = self.post_turnstile_verify(client, {
                        "turnstile_token": "shared-token",
                        "challenged": True,
                    })
                    self.assertEqual(ok.status_code, 200)
                    for _ in range(8):
                        self.clear_fail_cooldown(fake)
                        res = self.post_turnstile_verify(client, {
                            "turnstile_token": "shared-token",
                            "challenged": True,
                        })
                        # Already verified / token dedup → soft ok or soft fail, never ban
                        self.assertNotEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)

    def test_duplicate_token_submits_do_not_stack_fail_count(self):
        # 同一トークンの多重送信は1回分しか失敗カウントしない（dedup）
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    for _ in range(10):
                        self.clear_fail_cooldown(fake)
                        res = self.post_turnstile_verify(client, {
                            "turnstile_token": "same-bad-token",
                            "challenged": True,
                        })
                        self.assertEqual(res.status_code, 403)
                        self.assertEqual(res.get_json().get("error"), "turnstile_failed")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)
        # Only the first attempt may claim the token; fail count stays at 0 or 1
        fail_count = int(fake.get(f"bot:tst:fail:{self.user_id}") or 0)
        self.assertLessEqual(fail_count, 1)

    def test_fail_cooldown_prevents_burst_ban(self):
        # クールダウン中の連続失敗は1回しかカウントされず、バーストで即 BAN しない
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    # Do NOT clear cooldown between posts
                    for i in range(10):
                        res = self.post_turnstile_verify(client, {
                            "turnstile_token": f"burst-bad-{i}",
                            "challenged": True,
                        })
                        self.assertEqual(res.status_code, 403)
                        self.assertNotEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertFalse(user.is_bot_banned)
        fail_count = int(fake.get(f"bot:tst:fail:{self.user_id}") or 0)
        self.assertEqual(fail_count, 1)

    def test_js_verify_is_single_flight_per_token(self):
        # クライアントは同一トークンのサーバー検証を単一フライトにし、多重 POST しない
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("turnstileVerifyInFlight", source)
        self.assertIn("turnstileLastSubmittedToken", source)
        self.assertIn("Turnstile tokens are single-use", source)

    def test_js_api_fetch_does_not_retry_turnstile_failed(self):
        # turnstile_failed / banned を CSRF と誤認して同一ボディで再送しない
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        api = source[source.index("const apiFetch = async (url, opts = {}) => {") :]
        api = api[: api.index("window.updateGoogleLinkUI")]
        self.assertIn("botErr === 'banned' || botErr === 'turnstile_failed'", api)
        self.assertIn("return response;", api)



if __name__ == "__main__":
    unittest.main()
