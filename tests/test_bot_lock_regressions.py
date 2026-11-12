"""Bot lock, telemetry scoring, ban, and appeal regressions."""
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


class BotLockRegressionTests(BotDetectionTestBase):
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
        html = read_chat_markup()
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
        self.assertIn("botErr === 'turnstile_required'", source)

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
                        self.clear_fail_cooldown(fake)
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
                        self.clear_fail_cooldown(fake)
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
        self.assertIn("challenged: challengedFlag", source)
        self.assertIn("async function verifyTurnstileOnServer(token, force = false, challenged = null)", source)

    def test_js_verified_user_not_blocked_when_token_unavailable(self):
        # 検証済みユーザーは、連打時にトークンが一時的に取れなくても
        # 「安全性を確認できませんでした」で送信をブロックしない
        # （サーバー側は Redis マーカーで通過できるため）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = ")]
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
                # 管理者にはロックキーが作られない
                lock_keys = [
                    k for k in fake._d
                    if (k if isinstance(k, str) else k.decode()).startswith("bot:lock:")
                ]
                self.assertEqual(lock_keys, [])

    def test_admin_not_locked_via_ip_cookie_cascade(self):
        # 一般ユーザーの連打ロックは IP/クッキーにも記録されるが、
        # 同一IPの管理者アカウントはロック対象外（BANの関連アカウント除外と同じ方針）。
        fake = _FakeRedis()
        with target.app.app_context():
            admin = target.User(
                username="admin-lock-exempt",
                is_setup_completed=True,
                is_admin=True,
            )
            admin.set_password("test-password")
            target.db.session.add(admin)
            target.db.session.commit()
            admin_id = admin.id
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    normal = self.authenticated_client()
                    res = self.post_bot_lock(normal, {"reason": "連打検出"})
                    self.assertEqual(res.status_code, 200)
                    self.assertEqual(res.get_json().get("status"), "locked")
                    # 一般ユーザーはロック中
                    blocked = self.post_chat_stream(
                        normal, {"message": "hello", "turnstile_token": "valid"}
                    )
                    self.assertEqual(blocked.status_code, 403)
                    self.assertEqual(blocked.get_json().get("error"), "account_locked")
                    # 同一環境の管理者は lock-status が inactive で POST も account_locked にならない
                    admin_client = target.app.test_client()
                    with admin_client.session_transaction() as sess:
                        sess["_user_id"] = str(admin_id)
                        sess["_fresh"] = True
                        sess["csrf_token"] = "csrf-test-token"
                    status = admin_client.get("/api/bot/lock-status")
                    self.assertEqual(status.status_code, 200)
                    self.assertFalse(status.get_json().get("active"))
                    admin_post = admin_client.post(
                        "/chat_stream",
                        data=target.json.dumps(
                            {"message": "hello", "turnstile_token": "valid"}
                        ),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    admin_body = admin_post.get_json() or {}
                    self.assertNotEqual(admin_body.get("error"), "account_locked")

    def test_setup_not_blocked_by_inherited_ip_cookie_lock(self):
        # 同一IP/クッキーのロックを引き継いだ新規アカウントでも、
        # 初期セットアップの完了は 403 account_locked にならない。
        # セットアップ完了後は通常どおりロックが適用される。
        fake = _FakeRedis()
        with target.app.app_context():
            fresh = target.User(username="setup-under-lock", is_setup_completed=False)
            fresh.set_password("test-password")
            target.db.session.add(fresh)
            target.db.session.commit()
            fresh_id = fresh.id
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    normal = self.authenticated_client()
                    res = self.post_bot_lock(normal, {"reason": "連打検出"})
                    self.assertEqual(res.get_json().get("status"), "locked")
                    fresh_client = target.app.test_client()
                    with fresh_client.session_transaction() as sess:
                        sess["_user_id"] = str(fresh_id)
                        sess["_fresh"] = True
                        sess["csrf_token"] = "csrf-test-token"
                    setup_res = fresh_client.post(
                        "/setup",
                        data={"csrf_token": "csrf-test-token", "default_model": "gemini-3.6-flash"},
                    )
                    self.assertEqual(setup_res.status_code, 302)
                    self.assertEqual(setup_res.headers["Location"], "/")
                    with target.app.app_context():
                        self.assertTrue(target.db.session.get(target.User, fresh_id).is_setup_completed)
                    blocked = fresh_client.post(
                        "/chat_stream",
                        data=target.json.dumps({"message": "hello", "turnstile_token": "valid"}),
                        content_type="application/json",
                        headers={"X-CSRF-Token": "csrf-test-token"},
                    )
                    self.assertEqual(blocked.status_code, 403)
                    self.assertEqual(blocked.get_json().get("error"), "account_locked")

    def test_js_admin_skips_lock_overlay(self):
        # 管理者はロック画面を出さない（bootstrap / apiFetch / applyBotLockFromServer）
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        self.assertIn("botConfig.lock.active && !isAdminUser", source)
        self.assertIn("if (isAdminUser) return true;", source)
        self.assertIn("!isAdminUser && !document.getElementById('bot-lock-overlay')", source)

    def test_banned_page_does_not_show_evidence(self):
        # BAN画面では不審履歴（evidence）を表示しない（管理者画面専用）
        html = (APP_ROOT / "templates" / "banned.html").read_text(encoding="utf-8")
        self.assertNotIn("アカウントの不審な履歴", html)
        self.assertNotIn("evidence-box", html)

    def test_js_send_spam_triggers_bot_check_dialog(self):
        # 送信ボタンの連打（8回/3秒）でアカウントロック（10分）をかけ、
        # 理由を表示してサーバー通信をブロックする（サーバー負荷対策）。
        # 検証済みユーザーでも発動する（DOM連打を検出するため）。
        # 閾値は通常の再試行・ダブルタップで誤ロックしにくいよう緩めている。
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        gate = source[source.index("async function sendMessage()") :]
        gate = gate[: gate.index("const rawText = ")]
        self.assertIn("registerSendButtonSpam()", gate)
        self.assertIn("sendCount >= 8", gate)
        self.assertIn("runSendSpamVerification()", gate)
        self.assertIn("送信操作が速すぎるため", gate)
        self.assertIn("registerSendButtonSpam", source)
        self.assertIn("runSendSpamVerification", source)
        # 3秒窓で集計（2秒から緩和）
        self.assertIn("now - t <= 3000", source)
        # 検証済みユーザーでも連打ロック対象（!botDetectionVerified 条件を外す）
        self.assertIn("if (isBotDetectionActive()) {", gate)
        self.assertNotIn("isBotDetectionActive() && !botDetectionVerified", gate)

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
        self.assertIn("botErr === 'account_locked'", source)
        # ロック画面はAPIリクエスト時にサーバーから取得した残り時間でカウントダウン
        self.assertIn("applyBotLockFromServer", source)
        # サーバーにロック状態取得エンドポイントが定義されている
        app_source = read_app_source()
        self.assertIn("/api/bot/lock-status", app_source)

    def test_js_detects_synthetic_events(self):
        # コンソール等の合成イベント（isTrusted===false）を検出し、
        # untrusted_input を付けて即報告（BANの根拠）する
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1, "Only the latest versioned chat core asset should remain")
        source = assets[0].read_text(encoding="utf-8")
        telemetry = source[source.index("const botTelemetry = (() => {") :]
        telemetry = telemetry[: telemetry.index("return { start, refreshEnabled, send, looksSuspicious };")]
        self.assertIn("e.isTrusted === false", telemetry)
        self.assertIn("state.untrustedInput = true", telemetry)
        self.assertIn("untrusted_input: !!state.untrustedInput", telemetry)

    def test_untrusted_input_bans_immediately(self):
        # untrusted_input 付きテレメトリは即BAN
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=False):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    res = self.post_telemetry(client, {
                        "untrusted_input": True,
                        "window_ms": 1000,
                        "clicks": 0,
                        "keys": 0,
                        "moves": 0,
                        "turnstile_failed": True,
                    })
                    self.assertEqual(res.status_code, 403)
                    self.assertEqual(res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)
            self.assertIn("Synthetic", user.bot_ban_reason)

    def test_lock_applies_to_ip_and_cookie(self):
        # ロックは IP / クッキーにも記録され、別アカウントでも同一IPならロックされる
        fake = _FakeRedis()
        with mock.patch.object(target, "redis_conn", fake):
            with self.turnstile_env():
                client = self.authenticated_client()
                res = self.post_bot_lock(client, {"reason": "連打検出"})
                self.assertEqual(res.get_json().get("status"), "locked")
                # IP ロックキーが存在する（クッキーはテストでは未設定の場合があるため IP を確認）
                ip_keys = [k for k in fake._d if b"bot:lock:ip:" in (k if isinstance(k, bytes) else k.encode())]
                self.assertTrue(ip_keys, "IP lock key should be recorded")

    def test_bot_telemetry_server_rejects_untrusted_flag(self):
        # サーバー側 /api/bot-telemetry が untrusted_input を処理する
        app_source = read_app_source()
        self.assertIn("data.get('untrusted_input')", app_source)
        self.assertIn("_apply_bot_ban(\"Synthetic (script-injected) input events detected\")", app_source)

    def test_lock_gate_allows_bot_telemetry_and_turnstile_verify(self):
        # アカウントロック中でもボット検出テレメトリ／Turnstile検証は到達できる
        # （ロック中に合成イベント等を送ってもBANへ進めるため）
        app_source = read_app_source()
        # _BOT_LOCK_GATE_WHITELIST 内に両エンドポイントがあること
        whitelist_start = app_source.index("_BOT_LOCK_GATE_WHITELIST = {")
        whitelist = app_source[whitelist_start: whitelist_start + 800]
        self.assertIn("'bot_telemetry'", whitelist)
        self.assertIn("'bot_turnstile_verify'", whitelist)

    def test_untrusted_input_bans_while_account_locked(self):
        # ロック中に untrusted_input テレメトリを送ると、ロックをすり抜けずBANされる
        fake = _FakeRedis()
        # verify_turnstile=True で Turnstile ゲートを通過させ、ロックゲートを検証する
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    lock_res = self.post_bot_lock(client, {"reason": "連打検出"})
                    self.assertEqual(lock_res.status_code, 200)
                    self.assertEqual(lock_res.get_json().get("status"), "locked")
                    # ロック中でも chat_stream は拒否される（ロック自体は有効）
                    blocked = self.post_chat_stream(
                        client, {"message": "hello", "turnstile_token": "valid"}
                    )
                    self.assertEqual(blocked.status_code, 403)
                    self.assertEqual(blocked.get_json().get("error"), "account_locked")
                    # ロック中でも bot-telemetry の untrusted_input は到達してBAN
                    # （untrusted_input は Turnstile 状態に関係なく即BAN）
                    ban_res = self.post_telemetry(client, {
                        "untrusted_input": True,
                        "window_ms": 1000,
                        "clicks": 0,
                        "keys": 0,
                        "moves": 0,
                        "turnstile_token": "valid",
                    })
                    self.assertEqual(ban_res.status_code, 403)
                    self.assertEqual(ban_res.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)
            self.assertIn("Synthetic", user.bot_ban_reason)

    def test_behavior_score_ban_while_account_locked(self):
        # ロック中でも挙動スコアBAN（累積スコア>=8 かつ 挙動>=6）が適用される
        fake = _FakeRedis()
        with mock.patch.object(target, "verify_turnstile", return_value=True):
            with mock.patch.object(target, "redis_conn", fake):
                with self.turnstile_env():
                    client = self.authenticated_client()
                    lock_res = self.post_bot_lock(client, {"reason": "連打検出"})
                    self.assertEqual(lock_res.get_json().get("status"), "locked")
                    # 高スコアの挙動テレメトリを複数回送りBAN条件を満たす
                    high_score_payload = {
                        "window_ms": 2000,
                        "clicks": 40,
                        "keys": 0,
                        "moves": 0,
                        "fast_clicks": 30,
                        "fast_keys": 0,
                        "click_burst": 40,
                        "key_burst": 0,
                        "event_rate": 30,
                        "avg_click_ms": 50,
                        "click_cv": 0.02,
                        "pointer_speed_max": 0,
                        "turnstile_token": "valid",
                    }
                    last = None
                    for _ in range(3):
                        last = self.post_telemetry(client, high_score_payload)
                        if last.status_code == 403 and last.get_json().get("error") == "banned":
                            break
                    self.assertIsNotNone(last)
                    self.assertEqual(last.status_code, 403)
                    self.assertEqual(last.get_json().get("error"), "banned")
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.is_bot_banned)




if __name__ == "__main__":
    unittest.main()
