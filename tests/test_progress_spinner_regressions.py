from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
SPINNER_SOURCE = APP_ROOT / "static" / "js" / "progress_spinner.js"


class ProgressSpinnerRegressionTests(unittest.TestCase):
    def test_requests_are_tracked_by_lifetime_not_recent_click_window(self):
        source = SPINNER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("const operations = new Map()", source)
        self.assertIn("operations.set(operation.id, operation)", source)
        self.assertIn("keepOperationThroughResponseBody(response, finish)", source)
        self.assertIn("Promise.resolve(bodyResult).finally(finish)", source)
        self.assertIn("if (chunk && chunk.done) finish()", source)
        self.assertIn("this.addEventListener('loadend', finish", source)
        self.assertNotIn("USER_ACTION_WINDOW_MS", source)
        self.assertNotIn("isLikelyUserInitiated", source)

    def test_clicks_do_not_create_speculative_spinner_operations(self):
        source = SPINNER_SOURCE.read_text(encoding="utf-8")
        interaction = source[source.index("function installInteractionTracking()") :]
        click_handler = interaction[: interaction.index("document.addEventListener('keydown'")]

        self.assertIn("setInteractionContext", click_handler)
        self.assertNotIn("startOperation", click_handler)
        self.assertNotIn("startExpectedSlowPending", source)

    def test_passive_requests_and_explicit_opt_out_are_not_tracked(self):
        source = SPINNER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("PASSIVE_REQUEST_RE", source)
        self.assertIn("init.progressSpinner === false", source)
        self.assertIn("suppressCurrentInteraction", source)
        self.assertIn("data-progress-no-spinner", source)

    def test_spinner_cache_version_matches_system_version(self):
        import re
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        m = re.search(r"SYSTEM_VERSION'\]\s*=\s*'(V4\.8\.\d+)'", app_source)
        self.assertIsNotNone(m)
        self.assertIn(f"SYSTEM_VERSION'] = '{m.group(1)}'", app_source)
        for name in ("chat", "login", "signup", "verify_2fa", "setup", "landing", "banned"):
            template = (APP_ROOT / "templates" / f"{name}.html").read_text(encoding="utf-8")
            self.assertIn("filename='js/progress_spinner.js', v='4.8.722'", template)

    def test_chat_streams_remain_tracked_until_body_consumption_finishes(self):
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        source = assets[0].read_text(encoding="utf-8")

        self.assertEqual(source.count("window.ProgressSpinner.startFlow('chat')"), 1)
        self.assertEqual(source.count("window.ProgressSpinner.startFlow('chatResume')"), 1)
        self.assertEqual(source.count("window.ProgressSpinner.startFlow('browserFast')"), 1)
        # End at stream EOF, with an idempotent finally fallback for errors/aborts.
        self.assertEqual(source.count("if (finishStreamProgress) finishStreamProgress()"), 2)
        self.assertEqual(source.count("if (finishResumeProgress) finishResumeProgress()"), 2)
        self.assertIn("finishStreamProgress.setPhase('waiting')", source)
        self.assertIn("finishStreamProgress.setPhase('receiving')", source)
        self.assertIn("finishResumeProgress.setPhase('waiting')", source)
        self.assertIn("finishResumeProgress.setPhase('receiving')", source)
        self.assertGreaterEqual(source.count("manualSpinnerRequestOptions("), 4)
        self.assertNotIn("progressSpinner: false", source)
        self.assertNotIn("progressSpinner:false", source)

    def test_spinner_labels_switch_when_response_body_is_received(self):
        source = SPINNER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("const DEFAULT_SPINNER_TEXT = '通信中...'", source)
        self.assertIn("finishOperation.setLabel = function (label)", source)
        self.assertIn("finish.setLabel('受信中...')", source)
        self.assertNotIn("const DEFAULT_SPINNER_TEXT = '処理中...'", source)

    def test_manual_flow_labels_are_centralized_in_spinner_module(self):
        spinner_source = SPINNER_SOURCE.read_text(encoding="utf-8")
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        chat_source = assets[0].read_text(encoding="utf-8")

        self.assertIn("const PHASE_LABELS = Object.freeze({", spinner_source)
        self.assertIn("const FLOW_INITIAL_PHASES = Object.freeze({", spinner_source)
        self.assertIn("startFlow: function (flowName)", spinner_source)
        self.assertIn("finishOperation.setPhase = function (phase)", spinner_source)
        self.assertIn("manualRequestOptions: function (options)", spinner_source)
        self.assertNotIn("ProgressSpinner.start('", chat_source)
        self.assertNotIn(".setLabel('モデルの応答待機中...')", chat_source)

    def test_read_only_requests_ignore_action_button_labels(self):
        source = SPINNER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("function requestSpinnerLabel(url, method)", source)
        self.assertIn("if (methodText === 'GET' || methodText === 'HEAD')", source)
        self.assertIn("return inferSpinnerTextFromUrl(url, methodText)", source)
        self.assertIn("const label = requestSpinnerLabel(details.url, details.method)", source)
        self.assertIn("const label = requestSpinnerLabel(this.__progressSpinnerUrl, this.__progressSpinnerMethod)", source)

    def test_image_generation_progress_does_not_replace_pending_skeleton(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn('pub("content", "**Generating Image (Grok)...**', app_source)
        self.assertNotIn('pub("content", "**Generating Image (OpenAI)...**', app_source)
        self.assertIn('pub("status", "画像を生成中...")', app_source)
        self.assertIn('pub("status", "画像生成の準備中...")', app_source)


if __name__ == "__main__":
    unittest.main()
