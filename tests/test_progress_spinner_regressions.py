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
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("SYSTEM_VERSION'] = 'V4.8.650'", app_source)
        for name in ("chat", "login", "signup", "verify_2fa", "setup", "landing", "banned"):
            template = (APP_ROOT / "templates" / f"{name}.html").read_text(encoding="utf-8")
            self.assertIn("filename='js/progress_spinner.js', v='4.8.640'", template)

    def test_chat_streams_remain_tracked_until_body_consumption_finishes(self):
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        source = assets[0].read_text(encoding="utf-8")

        self.assertEqual(source.count("window.ProgressSpinner.start('送信中...')"), 1)
        self.assertEqual(source.count("window.ProgressSpinner.start('生成中...')"), 1)
        self.assertEqual(source.count("if (finishStreamProgress) finishStreamProgress()"), 1)
        self.assertEqual(source.count("if (finishResumeProgress) finishResumeProgress()"), 1)


if __name__ == "__main__":
    unittest.main()
