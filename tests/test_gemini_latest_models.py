from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")
SETUP_HTML = (APP_ROOT / "templates/setup.html").read_text(encoding="utf-8")


class GeminiLatestModelsRegressionTests(unittest.TestCase):
    def test_latest_gemini_models_are_registered_across_ui_and_backend(self):
        for model_id in (
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ):
            self.assertIn(model_id, APP_SOURCE)
            self.assertIn(model_id, CHAT_JS)
            self.assertIn(model_id, CHAT_HTML)
            self.assertIn(model_id, SETUP_HTML)

    def test_latest_gemini_routing_precedes_flash_substring_match(self):
        route = APP_SOURCE[APP_SOURCE.index('if "gemini-3.6-flash" in model_key'):]
        self.assertLess(
            route.index('if "gemini-3.6-flash" in model_key'),
            route.index('elif "gemini-3.5-flash-lite" in model_key'),
        )
        self.assertLess(
            route.index('elif "gemini-3.5-flash-lite" in model_key'),
            route.index('elif "gemini-3.5-flash" in model_key'),
        )

    def test_latest_models_omit_deprecated_sampling_parameters(self):
        self.assertIn(
            'conf = {} if is_latest_flash else {\'temperature\': 0.7}',
            APP_SOURCE,
        )
        self.assertIn(
            'rm in ("gemini-3.6-flash", "gemini-3.5-flash-lite")',
            APP_SOURCE,
        )

    def test_thinking_levels_follow_official_model_constraints(self):
        self.assertIn(
            'rm == "gemini-3.6-flash" and lvl not in ("medium", "high")',
            APP_SOURCE,
        )
        self.assertIn(
            'rm == "gemini-3.5-flash-lite" and lvl not in '
            '("minimal", "medium", "high")',
            APP_SOURCE,
        )
        self.assertIn("model === 'gemini-3.6-flash'", CHAT_JS)
        self.assertIn("model === 'gemini-3.5-flash-lite'", CHAT_JS)

    def test_stable_flash_lite_is_active_and_preview_is_kept_but_hidden(self):
        stable_definition = (
            '{ id: "gemini-3.1-flash-lite", '
            'name: "Gemini 3.1 Flash-Lite"'
        )
        stable_at = CHAT_JS.index(stable_definition)
        self.assertNotIn(
            "deprecated: true",
            CHAT_JS[stable_at:stable_at + 400],
        )
        model_definition = (
            '{ id: "gemini-3.1-flash-lite-preview", '
            'name: "Gemini 3.1 Flash-Lite Preview"'
        )
        definition_at = CHAT_JS.index(model_definition)
        self.assertIn(
            "deprecated: true",
            CHAT_JS[definition_at:definition_at + 400],
        )
        self.assertIn("gemini-3.1-flash-lite-preview", APP_SOURCE)
        self.assertNotIn("gemini-3.1-flash-lite-preview", SETUP_HTML)
        self.assertIn(
            "UPDATE user SET default_model='gemini-3.1-flash-lite'",
            APP_SOURCE,
        )
        self.assertIn(
            "UPDATE user SET last_model='gemini-3.1-flash-lite'",
            APP_SOURCE,
        )

    def test_stable_flash_lite_routes_before_retired_preview(self):
        route = APP_SOURCE[APP_SOURCE.index(
            'elif model_key == "gemini-3.1-flash-lite"'
        ):]
        self.assertLess(
            route.index('elif model_key == "gemini-3.1-flash-lite"'),
            route.index('elif "gemini-3.1-flash-lite-preview" in model_key'),
        )
