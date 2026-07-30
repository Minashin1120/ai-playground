from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")
SETUP_HTML = (APP_ROOT / "templates/setup.html").read_text(encoding="utf-8")


class NanoBanana2LiteRegressionTests(unittest.TestCase):
    MODEL_ID = "gemini-3.1-flash-lite-image"

    def test_model_is_registered_across_ui_and_backend(self):
        for source in (APP_SOURCE, CHAT_JS, CHAT_HTML, SETUP_HTML):
            self.assertIn(self.MODEL_ID, source)

    def test_lite_routes_to_the_stable_model_id(self):
        route = APP_SOURCE[APP_SOURCE.index(
            'if "gemini-3.1-flash-lite-image" in mk_lower'
        ):]
        self.assertIn('img_model = "gemini-3.1-flash-lite-image"', route[:300])

    def test_lite_supports_only_official_thinking_levels(self):
        self.assertIn(
            'img_model in ("gemini-3.1-flash-image-preview", '
            '"gemini-3.1-flash-lite-image")',
            APP_SOURCE,
        )
        self.assertIn(
            'default_level = "minimal" if img_model == '
            '"gemini-3.1-flash-lite-image" else "high"',
            APP_SOURCE,
        )
        self.assertIn("isNanoBanana2Lite", CHAT_JS)
        self.assertIn("['minimal', 'high'].includes(opt.value)", CHAT_JS)

    def test_lite_forces_1k_and_does_not_enable_search(self):
        self.assertIn(
            'if img_model == "gemini-3.1-flash-lite-image":\n'
            '                            # Nano Banana 2 Lite supports 1K output only.\n'
            '                            image_cfg_kwargs["image_size"] = "1K"',
            APP_SOURCE,
        )
        self.assertIn(
            'if img_model == "gemini-3.1-flash-image-preview" and '
            "options.get('enable_search')",
            APP_SOURCE,
        )
        self.assertIn("searchChk.disabled = true", CHAT_JS)
        self.assertIn(
            "!isDeepSeek && !isNanoBanana2Lite",
            CHAT_JS,
        )

    def test_thought_parts_are_not_published_as_final_images(self):
        self.assertIn(
            'if bool(getattr(_part, "thought", False)):\n'
            "                                        continue",
            APP_SOURCE,
        )

    def test_all_fourteen_aspect_ratios_are_available(self):
        for ratio in (
            "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
            "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
        ):
            self.assertIn(f'<option value="{ratio}">', CHAT_HTML)


if __name__ == "__main__":
    unittest.main()
