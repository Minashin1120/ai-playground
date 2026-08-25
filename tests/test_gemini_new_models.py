from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
SETUP_HTML = (APP_ROOT / "templates/setup.html").read_text(encoding="utf-8")


class GeminiNewModelsRegressionTests(unittest.TestCase):
    CHAT_AND_IMAGE_MODELS = (
        "gemini-2.5-pro",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
        "gemini-3.5-live-translate-preview",
    )
    VIDEO_MODELS = (
        "gemini-omni-flash",
        "veo-3.1-generate-preview",
        "veo-3.1-fast-generate-preview",
        "veo-3.1-lite-generate-preview",
    )
    MUSIC_MODELS = (
        "lyria-3-pro-preview",
        "lyria-3-clip-preview",
        "lyria-realtime-exp",
    )
    SPECIALIZED_MODELS = (
        "gemini-robotics-er-2-preview",
        "deep-research-preview-04-2026",
        "deep-research-max-preview-04-2026",
        "antigravity-preview-05-2026",
        "gemini-2.5-computer-use-preview-10-2025",
        "gemini-embedding-2",
    )
    ALL_MODELS = CHAT_AND_IMAGE_MODELS + VIDEO_MODELS + MUSIC_MODELS + SPECIALIZED_MODELS

    def test_all_new_models_are_registered_across_ui_and_backend(self):
        for model_id in self.ALL_MODELS:
            self.assertIn(model_id, APP_SOURCE, model_id)
            self.assertIn(model_id, CHAT_JS, model_id)
            self.assertIn(f'id: "{model_id}"', CHAT_JS, model_id)
            definition_at = CHAT_JS.index(f'id: "{model_id}"')
            self.assertIn(
                "implementedAt:",
                CHAT_JS[definition_at:definition_at + 160],
                model_id,
            )

    def test_main_chat_and_media_models_appear_in_setup_defaults(self):
        for model_id in (
            "gemini-2.5-pro",
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-3.5-live-translate-preview",
            "gemini-omni-flash",
            "veo-3.1-generate-preview",
            "veo-3.1-fast-generate-preview",
            "veo-3.1-lite-generate-preview",
            "lyria-3-pro-preview",
            "lyria-3-clip-preview",
            "deep-research-preview-04-2026",
            "deep-research-max-preview-04-2026",
            "antigravity-preview-05-2026",
        ):
            self.assertIn(model_id, SETUP_HTML, model_id)

    def test_shut_down_previews_are_marked_deprecated(self):
        for model_id, name_hint in (
            ("gemini-3-pro-preview", "Gemini 3.0 Pro"),
            ("gemini-3.1-flash-image-preview", "Nano Banana 2 (Preview)"),
            ("gemini-3-pro-image-preview", "Nano Banana Pro (Preview)"),
        ):
            definition_at = CHAT_JS.index(f'id: "{model_id}"')
            self.assertIn(
                "deprecated: true",
                CHAT_JS[definition_at:definition_at + 400],
                model_id,
            )
            self.assertIn(
                name_hint,
                CHAT_JS[definition_at:definition_at + 400],
                model_id,
            )

    def test_stable_image_models_route_to_stable_endpoints(self):
        route = APP_SOURCE[APP_SOURCE.index('if "gemini-3.1-flash-lite-image" in mk_lower'):]
        self.assertIn('img_model = "gemini-3.1-flash-image"', route[:500])
        self.assertIn('img_model = "gemini-3-pro-image"', route[:500])

    def test_gemini_25_pro_routes_before_flash_catch_all(self):
        route = APP_SOURCE[APP_SOURCE.index('if "gemini-3.7-flash" in model_key'):]
        self.assertLess(
            route.index('elif "gemini-2.5-pro" in model_key'),
            route.index('elif "gemini-2.5" in model_key'),
        )

    def test_live_translate_is_registered_for_sts(self):
        self.assertIn('"gemini-3.5-live-translate-preview"', APP_SOURCE)
        self.assertIn(
            '"gemini-3.5-live-translate-preview"',
            APP_SOURCE[APP_SOURCE.index("STS_MODELS = {"):APP_SOURCE.index("XAI_STS_MODEL_ALIASES")],
        )
        self.assertIn("'gemini-3.5-live-translate-preview'", CHAT_JS)
        self.assertIn("isGeminiLiveTranslateModel", CHAT_JS)

    def test_gemini_video_music_embedding_routing_branches_exist(self):
        self.assertIn("Routing: Gemini Video Branch", APP_SOURCE)
        self.assertIn("Routing: Gemini Music Branch", APP_SOURCE)
        self.assertIn("Routing: Gemini Embedding Branch", APP_SOURCE)
        self.assertIn("is_gemini_video_model_key", APP_SOURCE)
        self.assertIn("is_gemini_music_model_key", APP_SOURCE)
        self.assertIn("is_gemini_embedding_model_key", APP_SOURCE)
        self.assertIn("isGeminiVideoModel", CHAT_JS)
        self.assertIn("isGeminiMusicModel", CHAT_JS)
        self.assertIn("isGeminiEmbeddingModel", CHAT_JS)

    def test_gemini_video_and_music_options_ui_exist(self):
        self.assertIn('id="gemini-video-options"', (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8"))
        self.assertIn('id="gemini-music-options"', (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8"))
        self.assertIn("updateGeminiVideoUi", CHAT_JS)
        self.assertIn("updateGeminiMusicUi", CHAT_JS)


if __name__ == "__main__":
    unittest.main()
