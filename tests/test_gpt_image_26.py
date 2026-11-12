from pathlib import Path
import unittest

from tests.app_source import read_app_source
from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = read_chat_markup()


class GptImage26Tests(unittest.TestCase):
    MODELS = ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare")

    def test_models_are_registered_across_web_and_backend(self):
        valid_ids = APP_SOURCE[APP_SOURCE.index("ALL_VALID_MODEL_IDS"): APP_SOURCE.index("def is_sts_model")]
        for model in self.MODELS:
            self.assertIn(f'id: "{model}"', CHAT_JS)
            item = CHAT_JS[CHAT_JS.index(f'id: "{model}"'):]
            self.assertIn("implementedAt:", item[:200])
            self.assertIn("quickEmoji:", item[:200])
            self.assertIn(f'"{model}"', valid_ids)

    def test_gpt_image_26_uses_the_images_api(self):
        branch = (APP_ROOT / "server" / "background_stream_gpt_image.py").read_text(encoding="utf-8")
        self.assertIn("is_gpt_image_26", branch)
        self.assertIn("img_client.images.generate", branch)
        self.assertIn("img_client.images.edit", branch)
        self.assertIn('extra_body["complexity"]', branch)
        self.assertIn('extra_body["frame_count"]', branch)
        self.assertIn('extra_body["frame_duration_ms"]', branch)
        # Earlier models and 2.5 keep the Responses API image_generation tool.
        self.assertIn('model="gpt-4o-mini"', branch)

    def test_quality_and_format_limits(self):
        branch = (APP_ROOT / "server" / "background_stream_gpt_image.py").read_text(encoding="utf-8")
        self.assertIn('"gpt-image-2.6-sunburst", "gpt-image-2.6-flare"', branch)
        self.assertIn('format_values.add("gif")', branch)
        self.assertIn('quality_values.update({"xhigh", "max"})', branch)

    def test_composer_and_modal_controls_are_wired(self):
        for control in (
            "gpt-image-complexity",
            "gpt-image-frame-count",
            "gpt-image-frame-duration",
            "modal-gpt-image-complexity",
            "modal-gpt-image-frame-count",
            "modal-gpt-image-frame-duration",
        ):
            self.assertIn(f'id="{control}"', CHAT_HTML)
        self.assertIn('<option value="gif" data-gpt-image-26 hidden>GIF</option>', CHAT_HTML)
        self.assertIn("image_complexity:", CHAT_JS)
        self.assertIn("image_frame_count:", CHAT_JS)
        self.assertIn("image_frame_duration_ms:", CHAT_JS)
        self.assertIn("'image_complexity': data.get('image_complexity')", APP_SOURCE)
        self.assertIn("'image_frame_count': data.get('image_frame_count')", APP_SOURCE)

    def test_android_direct_mode_uses_the_images_api(self):
        source = (APP_ROOT / "android/app/src/main/java/com/minashin1120/aiplayground/data/direct/MediaDirect.kt").read_text(encoding="utf-8")
        self.assertIn("buildImagesApiPayload", source)
        self.assertIn("/v1/images/generations", source)
        self.assertIn("/v1/images/edits", source)
        self.assertIn('put("complexity"', source)
        self.assertIn('put("frame_count"', source)
        self.assertIn('put("frame_duration_ms"', source)


if __name__ == "__main__":
    unittest.main()
