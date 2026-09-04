from pathlib import Path
import unittest

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = read_chat_markup()
SETUP_HTML = (APP_ROOT / "templates" / "setup.html").read_text(encoding="utf-8")


class GrokImagineImage2RegressionTests(unittest.TestCase):
    MODEL_ID = "grok-imagine-image-2.0"

    def test_model_is_registered_across_ui_and_backend(self):
        for source in (APP_SOURCE, CHAT_JS, SETUP_HTML):
            self.assertIn(self.MODEL_ID, source)
        self.assertIn(f'id: "{self.MODEL_ID}"', CHAT_JS)
        self.assertIn("implementedAt:", CHAT_JS[CHAT_JS.index(f'id: "{self.MODEL_ID}"'): CHAT_JS.index(f'id: "{self.MODEL_ID}"') + 160])
        self.assertIn(f'<option value="{self.MODEL_ID}">Grok Imagine Image 2.0</option>', SETUP_HTML)
        valid_ids = APP_SOURCE[APP_SOURCE.index("ALL_VALID_MODEL_IDS"): APP_SOURCE.index("def is_sts_model")]
        self.assertIn(f'"{self.MODEL_ID}"', valid_ids)

    def test_imagine_branch_accepts_image_2_and_official_parameters(self):
        branch = APP_SOURCE[APP_SOURCE.index("Routing: Grok Imagine Branch"):]
        self.assertIn(f'"{self.MODEL_ID}"', APP_SOURCE[APP_SOURCE.index("# --- 1.5 Grok Imagine Image Generation ---"): APP_SOURCE.index("Routing: Grok Imagine Branch") + 400])
        self.assertIn('grok_supports_quality = model_key == "grok-imagine-image-2.0"', branch)
        self.assertIn('eb["quality"] = quality', branch)
        self.assertIn('payload["quality"] = quality', branch)
        self.assertIn('eb["resolution"] = resolution', branch)
        self.assertIn("'grok_image_resolution': data.get('grok_image_resolution')", APP_SOURCE)
        self.assertIn("'grok_image_quality': data.get('grok_image_quality')", APP_SOURCE)

    def test_quality_and_resolution_controls_are_wired(self):
        self.assertIn('id="grok-image-quality"', CHAT_HTML)
        self.assertIn('id="modal-grok-image-quality"', CHAT_HTML)
        self.assertIn('option value="medium"', CHAT_HTML)
        self.assertIn('option value="low"', CHAT_HTML)
        self.assertIn("grok_image_quality:", CHAT_JS)
        self.assertIn("showQuality = model === 'grok-imagine-image-2.0'", CHAT_JS)
        self.assertIn("model === 'grok-imagine-image-quality' || model === 'grok-imagine-image-2.0'", CHAT_JS)
        self.assertIn("sync('grok-image-quality', 'modal-grok-image-quality')", CHAT_JS)
        self.assertIn("syncBack('modal-grok-image-quality', 'grok-image-quality')", CHAT_JS)

    def test_official_aspect_ratios_are_available(self):
        for ratio in (
            "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
            "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20",
        ):
            self.assertIn(f'<option value="{ratio}">', CHAT_HTML)

    def test_multi_image_edits_use_images_array(self):
        branch = APP_SOURCE[APP_SOURCE.index("Routing: Grok Imagine Branch"): APP_SOURCE.index("Routing: Grok Video Branch")]
        self.assertIn("for img_entry in img_inputs[:3]:", branch)
        self.assertIn('payload["images"] = image_payloads', branch)
        self.assertIn('payload["image"] = image_payloads[0]', branch)


if __name__ == "__main__":
    unittest.main()
