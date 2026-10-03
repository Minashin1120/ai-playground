import json
import types
import unittest
from pathlib import Path

import httpx

from tests.app_source import read_app_source
from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = read_chat_markup()
SETUP_HTML = (APP_ROOT / "templates" / "setup.html").read_text(encoding="utf-8")
MODEL_IDS = ("ideogram-4.5", "ideogram-4.0", "ideogram-3.0", "ideogram-2a", "ideogram-2.0")


def _load_ideogram():
    """Run server/ideogram.py in its own namespace, as app.py does with every server part."""
    namespace = {"json": json, "httpx": httpx}
    source = (APP_ROOT / "server" / "ideogram.py").read_text(encoding="utf-8")
    exec(compile(source, "server/ideogram.py", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


IDEOGRAM = _load_ideogram()
PNG = ("a.png", b"\x89PNG-data", "image/png")


class IdeogramRegistrationTests(unittest.TestCase):
    def test_models_are_registered_in_backend_and_ui(self):
        valid_ids = APP_SOURCE[APP_SOURCE.index("ALL_VALID_MODEL_IDS"): APP_SOURCE.index("def is_sts_model")]
        for model_id in MODEL_IDS:
            self.assertIn(f'"{model_id}"', valid_ids)
            self.assertIn(f'id: "{model_id}"', CHAT_JS)
            start = CHAT_JS.index(f'id: "{model_id}"')
            self.assertIn('implementedAt: "2026-10-03"', CHAT_JS[start:start + 160])
            self.assertIn(f'<option value="{model_id}"', SETUP_HTML)

    def test_provider_key_is_wired_end_to_end(self):
        self.assertIn("ideogram_api_key = db.Column", APP_SOURCE)
        self.assertIn("def ensure_user_ideogram_api_key_column", APP_SOURCE)
        self.assertIn("ALTER TABLE user ADD COLUMN ideogram_api_key TEXT", APP_SOURCE)
        self.assertIn('user_or_admin_env("ideogram_api_key", "IDEOGRAM_API_KEY")', APP_SOURCE)
        self.assertIn("'ideogram_key': _masked_secret(current_user.ideogram_api_key)", APP_SOURCE)
        self.assertIn("current_user.ideogram_api_key = encrypt_val(d['ideogram_key'])", APP_SOURCE)
        self.assertIn('"ideogram_api_key"', APP_SOURCE[APP_SOURCE.index("ACCOUNT_SECRET_FIELDS"):][:400])
        self.assertIn("('ideogram_key', 'ideogram_api_key')", APP_SOURCE)
        self.assertIn("keyField: 'ideogram_key'", CHAT_JS)
        self.assertIn('id="set-ideogram"', CHAT_HTML)
        self.assertIn('name="ideogram_key"', SETUP_HTML)
        self.assertIn("b.ideogram_key = get('set-ideogram').value", CHAT_JS)

    def test_server_part_is_loaded_and_documented(self):
        parts = APP_SOURCE[APP_SOURCE.index("_SERVER_PARTS"):][:1200]
        self.assertIn('"ideogram.py"', parts)
        self.assertIn("`ideogram.py`", (APP_ROOT / "server" / "README.md").read_text(encoding="utf-8"))

    def test_generation_branch_and_options_are_forwarded(self):
        self.assertIn("elif is_ideogram:", APP_SOURCE)
        self.assertIn("ideogram_request(", APP_SOURCE)
        for field in ("aspect", "resolution", "quality", "speed", "magic_prompt", "style_type", "negative_prompt", "count", "seed"):
            self.assertIn(f"'ideogram_{field}': data.get('ideogram_{field}')", APP_SOURCE)
            self.assertIn(f"ideogram_{field}: isIdeogramModel()", CHAT_JS)
        for element in ("ideogram-image-options", "modal-ideogram-image-options", "ideogram-image-aspect", "modal-ideogram-image-aspect"):
            self.assertIn(f'id="{element}"', CHAT_HTML)

    def test_provider_classification(self):
        source = (APP_ROOT / "server" / "providers.py").read_text(encoding="utf-8")
        start = source.index("def get_model_api_provider")
        self.assertIn('mk.startswith("ideogram-")', source[start:start + 900])


class IdeogramRequestTests(unittest.TestCase):
    def build(self, model, options=None, images=None, prompt="a cat"):
        return IDEOGRAM.ideogram_build_request(model, prompt, options or {}, images)

    @staticmethod
    def form(kwargs):
        return {key: value[1] for key, value in kwargs["files"] if value[0] is None}

    def test_ideogram_45_generate_uses_multipart_size_and_quality(self):
        url, kwargs, count, edit = self.build("ideogram-4.5", {
            "ideogram_aspect": "16:9", "ideogram_resolution": "2k", "ideogram_quality": "high",
            "ideogram_count": "2", "ideogram_seed": "7", "ideogram_magic_prompt": "off"})
        self.assertEqual("https://api.ideogram.ai/v2/image/generate/ideogram-4-5", url)
        self.assertEqual((2, False), (count, edit))
        self.assertEqual({"prompt": "a cat", "num_images": "2", "seed": "7", "magic_prompt": "off",
                          "quality": "high", "size": "2560x1440"}, self.form(kwargs))

    def test_ideogram_45_very_low_requires_source_images(self):
        _, kwargs, _, _ = self.build("ideogram-4.5", {"ideogram_quality": "very_low"})
        self.assertNotIn("quality", self.form(kwargs))
        _, kwargs, _, _ = self.build("ideogram-4.5", {"ideogram_quality": "very_low", "ideogram_aspect": "1:1"}, [PNG])
        self.assertEqual("very_low", kwargs["data"]["quality"])

    def test_ideogram_45_edit_sends_image_references_and_no_size(self):
        url, kwargs, _, edit = self.build("ideogram-4.5", {"ideogram_aspect": "16:9"}, [PNG] + [PNG] * 6)
        self.assertEqual("https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5", url)
        self.assertTrue(edit)
        names = [name for name, _ in kwargs["files"]]
        self.assertEqual(["image"] + ["reference_images"] * 4, names)
        self.assertNotIn("size", kwargs["data"])

    def test_only_ideogram_45_accepts_images(self):
        for model in MODEL_IDS[1:]:
            with self.assertRaises(ValueError):
                self.build(model, images=[PNG])

    def test_ideogram_4_uses_exact_resolutions_as_json(self):
        url, kwargs, count, _ = self.build("ideogram-4.0", {
            "ideogram_aspect": "9:16", "ideogram_resolution": "1k", "ideogram_speed": "turbo", "ideogram_count": "99"})
        self.assertEqual("https://api.ideogram.ai/v2/image/generate/ideogram-4", url)
        self.assertEqual(8, count)
        self.assertEqual("720x1280", kwargs["json"]["resolution"])
        self.assertEqual("turbo", kwargs["json"]["rendering_speed"])
        _, auto_kwargs, _, _ = self.build("ideogram-4.0", {"ideogram_aspect": "auto"})
        self.assertNotIn("resolution", auto_kwargs["json"])

    def test_every_size_preset_matches_the_documented_resolutions(self):
        documented = {
            "1024x1024", "896x1120", "1120x896", "864x1152", "1152x864", "832x1248", "1248x832", "720x1280", "1280x720",
            "800x1280", "1280x800", "720x1440", "1440x720", "512x1536", "1536x512", "2048x2048", "1792x2240", "2240x1792",
            "1728x2304", "2304x1728", "1664x2496", "2496x1664", "1440x2560", "2560x1440", "1600x2560", "2560x1600",
            "1440x2880", "2880x1440", "1024x3072", "3072x1024",
        }
        presets = {size for pair in IDEOGRAM.IDEOGRAM_SIZE_PRESETS.values() for size in pair}
        self.assertEqual(documented, presets)

    def test_legacy_models_use_x_separated_aspect_ratio(self):
        url, kwargs, _, _ = self.build("ideogram-3.0", {
            "ideogram_aspect": "4:3", "ideogram_style_type": "realistic", "ideogram_negative_prompt": "blur", "ideogram_speed": "quality"})
        self.assertEqual("https://api.ideogram.ai/v2/image/generate/ideogram-3", url)
        self.assertEqual({"prompt": "a cat", "aspect_ratio": "4x3", "num_images": "1", "rendering_speed": "quality",
                          "style_type": "realistic", "negative_prompt": "blur"}, self.form(kwargs))
        _, v2, _, _ = self.build("ideogram-2.0", {"ideogram_aspect": "auto", "ideogram_style_type": "anime", "ideogram_negative_prompt": "blur"})
        self.assertEqual("auto", v2["json"]["aspect_ratio"])
        self.assertEqual("anime", v2["json"]["style_type"])
        self.assertEqual("blur", v2["json"]["negative_prompt"])
        _, v2a, _, _ = self.build("ideogram-2a", {"ideogram_negative_prompt": "blur", "ideogram_style_type": "fiction"})
        self.assertNotIn("negative_prompt", v2a["json"])
        self.assertNotIn("style_type", v2a["json"])
        _, v3_invalid, _, _ = self.build("ideogram-3.0", {"ideogram_style_type": "anime"})
        self.assertNotIn("style_type", self.form(v3_invalid))

    def test_request_sends_api_key_and_returns_items(self):
        captured = {}

        def fake_post(url, headers=None, timeout=None, **kwargs):
            captured.update(url=url, headers=headers, timeout=timeout)
            return httpx.Response(200, json={"data": [{"url": "https://ideogram.ai/x.png", "is_image_safe": True}]})

        original = httpx.post
        httpx.post = fake_post
        try:
            items, edit = IDEOGRAM.ideogram_request("key-1", "ideogram-4.0", "a cat", {}, None)
        finally:
            httpx.post = original
        self.assertEqual("key-1", captured["headers"]["Api-Key"])
        self.assertEqual("https://ideogram.ai/x.png", items[0]["url"])
        self.assertFalse(edit)

    def test_error_responses_are_translated(self):
        def fail(status, body):
            def fake_post(url, headers=None, timeout=None, **kwargs):
                return httpx.Response(status, text=body)
            original = httpx.post
            httpx.post = fake_post
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    IDEOGRAM.ideogram_request("k", "ideogram-4.0", "p", {}, None)
            finally:
                httpx.post = original
            return str(ctx.exception)

        self.assertIn("APIキー", fail(401, "{}"))
        self.assertIn("安全性", fail(422, "{}"))
        self.assertIn("insufficient_funds", fail(402, json.dumps({"error": "no credit", "reject_reason": "insufficient_funds"})))
        self.assertIn("boom", fail(500, json.dumps({"error": "boom"})))


if __name__ == "__main__":
    unittest.main()
