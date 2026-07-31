import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekV4Flash0731Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.js_source = (ROOT / "static/js/chat_core.v4.8.663.js").read_text(encoding="utf-8")
        cls.setup_source = (ROOT / "templates/setup.html").read_text(encoding="utf-8")

    def test_new_release_is_visible_and_preview_is_deprecated(self):
        self.assertIn(
            'id: "deepseek-v4-flash-0731", apiId: "deepseek-v4-flash", name: "DeepSeek V4 Flash"',
            self.js_source,
        )
        self.assertRegex(
            self.js_source,
            re.compile(
                r'id: "deepseek-v4-flash", name: "DeepSeek V4 Flash Preview"[^\n]+deprecated: true'
            ),
        )
        self.assertIn(
            '<option value="deepseek-v4-flash-0731">DeepSeek V4 Flash</option>',
            self.setup_source,
        )
        self.assertNotIn(
            '<option value="deepseek-v4-flash">DeepSeek V4 Flash</option>',
            self.setup_source,
        )

    def test_official_alias_and_specs_are_applied(self):
        self.assertIn('if mk.lower() == "deepseek-v4-flash-0731":', self.app_source)
        self.assertIn('return "deepseek-v4-flash"', self.app_source)
        self.assertIn('"model": _deepseek_api_model_id(model_key)', self.app_source)
        self.assertIn('legacy_value = key_map.get("deepseek-v4-flash")', self.app_source)
        self.assertIn('1M context, up to 384K output', self.js_source)
        self.assertIn('CN¥0.02/1M (hit), CN¥1/1M (miss), Out CN¥2/1M', self.js_source)

    def test_flash_effort_and_user_isolation_follow_official_api(self):
        self.assertIn('if raw in ("low", "high", "max"):', self.app_source)
        self.assertIn('deepseek_kwargs["extra_body"]["user_id"] = f"app_user_{user_id}"', self.app_source)
        self.assertIn('"stream_options": {"include_usage": True}', self.app_source)
        self.assertIn("const isDeepSeekFlash0731 = modelLower === 'deepseek-v4-flash-0731'", self.js_source)
        self.assertIn("!isDeepSeekFlash0731 && !isDeepSeekPro", self.js_source)
        self.assertIn("isLlmModel() && !isDeepSeek", self.js_source)

    def test_release_assets_and_versions_are_complete(self):
        self.assertIn("'2026-08-01-002'", self.app_source)
        self.assertIn("'V4.8.663'", self.app_source)
        for relative in (
            "static/js/chat_core.v4.8.663.js",
            "static/css/chat.custom.v4.8.663.css",
            "static/css/chat.tailwind.v4.8.663.css",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
