import re
import unittest
from pathlib import Path


from tests.app_source import read_app_source
ROOT = Path(__file__).resolve().parents[1]


class DeepSeekV41FlashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = read_app_source()
        js_assets = list((ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        assert len(js_assets) == 1, "Only the latest versioned chat core asset should remain"
        cls.js_source = js_assets[0].read_text(encoding="utf-8")
        cls.setup_source = (ROOT / "templates/setup.html").read_text(encoding="utf-8")
        m = re.search(r"SYSTEM_VERSION'\]\s*=\s*'(V4\.8\.\d+)'", cls.app_source)
        cls.system_version = m.group(1) if m else ""
        cls.version_slug = cls.system_version.lower() if cls.system_version else ""
        m2 = re.search(r"APP_VERSION',\s*'([^']+)'", cls.app_source)
        cls.app_version = m2.group(1) if m2 else ""

    def test_v41_flash_is_visible_and_legacy_flash_is_deprecated(self):
        self.assertRegex(
            self.js_source,
            re.compile(
                r'id:\s*"deepseek-v4\.1-flash"[^}]*apiId:\s*"deepseek-flash"[^}]*name:\s*"DeepSeek V4\.1 Flash"'
            ),
        )
        self.assertRegex(
            self.js_source,
            re.compile(
                r'id:\s*"deepseek-v4-flash-0731"[^}]*name:\s*"DeepSeek V4 Flash"[^}]*deprecated:\s*true'
            ),
        )
        self.assertIn(
            '<option value="deepseek-v4.1-flash">DeepSeek V4.1 Flash</option>',
            self.setup_source,
        )
        self.assertNotIn(
            '<option value="deepseek-v4-flash-0731">DeepSeek V4 Flash</option>',
            self.setup_source,
        )

    def test_official_alias_and_specs_are_applied(self):
        self.assertIn('if mk.lower() == "deepseek-v4.1-flash":', self.app_source)
        self.assertIn('return "deepseek-flash"', self.app_source)
        self.assertIn('"model": _deepseek_api_model_id(model_key)', self.app_source)
        self.assertIn('legacy_value = key_map.get("deepseek-v4-flash")', self.app_source)
        self.assertIn('1M context, 384K output', self.js_source)
        self.assertIn('In $0.003 hit/$0.15 miss, Out $0.60 off-peak', self.js_source)
        self.assertIn('In $0.022 hit/$0.66 miss, Out $1.98 off-peak', self.js_source)

    def test_flash_effort_and_user_isolation_follow_official_api(self):
        self.assertIn('if raw in ("low", "high", "max"):', self.app_source)
        self.assertIn('deepseek_kwargs["extra_body"]["user_id"] = f"app_user_{user_id}"', self.app_source)
        self.assertIn('"stream_options": {"include_usage": True}', self.app_source)
        self.assertIn("const isDeepSeekFlash = modelLower === 'deepseek-v4.1-flash'", self.js_source)
        self.assertIn("!isDeepSeekFlash && !isDeepSeekPro", self.js_source)
        self.assertIn("isLlmModel() && !isDeepSeek", self.js_source)

    def test_none_effort_forces_deepseek_non_thinking_mode(self):
        self.assertIn('if is_deepseek and req_reasoning_effort == "none":', self.app_source)
        self.assertIn('req_reasoning_effort != "none"', self.app_source)
        self.assertIn('extra_body"] = {"thinking": {"type": "disabled"}}', self.app_source)
        self.assertIn('const isDeepSeekNonThinking =', self.js_source)
        self.assertIn('enable_thinking: isDeepSeekNonThinking ? false : get(\'enable-thinking\').checked', self.js_source)

    def test_pro_effort_tool_calls_and_multiturn_context_follow_official_api(self):
        self.assertIn('if raw in ("max", "xhigh"):', self.app_source)
        self.assertIn('assistant_tool_message["reasoning_content"] = round_reasoning', self.app_source)
        self.assertIn('and isinstance(saved_tool_context, list):', self.app_source)
        self.assertIn('for saved_message in saved_tool_context:', self.app_source)
        self.assertIn('deepseek_kwargs["tools"] = python_tools', self.app_source)

    def test_release_assets_and_versions_are_complete(self):
        self.assertTrue(self.app_version)
        self.assertTrue(self.system_version)
        self.assertIn(f"'{self.app_version}'", self.app_source)
        self.assertIn(f"'{self.system_version}'", self.app_source)
        for relative in (
            f"static/js/chat_core.{self.version_slug}.js",
            f"static/css/chat.custom.{self.version_slug}.css",
            f"static/css/chat.tailwind.{self.version_slug}.css",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
