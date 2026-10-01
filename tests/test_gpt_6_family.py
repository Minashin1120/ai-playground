import pathlib
import re
import unittest

from tests.app_source import read_app_source
ROOT = pathlib.Path(__file__).resolve().parents[1]


class Gpt6FamilyRegressionTests(unittest.TestCase):
    def test_family_is_registered_across_backend_and_user_interfaces(self):
        app_source = read_app_source()
        setup_source = (ROOT / "templates" / "setup.html").read_text(encoding="utf-8")
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        for model_id, display_name in (
            ("gpt-6-astra", "GPT-6 Astra"),
            ("gpt-6.1-sol", "GPT-6.1 Sol"),
            ("gpt-6-luna", "GPT-6 Luna"),
            ("gpt-6-sol", "GPT-6 Sol"),
        ):
            self.assertIn(f'"{model_id}"', app_source)
            self.assertIn(f'<option value="{model_id}">{display_name}</option>', setup_source)
            self.assertIn(f'id: "{model_id}"', js_source)
            self.assertIn(f'name: "{display_name}"', js_source)
            self.assertRegex(
                js_source,
                rf'id:\s*"{re.escape(model_id)}"[^}}]*implementedAt:\s*"\d{{4}}-\d{{2}}-\d{{2}}"',
            )

    def test_reasoning_efforts_match_the_official_sets(self):
        app_source = read_app_source()
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        self.assertIn("const isGpt6Model = modelLower.startsWith('gpt-6');", js_source)
        self.assertIn("const gpt6AllowsNone = modelLower === 'gpt-6-sol' || modelLower === 'gpt-6-luna';", js_source)
        self.assertIn("!isGpt56Model && !isGpt6Model && !isDeepSeekFlash", js_source)
        self.assertIn('if effort == "none" and any(x in model_key_l for x in ("gpt-6-astra", "gpt-6.1-sol")):', app_source)
        self.assertIn("'gpt-6'", app_source)


if __name__ == "__main__":
    unittest.main()
