import pathlib
import re
import unittest


from tests.app_source import read_app_source
ROOT = pathlib.Path(__file__).resolve().parents[1]


class Grok4546FamilyRegressionTests(unittest.TestCase):
    def test_family_is_registered_across_backend_and_user_interfaces(self):
        app_source = read_app_source()
        setup_source = (ROOT / "templates" / "setup.html").read_text(encoding="utf-8")
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        self.assertIn("getRecentModelsForQuickStart", js_source)
        self.assertIn("renderWelcomeQuickStart", js_source)
        self.assertIn("implementedAt:", js_source)
        for model_id, display_name, effort_supported in (
            ("grok-4.6", "Grok 4.6", True),
            ("grok-4.5", "Grok 4.5", False),
        ):
            self.assertIn(f'"{model_id}"', app_source)
            self.assertIn(f'<option value="{model_id}">{display_name}</option>', setup_source)
            self.assertIn(f'id: "{model_id}"', js_source)
            self.assertIn(f'name: "{display_name}"', js_source)
            self.assertRegex(
                js_source,
                rf'id:\s*"{re.escape(model_id)}"[^}}]*implementedAt:\s*"\d{{4}}-\d{{2}}-\d{{2}}"',
            )

        # Both are reasoning_effort models in the backend.
        self.assertIn('("grok-4.5" in model_key_l)', app_source)
        self.assertIn('("grok-4.6" in model_key_l)', app_source)
        self.assertIn('("grok-4.5" in model_key_l) or ("grok-4.6" in model_key_l)', app_source)

    def test_reasoning_effort_levels_follow_official_docs(self):
        app_source = read_app_source()
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        # grok-4.6 supports xhigh; grok-4.5 does not (xhigh is treated as high).
        self.assertIn('return "xhigh" if is_grok_46 else "high"', app_source)
        self.assertIn("!isGrok46 && !modelLower.includes('multi-agent')", js_source)
        # medium effort is available on both new grok models.
        self.assertIn("isGrok45 || isGrok46", js_source)
        # The reasoning effort selector is enabled for both models.
        self.assertIn("modelLower.includes('grok-4.5')", js_source)
        self.assertIn("modelLower.includes('grok-4.6')", js_source)


if __name__ == "__main__":
    unittest.main()
