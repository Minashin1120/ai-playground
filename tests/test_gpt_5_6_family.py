import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Gpt56FamilyRegressionTests(unittest.TestCase):
    def test_family_is_registered_across_backend_and_user_interfaces(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        chat_source = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        setup_source = (ROOT / "templates" / "setup.html").read_text(encoding="utf-8")
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        for model_id, display_name in (
            ("gpt-5.6-sol", "GPT-5.6 Sol"),
            ("gpt-5.6-terra", "GPT-5.6 Terra"),
            ("gpt-5.6-luna", "GPT-5.6 Luna"),
        ):
            self.assertIn(f'"{model_id}"', app_source)
            self.assertIn(f"quickStart('{model_id}')", chat_source)
            self.assertIn(f'<option value="{model_id}">{display_name}</option>', setup_source)
            self.assertIn(f'{{ id: "{model_id}", name: "{display_name}"', js_source)

    def test_all_official_reasoning_efforts_are_available(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        chat_source = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        js_assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")

        self.assertIn(
            'VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}',
            app_source,
        )
        self.assertEqual(chat_source.count('<option value="max">Max</option>'), 2)
        self.assertIn("opt.value === 'max'", js_source)
        self.assertIn("modelLower.startsWith('gpt-5.6-')", js_source)
        self.assertIn("!modelLower.includes('multi-agent') && !isGpt56Model", js_source)

    def test_versioned_assets_and_release_metadata_match(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("'2026-07-31-009'", app_source)
        self.assertIn("'V4.8.658'", app_source)
        for relative_path in (
            "static/js/chat_core.v4.8.658.js",
            "static/css/chat.custom.v4.8.658.css",
            "static/css/chat.tailwind.v4.8.658.css",
            "static/changelogs/20260731_v4.8.658.md",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
