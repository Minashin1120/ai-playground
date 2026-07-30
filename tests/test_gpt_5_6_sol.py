import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Gpt56SolRegressionTests(unittest.TestCase):
    def test_model_is_registered_across_backend_and_user_interfaces(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        chat_source = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        setup_source = (ROOT / "templates" / "setup.html").read_text(encoding="utf-8")
        js_source = (ROOT / "static" / "js" / "chat_core.v4.8.655.js").read_text(encoding="utf-8")

        self.assertIn('"gpt-5.6-sol"', app_source)
        self.assertIn("quickStart('gpt-5.6-sol')", chat_source)
        self.assertIn('<option value="gpt-5.6-sol">GPT-5.6 Sol</option>', setup_source)
        self.assertIn('{ id: "gpt-5.6-sol", name: "GPT-5.6 Sol"', js_source)

    def test_all_official_reasoning_efforts_are_available(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        chat_source = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        js_source = (ROOT / "static" / "js" / "chat_core.v4.8.655.js").read_text(encoding="utf-8")

        self.assertIn(
            'VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}',
            app_source,
        )
        self.assertEqual(chat_source.count('<option value="max">Max</option>'), 2)
        self.assertIn("opt.value === 'max'", js_source)
        self.assertIn("!modelLower.includes('multi-agent') && !isGpt56Sol", js_source)

    def test_versioned_assets_and_release_metadata_match(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("'2026-07-31-006'", app_source)
        self.assertIn("'V4.8.655'", app_source)
        for relative_path in (
            "static/js/chat_core.v4.8.655.js",
            "static/css/chat.custom.v4.8.655.css",
            "static/css/chat.tailwind.v4.8.655.css",
            "static/changelogs/20260731_v4.8.655.md",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
