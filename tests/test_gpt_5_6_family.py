import pathlib
import re
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

        self.assertIn('id="welcome-quick-start"', chat_source)
        self.assertIn("getRecentModelsForQuickStart", js_source)
        self.assertIn("renderWelcomeQuickStart", js_source)
        self.assertIn("implementedAt:", js_source)
        for model_id, display_name in (
            ("gpt-5.6-sol", "GPT-5.6 Sol"),
            ("gpt-5.6-terra", "GPT-5.6 Terra"),
            ("gpt-5.6-luna", "GPT-5.6 Luna"),
        ):
            self.assertIn(f'"{model_id}"', app_source)
            self.assertIn(f'<option value="{model_id}">{display_name}</option>', setup_source)
            self.assertIn(f'id: "{model_id}"', js_source)
            self.assertIn(f'name: "{display_name}"', js_source)
            # Welcome quick-start is rendered from MODELS by implementedAt order (not hard-coded in HTML).
            self.assertRegex(
                js_source,
                rf'id:\s*"{re.escape(model_id)}"[^}}]*implementedAt:\s*"\d{{4}}-\d{{2}}-\d{{2}}"',
            )

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
        m = re.search(r"SYSTEM_VERSION'\]\s*=\s*'(V4\.8\.\d+)'", app_source)
        self.assertIsNotNone(m)
        system_version = m.group(1)
        version_slug = system_version.lower()
        m2 = re.search(r"APP_VERSION',\s*'([^']+)'", app_source)
        self.assertIsNotNone(m2)
        app_version = m2.group(1)

        self.assertIn(f"'{app_version}'", app_source)
        self.assertIn(f"'{system_version}'", app_source)
        for relative_path in (
            f"static/js/chat_core.{version_slug}.js",
            f"static/css/chat.custom.{version_slug}.css",
            f"static/css/chat.tailwind.{version_slug}.css",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)
        changelog_matches = list((ROOT / "static" / "changelogs").glob(f"*_{version_slug}.md"))
        self.assertTrue(changelog_matches, f"changelog for {version_slug}")


if __name__ == "__main__":
    unittest.main()
