from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class EdgeToEdgeRegressionTests(unittest.TestCase):
    def test_chat_viewport_opts_in_only_for_minimal_prompt_mode(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1.0'
            '{% if current_user.is_authenticated and current_user.minimal_prompt_mode %}, viewport-fit=cover{% endif %}">',
            template,
        )
        # The legacy always-on edge-to-edge flags are gone.
        self.assertNotIn("maximum-scale=1.0", template)
        self.assertNotIn("user-scalable=no", template)

    def test_dead_safe_pb_mechanism_removed(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertNotIn("safe-pb", template)
        self.assertNotIn("safe-pb", css)
        self.assertNotIn("--safe-area-inset-bottom", css)

    def test_minimal_mode_dock_is_transparent_for_edge_to_edge(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("body.minimal-prompt-mode .composer-dock", css)
        self.assertIn("background: transparent !important;", css)
        self.assertIn("border-top-color: transparent !important;", css)
        self.assertIn("box-shadow: none !important;", css)
        # Normal/compact modes keep the traditional solid dock.
        self.assertIn("body.minimal-prompt-mode #top-model-bar", css)

    def test_js_toggles_viewport_fit_for_minimal_mode(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("viewport-fit=cover", script)
        self.assertIn("function applyMinimalPromptMode", script)

    def test_all_pages_have_viewport_fit_cover(self):
        for name in (
            "403.html",
            "404.html",
            "banned.html",
            "changelog.html",
            "chat.html",
            "help.html",
            "landing.html",
            "login.html",
            "maintenance.html",
            "setup.html",
            "signup.html",
            "verify_2fa.html",
        ):
            template = (APP_ROOT / "templates" / name).read_text(encoding="utf-8")
            self.assertIn("viewport-fit=cover", template, msg=f"{name} should opt in to edge-to-edge")
        offline = (APP_ROOT / "static" / "offline.html").read_text(encoding="utf-8")
        self.assertIn("viewport-fit=cover", offline)

    def test_color_scheme_unified_on_all_pwa_pages(self):
        pwa_meta = (APP_ROOT / "templates" / "pwa_meta.html").read_text(encoding="utf-8")
        self.assertIn('<meta name="color-scheme" content="dark">', pwa_meta)

    def test_offline_page_keeps_safe_area_clearance(self):
        offline = (APP_ROOT / "static" / "offline.html").read_text(encoding="utf-8")
        self.assertIn("100dvh", offline)
        self.assertIn("env(safe-area-inset-bottom, 0px)", offline)
        self.assertIn("env(safe-area-inset-top, 0px)", offline)


if __name__ == "__main__":
    unittest.main()
