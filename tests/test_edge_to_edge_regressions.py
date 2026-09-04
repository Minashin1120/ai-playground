import json
from pathlib import Path
import unittest

from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class EdgeToEdgeRegressionTests(unittest.TestCase):
    def test_chat_always_opts_in_to_edge_to_edge(self):
        template = read_chat_markup()
        # Edge-to-edge (viewport-fit=cover) is applied in every prompt-bar mode,
        # not just the minimal one.
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">',
            template,
        )
        self.assertNotIn("maximum-scale=1.0", template)
        self.assertNotIn("user-scalable=no", template)

    def test_dead_safe_pb_mechanism_removed(self):
        template = read_chat_markup()
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

    def test_minimal_mode_dock_floats_over_conversation(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        minimal = css[css.index("Edge-to-edge") :]
        self.assertIn("body.minimal-prompt-mode .composer-dock {", minimal)
        self.assertIn("position: absolute;", minimal)
        self.assertIn("bottom: 0;", minimal)
        self.assertIn("z-index: 30;", minimal)
        self.assertIn("--composer-h", minimal)
        self.assertIn("body.minimal-prompt-mode #chat-container", minimal)
        self.assertIn("body.minimal-prompt-mode .chat-scroll-to-bottom", minimal)

    def test_safe_area_rules_are_mode_independent(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        # The navigation-bar / status-bar safe-area handling is global, not tied
        # to the minimal prompt mode.
        self.assertIn("#alpha-bar", css)
        self.assertIn("bottom: env(safe-area-inset-bottom, 0px) !important;", css)
        self.assertIn(".main-chrome-header", css)
        self.assertIn("env(safe-area-inset-top, 0px)", css)
        self.assertIn(".sidebar-footer", css)
        self.assertIn("env(safe-area-inset-bottom, 0px)", css)

    def test_no_viewport_meta_toggling_in_js(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("function applyMinimalPromptMode", script)
        # viewport-fit=cover is always present in chat.html, so the JS must not
        # try to add/remove it when switching prompt-bar modes.
        self.assertNotIn("viewport-fit=cover", script)

    def test_no_theme_color_meta(self):
        pwa_meta = (APP_ROOT / "templates" / "pwa_meta.html").read_text(encoding="utf-8")
        self.assertNotIn("theme-color", pwa_meta)
        self.assertIn('<meta name="color-scheme" content="dark">', pwa_meta)

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

    def test_pwa_manifest_has_no_theme_color(self):
        manifest = json.loads((APP_ROOT / "static" / "manifest.webmanifest").read_text(encoding="utf-8"))
        # No theme_color key: the navigation bar must not be painted a solid
        # color in standalone PWA mode; the splash keeps the dark background.
        self.assertNotIn("theme_color", manifest)
        self.assertEqual(manifest["background_color"], "#05070f")

    def test_offline_page_keeps_safe_area_clearance(self):
        offline = (APP_ROOT / "static" / "offline.html").read_text(encoding="utf-8")
        self.assertIn("100dvh", offline)
        self.assertIn("env(safe-area-inset-bottom, 0px)", offline)
        self.assertIn("env(safe-area-inset-top, 0px)", offline)


if __name__ == "__main__":
    unittest.main()
