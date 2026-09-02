import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _current_asset(kind, pattern):
    matches = sorted((ROOT / "static" / kind).glob(pattern))
    if not matches:
        raise AssertionError(f"No asset matched static/{kind}/{pattern}")
    return matches[-1].read_text(encoding="utf-8")


class UiDesignRefreshRegressionTests(unittest.TestCase):
    def test_design_tokens_are_synced(self):
        custom = _current_asset("css", "chat.custom.v4.8.*.css")
        shared = (ROOT / "static" / "css" / "app-design.css").read_text(encoding="utf-8")
        landing = (ROOT / "static" / "css" / "landing.css").read_text(encoding="utf-8")

        for source in (custom, shared):
            self.assertIn("--bg-1: #05070f;", source)
            self.assertIn("--panel-chrome: rgba(8, 12, 24, 0.78);", source)
            self.assertIn("--radius-xl: 26px;", source)
            self.assertIn("radial-gradient(rgba(226, 232, 240,", source)

        self.assertIn("V4.8.810 — modern visual refresh", custom)
        self.assertIn("border-radius: 999px !important;", custom)
        self.assertIn(".ld-hero-title-accent", landing)
        self.assertIn("position: sticky;", landing)
        self.assertIn("border-radius: 999px;", landing)

    def test_public_pages_use_refreshed_surfaces(self):
        landing = (ROOT / "templates" / "landing.html").read_text(encoding="utf-8")
        login = (ROOT / "templates" / "login.html").read_text(encoding="utf-8")
        chat = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

        self.assertIn("ld-hero-cta-primary", landing)
        self.assertIn("ld-hero-title-accent", landing)
        self.assertIn("radial-gradient(rgba(226, 232, 240, 0.04)", login)
        self.assertIn("welcome-subtitle", chat)
        self.assertIn("composer-input-shell", chat)
        self.assertIn("sidebar-toolbar", chat)
        self.assertIn("flex-nowrap", chat)
        self.assertNotIn('sidebar-toolbar flex items-center justify-start gap-0.5 flex-wrap', chat)

    def test_sidebar_toolbar_stays_on_one_row(self):
        custom = _current_asset("css", "chat.custom.v4.8.*.css")
        chat = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn("flex-wrap: nowrap !important;", custom)
        self.assertIn("#sidebar #lib-btn.sidebar-icon-btn", custom)
        self.assertIn("id=\"lib-btn\"", chat)

    def test_mobile_toolbar_icon_buttons_keep_flex_centering(self):
        # On mobile, forcing display:block on every .hide-compact item sank the
        # icon inside the settings/GitHub/etc. toolbar buttons to the button
        # bottom, so the flex-centered library button looked raised. The
        # toolbar icon buttons must keep display:inline-flex on small screens.
        custom = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("#sidebar .sidebar-toolbar .sidebar-icon-btn.hide-compact", custom)
        self.assertIn("display: inline-flex !important;", custom)
