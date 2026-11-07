from pathlib import Path
import re
import unittest

from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]

# The original reference modals keep their dedicated class families. Their
# surfaces already match the shared modal language documented in
# DESIGN_MODALS.md.
REFERENCE_MODAL_IDS = {
    "settings-modal",
    "model-modal",
    "lib-modal",
    "batch-modal",
}


def _latest_css():
    assets = sorted((APP_ROOT / "static" / "css").glob("chat.custom.v4.8.*.css"))
    assert len(assets) == 1, "Expected exactly one editable chat.custom source"
    return assets[0].read_text(encoding="utf-8")


class ModalDesignSystemRegressionTests(unittest.TestCase):
    def _opening_tags(self, markup, modal_id):
        match = re.search(rf'<div[^>]*id="{re.escape(modal_id)}"[^>]*>', markup)
        self.assertIsNotNone(match, f"{modal_id} opening tag not found")
        return match.group(0)

    def test_every_chat_modal_uses_the_shared_overlay_and_panel_language(self):
        markup = read_chat_markup()
        modal_ids = sorted(set(re.findall(r'id="([a-z0-9-]+-modal)"', markup)))
        self.assertTrue(modal_ids)

        for modal_id in modal_ids:
            tag = self._opening_tags(markup, modal_id)
            self.assertIn("modal-overlay", tag, modal_id)
            if modal_id in REFERENCE_MODAL_IDS:
                continue
            self.assertIn("app-modal-overlay", tag, modal_id)

        panels = re.findall(r'<div class="([^"]*app-modal-panel[^"]*)"', markup)
        self.assertTrue(panels)
        for classes in panels:
            self.assertIn("modal-panel", classes)
            self.assertRegex(classes, r"app-modal-panel--(sm|md|lg|xl|full|tall)")

    def test_every_new_chat_modal_has_a_header_and_scrollable_body(self):
        markup = read_chat_markup()
        # Parse each overlay as a fragment so nested panels cannot bleed into
        # the next modal. Split on the unique modal opening tags.
        starts = [m.start() for m in re.finditer(r'<div[^>]*id="[a-z0-9-]+-modal"[^>]*>', markup)]
        starts.append(len(markup))
        for start, end in zip(starts, starts[1:]):
            fragment = markup[start:end]
            modal_id = re.search(r'id="([a-z0-9-]+-modal)"', fragment).group(1)
            if modal_id in REFERENCE_MODAL_IDS:
                continue
            self.assertIn("app-modal-panel", fragment, modal_id)
            self.assertIn("app-modal-header", fragment, modal_id)
            self.assertIn("app-modal-title", fragment, modal_id)
            self.assertIn("app-modal-close", fragment, modal_id)
            self.assertIn("app-modal-body", fragment, modal_id)

    def test_public_page_modals_use_the_shared_language(self):
        for name in ("landing.html", "login.html", "help.html", "setup.html"):
            markup = (APP_ROOT / "templates" / name).read_text(encoding="utf-8")
            for modal_id in set(re.findall(r'id="([a-z0-9-]+-modal)"', markup)):
                tag = self._opening_tags(markup, modal_id)
                self.assertIn("modal-overlay", tag, f"{name}:{modal_id}")
                self.assertIn("app-modal-overlay", tag, f"{name}:{modal_id}")
                self.assertIn("app-modal-panel", markup, f"{name}:{modal_id}")

    def test_shared_modal_css_exists_in_chat_and_public_stylesheets(self):
        chat_css = _latest_css()
        public_css = (APP_ROOT / "static" / "css" / "app-design.css").read_text(encoding="utf-8")
        for css in (chat_css, public_css):
            self.assertIn("Shared modal design", css)
            for selector in (
                ".app-modal-overlay",
                ".app-modal-panel",
                ".app-modal-panel--sm",
                ".app-modal-panel--full",
                ".app-modal-header",
                ".app-modal-title-icon",
                ".app-modal-close",
                ".app-modal-body",
                ".app-modal-footer",
                ".app-modal-btn--primary",
            ):
                self.assertIn(selector, css)

        # Settings is the visual reference: the shared panel must keep the
        # same 22px radius and navy gradient.
        panel = chat_css[
            chat_css.index(".app-modal-panel {") : chat_css.index(".app-modal-panel.app-modal-panel--sm")
        ]
        self.assertIn("border-radius: 22px", panel)
        self.assertIn("rgba(10, 14, 28, 0.98)", panel)
        self.assertIn("rgba(6, 9, 18, 0.99)", panel)

    def test_light_theme_overrides_the_shared_modal_surfaces(self):
        for name in ("theme-light.css", "theme-light-manual.css"):
            css = (APP_ROOT / "static" / "css" / name).read_text(encoding="utf-8")
            self.assertIn(".app-modal-panel", css, name)
            self.assertIn(".app-modal-header", css, name)
            self.assertIn(".app-modal-footer", css, name)
            self.assertIn("linear-gradient(180deg, #ffffff, #f7f9fc)", css, name)

    def test_old_one_off_panel_markup_is_not_reintroduced(self):
        markup = read_chat_markup()
        # Old shape: dark utility panel + generic modal-panel without the shared
        # app-modal-panel class.
        offenders = re.findall(
            r'<div class="[^"]*bg-(?:gray|slate)-800[^"]*modal-panel(?![^"]*app-modal-panel)[^"]*"',
            markup,
        )
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
