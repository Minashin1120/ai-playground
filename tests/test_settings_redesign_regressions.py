from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class SettingsRedesignRegressionTests(unittest.TestCase):
    def test_template_has_modern_settings_modal_structure(self):
        template = Path(APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        start = template.index('id="settings-modal"')
        end = template.index('id="gemini-local-python-modal"')
        modal = template[start:end]

        for ident in (
            'id="settings-modal"',
            'id="settings-search"',
            'id="settings-search-clear"',
            'id="settings-tabs-wrap"',
            'id="settings-tabs"',
            'id="settings-tabs-arrow-left"',
            'id="settings-tabs-arrow-right"',
            'id="close-settings-btn"',
            'id="save-settings-btn"',
            'id="settings-header-close"',
            'id="set-enter-to-send"',
            'id="set-default-model"',
        ):
            self.assertIn(ident, modal, f"Missing settings element: {ident}")

        for cls in (
            "settings-modal-root",
            "settings-modal-panel",
            "settings-title-icon",
            "settings-modal-title",
            "settings-search-box",
            "settings-main",
            "settings-tab is-active",
            "settings-content",
            "settings-tab-panel",
            "settings-card",
            "settings-btn-primary",
        ):
            self.assertIn(cls, modal, f"Missing modern settings class: {cls}")

        self.assertIn("clickTab('general')", modal)
        self.assertIn('<i class="fas fa-sliders-h"></i><span>一般</span>', modal)
        self.assertNotIn('bg-gray-800 rounded-lg w-full max-w-2xl p-6', modal)
        self.assertNotIn('class="bg-gray-900 p-4 rounded border border-gray-700"', modal)

    def test_js_uses_active_tab_class_and_modern_cards(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("btn.classList.add('is-active')", script)
        self.assertIn("btn.classList.remove('is-active')", script)
        self.assertNotIn("btn.classList.add('text-blue-400','border-blue-400','font-bold')", script)
        self.assertIn("card.className = 'settings-card'", script)
        self.assertIn("settings-empty-state", script)
        self.assertIn("settings-header-close", script)
        self.assertIn("className = 'settings-search-count'", script)

    def test_css_has_settings_layout_and_cards(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("#settings-modal .settings-modal-panel", css)
        self.assertIn("#settings-modal .settings-tab.is-active", css)
        self.assertIn("#settings-modal .settings-card", css)
        self.assertIn("#settings-modal .settings-main", css)
        self.assertIn("flex-direction: column", css)
        self.assertIn(".settings-tabs-wrap.can-scroll-left.is-edge-left .settings-tabs-arrow-left", css)
        self.assertIn("overscroll-behavior-x: contain", css)

    def test_settings_body_can_scroll_inside_fixed_panel(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        template = Path(APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        start = template.index('id="settings-modal"')
        overlay = template[start: template.index(">", start) + 1]
        self.assertNotIn("overflow-y-auto", overlay)
        self.assertIn('class="settings-content overflow-y-auto"', template)
        self.assertIn("height: min(90dvh, 880px)", css)
        content_css = css[css.index("#settings-modal .settings-content {") :]
        content_css = content_css[: content_css.index("#settings-modal .settings-tab-panel {")]
        self.assertIn("overflow-y: auto", content_css)
        self.assertIn("min-height: 0", content_css)
        self.assertIn("overscroll-behavior: contain", content_css)
        self.assertIn("#settings-modal input[type=\"text\"]", css)
        self.assertIn("#settings-modal .settings-card button", css)


if __name__ == "__main__":
    unittest.main()
