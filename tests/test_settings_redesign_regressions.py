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

    def test_save_button_is_disabled_until_settings_load_completes(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        # The save button must be disabled at open and only re-enabled after the
        # async settings fetch has populated the form (prevents saving defaults).
        self.assertIn("let settingsModalLoaded = false;", script)
        self.assertIn("setSettingsSaveEnabled(false);", script)
        self.assertIn("settingsModalLoaded = true;", script)
        self.assertIn("setSettingsSaveEnabled(true);", script)
        self.assertIn("if (!settingsModalLoaded) {", script)
        # enable_e2ee must only be sent when the checkbox differs from the value
        # loaded from the server, so an unmodified form cannot start a migration.
        self.assertIn("b.enable_e2ee = e2eeCurrent;", script)
        self.assertIn("e2eeCurrent !== e2eeLoaded", script)

    def test_import_settings_confirmation_ui_is_present(self):
        template = Path(APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        for ident in (
            'id="settings-confirmation-modal"',
            'id="settings-confirmation-list"',
            'id="settings-confirmation-count"',
            'id="settings-confirmation-close"',
            'id="settings-confirmation-cancel"',
            'id="settings-confirmation-confirm"',
            'id="account-import-settings-bypass"',
        ):
            self.assertIn(ident, template)
        script = _current_asset("js", "chat_core.v4.8.*.js")
        for expected in (
            "showSettingsImportConfirmation",
            "settings_confirmation",
            "needs_settings_confirmation",
            "confirm_settings",
        ):
            self.assertIn(expected, script)

    def test_import_modals_sit_above_settings_modal_via_explicit_zindex(self):
        # The Tailwind bundle ships z-50 (settings modal) but not z-[200], so the
        # import file-selection / settings-confirmation modals must get an
        # explicit z-index from chat.custom CSS or they paint behind the settings
        # modal and are invisible while it is open.
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        block = css[css.index("#import-files-modal,") :]
        block = block[: block.index("}") + 1]
        self.assertIn("#settings-confirmation-modal", block)
        self.assertIn("z-index: 200", block)
        self.assertIn("position: fixed", block)
        # The static Tailwind bundle must not silently drop the z-index either.
        tailwind = _current_asset("css", "chat.tailwind.v4.8.*.css")
        self.assertNotIn("z-\\[200\\]", tailwind)


if __name__ == "__main__":
    unittest.main()
