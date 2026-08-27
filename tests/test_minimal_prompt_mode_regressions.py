from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class MinimalPromptModeRegressionTests(unittest.TestCase):
    def test_settings_offer_minimal_prompt_mode(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="set-minimal-prompt-mode"', template)
        self.assertIn('id="set-prompt-bar-mode-normal"', template)
        self.assertIn('id="set-compact-prompt-mode"', template)
        self.assertIn("ミニマル表示（送信・プラスのみ）", template)
        self.assertIn('id="top-model-bar"', template)
        self.assertIn("minimalPromptMode:", template)
        self.assertIn("minimal-prompt-mode", template)

    def test_js_moves_model_selector_and_keeps_send_plus_only(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("function setMinimalPromptMode", script)
        self.assertIn("function applyMinimalPromptMode", script)
        self.assertIn("function placeModelSelectorButton", script)
        self.assertIn("function readPromptBarModeFromForm", script)
        self.assertIn("function writePromptBarModeToForm", script)
        self.assertIn("document.body.classList.toggle('minimal-prompt-mode'", script)
        self.assertIn("fas fa-plus", script)
        self.assertIn("minimal_prompt_mode", script)
        self.assertIn("control: 'set-minimal-prompt-mode'", script)

    def test_css_hides_clutter_and_shows_top_model_bar(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("body.minimal-prompt-mode #top-model-bar", css)
        self.assertIn("body.minimal-prompt-mode #prompt-controls-row", css)
        self.assertIn("body.minimal-prompt-mode #rich-paste-btn", css)
        self.assertIn("body.minimal-prompt-mode #mask-btn", css)
        self.assertIn("body.minimal-prompt-mode #mic-btn", css)
        self.assertIn("display: none !important;", css)

    def test_backend_persists_minimal_prompt_mode(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("minimal_prompt_mode = db.Column(db.Boolean, default=False)", app_source)
        self.assertIn("'minimal_prompt_mode'", app_source)
        self.assertIn("ensure_user_minimal_prompt_mode_column", app_source)
        self.assertIn("ALTER TABLE user ADD COLUMN minimal_prompt_mode BOOLEAN DEFAULT 0", app_source)
        self.assertIn("プロンプトバーをミニマル表示", app_source)

    def test_plus_button_options_popup_markup_exists(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="minimal-options-popup"', template)
        self.assertIn('id="minimal-options-backdrop"', template)
        self.assertIn('id="minimal-options-panel"', template)
        self.assertIn('id="minimal-options-items"', template)
        self.assertIn('id="minimal-options-model-section"', template)
        self.assertIn('id="minimal-options-model-body"', template)
        self.assertIn('id="minimal-options-close-btn"', template)
        self.assertIn('id="thinking-slide-bar"', template)
        self.assertIn('id="thinking-slider"', template)
        self.assertIn('id="thinking-slide-close-btn"', template)

    def test_option_wrappers_have_ids_for_popup_reference(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="safety-option"', template)
        self.assertIn('id="compression-option"', template)

    def test_js_implements_plus_button_popup_and_thinking_slider(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("function openMinimalOptions", script)
        self.assertIn("function closeMinimalOptions", script)
        self.assertIn("function toggleMinimalOptions", script)
        self.assertIn("function renderMinimalOptionItems", script)
        self.assertIn("function moveModelPanelsIntoPopup", script)
        self.assertIn("function restoreModelPanelsFromPopup", script)
        self.assertIn("function showThinkingSlider", script)
        self.assertIn("function hideThinkingSlider", script)
        self.assertIn("function bindUploadButton", script)
        self.assertIn("function bindMinimalOptionsEvents", script)
        self.assertIn("MINIMAL_POPUP_ITEMS", script)
        self.assertIn("MINIMAL_MODEL_PANEL_IDS", script)
        self.assertIn("THINKING_LEVELS", script)
        self.assertIn("btn.onclick = () => {", script)

    def test_css_hides_special_model_panels_in_minimal_mode(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        for panel_id in (
            "gpt-image-options",
            "gemini-image-options",
            "grok-image-options",
            "grok-video-options",
            "mistral-ocr-options",
            "image-input-limits",
            "audio-gen-options",
        ):
            self.assertIn(f".composer-shell > #{panel_id}", css, msg=f"{panel_id} should be hidden inline in minimal mode")
        self.assertIn("display: none !important;", css)

    def test_css_styles_popup_and_slide_bar(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("#minimal-options-popup", css)
        self.assertIn(".minimal-option-item", css)
        self.assertIn(".minimal-option-gear", css)
        self.assertIn("#thinking-slide-bar", css)
        self.assertIn(".thinking-slide-inner", css)
        self.assertIn(".thinking-slide-open", css)

    def test_minimal_mode_file_attach_in_plus_popup(self):
        # Minimal mode removes the paperclip from the input row (it becomes the
        # plus/options button), so file attachment is offered inside the
        # plus-button options popup as a prominent action row.
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("key: 'attach'", script)
        self.assertIn("label: 'ファイルを添付'", script)
        self.assertIn("action: 'upload'", script)
        self.assertIn("item.action === 'upload'", script)
        self.assertIn("closeMinimalOptions();", script)
        self.assertIn("openUploadModal();", script)
        self.assertIn("row.classList.add('action-' + item.action)", script)

        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn(".minimal-option-item.action-upload", css)

    def test_thinking_row_works_when_checkbox_disabled(self):
        # Gemini 3.x forces thinking on and disables the checkbox, so the
        # Thinking row must not be treated as disabled: tapping it opens the
        # level slider.
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("if (item.special === 'thinking') {", script)
        self.assertIn("// Thinking needs special handling", script)
        # minimalOptionDisabled must exempt the thinking row from the plain
        # checkbox-disabled rule.
        self.assertIn(
            "if (item.special === 'thinking') {",
            script,
        )
        self.assertIn("forced on for the current model", script)
        # handleMinimalOptionClick runs the thinking branch before the generic
        # disabled early-return, and opens the slider even when the checkbox is
        # disabled.
        self.assertIn("chk && !chk.disabled", script)
        self.assertIn("closeMinimalOptions();\n                        showThinkingSlider();", script)

    def test_popup_supports_swipe_down_to_close(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("popupSwipeStartY", script)
        self.assertIn("popupSwipeStartX", script)
        self.assertIn("popupSwipeDragging", script)
        self.assertIn("popupSwipeAtTop", script)
        self.assertIn("popupSwipeAxis", script)
        self.assertIn("minimal-options-panel", script)
        # Swipe down past the threshold closes the popup (horizontal swipes are
        # ignored via the axis lock).
        self.assertIn("popupSwipeAtTop && popupSwipeAxis !== 'h' && dy > 70", script)
        # The close starts from the released drag position instead of resetting
        # to translateY(0) first, so the panel never bounces back up.
        self.assertIn("popupPanel.style.transform = `translateY(${Math.max(dy * 0.6, 100)}px)`;", script)
        self.assertIn("popupPanel.style.opacity = '0';", script)
        self.assertIn("closeMinimalOptions();", script)
        # Drag must not be animated by the open transition.
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("#minimal-options-panel.dragging", css)

    def test_popup_swipe_respects_scrolled_lists(self):
        # Swiping down inside a scrolled options/model list must scroll the list
        # instead of closing the popup, so the swipe handler checks scrollTop.
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("node.scrollTop > 0", script)
        self.assertIn("popupPanel.classList.add('dragging')", script)

    def test_thinking_slider_swipe_is_axis_locked_and_dampened(self):
        # Horizontal swipes must not move or dismiss the thinking slider, and a
        # downward swipe needs a dead zone plus a larger threshold so accidental
        # touches do not close it.
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("thinkingSliderAxis", script)
        self.assertIn("thinkingSliderStartX", script)
        self.assertIn("Math.abs(dy) > Math.abs(dx)", script)
        self.assertIn("const travel = Math.min((dy - 8) * 0.5, 120);", script)
        self.assertIn("thinkingSliderAxis === 'v' && dy > 100", script)
        # A cancelled touch must reset the drag state cleanly.
        self.assertIn("addEventListener('touchcancel'", script)
        self.assertIn("popupPanel.addEventListener('touchcancel'", script)

    def test_thinking_slider_swipe_close_does_not_bounce(self):
        # A swipe-to-close must keep the inner bar below the open position so
        # the fade-out runs from the released spot instead of snapping back up
        # to translateY(0).
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("Math.max(dy * 0.5, 60)}px)", script)
        self.assertIn("hideThinkingSlider();", script)
        # hideThinkingSlider must not reset the inner transform synchronously;
        # it clears the leftover transform only after the close transition so
        # the close animation never bounces.
        self.assertIn("if (!thinkingSliderOpen) bar.classList.add('hidden');", script)
        self.assertIn("const inner = get('thinking-slide-inner');", script)
        self.assertIn("if (inner) inner.style.transform = '';", script)
        # showThinkingSlider clears stale drag transforms before reappearing.
        self.assertIn("// Clear any leftover drag transform from a previous swipe-close", script)


if __name__ == "__main__":
    unittest.main()
