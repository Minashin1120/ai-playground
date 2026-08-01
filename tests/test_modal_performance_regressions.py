from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class ModalPerformanceRegressionTests(unittest.TestCase):
    def test_browser_fast_mode_modal_is_fixed_above_home_content(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        modal_css = source[source.index("#browser-fast-mode-modal {") :]
        modal_css = modal_css[: modal_css.index("}")]

        self.assertIn("position: fixed", modal_css)
        self.assertIn("inset: 0", modal_css)
        self.assertIn("z-index: 110", modal_css)
        modal_open = template[template.index('<div id="browser-fast-mode-modal"') :]
        self.assertIn("z-[100]", modal_open.split(">", 1)[0])

    def test_model_list_is_built_once_and_filtered_in_place(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("let modelListBuilt = false", source)
        self.assertIn("if (!container || modelListBuilt) return", source)
        self.assertIn("entry.button.classList.toggle('hidden', !visible)", source)
        self.assertIn("scheduleModelListRender(e.target.value)", source)
        self.assertIn("const apiModelName = String(m.apiId || m.id || '').trim()", source)
        self.assertIn("API model:</span>${escapeHtml(apiModelName)}", source)
        self.assertIn(
            "`${m.name} ${m.id} ${apiModelName} "
            "${m.agenticView ? 'agentic view' : ''}`.toLowerCase()",
            source,
        )

        renderer = source[source.index('function renderModelList(filter = "", options = {})') :]
        renderer = renderer[: renderer.index("function scheduleModelListRender")]
        self.assertNotIn("container.innerHTML = ''", renderer)
        self.assertNotIn("document.createElement('button')", renderer)

    def test_modal_open_does_not_force_synchronous_layout(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")
        helper = source[source.index("const showModal = (id) => {") :]
        helper = helper[: helper.index("window.showModal = showModal")]

        self.assertNotIn("offsetHeight", helper)
        self.assertIn("modalOpenFrames", helper)
        self.assertIn("requestAnimationFrame", helper)

    def test_liquid_glass_pointer_work_is_coalesced_and_rect_is_cached(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("const paintLiquidGlassPointer = (timestamp) =>", source)
        self.assertIn("timestamp - liquidGlassPointerPaintAt < 30", source)
        self.assertIn("surface !== liquidGlassPointerSurface || !liquidGlassPointerRect", source)
        self.assertIn("liquidGlassPointerRect = null", source)

    def test_model_modal_preserves_desktop_and_touch_scrolling(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")

        modal_shell_css = source[source.index("#model-modal {") :]
        modal_shell_css = modal_shell_css[: modal_shell_css.index("#model-list-container {")]
        self.assertIn("overflow: hidden", modal_shell_css)
        self.assertIn("max-height: calc(100dvh - 2rem)", modal_shell_css)

        model_list_css = source[source.index("#model-list-container {") :]
        model_list_css = model_list_css[: model_list_css.index("body.liquid-glass-mode .modal-overlay")]
        self.assertIn("min-height: 0", model_list_css)
        self.assertIn("overflow-y: auto", model_list_css)
        self.assertIn("touch-action: pan-y", model_list_css)
        self.assertNotIn("overflow-y: visible", model_list_css)
        self.assertNotIn("overscroll-behavior: contain", model_list_css)
        self.assertNotIn("content-visibility", model_list_css)

        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        modal_open = template[template.index('<div id="model-modal"') :]
        modal_open = modal_open[: modal_open.index(">") + 1]
        self.assertIn("overflow-hidden", modal_open)
        self.assertNotIn("overflow-y-auto", modal_open)

    def test_modal_glass_keeps_effect_with_reduced_paint_cost(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")

        self.assertIn("backdrop-filter: blur(22px) saturate(170%)", source)
        self.assertIn("body.liquid-glass-mode .modal-overlay .modal-panel", source)
        self.assertIn("transition: opacity 0.24s", source)

    def test_settings_search_icon_moves_out_of_the_typing_area(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

        self.assertIn("settings-search-icon", template)
        self.assertIn("#settings-search:focus { padding-left: 0.75rem; }", source)
        self.assertIn("#settings-search:focus + .settings-search-icon", source)
        self.assertIn("opacity: 0", source)
        self.assertIn("pointer-events: none", source)

    def test_standard_touch_home_avoids_backdrop_raster_during_controls(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        touch_css = source[source.index("V4.8.678 — standard-mode touch interaction paint budget") :]

        self.assertIn("@media (max-width: 768px) and (hover: none) and (pointer: coarse)", touch_css)
        self.assertIn("body:not(.liquid-glass-mode) #sidebar", touch_css)
        self.assertIn("body:not(.liquid-glass-mode) .composer-dock", touch_css)
        self.assertIn("body:not(.liquid-glass-mode) .overlay", touch_css)
        self.assertGreaterEqual(touch_css.count("backdrop-filter: none !important"), 6)

        collapsed = touch_css[touch_css.index("#prompt-details-controls.collapsed") :]
        collapsed = collapsed[: collapsed.index("}")]
        expanded = touch_css[touch_css.index("#prompt-details-controls.expanded") :]
        expanded = expanded[: expanded.index("}")]
        self.assertIn("display: none !important", collapsed)
        self.assertIn("display: flex !important", expanded)
        self.assertIn("promptDetailsTouchEnter 140ms", expanded)


if __name__ == "__main__":
    unittest.main()
