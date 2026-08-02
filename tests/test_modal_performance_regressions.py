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

    def test_standard_blur_is_restored_until_performance_cookie_disables_it(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        standard_css = source[source.index("V4.8.681 — adaptive standard-mode blur fallback") :]

        self.assertNotIn("pointer: coarse", standard_css)
        self.assertNotIn("hover: none", standard_css)
        self.assertIn("html.performance-blur-disabled body:not(.liquid-glass-mode) #sidebar", standard_css)
        self.assertIn("html.performance-blur-disabled body:not(.liquid-glass-mode) .composer-dock", standard_css)
        self.assertIn("html.performance-blur-disabled body:not(.liquid-glass-mode) .overlay", standard_css)
        self.assertGreaterEqual(standard_css.count("backdrop-filter: none !important"), 6)

        original_sidebar = source[source.index("#sidebar {") : source.index("#chat-container {")]
        self.assertIn("backdrop-filter: blur(var(--blur-panel)) saturate(170%)", original_sidebar)
        original_composer = source[source.index(".composer-dock,") : source.index("/* Sidebar hierarchy */")]
        self.assertIn("backdrop-filter: blur(var(--blur-panel)) saturate(170%)", original_composer)
        self.assertIn("#prompt-input { background: rgba(8, 14, 28, 0.65); backdrop-filter: blur(12px)", source)

        cookie_bootstrap = template[template.index("const detectedCookieName = 'adaptive_blur_disabled'") :]
        cookie_bootstrap = cookie_bootstrap[: cookie_bootstrap.index("</script>")]
        self.assertIn("document.documentElement.classList.add('performance-blur-disabled')", cookie_bootstrap)
        self.assertIn("mode === 'disabled' || mode === 'lite' || (mode === 'auto'", cookie_bootstrap)
        self.assertLess(template.index("const detectedCookieName = 'adaptive_blur_disabled'"), template.index("chat.tailwind."))

        self.assertIn("const ADAPTIVE_BLUR_COOKIE = 'adaptive_blur_disabled'", script)
        self.assertIn("const ADAPTIVE_BLUR_MODE_COOKIE = 'adaptive_blur_mode'", script)
        self.assertIn("'menu-btn'", script)
        self.assertIn("'sidebar-toggle-btn'", script)
        self.assertIn("'prompt-controls-toggle-btn'", script)
        self.assertIn("document.visibilityState !== 'visible'", script)
        self.assertIn("requestAnimationFrame(sampleFrame)", script)
        self.assertIn("droppedFrames >= 5", script)
        self.assertIn("maxAge = 31536000", script)
        self.assertIn("Max-Age=${maxAge}; SameSite=Lax", script)
        self.assertIn('id="set-background-blur-mode"', template)
        self.assertIn('<option value="auto">自動（重い場合のみ無効化）</option>', template)
        self.assertIn('<option value="enabled">常に有効</option>', template)
        self.assertIn('<option value="disabled">常に無効（軽量）</option>', template)
        self.assertIn("applyAdaptiveBlurPreference(get('set-background-blur-mode')", script)
        self.assertIn("adaptiveBlurPreferenceMode !== 'auto'", script)
        self.assertIn("writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE, '', 0)", script)
        self.assertIn("classList.toggle('performance-blur-disabled', adaptiveBlurFallbackEnabled)", script)

        self.assertNotIn("promptDetailsTouchEnter", standard_css)
        self.assertNotIn("#prompt-details-controls.collapsed", standard_css)

        original_details = source[source.index("#prompt-details-controls {") :]
        original_details = original_details[: original_details.index("/* Shimmer sweep animation")]
        self.assertIn("transition: max-height 0.34s", original_details)
        self.assertIn("#prompt-details-controls.collapsed", original_details)
        self.assertIn("#prompt-details-controls.expanded", original_details)

    def test_lite_mode_is_a_second_tier_after_the_blur_fallback(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        lite_css = source[source.index("V4.8.683 — adaptive lite mode") :]

        self.assertIn("html.performance-lite-mode body:not(.liquid-glass-mode) *", lite_css)
        self.assertIn("animation: none !important", lite_css)
        self.assertIn("transition: none !important", lite_css)
        self.assertIn("box-shadow: none !important", lite_css)
        self.assertIn("text-shadow: none !important", lite_css)
        self.assertIn("backdrop-filter: none !important", lite_css)
        self.assertIn("will-change: auto !important", lite_css)
        self.assertNotIn("pointer: coarse", lite_css)
        self.assertNotIn("hover: none", lite_css)

        self.assertIn("const ADAPTIVE_LITE_COOKIE = 'adaptive_lite_mode'", script)
        self.assertIn("let adaptiveBlurLiteEnabled", script)
        self.assertIn("const enableAdaptiveBlurLite = () => {", script)
        adaptive_block = script[script.index("const ADAPTIVE_BLUR_COOKIE") : script.index("const externalScriptLoads")]
        measure_block = adaptive_block[adaptive_block.index("const measureInteractionFrames") : adaptive_block.index("document.addEventListener('click'")]
        # The blur fallback no longer stops the measurement: the second tier is reached.
        self.assertNotIn("adaptiveBlurPreferenceMode !== 'auto' || adaptiveBlurFallbackEnabled", measure_block)
        self.assertIn("adaptiveBlurPreferenceMode !== 'auto' || adaptiveBlurLiteEnabled", measure_block)
        self.assertIn("if (adaptiveBlurFallbackEnabled) {", measure_block)
        self.assertIn("enableAdaptiveBlurLite();", measure_block)
        self.assertIn("enableAdaptiveBlurFallback();", measure_block)
        self.assertIn("writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE, '1')", adaptive_block)
        self.assertIn("writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE, '', 0)", adaptive_block)
        self.assertIn("normalizedMode === 'disabled' || normalizedMode === 'lite'", adaptive_block)

        cookie_bootstrap = template[template.index("const detectedCookieName = 'adaptive_blur_disabled'") :]
        cookie_bootstrap = cookie_bootstrap[: cookie_bootstrap.index("</script>")]
        self.assertIn("const liteCookieName = 'adaptive_lite_mode'", cookie_bootstrap)
        self.assertIn("document.documentElement.classList.add('performance-lite-mode')", cookie_bootstrap)
        self.assertIn("const liteEnabled = mode === 'lite' || (mode === 'auto'", cookie_bootstrap)
        self.assertIn('<option value="lite">常に軽量（最小負荷）</option>', template)


if __name__ == "__main__":
    unittest.main()
