from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class ModalPerformanceRegressionTests(unittest.TestCase):
    def test_message_detail_modals_follow_browser_history(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn(
            "'/token-details': { id: 'token-detail-modal', open: () => showTokenDetailModal() }",
            source,
        )
        self.assertIn(
            "'/encryption-status': { id: 'encryption-status-modal', open: () => showEncryptionStatusModal() }",
            source,
        )
        self.assertIn("if (messageId !== null) state.messageId = messageId", source)
        self.assertIn("history.pushState(state, '', '/token-details')", source)
        self.assertIn("history.pushState({ modal: 'encryption-status' }, '', '/encryption-status')", source)
        self.assertIn("if (!skipHistory && location.pathname === '/token-details')", source)
        self.assertIn("if (!skipHistory && location.pathname === '/encryption-status')", source)
        self.assertIn("history.replaceState({ modal: 'settings'", source)
        self.assertIn("@app.route('/token-details')", app_source)
        self.assertIn("@app.route('/encryption-status')", app_source)

        token_modal = template[template.index('<div id="token-detail-modal"') :]
        token_modal = token_modal[: token_modal.index(">") + 1]
        self.assertIn("modal-overlay", token_modal)
        self.assertNotIn("onclick=", token_modal)
        encryption_modal = template[template.index('<div id="encryption-status-modal"') :]
        encryption_modal = encryption_modal[: encryption_modal.index(">") + 1]
        self.assertIn("modal-overlay", encryption_modal)

    def test_hidden_conditional_composer_controls_override_component_display(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

        self.assertIn(
            ".composer-opt.hidden,\n.composer-tool-btn.hidden {\n    display: none !important;\n}",
            source,
        )
        self.assertRegex(
            template,
            r'id="mask-btn"[^>]+class="[^"]*composer-tool-btn hidden[^"]*"',
        )
        self.assertIn("maskBtn.classList.add('hidden')", script)
        self.assertIn("thinkOpts.classList.add('hidden')", script)
        self.assertIn("reasonOpts.classList.add('hidden')", script)
        self.assertIn("pyCont.classList.add('hidden')", script)

    def test_reasoning_effort_excludes_non_llm_models(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn(
            "const supportsReasoningEffort = isLlmModel() && (",
            source,
        )
        self.assertIn(
            "if (supportsReasoningEffort) {\n                    reasonOpts.classList.remove('hidden');",
            source,
        )

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

    def test_settings_search_icon_stays_out_of_the_typing_area(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

        self.assertIn("settings-search-icon", template)
        self.assertIn("settings-search-box", template)
        self.assertIn("#settings-modal #settings-search { padding-left: 36px; padding-right: 36px; }", source)
        self.assertIn("#settings-modal .settings-search-icon", source)
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
        # Input shell owns the glass fill; textarea itself stays transparent.
        self.assertIn(".composer-input-shell {", source)
        self.assertIn("backdrop-filter: blur(14px)", source)
        self.assertIn("#prompt-input { background: transparent;", source)

        cookie_bootstrap = template[template.index("const detectedCookieName = 'adaptive_blur_disabled'") :]
        cookie_bootstrap = cookie_bootstrap[: cookie_bootstrap.index("</script>")]
        self.assertIn("document.documentElement.classList.add('performance-blur-disabled')", cookie_bootstrap)
        self.assertIn("mode === 'disabled' || mode === 'lite' || (mode === 'auto'", cookie_bootstrap)
        self.assertLess(template.index("const detectedCookieName = 'adaptive_blur_disabled'"), template.index("chat.tailwind."))

        self.assertIn("const ADAPTIVE_BLUR_COOKIE = 'adaptive_blur_disabled'", script)
        self.assertIn("const ADAPTIVE_BLUR_MODE_COOKIE = 'adaptive_blur_mode'", script)
        self.assertIn("target.closest('button, a, input, select, textarea, [role=\"button\"], [tabindex]')", script)
        self.assertIn("adaptiveBlurInteractionCooldownMs", script)
        self.assertIn("activeStreamingBubbleId", script)
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
        self.assertIn("transition: grid-template-rows 0.34s", original_details)
        self.assertIn("#prompt-details-controls.collapsed", original_details)
        self.assertIn("#prompt-details-controls.expanded", original_details)
        self.assertIn("grid-template-rows: 0fr", original_details)
        self.assertIn("grid-template-rows: 1fr", original_details)
        self.assertIn(".prompt-details-inner", original_details)
        self.assertNotIn("max-height 0.34s", original_details)
        self.assertIn("class=\"prompt-details-inner", template)

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

    def test_lite_mode_auto_apply_shows_a_clickable_toast(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn(".toast-clickable { cursor: pointer; }", source)
        self.assertIn(".toast-clickable:hover", source)
        self.assertIn("showToast('描画負荷が高いため、軽量表示（最小負荷）を自動適用しました。タップで設定を開く'", script)
        self.assertIn("openAdaptiveBlurSettingsFromToast", script)
        # The toast body itself jumps to the blur setting section.
        toast_jump = script[script.index("const openAdaptiveBlurSettingsFromToast = () => {") :]
        toast_jump = toast_jump[: toast_jump.index("const applyAdaptiveBlurPreference")]
        self.assertIn("window.openSettingsModal()", toast_jump)
        self.assertIn("tab-display", toast_jump)
        self.assertIn("jumpToSetting", toast_jump)
        self.assertIn("child.contains(select)", toast_jump)
        # showToast gained an optional onClick that marks the toast clickable.
        show_toast = script[script.index("function showToast(msg, type = \"error\", sticky = false, onClick = null) {") :]
        show_toast = show_toast[: show_toast.index("function showProgressToast")]
        self.assertIn("toast-clickable", show_toast)
        self.assertIn("event.stopPropagation()", show_toast)
        self.assertIn("if (onClick) el.addEventListener('click', onClick)", show_toast)

    def test_lite_mode_toast_is_not_blocked_by_cookie_write_failure(self):
        # V4.8.687: when cookies are blocked, writeAdaptiveBlurCookie must not
        # throw, and the toast must be scheduled before the lite cookie write so
        # the auto-apply notification is always shown.
        script = _current_asset("js", "chat_core.v4.8.*.js")

        write_cookie = script[
            script.index("const writeAdaptiveBlurCookie = (cookieName, value, maxAge = 31536000) => {") :
            script.index("const adaptiveBlurInteractionCooldownMs")
        ]
        self.assertIn("try {", write_cookie)
        self.assertIn("catch (error) {", write_cookie)

        lite_enable = script[
            script.index("const enableAdaptiveBlurLite = () => {") :
            script.index("const openAdaptiveBlurSettingsFromToast")
        ]
        toast_index = lite_enable.index("showToast('描画負荷が高いため、軽量表示（最小負荷）を自動適用しました。タップで設定を開く'")
        lite_cookie_write = lite_enable.index("writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE, '1')")
        self.assertLess(toast_index, lite_cookie_write)

    def test_measurement_covers_page_load_and_broad_interactions(self):
        # V4.8.688: measurement is no longer limited to three sidebar buttons.
        script = _current_asset("js", "chat_core.v4.8.*.js")
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        lite_css = source[source.index("V4.8.683 — adaptive lite mode") :]

        # A forced measurement runs right after theme init so the load-time
        # entrance motion is sampled on every visit.
        theme_init = script[script.index("initThemeFromServer();") :]
        self.assertIn("measureInteractionFrames(true)", theme_init)

        # Interaction measurements are rate-limited and skipped while busy.
        measure = script[script.index("const adaptiveBlurIsBusy = () => {") :]
        measure = measure[: measure.index("const externalScriptLoads")]
        self.assertIn("adaptiveBlurInteractionCooldownMs", measure)
        self.assertIn("if (adaptiveBlurIsBusy()) return", measure)
        self.assertIn("activeStreamingBubbleId", measure)
        self.assertIn("document.querySelector('.modal-overlay.modal-open", measure)

        # The click handler now starts from any interactive element, not a fixed id list.
        self.assertNotIn("ADAPTIVE_BLUR_TRIGGER_IDS", script)

        # Lite mode must not hide entrance-animated elements (welcome screen etc.).
        self.assertIn("html.performance-lite-mode body:not(.liquid-glass-mode) .fade-in", lite_css)
        self.assertIn("html.performance-lite-mode body:not(.liquid-glass-mode) .slide-in-animate", lite_css)
        self.assertIn("opacity: 1 !important", lite_css)
        self.assertIn("transform: none !important", lite_css)

    def test_lite_mode_does_not_hide_thread_and_gem_list_items(self):
        # V4.8.690 / V4.8.801: thread/gem list items rely on model-list-animate.
        # Static opacity-0 was removed in V4.8.801 so items do not get stuck invisible
        # when animations are disabled in lite mode or toggled back to normal.
        source = _current_asset("css", "chat.custom.v4.8.*.css")
        lite_css = source[source.index("V4.8.683 — adaptive lite mode") :]
        self.assertIn("html.performance-lite-mode", lite_css)
        self.assertIn(".model-list-animate", lite_css)
        self.assertIn("opacity: 1 !important", lite_css)

        script = _current_asset("js", "chat_core.v4.8.*.js")
        thread_item = script[script.index("async function loadThreads(append=false) {") :]
        thread_item = thread_item[: thread_item.index("async function loadGems()")]
        self.assertNotIn("model-list-animate", thread_item)
        gem_item = script[script.index("async function loadGems() {") :]
        gem_item = gem_item[: gem_item.index("async function openEditGemModal")]
        self.assertNotIn("model-list-animate", gem_item)

    def test_opening_settings_does_not_blank_sidebar_history(self):
        # V4.8.805: lite mode disables animations. Persistent sidebar lists
        # must stay visible without depending on model-list-animate fill-mode.
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("html.performance-lite-mode #thread-list > [data-thread-id]", css)
        self.assertIn("html.performance-lite-mode #gem-list > .gem-item", css)
        self.assertIn("revealPersistentSidebarLists", script)
        self.assertIn("showModal('settings-modal');", script)
        self.assertIn("restoreThreadSearchValue(preservedThreadSearch", script)
        self.assertIn("revealPersistentSidebarLists();", script)
        self.assertIn("hideModal('settings-modal');\n                revealPersistentSidebarLists();", script)

        thread_item = script[script.index("async function loadThreads(append=false) {") :]
        thread_item = thread_item[: thread_item.index("function initPullToRefresh")]
        self.assertNotIn("model-list-animate", thread_item)
        self.assertIn("searchEl ? searchEl.value : ''", thread_item)

        gem_item = script[script.index("async function loadGems() {") :]
        gem_item = gem_item[: gem_item.index("async function openEditGemModal")]
        self.assertNotIn("model-list-animate", gem_item)

    def test_admin_sidebar_debug_logs_are_admin_only(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        helper = script[script.index("const isAdminSidebarDebugEnabled = () => {") :]
        helper = helper[: helper.index("const ADAPTIVE_BLUR_COOKIE")]
        self.assertIn("cfg.botConfig && cfg.botConfig.isAdmin", helper)
        self.assertIn("if (!isAdminSidebarDebugEnabled()) return null;", helper)
        self.assertIn("[admin-sidebar]", helper)
        self.assertIn("console.log.bind(console)", script)
        self.assertIn("nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX", helper)
        self.assertIn("window.__adminSidebarDebugDump", helper)
        self.assertIn("window.copyAdminSidebarDebug", helper)
        self.assertIn("snapshotSidebarHistory('settings-open-before')", script)
        self.assertIn("snapshotSidebarHistory('settings-close-after')", script)
        self.assertIn("installAdminSidebarDebugObserver", script)
        self.assertIn("if (args && args[0] === ADMIN_SIDEBAR_DEBUG_PREFIX) return;", script)

    def test_settings_open_does_not_reload_or_clear_thread_search(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        search_handler = script[script.index("get('search-box').addEventListener('input'"):]
        search_handler = search_handler[: search_handler.index("if (get('mobile-new-chat-btn')")]
        self.assertIn("isSettingsModalOpen()", search_handler)
        self.assertNotIn("get('thread-list').innerHTML=", search_handler)
        self.assertIn("loadThreads-skipped-settings-open", script)
        self.assertIn("loadThreads-keep-existing-empty-search", script)
        self.assertIn("restoreThreadSearchValue", script)
        self.assertIn('id="search-box"', template)
        search_region = template[template.index('sidebar-search'):template.index('sidebar-search') + 1200]
        self.assertIn('type="search"', search_region)
        self.assertIn('autocomplete="off"', search_region)
        self.assertIn('role="search"', search_region)
        self.assertIn('data-1p-ignore="true"', search_region)
        self.assertIn('data-lpignore="true"', search_region)
        self.assertIn('data-bwignore="true"', search_region)
        self.assertIn('readonly', search_region)
        self.assertIn("hardenThreadSearchInputs", script)
        self.assertIn("discardAutofilledThreadSearch", script)
        self.assertIn("isUserInitiatedSearchInput", script)
        history_region = template[template.index('id="history-search-box"') - 220:template.index('id="history-search-box"') + 700]
        self.assertIn('type="search"', history_region)
        self.assertIn('data-1p-ignore="true"', history_region)
        self.assertIn('readonly', history_region)

    def test_overlay_notifications_follow_visible_composer_dock(self):
        # V4.8.691: the offline connection banner and the global progress spinner
        # were fixed at the bottom-right corner and overlapped the prompt bar /
        # send button. They must be offset by the live composer dock height so
        # they render above the prompt bar while it is available.
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        banner_block = css[css.index("#offline-banner {") :]
        banner_block = banner_block[: banner_block.index("#image-viewer {")]
        self.assertIn("bottom: calc(var(--composer-h, 0px) + 16px);", banner_block)
        spinner_block = css[css.index("body.network-banner-visible #global-progress-spinner {") :]
        spinner_block = spinner_block[: spinner_block.index("#image-viewer {")]
        self.assertIn("bottom: calc(var(--composer-h, 0px) + 64px) !important;", spinner_block)
        self.assertIn("bottom: calc(var(--composer-h, 0px) + 16px) !important;", spinner_block)

        chat_html = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn("querySelector('.composer-dock')", chat_html)
        self.assertIn("--composer-h", chat_html)
        self.assertIn("new ResizeObserver(update)", chat_html)
        # V4.8.693: a full-screen modal covers the composer. In that state the
        # offset must return to zero so notifications sit at the viewport bottom
        # instead of leaving a composer-sized empty margin.
        self.assertIn("querySelectorAll('[id$=\"-modal\"]')", chat_html)
        self.assertIn("!modals.some(isModalVisible)", chat_html)
        self.assertIn("composerVisible ? Math.ceil(rect.height) + 'px' : '0px'", chat_html)
        self.assertIn("new MutationObserver(update)", chat_html)
        # V4.8.694: opening the mobile side menu also covers the composer and
        # must move both notifications to the viewport bottom.
        self.assertIn("sidebar.classList.contains('open')", chat_html)
        self.assertIn("!modals.some(isModalVisible) && !sideMenuOpen", chat_html)
        self.assertIn("modalObserver.observe(sidebar", chat_html)

        # Position changes should be animated for both notification types.
        self.assertIn("bottom 0.32s var(--ease-out)", banner_block)
        self.assertIn("bottom 0.32s var(--ease-out) !important", spinner_block)

    def test_quote_popover_container_has_no_containing_block_transform(self):
        # V4.8.710 moved #quote-popover out of .composer-dock because its
        # backdrop-filter made the dock a containing block for the position:fixed
        # button. That only fixed the dock: the container itself kept a transform
        # (the fade-in messagePop animation with fill-mode:both, plus a :has()
        # popoverIn animation rule), which is ALSO a containing block. JS then set
        # viewport coords that were resolved against the container's 0x0 box and
        # the button was pushed off-screen again. Neither the container nor any of
        # its ancestors may carry a transform/backdrop-filter (incl. via CSS
        # animation), otherwise position:fixed viewport coordinates break.
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        container_start = template.index('<div id="quote-popover-container"')
        container_open = container_start + template[container_start:].index(">")
        container_tag = template[container_start : container_open + 1]
        self.assertNotIn("fade-in", container_tag)
        self.assertNotIn("transform", container_tag)

        css = _current_asset("css", "chat.custom.v4.8.*.css")
        # The only remaining #quote-popover-container rule is a plain layout rule
        # (right offset) with no animation/transform that could create a
        # containing block for the fixed button.
        self.assertEqual(css.count("#quote-popover-container {"), 1)
        container_css = css[css.index("#quote-popover-container") :]
        container_css = container_css[: container_css.index("}")]
        self.assertNotIn("animation", container_css)
        self.assertNotIn("transform", container_css)

        # The entrance animation must live on the fixed button itself so the
        # button still animates in while staying viewport-positioned.
        button_block = css[css.index("#quote-popover[style*=\"display: block\"]") :]
        button_block = button_block[: button_block.index("}")]
        self.assertIn("animation: popoverIn", button_block)

    def test_mobile_quote_preview_uses_composer_bar(self):
        # V4.8.737: on mobile (<=768px) the floating popover is hidden behind the
        # native selection UI, so the selected text is shown as a one-line preview
        # in the composer bar (#quote-bar) and applied via #quote-confirm-btn.
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        bar_start = template.index('<div id="quote-bar"')
        bar_end = template.index("Coding Mode Target Bar", bar_start)
        bar = template[bar_start:bar_end]
        self.assertIn("id=\"quote-confirm-btn\"", bar)
        self.assertIn("quote-text-display", bar)

        css = _current_asset("css", "chat.custom.v4.8.*.css")
        # Hidden everywhere; only revealed on mobile while the quote is pending.
        self.assertIn("#quote-confirm-btn {", css)
        self.assertIn("display: none", css[css.index("#quote-confirm-btn {") : css.index("#quote-confirm-btn:hover")])
        self.assertIn("#quote-bar.preview #quote-confirm-btn { display: inline-flex; }", css)

        js = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("isQuoteMobileLayout", js)
        self.assertIn("const isQuoteMobileLayout = () => window.matchMedia('(max-width: 768px)').matches", js)
        self.assertIn("function showQuotePreview(text)", js)
        # The mobile branch must return before the floating button is positioned.
        handler = js[js.index("function handleQuotePopover()") :]
        handler = handler[: handler.index("document.addEventListener('mouseup'")]
        self.assertIn("showQuotePreview(text)", handler)
        # The confirm button applies the pending preview as the quote.
        self.assertIn("get('quote-confirm-btn').onclick", js)
        self.assertIn("currentQuote = quotePreviewText", js)
        # clearQuote must also reset the pending preview so no stale state remains.
        self.assertIn('quotePreviewText = "";', js)


if __name__ == "__main__":
    unittest.main()
