from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _chat_core_source():
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1
    return assets[0].read_text(encoding="utf-8")


class ChatAutoScrollRegressionTests(unittest.TestCase):
    def test_layout_scroll_events_do_not_disable_auto_follow(self):
        source = _chat_core_source()
        controller = source[source.index("const chatContainer = get('chat-container')") :]
        controller = controller[: controller.index("// Image Viewer Logic")]

        self.assertNotIn("userAutoScroll = (this.scrollHeight - this.scrollTop", controller)
        self.assertIn("let chatManualScrollPaused = false", controller)
        self.assertIn("let chatManualResumeArmed = false", controller)
        self.assertIn("let chatManualPauseIntent = false", controller)
        self.assertIn("if (chatManualResumeArmed && isChatNearBottom())", controller)
        self.assertIn("event.deltaY < 0", controller)
        self.assertIn("nextY > chatTouchY + 2", controller)
        self.assertIn("currentScrollTop < chatLastScrollTop - 0.5", controller)

    def test_upward_intent_latches_pause_until_explicit_downward_return(self):
        source = _chat_core_source()
        controller = source[source.index("const chatContainer = get('chat-container')") :]
        controller = controller[: controller.index("// Image Viewer Logic")]

        pause = controller[controller.index("function pauseChatAutoScroll()") :]
        pause = pause[: pause.index("function performChatAutoScroll()")]
        self.assertIn("cancelAnimationFrame(chatAutoScrollFrame)", pause)
        self.assertIn("chatManualScrollPaused = true", pause)
        self.assertIn("chatManualResumeArmed = false", pause)
        self.assertIn("userAutoScroll = false", pause)

        force = controller[controller.index("function scrollToBottom(force = false)") :]
        force = force[: force.index("if (chatContainer) {")]
        self.assertIn("chatManualScrollPaused = false", force)
        self.assertIn("chatManualResumeArmed = false", force)
        self.assertIn("userAutoScroll = true", force)

    def test_resume_requires_explicit_downward_input_or_button(self):
        source = _chat_core_source()
        controller = source[source.index("const chatContainer = get('chat-container')") :]
        controller = controller[: controller.index("// Image Viewer Logic")]

        self.assertNotIn("isChatNearBottom() && movedTowardBottom", controller)
        self.assertIn("event.deltaY > 0 && chatManualScrollPaused", controller)
        self.assertIn("nextY < chatTouchY - 2 && chatManualScrollPaused", controller)
        self.assertIn("['ArrowDown', 'PageDown', 'End']", controller)

    def test_non_scrolling_gestures_do_not_pause_auto_follow(self):
        source = _chat_core_source()
        controller = source[source.index("const chatContainer = get('chat-container')") :]
        controller = controller[: controller.index("// Image Viewer Logic")]

        self.assertIn("function armChatAutoScrollPause()", controller)
        self.assertIn("if (event.deltaY < 0) armChatAutoScrollPause()", controller)
        self.assertIn("nextY > chatTouchY + 2) armChatAutoScrollPause()", controller)
        self.assertNotIn("if (event.deltaY < 0) pauseChatAutoScroll()", controller)
        self.assertIn("chatManualPauseIntent && currentScrollTop < chatLastScrollTop - 0.5", controller)
        self.assertIn("chatPauseIntentTimer = setTimeout", controller)
        intent = controller[controller.index("function armChatAutoScrollPause()") :]
        intent = intent[: intent.index("function pauseChatAutoScroll()")]
        self.assertNotIn("cancelAnimationFrame", intent)
        self.assertNotIn("scrollToBottom()", intent)

    def test_new_generation_and_thread_load_reset_stale_pause(self):
        source = _chat_core_source()

        self.assertGreaterEqual(source.count("resumeChatAutoScroll();"), 3)
        loader = source[source.index("async function loadMessages(tid, opts = {})") :]
        loader = loader[: loader.index("async function loadOlderMessages()")]
        self.assertIn("if (!silent) resumeChatAutoScroll({ scroll: false })", loader)

    def test_content_resize_keeps_stream_pinned_to_bottom(self):
        source = _chat_core_source()
        controller = source[source.index("const chatContainer = get('chat-container')") :]
        controller = controller[: controller.index("// Image Viewer Logic")]

        self.assertIn("new ResizeObserver(() => scrollToBottom())", controller)
        self.assertIn("new MutationObserver((mutations) =>", controller)
        self.assertIn("requestAnimationFrame(performChatAutoScroll)", controller)
        scheduler = controller[controller.index("function scrollToBottom(force = false)") :]
        scheduler = scheduler[: scheduler.index("if (chatContainer) {")]
        self.assertIn("if (chatAutoScrollFrame) return", scheduler)
        self.assertNotIn("cancelAnimationFrame(chatAutoScrollFrame)", scheduler)

    def test_generation_start_scrolls_synchronously_before_following(self):
        source = _chat_core_source()
        resume = source[source.index("function resumeChatAutoScroll(options = {})") :]
        resume = resume[: resume.index("function performChatAutoScroll()")]

        self.assertIn("chatContainer.scrollTop = chatContainer.scrollHeight", resume)
        self.assertIn("else scrollToBottom()", resume)
        self.assertGreaterEqual(source.count("resumeChatAutoScroll();"), 3)

    def test_silent_reload_preserves_scroll_position_across_rebuild(self):
        source = _chat_core_source()
        tree = source[source.index("function renderThreadTree(opts = {})") :]
        tree = tree[: tree.index("function switchVersion(targetId)")]

        # keepScroll captures the scroll state before clearing the container.
        self.assertIn("let scrollState = null;", tree)
        self.assertIn("if (keepScroll) {", tree)
        self.assertIn("top: container.scrollTop", tree)
        self.assertIn("bottomOffset: container.scrollHeight - container.scrollTop - container.clientHeight", tree)

        # The rebuild restores the position synchronously (no one-frame flash to the
        # top), snapping bottom-anchored users to the bottom and keeping users who
        # had scrolled up pinned at the same document position.
        restore = tree[tree.index("function restoreThreadTreeScroll(container, state)") :]
        self.assertIn("state.bottomOffset <= CHAT_BOTTOM_THRESHOLD", restore)
        self.assertIn("container.scrollTop = container.scrollHeight", restore)
        self.assertIn("Math.max(0, Math.min(state.top, maxScroll))", restore)
        self.assertIn("chatLastScrollTop = container.scrollTop", restore)
        self.assertIn("syncScrollToBottomButton()", restore)

        # Silent reloads (stream completion / resume / fast mode / abort sync) keep
        # the current view instead of scrolling to the bottom.
        loader = source[source.index("async function loadMessages(tid, opts = {})") :]
        loader = loader[: loader.index("async function loadOlderMessages()")]
        self.assertIn("renderThreadTree({ silent, keepScroll: silent })", loader)

    def test_resume_button_is_wired_and_accessible(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        css = next((APP_ROOT / "static" / "css").glob("chat.custom.v4.8.*.css")).read_text(encoding="utf-8")
        source = _chat_core_source()

        self.assertIn('id="scroll-to-bottom-btn"', template)
        self.assertIn('aria-label="一番下まで移動して自動スクロールを再開"', template)
        self.assertIn("scrollToBottomBtn.addEventListener('click', () => scrollToBottom(true))", source)
        self.assertIn(".chat-scroll-to-bottom", css)


if __name__ == "__main__":
    unittest.main()
