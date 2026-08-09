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

    def test_new_generation_and_thread_load_reset_stale_pause(self):
        source = _chat_core_source()

        self.assertGreaterEqual(source.count("resumeChatAutoScroll({ scroll: false })"), 4)
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
