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
        self.assertIn("if (isChatNearBottom()) userAutoScroll = true", controller)
        self.assertIn("event.deltaY < 0", controller)
        self.assertIn("nextY > chatTouchY + 2", controller)
        self.assertIn("else if (chatScrollbarDragging) userAutoScroll = false", controller)

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
