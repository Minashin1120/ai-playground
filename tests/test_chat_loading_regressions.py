from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_core_source():
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Only the latest versioned chat core asset should remain"
    return assets[0].read_text(encoding="utf-8")


class ChatLoadingRegressionTests(unittest.TestCase):
    def test_markdown_rendering_survives_a_missing_library_without_unsafe_html(self):
        source = _current_chat_core_source()
        sanitizer = source[source.index("function sanitizeMarkdownHtml(text)") :]
        sanitizer = sanitizer[: sanitizer.index("function getCanvasModeElements()")]

        self.assertIn("!window.marked", sanitizer)
        self.assertIn("!window.DOMPurify", sanitizer)
        self.assertIn("return escapeHtml(source).replace(/\\n/g, '<br>')", sanitizer)
        self.assertIn("window.DOMPurify.sanitize(window.marked.parse(source))", sanitizer)

    def test_prompt_cache_ui_helper_is_available_to_thread_loader(self):
        source = _current_chat_core_source()
        helper = "const updatePromptCacheUi = () => {"

        self.assertEqual(source.count(helper), 1)
        self.assertLess(source.index(helper), source.index("document.addEventListener('DOMContentLoaded', () => {"))
        self.assertLess(source.index(helper), source.index("async function loadMessages(tid, opts = {})"))

    def test_thread_loader_cannot_leave_a_permanent_skeleton_on_failure(self):
        source = _current_chat_core_source()
        loader = source[source.index("async function loadMessages(tid, opts = {})"):]
        loader = loader[:loader.index("async function loadOlderMessages()")]

        self.assertIn("if (!r.ok) throw new Error", loader)
        self.assertIn("Array.isArray(threadData.messages)", loader)
        self.assertIn("showChatLoadError(tid)", loader)
        self.assertIn("data-chat-load-retry", source)
        self.assertIn("loadSequence !== threadLoadSequence", loader)


if __name__ == "__main__":
    unittest.main()
