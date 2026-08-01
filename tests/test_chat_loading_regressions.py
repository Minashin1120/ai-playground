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
        # Math is protected before marked.parse so \( / \[ backslashes are not stripped
        self.assertIn("protectMathSegments(source)", sanitizer)
        self.assertIn("restoreMathSegments(parsed, protectedMath.blocks)", sanitizer)
        self.assertIn("window.DOMPurify.sanitize(restored)", sanitizer)

    def test_mathjax_pipeline_protects_delimiters_and_typesets_safely(self):
        source = _current_chat_core_source()
        self.assertIn("function protectMathSegments(src)", source)
        self.assertIn("function restoreMathSegments(html, blocks)", source)
        self.assertIn("@@MATHJAX_BLOCK_", source)
        # marked が落とす \( / \[ と $$ / $ を退避対象に含む
        self.assertIn(r"/\\\(([\s\S]+?)\\\)/g", source)
        self.assertIn(r"/\\\[([\s\S]+?)\\\]/g", source)
        self.assertIn(r"/\$\$([\s\S]+?)\$\$/g", source)
        # 再描画後の typeset 失敗を防ぐ
        self.assertIn("MathJax.typesetClear", source)
        self.assertIn("typesetPromise([container])", source)
        # 一般的な LaTeX デリミタを MathJax 設定へ
        self.assertIn("['\\\\[', '\\\\]']", source)
        self.assertIn("['$', '$']", source)

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

    def test_back_forward_navigation_restores_thread_from_url(self):
        source = _current_chat_core_source()
        popstate = source[source.index("window.addEventListener('popstate'"):]
        popstate = popstate[: popstate.index("const initialPath = location.pathname")]

        self.assertIn("location.pathname.match(/^\\/c\\/(.+)$/)", popstate)
        self.assertIn("loadMessages(tid, { skipHistory: true })", popstate)
        self.assertIn("startNewChat({ skipHistory: true })", popstate)

    def test_url_push_is_skipped_when_restoring_from_history(self):
        source = _current_chat_core_source()
        loader = source[source.index("async function loadMessages(tid, opts = {})"):]
        loader = loader[:loader.index("async function loadOlderMessages()")]
        self.assertIn("if (!opts.skipHistory) history.pushState({}, '', '/c/' + tid);", loader)

        new_chat = source[source.index("function startNewChat(opts = {})"):]
        new_chat = new_chat[:new_chat.index("let threadModalLoadSeq = 0;")]
        self.assertIn("if (!opts.skipHistory) history.pushState({}, '', '/');", new_chat)


if __name__ == "__main__":
    unittest.main()
