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
        sanitizer = source[source.index("function sanitizeMarkdownHtml(text, opts = {})") :]
        sanitizer = sanitizer[: sanitizer.index("function getCanvasModeElements()")]

        self.assertIn("!window.marked", sanitizer)
        self.assertIn("!window.DOMPurify", sanitizer)
        self.assertIn("return escapeHtml(source).replace(/\\n/g, '<br>')", sanitizer)
        # Math is protected before marked.parse so \( / \[ backslashes are not stripped
        self.assertIn("protectMathSegments(source)", sanitizer)
        self.assertIn("restoreMathSegments(parsed, protectedMath.blocks, opts)", sanitizer)
        self.assertIn("window.DOMPurify.sanitize(restored)", sanitizer)

    def test_mathjax_pipeline_protects_delimiters_and_typesets_safely(self):
        source = _current_chat_core_source()
        self.assertIn("function protectMathSegments(src)", source)
        self.assertIn("function restoreMathSegments(html, blocks, opts = {})", source)
        self.assertIn("@@MATHJAX_BLOCK_", source)
        # marked が落とす \( / \[ と $$ / $ を退避対象に含む
        self.assertIn(r"/\\\(([\s\S]+?)\\\)/g", source)
        self.assertIn(r"/\\\[([\s\S]+?)\\\]/g", source)
        self.assertIn(r"/\$\$([\s\S]+?)\$\$/g", source)
        # 一文字だけのインライン数式（$x$）も増分描画対象として退避する
        self.assertIn(r"([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)", source)
        # 再描画後の typeset 失敗を防ぐ
        self.assertIn("MathJax.typesetClear", source)
        self.assertIn("typesetPromise([container])", source)
        # 一般的な LaTeX デリミタを MathJax 設定へ
        self.assertIn("['\\\\[', '\\\\]']", source)
        self.assertIn("['$', '$']", source)

    def test_streaming_math_preserves_rendered_nodes_and_typesets_only_new_segments(self):
        source = _current_chat_core_source()
        renderer = source[source.index("function renderAiMarkdownInto(container, text, opts = {})") :]
        renderer = renderer[: renderer.index("function wrapRenderedSvgBoxes(root)")]

        self.assertIn("streamMathSegments: true", renderer)
        self.assertIn("data-stream-math-key", renderer)
        self.assertIn("fresh.replaceWith(old)", renderer)
        self.assertIn("container.replaceChildren(template.content)", renderer)
        self.assertIn("queueIncrementalMathTypeset(newMathSegments)", renderer)

        incremental = source[source.index("function queueIncrementalMathTypeset(elements)") :]
        incremental = incremental[: incremental.index("function queueHighlight(container")]
        self.assertIn("data-stream-math-state", incremental)
        self.assertIn("window.MathJax.typesetPromise(connected)", incremental)
        self.assertNotIn("typesetClear", incremental)

        # All three generation paths (browser-fast, normal, resume) and their final
        # renders must opt into the DOM-preserving renderer.
        self.assertEqual(source.count("renderAiMarkdownInto(contentEl, content, { incrementalMath: true })"), 2)
        self.assertEqual(source.count("renderAiMarkdownInto(cEl, acc, { incrementalMath: true })"), 4)

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

    def test_pending_stream_bubble_renders_into_a_document_fragment_on_reload(self):
        # V4.8.750: reloading during streaming renders the pending-job skeleton
        # inside renderThreadTree's DocumentFragment. DocumentFragment has no
        # insertAdjacentHTML, so calling it directly threw a TypeError and turned
        # the whole thread load into "チャットを読み込めませんでした". renderPendingMessage
        # must fall back to building the node and appending it.
        source = _current_chat_core_source()
        render = source[source.index("function renderPendingMessage(target = null"):]
        render = render[: render.index("function beginPendingToStreamTransition")]

        self.assertIn("renderPendingMessage(fragment", source)
        self.assertIn("typeof container.insertAdjacentHTML === 'function'", render)
        self.assertIn("container.insertAdjacentHTML('beforeend', html)", render)
        self.assertIn("wrap.firstElementChild", render)
        self.assertIn("container.appendChild(node)", render)

    def test_pending_bubble_is_skipped_when_job_is_suppressed(self):
        source = _current_chat_core_source()
        tree = source[source.index("function renderThreadTree(opts = {})"):]
        tree = tree[: tree.index("function switchVersion(targetId)")]

        self.assertIn("const pending = currentThreadPending", tree)
        self.assertIn("!isPendingJobSuppressed(pending.job_id)", tree)
        self.assertIn("renderPendingMessage(fragment", tree)

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
