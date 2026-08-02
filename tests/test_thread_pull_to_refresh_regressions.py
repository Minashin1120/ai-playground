from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class ThreadPullToRefreshRegressionTests(unittest.TestCase):
    def test_indicator_element_exists_in_thread_list(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        thread_list = template[template.index('id="thread-list"') :]
        thread_list = thread_list[: thread_list.index("</div>", thread_list.index('id="scroll-sentinel"'))]

        self.assertIn('id="thread-pull-indicator"', thread_list)
        self.assertIn("thread-pull-icon", thread_list)
        self.assertIn("thread-pull-spinner", thread_list)
        self.assertIn("thread-pull-label", thread_list)

    def test_init_function_and_touch_handlers_registered(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("function initThreadPullToRefresh()", source)
        self.assertIn("list.addEventListener('touchstart'", source)
        self.assertIn("list.addEventListener('touchmove', (e) => {", source)
        self.assertIn("{ passive: false })", source)
        self.assertIn("list.addEventListener('touchend'", source)
        self.assertIn("list.addEventListener('touchcancel'", source)
        self.assertIn("initThreadPullToRefresh()", source)

    def test_refresh_reloads_first_page_from_top(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("TRIGGER_DIST = 60", source)
        self.assertIn("MAX_PULL_DIST = 88", source)
        self.assertIn("p = loadThreads(false)", source)
        self.assertIn("threadPage = 1", source)
        self.assertIn("ind3.classList.add('refreshing')", source)
        self.assertIn("p.catch(() => {}).finally(() => {", source)

    def test_list_reset_preserves_pull_indicator(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        reset_html = '<div id="thread-pull-indicator" class="thread-pull-indicator"'
        self.assertGreaterEqual(source.count(reset_html), 2)
        sentinel_count = source.count('<div id="scroll-sentinel"></div>')
        self.assertGreaterEqual(sentinel_count, 2)
        # Every list reset that creates a fresh scroll-sentinel must also recreate the indicator.
        for block in source.split(reset_html)[1:]:
            self.assertIn('<div id="scroll-sentinel"></div>', block[:300])

    def test_css_styles_pull_indicator_and_prevents_scroll_chaining(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")

        self.assertIn("#thread-list { overscroll-behavior-y: contain; }", source)
        self.assertIn("#thread-pull-indicator {", source)
        self.assertIn("transition: height 0.25s var(--ease-out);", source)
        self.assertIn("#thread-pull-indicator.pull-ready .thread-pull-icon { transform: rotate(180deg); }", source)
        self.assertIn("#thread-pull-indicator.refreshing .thread-pull-spinner { display: inline-block; }", source)
