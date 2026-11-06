from pathlib import Path
import unittest

from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class PullToRefreshRegressionTests(unittest.TestCase):
    def test_thread_indicator_element_exists_in_thread_list(self):
        template = read_chat_markup()
        thread_list = template[template.index('id="thread-list"') :]
        thread_list = thread_list[: thread_list.index("</div>", thread_list.index('id="scroll-sentinel"'))]

        self.assertIn('id="thread-pull-indicator"', thread_list)
        self.assertIn("ptr-pull-icon", thread_list)
        self.assertIn("ptr-pull-spinner", thread_list)
        self.assertIn("ptr-pull-label", thread_list)

    def test_skill_indicator_element_exists_in_skill_list(self):
        template = read_chat_markup()
        skill_list = template[template.index('id="skill-list"') :]
        skill_list = skill_list[: skill_list.index('</div></div>')]

        self.assertIn('id="skill-pull-indicator"', skill_list)
        self.assertIn("ptr-pull-icon", skill_list)
        self.assertIn("ptr-pull-spinner", skill_list)
        self.assertIn("ptr-pull-label", skill_list)

    def test_init_functions_and_touch_handlers_registered(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("function initPullToRefresh(listId, refreshFn)", source)
        self.assertIn("initThreadPullToRefresh", source)
        self.assertIn("initSkillPullToRefresh", source)
        self.assertIn("initPullToRefresh('thread-list', () => loadThreads(false))", source)
        self.assertIn("initPullToRefresh('skill-list', () => loadSkills())", source)
        self.assertIn("list.addEventListener('touchstart'", source)
        self.assertIn("list.addEventListener('touchmove', (e) => {", source)
        self.assertIn("{ passive: false })", source)
        self.assertIn("list.addEventListener('touchend'", source)
        self.assertIn("list.addEventListener('touchcancel'", source)
        self.assertIn("initPullToRefreshAll()", source)

    def test_refresh_reloads_first_page_from_top(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("TRIGGER_DIST = 60", source)
        self.assertIn("MAX_PULL_DIST = 88", source)
        self.assertIn("p = refreshFn()", source)
        self.assertIn("threadPage = 1", source)
        self.assertIn("ind3.classList.add('refreshing')", source)
        self.assertIn("p.catch(() => {}).finally(() => {", source)

    def test_list_reset_preserves_pull_indicator(self):
        source = _current_asset("js", "chat_core.v4.8.*.js")

        reset_html = '<div id="thread-pull-indicator" class="ptr-pull-indicator"'
        self.assertGreaterEqual(source.count(reset_html), 1)
        sentinel_count = source.count('<div id="scroll-sentinel"></div>')
        self.assertGreaterEqual(sentinel_count, 1)
        # Every list reset that creates a fresh scroll-sentinel must also recreate the indicator.
        for block in source.split(reset_html)[1:]:
            self.assertIn('<div id="scroll-sentinel"></div>', block[:300])
        # Skills reload must recreate the skill indicator too.
        self.assertIn('<div id="skill-pull-indicator" class="ptr-pull-indicator"', source)

    def test_css_styles_pull_indicator_and_prevents_scroll_chaining(self):
        source = _current_asset("css", "chat.custom.v4.8.*.css")

        self.assertIn("#thread-list,", source)
        self.assertIn("#skill-list { overscroll-behavior-y: contain; }", source)
        self.assertIn(".ptr-pull-indicator {", source)
        self.assertIn("transition: height 0.25s var(--ease-out);", source)
        self.assertIn(".ptr-pull-indicator.pull-ready .ptr-pull-icon { transform: rotate(180deg); }", source)
        self.assertIn(".ptr-pull-indicator.refreshing .ptr-pull-spinner { display: inline-block; }", source)
