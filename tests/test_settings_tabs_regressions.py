from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got {[a.name for a in assets]}"
    return assets[0].read_text(encoding="utf-8")


class SettingsTabsRegressionTests(unittest.TestCase):
    EXPECTED_TABS = [
        "general",
        "api",
        "prompt",
        "display",
        "data",
        "account",
        "security",
        "2fa",
        "feedback",
    ]

    def test_settings_tabs_exist_in_template_and_js(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        js = _current_asset("js", "chat_core.v4.8.*.js")

        for tab_id in self.EXPECTED_TABS:
            self.assertIn(f'id="tab-{tab_id}"', template)
            self.assertIn(f'id="btn-tab-{tab_id}"', template)
            self.assertIn(f"clickTab('{tab_id}')", template)

        self.assertIn("const ALL_TABS = [", js)
        for tab_id in self.EXPECTED_TABS:
            self.assertIn(f"'{tab_id}'", js[js.index("const ALL_TABS = [") : js.index("const ALL_TABS = [") + 400])

        labels_block = js[js.index("const TAB_LABELS = {") : js.index("const ALL_TABS = [")]
        self.assertIn("api: 'APIキー'", labels_block)
        self.assertIn("prompt: 'プロンプト'", labels_block)
        self.assertIn("display: '表示'", labels_block)
        self.assertIn("data: 'データ'", labels_block)
        self.assertIn("account: 'アカウント'", labels_block)

    def test_settings_sections_are_categorized(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        start = template.index('id="settings-modal"')
        end = template.index('id="gemini-local-python-modal"')
        modal = template[start:end]

        def tab_slice(tab_id, next_ids):
            i = modal.index(f'id="tab-{tab_id}"')
            ends = [modal.index(f'id="tab-{n}"') for n in next_ids if f'id="tab-{n}"' in modal]
            j = min(ends) if ends else len(modal)
            return modal[i:j]

        general = tab_slice("general", ["api"])
        api = tab_slice("api", ["prompt"])
        prompt = tab_slice("prompt", ["display"])
        display = tab_slice("display", ["data"])
        data = tab_slice("data", ["account"])
        account = tab_slice("account", ["security"])
        security = tab_slice("security", ["2fa"])

        # 一般: 日常操作
        self.assertIn('id="set-enter-to-send"', general)
        self.assertIn('id="set-default-model"', general)
        self.assertIn('id="set-mic-transcribe-mode"', general)
        self.assertNotIn('id="set-openai"', general)
        self.assertNotIn('id="sys-prompt-text"', general)

        # APIキー
        self.assertIn('id="set-openai"', api)
        self.assertIn('id="set-gemini"', api)
        self.assertIn('id="model-api-key-input"', api)

        # プロンプト
        self.assertIn('id="sys-prompt-text"', prompt)
        self.assertIn('id="set-global-sys-prompt-enabled"', prompt)

        # 表示
        self.assertIn('id="set-theme-color"', display)
        self.assertIn('id="set-liquid-glass"', display)
        self.assertIn('id="set-background-blur-mode"', display)

        # データ
        self.assertIn('id="account-export-btn"', data)
        self.assertIn('id="set-use-sw-cache"', data)
        self.assertIn('id="set-latency-metrics"', data)
        self.assertIn('id="storage-usage-text"', data)

        # アカウント（Google連携を含む）
        self.assertIn('id="set-username"', account)
        self.assertIn('id="easy-login-generate"', account)
        self.assertIn('id="google-link-status"', account)

        # セキュリティから Google を外し、E2EE 等を維持
        self.assertIn('id="set-e2ee"', security)
        self.assertIn('id="session-refresh-btn"', security)
        self.assertNotIn('id="google-link-status"', security)

    def test_admin_security_blocks_remain_guarded(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        start = template.index('id="tab-security"')
        end = template.index('id="tab-2fa"')
        security = template[start:end]

        self.assertGreaterEqual(security.count("{% if is_admin %}"), 4)
        self.assertIn("{% else %}", security)
        self.assertIn("delete-account-btn", security)
        # Bot Detection は管理者のみ
        bot_pos = security.index("set-bot-detect")
        before = security[max(0, bot_pos - 400) : bot_pos]
        self.assertIn("{% if is_admin %}", before)

    def test_settings_tabs_have_edge_arrows_and_wheel_scroll(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        js = _current_asset("js", "chat_core.v4.8.*.js")
        css = _current_asset("css", "chat.custom.v4.8.*.css")

        self.assertIn('id="settings-tabs-wrap"', template)
        self.assertIn('id="settings-tabs"', template)
        self.assertIn('id="settings-tabs-arrow-left"', template)
        self.assertIn('id="settings-tabs-arrow-right"', template)
        self.assertIn("fa-chevron-left", template)
        self.assertIn("fa-chevron-right", template)

        self.assertIn("function initSettingsTabsScroll()", js)
        self.assertIn("function refreshSettingsTabsScroll()", js)
        self.assertIn("function syncSettingsTabsOverflow()", js)
        self.assertIn("wrap.addEventListener('wheel'", js)
        self.assertIn("{ passive: false }", js)
        self.assertIn("primarilyVertical", js)
        self.assertIn("is-edge-left", js)
        self.assertIn("is-edge-right", js)
        self.assertIn("startHold(-1)", js)
        self.assertIn("startHold(1)", js)
        self.assertIn("refreshSettingsTabsScroll()", js)

        self.assertIn(".settings-tabs-arrow", css)
        self.assertIn(".settings-tabs-wrap.can-scroll-left.is-edge-left .settings-tabs-arrow-left", css)
        self.assertIn(".settings-tabs-wrap.can-scroll-right.is-edge-right .settings-tabs-arrow-right", css)
        self.assertIn("overscroll-behavior-x: contain", css)

    def test_blur_toast_opens_display_tab(self):
        js = _current_asset("js", "chat_core.v4.8.*.js")
        helper = js[js.index("const openAdaptiveBlurSettingsFromToast = () => {") :]
        helper = helper[: helper.index("const applyAdaptiveBlurPreference")]
        self.assertIn("tab-display", helper)
        self.assertIn("jumpToSetting", helper)
        self.assertIn("'display'", helper)


if __name__ == "__main__":
    unittest.main()
