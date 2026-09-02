from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got {[a.name for a in assets]}"
    return assets[0]


class McpFrontendPartsTests(unittest.TestCase):
    """MCP設定タブとチャット中イベントのフロントエンド実装を検査する。"""

    def _js(self):
        return _current_asset("js", "chat_core.v4.8.*.js").read_text(encoding="utf-8")

    def _template(self):
        return (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

    def _css(self):
        return _current_asset("css", "chat.custom.v4.8.*.css").read_text(encoding="utf-8")

    def test_settings_modal_has_mcp_tab(self):
        template = self._template()
        self.assertIn('id="tab-mcp"', template)
        self.assertIn('id="btn-tab-mcp"', template)
        self.assertIn("clickTab('mcp')", template)
        self.assertIn('id="mcp-server-list"', template)
        self.assertIn('id="mcp-add-server-btn"', template)
        self.assertIn('id="mcp-google-client-id"', template)
        self.assertIn('id="mcp-custom-url"', template)

    def test_js_constants_include_mcp(self):
        js = self._js()
        i = js.index("const ALL_TABS = [")
        self.assertIn("'mcp'", js[i : i + 400])
        j = js.index("const TAB_LABELS = {")
        self.assertIn("mcp: 'MCP'", js[j : js.index("const ALL_TABS = [")])

    def test_chat_stream_handles_mcp_events_in_both_loops(self):
        js = self._js()
        # sendMessage / resumePendingStream の両ループに MCP イベント処理がある
        self.assertEqual(js.count("if (j.type === 'mcp') {"), 2)
        self.assertEqual(js.count("if (j.type === 'mcp_decision_request') {"), 2)
        self.assertIn("handleMcpStreamEvent", js)
        self.assertIn("openMcpDecisionModal", js)
        self.assertIn("submitMcpDecision", js)
        self.assertIn("mcp-decision-modal", self._template())
        self.assertIn("data-mcp-toolbox", js)

    def test_settings_loader_and_render_present(self):
        js = self._js()
        for needle in (
            "async function loadMcpServers",
            "function renderMcpServers",
            "function bindMcpSettingsUi",
            "async function mcpAddCustomServer",
            "/api/mcp/servers",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, js)

    def test_css_classes_present(self):
        css = self._css()
        for needle in (".mcp-box", ".mcp-spinner", "@keyframes mcpSpin", ".mcp-mini-btn"):
            self.assertIn(needle, css)

    def test_parts_source_is_clean(self):
        # 部品ソースは結合ファイルの一部分として壊れていない（重複定義ガード）
        part06 = (APP_ROOT / "static/js/chat_core_parts" / "chat_core.part06_model_media_prompt_cache.js").read_text(encoding="utf-8")
        part14 = (APP_ROOT / "static/js/chat_core_parts" / "chat_core.part14_send_message_browser_fast.js").read_text(encoding="utf-8")
        self.assertEqual(part06.count("function bindMcpSettingsUi"), 1)
        self.assertEqual(part06.count("async function loadMcpServers"), 1)
        self.assertEqual(part14.count("function handleMcpStreamEvent"), 1)
        self.assertEqual(part14.count("function openMcpDecisionModal"), 1)


if __name__ == "__main__":
    unittest.main()
