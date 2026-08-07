"""Regression tests: chat generation errors must be persisted and re-rendered."""
from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_core_source():
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Only the latest versioned chat core asset should remain"
    return assets[0].read_text(encoding="utf-8")


def _format_chat_error_content_from_source_contract(error_text, partial_content=""):
    """Mirror of app.format_chat_error_content for pure-function contract checks."""
    err_body = str(error_text or "Unknown error").strip() or "Unknown error"
    if len(err_body) > 50_000:
        err_body = err_body[:50_000] + "…"
    err_body = err_body.replace("```", "'''")
    fence = f"```chat_error\n{err_body}\n```"
    partial = str(partial_content or "").rstrip()
    if partial:
        return partial + "\n\n" + fence
    return fence


class ChatErrorPersistRegressionTests(unittest.TestCase):
    def test_app_defines_format_and_persist_helpers(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("def format_chat_error_content(error_text, partial_content=\"\"):", source)
        self.assertIn('fence = f"```chat_error\\n{err_body}\\n```"', source)
        self.assertIn('err_body = err_body.replace("```", "\'\'\'")', source)

        task = source[source.index("def background_chat_task(") :]
        task = task[: task.index("@app.route('/')")]
        self.assertIn("def _persist_stream_error(error_text):", task)
        self.assertIn("stream_error_persisted", task)
        self.assertIn("format_chat_error_content(error_text, partial)", task)
        self.assertIn('role="assistant"', task)
        self.assertIn("parent_id=message_id", task)

        # Save must happen before clients observe the error event.
        persist_idx = task.index('if dt == "error":\n                _persist_stream_error(d)')
        publish_idx = task.index("r.publish(channel, json.dumps(event_payload))", persist_idx)
        self.assertLess(persist_idx, publish_idx)

    def test_format_chat_error_content_contract(self):
        self.assertEqual(
            _format_chat_error_content_from_source_contract("API key invalid"),
            "```chat_error\nAPI key invalid\n```",
        )
        self.assertEqual(
            _format_chat_error_content_from_source_contract("boom", "partial answer"),
            "partial answer\n\n```chat_error\nboom\n```",
        )
        out = _format_chat_error_content_from_source_contract("bad ``` fence")
        self.assertIn("'''", out)
        self.assertNotIn("```chat_error\nbad ```", out)

        # Ensure the live app.py body matches the pure contract used above.
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        fn = source[source.index("def format_chat_error_content(") :]
        fn = fn[: fn.index("\ndef safe_db_commit(")]
        self.assertIn('err_body.replace("```", "\'\'\'")', fn)
        self.assertIn('fence = f"```chat_error\\n{err_body}\\n```"', fn)
        self.assertIn("partial + \"\\n\\n\" + fence", fn)

    def test_client_renders_chat_error_fence_and_reloads_after_error(self):
        source = _current_chat_core_source()
        self.assertIn("function buildChatErrorBubbleHtml(errorText)", source)
        self.assertIn("function buildChatErrorMarkdown(errorText, partialContent = '')", source)
        self.assertIn("if (l === 'chat_error')", source)
        self.assertIn("return buildChatErrorBubbleHtml(c || '');", source)
        self.assertEqual(source.count("buildChatErrorBubbleHtml(j.content)"), 2)
        self.assertIn("```chat_error", source)
        self.assertIn(
            "// Errors are persisted server-side; always reload so history stays consistent.",
            source,
        )
        self.assertNotIn(
            "// Full reload to establish new tree structure (skip on error to keep the error visible)",
            source,
        )
        self.assertNotIn("if (!hadError) {\n                    await loadMessages", source)

        browser_fast = source[source.index("async function sendBrowserFastMessage") :]
        browser_fast = browser_fast[: browser_fast.index("async function sendMessage()")]
        self.assertIn("buildChatErrorMarkdown(errMsg, partial)", browser_fast)
        self.assertIn("/api/browser_fast_mode/save", browser_fast)

    def test_version_assets_include_chat_error_support(self):
        assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        ver_m = re.search(r"SYSTEM_VERSION'\] = '(V4\.8\.\d+)'", app_source)
        self.assertIsNotNone(ver_m)
        system_version = ver_m.group(1)
        short = system_version.lower()  # e.g. v4.8.760
        self.assertTrue(assets[0].name.endswith(f".{short.split('v', 1)[-1]}.js") or short in assets[0].name)
        self.assertIn(f"SYSTEM_VERSION'] = '{system_version}'", app_source)
        # Ensure only one versioned triad remains (JS + custom CSS + tailwind CSS).
        js = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        css = list((APP_ROOT / "static" / "css").glob("chat.custom.v4.8.*.css"))
        tw = list((APP_ROOT / "static" / "css").glob("chat.tailwind.v4.8.*.css"))
        self.assertEqual(len(js), 1)
        self.assertEqual(len(css), 1)
        self.assertEqual(len(tw), 1)
        ver_num = re.search(r"v4\.8\.(\d+)", short).group(0)
        self.assertTrue(re.search(re.escape(ver_num), js[0].name))
        self.assertTrue(re.search(re.escape(ver_num), css[0].name))
        self.assertTrue(re.search(re.escape(ver_num), tw[0].name))


if __name__ == "__main__":
    unittest.main()
