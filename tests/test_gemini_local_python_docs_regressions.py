from pathlib import Path
import re
import unittest

from tests.app_source import read_app_source

APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")


class GeminiLocalPythonDocumentRegressions(unittest.TestCase):
    """Gemini + Python 実行時、PDF/DOCX 添付で code_execution が 400 を返さないようにする回帰検証。"""

    def test_pdf_docx_detected_as_code_execution_unsupported(self):
        block = APP_SOURCE[
            APP_SOURCE.index("has_code_exec_unsupported_doc = any("):
            APP_SOURCE.index("gemini_local_python = False")
        ]
        self.assertIn("fi.get('is_pdf')", block)
        self.assertIn("fi.get('is_docx')", block)
        self.assertIn("'application/pdf'", block)
        self.assertIn(
            "'application/vnd.openxmlformats-officedocument.wordprocessingml.document'",
            block,
        )

    def test_pdf_docx_included_in_local_python_fallback_condition(self):
        condition = APP_SOURCE[
            APP_SOURCE.index("if is_gem and (has_audio or has_video or has_code_exec_unsupported_doc)"):
            APP_SOURCE.index("if _auto_notice_enabled(\"gemini_local_python\"):")
        ]
        self.assertIn("has_audio or has_video or has_code_exec_unsupported_doc", condition)
        self.assertIn("gemini_local_python = True", condition)

    def test_gemini_code_execution_tool_still_gated_by_local_python_flag(self):
        self.assertIn(
            "if options.get('enable_python') and not gemini_local_python:",
            APP_SOURCE,
        )
        self.assertIn(
            "conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))",
            APP_SOURCE,
        )

    def test_ui_labels_mention_pdf_docx(self):
        self.assertIn(
            "Gemini 音声/動画/PDF/DOCX + Python (ローカル実行時)",
            APP_SOURCE,
        )
        self.assertIn(
            "'Gemini 音声/動画/PDF/DOCX + Python（ローカル実行）'",
            CHAT_JS,
        )

    def test_version_assets_are_present(self):
        css_custom = list((APP_ROOT / "static/css").glob("chat.custom.v4.8.*.css"))
        css_tailwind = list((APP_ROOT / "static/css").glob("chat.tailwind.v4.8.*.css"))
        self.assertEqual(len(css_custom), 1)
        self.assertEqual(len(css_tailwind), 1)
        m = re.search(r"SYSTEM_VERSION'\]\s*=\s*'(V4\.8\.\d+)'", APP_SOURCE)
        if m is None:
            self.fail("SYSTEM_VERSION not found in app.py")
        system_version = m.group(1)
        self.assertEqual(
            CHAT_JS_ASSETS[0].name,
            f"chat_core.{system_version.lower()}.js",
        )
        self.assertEqual(
            css_custom[0].name,
            f"chat.custom.{system_version.lower()}.css",
        )
        self.assertEqual(
            css_tailwind[0].name,
            f"chat.tailwind.{system_version.lower()}.css",
        )


if __name__ == "__main__":
    unittest.main()
