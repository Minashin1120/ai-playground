from pathlib import Path
import ast
import re
import unittest

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
APP_ROOT = Path(__file__).resolve().parents[1]


def _latest_chat_core_source():
    assets = [
        path for path in (APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js")
        if ".min." not in path.name
    ]
    self_assets = [path for path in assets if path.name.startswith("chat_core.v4.8.")]
    assert self_assets, "chat_core.v4.8.*.js source not found"
    return sorted(self_assets)[-1]


class MistralOcrRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = read_app_source()
        cls.js_path = _latest_chat_core_source()
        cls.js_source = cls.js_path.read_text(encoding="utf-8")
        cls.chat_html = read_chat_markup()
        cls.setup_html = (APP_ROOT / "templates" / "setup.html").read_text(encoding="utf-8")

    def test_backend_registers_mistral_ocr_model_and_provider(self):
        self.assertIn('"mistral-ocr-4-0"', self.app_source)
        self.assertIn("def is_mistral_ocr_model_key(", self.app_source)
        self.assertIn('provider = "mistral"', self.app_source)
        self.assertIn('user_or_admin_env("mistral_api_key", "MISTRAL_API_KEY")', self.app_source)
        self.assertIn("mistral_api_key = db.Column", self.app_source)
        self.assertIn("ensure_user_mistral_api_key_column", self.app_source)
        self.assertIn('f"{MISTRAL_API_BASE}/ocr"', self.app_source)
        self.assertIn('"purpose": "ocr"', self.app_source)
        self.assertIn("Routing: Mistral OCR Branch", self.app_source)
        self.assertIn("'ocr_table_format': data.get('ocr_table_format')", self.app_source)
        self.assertIn("'mistral_key': _masked_secret(current_user.mistral_api_key)", self.app_source)

    def test_ocr_does_not_use_chat_completions_or_history(self):
        self.assertIn("会話履歴は送信されません", self.app_source)
        self.assertIn("elif is_mistral_ocr:", self.app_source)
        self.assertIn("o_client = None", self.app_source)
        self.assertIn("if is_mistral_ocr:", self.app_source)
        self.assertIn("is_mistral_ocr_model_key(mk):", self.app_source)
        self.assertIn("return True", self.app_source)

    def test_frontend_model_definition_and_non_llm_controls(self):
        self.assertRegex(
            self.js_source,
            r'id:\s*"mistral-ocr-4-0"[^}]*name:\s*"Mistral OCR 4"',
        )
        self.assertIn("implementedAt: \"2026-08-15\"", self.js_source)
        self.assertIn("const isMistralOcrModel =", self.js_source)
        self.assertIn("if (isMistralOcrModel(m)) return false;", self.js_source)
        self.assertIn("function updateMistralOcrUi()", self.js_source)
        self.assertIn("ocr_table_format:", self.js_source)
        self.assertIn("Mistral OCR は文書専用です", self.js_source)
        self.assertIn("Mistral OCR は設定変更コマンドに使えません", self.js_source)
        self.assertIn("if (hasXLink && !isMistralOcrModel() && !get('enable-search').checked)", self.js_source)
        self.assertIn("keyField: 'mistral_key'", self.js_source)

    def test_templates_expose_key_and_ocr_options(self):
        self.assertIn('id="set-mistral"', self.chat_html)
        self.assertIn('id="mistral-ocr-options"', self.chat_html)
        self.assertIn('id="ocr-table-format"', self.chat_html)
        self.assertIn('id="ocr-include-blocks"', self.chat_html)
        self.assertIn('id="modal-mistral-ocr-options"', self.chat_html)
        self.assertIn('value="mistral-ocr-4-0"', self.setup_html)
        self.assertIn('name="mistral_key"', self.setup_html)

    def test_ocr_helpers_parse_without_syntax_error(self):
        ast.parse(self.app_source)
        self.assertIn("def _mistral_ocr_process_document(", self.app_source)
        self.assertIn("def _build_mistral_ocr_markdown(", self.app_source)
        self.assertIn("def _extract_mistral_ocr_urls(", self.app_source)


if __name__ == "__main__":
    unittest.main()
