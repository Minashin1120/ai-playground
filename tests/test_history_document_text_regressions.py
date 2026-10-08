import unittest

from tests.app_source import read_app_source

APP_SOURCE = read_app_source()


class HistoryDocumentTextRegressions(unittest.TestCase):
    """後続ターンで、過去に添付・生成されたDOCX/PDF/テキストを読めなくなる不具合の回帰検証。"""

    def _function_body(self):
        start = APP_SOURCE.index("def _append_history_document_text():")
        end = APP_SOURCE.index("grok_reasoning_supported =", start)
        return APP_SOURCE[start:end]

    def test_history_documents_are_extracted_as_text(self):
        body = self._function_body()
        self.assertIn("_extract_docx_as_numbered(data_h)", body)
        self.assertIn("_extract_text_from_pdf(data_h)", body)
        self.assertIn("_TEXT_LIKE_UPLOAD_EXTS", body)
        self.assertIn("m['content'] =", body)

    def test_current_turn_attachments_are_not_duplicated(self):
        body = self._function_body()
        self.assertIn("for fn in img_list:", body)
        self.assertIn("if not norm_h or norm_h in seen_refs:", body)

    def test_text_size_is_bounded(self):
        body = self._function_body()
        self.assertIn("max_total_chars", body)
        self.assertIn("max_file_chars", body)
        self.assertIn("max_file_bytes", body)

    def test_only_llm_models_get_history_document_text(self):
        self.assertIn(
            "if is_llm_model and not is_mistral_ocr:\n"
            "                try:\n"
            "                    _append_history_document_text()",
            APP_SOURCE,
        )
        self.assertLess(
            APP_SOURCE.index("def _append_history_document_text():"),
            APP_SOURCE.index("_append_history_document_text()\n"),
        )


if __name__ == "__main__":
    unittest.main()
