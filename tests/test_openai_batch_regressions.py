from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OpenAIBatchRegressionTests(unittest.TestCase):
    def test_openai_batch_model_gate_excludes_non_text_models(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _is_openai_batch_model")
        source = routes[start:routes.index("def _is_batch_model", start)]
        self.assertIn("model_l.startswith('gpt-')", source)
        for marker in ("'image'", "'audio'", "'tts'", "'transcribe'", "'realtime'", "'search'"):
            self.assertIn(marker, source)

    def test_openai_batch_uses_files_and_responses_endpoints(self):
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = background.index("def _submit_openai_batch")
        source = background[start:background.index("def _persist_batch_failure", start)]
        self.assertIn("client.files.create", source)
        self.assertIn("purpose='batch'", source)
        self.assertIn("endpoint='/v1/responses'", source)
        self.assertIn("completion_window='24h'", source)
        self.assertIn("'custom_id': job_id", source)

    def test_openai_batch_result_file_is_saved_to_assistant_message(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        self.assertIn("client.batches.retrieve(row.provider_job_name)", routes)
        self.assertIn("_openai_file_text(client, row.output_file_id)", routes)
        self.assertIn("_extract_openai_response_text(body)", routes)
        self.assertIn("usage, 'output_tokens'", routes)

    def test_batch_ui_includes_openai(self):
        part = (APP_ROOT / "static/js/chat_core_parts/chat_core.part06_model_media_prompt_cache.js").read_text(encoding="utf-8")
        self.assertIn("function isBatchModelKey", part)
        self.assertIn("if (m.startsWith('gpt-'))", part)
        self.assertIn("const supported = isBatchModelKey(model)", part)
        template = (APP_ROOT / "templates/chat/composer_controls.html").read_text(encoding="utf-8")
        self.assertIn("Gemini／OpenAI Batch API", template)


if __name__ == "__main__":
    unittest.main()
