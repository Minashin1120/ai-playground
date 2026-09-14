from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class XAIBatchRegressionTests(unittest.TestCase):
    def test_xai_batch_gate_accepts_text_and_excludes_media_models(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _is_xai_batch_model")
        source = routes[start:routes.index("def _is_batch_model", start)]
        self.assertIn("model_l.startswith('grok-')", source)
        for marker in ("'image'", "'video'", "'voice'", "'audio'", "'tts'", "'realtime'"):
            self.assertIn(marker, source)

    def test_xai_batch_uses_inline_batch_workflow(self):
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = background.index("def _submit_xai_batch")
        source = background[start:background.index("def _persist_batch_failure", start)]
        self.assertIn("requests.post(", source)
        self.assertIn("/v1/batches", source)
        self.assertIn("'batch_request_id': job_id", source)
        self.assertIn("'batch_request': {'responses': request_body}", source)
        self.assertIn("/requests", source)

    def test_xai_batch_status_paginates_and_matches_request_id(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _xai_batch_result")
        source = routes[start:routes.index("for row in rows:", start)]
        self.assertIn("pagination_token", source)
        self.assertIn("batch_request_id", source)
        self.assertIn("requests.get(", source)

    def test_xai_batch_result_extracts_chat_completion_usage(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _terminal_xai_message")
        source = routes[start:routes.index("def _openai_batch_state", start)]
        self.assertIn("chat_get_completion", source)
        self.assertIn("reasoning_content", source)
        self.assertIn("completion_tokens", source)


if __name__ == "__main__":
    unittest.main()
