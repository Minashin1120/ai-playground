from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class GeminiBatchRegressionTests(unittest.TestCase):
    def test_gemini_batch_reads_rest_state_and_response_envelope(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def gemini_batch_status_api")
        source = routes[start:]
        self.assertIn("_batch_field(provider_payload, 'metadata')", source)
        self.assertIn("_batch_field(provider_payload, 'response')", source)
        self.assertIn("'inlinedResponses', 'inlined_responses'", source)

    def test_gemini_batch_normalizes_sdk_and_rest_state_values(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        self.assertIn("def _normalize_batch_state", routes)
        self.assertIn("'COMPLETED': 'JOB_STATE_SUCCEEDED'", routes)
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        self.assertIn("row.state = _normalize_batch_state", background)

    def test_unnotified_terminal_jobs_are_repolled_for_results(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        self.assertIn(
            "state not in terminal_states or row.notified_at is None",
            routes,
        )

    def test_gemini_batch_downloads_file_results(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _gemini_result_payload")
        source = routes[start:routes.index("for row in rows:", start)]
        self.assertIn("'responsesFile', 'responses_file'", source)
        self.assertIn("/download/v1beta/", source)
        self.assertIn("_gemini_result_line", source)

    def test_gemini_batch_status_is_polled_until_completion(self):
        part = (APP_ROOT / "static/js/chat_core_parts/chat_core.part15_slash_tempchat_threads.js").read_text(encoding="utf-8")
        self.assertIn("setInterval(refreshGeminiBatchStatus", part)
        self.assertIn("refreshGeminiBatchStatus();", part)
        self.assertIn("15000", part)


if __name__ == "__main__":
    unittest.main()
