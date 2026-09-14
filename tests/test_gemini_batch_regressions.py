from pathlib import Path
import ast
from types import SimpleNamespace
from unittest.mock import Mock
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class GeminiBatchRegressionTests(unittest.TestCase):
    def test_gemini_batch_reads_rest_state_and_response_envelope(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        helper_start = routes.index("def _gemini_operation_state")
        helper_source = routes[helper_start:routes.index("def _batch_state_label", helper_start)]
        status_source = routes[routes.index("def gemini_batch_status_api"):]
        self.assertIn("_batch_field(provider_payload, 'metadata')", helper_source)
        self.assertIn("_batch_field(provider_payload, 'response')", helper_source)
        self.assertIn("'inlinedResponses', 'inlined_responses'", status_source)

    def test_gemini_batch_normalizes_sdk_and_rest_state_values(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        self.assertIn("def _normalize_batch_state", routes)
        self.assertIn("'COMPLETED': 'JOB_STATE_SUCCEEDED'", routes)
        start = routes.index("def _normalize_batch_state")
        helper_source = routes[start:routes.index("def _batch_state_label", start)]
        namespace = {}
        exec(helper_source, namespace)
        normalize = namespace["_normalize_batch_state"]
        self.assertEqual(normalize("BATCH_STATE_SUCCEEDED"), "JOB_STATE_SUCCEEDED")
        self.assertEqual(normalize("JOB_STATE_BATCH_STATE_SUCCEEDED"), "JOB_STATE_SUCCEEDED")
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        self.assertIn("row.state = _normalize_batch_state", background)

    def test_gemini_batch_uses_operation_result_when_metadata_state_is_missing(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _batch_field")
        end = routes.index("def _batch_state_label", start)
        namespace = {}
        exec(routes[start:end], namespace)
        resolve = namespace["_gemini_operation_state"]

        self.assertEqual(
            resolve({"done": True, "response": {}}, "JOB_STATE_RUNNING"),
            "JOB_STATE_SUCCEEDED",
        )
        self.assertEqual(
            resolve({"done": True, "error": {"code": 13}}, "JOB_STATE_RUNNING"),
            "JOB_STATE_FAILED",
        )
        self.assertEqual(
            resolve({"done": False, "metadata": {"state": "BATCH_STATE_RUNNING"}}, "JOB_STATE_PENDING"),
            "JOB_STATE_RUNNING",
        )

    def test_unnotified_terminal_jobs_are_repolled_for_results(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        self.assertIn(
            "state not in terminal_states or row.notified_at is None",
            routes,
        )
        self.assertIn("row.state = 'JOB_STATE_FINALIZING'", routes)

    def test_gemini_batch_downloads_file_results(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        start = routes.index("def _gemini_result_payload")
        source = routes[start:routes.index("for row in rows:", start)]
        self.assertIn("'responsesFile', 'responses_file'", source)
        self.assertIn("/download/v1beta/", source)
        self.assertIn("_gemini_result_line", source)
        self.assertIn("if isinstance(responses, dict)", source)
        self.assertIn("'inlinedResponses', 'inlined_responses', 'responses'", source)

    def test_batch_omits_thought_images_but_preserves_final_images(self):
        routes = (APP_ROOT / "server/routes_chat.py").read_text(encoding="utf-8")
        tree = ast.parse(routes)
        status = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "gemini_batch_status_api")
        terminal = next(n for n in status.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_terminal_message")
        field = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_batch_field")
        for sdk in (False, True):
            with self.subTest(sdk=sdk):
                def part(**values):
                    return SimpleNamespace(**values) if sdk else values

                message = SimpleNamespace(is_encrypted=False)
                query = Mock()
                query.filter_by.return_value.first.return_value = message
                save = Mock()
                payload = Mock()
                namespace = {
                    "Message": SimpleNamespace(query=query),
                    "current_user": SimpleNamespace(id=1),
                    "_save_user_generated_bytes": save,
                    "_set_message_payload": payload,
                    "time": SimpleNamespace(time=lambda: 1),
                    "secrets": SimpleNamespace(token_hex=Mock(side_effect=["a", "b"])),
                    "logger": Mock(),
                }
                exec(compile(ast.Module(body=[field, terminal], type_ignores=[]),
                             "<batch-test>", "exec"), namespace)
                blob_key = "inline_data" if sdk else "inlineData"
                mime_key = "mime_type" if sdk else "mimeType"
                def image_part(data, **values):
                    return part(**values, **{blob_key: part(data=data, **{mime_key: "image/png"})})

                response = {"candidates": [{"content": {"parts": [
                    image_part(b"intermediate", thought=True),
                    part(text="reasoning", thought=True),
                    part(text="final answer"),
                    image_part(b"final-one"),
                    image_part(b"final-two", thought=False),
                ]}}]}
                namespace["_terminal_message"](
                    SimpleNamespace(assistant_message_id=2, thread_id=3),
                    "JOB_STATE_SUCCEEDED", response_payload=response,
                )
                self.assertEqual([c.args[1] for c in save.call_args_list],
                                 [b"final-one", b"final-two"])
                args = payload.call_args.args
                self.assertEqual(args[2], "reasoning")
                self.assertEqual(len(args[3]), 2)
                self.assertEqual(args[1].count("![Image]"), 2)
                self.assertTrue(args[1].startswith("final answer"))
                namespace["logger"].warning.assert_not_called()

    def test_gemini_batch_status_is_polled_until_completion(self):
        part = (APP_ROOT / "static/js/chat_core_parts/chat_core.part15_slash_tempchat_threads.js").read_text(encoding="utf-8")
        self.assertIn("setInterval(refreshGeminiBatchStatus", part)
        self.assertIn("refreshGeminiBatchStatus();", part)
        self.assertIn("15000", part)


if __name__ == "__main__":
    unittest.main()
