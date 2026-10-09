import os
import unittest
from pathlib import Path


from tests.app_source import read_app_source
os.environ.setdefault("FLASK_SECRET_KEY", "gemini-agentic-timeout-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-agentic-timeout-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target

APP_SOURCE = read_app_source()


class GeminiAgenticTimeoutRegressionTests(unittest.TestCase):
    def test_agentic_timeout_is_longer_than_base_timeout(self):
        self.assertGreater(
            target._GEMINI_AGENTIC_TIMEOUT_MS,
            target._GEMINI_TIMEOUT_MS,
            "Agentic View code execution needs a longer deadline than the base timeout.",
        )

    def test_agentic_timeout_respects_env_override(self):
        original = os.environ.get("GEMINI_AGENTIC_TIMEOUT_MS")
        try:
            os.environ["GEMINI_AGENTIC_TIMEOUT_MS"] = "12345"
            value = target._env_int("GEMINI_AGENTIC_TIMEOUT_MS", 600000)
            self.assertEqual(value, 12345)
        finally:
            if original is None:
                os.environ.pop("GEMINI_AGENTIC_TIMEOUT_MS", None)
            else:
                os.environ["GEMINI_AGENTIC_TIMEOUT_MS"] = original

    def test_deadline_error_is_formatted_for_users(self):
        err = Exception(
            "Error: 504 DEADLINE_EXCEEDED. "
            "{'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', "
            "'status': 'DEADLINE_EXCEEDED'}}"
        )
        formatted = target._format_gemini_runtime_error(err, "gemini_api")
        self.assertIn("504 DEADLINE_EXCEEDED", formatted)
        self.assertIn("時間を超えました", formatted)

    def test_code_execution_guidance_mentions_sandbox_runtime_limit(self):
        guidance = target._build_default_auto_system_prompt_notices_config()["gemini_code_execution"]["text"]
        self.assertFalse(hasattr(target, "GEMINI_CODE_EXECUTION_GUIDANCE"))
        self.assertIn("30 seconds", guidance)
        self.assertIn("downscale", guidance.lower())
        self.assertIn("per-pixel", guidance.lower())

    def test_guidance_appended_when_code_execution_tool_enabled(self):
        block_start = APP_SOURCE.index(
            "conf['tools'].append(types.Tool(code_execution=types.ToolCodeExecution()))"
        )
        block_end = APP_SOURCE.index(
            "if options.get('system_prompt') and 'system_instruction' not in conf:",
            block_start,
        )
        block = APP_SOURCE[block_start:block_end]
        self.assertIn("conf['http_options'] = types.HttpOptions(timeout=_GEMINI_AGENTIC_TIMEOUT_MS)", block)
        self.assertIn('_auto_notice_enabled("gemini_code_execution")', block)
        self.assertIn('_auto_notice_text("gemini_code_execution")', block)
        self.assertIn("system_instruction", block)

    def test_file_guidance_is_a_user_editable_auto_notice(self):
        notices = target._build_default_auto_system_prompt_notices_config()
        notice = notices["python_file_creation"]
        self.assertTrue(notice["enabled"])
        self.assertIn("create_file", notice["text"])
        self.assertIn("edit_file", notice["text"])
        self.assertIn("never saved", notice["text"])
        self.assertFalse(hasattr(target, "GEMINI_CODE_EXECUTION_FILE_GUIDANCE"))

    def test_file_guidance_added_only_when_file_creation_and_notice_are_enabled(self):
        block_start = APP_SOURCE.index("code_exec_notices = []")
        block = APP_SOURCE[block_start:block_start + 700]
        self.assertIn("options.get('enable_file_creation')", block)
        self.assertIn('_auto_notice_enabled("python_file_creation")', block)
        self.assertIn('_auto_notice_text("python_file_creation")', block)

    def test_base_system_prompt_still_applied_when_python_off(self):
        self.assertIn(
            "if options.get('system_prompt') and 'system_instruction' not in conf:",
            APP_SOURCE,
        )
        self.assertIn(
            "conf['system_instruction'] = options.get('system_prompt')",
            APP_SOURCE,
        )

    def test_stream_initial_response_504_retry_exists_for_code_execution(self):
        block_start = APP_SOURCE.index(
            "_mark_provider_request_started()\n"
            "                    log_force(f\"STREAM-TRACE: Gemini stream starting for {job_id} model={rm}\")"
        )
        chain_line = "for chunk in itertools.chain([_gemini_first_chunk], _gemini_stream_iter):"
        block_end = APP_SOURCE.index(chain_line)
        block = APP_SOURCE[block_start:block_end]
        self.assertIn("_gemini_code_exec_active", block)
        self.assertIn('"504" in str(_stream_exc) or "DEADLINE_EXCEEDED" in str(_stream_exc)', block)
        self.assertIn("_GEMINI_STREAM_DEADLINE_RETRIES", block)
        self.assertIn("time.sleep(2)", block)
        # itertools.chain is used to feed the already-pulled first chunk into the loop.
        self.assertIn("itertools", APP_SOURCE)
        self.assertIn(chain_line, APP_SOURCE)

    def test_file_and_mcp_tool_requests_are_streamed(self):
        # File creation / MCP turns must stream like any other turn; the
        # non-streaming generate_content path made the answer appear at once.
        block_start = APP_SOURCE.index("def _gemini_manual_function_stream():")
        block_end = APP_SOURCE.index(
            "_mark_provider_request_started()\n"
            "                    log_force(f\"STREAM-TRACE: Gemini stream starting",
            block_start,
        )
        block = APP_SOURCE[block_start:block_end]
        self.assertNotIn("generate_content(", block)
        self.assertEqual(block.count("generate_content_stream("), 2)
        self.assertIn("yield chunk", block)

    def test_stream_deadline_retry_constant_is_defined(self):
        self.assertGreaterEqual(target._GEMINI_STREAM_DEADLINE_RETRIES, 1)

    def test_stream_retry_gated_to_code_execution(self):
        # Retrying before the first chunk is safe, but it is only engaged for
        # code-execution requests to avoid extra input-token charges elsewhere.
        self.assertIn(
            "_gemini_code_exec_active\n"
            "                                and _is_deadline",
            APP_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
