import os
import unittest


os.environ.setdefault("FLASK_SECRET_KEY", "gemini-agentic-timeout-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-agentic-timeout-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


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


if __name__ == "__main__":
    unittest.main()
