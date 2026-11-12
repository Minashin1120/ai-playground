import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("FLASK_SECRET_KEY", "deepseek-search-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-deepseek-search-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class DeepSeekSearchTests(unittest.TestCase):
    def test_search_uses_the_documented_endpoint_and_payload(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "results": [
                {
                    "title": "DeepSeek V4 release",
                    "url": "https://example.com/deepseek-v4",
                    "summary": "DeepSeek released V4.",
                    "snippet": "short",
                    "date": "2026-06-01T00:00:00+00:00",
                }
            ]
        }
        with patch.object(target.requests, "post", return_value=response) as post:
            results, error = target._deepseek_search_request("secret-key", "DeepSeek V4 release")

        self.assertIsNone(error)
        self.assertEqual(1, len(results))
        args, kwargs = post.call_args
        self.assertEqual("https://api.deepseek.com/v1/tools/search", args[0])
        self.assertEqual("Bearer secret-key", kwargs["headers"]["Authorization"])
        self.assertEqual(
            {"query": "DeepSeek V4 release", "count": 8, "need_content": False},
            kwargs["json"],
        )

    def test_empty_results_are_not_treated_as_a_billable_search(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": []}
        with patch.object(target.requests, "post", return_value=response):
            results, error = target._deepseek_search_request("secret-key", "no matches")

        self.assertEqual([], results)
        self.assertIsNone(error)

    def test_failed_search_returns_an_error_without_results(self):
        response = Mock(status_code=429)
        with patch.object(target.requests, "post", return_value=response):
            results, error = target._deepseek_search_request("secret-key", "rate limited")

        self.assertEqual([], results)
        self.assertIn("429", error)

    def test_formats_results_as_model_readable_data(self):
        text = target._format_deepseek_search_results([
            {
                "title": "Example",
                "url": "https://example.com/news",
                "summary": "Latest news.",
                "date": "2026-06-01T00:00:00+00:00",
            },
            {"title": "", "url": "https://example.org", "snippet": "fallback"},
        ])
        self.assertIn("[1] Example", text)
        self.assertIn("URL: https://example.com/news", text)
        self.assertIn("Summary: Latest news.", text)
        self.assertIn("[2] (untitled)", text)
        self.assertIn("Summary: fallback", text)

    def test_deepseek_pricing_counts_each_successful_search(self):
        pricing = json.loads((APP_ROOT / "static/model_pricing.json").read_text(encoding="utf-8"))
        for model in (
            "deepseek-v4.1-flash",
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
            "deepseek-v4-pro",
        ):
            with self.subTest(model=model):
                self.assertEqual(0.00075, pricing["models"][model]["tool_rates"]["web_search"])

    def test_deepseek_branch_calls_search_only_when_the_toggle_is_on(self):
        source = (APP_ROOT / "server/background_stream_deepseek.py").read_text(encoding="utf-8")
        start = source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")')
        end = source.index("# If images are present", start)
        snippet = source[start:end]
        self.assertIn("options.get('enable_search')", snippet)
        self.assertIn("_deepseek_search_request", snippet)
        self.assertIn('pub("search_status", "searching")', snippet)
        self.assertIn('pub("search_status", "done")', snippet)


if __name__ == "__main__":
    unittest.main()
