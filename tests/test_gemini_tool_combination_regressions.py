import os
import unittest

from tests.app_source import read_app_source

os.environ.setdefault("FLASK_SECRET_KEY", "gemini-tool-combination-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-tool-combination-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target
from google.genai import types

APP_SOURCE = read_app_source()


def _create_file(name: str) -> str:
    return name


class GeminiToolCombinationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.combines = target._gemini_combines_builtin_and_function_tools

    def test_search_with_file_function_needs_tool_invocations(self):
        tools = [types.Tool(google_search=types.GoogleSearch()), _create_file]
        self.assertTrue(self.combines(tools))

    def test_builtin_tools_with_function_declarations(self):
        for builtin in (
            types.Tool(url_context=types.UrlContext()),
            types.Tool(google_maps=types.GoogleMaps()),
            types.Tool(code_execution=types.ToolCodeExecution()),
        ):
            with self.subTest(builtin=builtin):
                declarations = types.Tool(function_declarations=[{"name": "create_file"}])
                self.assertTrue(self.combines([builtin, declarations]))

    def test_single_kind_of_tool_does_not_need_it(self):
        self.assertFalse(self.combines(None))
        self.assertFalse(self.combines([types.Tool(google_search=types.GoogleSearch())]))
        self.assertFalse(self.combines([_create_file]))

    def test_sdk_supports_the_tool_config_and_parts(self):
        # 1.56 の SDK は ToolConfig にこの項目がなく、toolCall パートを読み捨てていた。
        types.ToolConfig(include_server_side_tool_invocations=True)
        raw = {
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"toolCall": {"id": "t1", "toolType": "GOOGLE_SEARCH_WEB", "args": {"queries": ["x"]}}},
                    {"toolResponse": {"id": "t1", "toolType": "GOOGLE_SEARCH_WEB", "response": {"ok": True}}},
                ]},
            }]
        }
        response = types.GenerateContentResponse._from_response(response=raw, kwargs={})
        parts = response.candidates[0].content.parts
        self.assertEqual(parts[0].tool_call.id, "t1")
        self.assertEqual(parts[1].tool_response.id, "t1")

    def test_gemini_3_request_sets_tool_config_after_all_tools(self):
        start = APP_SOURCE.index("conf['tools'].extend(_gemini_mcp_callables)")
        block = APP_SOURCE[start:APP_SOURCE.index("contents = []", start)]
        self.assertIn("is_gemini_3 and _gemini_combines_builtin_and_function_tools(conf.get('tools'))", block)
        self.assertIn("types.ToolConfig(include_server_side_tool_invocations=True)", block)
        self.assertNotIn("includeServerSideToolInvocations", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
