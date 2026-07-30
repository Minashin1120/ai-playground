import os
from pathlib import Path
import unittest


os.environ.setdefault("FLASK_SECRET_KEY", "coding-mode-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-coding-mode-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class CodingModeTests(unittest.TestCase):
    def test_applies_only_unique_search_replace_edits_and_rebuilds_code(self):
        source = "def greet(name):\n    return f'Hello {name}'\n"
        response = (
            '{"summary":"挨拶を変更",'
            '"edits":[{"search":"return f\'Hello {name}\'",'
            '"replace":"return f\'Hi {name}!\'"}]}'
        )

        result = target.apply_coding_mode_edits(response, source, "python")

        self.assertIn("**Coding Mode:** 挨拶を変更", result)
        self.assertIn("```python", result)
        self.assertIn("return f'Hi {name}!'", result)
        self.assertNotIn("return f'Hello {name}'", result)

    def test_rejects_ambiguous_or_missing_search_text(self):
        response = '{"summary":"変更","edits":[{"search":"same","replace":"new"}]}'

        with self.assertRaisesRegex(ValueError, "一致数が2件"):
            target.apply_coding_mode_edits(response, "same\nsame\n", "text")
        with self.assertRaisesRegex(ValueError, "一致数が0件"):
            target.apply_coding_mode_edits(response, "different\n", "text")

    def test_uses_a_longer_markdown_fence_when_code_contains_backticks(self):
        response = '{"summary":"変更なし","edits":[]}'

        result = target.apply_coding_mode_edits(response, "```nested```", "markdown")

        self.assertIn("````markdown\n```nested```\n````", result)

    def test_ignores_python_tool_payload_before_the_final_edit_json(self):
        response = (
            '```pyexec\n{"code":"print(1)","output":"ok"}\n```\n'
            '{"summary":"変更","edits":[{"search":"old","replace":"new"}]}'
        )

        result = target.apply_coding_mode_edits(response, "old", "text")

        self.assertIn("\nnew\n", result)

    def test_extracts_latest_complete_code_block_from_user_prompt(self):
        prompt = (
            "最初\n```javascript\nconst oldValue = 1;\n```\n"
            "こちらを変更\n~~~~python\nprint('latest')\n```\n~~~~\n"
            "未完成は対象外\n```ruby\nputs 'open'"
        )

        target_block = target.extract_latest_markdown_code_block(prompt)

        self.assertEqual(target_block["language"], "python")
        self.assertEqual(target_block["code"], "print('latest')\n```")

    def test_model_target_id_selects_the_matching_candidate(self):
        response = (
            '{"target_id":"python-candidate","summary":"Pythonだけ変更",'
            '"edits":[{"search":"value = 1","replace":"value = 2"}]}'
        )
        candidates = [
            {"id": "js-candidate", "language": "javascript", "code": "const value = 1;"},
            {"id": "python-candidate", "language": "python", "code": "value = 1"},
        ]

        result = target.apply_coding_mode_candidate_edits(
            response,
            candidates,
            "js-candidate",
        )

        self.assertIn("```python", result)
        self.assertIn("value = 2", result)
        self.assertNotIn("const value", result)

    def test_unique_edit_can_infer_candidate_when_model_omits_target_id(self):
        response = (
            '{"summary":"対象推定","edits":'
            '[{"search":"unique_python()","replace":"updated_python()"}]}'
        )
        candidates = [
            {"id": "one", "language": "javascript", "code": "unique_js();"},
            {"id": "two", "language": "python", "code": "unique_python()"},
        ]

        result = target.apply_coding_mode_candidate_edits(response, candidates, "one")

        self.assertIn("updated_python()", result)

    def test_client_requires_explicit_mode_and_supports_target_selection(self):
        assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        source = assets[0].read_text(encoding="utf-8")
        template = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")

        self.assertIn('id="enable-coding-mode"', template)
        self.assertIn('id="coding-target-bar"', template)
        self.assertIn("findLatestCodingTarget()", source)
        self.assertIn("collectCodingCandidates", source)
        self.assertIn("extractLatestPromptCodingTarget", source)
        self.assertIn("collectCodingCandidates(rawText)", source)
        self.assertIn("target.prompt_source", source)
        self.assertIn("selectCodingTargetFromButton", source)
        self.assertIn("coding_mode: codingModeEnabled", source)
        self.assertIn("coding_target: codingTargetForSend", source)
        self.assertIn("codingTargetForSend.prompt_source ? null", source)
        selection_at = source.index("if (codingTargetSelection)", source.index("function resolveCodingTarget"))
        prompt_at = source.index("extractLatestPromptCodingTarget", selection_at)
        self.assertLess(selection_at, prompt_at)

    def test_backend_validates_target_before_message_save(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        route = source[source.index("def chat_stream():"):]
        target_validation_at = route.index("Coding Mode target is required")
        message_save_at = route.index("user_msg = Message(")

        self.assertLess(target_validation_at, message_save_at)
        self.assertIn("CODING_MODE_SYSTEM_PROMPT", source)
        self.assertIn("extract_markdown_code_blocks(raw_message)", source)
        self.assertIn("apply_coding_mode_candidate_edits", source)
        self.assertIn('metadata.get("coding_final")', source)


if __name__ == "__main__":
    unittest.main()
