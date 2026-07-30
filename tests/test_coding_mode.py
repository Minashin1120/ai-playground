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
        updated_section = result[result.index("**更新後コード:**"):]
        self.assertNotIn("return f'Hello {name}'", updated_section)

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

    def test_ndjson_edits_are_saved_with_diff_and_updated_code(self):
        response = "\n".join([
            '{"type":"target","target_id":"python","summary":"値を更新"}',
            '{"type":"edit","search":"value = 1","replace":"value = 2"}',
            '{"type":"done"}',
        ])
        candidates = [
            {"id": "python", "language": "python", "code": "value = 1\nprint(value)\n"},
        ]

        result = target.apply_coding_mode_candidate_edits(response, candidates, "python")

        self.assertIn("```diff", result)
        self.assertIn("-value = 1", result)
        self.assertIn("+value = 2", result)
        self.assertIn("**更新後コード:**", result)
        self.assertIn("```python\nvalue = 2\nprint(value)", result)
        self.assertLess(result.index("```diff"), result.index("**更新後コード:**"))

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

    def test_failed_edit_preserves_working_code_for_automatic_repair(self):
        payload = target._parse_coding_mode_edit_payload("\n".join([
            '{"type":"target","target_id":"selected-1","summary":"3か所変更"}',
            '{"type":"edit","search":"first = 1","replace":"first = 10"}',
            '{"type":"edit","search":"second = 2","replace":"second = 20"}',
            '{"type":"edit","search":"missing = 3","replace":"third = 30"}',
            '{"type":"done"}',
        ]))

        with self.assertRaises(target.CodingModeEditApplicationError) as raised:
            target._apply_coding_mode_payload(
                payload,
                "first = 1\nsecond = 2\nthird = 3\n",
            )

        failure = raised.exception
        self.assertEqual(failure.edit_index, 3)
        self.assertIn("first = 10", failure.current_code)
        self.assertIn("second = 20", failure.current_code)
        self.assertIn("third = 3", failure.current_code)
        repair_prompt = target.build_coding_mode_repair_prompt(
            "3か所を更新",
            "selected-1",
            "python",
            failure.current_code,
            failure,
            payload["edits"][failure.edit_index - 1:],
            explicitly_selected=True,
            attempt=1,
        )
        self.assertIn("explicitly selected by the user", repair_prompt)
        self.assertIn('"target_id": "selected-1"', repair_prompt)
        self.assertIn("first = 10", repair_prompt)
        self.assertIn("missing = 3", repair_prompt)

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
        self.assertIn("let codingModeEffective = false", source)
        self.assertIn("const codingModeActiveForSend = codingModeEnabled && codingModeEffective", source)
        self.assertIn("coding_mode: codingModeActiveForSend", source)
        self.assertIn("coding_target: codingModeActiveForSend ?", source)
        self.assertIn("coding_candidates: codingModeActiveForSend ?", source)
        self.assertIn("if (/[`~]/.test(contentDelta)) activateDeferredCodingModeFromStream(acc)", source)
        self.assertGreaterEqual(source.count("activateDeferredCodingModeFromStream(acc)"), 2)
        self.assertIn("コードブロック生成後に自動有効化", source)
        self.assertNotIn("編集対象のコードブロックがありません。先にコードを生成するか", source)
        self.assertIn("explicit: codingTargetForSend.explicit === true", source)
        self.assertIn("codingTargetForSend.prompt_source ? null", source)
        self.assertGreaterEqual(source.count("j.type === 'coding_diff'"), 1)
        self.assertIn("appendCodingLiveDiff", source)
        self.assertIn("Live Code Changes", source)
        self.assertIn("lowerLang === 'diff'", source)
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
        self.assertIn('{"type":"edit","search":"exact existing text"', source)
        self.assertIn('event_payload = {"type": "coding_diff"', source)
        self.assertIn('stream_acc:{job_id}:coding_diff', source)
        self.assertIn("for repair_attempt in range(1, 3)", source)
        self.assertIn("_call_coding_mode_repair_model", source)
        self.assertIn("coding_target.get(\"explicit\") is True", source)
        self.assertIn("自動修復も完了できませんでした", source)
        final_at = source.index("final_content = build_coding_mode_final_markdown")
        save_at = source.index("msg_entry = Message(", final_at)
        self.assertLess(final_at, save_at)
        self.assertIn('metadata.get("coding_final")', source)


if __name__ == "__main__":
    unittest.main()
