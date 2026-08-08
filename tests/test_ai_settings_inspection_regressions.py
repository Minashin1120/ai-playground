from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")


class AiSettingsInspectionRegressionTests(unittest.TestCase):
    def test_llm_can_choose_update_or_read_only_inspection_tool(self):
        schema = APP_SOURCE[APP_SOURCE.index("def _build_ai_settings_tool_schema") :]
        schema = schema[: schema.index("def _call_llm_for_settings_ai")]
        self.assertIn('"name": "update_settings"', schema)
        self.assertIn('"name": "inspect_settings"', schema)
        self.assertIn('"fields": {', schema)
        self.assertIn('"all": {"type": "boolean"', schema)

        caller = APP_SOURCE[APP_SOURCE.index("def _call_llm_for_settings_ai") :]
        caller = caller[: caller.index("@app.route('/api/settings/apply-ai-prompt'")]
        self.assertIn("現在値の確認・表示・質問ではinspect_settings", caller)
        self.assertIn('getattr(fc, "name", "update_settings")', caller)
        self.assertIn('tc.function.name or "update_settings"', caller)

    def test_inspection_returns_current_values_before_any_update_or_commit(self):
        route = APP_SOURCE[APP_SOURCE.index("def apply_ai_settings_prompt") :]
        route = route[: route.index("# --- Session Management ---")]
        inspect_start = route.index("if action == 'inspect_settings':")
        inspect_end = route.index("if action != 'update_settings':")
        inspect_branch = route[inspect_start:inspect_end]

        self.assertIn("'mode': 'inspect'", inspect_branch)
        self.assertIn("'current': inspected", inspect_branch)
        self.assertNotIn("_apply_ai_settings_update", inspect_branch)
        self.assertNotIn("safe_db_commit", inspect_branch)
        self.assertGreater(route.index("_apply_ai_settings_update", inspect_end), inspect_end)
        self.assertGreater(route.index("safe_db_commit", inspect_end), inspect_end)

    def test_snapshot_is_allowlisted_and_hides_long_prompt_bodies_from_model_context(self):
        snapshot = APP_SOURCE[APP_SOURCE.index("def _get_ai_safe_settings_snapshot") :]
        snapshot = snapshot[: snapshot.index("def _apply_ai_settings_update")]
        self.assertIn("for field in AI_SAFE_EDITABLE_FIELDS", snapshot)
        self.assertIn("decrypt_val(system_prompt)", snapshot)
        self.assertIn("def _summarize_ai_settings_for_model", snapshot)
        self.assertIn("'system_prompt', 'llm_transcribe_prompt', 'rich_paste_prompt_default'", snapshot)
        self.assertNotIn("openai_api_key", snapshot)
        self.assertNotIn("gemini_api_key", snapshot)

    def test_followup_conversation_is_bounded_and_passed_as_context(self):
        route = APP_SOURCE[APP_SOURCE.index("def apply_ai_settings_prompt") :]
        route = route[: route.index("# --- Session Management ---")]
        caller = APP_SOURCE[APP_SOURCE.index("def _call_llm_for_settings_ai") :]
        caller = caller[: caller.index("@app.route('/api/settings/apply-ai-prompt'")]
        self.assertIn("raw_history = d.get('conversation')", route)
        self.assertIn("conversation_history=conversation_history", route)
        self.assertIn("過去の設定会話（参考。今回の指示を最優先）", caller)
        self.assertIn("conversation_history[-10:]", caller)


if __name__ == "__main__":
    unittest.main()
