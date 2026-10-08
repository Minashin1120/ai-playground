"""Regression tests: prompt-bar settings are recorded per user prompt (Message.prompt_options)."""
import json
import unittest

from tests.app_source import read_app_source


def _load_helper():
    source = read_app_source()
    start = source.index("_PROMPT_OPTION_KEYS = (")
    end = source.index("@app.route('/api/browser_fast_mode/save'")
    namespace = {"json": json}
    exec(source[start:end], namespace)
    return source, namespace


class PromptOptionsRecordTests(unittest.TestCase):
    def test_records_only_prompt_bar_settings(self):
        _, ns = _load_helper()
        record = json.loads(ns["_build_prompt_options_record"]({
            "enable_search": True,
            "enable_python": False,
            "thinking_level": " high ",
            "thinking_budget": "4096",
            "reasoning_effort": "",
            "canvas_mode": True,
            "ideogram_seed": None,
            # 歯車から開く項目・自由記述・内容は記録しない
            "enable_system_prompt": True,
            "system_prompt": "secret",
            "marker_system_prompt": "secret",
            "thread_custom_instruction": "secret",
            "temporary_chat": True,
            "tts_style": "secret",
            "ideogram_negative_prompt": "secret",
            "xai_stop": "secret",
            "message": "secret",
            "image_mask": "secret",
            "image_urls": ["a"],
            "coding_candidates": [{"code": "secret"}],
        }, {"browser_fast_mode": False, "batch_mode": True}))
        self.assertEqual(record, {
            "enable_search": True,
            "enable_python": False,
            "thinking_level": "high",
            "thinking_budget": "4096",
            "canvas_mode": True,
            "browser_fast_mode": False,
            "batch_mode": True,
        })

    def test_invalid_source_and_long_values(self):
        _, ns = _load_helper()
        build = ns["_build_prompt_options_record"]
        self.assertIsNone(build(None))
        self.assertIsNone(build("x"))
        self.assertIsNone(build({"enable_search": [1], "safety_setting": {"a": 1}}))
        self.assertEqual(
            json.loads(build({"ocr_pages": "9" * 500}))["ocr_pages"], "9" * 80
        )
        forced = json.loads(build({"enable_search": "ignored-by-override"}, {"enable_search": False}))
        self.assertIs(forced["enable_search"], False)

    def test_both_send_paths_store_the_record(self):
        source, _ = _load_helper()
        self.assertIn("prompt_options = db.Column(db.Text, nullable=True)", source)
        self.assertIn("def ensure_message_prompt_options_column():", source)
        self.assertIn("ensure_message_prompt_options_column()", source)
        chat_stream = source[source.index("def chat_stream():"):]
        chat_stream = chat_stream[: chat_stream.index("@app.route('/chat_stream_resume'")]
        self.assertIn("prompt_options=_build_prompt_options_record(", chat_stream)
        fast_save = source[source.index("def save_browser_fast_mode_chat():"):]
        fast_save = fast_save[: fast_save.index("@app.route(", 10)]
        self.assertIn("_build_prompt_options_record(\n                data.get('prompt_options')", fast_save)


if __name__ == "__main__":
    unittest.main()
