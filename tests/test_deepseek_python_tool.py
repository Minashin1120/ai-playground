import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


os.environ.setdefault("FLASK_SECRET_KEY", "deepseek-python-tool-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-deepseek-python-tool-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class DeepSeekPythonToolTests(unittest.TestCase):
    def test_merges_streamed_tool_call_name_and_argument_fragments(self):
        state = {}
        target.accumulate_deepseek_tool_call_deltas(
            state,
            [
                SimpleNamespace(
                    index=0,
                    id="call_python",
                    type="function",
                    function=SimpleNamespace(name="execute_", arguments='{"code":"print'),
                )
            ],
        )
        target.accumulate_deepseek_tool_call_deltas(
            state,
            [
                SimpleNamespace(
                    index=0,
                    id=None,
                    type=None,
                    function=SimpleNamespace(name="python", arguments='(6 * 7)"}'),
                )
            ],
        )

        self.assertEqual(state[0]["id"], "call_python")
        self.assertEqual(state[0]["name"], "execute_python")
        self.assertEqual(state[0]["arguments"], '{"code":"print(6 * 7)"}')

    def test_supports_parallel_dict_tool_call_fragments(self):
        state = {}
        target.accumulate_deepseek_tool_call_deltas(
            state,
            [
                {
                    "index": 1,
                    "id": "second",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(2)"}'},
                },
                {
                    "index": 0,
                    "id": "first",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                },
            ],
        )

        self.assertEqual(state[0]["id"], "first")
        self.assertEqual(state[1]["id"], "second")

    def test_python_sandbox_uses_interpreter_visible_inside_namespace(self):
        with patch("subprocess.run") as run:
            result = target.safe_execute_python("print(3 - 1)")

        self.assertEqual(result, "Success (No output)")
        command = run.call_args.args[0]
        self.assertIn("/usr/bin/python3", command)
        self.assertNotIn("/home/ai-chat-minashin1120/app/venv/bin/python3", command)

    def test_python_sandbox_redacts_host_paths_from_execution_output(self):
        def emit_host_path_error(*args, **kwargs):
            kwargs["stdout"].write(
                b"bwrap: execvp /home/ai-chat-minashin1120/app/venv/bin/python3: No such file or directory\n"
            )

        with patch("subprocess.run", side_effect=emit_host_path_error):
            result = target.safe_execute_python("print(3 - 1)")

        self.assertIn("[host path redacted]", result)
        self.assertNotIn("/home/ai-chat-minashin1120", result)
        self.assertNotIn("venv/bin/python3", result)

    def test_deepseek_ui_and_backend_enable_the_python_tool(self):
        js_assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        deepseek_branch = app_source[
            app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'):
            app_source.index('elif is_kimi:', app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'))
        ]

        self.assertIn("if (isLlmModel())", js_source)
        self.assertNotIn("if (isLlmModel() && !isDeepSeek)", js_source)
        self.assertIn('"name": "execute_python"', deepseek_branch)
        self.assertIn('deepseek_kwargs["tools"] = python_tools', deepseek_branch)
        self.assertIn('assistant_tool_message["reasoning_content"] = round_reasoning', deepseek_branch)
        self.assertIn("safe_execute_python(code)", deepseek_branch)
        self.assertIn("max_tool_rounds = 8", deepseek_branch)
        self.assertIn('"deepseek_tool_context"', app_source)


if __name__ == "__main__":
    unittest.main()
