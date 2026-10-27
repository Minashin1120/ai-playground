import ast
import copy
import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


from tests.app_source import read_app_source
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
        self.assertIn("--nproc=256", command)
        self.assertNotIn("--nproc=64", command)

    def test_python_sandbox_does_not_inherit_service_environment(self):
        with patch.dict(os.environ, {"SANDBOX_LEAK_CHECK_SECRET": "must-not-leak"}), \
                patch("subprocess.run") as run:
            target.safe_execute_python("print(1)")

        command = run.call_args.args[0]
        self.assertIn("--clearenv", command)
        self.assertLess(command.index("--clearenv"), command.index("/usr/bin/python3"))
        env = run.call_args.kwargs.get("env")
        self.assertIsInstance(env, dict)
        self.assertNotIn("SANDBOX_LEAK_CHECK_SECRET", env)
        self.assertNotIn("FLASK_SECRET_KEY", env)
        self.assertNotIn("must-not-leak", " ".join(command))

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

    def test_execution_guidance_respects_notice_and_python_switches(self):
        source = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = source.index('                    python_tools = []', source.index('            elif is_deepseek:', source.index('def background_chat_task')))
        end = source.index('                    if options.get("enable_file_creation"):', start)
        snippet = textwrap.dedent(source[start:end])
        for python_enabled in (False, True):
            for notice_enabled in (False, True):
                with self.subTest(python=python_enabled, notice=notice_enabled):
                    messages = []
                    read_notice = Mock(return_value="Custom Python instructions")
                    namespace = dict(options={"enable_python": python_enabled}, messages=messages,
                                     _auto_notice_enabled=lambda key: notice_enabled,
                                     _auto_notice_text=read_notice)
                    exec(compile(snippet, "deepseek-tool-setup-test", "exec"), namespace)
                    self.assertEqual(bool(namespace["python_tools"]), python_enabled)
                    self.assertEqual(messages, [{"role": "system", "content": "Custom Python instructions"}]
                                     if python_enabled and notice_enabled else [])
                    if python_enabled and notice_enabled:
                        read_notice.assert_called_once_with("deepseek_python")
                    else:
                        read_notice.assert_not_called()

    def test_sandbox_returns_generated_images_and_skips_symlinks_and_inputs(self):
        image = target._encode_png(1, 1, 6, 8, [b"\xff\x00\x00\xff"])
        outputs = []

        def run(command, **kwargs):
            work = command[command.index("--bind") + 1]
            Path(work, "plot.png").write_bytes(image)
            Path(work, "input_1.png").write_bytes(image)
            Path(work, "link.png").symlink_to(Path(work, "plot.png"))
            kwargs["stdout"].write(b"done")

        with patch("subprocess.run", side_effect=run):
            result = target.safe_execute_python("pass", on_image=outputs.append)
        self.assertEqual(result, "done")
        self.assertEqual(outputs, [image])

    def test_python_request_calls_runtime_and_returns_real_result_to_model(self):
        self._assert_python_rounds(1)

    def test_tool_budget_leaves_one_request_to_write_the_final_answer(self):
        self._assert_python_rounds(8)

    def _assert_python_rounds(self, rounds):
        # Run the production provider branch without a database, Redis, or API key.
        source = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = source.index('                log_force("Routing: DeepSeek V4 Branch (Chat Completions)")')
        end = source.index('            elif is_kimi or is_zai:', start)
        branch = textwrap.dedent(source[start:end])
        module = ast.parse("def run_branch():\n    global thought_accumulated\n    full_res = ''\n" + textwrap.indent(branch, "    "))
        requests = []
        events = []
        execute = Mock(return_value="42\n")

        def chunk(content=None, reasoning=None, calls=None):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
                content=content, reasoning_content=reasoning, tool_calls=calls))])

        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) <= rounds:
                return iter([chunk(reasoning="I will calculate."), chunk(calls=[SimpleNamespace(
                    index=0, id="call_python", type="function", function=SimpleNamespace(
                        name="execute_python", arguments=json.dumps({"code": "print(6 * 7)"})))])])
            return iter([chunk(content="The result is 42.")])

        namespace = dict(
            log_force=lambda *a: None, check_stop=lambda: False,
            o_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            options={"enable_python": True, "enable_thinking": True},
            model_key="deepseek-v4.1-flash", model_key_l="deepseek-v4.1-flash",
            history=[], quote_text="", message_text="Run print(6 * 7) in Python.",
            loaded_files=[], req_reasoning_effort="high", user_id=7,
            _auto_notice_enabled=lambda key: True,
            _auto_notice_text=lambda key: target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS[key],
            _ensure_mcp_env=lambda: None, _deepseek_api_model_id=target._deepseek_api_model_id,
            _deepseek_reasoning_effort=lambda: "high", _mark_provider_request_started=lambda: None,
            _latency_mark_once=lambda *a: None, job_id="test", _refresh_pending_job=lambda: None,
            thought_accumulated="", full_res="", deepseek_tool_context=[],
            accumulate_deepseek_tool_call_deltas=target.accumulate_deepseek_tool_call_deltas,
            safe_execute_python=execute, _markdown_fence_for_code=target._markdown_fence_for_code,
            json=json, pub=lambda *a: events.append(a),
        )
        exec(compile(module, "deepseek-provider-test", "exec"), namespace)
        namespace["run_branch"]()
        self.assertFalse([event for event in events if event[0] == "error"], events)
        self.assertEqual(execute.call_count, rounds)
        self.assertEqual(execute.call_args.args, ("print(6 * 7)",))
        self.assertTrue(callable(execute.call_args.kwargs["on_image"]))
        self.assertEqual(execute.call_args.kwargs["input_images"], [])
        self.assertEqual(len(requests), rounds + 1)
        self.assertEqual(requests[-1]["tool_choice"], "none" if rounds == 8 else "auto")
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "execute_python")
        self.assertTrue(any("call execute_python" in m.get("content", "")
                            for m in requests[0]["messages"] if m["role"] == "system"))
        assistant, result = requests[1]["messages"][-2:]
        self.assertEqual(assistant["reasoning_content"], "I will calculate.")
        self.assertEqual(result, {"role": "tool", "tool_call_id": "call_python", "content": "42\n"})
        self.assertIn(("python", {"id": "call_python", "code": "print(6 * 7)", "output": "42\n"}), events)
        self.assertIn(("content", "The result is 42."), events)

    def test_deepseek_ui_and_backend_enable_the_python_tool(self):
        js_assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        app_source = read_app_source()
        deepseek_branch = app_source[
            app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'):
            app_source.index('elif is_kimi:', app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'))
        ]

        self.assertIn("if (isLlmModel())", js_source)
        self.assertNotIn("if (isLlmModel() && !isDeepSeek)", js_source)
        self.assertIn('"name": "execute_python"', deepseek_branch)
        self.assertIn('deepseek_kwargs["tools"] = python_tools', deepseek_branch)
        self.assertIn('assistant_tool_message["reasoning_content"] = round_reasoning', deepseek_branch)
        self.assertIn("safe_execute_python(code, on_image=publish_python_image, input_images=image_files)", deepseek_branch)
        self.assertIn("max_tool_rounds = 8", deepseek_branch)
        self.assertIn('"deepseek_tool_context"', app_source)


if __name__ == "__main__":
    unittest.main()
