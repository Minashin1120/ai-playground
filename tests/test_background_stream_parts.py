"""Extracted `background_stream_*` parts keep every name they rely on.

The parts used to run inline in `background_chat_task`, so they could read that
function's nested helpers (`pub`, `check_stop`, ...) and its accumulators
(`full_res`, ...).  After extraction they only get a `locals()` snapshot, so
every name has to be either bound from `local` or available on the app module.
Missing one only breaks at runtime, mid-stream, so check the compiled code
objects instead of waiting for a request.
"""
import ast
import builtins
import dis
import unittest
from pathlib import Path
from unittest import mock

import app as app_module

APP_ROOT = Path(__file__).resolve().parents[1]


def _part_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def _part_classes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _global_loads(code):
    names = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            names.add(instruction.argval)
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            names |= _global_loads(const)
    return names


class BackgroundStreamPartBindingTests(unittest.TestCase):
    def test_extracted_parts_do_not_load_unknown_globals(self):
        parts = sorted(APP_ROOT.glob("server/background_stream_*.py"))
        self.assertTrue(parts, "background_stream parts are missing")
        for path in parts:
            for function_name in _part_functions(path):
                function = getattr(app_module, function_name, None)
                self.assertIsNotNone(function, f"{path.name}:{function_name} is not loaded by app.py")
                missing = sorted(
                    name
                    for name in _global_loads(function.__code__)
                    if name not in app_module.__dict__ and not hasattr(builtins, name)
                )
                self.assertEqual(
                    [],
                    missing,
                    f"{path.name}:{function_name} reads names that are neither bound from "
                    f"local nor defined on the app module: {missing}",
                )

    def test_extracted_part_classes_do_not_load_unknown_globals(self):
        # Some parts expose a small state holder instead of module-level functions.
        for path in sorted(APP_ROOT.glob("server/background_stream_*.py")):
            for class_name in _part_classes(path):
                cls = getattr(app_module, class_name, None)
                self.assertIsNotNone(cls, f"{path.name}:{class_name} is not loaded by app.py")
                for method_name, method in vars(cls).items():
                    code = getattr(method, "__code__", None)
                    if code is None:
                        continue
                    missing = sorted(
                        name
                        for name in _global_loads(code)
                        if name not in app_module.__dict__ and not hasattr(builtins, name)
                    )
                    self.assertEqual(
                        [],
                        missing,
                        f"{path.name}:{class_name}.{method_name} reads names that are neither "
                        f"arguments nor defined on the app module: {missing}",
                    )

    def test_extracted_branches_return_the_values_the_tail_reads(self):
        """A locals() snapshot does not publish assignments. The caller must get them back."""
        def top_returns(path, name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            found = []

            def walk(node):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    if isinstance(child, ast.Return):
                        found.append(child.value)
                    else:
                        walk(child)

            walk(function)
            return found

        def returns_name(values, name):
            return any(isinstance(value, ast.Name) and value.id == name for value in values)

        def returns_pair(values, first, second):
            for value in values:
                if not isinstance(value, ast.Tuple) or len(value.elts) < 2:
                    continue
                left, right = value.elts[0], value.elts[1]
                if isinstance(left, ast.Name) and left.id == first and isinstance(right, ast.Name) and right.id == second:
                    return True
            return False

        claude = top_returns(APP_ROOT / "server/background_stream_claude.py", "background_stream_claude")
        grok = top_returns(APP_ROOT / "server/background_stream_grok_native.py", "background_stream_grok_native")
        self.assertTrue(returns_pair(claude, "full_res", "thought_accumulated"))
        self.assertTrue(returns_pair(grok, "full_res", "thought_accumulated"))

        image = top_returns(APP_ROOT / "server/background_stream_gpt_image.py", "background_stream_gpt_image")
        search = top_returns(APP_ROOT / "server/background_stream_openai_search.py", "background_stream_openai_search")
        self.assertTrue(returns_name(image, "full_res"))
        self.assertTrue(returns_name(search, "full_res"))

        chat = top_returns(APP_ROOT / "server/background_stream_gemini_chat.py", "background_stream_gemini_chat")
        self.assertTrue(any(
            isinstance(value, ast.Tuple)
            and len(value.elts) == 3
            and all(isinstance(elt, ast.Name) for elt in value.elts)
            and [elt.id for elt in value.elts] == ["full_res", "thought_accumulated", "final_usage_metadata"]
            for value in chat
        ))
        dispatcher = (APP_ROOT / "server/background_stream_gemini.py").read_text(encoding="utf-8")
        self.assertIn(
            "full_res, thought_accumulated, final_usage_metadata = _chat_result",
            dispatcher,
        )

    def test_stream_tail_stops_after_persisting_the_empty_answer_error(self):
        """The error bubble must stay the only message when the answer text is empty.

        `pub("error", ...)` persists an assistant bubble and sets
        `stream_error_persisted` on `background_chat_task`.  The extracted tail
        receives a `locals()` snapshot, so that assignment is invisible to its
        own copy of the flag; without an explicit update it would carry on and
        save a second, empty assistant message.
        """
        published = []

        def fake_pub(kind, data, **_metadata):
            published.append((kind, data))

        local = {
            "is_gem": True,
            "is_deepseek": False,
            "is_grok": False,
            "grok_reasoning_supported": False,
            "full_res": "",
            "generated_images": [],
            "stream_error_persisted": False,
            "pub": fake_pub,
            "agentic_saved_urls": [],
            "agentic_consumed_urls": [],
            "agentic_filename_url_map": {},
            "usage_ledger": mock.Mock(calls=[]),
            "_used_skills": [],
            "history": [],
            "options": {},
        }
        with mock.patch.object(
            app_module,
            "_strip_invented_agentic_images",
            side_effect=AssertionError("a normal message was saved after the error was persisted"),
        ):
            result = app_module.background_stream_tail(local)

        self.assertEqual([("error", "モデルから回答本文が返されませんでした。時間をおいて再試行してください。")], published)
        self.assertIs(result, app_module.BACKGROUND_STREAM_EARLY_RETURN)

    def test_ideogram_edit_uses_the_configured_context_limits(self):
        """The Ideogram edit branch must see MAX_CONTEXT_* from the environment.

        Those values used to be locals of `background_chat_task`, but the history
        loader owns them now.  Reading them from the `locals()` snapshot always
        returned None, so a configured limit silently stopped filtering the
        source images.
        """
        class Reached(BaseException):
            pass

        def stop(*_args, **_kwargs):
            raise Reached()

        local = {
            "full_res": "",
            "generated_images": [],
            "key": "k",
            "loaded_files": [],
            "model_key_l": "ideogram-4.0",
            "msg": mock.Mock(parent_id=2),
            "options": {"ideogram_operation": ""},
            "pub": mock.Mock(),
            "thread_id": 1,
            "user_config": {},
            "user_id": 1,
        }
        with mock.patch.object(app_module, "background_context_limits", return_value=(5, 0)), \
                mock.patch.object(app_module, "ideogram_supports_edit", return_value=True), \
                mock.patch.object(app_module, "_iter_chat_history_ancestors", side_effect=stop):
            with self.assertRaises(Reached):
                app_module.background_stream_ideogram(local)

    def test_extracted_parts_are_registered_before_background(self):
        parts = list(app_module._SERVER_PARTS)
        self.assertIn("background.py", parts)
        background_index = parts.index("background.py")
        for part in parts:
            if part.startswith("background_stream_"):
                self.assertLess(parts.index(part), background_index, f"{part} must load before background.py")


if __name__ == "__main__":
    unittest.main()
