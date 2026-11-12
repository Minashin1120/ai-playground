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

    def test_extracted_parts_are_registered_before_background(self):
        parts = list(app_module._SERVER_PARTS)
        self.assertIn("background.py", parts)
        background_index = parts.index("background.py")
        for part in parts:
            if part.startswith("background_stream_"):
                self.assertLess(parts.index(part), background_index, f"{part} must load before background.py")


if __name__ == "__main__":
    unittest.main()
