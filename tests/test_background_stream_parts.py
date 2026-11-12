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

    def test_extracted_parts_are_registered_before_background(self):
        parts = list(app_module._SERVER_PARTS)
        self.assertIn("background.py", parts)
        background_index = parts.index("background.py")
        for part in parts:
            if part.startswith("background_stream_"):
                self.assertLess(parts.index(part), background_index, f"{part} must load before background.py")


if __name__ == "__main__":
    unittest.main()
