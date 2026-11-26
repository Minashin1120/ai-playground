"""Structural Kotlin checks that run before Android CI."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kotlin_static = load("check_kotlin_static")


def test_valid_strings_templates_and_comments() -> None:
    text = '''package demo

/** A normal KDoc with a URL: https://example.test/api/realtime/. */
class Demo {
    val template = "${foo("bar")}"
    val raw = """viewBox="0 0 1 1""""
    val escaped = "a \\" quote }"
    val character = '\\''
    fun f() {
        println("}")
    }
}
'''
    assert kotlin_static.Scanner(Path("Demo.kt"), text).scan() == []


def test_nested_comment_is_reported() -> None:
    text = '''/**
 * Starts through the server (`/api/realtime/*`, `/sts`).
 */
fun f() {}
'''
    problems = kotlin_static.Scanner(Path("Demo.kt"), text).scan()
    assert any("nested block comment" in problem.message for problem in problems)


def test_unclosed_comment_is_reported() -> None:
    text = "/** open\nfun f() {}\n"
    problems = kotlin_static.Scanner(Path("Demo.kt"), text).scan()
    assert any("unclosed block comment" in problem.message for problem in problems)


def test_unbalanced_delimiter_is_reported() -> None:
    text = "fun f() { println(1 }\n"
    problems = kotlin_static.Scanner(Path("Demo.kt"), text).scan()
    assert any("mismatched" in problem.message or "unclosed" in problem.message for problem in problems)


def test_command_reports_the_file_and_exits_nonzero(tmp_path: Path) -> None:
    source = tmp_path / "Broken.kt"
    source.write_text("/** `/api/realtime/*` */\nfun f() {}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_kotlin_static.py"), str(source)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert str(source) in result.stdout
    assert "nested block comment" in result.stdout
