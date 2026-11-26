#!/usr/bin/env python3
"""Catch Kotlin syntax mistakes before Android CI runs.

Android builds cannot run on this server, so the first compiler is GitHub
Actions.  This lightweight checker covers the mistakes that are cheap to find
without a compiler: nested block comments (often an unescaped URL such as
``/api/realtime/*``), unclosed comments/strings, and unbalanced delimiters.

It is intentionally conservative: it only reads Kotlin source, understands
comments, strings, characters, backtick identifiers, and string templates, and
never tries to resolve types or imports.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    column: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: error: {self.message}"


class Scanner:
    """A small Kotlin lexer that is only interested in structural mistakes."""

    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self.text = text
        self.index = 0
        self.line = 1
        self.column = 1
        self.problems: list[Problem] = []

    # -- helpers ---------------------------------------------------------
    def peek(self, offset: int = 0) -> str:
        position = self.index + offset
        return self.text[position] if position < len(self.text) else ""

    def starts_with(self, value: str) -> bool:
        return self.text.startswith(value, self.index)

    def advance(self, count: int = 1) -> None:
        for _ in range(count):
            if self.index >= len(self.text):
                return
            character = self.text[self.index]
            self.index += 1
            if character == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1

    def problem(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.problems.append(
            Problem(self.path, line or self.line, column or self.column, message)
        )

    # -- main loop -------------------------------------------------------
    def scan(self) -> list[Problem]:
        self.scan_code(until=None, delimiters=[])
        return self.problems

    def scan_code(self, until: str | None, delimiters: list[tuple[str, int, int]]) -> None:
        while self.index < len(self.text):
            if self.starts_with("//"):
                self.scan_line_comment()
                continue
            if self.starts_with("/*"):
                self.scan_block_comment()
                continue
            if self.starts_with('"""'):
                self.scan_string(raw=True)
                continue
            if self.peek() == '"':
                self.scan_string(raw=False)
                continue
            if self.peek() == "'":
                self.scan_char()
                continue
            if self.peek() == "`":
                self.scan_backtick()
                continue
            if self.starts_with("*/"):
                self.problem("stray '*/' outside a block comment")
                self.advance(2)
                continue

            character = self.peek()
            if character in "({[":
                delimiters.append((character, self.line, self.column))
                self.advance()
                continue
            if character in ")}]":
                if until == "}" and character == "}" and not delimiters:
                    self.advance()
                    return
                self.close_delimiter(character, delimiters)
                self.advance()
                continue
            self.advance()

        if until is not None:
            self.problem("unclosed string template expression")
        for opener, line, column in reversed(delimiters):
            self.problem(f"unclosed '{opener}'", line, column)

    def scan_line_comment(self) -> None:
        while self.index < len(self.text):
            character = self.peek()
            self.advance()
            if character == "\n":
                return

    def scan_block_comment(self) -> None:
        start_line, start_column = self.line, self.column
        depth = 0
        while self.index < len(self.text):
            if self.starts_with("/*"):
                if depth:
                    self.problem(
                        "nested block comment (often an unescaped URL such as "
                        "'/api/realtime/*'); write '/api/realtime/' instead"
                    )
                depth += 1
                self.advance(2)
                continue
            if self.starts_with("*/"):
                depth -= 1
                self.advance(2)
                if depth == 0:
                    return
                continue
            self.advance()
        self.problem("unclosed block comment", start_line, start_column)

    def scan_string(self, raw: bool) -> None:
        start_line, start_column = self.line, self.column
        if raw:
            self.advance(3)
        else:
            self.advance()
        while self.index < len(self.text):
            if raw:
                quotes = 0
                while self.peek(quotes) == '"':
                    quotes += 1
                if quotes >= 3:
                    self.advance(quotes - 3)
                    self.advance(3)
                    return
                if quotes:
                    self.advance(quotes)
                    continue
            else:
                if self.peek() == '"':
                    self.advance()
                    return
                if self.peek() == "\\":
                    self.advance(2)
                    continue
                if self.peek() == "\n":
                    self.problem("unterminated string literal", start_line, start_column)
                    self.advance()
                    return
            if self.peek() == "$" and self.peek(1) == "{":
                self.advance(2)
                self.scan_code(until="}", delimiters=[])
                continue
            self.advance()
        self.problem(
            "unclosed raw string literal" if raw else "unclosed string literal",
            start_line,
            start_column,
        )

    def scan_char(self) -> None:
        start_line, start_column = self.line, self.column
        self.advance()
        while self.index < len(self.text):
            character = self.peek()
            if character == "\\":
                self.advance(2)
                continue
            if character == "'":
                self.advance()
                return
            if character == "\n":
                self.problem("unterminated character literal", start_line, start_column)
                self.advance()
                return
            self.advance()
        self.problem("unclosed character literal", start_line, start_column)

    def scan_backtick(self) -> None:
        start_line, start_column = self.line, self.column
        self.advance()
        while self.index < len(self.text):
            character = self.peek()
            if character == "`":
                self.advance()
                return
            if character == "\n":
                self.problem("unterminated backtick identifier", start_line, start_column)
                self.advance()
                return
            self.advance()
        self.problem("unclosed backtick identifier", start_line, start_column)

    def close_delimiter(self, character: str, delimiters: list[tuple[str, int, int]]) -> None:
        pairs = {")": "(", "]": "[", "}": "{"}
        expected = pairs[character]
        if not delimiters:
            self.problem(f"unmatched '{character}'")
            return
        opener, line, column = delimiters[-1]
        if opener != expected:
            self.problem(
                f"mismatched '{character}' (opened '{opener}' at {line}:{column})"
            )
            return
        delimiters.pop()


def repository_root(start: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return start
    return Path(output.strip())


def tracked_files(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "*.kt"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [root / item for item in output.decode().split("\0") if item]


def changed_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    commands = [
        ["git", "-C", str(root), "diff", "--name-only", "-z", "HEAD", "--", "*.kt"],
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z", "--", "*.kt"],
    ]
    for command in commands:
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            continue
        for item in output.decode().split("\0"):
            if item:
                paths.add(root / item)
    return sorted(paths)


def select_files(root: Path, arguments: list[str], changed: bool) -> list[Path]:
    if arguments:
        return [
            Path(argument) if Path(argument).is_absolute() else root / argument
            for argument in arguments
        ]
    files = changed_files(root) if changed else tracked_files(root)
    return [
        path
        for path in files
        if path.suffix == ".kt"
        and "/build/" not in path.as_posix()
        and "/.gradle/" not in path.as_posix()
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Kotlin files to check (default: all tracked .kt files)")
    parser.add_argument("--changed", action="store_true", help="check only changed and untracked .kt files")
    arguments = parser.parse_args(argv)

    root = repository_root(Path.cwd())
    files = select_files(root, arguments.files, arguments.changed)
    if not files:
        return 0

    failures = 0
    for path in files:
        if not path.exists():
            print(f"{path}: error: file not found", file=sys.stderr)
            failures += 1
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"{path}: error: not valid UTF-8", file=sys.stderr)
            failures += 1
            continue
        problems = Scanner(path, text).scan()
        for problem in problems:
            print(problem.format())
        failures += len(problems)

    if failures:
        print(
            f"Kotlin static check failed: {failures} problem(s) in {len(files)} file(s).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
