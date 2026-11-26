#!/usr/bin/env python3
"""Catch missing project imports in Kotlin files before Android CI runs.

This is deliberately conservative.  It never guesses about library symbols:
it only reports a missing import when the name is declared as a type in
another Kotlin file of this repository, in a different package, and is not
imported (directly or by a star import) and not declared in the same package.
That covers the common "moved code but forgot the import" failure without
flagging standard library or Android framework names.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w.]*)", re.MULTILINE)
IMPORT_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][\w.]*\.\*|[A-Za-z_][\w.]*)(?:\s+as\s+([A-Za-z_]\w*))?",
    re.MULTILINE,
)
TYPE_DECL_RE = re.compile(
    r"(?m)(?:^|[;{}])\s*"
    r"(?P<mods>(?:(?:public|private|internal|protected|open|abstract|sealed|data|enum|annotation|value|inner|companion)\s+)*)"
    r"(?:fun\s+)?"
    r"(?:class|interface|object|typealias)\s+"
    r"(?:<[^>\n]*>\s*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
TYPE_PARAM_RE = re.compile(
    r"\b(?:class|interface|object|typealias|fun)\s+<([^>\n]*)>"
)
DECLARED_NAME_RE = re.compile(
    r"\b(?:fun\s+interface|fun|val|var|class|interface|object|typealias)"
    r"\s+(?:<[^>\n]*>\s*)?([A-Za-z_][A-Za-z0-9_]*)"
)
FUNCTION_DECL_RE = re.compile(
    r"\bfun\s+(?:<[^>\n]*>\s*)?(?:[A-Za-z_][\w.<>, ?]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def collect_declared_names(masked: str) -> set[str]:
    """Names declared in this file, including enum entries and function names.

    A capitalized name in a declaration position must not be treated as a type
    reference from another package.
    """
    names = {match.group(1) for match in DECLARED_NAME_RE.finditer(masked)}
    names.update(match.group(1) for match in FUNCTION_DECL_RE.finditer(masked))
    for enum_match in re.finditer(r"\benum\s+class\s+[A-Za-z_]\w*[^{]*\{", masked):
        index = enum_match.end()
        depth = 1
        while index < len(masked) and depth:
            character = masked[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            elif depth == 1 and (character.isalpha() or character == "_"):
                end = index
                while end < len(masked) and (masked[end].isalnum() or masked[end] == "_"):
                    end += 1
                name = masked[index:end]
                lookahead = end
                while lookahead < len(masked) and masked[lookahead] in " \t":
                    lookahead += 1
                previous = index - 1
                while previous >= 0 and masked[previous] in " \t":
                    previous -= 1
                previous_character = masked[previous] if previous >= 0 else "\n"
                if previous_character in "{,;\n" and lookahead < len(masked) and masked[lookahead] in "(,;{}":
                    names.add(name)
                index = end
                continue
            index += 1
    return names


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    column: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: error: {self.message}"


@dataclass
class Declaration:
    path: Path
    package: str


def mask_kotlin(text: str) -> str:
    """Return the source with comments and string/char contents blanked out.

    Newlines are preserved so line/column numbers stay meaningful.
    """
    output = list(text)
    index = 0
    length = len(text)

    def blank(start: int, end: int) -> None:
        for position in range(start, min(end, length)):
            if output[position] != "\n":
                output[position] = " "

    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            if end < 0:
                end = length
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            start = index
            depth = 0
            while index < length:
                if text.startswith("/*", index):
                    depth += 1
                    index += 2
                elif text.startswith("*/", index):
                    depth -= 1
                    index += 2
                    if depth == 0:
                        break
                else:
                    index += 1
            blank(start, index)
            continue
        if text.startswith('"""', index):
            start = index
            index += 3
            while index < length:
                quotes = 0
                while index + quotes < length and text[index + quotes] == '"':
                    quotes += 1
                if quotes >= 3:
                    index += quotes
                    break
                if quotes:
                    index += quotes
                    continue
                index += 1
            blank(start, index)
            continue
        character = text[index]
        if character == '"':
            start = index
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                elif text[index] == "\n":
                    break
                else:
                    index += 1
            blank(start, index)
            continue
        if character == "'":
            start = index
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                elif text[index] == "'":
                    index += 1
                    break
                elif text[index] == "\n":
                    break
                else:
                    index += 1
            blank(start, index)
            continue
        if character == "`":
            start = index
            index += 1
            while index < length and text[index] not in "`\n":
                index += 1
            if index < length and text[index] == "`":
                index += 1
            blank(start, index)
            continue
        index += 1
    return "".join(output)


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


def line_column(text: str, position: int) -> tuple[int, int]:
    line = text.count("\n", 0, position) + 1
    last_newline = text.rfind("\n", 0, position)
    column = position - last_newline
    return line, column


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Kotlin files to check (default: all tracked .kt files)")
    parser.add_argument("--changed", action="store_true", help="check only changed and untracked .kt files")
    arguments = parser.parse_args(argv)

    root = repository_root(Path.cwd())
    target_files = select_files(root, arguments.files, arguments.changed)
    if not target_files:
        return 0

    all_files = sorted(set(tracked_files(root)) | set(target_files))
    packages: dict[Path, str] = {}
    imports: dict[Path, set[str]] = {}
    wildcard_imports: dict[Path, set[str]] = {}
    declarations: dict[str, list[Declaration]] = defaultdict(list)
    type_parameters: dict[Path, set[str]] = {}
    texts: dict[Path, str] = {}

    for path in all_files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        texts[path] = text
        masked = mask_kotlin(text)
        package_match = PACKAGE_RE.search(masked)
        package = package_match.group(1) if package_match else ""
        packages[path] = package

        file_imports: set[str] = set()
        file_wildcards: set[str] = set()
        for match in IMPORT_RE.finditer(masked):
            imported, alias = match.group(1), match.group(2)
            if imported.endswith(".*"):
                file_wildcards.add(imported[:-2])
            else:
                file_imports.add(alias or imported.rsplit(".", 1)[-1])
        imports[path] = file_imports
        wildcard_imports[path] = file_wildcards

        type_parameters[path] = {
            name.strip()
            for group in TYPE_PARAM_RE.findall(masked)
            for name in group.split(",")
            if name.strip()
        }

        for match in TYPE_DECL_RE.finditer(masked):
            name = match.group("name")
            declarations[name].append(Declaration(path, package))
        for match in FUNCTION_DECL_RE.finditer(masked):
            name = match.group(1)
            declarations[name].append(Declaration(path, package))

    failures = 0
    for path in target_files:
        if path not in texts:
            print(f"{path}: error: cannot read", file=sys.stderr)
            failures += 1
            continue

        text = texts[path]
        masked = mask_kotlin(text)
        package = packages.get(path, "")
        file_imports = imports.get(path, set())
        file_wildcards = wildcard_imports.get(path, set())
        file_type_parameters = type_parameters.get(path, set())
        same_package = {
            name
            for name, entries in declarations.items()
            if any(entry.package == package for entry in entries)
        }
        file_declarations = {
            match.group("name") for match in TYPE_DECL_RE.finditer(masked)
        }
        local_types = (
            same_package
            | file_declarations
            | file_type_parameters
            | collect_declared_names(masked)
        )

        for match in re.finditer(r"(?<![\w.])([A-Z][A-Za-z0-9_]*)\b", masked):
            name = match.group(1)
            if len(name) == 1 or name in local_types or name in file_imports:
                continue
            after = masked[match.end() : match.end() + 2]
            if after.startswith(".") or after.startswith("::") or after.startswith(":"):
                continue
            candidates = declarations.get(name)
            if not candidates:
                continue
            if any(entry.package == package for entry in candidates):
                continue
            if any(entry.package in file_wildcards for entry in candidates):
                continue
            first = candidates[0]
            line, column = line_column(text, match.start())
            print(
                Problem(
                    path,
                    line,
                    column,
                    f"missing import for '{name}' (declared in {first.path.relative_to(root)})",
                ).format()
            )
            failures += 1

    if failures:
        print(
            f"Kotlin reference check failed: {failures} problem(s) in {len(target_files)} file(s).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
