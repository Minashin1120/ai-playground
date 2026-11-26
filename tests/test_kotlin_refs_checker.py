"""Project-import checks for Kotlin files (Android CI cannot run locally)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kotlin_refs = load("check_kotlin_refs")


def test_mask_removes_comments_and_strings() -> None:
    text = '''package demo
/** hidden comment */
val text = "hidden string"
val raw = """hidden raw"""
val template = "${hiddenTemplate}"
// hidden line
class Visible
'''
    masked = kotlin_refs.mask_kotlin(text)
    for hidden in ("hidden comment", "hidden string", "hidden raw", "hiddenTemplate", "hidden line"):
        assert hidden not in masked
    assert "Visible" in masked


def test_collect_declared_names_includes_enum_entries_and_functions() -> None:
    text = '''package demo
class Holder {
    enum class Kind { First, Second("x") }
    fun Capitalized() {}
    val value = 1
}
'''
    names = kotlin_refs.collect_declared_names(kotlin_refs.mask_kotlin(text))
    assert {"Holder", "Kind", "First", "Second", "Capitalized", "value"} <= names


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package_a = repo / "android/app/src/main/java/demo/a"
    package_b = repo / "android/app/src/main/java/demo/b"
    package_a.mkdir(parents=True)
    package_b.mkdir(parents=True)
    (package_a / "Foo.kt").write_text("package demo.a\n\nclass Foo\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    return repo


def run_checker(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_kotlin_refs.py")],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_project_import_is_reported(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    bar = repo / "android/app/src/main/java/demo/b/Bar.kt"
    bar.write_text("package demo.b\n\nfun bar(value: Foo) {}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)

    result = run_checker(repo)
    assert result.returncode == 1
    assert "missing import for 'Foo'" in result.stdout


def test_direct_and_star_imports_are_accepted(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    bar = repo / "android/app/src/main/java/demo/b/Bar.kt"
    bar.write_text(
        "package demo.b\n\nimport demo.a.Foo\n\nfun bar(value: Foo) {}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    assert run_checker(repo).returncode == 0

    bar.write_text(
        "package demo.b\n\nimport demo.a.*\n\nfun bar(value: Foo) {}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "star"], cwd=repo, check=True)
    assert run_checker(repo).returncode == 0
