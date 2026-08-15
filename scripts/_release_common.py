#!/usr/bin/env python3
"""Shared helpers for verify_changes / prepare_version / publish_version."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PY = ROOT / "app.py"
README = ROOT / "README.md"
MODELS = ROOT / "MODELS.md"
CHANGELOG_DIR = ROOT / "static" / "changelogs"
JS_DIR = ROOT / "static" / "js"
CSS_DIR = ROOT / "static" / "css"
DEFAULT_HANDOFF = ROOT.parent / "引き継ぎ資料.txt"

SYSTEM_VERSION_RE = re.compile(
    r"(app\.config\['SYSTEM_VERSION'\]\s*=\s*')(V\d+\.\d+\.\d+)(')"
)
APP_VERSION_RE = re.compile(
    r"(app\.config\['APP_VERSION'\]\s*=\s*os\.getenv\('APP_VERSION',\s*')([^']+)(')"
)
SYSTEM_VALUE_RE = re.compile(r"^V(\d+)\.(\d+)\.(\d+)$")
APP_VALUE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{3})$")

FORBIDDEN_CHANGELOG_PHRASES = (
    "ユーザーの要望",
    "ユーザー要望により",
    "ユーザーのリクエスト",
    "ユーザー指摘により",
    "ご要望に基づき",
    "というリクエストを受けて",
    "と指示された",
)

ALLOWED_GIT_EXACT = {
    ".env.example",
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "MODELS.md",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "app.py",
    "requirements.txt",
    "worker.py",
}
ALLOWED_GIT_PREFIXES = (
    "deploy/",
    "LICENSES/",
    "scripts/",
    "static/",
    "templates/",
    "tests/",
)
BLOCKED_GIT_PATTERNS = (
    re.compile(r"(^|/)引き継ぎ資料\.txt$"),
    re.compile(r"\.bak", re.IGNORECASE),
    re.compile(r"_bak_", re.IGNORECASE),
    re.compile(r"(^|/)cookie\.txt$"),
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)secret\.key$"),
    re.compile(r"(^|/)instance/"),
    re.compile(r"(^|/)venv/"),
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.log$"),
    re.compile(r"\.pyc$"),
)

CHANGELOG_STUB_MARKERS = (
    "TODO",
    "FIXME",
    "PLACEHOLDER",
    "ここに変更内容",
    "（変更内容を書く）",
    "(write the change here)",
)


class ReleaseError(SystemExit):
    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(code)
        self.message = message


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise ReleaseError(message, code)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def parse_versions(app_source: str | None = None) -> dict[str, str]:
    source = app_source if app_source is not None else read_text(APP_PY)
    system = SYSTEM_VERSION_RE.search(source)
    app_version = APP_VERSION_RE.search(source)
    if not system:
        die("SYSTEM_VERSION not found in app.py")
    if not app_version:
        die("APP_VERSION not found in app.py")
    system_version = system.group(2)
    numeric = SYSTEM_VALUE_RE.fullmatch(system_version)
    if not numeric:
        die(f"Unsupported SYSTEM_VERSION format: {system_version}")
    return {
        "system_version": system_version,
        "system_numeric": f"{numeric.group(1)}.{numeric.group(2)}.{numeric.group(3)}",
        "system_lower": f"v{numeric.group(1)}.{numeric.group(2)}.{numeric.group(3)}",
        "app_version": app_version.group(2),
    }


def next_system_version(current: str) -> str:
    match = SYSTEM_VALUE_RE.fullmatch(current)
    if not match:
        die(f"Unsupported SYSTEM_VERSION format: {current}")
    major, minor, patch = (int(part) for part in match.groups())
    return f"V{major}.{minor}.{patch + 1}"


def next_app_version(current: str, today: dt.date | None = None) -> str:
    match = APP_VALUE_RE.fullmatch(current)
    if not match:
        die(f"Unsupported APP_VERSION format: {current}")
    today = today or dt.date.today()
    today_s = today.isoformat()
    if match.group(1) == today_s:
        return f"{today_s}-{int(match.group(2)) + 1:03d}"
    return f"{today_s}-001"


def versioned_assets(system_lower: str) -> dict[str, Path]:
    return {
        "chat_core_js": JS_DIR / f"chat_core.{system_lower}.js",
        "chat_core_min_js": JS_DIR / f"chat_core.min.{system_lower}.js",
        "chat_custom_css": CSS_DIR / f"chat.custom.{system_lower}.css",
        "chat_custom_min_css": CSS_DIR / f"chat.custom.min.{system_lower}.css",
        "chat_tailwind_css": CSS_DIR / f"chat.tailwind.{system_lower}.css",
    }


def required_source_assets(system_lower: str) -> list[Path]:
    assets = versioned_assets(system_lower)
    return [
        assets["chat_core_js"],
        assets["chat_custom_css"],
        assets["chat_tailwind_css"],
    ]


def required_all_assets(system_lower: str) -> list[Path]:
    return list(versioned_assets(system_lower).values())


def list_chat_core_sources() -> list[Path]:
    return sorted(
        path
        for path in JS_DIR.glob("chat_core.v4.8.*.js")
        if ".min." not in path.name
    )


def changelog_path(system_lower: str, day: dt.date | None = None) -> Path:
    day = day or dt.date.today()
    return CHANGELOG_DIR / f"{day.strftime('%Y%m%d')}_{system_lower}.md"


def find_changelog(system_lower: str) -> Path | None:
    matches = sorted(CHANGELOG_DIR.glob(f"*_{system_lower}.md"))
    return matches[-1] if matches else None


def notes_are_forbidden(notes: str) -> str | None:
    for phrase in FORBIDDEN_CHANGELOG_PHRASES:
        if phrase in notes:
            return phrase
    return None


def changelog_is_complete(text: str, system_version: str) -> str | None:
    stripped = text.strip()
    if len(stripped) < 40:
        return "changelog body is too short"
    if system_version not in stripped and system_version.lower() not in stripped:
        return "changelog does not mention the system version"
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) < 2:
        return "changelog has no body"
    for marker in CHANGELOG_STUB_MARKERS:
        if marker in stripped:
            return f"changelog still contains stub marker: {marker}"
    return None


def normalize_git_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_blocked_git_path(path: str) -> bool:
    normalized = normalize_git_path(path)
    return any(pattern.search(normalized) for pattern in BLOCKED_GIT_PATTERNS)


def is_allowed_git_path(path: str) -> bool:
    normalized = normalize_git_path(path)
    if is_blocked_git_path(normalized):
        return False
    if normalized in ALLOWED_GIT_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in ALLOWED_GIT_PREFIXES)


def classify_git_paths(paths: list[str]) -> dict[str, list[str]]:
    allowed: list[str] = []
    blocked: list[str] = []
    unknown: list[str] = []
    for raw in paths:
        path = normalize_git_path(raw)
        if not path:
            continue
        if is_blocked_git_path(path):
            blocked.append(path)
        elif is_allowed_git_path(path):
            allowed.append(path)
        else:
            unknown.append(path)
    return {"allowed": allowed, "blocked": blocked, "unknown": unknown}


def porcelain_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        if not line:
            continue
        # XY PATH or XY ORIG -> PATH
        payload = line[3:] if len(line) > 3 and line[2] == " " else line
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload)
    return paths


def git_status_paths(repo: Path | None = None) -> list[str]:
    repo = repo or ROOT
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return porcelain_paths(result.stdout.splitlines())


def current_asset_version() -> str | None:
    sources = list_chat_core_sources()
    if len(sources) != 1:
        return None
    match = re.fullmatch(r"chat_core\.(v\d+\.\d+\.\d+)\.js", sources[0].name)
    return match.group(1) if match else None


def check_workspace_assets(versions: dict[str, str] | None = None) -> list[str]:
    versions = versions or parse_versions()
    errors: list[str] = []
    sources = list_chat_core_sources()
    if len(sources) != 1:
        names = ", ".join(path.name for path in sources) or "(none)"
        errors.append(f"expected exactly one chat_core.v4.8.*.js source, found: {names}")
    elif sources[0].name != f"chat_core.{versions['system_lower']}.js":
        errors.append(
            f"chat_core source {sources[0].name} does not match {versions['system_version']}"
        )
    for path in required_all_assets(versions["system_lower"]):
        if not path.is_file():
            errors.append(f"missing required asset: {path.relative_to(ROOT)}")
    changelog = find_changelog(versions["system_lower"])
    if changelog is None:
        errors.append(f"missing changelog for {versions['system_lower']}")
    else:
        reason = changelog_is_complete(read_text(changelog), versions["system_version"])
        if reason:
            errors.append(f"{changelog.name}: {reason}")
    readme = read_text(README)
    models = read_text(MODELS)
    if versions["system_version"] not in readme:
        errors.append("README.md does not mention SYSTEM_VERSION")
    if versions["system_lower"] not in readme:
        errors.append("README.md does not link the current changelog")
    if versions["system_version"] not in models:
        errors.append("MODELS.md does not mention SYSTEM_VERSION")
    if f"chat_core.{versions['system_lower']}.js" not in models:
        errors.append("MODELS.md does not point at the current chat_core source")
    return errors


def render_changelog(system_version: str, notes: str, day: dt.date | None = None) -> str:
    day = day or dt.date.today()
    body = notes.strip()
    if not body.endswith("\n"):
        body += "\n"
    return f"# 更新履歴 - {system_version} ({day.isoformat()})\n\n{body}"


def render_handoff_block(
    versions: dict[str, str],
    notes: str,
    old_system: str,
    extra: str = "",
    day: dt.date | None = None,
) -> str:
    day = day or dt.date.today()
    summary = " ".join(notes.strip().split())
    extra_text = extra.strip()
    line = (
        f"**最終更新:** {day.isoformat()}\n"
        f"**システム状態:** **正常稼働中 ({versions['system_version']})**\n"
        f"**バージョン:** {versions['system_version']}\n"
        f"**特記事項:** {summary} "
        f"`APP_VERSION={versions['app_version']}` / "
        f"`SYSTEM_VERSION={versions['system_version']}`。"
        f"{old_system}資産を削除。"
    )
    if extra_text:
        line += f" {extra_text}"
    return line + "\n\n"


def update_handoff(path: Path, block: str) -> None:
    previous = read_text(path) if path.is_file() else ""
    write_text(path, block + previous)


def apply_prepare(
    notes: str,
    handoff_notes: str | None = None,
    dry_run: bool = False,
    today: dt.date | None = None,
    handoff_path: Path | None = None,
) -> dict[str, object]:
    today = today or dt.date.today()
    notes = notes.strip()
    if len(notes) < 20:
        die("changelog notes are too short")
    forbidden = notes_are_forbidden(notes)
    if forbidden:
        die(f"changelog notes contain a forbidden phrase: {forbidden}")

    current = parse_versions()
    asset_version = current_asset_version()
    if asset_version != current["system_lower"]:
        die(
            "app.py SYSTEM_VERSION and the current chat_core source do not match "
            f"({current['system_lower']} vs {asset_version})"
        )

    new_system = next_system_version(current["system_version"])
    new_app = next_app_version(current["app_version"], today)
    new_versions = {
        "system_version": new_system,
        "system_numeric": new_system[1:],
        "system_lower": new_system.lower(),
        "app_version": new_app,
    }
    old_assets = versioned_assets(current["system_lower"])
    new_assets = versioned_assets(new_versions["system_lower"])
    new_changelog = changelog_path(new_versions["system_lower"], today)
    handoff = handoff_path or Path(os.environ.get("HANDOFF_FILE", DEFAULT_HANDOFF))

    for path in required_source_assets(current["system_lower"]):
        if not path.is_file():
            die(f"missing current source asset: {path.relative_to(ROOT)}")
    for path in new_assets.values():
        if path.exists():
            die(f"refusing to overwrite existing asset: {path.relative_to(ROOT)}")
    if new_changelog.exists():
        die(f"changelog already exists: {new_changelog.name}")

    plan = {
        "old_system_version": current["system_version"],
        "new_system_version": new_system,
        "old_app_version": current["app_version"],
        "new_app_version": new_app,
        "copy": [
            [
                str(old_assets["chat_core_js"].relative_to(ROOT)),
                str(new_assets["chat_core_js"].relative_to(ROOT)),
            ],
            [
                str(old_assets["chat_custom_css"].relative_to(ROOT)),
                str(new_assets["chat_custom_css"].relative_to(ROOT)),
            ],
            [
                str(old_assets["chat_tailwind_css"].relative_to(ROOT)),
                str(new_assets["chat_tailwind_css"].relative_to(ROOT)),
            ],
        ],
        "delete": [
            str(path.relative_to(ROOT))
            for path in old_assets.values()
            if path.exists()
        ],
        "changelog": str(new_changelog.relative_to(ROOT)),
        "handoff": str(handoff),
    }
    if dry_run:
        return plan

    new_assets["chat_core_js"].write_bytes(old_assets["chat_core_js"].read_bytes())
    new_assets["chat_custom_css"].write_bytes(old_assets["chat_custom_css"].read_bytes())
    new_assets["chat_tailwind_css"].write_bytes(old_assets["chat_tailwind_css"].read_bytes())

    app_source = read_text(APP_PY)
    app_source = APP_VERSION_RE.sub(rf"\g<1>{new_app}\3", app_source, count=1)
    app_source = SYSTEM_VERSION_RE.sub(rf"\g<1>{new_system}\3", app_source, count=1)
    write_text(APP_PY, app_source)

    readme = read_text(README)
    readme = readme.replace(current["system_version"], new_system)
    readme = readme.replace(
        f"{find_changelog_date_prefix(current['system_lower'])}_{current['system_lower']}.md",
        f"{today.strftime('%Y%m%d')}_{new_versions['system_lower']}.md",
    )
    # If the README still points at the old changelog filename, replace the
    # versioned file name even when the date prefix differs.
    readme = readme.replace(
        f"_{current['system_lower']}.md",
        f"_{new_versions['system_lower']}.md",
    )
    write_text(README, readme)

    models = read_text(MODELS)
    models = models.replace(current["system_version"], new_system)
    models = models.replace(current["system_lower"], new_versions["system_lower"])
    models = re.sub(
        r"(\*最終更新:\s*)\d{4}-\d{2}-\d{2}(\s+\()",
        rf"\g<1>{today.isoformat()}\2",
        models,
        count=1,
    )
    write_text(MODELS, models)

    write_text(new_changelog, render_changelog(new_system, notes, today))

    for path in old_assets.values():
        if path.exists():
            path.unlink()

    if handoff.parent.is_dir():
        update_handoff(
            handoff,
            render_handoff_block(
                new_versions,
                handoff_notes or notes,
                current["system_version"],
                day=today,
            ),
        )
    return plan


def find_changelog_date_prefix(system_lower: str) -> str:
    existing = find_changelog(system_lower)
    if existing:
        return existing.name.split("_", 1)[0]
    return dt.date.today().strftime("%Y%m%d")


def append_handoff_publish_result(
    versions: dict[str, str],
    commit: str,
    tag: str,
    handoff_path: Path | None = None,
) -> None:
    path = handoff_path or Path(os.environ.get("HANDOFF_FILE", DEFAULT_HANDOFF))
    if not path.is_file():
        return
    text = read_text(path)
    marker = f"`SYSTEM_VERSION={versions['system_version']}`。"
    addition = (
        f"全回帰テスト成功。リリースコミット`{commit}`、"
        f"注釈付きタグ`{tag}`をGitHubへpush済み。"
    )
    if addition in text:
        return
    if marker in text:
        write_text(path, text.replace(marker, marker + addition, 1))
        return
    update_handoff(
        path,
        render_handoff_block(versions, "版を公開しました。", versions["system_version"], addition),
    )


def print_plan(plan: dict[str, object]) -> None:
    print(f"current: {plan['old_system_version']} / {plan['old_app_version']}")
    print(f"next:    {plan['new_system_version']} / {plan['new_app_version']}")
    print("copy:")
    for src, dest in plan["copy"]:
        print(f"  {src} -> {dest}")
    print("delete:")
    for path in plan["delete"]:
        print(f"  {path}")
    print(f"changelog: {plan['changelog']}")
    print(f"handoff:   {plan['handoff']}")


def cmd_versions(_args: argparse.Namespace) -> int:
    print(json.dumps(parse_versions(), ensure_ascii=False, indent=2))
    return 0


def cmd_check_assets(_args: argparse.Namespace) -> int:
    errors = check_workspace_assets()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    versions = parse_versions()
    print(f"assets ok: {versions['system_version']} / {versions['app_version']}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    notes = args.notes
    if args.notes_file:
        notes = read_text(Path(args.notes_file))
    if not notes:
        die("provide --notes or --notes-file")
    plan = apply_prepare(
        notes=notes,
        handoff_notes=args.handoff_notes,
        dry_run=args.dry_run,
    )
    print_plan(plan)
    if args.dry_run:
        print("dry-run: no files were changed")
    return 0


def cmd_classify_git(_args: argparse.Namespace) -> int:
    classified = classify_git_paths(git_status_paths())
    print(json.dumps(classified, ensure_ascii=False, indent=2))
    if classified["blocked"] or classified["unknown"]:
        return 1
    return 0


def cmd_check_notes(args: argparse.Namespace) -> int:
    notes = args.notes
    if args.notes_file:
        notes = read_text(Path(args.notes_file))
    if not notes or len(notes.strip()) < 20:
        die("changelog notes are too short")
    forbidden = notes_are_forbidden(notes)
    if forbidden:
        die(f"changelog notes contain a forbidden phrase: {forbidden}")
    print("notes ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("versions").set_defaults(func=cmd_versions)
    sub.add_parser("check-assets").set_defaults(func=cmd_check_assets)
    sub.add_parser("classify-git").set_defaults(func=cmd_classify_git)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--notes", default="")
    prepare.add_argument("--notes-file")
    prepare.add_argument("--handoff-notes")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    notes = sub.add_parser("check-notes")
    notes.add_argument("--notes", default="")
    notes.add_argument("--notes-file")
    notes.set_defaults(func=cmd_check_notes)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        raise SystemExit(exc.code)
