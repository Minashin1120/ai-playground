#!/usr/bin/env bash
# Print only "# region: <name>" / "// region: <name>" markers with their line
# ranges, so a large file can be read by region instead of in full.
#
# Usage: scripts/show_regions.sh [file-or-directory ...]
# Defaults to the source trees that tend to grow large.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(app.py server tests android/app/src static/js/chat_core_parts)
fi

"$PYTHON" - "${targets[@]}" <<'PY'
import re
import sys
from pathlib import Path

REGION_RE = re.compile(r"^\s*(?:#|//)\s*region:\s*(.+?)\s*$")
SUFFIXES = {".py", ".js", ".mjs", ".kt", ".sh", ".html", ".css"}
SKIP_DIRS = {"node_modules", "build", ".gradle", "__pycache__", "venv"}


def source_files(target: Path):
    if target.is_file():
        yield target
        return
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if SKIP_DIRS.intersection(path.parts):
            continue
        yield path


found = 0
for raw in sys.argv[1:]:
    target = Path(raw)
    if not target.exists():
        print(f"{raw}: not found", file=sys.stderr)
        continue
    for path in source_files(target):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        markers = [(index, match.group(1)) for index, line in enumerate(lines, 1) if (match := REGION_RE.match(line))]
        if not markers:
            continue
        found += 1
        print(f"{path.as_posix()}")
        for position, (start, name) in enumerate(markers):
            end = markers[position + 1][0] - 1 if position + 1 < len(markers) else len(lines)
            print(f"  {start:6d}-{end:<6d} {name}")

if not found:
    print("no region markers found.")
    print('Add "# region: <name>" (or "// region: <name>") where a large file should be read in pieces.')
PY
