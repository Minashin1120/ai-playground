#!/usr/bin/env bash
# Print only the release identity needed to edit files: SYSTEM_VERSION,
# APP_VERSION, and the _SERVER_PARTS load order.  Reads files; does not import
# app.py or touch the services.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

"$PYTHON" - <<'PY'
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import _release_common as release

versions = release.parse_versions()
print(f"SYSTEM_VERSION={versions['system_version']}")
print(f"APP_VERSION={versions['app_version']}")

tree = ast.parse(release.APP_PY.read_text(encoding="utf-8"))
parts = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        getattr(target, "id", None) == "_SERVER_PARTS" for target in node.targets
    ):
        parts = ast.literal_eval(node.value)
        break
if parts is None:
    sys.exit("_SERVER_PARTS not found in app.py")

print(f"_SERVER_PARTS ({len(parts)} files, load order):")
for index, name in enumerate(parts, 1):
    print(f"{index:4d}. {name}")
PY
