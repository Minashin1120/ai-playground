#!/usr/bin/env bash
# Regenerate Android assets that are derived from Web sources.
#
# Called before prepare_version.sh / record_changes.sh so a generated-file
# omission does not stop the release. verify_changes.sh still runs the same
# generators with --check and fails only when this step was skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

info "syncing Android generated assets from Web sources"
"$PYTHON" android/ci/sync-serverless-defaults.py
node android/ci/sync-web-catalog.mjs
"$PYTHON" android/ci/sync-web-icons.py
ok "Android generated assets synced"
