#!/usr/bin/env bash
# Re-split the combined chat_core source into ordered parts under
# static/js/chat_core_parts/ and regenerate the part README base table.
#
# Run this when scripts/verify_changes.sh warns that a chat_core part exceeds
# CHAT_CORE_PART_MAX_LINES (2000).  It:
#   1. re-splits the current combined source into parts of ~1700 lines
#   2. regenerates static/js/chat_core_parts/README.md (line ranges are exact;
#      the "概要" descriptions must be reviewed/filled in afterwards)
#   3. rebuilds the combined source and minified assets via build_frontend.sh
#
# After running, review and update the README descriptions, then follow the
# normal version flow (prepare_version.sh / publish_version.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION_FILE="$(ls -1 static/js/chat_core.v4.8.*.js | grep -v '\.min\.' | sort | tail -n 1)"
if [[ -z "$VERSION_FILE" ]]; then
  echo "No chat_core.v4.8.*.js source file found." >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  echo "node is required to re-split chat_core parts." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Installing esbuild (cached by npm) ..."
npm install --prefix "$TMP" --no-audit --no-fund --no-save esbuild@0.25.9 >/dev/null 2>&1

node "$ROOT/scripts/_rebuild_chat_core_parts.js" \
  "$ROOT/$VERSION_FILE" \
  "$ROOT/static/js/chat_core_parts" \
  "$TMP/node_modules"

echo "Rebuilding combined source and minified assets ..."
"$ROOT/scripts/build_frontend.sh"
