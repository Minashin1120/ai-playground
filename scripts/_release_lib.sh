#!/usr/bin/env bash
# Shared bash helpers for verify_changes / prepare_version / publish_version.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/venv/bin/python" ]]; then
    PYTHON="$ROOT/venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

COMMON_PY="$ROOT/scripts/_release_common.py"
VERIFY_LOG="${VERIFY_LOG:-$ROOT/scripts/verify.last.log}"
PUBLISH_HOST="${PUBLISH_HOST:-ai.minashin1120.com}"
PUBLISH_LOCAL="${PUBLISH_LOCAL:-http://127.0.0.1:3111}"

ok() { printf '[ok] %s\n' "$*"; }
info() { printf '==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

run_common() {
    "$PYTHON" "$COMMON_PY" "$@"
}

json_get() {
    local key="$1"
    "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key"
}
