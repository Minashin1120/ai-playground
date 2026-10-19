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

# Commit trailers that Android CI (.github/workflows/android.yml) reads to leave out checks.
ANDROID_SKIP_TESTS_TRAILER="Android-Skip-Unit-Tests: true"
ANDROID_SKIP_LINT_TRAILER="Android-Skip-Lint: true"

# Appends the requested skip trailers to the COMMIT_ARGS array.
add_android_skip_trailers() {
    if [[ "${SKIP_ANDROID_TESTS:-0}" -eq 1 ]]; then
        COMMIT_ARGS+=(--trailer "$ANDROID_SKIP_TESTS_TRAILER")
    fi
    if [[ "${SKIP_ANDROID_LINT:-0}" -eq 1 ]]; then
        COMMIT_ARGS+=(--trailer "$ANDROID_SKIP_LINT_TRAILER")
    fi
}

# Prints which Android CI checks will run for the plan.
print_android_checks() {
    local tests="run" lint="run"
    [[ "${SKIP_ANDROID_TESTS:-0}" -eq 1 ]] && tests="skipped"
    [[ "${SKIP_ANDROID_LINT:-0}" -eq 1 ]] && lint="skipped"
    echo "  Android unit tests: $tests"
    echo "  Android lint: $lint"
}
