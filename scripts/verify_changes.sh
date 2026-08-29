#!/usr/bin/env bash
# Check syntax, required versioned assets, and regression tests.
# This does not change files or touch the running services.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

SKIP_TESTS=0
LIVE=0
PYTEST_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/verify_changes.sh [--skip-tests] [--live] [--] [pytest-args...]

Checks the current tree:
  - required versioned JS/CSS assets exist and match SYSTEM_VERSION
  - changelog / README / MODELS mention the current version
  - JavaScript and Python syntax
  - regression tests (unless --skip-tests)

--live also requests the running site after a deploy. Use it only after
services have been restarted. Default output is a short PASS/FAIL summary.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tests) SKIP_TESTS=1; shift ;;
        --live) LIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS+=("$@"); break ;;
        *) PYTEST_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$(dirname "$VERIFY_LOG")"
: > "$VERIFY_LOG"

run_logged() {
    local label="$1"
    shift
    if "$@" >>"$VERIFY_LOG" 2>&1; then
        ok "$label"
    else
        echo "----- last 40 log lines -----" >&2
        tail -n 40 "$VERIFY_LOG" >&2 || true
        die "$label failed (full log: $VERIFY_LOG)"
    fi
}

info "verify $(run_common versions | json_get system_version) / $(run_common versions | json_get app_version)"

run_logged "versioned assets and docs" run_common check-assets

JS_SOURCES=(
    static/js/progress_spinner.js
    static/js/connection_monitor.js
    static/js/pwa_install.js
    static/js/landing_demo.js
)
mapfile -t CHAT_CORE < <(ls -1 static/js/chat_core.v4.8.*.js 2>/dev/null | grep -v '\.min\.' || true)
if [[ ${#CHAT_CORE[@]} -ne 1 ]]; then
    die "expected exactly one chat_core.v4.8.*.js source"
fi
JS_SOURCES+=("${CHAT_CORE[0]}")

if ! command -v node >/dev/null 2>&1; then
    die "node is required for JavaScript syntax checks"
fi
for src in "${JS_SOURCES[@]}"; do
    run_logged "node --check $src" node --check "$src"
done

PY_SOURCES=(app.py worker.py scripts/_release_common.py)
run_logged "python syntax" "$PYTHON" -m py_compile "${PY_SOURCES[@]}"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    # Never inherit a live MySQL URL into pytest. setdefault() in tests cannot
    # override an already-exported DATABASE_URL, and db.drop_all() would wipe it.
    export DATABASE_URL="${AI_CHAT_TEST_DATABASE_URL:-sqlite:////tmp/ai-chat-verify.db}"
    export RUN_SCHEMA_MIGRATIONS="${RUN_SCHEMA_MIGRATIONS:-0}"
    info "running pytest (quiet; full output in $VERIFY_LOG)"
    if ! "$PYTHON" -m pytest -q --tb=line "${PYTEST_ARGS[@]}" >>"$VERIFY_LOG" 2>&1; then
        echo "----- pytest failures -----" >&2
        grep -E 'FAILED|ERROR|failed' "$VERIFY_LOG" | tail -n 50 >&2 || tail -n 40 "$VERIFY_LOG" >&2
        die "pytest failed (full log: $VERIFY_LOG)"
    fi
    summary="$(grep -E '[0-9]+ passed' "$VERIFY_LOG" | tail -n 1 || true)"
    ok "pytest ${summary:-passed}"
else
    warn "tests skipped"
fi

if [[ "$LIVE" -eq 1 ]]; then
    versions_json="$(run_common versions)"
    system_lower="$(printf '%s' "$versions_json" | json_get system_lower)"
    app_version="$(printf '%s' "$versions_json" | json_get app_version)"
    code="$(curl -sS -o /tmp/ai-playground-version.json -w '%{http_code}' \
        -H "Host: $PUBLISH_HOST" "$PUBLISH_LOCAL/api/version" || true)"
    if [[ "$code" != "200" ]]; then
        die "live /api/version returned HTTP $code"
    fi
    if ! grep -Fq "\"$app_version\"" /tmp/ai-playground-version.json; then
        die "live /api/version is not $app_version"
    fi
    ok "live /api/version $app_version"
    for rel in \
        "js/chat_core.${system_lower}.js" \
        "js/chat_core.min.${system_lower}.js" \
        "css/chat.custom.${system_lower}.css" \
        "css/chat.custom.min.${system_lower}.css" \
        "css/chat.tailwind.${system_lower}.css"
    do
        live_code="$(curl -sS -o /dev/null -w '%{http_code}' \
            -H "Host: $PUBLISH_HOST" "$PUBLISH_LOCAL/static/$rel" || true)"
        # Static files are usually served by the front proxy, not gunicorn.
        if [[ "$live_code" != "200" ]]; then
            live_code="$(curl -sS -o /dev/null -w '%{http_code}' \
                -H "Host: $PUBLISH_HOST" "https://$PUBLISH_HOST/static/$rel" || true)"
        fi
        if [[ "$live_code" != "200" ]]; then
            die "live $rel returned HTTP $live_code"
        fi
        ok "live $rel 200"
    done
fi

info "ALL CHECKS PASSED"
exit 0
