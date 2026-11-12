#!/usr/bin/env bash
# Check syntax, required versioned assets, and regression tests.
# This does not change files or touch the running services.
#
# Output is deliberately small: by default the per-check lines and the full
# test output go to $VERIFY_LOG, stdout gets one summary line on success and at
# most the last 20 log lines on failure.  --verbose keeps the per-check output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

SKIP_TESTS=0
LIVE=0
VERBOSE=0
PYTEST_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/verify_changes.sh [--skip-tests] [--live] [--verbose] [--] [pytest-args...]

Checks the current tree:
  - required versioned JS/CSS assets exist and match SYSTEM_VERSION
  - changelog / README / MODELS mention the current version
  - JavaScript and Python syntax
  - files under static/ are readable by the web server
  - source size report (largest files, budget overruns)
  - extract before extend (budget-over files must not grow by 150+ net lines)
  - regression tests (unless --skip-tests)

--live also requests the running site after a deploy. Use it only after
services have been restarted.
Default output is one PASS line on success (full detail in the log) and the
last 20 log lines on failure. --verbose prints each check as it runs.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tests) SKIP_TESTS=1; shift ;;
        --live) LIVE=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS+=("$@"); break ;;
        *) PYTEST_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$(dirname "$VERIFY_LOG")"
: > "$VERIFY_LOG"

TEST_DB_FILE=""
on_exit() {
    local status=$?
    if [[ -n "$TEST_DB_FILE" ]]; then
        rm -f "$TEST_DB_FILE"
    fi
    # A failed check already wrote its reason to the log.  Show only the end of
    # it so an agent does not have to read the whole test output.
    if [[ "$status" -ne 0 ]]; then
        printf -- '----- verify failed; last 20 log lines -----\n' >&3
        tail -n 20 "$VERIFY_LOG" >&3 2>/dev/null || true
        printf 'full log: %s\n' "$VERIFY_LOG" >&3
    fi
}
trap on_exit EXIT

# Keep the terminal quiet unless --verbose; checks and pytest keep writing to
# the log, and the summary is printed to fd 3.
if [[ "$VERBOSE" -eq 0 ]]; then
    exec 3>&1 1>>"$VERIFY_LOG" 2>&1
else
    exec 3>&1
fi

run_logged() {
    local label="$1"
    shift
    if "$@" >>"$VERIFY_LOG" 2>&1; then
        ok "$label"
    else
        die "$label failed (full log: $VERIFY_LOG)"
    fi
}

versions_json="$(run_common versions)"
SYSTEM_VERSION="$(printf '%s' "$versions_json" | json_get system_version)"
APP_VERSION="$(printf '%s' "$versions_json" | json_get app_version)"
info "verify $SYSTEM_VERSION / $APP_VERSION"

run_logged "versioned assets and docs" run_common check-assets

# chat_core is edited as ordered parts; warn (non-blocking) when a part grows
# past the readable size limit so it can be re-split and the README updated.
info "chat_core parts size"
run_common check-parts || warn "chat_core parts size check failed"

# Report the largest sources and anything past the line budgets from
# ~/docs/context_cost_reduction_plan.md.  Informational: never fails a release.
info "source size report"
run_common size-report >>"$VERIFY_LOG" 2>&1 || warn "source size report failed"

# Extract before extend: a file already past its warn budget may not grow by
# 150+ net lines in one change.  Blocking, unlike the warning-only report above.
run_logged "extract before extend" run_common check-extract

JS_SOURCES=(
    static/js/activity_log.js
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
# The browser loads chat_core as several files; each must parse on its own.
mapfile -t CHAT_CORE_BUNDLES < <(run_common chat-core-bundles)
for bundle in "${CHAT_CORE_BUNDLES[@]}"; do
    JS_SOURCES+=("static/$bundle")
done

if ! command -v node >/dev/null 2>&1; then
    die "node is required for JavaScript syntax checks"
fi
for src in "${JS_SOURCES[@]}"; do
    run_logged "node --check $src" node --check "$src"
done

PY_SOURCES=(app.py worker.py scripts/_release_common.py)
while IFS= read -r -d '' py; do
    PY_SOURCES+=("$py")
done < <(find server -name '*.py' -print0 | sort -z)
run_logged "python syntax" "$PYTHON" -m py_compile "${PY_SOURCES[@]}"

# Android generated assets are derived from Web sources. Check them here so a
# stale serverless-defaults/catalog/icon file fails with a direct message before
# the slower regression suite or version update.
info "Android generated assets"
run_logged "android serverless defaults in sync" "$PYTHON" android/ci/sync-serverless-defaults.py --check
run_logged "android model catalog in sync" node android/ci/sync-web-catalog.mjs --check
run_logged "android icon drawables in sync" "$PYTHON" android/ci/sync-web-icons.py --check

# gunicorn can read a private file, but the front web server cannot, so a live check
# against gunicorn would pass while the public site answers 403.
unreadable="$(find static \( -type f ! -perm -004 -o -type d ! -perm -005 \) -print | head -n 5 || true)"
if [[ -n "$unreadable" ]]; then
    die "static files not readable by the web server (fix with chmod a+r): $(printf '%s' "$unreadable" | tr '\n' ' ')"
fi
ok "static files readable by the web server"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    # Never inherit a live MySQL URL into pytest. setdefault() in tests cannot
    # override an already-exported DATABASE_URL, and db.drop_all() would wipe it.
    # Most tests recreate every table in setUp; on the disk-backed /tmp each
    # reset costs ~1 s of fsync, so keep the throwaway database on tmpfs.
    TEST_DB_DIR=/tmp
    if [[ -d /dev/shm && -w /dev/shm ]]; then
        TEST_DB_DIR=/dev/shm
    fi
    TEST_DB_FILE="$TEST_DB_DIR/ai-chat-verify.db"
    if [[ -n "${AI_CHAT_TEST_DATABASE_URL:-}" ]]; then
        TEST_DB_FILE=""
    fi
    export DATABASE_URL="${AI_CHAT_TEST_DATABASE_URL:-sqlite:///$TEST_DB_FILE}"
    export RUN_SCHEMA_MIGRATIONS="${RUN_SCHEMA_MIGRATIONS:-0}"
    info "running pytest (quiet; full output in $VERIFY_LOG)"
    if ! "$PYTHON" -m pytest -q --tb=line "${PYTEST_ARGS[@]}" >>"$VERIFY_LOG" 2>&1; then
        die "pytest failed (full log: $VERIFY_LOG)"
    fi
    summary="$(grep -E '[0-9]+ passed' "$VERIFY_LOG" | tail -n 1 || true)"
    ok "pytest ${summary:-passed}"
else
    warn "tests skipped"
    summary="tests skipped"
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
        "${CHAT_CORE_BUNDLES[@]}" \
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
printf '[ok] verify %s / %s passed (%s); log: %s\n' \
    "$SYSTEM_VERSION" "$APP_VERSION" "$summary" "$VERIFY_LOG" >&3
exit 0
