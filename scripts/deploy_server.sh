#!/usr/bin/env bash
# Deploy a server-only fix without advancing the Web version.
# Review the plan first. Nothing starts until --confirm SERVER is given.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

MESSAGE=""
CONFIRM=""
SKIP_TESTS=0

usage() {
    cat <<'EOF'
Usage: scripts/deploy_server.sh --message "..."
       scripts/deploy_server.sh --message "..." --confirm SERVER [--skip-tests]

Without --confirm, prints the files that would be recorded and exits 2.
No services or repository state are changed.

Only server/, tests/, worker.py and deploy/*.md may be changed. app.py,
templates/, static/ and android/ stop the plan (use the Web or Android route).

With --confirm SERVER:
  1. run verify_changes.sh (regression tests unless --skip-tests)
  2. restart gunicorn and all workers; stop immediately if restart fails
  3. confirm public/local URLs
  4. commit only the reviewed files and push the branch

It does not advance SYSTEM_VERSION/APP_VERSION, write a public changelog,
purge CDN caches, or create a tag.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --message) MESSAGE="${2:-}"; shift 2 ;;
        --confirm) CONFIRM="${2:-}"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ ${#MESSAGE} -ge 8 ]] || die "provide a commit message of at least 8 characters"
command -v git >/dev/null 2>&1 || die "git is required"

if [[ -d "$ROOT/.git" ]]; then
    GIT_DIR="$ROOT"
elif [[ -d "$ROOT/../.git" ]]; then
    GIT_DIR="$(cd "$ROOT/.." && pwd)"
else
    die "not inside a git repository"
fi
git_in() { git -C "$GIT_DIR" "$@"; }

BRANCH="$(git_in rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" == "main" ]] || die "refusing to deploy from branch '$BRANCH' (main only)"

SYSTEM_VERSION="$(run_common versions | json_get system_version)"
STATUS_JSON="$(run_common classify-record --target server || true)"
[[ -n "$STATUS_JSON" ]] || die "failed to classify repository changes"

echo "==> server deploy plan"
echo "  branch:   $BRANCH"
echo "  version:  $SYSTEM_VERSION (unchanged)"
echo "  message:  $MESSAGE"
echo "  tests:    $([[ "$SKIP_TESTS" -eq 1 ]] && echo skipped || echo full)"
printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json, sys
data = json.load(sys.stdin)
print("  files:")
for path in data.get("allowed") or []:
    print(f"    {path}")
for key, label in (("blocked", "blocked"), ("unknown", "outside allowlist"),
                   ("outside_target", "outside server-only scope")):
    paths = data.get(key) or []
    if paths:
        print(f"  {label}:")
        for path in paths:
            print(f"    {path}")
'

blocked_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("blocked") or []) + len(d.get("unknown") or []) + len(d.get("outside_target") or []))')"
allowed_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("allowed") or []))')"
[[ "$blocked_count" == "0" ]] \
    || die "the tree contains changes outside the server-only scope (use prepare_version.sh / publish_version.sh)"
[[ "$allowed_count" != "0" ]] || die "there are no server-only changes"

if [[ -z "$CONFIRM" ]]; then
    info "review the plan, then rerun with --confirm SERVER"
    exit 2
fi
[[ "$CONFIRM" == "SERVER" ]] || die "--confirm must be SERVER"

info "checking the tree"
if [[ "$SKIP_TESTS" -eq 1 ]]; then
    "$ROOT/scripts/verify_changes.sh" --skip-tests
else
    "$ROOT/scripts/verify_changes.sh"
fi

dump_restart_logs() {
    echo "----- service status -----" >&2
    for unit in ai-chat.service ai-chat-worker@1.service ai-chat-worker@2.service; do
        printf '  %s: %s\n' "$unit" "$(systemctl is-active "$unit" 2>/dev/null || echo unknown)" >&2
    done
    echo "----- ai-chat.service log -----" >&2
    journalctl -u ai-chat.service -n 80 --no-pager >&2 || true
    echo "----- worker logs -----" >&2
    journalctl -u 'ai-chat-worker@*.service' -n 80 --no-pager >&2 || true
}

info "restarting services"
if ! "$ROOT/scripts/restart_services.sh"; then
    dump_restart_logs
    die "restart failed; nothing was recorded"
fi
ok "restart succeeded"

info "checking live URLs"
if ! "$ROOT/scripts/verify_changes.sh" --skip-tests --live; then
    die "live URL check failed; nothing was recorded"
fi

mapfile -t RECORD_PATHS < <(printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json, sys
for path in json.load(sys.stdin).get("allowed") or []:
    print(path)
')
for path in "${RECORD_PATHS[@]}"; do
    [[ "$path" != *".."* ]] || die "refusing suspicious path: $path"
    git_in add -- "$path"
done

STAGED="$(git_in diff --cached --name-only)"
[[ -n "$STAGED" ]] || die "nothing is staged"
while IFS= read -r staged; do
    printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json, sys
allowed = set(json.load(sys.stdin).get("allowed") or [])
raise SystemExit(0 if sys.argv[1] in allowed else 1)
' "$staged" || {
        git_in reset -q HEAD -- "$staged" || true
        die "refusing staged path outside the reviewed plan: $staged"
    }
done <<< "$STAGED"

git_in commit -m "$MESSAGE"
COMMIT="$(git_in rev-parse --short HEAD)"
if ! git_in push origin HEAD; then
    die "deployed, but failed to push; local commit $COMMIT exists"
fi
info "SERVER DEPLOY COMPLETE: $SYSTEM_VERSION unchanged ($COMMIT)"
