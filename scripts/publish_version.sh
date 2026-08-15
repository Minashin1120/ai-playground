#!/usr/bin/env bash
# Apply the current prepared version to running services and record it.
# Review the plan first. Nothing starts until --confirm matches SYSTEM_VERSION.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

MESSAGE=""
CONFIRM=""
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/publish_version.sh --message "..." --confirm V4.8.xxx
       scripts/publish_version.sh --message "..." --dry-run

Without --confirm, prints the plan and exits 2. No services or repository
state are changed.

With --confirm matching the current SYSTEM_VERSION:
  1. re-check the tree and the files that would be recorded
  2. restart gunicorn and all workers; stop immediately if restart fails
  3. purge the CDN cache only after a successful restart
  4. confirm public/local URLs
  5. record only an allowlisted set of files, tag that exact revision,
     and send it to the configured remote

--message is required in advance. The script never stages ignored files
and never uses a blanket add of the whole tree.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --message) MESSAGE="${2:-}"; shift 2 ;;
        --confirm) CONFIRM="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [[ -z "$MESSAGE" ]]; then
    die "provide --message before running (the repository summary is not invented)"
fi
if [[ "${#MESSAGE}" -lt 8 ]]; then
    die "commit message is too short"
fi

if ! command -v git >/dev/null 2>&1; then
    die "git is required"
fi
if [[ ! -d "$ROOT/.git" && ! -d "$ROOT/../.git" ]]; then
    die "not inside a git repository"
fi
if [[ -d "$ROOT/.git" ]]; then
    GIT_DIR="$ROOT"
else
    GIT_DIR="$(cd "$ROOT/.." && pwd)"
fi

git_in() {
    git -C "$GIT_DIR" "$@"
}

if [[ "$(git_in rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]]; then
    die "not inside a git repository"
fi

BRANCH="$(git_in rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    die "refusing to publish from branch '$BRANCH' (main only)"
fi

VERSIONS_JSON="$(run_common versions)"
SYSTEM_VERSION="$(printf '%s' "$VERSIONS_JSON" | json_get system_version)"
SYSTEM_LOWER="$(printf '%s' "$VERSIONS_JSON" | json_get system_lower)"
APP_VERSION="$(printf '%s' "$VERSIONS_JSON" | json_get app_version)"
TAG="$SYSTEM_LOWER"

STATUS_JSON="$(run_common classify-git || true)"
# classify-git returns 1 when blocked/unknown paths exist; still print the plan.
if [[ -z "$STATUS_JSON" ]]; then
    die "failed to classify repository changes"
fi

print_plan() {
    info "publish plan"
    echo "  branch:   $BRANCH"
    echo "  version:  $SYSTEM_VERSION / $APP_VERSION"
    echo "  tag:      $TAG"
    echo "  message:  $MESSAGE"
    echo
    echo "  files that would be recorded:"
    printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json,sys
data=json.load(sys.stdin)
paths=data.get("allowed") or []
if not paths:
    print("    (none)")
for path in paths:
    print(f"    {path}")
blocked=data.get("blocked") or []
unknown=data.get("unknown") or []
if blocked:
    print("  blocked files:")
    for path in blocked:
        print(f"    {path}")
if unknown:
    print("  files outside the allowlist:")
    for path in unknown:
        print(f"    {path}")
'
    CHANGELOG="$(ls -1 static/changelogs/*_"${SYSTEM_LOWER}".md 2>/dev/null | tail -n 1 || true)"
    if [[ -n "$CHANGELOG" ]]; then
        echo
        echo "  changelog: $CHANGELOG"
        sed -n '1,8p' "$CHANGELOG" | sed 's/^/    /'
    fi
}

print_plan

if git_in rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    warn "local tag $TAG already exists"
    TAG_EXISTS_LOCAL=1
else
    TAG_EXISTS_LOCAL=0
fi
if git_in ls-remote --tags origin "refs/tags/$TAG" 2>/dev/null | grep -q .; then
    warn "remote tag $TAG already exists"
    TAG_EXISTS_REMOTE=1
else
    TAG_EXISTS_REMOTE=0
fi

blocked_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("blocked") or []))')"
unknown_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("unknown") or []))')"
allowed_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("allowed") or []))')"

plan_errors=0
if [[ "$blocked_count" != "0" || "$unknown_count" != "0" ]]; then
    warn "the tree contains files that must not be recorded"
    plan_errors=1
fi
if [[ "$allowed_count" == "0" ]]; then
    warn "there are no allowlisted changes to record"
    plan_errors=1
fi
if [[ "$TAG_EXISTS_LOCAL" -eq 1 || "$TAG_EXISTS_REMOTE" -eq 1 ]]; then
    warn "tag $TAG already exists; this version was already recorded"
    plan_errors=1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$plan_errors" -ne 0 ]]; then
        info "dry-run: this run would stop before changing services"
        exit 0
    fi
    info "dry-run: no services or repository state were changed"
    exit 0
fi

if [[ -z "$CONFIRM" ]]; then
    info "review the plan, then rerun with --confirm $SYSTEM_VERSION"
    exit 2
fi
if [[ "$CONFIRM" != "$SYSTEM_VERSION" ]]; then
    die "--confirm must match the current SYSTEM_VERSION ($SYSTEM_VERSION)"
fi
if [[ "$plan_errors" -ne 0 ]]; then
    die "refusing to publish until the plan is clean"
fi

info "re-checking the prepared tree"
"$ROOT/scripts/verify_changes.sh"

dump_restart_logs() {
    echo "----- service status -----" >&2
    for unit in ai-chat.service ai-chat-worker@1.service ai-chat-worker@2.service \
                ai-chat-worker@3.service ai-chat-worker@4.service; do
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
    die "restart failed; cache was not purged and the version was not recorded"
fi
ok "restart succeeded"

info "purging CDN cache"
if ! "$ROOT/scripts/purge_cloudflare_cache.sh"; then
    die "cache purge failed after a successful restart; the version was not recorded"
fi
ok "cache purge succeeded"

info "checking live URLs"
if ! "$ROOT/scripts/verify_changes.sh" --skip-tests --live; then
    die "live URL check failed; the version was not recorded"
fi

# Prefer the files that were just deleted in the working tree.
mapfile -t DELETED_NOW < <(git_in status --porcelain -- 'static/js/' 'static/css/' \
    | awk '/^ D|^D / {print $2}')
for rel in "${DELETED_NOW[@]:-}"; do
    [[ -z "${rel:-}" ]] && continue
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
        -H "Host: $PUBLISH_HOST" "https://$PUBLISH_HOST/$rel" || true)"
    if [[ "$code" != "404" && "$code" != "000" ]]; then
        # After CDN purge a brief 200 can remain; also try origin via gunicorn/static.
        if [[ "$code" == "200" ]]; then
            warn "old asset still HTTP 200 via public host: $rel"
        fi
    else
        ok "old $rel HTTP $code"
    fi
done

info "recording allowlisted files"
mapfile -t RECORD_PATHS < <(printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json,sys
for path in json.load(sys.stdin).get("allowed") or []:
    print(path)
')
if [[ ${#RECORD_PATHS[@]} -eq 0 ]]; then
    die "no allowlisted files to record"
fi
# Never add the whole tree. Stage each reviewed path only.
for path in "${RECORD_PATHS[@]}"; do
    if [[ "$path" == *".."* ]]; then
        die "refusing suspicious path: $path"
    fi
    git_in add -- "$path"
done

STAGED="$(git_in diff --cached --name-only)"
if [[ -z "$STAGED" ]]; then
    die "nothing is staged"
fi
while IFS= read -r staged; do
    if ! "$PYTHON" -c '
import sys
sys.path.insert(0, sys.argv[1])
from _release_common import is_allowed_git_path
sys.exit(0 if is_allowed_git_path(sys.argv[2]) else 1)
' "$ROOT/scripts" "$staged"; then
        git_in reset -q HEAD -- "$staged" || true
        die "refusing to record non-allowlisted staged path: $staged"
    fi
done <<< "$STAGED"

git_in commit -m "$MESSAGE"
COMMIT="$(git_in rev-parse --short HEAD)"
ok "recorded $COMMIT"

git_in tag -a "$TAG" -m "$SYSTEM_VERSION"
TAG_COMMIT="$(git_in rev-list -n 1 "$TAG")"
HEAD_COMMIT="$(git_in rev-parse HEAD)"
if [[ "$TAG_COMMIT" != "$HEAD_COMMIT" ]]; then
    die "tag $TAG does not point at HEAD"
fi
ok "tagged $TAG -> $HEAD_COMMIT"

info "sending branch and tag"
if ! git_in push origin HEAD; then
    die "failed to send the branch; local commit $COMMIT and tag $TAG exist"
fi
if ! git_in push origin "$TAG"; then
    die "failed to send tag $TAG; the branch was sent"
fi

REMOTE_TAG="$(git_in ls-remote --tags origin "refs/tags/$TAG" | awk '{print $1}')"
if [[ "$REMOTE_TAG" != "$HEAD_COMMIT" ]]; then
    die "remote tag $TAG is $REMOTE_TAG, expected $HEAD_COMMIT"
fi
ok "remote tag $TAG matches $COMMIT"

"$PYTHON" -c '
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])))
from _release_common import append_handoff_publish_result, parse_versions
append_handoff_publish_result(parse_versions(), sys.argv[2], sys.argv[3])
' "$ROOT/scripts" "$COMMIT" "$TAG"

info "PUBLISH COMPLETE: $SYSTEM_VERSION ($COMMIT, $TAG)"
exit 0
