#!/usr/bin/env bash
# Record Android-only or operations-only changes without publishing the Web app.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"

TARGET=""
MESSAGE=""
CONFIRM=""

usage() {
    cat <<'EOF'
Usage: scripts/record_changes.sh --target android|operations --message "..."
       scripts/record_changes.sh --target android|operations --message "..." --confirm TOKEN

Without --confirm, prints the exact files and expected Android Actions behavior,
then exits 2. No services, repository state, or caches are changed.

Confirmation token:
  android    android-vX.Y.Z from android/version.properties
  operations OPERATIONS

This command commits and pushes only target-specific allowlisted files. It does
not advance the Web version, run Web deployment, restart services, purge CDN
caches, create a Web release tag, or run Android tooling locally.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="${2:-}"; shift 2 ;;
        --message) MESSAGE="${2:-}"; shift 2 ;;
        --confirm) CONFIRM="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$TARGET" == "android" || "$TARGET" == "operations" ]] \
    || die "--target must be android or operations"
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
[[ "$BRANCH" == "main" ]] || die "refusing to record from branch '$BRANCH' (main only)"

STATUS_JSON="$(run_common classify-record --target "$TARGET" || true)"
[[ -n "$STATUS_JSON" ]] || die "failed to classify repository changes"

if [[ "$TARGET" == "android" ]]; then
    VERSION_NAME="$(sed -n 's/^VERSION_NAME=//p' "$ROOT/android/version.properties")"
    [[ "$VERSION_NAME" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || die "android/version.properties has an invalid VERSION_NAME"
    EXPECTED_CONFIRM="android-v$VERSION_NAME"
else
    EXPECTED_CONFIRM="OPERATIONS"
fi

echo "==> record plan"
echo "  target:   $TARGET"
echo "  branch:   $BRANCH"
echo "  message:  $MESSAGE"
echo "  confirm:  $EXPECTED_CONFIRM"
printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json, sys
data = json.load(sys.stdin)
print("  files:")
for path in data.get("allowed") or []:
    print(f"    {path}")
for key, label in (("blocked", "blocked"), ("unknown", "outside allowlist"),
                   ("outside_target", "outside selected target")):
    paths = data.get(key) or []
    if paths:
        print(f"  {label}:")
        for path in paths:
            print(f"    {path}")
build = data.get("android_build") or []
print("  Android Actions: " + ("will run" if build else "will not run"))
'

blocked_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("blocked") or []) + len(d.get("unknown") or []) + len(d.get("outside_target") or []))')"
allowed_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("allowed") or []))')"
android_build_count="$(printf '%s' "$STATUS_JSON" | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin).get("android_build") or []))')"
[[ "$blocked_count" == "0" ]] || die "the tree contains changes outside the selected target"
[[ "$allowed_count" != "0" ]] || die "there are no changes for target '$TARGET'"

if [[ "$TARGET" == "android" && "$android_build_count" != "0" ]]; then
    printf '%s' "$STATUS_JSON" | "$PYTHON" -c '
import json, sys
paths = json.load(sys.stdin).get("allowed") or []
raise SystemExit(0 if "android/version.properties" in paths else 1)
' || die "Android build inputs changed without updating android/version.properties"
fi

if [[ -z "$CONFIRM" ]]; then
    info "review the plan, then rerun with --confirm $EXPECTED_CONFIRM"
    exit 2
fi
[[ "$CONFIRM" == "$EXPECTED_CONFIRM" ]] \
    || die "--confirm must be $EXPECTED_CONFIRM"

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
    die "failed to push; local commit $COMMIT exists"
fi
ok "recorded and pushed $COMMIT"
if [[ "$android_build_count" != "0" ]]; then
    info "Android Actions must be monitored for commit $(git_in rev-parse HEAD)"
else
    ok "no Android build input changed; Android Actions will not start"
fi
