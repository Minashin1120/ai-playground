#!/usr/bin/env bash
# Advance SYSTEM_VERSION / APP_VERSION and versioned frontend assets.
# Run this only after the current version's source files have been edited.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_release_lib.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_prepare_checks.sh"

NOTES=""
NOTES_FILE=""
HANDOFF_NOTES=""
DRY_RUN=0
SKIP_VERIFY=0

usage() {
    cat <<'EOF'
Usage: scripts/prepare_version.sh --notes "..." [--handoff-notes "..."]
       scripts/prepare_version.sh --notes-file PATH

Before advancing the version, rebuilds the edited sources and runs regression
tests and static checks. A failure leaves versions and changelogs unchanged.
Then advances the current version to the next one:
  - copies the edited chat_core / CSS sources to the new version name
  - updates app.py, README.md, and MODELS.md
  - writes the public changelog from --notes
  - deletes the previous versioned assets
  - rebuilds minified files
  - verifies new assets, syntax, and references without repeating regression tests

Does not restart services or record the version in the remote repository.
Use --dry-run to print the plan without changing files.
Use --skip-verify only when regression tests were explicitly waived; static
checks still run before and after advancing the version.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --notes) NOTES="${2:-}"; shift 2 ;;
        --notes-file) NOTES_FILE="${2:-}"; shift 2 ;;
        --handoff-notes) HANDOFF_NOTES="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [[ -z "$NOTES" && -z "$NOTES_FILE" ]]; then
    die "provide --notes or --notes-file (the public changelog body)"
fi

COMMON_ARGS=()
if [[ -n "$NOTES_FILE" ]]; then
    COMMON_ARGS+=(--notes-file "$NOTES_FILE")
else
    COMMON_ARGS+=(--notes "$NOTES")
fi
if [[ -n "$HANDOFF_NOTES" ]]; then
    COMMON_ARGS+=(--handoff-notes "$HANDOFF_NOTES")
fi

if [[ -n "$NOTES_FILE" ]]; then
    run_common check-notes --notes-file "$NOTES_FILE" >/dev/null
else
    run_common check-notes --notes "$NOTES" >/dev/null
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    info "prepare dry-run"
    run_common prepare --dry-run "${COMMON_ARGS[@]}"
    exit 0
fi

prepare_preflight

info "preparing the next version"
run_common prepare "${COMMON_ARGS[@]}"
prepare_check_new_assets
