#!/usr/bin/env bash
# prepare_version.sh phases; sourced after _release_lib.sh, never run alone.

prepare_build_assets() {
    info "building minified frontend assets"
    "$ROOT/scripts/build_frontend.sh"
    # Apache serves static files directly, so gunicorn's access alone is insufficient.
    info "making static assets readable by the web server"
    find "$ROOT/static" -type f ! -perm -004 -exec chmod a+r {} +
    find "$ROOT/static" -type d ! -perm -005 -exec chmod a+rx {} +
}

prepare_preflight() {
    prepare_build_assets
    local checks=()
    if [[ "$SKIP_VERIFY" -eq 1 ]]; then
        warn "regression tests explicitly skipped; static checks still run"
        checks+=(--skip-tests)
    fi
    info "checking edited sources before changing version or changelog"
    VERIFY_LOG="$ROOT/scripts/verify.preflight.last.log" "$ROOT/scripts/verify_changes.sh" "${checks[@]}"
}

prepare_check_new_assets() {
    prepare_build_assets
    info "checking new version assets and references (regression tests already completed)"
    "$ROOT/scripts/verify_changes.sh" --skip-tests
}
