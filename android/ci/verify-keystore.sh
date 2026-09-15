#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -s ci/debug.keystore && -s ci/signing-fingerprint.txt ]] || {
    echo 'Fixed keystore/fingerprint missing. Bootstrap once through Android CI on main.' >&2; exit 1;
}
fingerprint=$(keytool -exportcert -keystore ci/debug.keystore -storepass android -alias androiddebugkey | sha256sum | cut -d ' ' -f 1)
[[ "$fingerprint" == "$(tr -d '\r\n' < ci/signing-fingerprint.txt)" ]] || {
    echo 'Signing fingerprint changed; refusing this key.' >&2; exit 1;
}
keytool -list -v -keystore ci/debug.keystore -storepass android -alias androiddebugkey
printf 'Fixed certificate SHA-256: %s\n' "$fingerprint"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '### Fixed signing certificate\n\nSHA-256: `%s`\n' "$fingerprint" >> "$GITHUB_STEP_SUMMARY"
fi
