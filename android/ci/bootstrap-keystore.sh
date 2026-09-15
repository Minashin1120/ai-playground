#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${GITHUB_ACTIONS:-}" != true || "${GITHUB_EVENT_NAME:-}" == pull_request ]]; then
    echo 'Fixed-key bootstrap is allowed only in trusted GitHub Actions runs.' >&2
    exit 1
fi
git fetch origin main
git checkout --detach origin/main
key=android/ci/debug.keystore
if [[ -e "$key" ]]; then
    [[ -s "$key" ]] || { echo 'Existing keystore is empty; refusing to replace it.' >&2; exit 1; }
    echo 'Reusing the committed fixed keystore.'
else
    if [[ -n "$(git log --all --format=%H -- "$key")" ]]; then
        echo 'A keystore existed in history. Restore that exact file; do not generate another.' >&2
        exit 1
    fi
    keytool -genkeypair -keystore "$key" -storetype JKS -storepass android \
        -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 \
        -validity 10000 -dname 'CN=Android Debug,O=Android,C=US'
    keytool -exportcert -keystore "$key" -storepass android -alias androiddebugkey \
        | sha256sum | cut -d ' ' -f 1 > android/ci/signing-fingerprint.txt
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -- "$key" android/ci/signing-fingerprint.txt
    git -c commit.gpgsign=false commit -m 'ci: persist the fixed Android signing key'
    # A rejected push fails the job. No build may use a non-persisted key.
    git push origin HEAD:main
fi
bash android/ci/verify-keystore.sh
printf 'sha=%s\n' "$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"
