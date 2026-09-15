#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
apk="${1:?Pass APK path relative to android/}"
[[ -s "$apk" ]] || { echo 'APK missing.' >&2; exit 1; }
apksigner="${ANDROID_HOME:?}/build-tools/36.0.0/apksigner"
verification=$("$apksigner" verify --verbose --print-certs "$apk")
printf '%s\n' "$verification"
actual=$(printf '%s\n' "$verification" | sed -n 's/^Signer #1 certificate SHA-256 digest: //p' | tr 'A-F' 'a-f')
[[ "$actual" == "$(tr -d '\r\n' < ci/signing-fingerprint.txt)" ]] || {
    echo 'APK was not signed by the fixed committed certificate.' >&2; exit 1;
}
sha256sum "$apk" > "$apk.sha256"
