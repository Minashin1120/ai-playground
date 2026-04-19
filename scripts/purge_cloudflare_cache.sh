#!/usr/bin/env bash
set -euo pipefail

prompt() {
  local __var_name="$1"
  local __label="$2"
  local __secret="${3:-0}"
  local __default="${4:-}"
  local __value=""

  if [[ -n "${!__var_name:-}" ]]; then
    __value="${!__var_name}"
  else
    if [[ "$__secret" == "1" ]]; then
      if [[ -n "$__default" ]]; then
        read -r -s -p "$__label [$__default]: " __value
        echo
        __value="${__value:-$__default}"
      else
        read -r -s -p "$__label: " __value
        echo
      fi
    else
      if [[ -n "$__default" ]]; then
        read -r -p "$__label [$__default]: " __value
        __value="${__value:-$__default}"
      else
        read -r -p "$__label: " __value
      fi
    fi
  fi

  if [[ -z "$__value" ]]; then
    echo "Missing value for $__label" >&2
    exit 1
  fi

  printf -v "$__var_name" '%s' "$__value"
}

echo "Cloudflare cache purge"
echo "This script purges the entire zone cache."
echo
prompt CF_ZONE_ID "Cloudflare Zone ID"

auth_mode="${CF_AUTH_MODE:-}"
if [[ -z "$auth_mode" ]]; then
  read -r -p "Auth mode (token/key) [token]: " auth_mode
  auth_mode="${auth_mode:-token}"
fi
auth_mode="${auth_mode,,}"

if [[ "$auth_mode" == "token" ]]; then
  prompt CF_API_TOKEN "Cloudflare API Token" 1
  response="$(
    curl -sS -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
      -H "Authorization: Bearer ${CF_API_TOKEN}" \
      -H "Content-Type: application/json" \
      --data '{"purge_everything":true}'
  )"
else
  prompt CF_API_EMAIL "Cloudflare API Email"
  prompt CF_API_KEY "Cloudflare API Key" 1
  response="$(
    curl -sS -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
      -H "X-Auth-Email: ${CF_API_EMAIL}" \
      -H "X-Auth-Key: ${CF_API_KEY}" \
      -H "Content-Type: application/json" \
      --data '{"purge_everything":true}'
  )"
fi

python3 - <<'PY' "$response"
import json, sys
raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception:
    print(raw)
    raise SystemExit(1)
if data.get("success"):
    result = data.get("result") or {}
    print("Cloudflare cache purge succeeded.")
    if isinstance(result, dict) and result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("Cloudflare cache purge failed.")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    raise SystemExit(1)
PY
