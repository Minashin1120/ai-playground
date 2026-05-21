#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ENV="${CF_ENV_FILE:-$SCRIPT_DIR/../.env}"
HOME_ENV="${CF_HOME_ENV_FILE:-$HOME/.env}"

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

prompt_value() {
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

load_env_file "$PROJECT_ENV"
load_env_file "$HOME_ENV"

prompt_value CF_ACCOUNT_ID "Cloudflare Account ID"
prompt_value CF_ZONE_ID "Cloudflare Zone ID"
prompt_value CF_API_KEY "Cloudflare API Key / Token" 1

auth_headers=(-H "Authorization: Bearer ${CF_API_KEY}")
if [[ -n "${CF_API_EMAIL:-}" ]]; then
  auth_headers=(-H "X-Auth-Email: ${CF_API_EMAIL}" -H "X-Auth-Key: ${CF_API_KEY}")
fi

zone_detail_file="$(mktemp)"
purge_response_file="$(mktemp)"
cleanup() {
  rm -f "$zone_detail_file" "$purge_response_file"
}
trap cleanup EXIT

echo "Fetching zone details for ${CF_ZONE_ID}..."
if ! curl -fsS "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}" \
  "${auth_headers[@]}" \
  -o "$zone_detail_file"; then
  echo "Failed to fetch Cloudflare zone details." >&2
  cat "$zone_detail_file" >&2 || true
  exit 1
fi

zone_output="$(
  python3 - "$zone_detail_file" "$CF_ACCOUNT_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_account_id = sys.argv[2]

try:
    payload = json.loads(path.read_text())
except Exception as exc:
    print(f"invalid_json::{exc}", file=sys.stderr)
    raise SystemExit(1)

if not payload.get("success"):
    print("api_error::" + json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(1)

result = payload.get("result") or {}
zone_name = result.get("name") or ""
account = result.get("account") or {}
account_id = account.get("id") or ""

if not zone_name:
    print("missing_zone_name", file=sys.stderr)
    raise SystemExit(1)

if expected_account_id and account_id and account_id != expected_account_id:
    print(f"account_mismatch::{account_id}::{expected_account_id}", file=sys.stderr)
    raise SystemExit(1)

print(zone_name)
print(account_id)
PY
)"

mapfile -t zone_info <<< "$zone_output"
zone_name="${zone_info[0]:-}"
resolved_account_id="${zone_info[1]:-}"

echo "Resolved zone: ${zone_name}"
if [[ -n "$resolved_account_id" ]]; then
  echo "Resolved account: ${resolved_account_id}"
fi

hosts=("ai.minashin1120.com")
if [[ -n "${CF_PURGE_HOSTS:-}" ]]; then
  IFS=',' read -r -a hosts <<< "${CF_PURGE_HOSTS}"
fi

payload_file="$(mktemp)"
trap 'cleanup; rm -f "$payload_file"' EXIT

python3 - "$payload_file" "${hosts[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
hosts = [host for host in sys.argv[2:] if host]
path.write_text(json.dumps({"hosts": hosts}, ensure_ascii=False))
PY

echo "Purging cached content for host(s): ${hosts[*]}"
if ! curl -fsS -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
  "${auth_headers[@]}" \
  -H "Content-Type: application/json" \
  --data @"$payload_file" \
  -o "$purge_response_file"; then
  echo "Cloudflare cache purge request failed." >&2
  cat "$purge_response_file" >&2 || true
  exit 1
fi

python3 - "$purge_response_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    print(path.read_text())
    raise SystemExit(1)

if payload.get("success"):
    print("Cloudflare cache purge succeeded.")
    result = payload.get("result")
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("Cloudflare cache purge failed.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(1)
PY
