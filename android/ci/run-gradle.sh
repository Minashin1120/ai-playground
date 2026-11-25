#!/usr/bin/env bash
# Runs Gradle and retries when dependency downloads fail transiently.
# Usage: bash ci/run-gradle.sh <log-file> <gradle tasks...>
set -uo pipefail
log="$1"
shift
for attempt in 1 2 3; do
  echo "=== gradle attempt $attempt ===" >> "$log"
  ./gradlew --no-daemon "$@" --stacktrace >> "$log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    exit 0
  fi
  if [[ $attempt -lt 3 ]] && grep -Eq 'Could not (download|GET)|Gateway Time-?out|Connection (reset|timed out)|Read timed out|status code 50[234]|Failed to fetch maven artifact' "$log"; then
    echo "=== transient dependency download failure; retrying in 15s (attempt $attempt/3) ===" >> "$log"
    sleep 15
    continue
  fi
  exit "$status"
done
exit 1
