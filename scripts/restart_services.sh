#!/usr/bin/env bash
# Restart gunicorn + RQ workers without blocking, then confirm completion via a
# short bounded poll so we never linger after the services are already up.
#
# Background:
#   ai-chat-worker@.service sets TimeoutStopSec=660 (workers may finish a long
#   job on graceful shutdown), and a plain sequential `systemctl restart a && b`
#   can block for minutes or hit the shell timeout even though each service has
#   actually restarted. We use `systemctl restart --no-block` (returns at once),
#   then wait only until every unit's MainPID has changed and gunicorn serves
#   HTTP 200 again.

set -euo pipefail

WEB=ai-chat.service
# Only 2 RQ workers are enabled now: the 1.9 GB host ran out of memory and
# swapped-out worker pages were stalling requests for 10-15 s. Two workers keep
# the fast/chat queue responsive while staying within the host's RAM budget.
WORKERS=(ai-chat-worker@1.service ai-chat-worker@2.service)
SERVICES=("$WEB" "${WORKERS[@]}")

# Bounded wait: fail after this many seconds even if some unit is still stopping
# (e.g. a worker mid-job during graceful shutdown).
DEADLINE_SECS="${RESTART_DEADLINE_SECS:-120}"

main_pid() {
    systemctl show -p MainPID --value "$1" 2>/dev/null | tr -d ' '
}

web_serves_200() {
    [[ "$(curl -s -o /dev/null -w '%{http_code}' -H 'Host: ai.minashin1120.com' \
        "http://127.0.0.1:3111/api/version" 2>/dev/null || true)" == "200" ]]
}

# --- snapshot current PIDs so we can detect an actual restart -----------------
declare -A OLD_PID
for s in "${SERVICES[@]}"; do
    OLD_PID["$s"]="$(main_pid "$s")"
done

echo "==> Enqueuing restart for: ${SERVICES[*]}"
sudo systemctl restart --no-block "${SERVICES[@]}"

echo "==> Waiting for services to come back (bounded ${DEADLINE_SECS}s) ..."
declare -A DONE
started_at=$SECONDS
while (( SECONDS - started_at < DEADLINE_SECS )); do
    pending=0
    for s in "${SERVICES[@]}"; do
        if [[ "${DONE[$s]:-}" == "1" ]]; then
            continue
        fi
        cur="$(main_pid "$s")"
        restarted=$([[ -n "$cur" && "$cur" != "${OLD_PID[$s]}" ]] && echo yes || echo no)
        if [[ "$restarted" == "yes" ]]; then
            if [[ "$s" == "$WEB" ]] && ! web_serves_200; then
                pending=1
                continue
            fi
            DONE["$s"]=1
            echo "    $s: restarted (pid $cur)"
            continue
        fi
        pending=1
    done
    if (( pending == 0 )); then
        echo "==> All services are active. Done."
        exit 0
    fi
    sleep 1
done

echo "==> Timed out waiting after ${DEADLINE_SECS}s. Current status:" >&2
for s in "${SERVICES[@]}"; do
    printf '    %s: %s (pid %s, old %s)\n' "$s" \
        "$(systemctl is-active "$s" 2>/dev/null || echo unknown)" \
        "$(main_pid "$s")" "${OLD_PID[$s]}" >&2
done
exit 1
