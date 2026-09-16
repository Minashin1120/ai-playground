#!/usr/bin/env bash
# Wait briefly for active memory/I/O pressure to subside before restarting the
# web and RQ services. Used swap alone is deliberately not a blocker: Linux can
# keep cold pages in swap long after pressure has ended.
set -euo pipefail

MAX_WAIT_SECS="${RESTART_HEADROOM_WAIT_SECS:-120}"
MIN_AVAILABLE_MB="${RESTART_MIN_AVAILABLE_MB:-256}"
MAX_MEMORY_FULL_AVG10="${RESTART_MAX_MEMORY_FULL_AVG10:-2.0}"
MAX_IO_FULL_AVG10="${RESTART_MAX_IO_FULL_AVG10:-5.0}"
MAX_SWAPIN_PAGES_PER_SEC="${RESTART_MAX_SWAPIN_PAGES_PER_SEC:-256}"
SAMPLE_SECS="${RESTART_HEADROOM_SAMPLE_SECS:-2}"
REQUIRED_HEALTHY_SAMPLES="${RESTART_HEADROOM_HEALTHY_SAMPLES:-2}"

for value in "$MAX_WAIT_SECS" "$MIN_AVAILABLE_MB" "$MAX_SWAPIN_PAGES_PER_SEC" "$SAMPLE_SECS" "$REQUIRED_HEALTHY_SAMPLES"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "invalid integer resource-gate setting: $value" >&2; exit 2; }
done
for value in "$MAX_MEMORY_FULL_AVG10" "$MAX_IO_FULL_AVG10"; do
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid decimal resource-gate setting: $value" >&2; exit 2; }
done
(( SAMPLE_SECS > 0 && REQUIRED_HEALTHY_SAMPLES > 0 )) || { echo "resource-gate sample settings must be positive" >&2; exit 2; }

psi_full_avg10() {
    local path="$1"
    awk '$1 == "full" { for (i=2; i<=NF; i++) if ($i ~ /^avg10=/) { split($i,a,"="); print a[2]; exit } }' "$path" 2>/dev/null || true
}

mem_available_mb() {
    awk '$1 == "MemAvailable:" { printf "%d\n", $2 / 1024; exit }' /proc/meminfo
}

swapin_pages() {
    awk '$1 == "pswpin" { print $2; exit }' /proc/vmstat
}

decimal_le() {
    awk -v actual="$1" -v limit="$2" 'BEGIN { exit !(actual <= limit) }'
}

started_at=$SECONDS
healthy_samples=0
while (( SECONDS - started_at < MAX_WAIT_SECS )); do
    first_swapin="$(swapin_pages)"
    sleep "$SAMPLE_SECS"
    second_swapin="$(swapin_pages)"
    available_mb="$(mem_available_mb)"
    memory_full="$(psi_full_avg10 /proc/pressure/memory)"
    io_full="$(psi_full_avg10 /proc/pressure/io)"
    memory_full="${memory_full:-0}"
    io_full="${io_full:-0}"
    swapin_rate=$(( (second_swapin - first_swapin) / SAMPLE_SECS ))
    (( swapin_rate < 0 )) && swapin_rate=0

    healthy=1
    (( available_mb >= MIN_AVAILABLE_MB )) || healthy=0
    decimal_le "$memory_full" "$MAX_MEMORY_FULL_AVG10" || healthy=0
    decimal_le "$io_full" "$MAX_IO_FULL_AVG10" || healthy=0
    (( swapin_rate <= MAX_SWAPIN_PAGES_PER_SEC )) || healthy=0

    printf '    restart headroom: available=%sMiB memory-full=%s%% io-full=%s%% swap-in=%spages/s\n' \
        "$available_mb" "$memory_full" "$io_full" "$swapin_rate"
    if (( healthy == 1 )); then
        healthy_samples=$((healthy_samples + 1))
        if (( healthy_samples >= REQUIRED_HEALTHY_SAMPLES )); then
            echo "==> Restart resource headroom is healthy."
            exit 0
        fi
    else
        healthy_samples=0
    fi
    sleep 3
done

echo "==> Host resource pressure did not settle within ${MAX_WAIT_SECS}s; restart was not started." >&2
exit 1
