#!/usr/bin/env bash
set -euo pipefail
: "${NCCL_TEST:?set NCCL_TEST}"
: "${NCCL_LIB:?set NCCL_LIB}"
N=${NCCL_REPEATS:-5}; SIZE=${NCCL_SIZE:-1G}; OUT=${NCCL_REPEAT_OUT:-nccl_repeated_perf.csv}
echo mode,run,size,time_us,algbw_gbps,busbw_gbps > "$OUT"
for mode in off basic execution diagnostic; do
  case $mode in off) en=0; lv=0;; basic) en=1; lv=0;; execution) en=1; lv=1;; diagnostic) en=1; lv=2;; esac
  for run in $(seq 1 "$N"); do
    log=$(mktemp)
    NCCL_TELEMETRY_ENABLE=$en NCCL_TELEMETRY_LEVEL=$lv LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$NCCL_TEST" -b "$SIZE" -e "$SIZE" -f 2 -g 2 >"$log" 2>&1
    line=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]' "$log" | tail -1); set -- $line
    echo "$mode,$run,$SIZE,$6,$7,$8" >> "$OUT"; rm -f "$log"
  done
done
cat "$OUT"
