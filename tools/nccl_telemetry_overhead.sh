#!/usr/bin/env bash
set -euo pipefail
: "${NCCL_TEST:?set NCCL_TEST to all_reduce_perf}"
: "${NCCL_LIB:?set NCCL_LIB to NCCL build lib directory}"
OUT=${NCCL_TELEMETRY_OVERHEAD_OUT:-nccl_telemetry_overhead.csv}
echo "mode,size_bytes,time_us,algbw_gbps,busbw_gbps" > "$OUT"
for mode in off basic execution diagnostic; do
  case "$mode" in off) enable=0; level=0;; basic) enable=1; level=0;; execution) enable=1; level=1;; diagnostic) enable=1; level=2;; esac
  echo "=== telemetry=$mode ==="
  log=$(mktemp)
  NCCL_TELEMETRY_ENABLE=$enable NCCL_TELEMETRY_LEVEL=$level \
    LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$NCCL_TEST" -b 64M -e 1G -f 2 -g 2 | tee "$log"
  awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]/ {print "'"$mode"'" "," $1 "," $6 "," $7 "," $8}' "$log" >> "$OUT"
  rm -f "$log"
done
echo "=== overhead versus off (mean) ==="
awk -F, 'NR>1 {n[$1]++; t[$1]+=$3; a[$1]+=$4; b[$1]+=$5} END {for (m in n) printf "%s time_us=%.3f algbw=%.3f busbw=%.3f\n",m,t[m]/n[m],a[m]/n[m],b[m]/n[m]}' "$OUT" | sort
awk -F, 'NR>1 {n[$1]++; t[$1]+=$3; a[$1]+=$4; b[$1]+=$5} END {if(n["off"]) for (m in n) if(m!="off") printf "%s latency_overhead=%.2f%% algbw_change=%.2f%% busbw_change=%.2f%%\n",m,(t[m]/n[m]/(t["off"]/n["off"])-1)*100,(a[m]/n[m]/(a["off"]/n["off"])-1)*100,(b[m]/n[m]/(b["off"]/n["off"])-1)*100}' "$OUT"
