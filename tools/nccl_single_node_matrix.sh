#!/usr/bin/env bash
set -euo pipefail
: "${NCCL_TEST_DIR:?set NCCL_TEST_DIR, e.g. /root/nccl-tests/build}"
: "${NCCL_LIB:?set NCCL_LIB, e.g. /root/nccl-build/lib}"
OUT=${NCCL_MATRIX_OUT:-nccl_single_node_matrix.csv}
echo "collective,transport,size,mode,time_us,algbw_gbps,busbw_gbps,wrong,trace" > "$OUT"
export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
for collective in all_reduce all_gather reduce_scatter; do
  bin="$NCCL_TEST_DIR/${collective}_perf"
  for transport in p2p shm; do
    for size in 64M 1G 4G; do
      for mode in off execution; do
        log=$(mktemp)
        if [ "$transport" = shm ]; then p2p=1; else p2p=0; fi
        if [ "$mode" = execution ]; then en=1; lv=1; else en=0; lv=0; fi
        NCCL_P2P_DISABLE=$p2p NCCL_TELEMETRY_ENABLE=$en NCCL_TELEMETRY_LEVEL=$lv "$bin" -b "$size" -e "$size" -f 2 -g 2 >"$log" 2>&1
        line=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]' "$log" | tail -1)
        set -- $line
        trace=$(ls -t /tmp/nccl_telemetry.*.txt 2>/dev/null | head -1 || true)
        echo "$collective,$transport,$size,$mode,$6,$7,$8,$9,$trace" >> "$OUT"
        rm -f "$log"
      done
    done
  done
done
cat "$OUT"
