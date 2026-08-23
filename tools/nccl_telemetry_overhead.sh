#!/usr/bin/env bash
set -euo pipefail
: "${NCCL_TEST:?set NCCL_TEST to all_reduce_perf}"
: "${NCCL_LIB:?set NCCL_LIB to NCCL build lib directory}"
for mode in off basic execution diagnostic; do
  case "$mode" in off) enable=0; level=0;; basic) enable=1; level=0;; execution) enable=1; level=1;; diagnostic) enable=1; level=2;; esac
  echo "=== telemetry=$mode ==="
  NCCL_TELEMETRY_ENABLE=$enable NCCL_TELEMETRY_LEVEL=$level \
    LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$NCCL_TEST" -b 64M -e 1G -f 2 -g 2 | tail -12
done
