#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 snapshot <output-dir> <label> <counter-tool>" >&2
  echo "       $0 run <binary> <size> <iterations> <warmup>" >&2
  exit 2
}

action=${1:-}
case "$action" in
  snapshot)
    [ "$#" -eq 4 ] || usage
    output_dir=$2
    label=$3
    counter_tool=$4
    hostname=$(hostname)
    mkdir -p "$output_dir"

    # Exactly one rank per host captures a snapshot. The run directory is
    # unique, so the host-local lock never needs cleanup or reuse.
    if mkdir "$output_dir/.snapshot.${label}.${hostname}.lock" 2>/dev/null; then
      python3 "$counter_tool" snapshot -o "$output_dir/rdma.${label}.${hostname}.json"
    fi
    ;;
  run)
    [ "$#" -eq 5 ] || usage
    binary=$2
    size=$3
    iterations=$4
    warmup=$5
    : "${NCCL_TELEMETRY_DIR:?NCCL_TELEMETRY_DIR must name a shared run directory}"
    mkdir -p "$NCCL_TELEMETRY_DIR"
    exec "$binary" -b "$size" -e "$size" -f 2 -g 1 -n "$iterations" -w "$warmup"
    ;;
  *)
    usage
    ;;
esac
