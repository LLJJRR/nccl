#!/usr/bin/env bash
set -euo pipefail

: "${NCCL_TEST_DIR:?set NCCL_TEST_DIR to an MPI-enabled nccl-tests build directory}"
: "${NCCL_LIB:?set NCCL_LIB to this NCCL build's lib directory}"
: "${NCCL_MPI_NP:?set NCCL_MPI_NP to the total number of MPI ranks}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpirun_bin=${NCCL_MPIRUN:-mpirun}
results_root=${NCCL_MULTINODE_OUT:-nccl_multinode_results/$(date +%Y%m%d-%H%M%S)}
iterations=${NCCL_TEST_ITERATIONS:-3}
warmup=${NCCL_TEST_WARMUP:-2}
repeats=${NCCL_MATRIX_REPEATS:-3}
collectives=${NCCL_MATRIX_COLLECTIVES:-"all_reduce all_gather reduce_scatter"}
sizes=${NCCL_MATRIX_SIZES:-"1M 64M 1G"}
modes=${NCCL_MATRIX_MODES:-"off level1 level2"}
controls=${NCCL_MATRIX_CONTROLS:-"ib socket"}
fault_poll_delay_us=${NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US:-0}
rdma_counters=${NCCL_RDMA_COUNTERS:-1}
rank_helper=${NCCL_MULTINODE_RANK_HELPER:-$script_dir/nccl_multinode_rank.sh}
counter_tool=${NCCL_RDMA_COUNTER_TOOL:-$script_dir/nccl_rdma_counters.py}
telemetry_report=${NCCL_TELEMETRY_REPORT_TOOL:-$script_dir/nccl_telemetry_report.py}
rdma_report=${NCCL_RDMA_REPORT_TOOL:-$script_dir/nccl_rdma_report.py}
merge_tool=${NCCL_TELEMETRY_MERGE_TOOL:-$script_dir/nccl_telemetry_merge.py}
summary_tool=${NCCL_MULTINODE_SUMMARY_TOOL:-$script_dir/nccl_multinode_summary.py}

read -r -a mpi_extra <<< "${NCCL_MPI_ARGS:-}"
mpi=("$mpirun_bin" -np "$NCCL_MPI_NP" "${mpi_extra[@]}")

mkdir -p "$results_root"
results_root=$(cd "$results_root" && pwd)
results_csv=$results_root/results.csv
summary_csv=$results_root/summary.csv
echo "control,collective,size,mode,repeat,time_us,algbw_gbps,busbw_gbps,wrong,run_dir" > "$results_csv"

export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if [ -n "${NCCL_IB_HCA:-}" ]; then export NCCL_IB_HCA; fi

snapshot_all_nodes() {
  local run_dir=$1
  local label=$2
  "${mpi[@]}" "$rank_helper" snapshot "$run_dir" "$label" "$counter_tool"
}

write_reports() {
  local run_dir=$1
  local trace_dir=$2
  local mode=$3
  local traces=("$trace_dir"/nccl_telemetry.*.txt)
  local have_traces=0
  if [ -e "${traces[0]}" ]; then
    have_traces=1
    if [ "$mode" = off ]; then
      echo "telemetry OFF unexpectedly produced traces in $trace_dir" >&2
      exit 1
    fi
    if [ "${#traces[@]}" -ne "$NCCL_MPI_NP" ]; then
      echo "expected $NCCL_MPI_NP traces, found ${#traces[@]} in $trace_dir" >&2
      exit 1
    fi
    python3 "$merge_tool" "$run_dir/telemetry.merged.txt" "${traces[@]}"
    python3 "$telemetry_report" "${traces[@]}" > "$run_dir/telemetry.report.txt"
  elif [ "$mode" != off ]; then
    echo "telemetry $mode produced no traces in $trace_dir" >&2
    exit 1
  fi

  if [ "$rdma_counters" != 1 ]; then return; fi
  local before_files=("$run_dir"/rdma.before.*.json)
  local before host after diff
  for before in "${before_files[@]}"; do
    [ -e "$before" ] || continue
    host=${before##*/rdma.before.}
    host=${host%.json}
    after=$run_dir/rdma.after.$host.json
    [ -e "$after" ] || continue
    diff=$run_dir/rdma.diff.$host.json
    python3 "$counter_tool" diff "$before" "$after" --json -o "$diff"
    local host_traces=("$trace_dir"/nccl_telemetry."$host".*.txt)
    if [ "$have_traces" = 1 ] && [ -e "${host_traces[0]}" ]; then
      python3 "$rdma_report" "${host_traces[@]}" --rdma-diff "$diff" \
        > "$run_dir/rdma.report.$host.txt"
    fi
  done
}

for control in $controls; do
  case "$control" in
    ib) export NCCL_IB_DISABLE=0 NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US=0 ;;
    socket) export NCCL_IB_DISABLE=1 NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US=0 ;;
    ib_poll_delay)
      if [ "$fault_poll_delay_us" -le 0 ]; then
        echo "ib_poll_delay requires NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US > 0" >&2
        exit 2
      fi
      export NCCL_IB_DISABLE=0 NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US=$fault_poll_delay_us
      ;;
    *) echo "unsupported NCCL_MATRIX_CONTROLS entry: $control" >&2; exit 2 ;;
  esac
  for collective in $collectives; do
    binary=$NCCL_TEST_DIR/${collective}_perf
    [ -x "$binary" ] || { echo "missing executable: $binary" >&2; exit 2; }
    for size in $sizes; do
      for mode in $modes; do
        case "$mode" in
          off) export NCCL_TELEMETRY_ENABLE=0 NCCL_TELEMETRY_LEVEL=0 ;;
          level1) export NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=1 ;;
          level2) export NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=2 ;;
          *) echo "unsupported NCCL_MATRIX_MODES entry: $mode" >&2; exit 2 ;;
        esac
        for repeat in $(seq 1 "$repeats"); do
          run_name=${control}.${collective}.${size}.${mode}.r${repeat}
          run_dir=$results_root/$run_name
          trace_dir=$run_dir/traces
          mkdir -p "$trace_dir"
          export NCCL_TELEMETRY_DIR=$trace_dir

          if [ "$rdma_counters" = 1 ]; then snapshot_all_nodes "$run_dir" before; fi
          log=$run_dir/nccl-tests.log
          echo "=== $run_name ==="
          "${mpi[@]}" "$rank_helper" run "$binary" "$size" "$iterations" "$warmup" \
            2>&1 | tee "$log"
          if [ "$rdma_counters" = 1 ]; then snapshot_all_nodes "$run_dir" after; fi

          perf_line=$(awk '$1 ~ /^[0-9]+$/ && NF >= 9 {line=$0} END {print line}' "$log")
          if [ -z "$perf_line" ]; then
            echo "no nccl-tests result row found in $log" >&2
            exit 1
          fi
          read -r _bytes _count _type _op _root time_us algbw busbw wrong _rest <<< "$perf_line"
          echo "$control,$collective,$size,$mode,$repeat,$time_us,$algbw,$busbw,$wrong,$run_dir" \
            >> "$results_csv"
          if [ "$wrong" != 0 ]; then
            echo "nccl-tests reported wrong=$wrong in $log" >&2
            exit 1
          fi
          write_reports "$run_dir" "$trace_dir" "$mode"
        done
      done
    done
  done
done

python3 "$summary_tool" "$results_csv" -o "$summary_csv"
cat "$summary_csv"
echo "artifacts=$results_root"
