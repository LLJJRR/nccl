# Multi-node NCCL telemetry

This branch provides a low-overhead multi-node execution trace for
AllReduce, AllGather, and ReduceScatter.  The trace joins:

```
collective -> channel -> proxy -> NET_PATH -> NIC/port -> QP -> WQE/CQE
                                                   \-> external RDMA counters
```

## Collection levels

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=0` records collective plans,
transport connections, and network path metadata.  This is the normal
production-safe level.

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=1` additionally records proxy
enqueue/start/first-progress/complete timestamps and channel summaries.  Use
this level to measure queue delay, first-progress delay, proxy duration, and
progress rate.

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=2` adds per-step transfer, GPU
work, and in-tree IB request/QP/WQE/CQE events.  This is a diagnostic mode and
should only be used for a short reproduction because event volume is much
higher.

All timestamps in the NCCL text trace are `CLOCK_MONOTONIC` nanoseconds.  The
trace header declares this clock domain and the supported capabilities.  Host
monotonic clocks are not assumed to be synchronized across nodes; align node
traces with an external clock offset when doing cross-node timeline analysis.
The current tools intentionally do not estimate that offset. Durations and
ordering within one process are valid; do not subtract timestamps originating
on different hosts.

Set `NCCL_TELEMETRY_DIR` to an existing directory to keep each experiment's
traces together. Trace filenames contain hostname and PID, so ranks on
different hosts cannot overwrite one another in a shared directory.

## Automated two-node matrix

The matrix runner covers AllReduce, AllGather, and ReduceScatter at 1 MiB,
64 MiB, and 1 GiB. For every case it runs telemetry OFF, level 1, and level 2,
and compares the native IB path with `NCCL_IB_DISABLE=1`. It captures RDMA
counters once per host before and after every run and generates per-run
telemetry and NIC-path reports.

The NCCL build, MPI-enabled nccl-tests build, helper scripts, and output path
must be visible at the same path on both nodes. Example for eight ranks:

```bash
export NCCL_TEST_DIR=/shared/nccl-tests/build
export NCCL_LIB=/shared/nccl/build/lib
export NCCL_MPI_NP=8
export NCCL_MPI_ARGS="--hostfile /shared/hosts --map-by ppr:4:node"
export NCCL_MULTINODE_OUT=/shared/results/nccl-telemetry-001

tools/nccl_multinode_matrix.sh
```

The output directory contains:

```text
results.csv                         raw result for every independent run
summary.csv                         mean throughput and OFF-relative overhead
<case>/nccl-tests.log               original benchmark output
<case>/traces/*.txt                 one telemetry trace per rank
<case>/telemetry.report.txt         collective/channel/proxy/RDMA diagnosis
<case>/rdma.{before,after}.*.json   per-host NIC snapshots
<case>/rdma.diff.*.json             per-host counter deltas
<case>/rdma.report.*.txt            GUID+port path join for that host
```

Useful overrides are `NCCL_MATRIX_SIZES`, `NCCL_MATRIX_COLLECTIVES`,
`NCCL_MATRIX_MODES`, `NCCL_MATRIX_CONTROLS`, `NCCL_MATRIX_REPEATS`,
`NCCL_TEST_ITERATIONS`, and `NCCL_TEST_WARMUP`. The defaults use three measured
iterations and two warmups to keep level-2 traces bounded. Set `NCCL_IB_HCA` to
constrain the IB baseline to a particular HCA or rail. Set
`NCCL_RDMA_COUNTERS=0` only on systems without `/sys/class/infiniband`.

For a reproducible diagnostic fault, the in-tree IB backend can delay CQ
polling once in the matching rank process. This does not emulate wire
congestion; it verifies that the trace can distinguish a posted WQE from
delayed completion observation. The hook is disabled by default and only takes
effect when level-2 request context is present:

```bash
export NCCL_MATRIX_CONTROLS="ib ib_poll_delay"
export NCCL_MATRIX_MODES="off level2"
export NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_US=5000
export NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_RANK=7
export NCCL_TELEMETRY_FAULT_IB_POLL_DELAY_CHANNEL=6
tools/nccl_multinode_matrix.sh
```

The matching proxy report must contain
`diagnostic=injected_cq_poll_delay`; this prevents the controlled host-side
delay from being mislabeled as physical network congestion.

## RDMA correlation

Keep hardware counter reads outside NCCL's progress path:

```bash
tools/nccl_rdma_counters.py snapshot -o before.json
# run the two-node NCCL benchmark here
tools/nccl_rdma_counters.py snapshot -o after.json
tools/nccl_rdma_counters.py diff before.json after.json --json > rdma-diff.json
tools/nccl_rdma_report.py /tmp/nccl_telemetry.<host>.<pid>.txt --rdma-diff rdma-diff.json
```

The join key is normalized `GUID + port`, which is also printed by the
`NET_PATH` event.  Missing devices, unsupported counters, and counter resets
are reported explicitly.

Useful two-node controls include:

```bash
# baseline
NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=1 ...

# force a non-P2P path where applicable
NCCL_P2P_DISABLE=1 ...

# disable IB to compare the socket/fallback path
NCCL_IB_DISABLE=1 ...

# select a specific HCA/rail
NCCL_IB_HCA=mlx5_0 ...
```

For each control, record NCCL `time/algbw/busbw`, the telemetry report, and
the RDMA counter diff.  A useful diagnosis should identify the slow rank,
channel, peer, proxy delay, selected NIC/port, and whether the hardware
counters support the proposed explanation.

## QP/WQE/CQE boundary

The in-tree IB transport now records request IDs, QP numbers, WQE `wr_id`,
opcode, post timestamps, and CQE status/byte length in diagnostic mode.  The
trace header advertises this as `qp_wqe_cqe=ib_internal_diagnostic`.  The
public NCCL NET plugin interface remains opaque, so Socket and third-party
plugins report no QP/WQE/CQE records rather than guessing them.  A
plugin-specific adapter can add equivalent records later without changing the
core collective/channel/proxy schema.

WQE records also identify whether a CQE is expected. Unsignaled send WQEs are
reported separately and are not counted as unmatched posts; their completion
is covered by the later signaled WQE on the same send queue.

A grouped receive is one physical IB request owned by multiple NCCL proxy
sub-operations. Each `RDMA_REQUEST` association therefore includes
`owner_index` and `owner_count`. The report attaches the shared WQE/CQE to
every real owner without claiming that those associations are separate
physical requests.

The existing proxy timeline still distinguishes queue delay (enqueue to
start), first-progress delay (start to first completed transfer), and proxy
duration (start to completion).  This is sufficient to separate NCCL/proxy
scheduling delay from a path-level RDMA counter anomaly in the first delivery.

## Acceptance checklist

For a two-node multi-GPU run, the deliverable should include:

1. One hostname/PID trace per rank, bundled with
   `tools/nccl_telemetry_merge.py` without cross-host timestamp arithmetic.
2. A report showing collective/channel/proxy/NIC path associations.
3. Before/after RDMA counter snapshots and a GUID/port join report.
4. A baseline versus level-1 overhead measurement, plus a short level-2
   diagnostic capture when investigating a known stall.
5. One controlled comparison such as `NCCL_IB_DISABLE=1` or `NCCL_IB_HCA` that
   changes the selected path and is reflected in both throughput and telemetry.

In diagnostic mode, an IB trace should also contain `RDMA_REQUEST` records for
WQE posts, CQEs, and request completion.  The report summarizes QP IDs,
completion latency, unmatched posts, and non-zero WC status.  Pre-posted
receive buffers and third-party NET plugins may not expose a one-to-one WQE
mapping; those cases remain explicitly partial/unsupported.

Hardware acceptance is complete only after a real two-node run shows all
expected ranks and collectives, zero nccl-tests errors, IB `NET_PATH` records,
matching GUID/port counters, level-2 WQE/CQE records for the in-tree IB backend,
and a measurable Socket-path change under `NCCL_IB_DISABLE=1`.

The matrix runner fails if an enabled run produces fewer traces than MPI ranks,
if an OFF run unexpectedly emits traces, or if nccl-tests reports a non-zero
wrong-result count. Reports surface ring-buffer drops, unmatched signaled WQEs,
WC errors, and non-zero retry/error/congestion-related NIC counters rather than
silently treating a partial trace as complete.

## Detailed verbs datapath (Level 2)

The in-tree IB backend exposes the following diagnostic-only chain:

```text
proxy op -> ncclIbRequest -> WR -> SGE -> QP depth -> CQ poll -> CQE -> request completion
```

`RDMA_REQUEST_STATE` records `CREATE`, `READY`, `POST_BEGIN`, `POST_END`, and
`COMPLETE`, together with expected, posted, and completed bytes and cumulative
WR/CQE counts. The report derives request construction, submit, first-CQE,
CQE-tail, and post-CQE bookkeeping intervals from these events.

`RDMA_WR` records QP, `wr_id`, opcode, send flags, SGE count, remote address,
rkey, immediate data, and total SGE bytes. `RDMA_SGE` records every SGE address,
length, and lkey. These records are emitted only at level 2. Addresses and keys
are diagnostic process data and traces should be handled as sensitive artifacts.

`RDMA_QP_DEPTH` maintains separate SQ and RQ cumulative posted/completed WR and
byte counters. For an SQ, a signaled CQE retires that WR and the preceding
unsignaled batch. Unsignaled WRs therefore do not produce false "missing CQE"
diagnostics. NCCL's receive WR has zero SGEs because payload arrives through a
remote RDMA write, so RQ outstanding bytes are unavailable even though RQ WR
depth is exact. The receive CTS SQ is recorded at WR/SGE granularity, but is not
assigned exact request-owned depth because an occasional signaled CTS can retire
work from already-reused receive requests.

`RDMA_CQ_POLL` aggregates polling instead of recording every empty call. It
emits the first poll, every non-empty poll, each 64-empty-poll window, and any
remaining window at request completion. `cq_id` is the low 32 bits of the CQ
pointer and is process-local; standard verbs does not provide a portable CQ
number. Poll activity is attributed to the request whose `test()` call drove
the shared CQ, while each returned CQE remains associated with its actual owner.

`RDMA_CQE_DETAIL` records status, opcode, byte length, vendor error, source QP,
WC flags, and immediate data. On error CQEs only `wr_id`, QP, status, and vendor
error are treated as valid; optional successful-completion fields are zeroed.
Grouped receive and grouped send associations retain every real proxy owner but
do not claim that duplicated associations represent additional physical CQEs.

The report includes per-request lifecycle intervals, opcode/SGE totals,
per-QP peak outstanding WRs/bytes, poll calls and empty ratio, CQE batch sizes,
poll busy time, and CQE latency avg/p50/p95/p99/max. It can surface
`request_not_ready`, `verbs_post_slow`, `cq_not_polled`,
`cq_polled_no_completion`, `cqe_returned_request_not_complete`, and WC/vendor
errors. `low_qp_feed_candidate` is deliberately a candidate diagnosis because
small transfers can legitimately keep only one WR outstanding.

Limitations are explicit: pre-posted receive queues expose exact CQE and global
RQ depth but not a one-to-one request post timestamp; Socket and third-party NET
plugins do not expose internal verbs objects; cross-node monotonic-clock
calibration is deferred. This boundary intentionally excludes mlx5 DV/DevX,
doorbells, packet PSNs, and NIC firmware internals.
