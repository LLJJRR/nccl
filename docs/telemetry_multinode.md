# Multi-node NCCL telemetry

This branch provides a low-overhead multi-node execution trace for
AllReduce, AllGather, and ReduceScatter.  The trace joins:

```
collective -> channel -> proxy -> NET_PATH -> NIC/port -> external RDMA counters
```

## Collection levels

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=0` records collective plans,
transport connections, and network path metadata.  This is the normal
production-safe level.

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=1` additionally records proxy
enqueue/start/first-progress/complete timestamps and channel summaries.  Use
this level to measure queue delay, first-progress delay, proxy duration, and
progress rate.

`NCCL_TELEMETRY_ENABLE=1 NCCL_TELEMETRY_LEVEL=2` adds per-step transfer and
GPU work events.  This is a diagnostic mode and should only be used for a
short reproduction because event volume is much higher.

All timestamps in the NCCL text trace are `CLOCK_MONOTONIC` nanoseconds.  The
trace header declares this clock domain and the supported capabilities.  Host
monotonic clocks are not assumed to be synchronized across nodes; align node
traces with an external clock offset when doing cross-node timeline analysis.

## RDMA correlation

Keep hardware counter reads outside NCCL's progress path:

```bash
tools/nccl_rdma_counters.py snapshot -o before.json
# run the two-node NCCL benchmark here
tools/nccl_rdma_counters.py snapshot -o after.json
tools/nccl_rdma_counters.py diff before.json after.json --json > rdma-diff.json
tools/nccl_rdma_report.py /tmp/nccl_telemetry.<pid>.txt --rdma-diff rdma-diff.json
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

The existing proxy timeline still distinguishes queue delay (enqueue to
start), first-progress delay (start to first completed transfer), and proxy
duration (start to completion).  This is sufficient to separate NCCL/proxy
scheduling delay from a path-level RDMA counter anomaly in the first delivery.

## Acceptance checklist

For a two-node multi-GPU run, the deliverable should include:

1. One trace per rank, merged with `tools/nccl_telemetry_merge.py`.
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
