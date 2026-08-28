#!/usr/bin/env python3
"""Summarize a multi-node nccl-tests telemetry matrix."""

import argparse
import csv
import statistics
import sys
from collections import defaultdict


def mean(rows, field):
    return statistics.fmean(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="results.csv from nccl_multinode_matrix.sh")
    parser.add_argument("-o", "--output", help="summary CSV (default: stdout)")
    args = parser.parse_args()

    try:
        with open(args.input, newline="", errors="replace") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    groups = defaultdict(list)
    for row in rows:
        groups[(row["control"], row["collective"], row["size"], row["mode"])].append(row)

    fields = ["control", "collective", "size", "mode", "samples", "mean_time_us",
              "mean_algbw_gbps", "mean_busbw_gbps", "latency_overhead_pct",
              "busbw_change_pct", "wrong_total"]
    output = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            control, collective, size, mode = key
            samples = groups[key]
            baseline = groups.get((control, collective, size, "off"), [])
            time_us = mean(samples, "time_us")
            busbw = mean(samples, "busbw_gbps")
            latency_overhead = ""
            busbw_change = ""
            if mode != "off" and baseline:
                baseline_time = mean(baseline, "time_us")
                baseline_busbw = mean(baseline, "busbw_gbps")
                if baseline_time:
                    latency_overhead = f"{(time_us / baseline_time - 1.0) * 100.0:.3f}"
                if baseline_busbw:
                    busbw_change = f"{(busbw / baseline_busbw - 1.0) * 100.0:.3f}"
            writer.writerow({
                "control": control,
                "collective": collective,
                "size": size,
                "mode": mode,
                "samples": len(samples),
                "mean_time_us": f"{time_us:.6f}",
                "mean_algbw_gbps": f"{mean(samples, 'algbw_gbps'):.6f}",
                "mean_busbw_gbps": f"{busbw:.6f}",
                "latency_overhead_pct": latency_overhead,
                "busbw_change_pct": busbw_change,
                "wrong_total": sum(int(row["wrong"]) for row in samples),
            })
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
