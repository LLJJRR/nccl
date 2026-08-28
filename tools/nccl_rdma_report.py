#!/usr/bin/env python3
"""Join NCCL NET_PATH records with an RDMA counter diff.

The NCCL trace describes which NIC/port a connection selected, while the
counter snapshot describes what the NIC actually transmitted.  Keeping the
join in a post-processing tool avoids adding sysfs or verbs work to NCCL.
"""

import argparse
import json
import re
import sys
from collections import defaultdict


NET_PATH = re.compile(
    r"NET_PATH rank=(?P<rank>\d+) peer=(?P<peer>\d+) ch=(?P<ch>\d+) "
    r"conn=(?P<conn>\d+) direction=(?P<direction>SEND|RECV) "
    r"backend=(?P<backend>\w+) net_dev=(?P<net_dev>\d+) "
    r"net_id=(?P<net_id>0x[0-9a-fA-F]+) guid=(?P<guid>0x[0-9a-fA-F]+) "
    r"port=(?P<port>-?\d+) speed_mbps=(?P<speed>-?\d+) "
    r"rail=(?P<rail>-?\d+) plane=(?P<plane>-?\d+) "
    r"proxy_rank=(?P<proxy_rank>\d+) pxn=(?P<pxn>\d+) "
    r"gdr=(?P<gdr>\w+) gpu_nic_path=(?P<gpu_path>\w+) "
    r"path_type=(?P<path_type>\d+) shared=(?P<shared>\d+) "
    r"same_device=(?P<same_device>\d+)"
)
HEALTH_TERMS = ("retry", "retrans", "error", "discard", "drop", "cnp", "ecn",
                "pfc", "wait", "congestion")


def normalize_guid(value):
    if value is None:
        return None
    value = str(value).lower().replace(":", "")
    if value.startswith("0x"):
        value = value[2:]
    try:
        return f"0x{int(value, 16):016x}"
    except ValueError:
        return value


def load_ports(path):
    with open(path, errors="replace") as source:
        data = json.load(source)
    ports = defaultdict(list)
    for row in data.get("ports", []):
        key = (normalize_guid(row.get("guid")), int(row.get("port", -1)))
        ports[key].append(row)
    return ports


def read_paths(paths):
    records = []
    for path in paths:
        with open(path, errors="replace") as source:
            for line in source:
                match = NET_PATH.search(line)
                if not match:
                    continue
                record = match.groupdict()
                for key in ("rank", "peer", "ch", "conn", "net_dev", "port", "speed",
                            "rail", "plane", "proxy_rank", "pxn", "path_type",
                            "shared", "same_device"):
                    record[key] = int(record[key])
                record["guid"] = normalize_guid(record["guid"])
                records.append(record)
    return records


def build_report(paths, ports):
    rows = []
    for record in paths:
        matches = ports.get((record["guid"], record["port"]), [])
        row = dict(record)
        row["rdma_matches"] = matches
        row["health_signals"] = [
            {"device": match.get("device"), "port": match.get("port"),
             "counter": name, "delta": value}
            for match in matches
            for name, value in match.get("deltas", {}).items()
            if value and any(term in name.lower() for term in HEALTH_TERMS)
        ]
        row["diagnostic"] = None if matches else "unmatched_rdma_port"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Join NCCL NET_PATH with RDMA counter diff JSON")
    parser.add_argument("telemetry", nargs="+", help="NCCL telemetry text file(s)")
    parser.add_argument("--rdma-diff", required=True, help="JSON from nccl_rdma_counters.py diff --json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    try:
        rows = build_report(read_paths(args.telemetry), load_ports(args.rdma_diff))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        json.dump({"version": 1, "connections": rows}, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    print("NCCL/RDMA Path Report")
    if not rows:
        print("diagnostic=no_net_path_records")
        return 0
    for row in rows:
        route = f"{row['rank']}->{row['peer']}" if row["direction"] == "SEND" else \
                f"{row['peer']}->{row['rank']}"
        match_text = "none"
        if row["rdma_matches"]:
            match_text = ", ".join(
                f"{item.get('device')}:{item.get('port')} link_layer={item.get('link_layer')} "
                f"state={item.get('state')} rate={item.get('rate')} "
                f"tx={item.get('derived', {}).get('tx_bytes', 0)} "
                f"rx={item.get('derived', {}).get('rx_bytes', 0)}"
                for item in row["rdma_matches"]
            )
        print(f"route={route} ch={row['ch']} conn={row['conn']} backend={row['backend']} "
              f"gpu_nic_path={row['gpu_path']} guid={row['guid']} port={row['port']} "
              f"gdr={row['gdr']} rdma={match_text}")
        if row["diagnostic"]:
            print(f"diagnostic={row['diagnostic']} guid={row['guid']} port={row['port']}")
        for signal in row["health_signals"]:
            print(f"rdma_health_evidence device={signal['device']} port={signal['port']} "
                  f"counter={signal['counter']} delta={signal['delta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
