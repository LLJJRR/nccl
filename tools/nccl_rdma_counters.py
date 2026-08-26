#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path


COUNTER_GROUPS = ("counters", "hw_counters")
DATA_COUNTERS = {
    "counters/port_xmit_data": "tx_bytes",
    "counters/port_rcv_data": "rx_bytes",
}


def read_text(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def read_counter(path):
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def normalize_guid(value):
    if value is None:
        return None
    compact = value.lower().replace(":", "")
    if compact.startswith("0x"):
        compact = compact[2:]
    try:
        return f"0x{int(compact, 16):016x}"
    except ValueError:
        return value


def take_snapshot(sysfs_root):
    devices = {}
    if not sysfs_root.is_dir():
        return {"version": 1, "timestamp_ns": time.time_ns(), "devices": devices}

    for device_path in sorted(path for path in sysfs_root.iterdir() if path.is_dir()):
        device = {
            "node_guid": normalize_guid(read_text(device_path / "node_guid")),
            "node_type": read_text(device_path / "node_type"),
            "sys_image_guid": normalize_guid(read_text(device_path / "sys_image_guid")),
            "ports": {},
        }
        ports_path = device_path / "ports"
        if ports_path.is_dir():
            for port_path in sorted((path for path in ports_path.iterdir() if path.is_dir()),
                                    key=lambda path: int(path.name)):
                port = {
                    "state": read_text(port_path / "state"),
                    "phys_state": read_text(port_path / "phys_state"),
                    "rate": read_text(port_path / "rate"),
                    "link_layer": read_text(port_path / "link_layer"),
                    "counters": {},
                }
                for group in COUNTER_GROUPS:
                    group_path = port_path / group
                    if not group_path.is_dir():
                        continue
                    for counter_path in sorted(path for path in group_path.iterdir() if path.is_file()):
                        value = read_counter(counter_path)
                        if value is not None:
                            port["counters"][f"{group}/{counter_path.name}"] = value
                device["ports"][port_path.name] = port
        devices[device_path.name] = device

    return {"version": 1, "timestamp_ns": time.time_ns(), "devices": devices}


def counter_deltas(before, after):
    rows = []
    device_names = sorted(set(before.get("devices", {})) | set(after.get("devices", {})))
    for device_name in device_names:
        before_device = before.get("devices", {}).get(device_name, {})
        after_device = after.get("devices", {}).get(device_name, {})
        port_names = sorted(set(before_device.get("ports", {})) | set(after_device.get("ports", {})),
                            key=int)
        for port_name in port_names:
            before_port = before_device.get("ports", {}).get(port_name, {})
            after_port = after_device.get("ports", {}).get(port_name, {})
            before_counters = before_port.get("counters", {})
            after_counters = after_port.get("counters", {})
            names = sorted(set(before_counters) | set(after_counters))
            deltas = {}
            resets = []
            for name in names:
                if name not in before_counters or name not in after_counters:
                    continue
                delta = after_counters[name] - before_counters[name]
                if delta < 0:
                    resets.append(name)
                    continue
                deltas[name] = delta
            derived = {}
            for counter, label in DATA_COUNTERS.items():
                if counter in deltas:
                    # InfiniBand port_{xmit,rcv}_data counters count 32-bit words.
                    derived[label] = deltas[counter] * 4
            rows.append({
                "device": device_name,
                "guid": after_device.get("node_guid") or before_device.get("node_guid"),
                "port": int(port_name),
                "link_layer": after_port.get("link_layer"),
                "state": after_port.get("state"),
                "rate": after_port.get("rate"),
                "deltas": deltas,
                "derived": derived,
                "resets": resets,
            })
    return rows


def write_json(data, output):
    serialized = json.dumps(data, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(serialized + "\n")
    else:
        print(serialized)


def snapshot_command(args):
    snapshot = take_snapshot(Path(args.sysfs_root))
    if not snapshot["devices"]:
        print(f"no RDMA devices found under {args.sysfs_root}", file=sys.stderr)
        return 1
    write_json(snapshot, args.output)
    return 0


def diff_command(args):
    with open(args.before, errors="replace") as before_file:
        before = json.load(before_file)
    with open(args.after, errors="replace") as after_file:
        after = json.load(after_file)
    rows = counter_deltas(before, after)
    if args.json:
        write_json({"version": 1, "ports": rows}, args.output)
        return 0

    output = open(args.output, "w") if args.output else sys.stdout
    try:
        print("RDMA Counter Diff", file=output)
        for row in rows:
            print(f'device={row["device"]} guid={row["guid"]} port={row["port"]} link_layer={row["link_layer"]} '
                  f'state={row["state"]} rate={row["rate"]}', file=output)
            for name, value in sorted(row["derived"].items()):
                print(f"  {name}={value}", file=output)
            for name, value in sorted(row["deltas"].items()):
                if args.all or value != 0:
                    print(f"  {name}={value}", file=output)
            for name in row["resets"]:
                print(f"  diagnostic=counter_reset counter={name}", file=output)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Snapshot and diff Linux RDMA port counters")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="capture counters as JSON")
    snapshot.add_argument("-o", "--output")
    snapshot.add_argument("--sysfs-root", default="/sys/class/infiniband")
    snapshot.set_defaults(func=snapshot_command)

    diff = subparsers.add_parser("diff", help="diff two JSON snapshots")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.add_argument("-o", "--output")
    diff.add_argument("--all", action="store_true", help="include zero-valued counters")
    diff.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    diff.set_defaults(func=diff_command)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(args.func(args))
