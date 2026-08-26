#!/usr/bin/env python3
import re, statistics, sys, json, argparse, os
from collections import defaultdict, Counter
if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} <telemetry.txt> [telemetry2.txt ...]', file=sys.stderr)
    sys.exit(2)
work = re.compile(r"\[(\d+) ns\] WORK_(START|END) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
trans = re.compile(r"TRANSPORT .*type=(\w+)")
events, transports = defaultdict(dict), Counter()
transport_edges = defaultdict(lambda: {'count': 0, 'channels': set(), 'conn': set()})
collectives = defaultdict(lambda: {'ranks': set(), 'payload': 0, 'traffic': 0, 'first': [], 'last': []})
transfers = defaultdict(lambda: {'bytes': 0, 'ops': 0, 'first': None, 'last': None, 'peers': set()})
peer_transfers = defaultdict(lambda: {'bytes': 0, 'ops': 0, 'steps': set()})
work_windows = defaultdict(lambda: {'first': None, 'last': None})
summaries = []
host_times = []
for path in sys.argv[1:]:
    with open(path, errors="replace") as f:
        for line in f:
            m = work.search(line)
            if m:
                host, kind, rank, ch, counter, timer = m.groups()
                events[(int(rank), int(ch), int(counter))][kind] = (int(timer), int(host))
                host_times.append((int(timer), int(host)))
                w = work_windows[(int(rank), int(ch))]
                if kind == 'START': w['first'] = int(host) if w['first'] is None else min(w['first'], int(host))
                else: w['last'] = int(host) if w['last'] is None else max(w['last'], int(host))
            m = trans.search(line)
            if m: transports[m.group(1)] += 1
            m = re.search(r"TRANSPORT rank=(\d+) peer=(\d+) ch=(\d+) conn=(\d+)(?: direction=(SEND|RECV))? type=(\w+) path=(\w+) path_type=(\d+)", line)
            if m:
                rank, peer, ch, conn, direction, typ, path, path_type = m.groups()
                direction = direction or 'UNKNOWN'
                if typ != 'P2P': path = typ
                edge = transport_edges[(int(rank), int(peer), direction, typ, path)]
                edge['count'] += 1; edge['channels'].add(int(ch)); edge['conn'].add(int(conn))
            m = re.search(r"TRANSFER op=(\d+) rank=(\d+) ch=(\d+) peer=(\d+) transport=(\d+) step=(\d+) bytes=(\d+)", line)
            if m:
                op, rank, ch, peer, transport, step, bytes_ = map(int, m.groups())
                t = transfers[(op, rank, ch)]; t['bytes'] += bytes_; t['ops'] += 1; t['peers'].add(peer)
                p = peer_transfers[(rank, ch, peer, transport)]; p['bytes'] += bytes_; p['ops'] += 1; p['steps'].add(step)
                ts = int(line.split('[',1)[1].split(' ',1)[0]); t['first'] = ts if t['first'] is None else min(t['first'], ts); t['last'] = ts if t['last'] is None else max(t['last'], ts)
            m = re.search(r"COLLECTIVE id=(\d+) rank=(\d+) type=(\S+) payload=(\d+) traffic=(\d+)", line)
            if m:
                cid, rank, typ, payload, traffic = m.groups(); c = collectives[(int(cid), typ)]
                c['ranks'].add(int(rank)); c['payload'] = max(c['payload'], int(payload)); c['traffic'] = max(c['traffic'], int(traffic)); c['first'].append(int(line.split('[',1)[1].split(' ',1)[0]))
            m = re.search(r"CHANNEL_SUMMARY coll=(\d+) plan=(\d+) rank=(\d+) ch=(\d+) work=(\d+) bytes=(\d+) duration_ns=(\d+)", line)
            if m:
                summaries.append(tuple(map(int, m.groups())))
rows = [(k, v['END'][0]-v['START'][0], v['END'][1]-v['START'][1], v['START'][1], v['END'][1]) for k,v in events.items()
        if 'START' in v and 'END' in v and v['END'][0] >= v['START'][0]]
print('NCCL Telemetry Report')
print(f'matched_work_events={len(rows)} input_files={len(sys.argv)-1}')
if summaries:
    print(f'channel_summaries={len(summaries)}')
    collective_windows = defaultdict(lambda: {'bytes': 0, 'duration': 0, 'channels': 0})
    for coll, plan, rank, ch, work_count, bytes_, duration in summaries:
        bw = float(bytes_) * 1e9 / max(1, duration)
        print(f'  channel_summary=coll:{coll}:plan:{plan}:rank:{rank}:ch:{ch}:work:{work_count}:bytes:{bytes_}:duration_ns:{duration}:bandwidth_Bps:{bw:.1f}')
        window = collective_windows[(coll, rank)]
        window['bytes'] += bytes_; window['duration'] = max(window['duration'], duration); window['channels'] += 1
    print('collective_effective_bandwidth=')
    for (coll, rank), window in sorted(collective_windows.items()):
        bw = float(window['bytes']) * 1e9 / max(1, window['duration'])
        print(f'  coll={coll} rank={rank} channels={window["channels"]} bytes={window["bytes"]} duration_ns={window["duration"]} bandwidth_Bps={bw:.1f}')
ring = []
with open(sys.argv[1], errors='replace') as f:
    for line in f:
        if ' RING ' in line: ring.append(line.strip())
if ring:
    print(f'topology_ring_edges={len(ring)}')
    for line in ring[:8]: print('  ' + line)
if host_times:
    tr = max(x[0] for x in host_times)-min(x[0] for x in host_times); hr = max(x[1] for x in host_times)-min(x[1] for x in host_times)
    print(f'gpu_timer_range_ticks={tr} host_range_ns={hr} calibrated_tick_ns={(hr/tr):.6f}' if tr else f'gpu_timer_range_ticks=0 host_range_ns={hr}')
if transports:
    print('transports=' + ', '.join(f'{k}:{v}' for k,v in sorted(transports.items())))
if transport_edges:
    print('transport_edges=')
    for (rank, peer, direction, typ, path), edge in sorted(transport_edges.items()):
        route = f'{rank}->{peer}' if direction == 'SEND' else f'{peer}->{rank}' if direction == 'RECV' else f'{rank}<->{peer}'
        print(f'  route={route} local_rank={rank} peer={peer} direction={direction} transport={typ} path={path} connections={edge["count"]} channels={sorted(edge["channels"])} conn_indices={sorted(edge["conn"])}')
if peer_transfers:
    print('peer_transfers=')
    for (rank, ch, peer, transport), item in sorted(peer_transfers.items()):
        print(f'  rank={rank} ch={ch} peer={peer} transport={transport} bytes={item["bytes"]} ops={item["ops"]} steps={len(item["steps"])}')
for (op, rank, ch), t in sorted(transfers.items()):
    ww = work_windows[(rank, ch)]
    duration = (ww['last'] - ww['first']) if ww['first'] is not None and ww['last'] is not None else (t['last'] or 0) - (t['first'] or 0)
    duration = max(1, int(duration)); bw = float(t['bytes']) * 1e9 / float(duration)
    print(f'channel_transfer=op:{op}:rank:{rank}:ch:{ch}:bytes:{t["bytes"]}:ops:{t["ops"]}:peers:{sorted(t["peers"])}:duration_ns:{duration}:bandwidth_Bps:{bw:.1f}')
by_rank = defaultdict(list)
for (rank, ch, counter), duration, host_duration, start_ns, end_ns in rows: by_rank[rank].append((ch, counter, duration, host_duration, start_ns, end_ns))
for rank, vals in sorted(by_rank.items()):
    ds = [x[2] for x in vals]; mean = statistics.mean(ds)
    imbalance = (max(ds)-min(ds))/mean if mean else 0
    print(f'rank={rank} channels={len(vals)} min_ticks={min(ds)} max_ticks={max(ds)} mean_ticks={mean:.1f} imbalance={imbalance:.3f}')
    for ch, counter, duration, host_duration, _, _ in sorted(vals, key=lambda x:x[2], reverse=True)[:5]:
        print(f'  slow_channel={ch} counter={counter} duration_ticks={duration} host_duration_ns={host_duration}')
    for counter in sorted(set(x[1] for x in vals)):
        batch = [x for x in vals if x[1] == counter]
        starts = [x[4] for x in batch]
        ends = [x[5] for x in batch]
        if len(batch) > 1 and max(starts) > min(starts):
            spread = max(starts)-min(starts); print(f'diagnostic=late_arrival:rank={rank}:counter={counter}:spread_ns={spread}')
            if spread > 1000000: print(f'diagnostic=late_arrival_strict:rank={rank}:counter={counter}:severity=high')
        if len(batch) > 1 and max(ends) > min(ends):
            print(f'diagnostic=completion_skew_proxy:rank={rank}:counter={counter}:spread_ns={max(ends)-min(ends)}')
    if imbalance > .10: print(f'diagnostic=channel_imbalance:rank={rank}:ratio={imbalance:.3f}')
    for ch, counter, duration, host_duration, _, _ in vals:
        if host_duration > 0 and duration == 0: print(f'diagnostic=timer_stall:rank={rank}:channel={ch}:counter={counter}:host_duration_ns={host_duration}')
if collectives:
    print('collective_summary=')
    for (cid, typ), c in sorted(collectives.items()):
        print(f'  id={cid} type={typ} ranks={len(c["ranks"])} payload={c["payload"]} traffic={c["traffic"]}')
if not rows and not summaries: print('diagnostic=no_matched_gpu_work_pairs')
