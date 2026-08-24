#!/usr/bin/env python3
import re, statistics, sys, json, argparse, os
from collections import defaultdict, Counter
work = re.compile(r"\[(\d+) ns\] WORK_(START|END) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
trans = re.compile(r"TRANSPORT .*type=(\w+)")
events, transports = defaultdict(dict), Counter()
collectives = defaultdict(lambda: {'ranks': set(), 'payload': 0, 'traffic': 0, 'first': [], 'last': []})
transfers = defaultdict(lambda: {'bytes': 0, 'ops': 0, 'first': None, 'last': None, 'peers': set()})
host_times = []
for path in sys.argv[1:]:
    with open(path, errors="replace") as f:
        for line in f:
            m = work.search(line)
            if m:
                host, kind, rank, ch, counter, timer = m.groups()
                events[(int(rank), int(ch), int(counter))][kind] = (int(timer), int(host))
                host_times.append((int(timer), int(host)))
            m = trans.search(line)
            if m: transports[m.group(1)] += 1
            m = re.search(r"TRANSFER op=(\d+) rank=(\d+) ch=(\d+) peer=(\d+) transport=(\d+) step=(\d+) bytes=(\d+)", line)
            if m:
                op, rank, ch, peer, transport, step, bytes_ = map(int, m.groups())
                t = transfers[(op, rank, ch)]; t['bytes'] += bytes_; t['ops'] += 1; t['peers'].add(peer)
                ts = int(line.split('[',1)[1].split(' ',1)[0]); t['first'] = ts if t['first'] is None else min(t['first'], ts); t['last'] = ts if t['last'] is None else max(t['last'], ts)
            m = re.search(r"COLLECTIVE id=(\d+) rank=(\d+) type=(\S+) payload=(\d+) traffic=(\d+)", line)
            if m:
                cid, rank, typ, payload, traffic = m.groups(); c = collectives[(int(cid), typ)]
                c['ranks'].add(int(rank)); c['payload'] = max(c['payload'], int(payload)); c['traffic'] = max(c['traffic'], int(traffic)); c['first'].append(int(line.split('[',1)[1].split(' ',1)[0]))
rows = [(k, v['END'][0]-v['START'][0], v['END'][1]-v['START'][1], v['START'][1], v['END'][1]) for k,v in events.items()
        if 'START' in v and 'END' in v and v['END'][0] >= v['START'][0]]
print('NCCL Telemetry Report')
print(f'matched_work_events={len(rows)} input_files={len(sys.argv)-1}')
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
for (op, rank, ch), t in sorted(transfers.items()):
    duration = max(1, (t['last'] or 0) - (t['first'] or 0)); bw = t['bytes'] * 1e9 / duration
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
if not rows: print('diagnostic=no_matched_gpu_work_pairs')
