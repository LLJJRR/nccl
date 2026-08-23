#!/usr/bin/env python3
import re, statistics, sys, json, argparse
from collections import defaultdict, Counter
work = re.compile(r"\[(\d+) ns\] WORK_(START|END) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
trans = re.compile(r"TRANSPORT .*type=(\w+)")
events, transports = defaultdict(dict), Counter()
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
rows = [(k, v['END'][0]-v['START'][0], v['END'][1]-v['START'][1], v['START'][1], v['END'][1]) for k,v in events.items()
        if 'START' in v and 'END' in v and v['END'][0] >= v['START'][0]]
print('NCCL Telemetry Report')
print(f'matched_work_events={len(rows)}')
ring = []
with open(sys.argv[1], errors='replace') as f:
    for line in f:
        if ' RING ' in line: ring.append(line.strip())
if ring:
    print(f'topology_ring_edges={len(ring)}')
    for line in ring[:8]: print('  ' + line)
if host_times:
    print(f'gpu_timer_range_ticks={max(x[0] for x in host_times)-min(x[0] for x in host_times)} host_range_ns={max(x[1] for x in host_times)-min(x[1] for x in host_times)}')
if transports:
    print('transports=' + ', '.join(f'{k}:{v}' for k,v in sorted(transports.items())))
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
            print(f'diagnostic=late_arrival:rank={rank}:counter={counter}:spread_ns={max(starts)-min(starts)}')
        if len(batch) > 1 and max(ends) > min(ends):
            print(f'diagnostic=completion_skew:rank={rank}:counter={counter}:spread_ns={max(ends)-min(ends)}')
    if imbalance > .10: print(f'diagnostic=channel_imbalance:rank={rank}:ratio={imbalance:.3f}')
if not rows: print('diagnostic=no_matched_gpu_work_pairs')
