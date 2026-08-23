#!/usr/bin/env python3
import re, statistics, sys
from collections import defaultdict, Counter
work = re.compile(r"WORK_(START|END) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
trans = re.compile(r"TRANSPORT .*type=(\w+)")
events, transports = defaultdict(dict), Counter()
for path in sys.argv[1:]:
    with open(path, errors="replace") as f:
        for line in f:
            m = work.search(line)
            if m:
                kind, rank, ch, counter, timer = m.groups()
                events[(int(rank), int(ch), int(counter))][kind] = int(timer)
            m = trans.search(line)
            if m: transports[m.group(1)] += 1
rows = [(k, v['END']-v['START']) for k,v in events.items()
        if 'START' in v and 'END' in v and v['END'] >= v['START']]
print('NCCL Telemetry Report')
print(f'matched_work_events={len(rows)}')
if transports:
    print('transports=' + ', '.join(f'{k}:{v}' for k,v in sorted(transports.items())))
by_rank = defaultdict(list)
for (rank, ch, counter), duration in rows: by_rank[rank].append((ch, counter, duration))
for rank, vals in sorted(by_rank.items()):
    ds = [x[2] for x in vals]; mean = statistics.mean(ds)
    imbalance = (max(ds)-min(ds))/mean if mean else 0
    print(f'rank={rank} channels={len(vals)} min_ticks={min(ds)} max_ticks={max(ds)} mean_ticks={mean:.1f} imbalance={imbalance:.3f}')
    for ch, counter, duration in sorted(vals, key=lambda x:x[2], reverse=True)[:5]:
        print(f'  slow_channel={ch} counter={counter} duration_ticks={duration}')
    if imbalance > .10: print(f'diagnostic=channel_imbalance:rank={rank}:ratio={imbalance:.3f}')
if not rows: print('diagnostic=no_matched_gpu_work_pairs')
