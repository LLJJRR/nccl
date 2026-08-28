#!/usr/bin/env python3
import re, statistics, sys, json, argparse, os
from collections import defaultdict, Counter
if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} <telemetry.txt> [telemetry2.txt ...]', file=sys.stderr)
    sys.exit(2)
work = re.compile(r"\[(\d+) ns\] WORK_(START|END) coll=(\d+) plan=(\d+) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
legacy_work = re.compile(r"\[(\d+) ns\] WORK_(START|END) rank=(\d+) ch=(\d+) counter=(\d+) gpu_timer=(\d+)")
proxy_progress = re.compile(
    r"\[(\d+) ns\] PROXY_PROGRESS id=(\d+) coll=(\d+) plan=(\d+) rank=(\d+) ch=(\d+) "
    r"peer=(\d+) direction=(SEND|RECV|UNKNOWN) transport=(\d+) phase=(\d+) status=(\d+) "
    r"expected_bytes=(\d+) progressed_bytes=(\d+)"
)
rdma_request = re.compile(
    r"\[(\d+) ns\] RDMA_REQUEST proxy=(\d+) coll=(\d+) plan=(\S+) rank=(\d+) ch=(\d+) "
    r"peer=(\d+) direction=(SEND|RECV|UNKNOWN) request=(\d+) qp=(\d+) wr_id=(\d+) "
    r"opcode=(\d+) status=(\d+) phase=(\d+) "
    r"(?:owner_index=(\d+) owner_count=(\d+) )?"
    r"(?:completion_expected=(\d+) )?bytes=(\d+)"
)
events, transports = defaultdict(dict), Counter()
headers = []
transport_edges = defaultdict(lambda: {'count': 0, 'channels': set(), 'conn': set()})
network_paths = defaultdict(lambda: {'count': 0, 'channels': set(), 'conn': set(), 'speed': 0})
collectives = defaultdict(lambda: {'ranks': set(), 'payload': 0, 'traffic': 0, 'first': [], 'last': []})
channel_plans = {}
transfers = defaultdict(lambda: {'bytes': 0, 'ops': 0, 'first': None, 'last': None,
                                 'peers': set(), 'timestamps': []})
peer_transfers = defaultdict(lambda: {'bytes': 0, 'ops': 0, 'steps': set()})
work_windows = defaultdict(lambda: {'first': None, 'last': None})
summaries = []
host_times = defaultdict(list)
proxy_events = defaultdict(list)
rdma_events = defaultdict(list)
for path in sys.argv[1:]:
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith('HEADER '):
                values = dict(re.findall(r'(\w+)=([^ ]+)', line.strip()))
                values['source'] = path
                headers.append(values)
                continue
            m = work.search(line)
            if m:
                host, kind, coll, plan, rank, ch, counter, timer = m.groups()
                coll, plan = int(coll), int(plan)
            else:
                m = legacy_work.search(line)
                if m:
                    host, kind, rank, ch, counter, timer = m.groups()
                    coll, plan = -1, 0
            if m:
                events[(coll, plan, int(rank), int(ch), int(counter))][kind] = (int(timer), int(host))
                host_times[path].append((int(timer), int(host)))
                w = work_windows[(coll, plan, int(rank), int(ch))]
                if kind == 'START': w['first'] = int(host) if w['first'] is None else min(w['first'], int(host))
                else: w['last'] = int(host) if w['last'] is None else max(w['last'], int(host))
            m = proxy_progress.search(line)
            if m:
                (ts, proxy_id, coll, plan, rank, ch, peer, direction, transport, phase,
                 status, expected, progressed) = m.groups()
                proxy_events[(int(rank), int(proxy_id))].append({
                    'timestamp': int(ts), 'collective': int(coll), 'plan': int(plan),
                    'rank': int(rank), 'channel': int(ch), 'peer': int(peer),
                    'direction': direction, 'transport': int(transport), 'phase': int(phase),
                    'status': int(status), 'expected': int(expected), 'progressed': int(progressed)
                })
            m = rdma_request.search(line)
            if m:
                (ts, proxy_id, coll, plan, rank, ch, peer, direction, request_id, qp, wr_id,
                 opcode, status, phase, owner_index, owner_count, completion_expected,
                 bytes_) = m.groups()
                rdma_events[(int(rank), int(proxy_id))].append({
                    'timestamp': int(ts), 'collective': int(coll), 'plan': plan,
                    'rank': int(rank), 'channel': int(ch), 'peer': int(peer), 'direction': direction,
                    'request': int(request_id), 'qp': int(qp), 'wr_id': int(wr_id),
                    'opcode': int(opcode), 'status': int(status), 'phase': int(phase),
                    'owner_index': int(owner_index or 0), 'owner_count': int(owner_count or 1),
                    'completion_expected': int(completion_expected) if completion_expected is not None else 1,
                    'bytes': int(bytes_)
                })
            m = re.search(r"TRANSPORT rank=(\d+) peer=(\d+) ch=(\d+) conn=(\d+)(?: direction=(SEND|RECV))? type=(\w+) path=(\w+) path_type=(\d+)", line)
            if m:
                rank, peer, ch, conn, direction, typ, path, path_type = m.groups()
                direction = direction or 'UNKNOWN'
                transports[typ] += 1
                if typ != 'P2P': path = typ
                edge = transport_edges[(int(rank), int(peer), direction, typ, path)]
                edge['count'] += 1; edge['channels'].add(int(ch)); edge['conn'].add(int(conn))
            m = re.search(r"NET_PATH rank=(\d+) peer=(\d+) ch=(\d+) conn=(\d+) direction=(SEND|RECV) backend=(\w+) net_dev=(\d+) net_id=(0x[0-9a-fA-F]+) guid=(0x[0-9a-fA-F]+) port=(-?\d+) speed_mbps=(-?\d+) rail=(-?\d+) plane=(-?\d+) proxy_rank=(\d+) pxn=(\d+) gdr=(\w+) gpu_nic_path=(\w+) path_type=(\d+) shared=(\d+) same_device=(\d+)", line)
            if m:
                rank, peer, ch, conn, direction, backend, net_dev, net_id, guid, port, speed, rail, plane, proxy_rank, pxn, gdr, gpu_nic_path, path_type, shared, same_device = m.groups()
                key = (int(rank), int(peer), direction, backend, int(net_dev), net_id, guid,
                       int(port), int(rail), int(plane), int(proxy_rank), int(pxn), gdr,
                       gpu_nic_path, int(path_type), int(shared), int(same_device))
                path = network_paths[key]; path['count'] += 1; path['channels'].add(int(ch)); path['conn'].add(int(conn)); path['speed'] = int(speed)
            m = re.search(r"TRANSFER coll=(\d+) plan=(\d+) op=(\d+) rank=(\d+) ch=(\d+) direction=(SEND|RECV|UNKNOWN) peer=(\d+) transport=(\d+) step=(\d+) bytes=(\d+)", line)
            if m:
                coll, plan, op, rank, ch, direction, peer, transport, step, bytes_ = m.groups()
                coll, plan, op, rank, ch, peer, transport, step, bytes_ = map(int, (coll, plan, op, rank, ch, peer, transport, step, bytes_))
                t = transfers[(coll, plan, op, rank, ch, direction)]; t['bytes'] += bytes_; t['ops'] += 1; t['peers'].add(peer)
                p = peer_transfers[(coll, plan, rank, ch, peer, direction, transport)]; p['bytes'] += bytes_; p['ops'] += 1; p['steps'].add(step)
                ts = int(line.split('[',1)[1].split(' ',1)[0]); t['first'] = ts if t['first'] is None else min(t['first'], ts); t['last'] = ts if t['last'] is None else max(t['last'], ts)
                t['timestamps'].append(ts)
            else:
                m = re.search(r"TRANSFER op=(\d+) rank=(\d+) ch=(\d+) peer=(\d+) transport=(\d+) step=(\d+) bytes=(\d+)", line)
                if m:
                    op, rank, ch, peer, transport, step, bytes_ = map(int, m.groups())
                    coll, plan, direction = -1, 0, 'UNKNOWN'
                    t = transfers[(coll, plan, op, rank, ch, direction)]; t['bytes'] += bytes_; t['ops'] += 1; t['peers'].add(peer)
                    p = peer_transfers[(coll, plan, rank, ch, peer, direction, transport)]; p['bytes'] += bytes_; p['ops'] += 1; p['steps'].add(step)
                    ts = int(line.split('[',1)[1].split(' ',1)[0]); t['first'] = ts if t['first'] is None else min(t['first'], ts); t['last'] = ts if t['last'] is None else max(t['last'], ts)
                    t['timestamps'].append(ts)
            m = re.search(r"COLLECTIVE id=(\d+) comm_id=(0x[0-9a-fA-F]+) nranks=(\d+) rank=(\d+) type=(\S+) payload=(\d+) traffic=(\d+)", line)
            if m:
                cid, comm_id, nranks, rank, typ, payload, traffic = m.groups(); c = collectives[(comm_id, int(cid), typ)]
                c['ranks'].add(int(rank)); c['payload'] = max(c['payload'], int(payload)); c['traffic'] = max(c['traffic'], int(traffic)); c['first'].append(int(line.split('[',1)[1].split(' ',1)[0]))
                c['comm_id'], c['nranks'] = comm_id, int(nranks)
            else:
                m = re.search(r"COLLECTIVE id=(\d+) rank=(\d+) type=(\S+) payload=(\d+) traffic=(\d+)", line)
                if m:
                    cid, rank, typ, payload, traffic = m.groups(); c = collectives[('UNKNOWN', int(cid), typ)]
                    c['ranks'].add(int(rank)); c['payload'] = max(c['payload'], int(payload)); c['traffic'] = max(c['traffic'], int(traffic)); c['first'].append(int(line.split('[',1)[1].split(' ',1)[0]))
            m = re.search(r"CHANNEL coll=(\d+) plan=(\d+) rank=(\d+) ch=(\d+) type=(\S+) algo=(\S+) proto=(\S+) payload=(\d+) traffic=(\d+)", line)
            if m:
                coll, plan, rank, ch, typ, algo, proto, payload, traffic = m.groups()
                channel_plans[(int(coll), int(plan), int(rank), int(ch))] = {
                    'type': typ, 'algo': algo, 'proto': proto,
                    'payload': int(payload), 'traffic': int(traffic)
                }
            m = re.search(r"CHANNEL_SUMMARY coll=(\d+) plan=(\d+) rank=(\d+) ch=(\d+) work=(\d+) bytes=(\d+) duration_ns=(\d+)", line)
            if m:
                summaries.append(tuple(map(int, m.groups())))
rows = [(k, v['END'][0]-v['START'][0], v['END'][1]-v['START'][1], v['START'][1], v['END'][1]) for k,v in events.items()
        if 'START' in v and 'END' in v and v['END'][0] >= v['START'][0]]
print('NCCL Telemetry Report')
print(f'matched_work_events={len(rows)} input_files={len(sys.argv)-1}')
dropped = 0
if headers:
    dropped = sum(int(header.get('dropped', 0)) for header in headers)
    levels = sorted(set(header.get('level', 'unknown') for header in headers))
    print(f'trace_headers={len(headers)} levels={levels} dropped_events={dropped}')
    if dropped:
        print(f'diagnostic=telemetry_events_dropped:count={dropped}')
if proxy_events:
    print(f'proxy_operations={len(proxy_events)}')
    for (proxy_rank, proxy_id), entries in sorted(proxy_events.items()):
        by_phase = {entry['phase']: entry for entry in entries}
        identity = next((entry for entry in entries
                         if entry['direction'] != 'UNKNOWN' or entry['transport'] != 0), entries[0])
        enqueue = by_phase.get(0)
        start = by_phase.get(1)
        first = by_phase.get(2)
        complete = by_phase.get(3)
        error = by_phase.get(4)
        queue_delay = start['timestamp'] - enqueue['timestamp'] if enqueue and start else None
        first_delay = first['timestamp'] - start['timestamp'] if first and start else None
        duration = complete['timestamp'] - start['timestamp'] if complete and start else None
        expected = (complete or first or start or enqueue)['expected']
        progressed = (complete or first or start or enqueue)['progressed']
        rate = float(progressed) * 1e9 / duration if duration and duration > 0 else None
        fields = [f'id={proxy_id}', f"coll={identity['collective']}",
                  f"plan={identity['plan']}", f"rank={identity['rank']}",
                  f"channel={identity['channel']}", f"peer={identity['peer']}",
                  f"direction={identity['direction']}", f"transport={identity['transport']}",
                  f"expected_bytes={expected}"]
        if queue_delay is not None: fields.append(f'queue_delay_ns={queue_delay}')
        if first_delay is not None: fields.append(f'first_progress_delay_ns={first_delay}')
        if duration is not None: fields.append(f'proxy_duration_ns={duration}')
        if rate is not None: fields.append(f'progress_rate_Bps={rate:.1f}')
        if error: fields.append(f"error_status={error['status']}")
        rdma = rdma_events.get((proxy_rank, proxy_id), [])
        rdma = [event for event in rdma if event['rank'] == identity['rank'] and
                event['channel'] == identity['channel']]
        if rdma:
            posts = [event for event in rdma if event['phase'] == 0]
            cqes = [event for event in rdma if event['phase'] == 1]
            poll_delays = [event for event in rdma if event['phase'] == 3]
            qps = sorted(set(event['qp'] for event in rdma if event['qp']))
            statuses = sorted(set(event['status'] for event in cqes if event['status']))
            grouped = [event for event in rdma if event['owner_count'] > 1]
            signaled_posts = [event for event in posts if event['completion_expected']]
            unsignaled_posts = [event for event in posts if not event['completion_expected']]
            post_times = defaultdict(list)
            for event in signaled_posts:
                post_times[(event['request'], event['qp'], event['wr_id'])].append(event['timestamp'])
            completion_latencies = []
            for event in cqes:
                key = (event['request'], event['qp'], event['wr_id'])
                if post_times[key]:
                    completion_latencies.append(event['timestamp'] - post_times[key].pop(0))
            unmatched_posts = sum(len(times) for times in post_times.values())
            fields.append(f'rdma_wqe_posts={len(posts)}')
            fields.append(f'rdma_cqes={len(cqes)}')
            fields.append(f'rdma_signaled_posts={len(signaled_posts)}')
            fields.append(f'rdma_unsignaled_posts={len(unsignaled_posts)}')
            fields.append(f'rdma_qps={qps}')
            if completion_latencies:
                fields.append(f'rdma_max_completion_ns={max(completion_latencies)}')
            fields.append(f'rdma_unmatched_posts={unmatched_posts}')
            if poll_delays:
                fields.append(f'rdma_injected_poll_delay_us={sum(event["bytes"] for event in poll_delays)}')
                fields.append('diagnostic=injected_cq_poll_delay')
            if grouped:
                fields.append(f'rdma_grouped_events={len(grouped)}')
                fields.append(f'rdma_max_owner_count={max(event["owner_count"] for event in grouped)}')
            if statuses: fields.append(f'rdma_statuses={statuses}')
            if statuses: fields.append('diagnostic=rdma_wc_error')
            if unmatched_posts and not dropped:
                fields.append('diagnostic=missing_signaled_cqe')
        if not complete and not error: fields.append('diagnostic=incomplete_proxy')
        print('  proxy_timeline=' + ' '.join(fields))
if summaries:
    print(f'channel_summaries={len(summaries)}')
    collective_windows = defaultdict(lambda: {'bytes': 0, 'duration': 0, 'channels': 0})
    for coll, plan, rank, ch, work_count, bytes_, duration in summaries:
        meta = channel_plans.get((coll, plan, rank, ch), {})
        typ, algo, proto = meta.get('type', 'UNKNOWN'), meta.get('algo', 'UNKNOWN'), meta.get('proto', 'UNKNOWN')
        bw = float(bytes_) * 1e9 / max(1, duration)
        print(f'  channel_summary=coll:{coll}:plan:{plan}:rank:{rank}:ch:{ch}:type:{typ}:algo:{algo}:proto:{proto}:work:{work_count}:bytes:{bytes_}:duration_ns:{duration}:bandwidth_Bps:{bw:.1f}')
        window = collective_windows[(coll, plan, rank, typ, algo, proto)]
        window['bytes'] += bytes_; window['duration'] = max(window['duration'], duration); window['channels'] += 1
    print('collective_effective_bandwidth=')
    for (coll, plan, rank, typ, algo, proto), window in sorted(collective_windows.items()):
        bw = float(window['bytes']) * 1e9 / max(1, window['duration'])
        print(f'  coll={coll} plan={plan} rank={rank} type={typ} algo={algo} proto={proto} channels={window["channels"]} bytes={window["bytes"]} duration_ns={window["duration"]} bandwidth_Bps={bw:.1f}')
if channel_plans:
    planned_collectives = defaultdict(lambda: {'channels': 0, 'payload': 0, 'traffic': 0})
    for (coll, plan, rank, ch), meta in channel_plans.items():
        item = planned_collectives[(coll, plan, rank, meta['type'], meta['algo'], meta['proto'])]
        item['channels'] += 1; item['payload'] += meta['payload']; item['traffic'] += meta['traffic']
    print('collective_channel_plans=')
    for (coll, plan, rank, typ, algo, proto), item in sorted(planned_collectives.items()):
        print(f'  coll={coll} plan={plan} rank={rank} type={typ} algo={algo} proto={proto} channels={item["channels"]} payload={item["payload"]} traffic={item["traffic"]}')
ring = []
with open(sys.argv[1], errors='replace') as f:
    for line in f:
        if ' RING ' in line: ring.append(line.strip())
if ring:
    print(f'topology_ring_edges={len(ring)}')
    for line in ring[:8]: print('  ' + line)
for source, samples in sorted(host_times.items()):
    tr = max(x[0] for x in samples)-min(x[0] for x in samples); hr = max(x[1] for x in samples)-min(x[1] for x in samples)
    prefix = f'source={os.path.basename(source)} '
    print(prefix + (f'gpu_timer_range_ticks={tr} host_range_ns={hr} calibrated_tick_ns={(hr/tr):.6f}' if tr else f'gpu_timer_range_ticks=0 host_range_ns={hr}'))
if transports:
    print('transports=' + ', '.join(f'{k}:{v}' for k,v in sorted(transports.items())))
if transport_edges:
    print('transport_edges=')
    for (rank, peer, direction, typ, path), edge in sorted(transport_edges.items()):
        route = f'{rank}->{peer}' if direction == 'SEND' else f'{peer}->{rank}' if direction == 'RECV' else f'{rank}<->{peer}'
        print(f'  route={route} local_rank={rank} peer={peer} direction={direction} transport={typ} path={path} connections={edge["count"]} channels={sorted(edge["channels"])} conn_indices={sorted(edge["conn"])}')
if network_paths:
    print('network_paths=')
    for key, item in sorted(network_paths.items()):
        rank, peer, direction, backend, net_dev, net_id, guid, port, rail, plane, proxy_rank, pxn, gdr, gpu_nic_path, path_type, shared, same_device = key
        route = f'{rank}->{peer}' if direction == 'SEND' else f'{peer}->{rank}'
        print(f'  route={route} local_rank={rank} peer={peer} direction={direction} backend={backend} net_dev={net_dev} net_id={net_id} guid={guid} port={port} speed_mbps={item["speed"]} rail={rail} plane={plane} proxy_rank={proxy_rank} pxn={pxn} gdr={gdr} gpu_nic_path={gpu_nic_path} path_type={path_type} shared={shared} same_device={same_device} connections={item["count"]} channels={sorted(item["channels"])} conn_indices={sorted(item["conn"])}')
if peer_transfers:
    print('peer_transfers=')
    for (coll, plan, rank, ch, peer, direction, transport), item in sorted(peer_transfers.items()):
        print(f'  coll={coll} plan={plan} rank={rank} ch={ch} peer={peer} direction={direction} transport={transport} bytes={item["bytes"]} ops={item["ops"]} steps={len(item["steps"])}')
for (coll, plan, op, rank, ch, direction), t in sorted(transfers.items()):
    ww = work_windows[(coll, plan, rank, ch)]
    duration = (ww['last'] - ww['first']) if ww['first'] is not None and ww['last'] is not None else (t['last'] or 0) - (t['first'] or 0)
    duration = max(1, int(duration)); bw = float(t['bytes']) * 1e9 / float(duration)
    ordered_times = sorted(t['timestamps'])
    max_gap = max((b-a for a, b in zip(ordered_times, ordered_times[1:])), default=0)
    print(f'channel_transfer=coll:{coll}:plan:{plan}:op:{op}:rank:{rank}:ch:{ch}:direction:{direction}:bytes:{t["bytes"]}:ops:{t["ops"]}:peers:{sorted(t["peers"])}:duration_ns:{duration}:max_progress_gap_ns:{max_gap}:bandwidth_Bps:{bw:.1f}')
    if channel_plans and coll not in (-1, (1 << 64) - 1) and (coll, plan, rank, ch) not in channel_plans:
        print(f'diagnostic=unmatched_transfer:coll={coll}:plan={plan}:rank={rank}:channel={ch}:op={op}')
by_collective_rank = defaultdict(list)
for (coll, plan, rank, ch, counter), duration, host_duration, start_ns, end_ns in rows:
    by_collective_rank[(coll, plan, rank)].append((ch, counter, duration, host_duration, start_ns, end_ns))
for (coll, plan, rank), vals in sorted(by_collective_rank.items()):
    ds = [x[2] for x in vals]; mean = statistics.mean(ds)
    imbalance = (max(ds)-min(ds))/mean if mean else 0
    print(f'coll={coll} plan={plan} rank={rank} channels={len(vals)} min_ticks={min(ds)} max_ticks={max(ds)} mean_ticks={mean:.1f} imbalance={imbalance:.3f}')
    for ch, counter, duration, host_duration, _, _ in sorted(vals, key=lambda x:x[2], reverse=True)[:5]:
        print(f'  slow_channel={ch} coll={coll} plan={plan} counter={counter} duration_ticks={duration} host_duration_ns={host_duration}')
    for counter in sorted(set(x[1] for x in vals)):
        batch = [x for x in vals if x[1] == counter]
        starts = [x[4] for x in batch]
        ends = [x[5] for x in batch]
        if len(batch) > 1 and max(starts) > min(starts):
            spread = max(starts)-min(starts); print(f'diagnostic=late_arrival:coll={coll}:plan={plan}:rank={rank}:counter={counter}:spread_ns={spread}')
            if spread > 1000000: print(f'diagnostic=late_arrival_strict:coll={coll}:plan={plan}:rank={rank}:counter={counter}:severity=high')
        if len(batch) > 1 and max(ends) > min(ends):
            print(f'diagnostic=completion_skew_proxy:coll={coll}:plan={plan}:rank={rank}:counter={counter}:spread_ns={max(ends)-min(ends)}')
    if imbalance > .10: print(f'diagnostic=channel_imbalance:coll={coll}:plan={plan}:rank={rank}:ratio={imbalance:.3f}')
    for ch, counter, duration, host_duration, _, _ in vals:
        if host_duration > 0 and duration == 0: print(f'diagnostic=timer_stall:coll={coll}:plan={plan}:rank={rank}:channel={ch}:counter={counter}:host_duration_ns={host_duration}')
if collectives:
    print('collective_summary=')
    for (comm_id, cid, typ), c in sorted(collectives.items()):
        print(f'  comm_id={comm_id} id={cid} type={typ} ranks={len(c["ranks"])} nranks={c.get("nranks", "UNKNOWN")} payload={c["payload"]} traffic={c["traffic"]}')
if not rows and not summaries: print('diagnostic=no_matched_gpu_work_pairs')
