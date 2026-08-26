/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#include "telemetry.h"
#include "param.h"
#include "collectives.h"
#include "graph.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

// Basic host-side flight recording is on by default. Set NCCL_TELEMETRY_ENABLE=0 to disable it.
NCCL_PARAM(TelemetryEnable, "TELEMETRY_ENABLE", 1);
NCCL_PARAM(TelemetryLevel, "TELEMETRY_LEVEL", 0);

namespace {
constexpr uint64_t kCapacity = 1ull << 16;
struct TelemetryBuffer {
  std::atomic<uint64_t> next{0};
  std::atomic<uint64_t> dropped{0};
  ncclTelemetryEvent events[kCapacity];
};

TelemetryBuffer buffer;
std::atomic<bool> initialized{false};
std::atomic<uint64_t> collectiveIds{0};
std::mutex stateMutex;

struct ChannelState {
  uint64_t totalBytes = 0;
  uint64_t workCount = 0;
  uint64_t firstStart = 0;
  uint64_t lastEnd = 0;
};
struct ChannelPlanKey {
  uint64_t collectiveId;
  uint64_t planId;
  uint32_t rank;
  uint16_t channel;
  bool operator==(const ChannelPlanKey& other) const {
    return collectiveId == other.collectiveId && planId == other.planId &&
           rank == other.rank && channel == other.channel;
  }
};
struct ChannelPlanKeyHash {
  size_t operator()(const ChannelPlanKey& key) const {
    size_t h = std::hash<uint64_t>{}(key.collectiveId);
    h ^= std::hash<uint64_t>{}(key.planId) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>{}(key.rank) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h ^ (size_t(key.channel) * 0x9e3779b9);
  }
};
std::unordered_set<ChannelPlanKey, ChannelPlanKeyHash> channelPlans;
std::unordered_map<ChannelPlanKey, ChannelState, ChannelPlanKeyHash> channelStates;

struct WorkKey {
  uint64_t counter;
  uint32_t rank;
  uint16_t channel;
  bool operator==(const WorkKey& other) const {
    return counter == other.counter && rank == other.rank && channel == other.channel;
  }
};
struct WorkKeyHash {
  size_t operator()(const WorkKey& key) const {
    size_t h = std::hash<uint64_t>{}(key.counter);
    h ^= std::hash<uint32_t>{}(key.rank) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h ^ (size_t(key.channel) * 0x9e3779b9);
  }
};
struct WorkState {
  ChannelPlanKey channel;
  uint64_t bytes;
  uint64_t start;
};
std::unordered_map<WorkKey, WorkState, WorkKeyHash> workStates;
std::vector<ncclTelemetryEvent> pendingSummaries;

uint64_t nowNs();
void recordBatch(const ncclTelemetryEvent* input, size_t count);

void appendChannelSummary(std::vector<ncclTelemetryEvent>& events, const ChannelPlanKey& key,
                          ChannelState& state) {
  if (state.workCount == 0) return;
  ncclTelemetryEvent e{};
  e.timestampNs = nowNs();
  e.collectiveId = key.collectiveId;
  e.planId = key.planId;
  e.rank = key.rank;
  e.channel = key.channel;
  e.eventType = NCCL_TELEM_CHANNEL_SUMMARY;
  e.payloadBytes = state.totalBytes;
  e.trafficBytes = state.lastEnd >= state.firstStart ? state.lastEnd - state.firstStart : 0;
  e.value = state.workCount;
  events.push_back(e);
}

void flushChannelStatesLocked() {
  pendingSummaries.reserve(pendingSummaries.size() + channelStates.size());
  for (auto& item : channelStates) appendChannelSummary(pendingSummaries, item.first, item.second);
  channelStates.clear();
  recordBatch(pendingSummaries.data(), pendingSummaries.size());
  pendingSummaries.clear();
}

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

void dump() {
  if (!initialized.load(std::memory_order_relaxed)) return;
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    flushChannelStatesLocked();
  }
  char path[128];
  snprintf(path, sizeof(path), "/tmp/nccl_telemetry.%d.bin", int(getpid()));
  FILE* f = fopen(path, "wb");
  if (f == nullptr) return;
  char textPath[128];
  snprintf(textPath, sizeof(textPath), "/tmp/nccl_telemetry.%d.txt", int(getpid()));
  FILE* text = fopen(textPath, "w");
  uint64_t end = buffer.next.load(std::memory_order_acquire);
  uint64_t begin = end > kCapacity ? end - kCapacity : 0;
  uint64_t magic = 0x4e43434c54454c31ull; // NC CLTEL1
  uint32_t version = 1;
  uint32_t eventSize = sizeof(ncclTelemetryEvent);
  uint64_t dropped = buffer.dropped.load(std::memory_order_relaxed);
  fwrite(&magic, sizeof(magic), 1, f);
  fwrite(&version, sizeof(version), 1, f);
  fwrite(&eventSize, sizeof(eventSize), 1, f);
  fwrite(&dropped, sizeof(dropped), 1, f);
  if (text != nullptr) fprintf(text, "HEADER version=%u event_size=%u events=%llu dropped=%llu\n",
                               version, eventSize, (unsigned long long)(end - begin),
                               (unsigned long long)dropped);
  for (uint64_t i = begin; i < end; ++i) {
    const ncclTelemetryEvent& e = buffer.events[i % kCapacity];
    fwrite(&e, sizeof(ncclTelemetryEvent), 1, f);
    if (text == nullptr) continue;
    switch (e.eventType) {
    case NCCL_TELEM_COLLECTIVE_ENQUEUE:
      fprintf(text, "[%llu ns] COLLECTIVE id=%llu rank=%u type=%s payload=%llu traffic=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId, e.rank,
              ncclFuncToString((ncclFunc_t)e.collective), (unsigned long long)e.payloadBytes,
              (unsigned long long)e.trafficBytes);
      break;
    case NCCL_TELEM_CHANNEL_PLAN:
      fprintf(text, "[%llu ns] CHANNEL coll=%llu plan=%llu rank=%u ch=%u type=%s algo=%s proto=%s payload=%llu traffic=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, ncclFuncToString((ncclFunc_t)e.collective),
              ncclAlgoToString(e.algorithm), ncclProtoToString(e.protocol),
              (unsigned long long)e.payloadBytes, (unsigned long long)e.trafficBytes);
      break;
    case NCCL_TELEM_PLAN_LAUNCH:
      fprintf(text, "[%llu ns] PLAN id=%llu rank=%u channels=0x%llx\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.planId, e.rank,
              (unsigned long long)e.value);
      break;
    case NCCL_TELEM_PROXY_OP:
      fprintf(text, "[%llu ns] PROXY plan=%llu rank=%u ch=%u pattern=%u bytes=%llu op=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.planId, e.rank, e.channel,
              e.algorithm, (unsigned long long)e.payloadBytes, (unsigned long long)e.value);
      break;
    case NCCL_TELEM_TRANSPORT_CONNECT:
      fprintf(text, "[%llu ns] TRANSPORT rank=%u peer=%llu ch=%u conn=%u type=%s path=%s path_type=%u\n",
              (unsigned long long)e.timestampNs, e.rank, (unsigned long long)e.value, e.channel,
              e.protocol, e.algorithm == 0 ? "P2P" : e.algorithm == 1 ? "SHM" :
              e.algorithm == 2 ? "NET" : e.algorithm == 3 ? "COLLNET" : "UNKNOWN",
              e.algorithm == 0 && e.reserved < PATH_DIS + 1 ? topoPathTypeStr[e.reserved] :
              e.algorithm == 1 ? "SHM" : e.algorithm == 2 ? "NET" :
              e.algorithm == 3 ? "COLLNET" : "UNKNOWN", e.reserved);
      break;
    case NCCL_TELEM_RING_EDGE:
      fprintf(text, "[%llu ns] RING rank=%u ch=%u prev=%u next=%u\n",
              (unsigned long long)e.timestampNs, e.rank, e.channel, e.algorithm, e.protocol);
      break;
    case NCCL_TELEM_WORK_START:
    case NCCL_TELEM_WORK_END:
      fprintf(text, "[%llu ns] WORK_%s rank=%u ch=%u counter=%llu gpu_timer=%llu\n",
              (unsigned long long)e.timestampNs,
              e.eventType == NCCL_TELEM_WORK_START ? "START" : "END", e.rank, e.channel,
              (unsigned long long)e.collectiveId, (unsigned long long)e.value);
      break;
    case NCCL_TELEM_WORK_SNAPSHOT:
      fprintf(text, "[%llu ns] WORK_SNAPSHOT rank=%u ch=%u counter=%llu value=%llu\n",
              (unsigned long long)e.timestampNs, e.rank, e.channel,
              (unsigned long long)e.collectiveId, (unsigned long long)e.value);
      break;
    case NCCL_TELEM_FIRST_STEP:
    case NCCL_TELEM_LAST_STEP:
      fprintf(text, "[%llu ns] %s rank=%u ch=%u counter=%llu value=%llu\n",
              (unsigned long long)e.timestampNs,
              e.eventType == NCCL_TELEM_FIRST_STEP ? "FIRST_STEP" : "LAST_STEP",
              e.rank, e.channel, (unsigned long long)e.collectiveId,
              (unsigned long long)e.value);
      break;
    case NCCL_TELEM_CHANNEL_TRANSFER:
      fprintf(text, "[%llu ns] TRANSFER op=%llu rank=%u ch=%u peer=%u transport=%u step=%llu bytes=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              e.rank, e.channel, e.algorithm, e.protocol, (unsigned long long)e.value,
              (unsigned long long)e.payloadBytes);
      break;
    case NCCL_TELEM_CHANNEL_SUMMARY:
      fprintf(text, "[%llu ns] CHANNEL_SUMMARY coll=%llu plan=%llu rank=%u ch=%u work=%llu bytes=%llu duration_ns=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel,
              (unsigned long long)e.value, (unsigned long long)e.payloadBytes,
              (unsigned long long)e.trafficBytes);
      break;
    default:
      fprintf(text, "[%llu ns] EVENT type=%u\n", (unsigned long long)e.timestampNs, e.eventType);
      break;
    }
  }
  fclose(f);
  if (text != nullptr) fclose(text);
}

void ensureInitialized() {
  bool expected = false;
  if (initialized.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    atexit(dump);
  }
}

void recordBatch(const ncclTelemetryEvent* input, size_t count) {
  if (!ncclParamTelemetryEnable() || count == 0) return;
  ensureInitialized();
  uint64_t base = buffer.next.fetch_add(count, std::memory_order_relaxed);
  uint64_t overwritten = base >= kCapacity ? count : (base + count > kCapacity ? base + count - kCapacity : 0);
  if (overwritten != 0) buffer.dropped.fetch_add(overwritten, std::memory_order_relaxed);
  for (size_t i = 0; i < count; ++i) buffer.events[(base + i) % kCapacity] = input[i];
}

void record(uint16_t type, uint64_t collectiveId, uint64_t planId, int rank, int channel,
            int collective, int algorithm, int protocol, size_t payloadBytes,
            size_t trafficBytes, uint64_t value, uint32_t reserved = 0) {
  if (!ncclParamTelemetryEnable()) return;
  ncclTelemetryEvent e{};
  e.timestampNs = nowNs();
  e.collectiveId = collectiveId;
  e.planId = planId;
  e.rank = uint32_t(rank);
  e.channel = uint16_t(channel < 0 ? 0xffff : channel);
  e.eventType = type;
  e.collective = uint32_t(collective);
  e.algorithm = uint32_t(algorithm);
  e.protocol = uint32_t(protocol);
  e.reserved = reserved;
  e.payloadBytes = payloadBytes;
  e.trafficBytes = trafficBytes;
  e.value = value;
  recordBatch(&e, 1);
}

}

void ncclTelemetryFlush() {
  if (ncclParamTelemetryEnable()) dump();
}

int ncclTelemetryLevel() {
  return ncclParamTelemetryEnable() ? (int)ncclParamTelemetryLevel() : -1;
}

uint64_t ncclTelemetryNextCollectiveId() {
  return collectiveIds.fetch_add(1, std::memory_order_relaxed);
}

void ncclTelemetryRecordCollective(uint64_t collectiveId, int rank, int collective,
                                   size_t payloadBytes, size_t trafficBytes) {
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    for (auto it = channelPlans.begin(); it != channelPlans.end();) {
      if (it->rank == uint32_t(rank) && it->collectiveId != collectiveId)
        it = channelPlans.erase(it);
      else
        ++it;
    }
  }
  record(NCCL_TELEM_COLLECTIVE_ENQUEUE, collectiveId, 0, rank, -1, collective, 0, 0,
         payloadBytes, trafficBytes, 0);
}

void ncclTelemetryRecordChannel(uint64_t collectiveId, uint64_t planId, int rank,
                                int channel, int collective, int algorithm, int protocol,
                                size_t payloadBytes, size_t trafficBytes, uint64_t workCounter) {
  std::lock_guard<std::mutex> lock(stateMutex);
  ChannelPlanKey dedupe{collectiveId, planId, uint32_t(rank), uint16_t(channel < 0 ? 0xffff : channel)};
  if (!channelPlans.insert(dedupe).second) return;
  if (ncclTelemetryLevel() == NCCL_TELEM_EXECUTION && workCounter != 0) {
    WorkKey work{workCounter, uint32_t(rank), uint16_t(channel)};
    workStates[work] = WorkState{dedupe, uint64_t(payloadBytes), 0};
  }
  record(NCCL_TELEM_CHANNEL_PLAN, collectiveId, planId, rank, channel, collective, algorithm,
         protocol, payloadBytes, trafficBytes, 0);
}

void ncclTelemetryRecordPlan(uint64_t planId, int rank, uint64_t channelMask, int nCollectives) {
  record(NCCL_TELEM_PLAN_LAUNCH, 0, planId, rank, -1, 0, 0, 0, 0, 0, channelMask);
  (void)nCollectives;
}

void ncclTelemetryRecordProxy(uint64_t planId, int rank, int channel, int pattern,
                              size_t bytes, uint64_t opCount) {
  record(NCCL_TELEM_PROXY_OP, 0, planId, rank, channel, 0, pattern, 0, bytes, bytes, opCount);
}

void ncclTelemetryRecordRingEdge(int rank, int channel, int prev, int next) {
  record(NCCL_TELEM_RING_EDGE, 0, 0, rank, channel, 0, prev, next, 0, 0, 0);
}

void ncclTelemetryRecordWork(int rank, int channel, uint64_t counter, uint64_t timestamp, bool end) {
  if (ncclTelemetryLevel() >= NCCL_TELEM_DIAGNOSTIC) {
    record(end ? NCCL_TELEM_WORK_END : NCCL_TELEM_WORK_START, counter, 0, rank, channel,
           0, 0, 0, 0, 0, timestamp);
    return;
  }
  std::lock_guard<std::mutex> lock(stateMutex);
  WorkKey key{counter, uint32_t(rank), uint16_t(channel)};
  auto work = workStates.find(key);
  if (work == workStates.end()) return;
  if (!end) {
    work->second.start = timestamp;
    return;
  }
  if (work->second.start == 0) return;
  ChannelState& state = channelStates[work->second.channel];
  if (state.firstStart == 0) state.firstStart = work->second.start;
  state.lastEnd = std::max(state.lastEnd, timestamp);
  state.totalBytes += work->second.bytes;
  state.workCount++;
  appendChannelSummary(pendingSummaries, work->second.channel, state);
  channelStates.erase(work->second.channel);
  workStates.erase(work);
  if (pendingSummaries.size() >= 32) {
    recordBatch(pendingSummaries.data(), pendingSummaries.size());
    pendingSummaries.clear();
  }
}

void ncclTelemetryRecordWorkSnapshot(int rank, int channel, uint64_t counter, uint64_t value) {
  record(NCCL_TELEM_WORK_SNAPSHOT, counter, 0, rank, channel, 0, 0, 0, 0, 0, value);
}

void ncclTelemetryRecordStep(int rank, int channel, uint64_t step, bool last) {
  if (ncclTelemetryLevel() < NCCL_TELEM_EXECUTION) return;
  record(last ? NCCL_TELEM_LAST_STEP : NCCL_TELEM_FIRST_STEP, step, 0, rank, channel, 0, 0, 0, 0, 0, step);
}

void ncclTelemetryRecordTransfer(uint64_t opCount, int rank, int channel, int peer,
                                 int transport, uint64_t step, size_t bytes) {
  // Per-step transfer events are diagnostic detail; keep them off the normal execution path.
  if (ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  record(NCCL_TELEM_CHANNEL_TRANSFER, opCount, 0, rank, channel, 0, peer, transport,
         bytes, bytes, step);
}

void ncclTelemetryRecordChannelSummary(uint64_t collectiveId, uint64_t planId, int rank,
                                       int channel, uint64_t workCount, size_t bytes,
                                       uint64_t durationNs) {
  record(NCCL_TELEM_CHANNEL_SUMMARY, collectiveId, planId, rank, channel, 0, 0, 0,
         bytes, durationNs, workCount);
}

void ncclTelemetryRecordTransport(int rank, int channel, int peer, int connIndex, int transport,
                                  int pathType) {
  record(NCCL_TELEM_TRANSPORT_CONNECT, 0, 0, rank, channel, 0, transport, connIndex,
         0, 0, uint64_t(uint32_t(peer)), uint32_t(pathType));
}
