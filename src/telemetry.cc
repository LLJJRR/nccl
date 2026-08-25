/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#include "telemetry.h"
#include "param.h"
#include "collectives.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

void dump() {
  if (!initialized.load(std::memory_order_relaxed)) return;
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
      fprintf(text, "[%llu ns] TRANSPORT rank=%u peer=%llu ch=%u conn=%u type=%s\n",
              (unsigned long long)e.timestampNs, e.rank, (unsigned long long)e.value, e.channel,
              e.protocol, e.algorithm == 0 ? "P2P" : e.algorithm == 1 ? "SHM" :
              e.algorithm == 2 ? "NET" : e.algorithm == 3 ? "COLLNET" : "UNKNOWN");
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

void record(uint16_t type, uint64_t collectiveId, uint64_t planId, int rank, int channel,
            int collective, int algorithm, int protocol, size_t payloadBytes,
            size_t trafficBytes, uint64_t value) {
  if (!ncclParamTelemetryEnable()) return;
  ensureInitialized();
  uint64_t slot = buffer.next.fetch_add(1, std::memory_order_relaxed);
  if (slot >= kCapacity) buffer.dropped.fetch_add(1, std::memory_order_relaxed);
  ncclTelemetryEvent& e = buffer.events[slot % kCapacity];
  e.timestampNs = nowNs();
  e.collectiveId = collectiveId;
  e.planId = planId;
  e.rank = uint32_t(rank);
  e.channel = uint16_t(channel < 0 ? 0xffff : channel);
  e.eventType = type;
  e.collective = uint32_t(collective);
  e.algorithm = uint32_t(algorithm);
  e.protocol = uint32_t(protocol);
  e.reserved = 0;
  e.payloadBytes = payloadBytes;
  e.trafficBytes = trafficBytes;
  e.value = value;
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
  record(NCCL_TELEM_COLLECTIVE_ENQUEUE, collectiveId, 0, rank, -1, collective, 0, 0,
         payloadBytes, trafficBytes, 0);
}

void ncclTelemetryRecordChannel(uint64_t collectiveId, uint64_t planId, int rank,
                                int channel, int collective, int algorithm, int protocol,
                                size_t payloadBytes, size_t trafficBytes) {
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
  // Execution telemetry samples short-lived work items; diagnostic keeps every event.
  if (ncclTelemetryLevel() == NCCL_TELEM_EXECUTION && (counter & 3ull) != 0) return;
  record(end ? NCCL_TELEM_WORK_END : NCCL_TELEM_WORK_START, counter, 0, rank, channel,
         0, 0, 0, 0, 0, timestamp);
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

void ncclTelemetryRecordTransport(int rank, int channel, int peer, int connIndex, int transport) {
  record(NCCL_TELEM_TRANSPORT_CONNECT, 0, 0, rank, channel, 0, transport, connIndex,
         0, 0, uint64_t(uint32_t(peer)));
}
