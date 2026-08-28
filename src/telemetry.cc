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
#include <unordered_set>
#include <unistd.h>

// Basic host-side flight recording is on by default. Set NCCL_TELEMETRY_ENABLE=0 to disable it.
NCCL_PARAM(TelemetryEnable, "TELEMETRY_ENABLE", 1);
NCCL_PARAM(TelemetryLevel, "TELEMETRY_LEVEL", 0);

namespace {
constexpr uint64_t kCapacity = 1ull << 16;
static_assert(sizeof(ncclTelemetryEvent) == 72, "telemetry event ABI changed");
struct TelemetryBuffer {
  std::atomic<uint64_t> next{0};
  std::atomic<uint64_t> dropped{0};
  ncclTelemetryEvent events[kCapacity];
};

TelemetryBuffer buffer;
std::atomic<bool> initialized{false};
std::atomic<uint64_t> collectiveIds{0};
std::atomic<uint64_t> proxyIds{0};
std::mutex stateMutex;
constexpr size_t kMaxNetRequestContexts = 8;
thread_local ncclTelemetryNetRequestContext netRequestContexts[kMaxNetRequestContexts];
thread_local size_t netRequestContextCount = 0;

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

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

void dump() {
  if (!initialized.load(std::memory_order_relaxed)) return;
  const char* outputDir = getenv("NCCL_TELEMETRY_DIR");
  if (outputDir == nullptr || outputDir[0] == '\0') outputDir = "/tmp";
  char hostname[64] = "unknown";
  gethostname(hostname, sizeof(hostname));
  hostname[sizeof(hostname)-1] = '\0';
  char path[512];
  snprintf(path, sizeof(path), "%s/nccl_telemetry.%s.%d.bin", outputDir, hostname, int(getpid()));
  FILE* f = fopen(path, "wb");
  if (f == nullptr) return;
  char textPath[512];
  snprintf(textPath, sizeof(textPath), "%s/nccl_telemetry.%s.%d.txt", outputDir, hostname, int(getpid()));
  FILE* text = fopen(textPath, "w");
  uint64_t end = buffer.next.load(std::memory_order_acquire);
  uint64_t begin = end > kCapacity ? end - kCapacity : 0;
  uint64_t magic = 0x4e43434c54454c31ull; // NC CLTEL1
  uint32_t version = 2;
  uint32_t eventSize = sizeof(ncclTelemetryEvent);
  uint64_t dropped = buffer.dropped.load(std::memory_order_relaxed);
  fwrite(&magic, sizeof(magic), 1, f);
  fwrite(&version, sizeof(version), 1, f);
  fwrite(&eventSize, sizeof(eventSize), 1, f);
  fwrite(&dropped, sizeof(dropped), 1, f);
  if (text != nullptr) fprintf(text, "HEADER version=%u event_size=%u events=%llu dropped=%llu "
                               "clock=monotonic_ns level=%d capabilities=proxy_progress,net_path,rdma_external_counters "
                               "qp_wqe_cqe=ib_internal_diagnostic rdma_detail=request_state,wr,sge,qp_depth,cq_poll,cqe_detail\n",
                               version, eventSize, (unsigned long long)(end - begin),
                               (unsigned long long)dropped, ncclTelemetryLevel());
  for (uint64_t i = begin; i < end; ++i) {
    const ncclTelemetryEvent& e = buffer.events[i % kCapacity];
    fwrite(&e, sizeof(ncclTelemetryEvent), 1, f);
    if (text == nullptr) continue;
    switch (e.eventType) {
    case NCCL_TELEM_COLLECTIVE_ENQUEUE:
      fprintf(text, "[%llu ns] COLLECTIVE id=%llu comm_id=0x%llx nranks=%llu rank=%u type=%s payload=%llu traffic=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, (unsigned long long)e.value, e.rank,
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
      fprintf(text, "[%llu ns] TRANSPORT rank=%u peer=%llu ch=%u conn=%u direction=%s type=%s path=%s path_type=%u\n",
              (unsigned long long)e.timestampNs, e.rank, (unsigned long long)e.value, e.channel,
              e.protocol, e.collective == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              e.collective == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.algorithm == 0 ? "P2P" : e.algorithm == 1 ? "SHM" :
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
      fprintf(text, "[%llu ns] WORK_%s coll=%llu plan=%llu rank=%u ch=%u counter=%llu gpu_timer=%llu\n",
              (unsigned long long)e.timestampNs,
              e.eventType == NCCL_TELEM_WORK_START ? "START" : "END",
              (unsigned long long)e.collectiveId, (unsigned long long)e.planId, e.rank, e.channel,
              (unsigned long long)e.trafficBytes, (unsigned long long)e.value);
      break;
    case NCCL_TELEM_WORK_SNAPSHOT:
      fprintf(text, "[%llu ns] WORK_SNAPSHOT coll=%llu plan=%llu rank=%u ch=%u counter=%llu value=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel,
              (unsigned long long)e.trafficBytes, (unsigned long long)e.value);
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
      fprintf(text, "[%llu ns] TRANSFER coll=%llu plan=%llu op=%llu rank=%u ch=%u direction=%s peer=%u transport=%u step=%llu bytes=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, (unsigned long long)e.trafficBytes, e.rank, e.channel,
              e.collective == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              e.collective == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.algorithm, e.protocol, (unsigned long long)e.value,
              (unsigned long long)e.payloadBytes);
      break;
    case NCCL_TELEM_CHANNEL_SUMMARY:
      fprintf(text, "[%llu ns] CHANNEL_SUMMARY coll=%llu plan=%llu rank=%u ch=%u work=%llu bytes=%llu duration_ns=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel,
              (unsigned long long)e.value, (unsigned long long)e.payloadBytes,
              (unsigned long long)e.trafficBytes);
      break;
    case NCCL_TELEM_NET_PATH: {
      uint32_t gdrMode = e.reserved & 0x3;
      uint32_t backend = (e.reserved >> 5) & 0x7;
      int port = int(int16_t((e.reserved >> 8) & 0xffff));
      int speed = int(int32_t(uint32_t(e.value)));
      int railId = int(int16_t((e.value >> 32) & 0xffff));
      int planeId = int(int16_t((e.value >> 48) & 0xffff));
      uint32_t pathType = (e.reserved >> 24) & 0xff;
      fprintf(text, "[%llu ns] NET_PATH rank=%u peer=%llu ch=%u conn=%u direction=%s backend=%s net_dev=%u net_id=0x%llx guid=0x%llx port=%d speed_mbps=%d rail=%d plane=%d proxy_rank=%llu pxn=%u gdr=%s gpu_nic_path=%s path_type=%u shared=%u same_device=%u\n",
              (unsigned long long)e.timestampNs, e.rank, (unsigned long long)e.payloadBytes,
              e.channel, e.protocol,
              e.collective == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              e.collective == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              backend == NCCL_TELEM_NET_BACKEND_SOCKET ? "SOCKET" :
              backend == NCCL_TELEM_NET_BACKEND_IB ? "IB" :
              backend == NCCL_TELEM_NET_BACKEND_PLUGIN ? "PLUGIN" : "UNKNOWN",
              e.algorithm, (unsigned long long)e.collectiveId, (unsigned long long)e.planId,
              port, speed, railId, planeId,
              (unsigned long long)e.trafficBytes, (e.reserved >> 4) & 0x1,
              gdrMode == 0 ? "DISABLED" : gdrMode == 1 ? "DEFAULT" :
              gdrMode == 2 ? "PCI" : "UNKNOWN",
              pathType < PATH_DIS + 1 ? topoPathTypeStr[pathType] : "UNKNOWN", pathType,
              (e.reserved >> 2) & 0x1, (e.reserved >> 3) & 0x1);
      break;
    }
    case NCCL_TELEM_PROXY_PROGRESS:
      fprintf(text, "[%llu ns] PROXY_PROGRESS id=%llu coll=%llu plan=%llu rank=%u ch=%u "
              "peer=%u direction=%s transport=%u phase=%u status=%u expected_bytes=%llu "
              "progressed_bytes=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.value,
              (unsigned long long)e.collectiveId, (unsigned long long)e.planId, e.rank,
              e.channel, e.algorithm,
              e.collective == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              e.collective == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.protocol, e.reserved & 0xff, (e.reserved >> 8) & 0xff,
              (unsigned long long)e.payloadBytes, (unsigned long long)e.trafficBytes);
      break;
    case NCCL_TELEM_RDMA_REQUEST:
      fprintf(text, "[%llu ns] RDMA_REQUEST proxy=%llu coll=%llu plan=unavailable rank=%u ch=%u "
              "peer=%u direction=%s request=%llu qp=%u wr_id=%llu opcode=%u status=%u "
              "phase=%u owner_index=%u owner_count=%u completion_expected=%u bytes=%llu\n",
              (unsigned long long)e.timestampNs,
              (unsigned long long)e.planId,
              (unsigned long long)e.collectiveId,
              e.rank, e.channel, e.algorithm,
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              (unsigned long long)(e.value & 0x00ffffffffffffffull), e.protocol,
              (unsigned long long)e.trafficBytes, uint32_t((e.value >> 56) & 0xff),
              (e.collective >> 8) & 0xff,
              e.reserved & 0xff,
              (e.reserved >> 8) & 0xff,
              (e.reserved >> 16) & 0xff,
              (e.reserved >> 24) & 0x1,
              (unsigned long long)e.payloadBytes);
      break;
    case NCCL_TELEM_RDMA_REQUEST_STATE:
      fprintf(text, "[%llu ns] RDMA_REQUEST_STATE request=%llu proxy=%llu rank=%u ch=%u type=%u state=%u expected_bytes=%llu posted_bytes=%llu completed_bytes=%llu wrs=%u cqes=%u\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.protocol, e.algorithm,
              (unsigned long long)e.payloadBytes, (unsigned long long)e.trafficBytes,
              (unsigned long long)e.value, e.reserved & 0xffff, e.reserved >> 16);
      break;
    case NCCL_TELEM_RDMA_WR:
      fprintf(text, "[%llu ns] RDMA_WR request=%llu proxy=%llu rank=%u ch=%u peer=%u direction=%s qp=%u wr_id=%llu opcode=%u send_flags=0x%x num_sge=%u total_sge_bytes=%llu remote_addr=0x%llx rkey=%u imm_data=%u\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.algorithm,
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.protocol, (unsigned long long)e.trafficBytes, (e.collective >> 8) & 0xff,
              (e.collective >> 16) & 0xff, (e.collective >> 24) & 0xff,
              (unsigned long long)e.reserved,
              (unsigned long long)e.payloadBytes, uint32_t(e.value >> 32), uint32_t(e.value));
      break;
    case NCCL_TELEM_RDMA_SGE:
      fprintf(text, "[%llu ns] RDMA_SGE request=%llu proxy=%llu rank=%u ch=%u peer=%u direction=%s qp=%u wr_id=%llu sge_index=%u addr=0x%llx length=%llu lkey=%u\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.algorithm,
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.protocol, (unsigned long long)e.value, e.collective >> 8,
              (unsigned long long)e.payloadBytes, (unsigned long long)e.trafficBytes,
              e.reserved);
      break;
    case NCCL_TELEM_RDMA_QP_DEPTH:
      fprintf(text, "[%llu ns] RDMA_QP_DEPTH request=%llu proxy=%llu rank=%u ch=%u peer=%u direction=%s qp=%u phase=%u outstanding_wrs=%u posted_bytes=%llu completed_bytes=%llu posted_wrs=%u completed_wrs=%u\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.algorithm,
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.protocol, (e.reserved >> 16) & 0xff, e.reserved & 0xffff,
              (unsigned long long)e.payloadBytes, (unsigned long long)e.trafficBytes,
              uint32_t(e.value >> 32), uint32_t(e.value));
      break;
    case NCCL_TELEM_RDMA_CQ_POLL:
      fprintf(text, "[%llu ns] RDMA_CQ_POLL request=%llu proxy=%llu rank=%u ch=%u cq_id=0x%x window_start_ns=%llu poll_calls=%llu empty_polls=%u returned_cqes=%llu busy_ns=%u\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.algorithm,
              (unsigned long long)e.value, (unsigned long long)e.payloadBytes, e.protocol,
              (unsigned long long)e.trafficBytes, e.reserved);
      break;
    case NCCL_TELEM_RDMA_CQE_DETAIL:
      fprintf(text, "[%llu ns] RDMA_CQE_DETAIL request=%llu proxy=%llu rank=%u ch=%u peer=%u direction=%s qp=%u wr_id=%llu opcode=%u status=%u vendor_err=%u src_qp=%u wc_flags=0x%x imm_data=%u byte_len=%llu\n",
              (unsigned long long)e.timestampNs, (unsigned long long)e.collectiveId,
              (unsigned long long)e.planId, e.rank, e.channel, e.algorithm,
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_SEND ? "SEND" :
              (e.collective & 0xff) == NCCL_TELEM_DIRECTION_RECV ? "RECV" : "UNKNOWN",
              e.protocol, (unsigned long long)e.trafficBytes, (e.reserved >> 8) & 0xff,
              e.reserved & 0xff, e.collective >> 8,
              uint32_t(e.value >> 32), (e.reserved >> 16) & 0xffff, uint32_t(e.value),
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

uint64_t ncclTelemetryNextProxyId() {
  return proxyIds.fetch_add(1, std::memory_order_relaxed);
}

void ncclTelemetryRecordCollective(uint64_t collectiveId, uint64_t commId, int nranks,
                                   int rank, int collective, size_t payloadBytes,
                                   size_t trafficBytes) {
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    for (auto it = channelPlans.begin(); it != channelPlans.end();) {
      if (it->rank == uint32_t(rank) && it->collectiveId != collectiveId)
        it = channelPlans.erase(it);
      else
        ++it;
    }
  }
  record(NCCL_TELEM_COLLECTIVE_ENQUEUE, collectiveId, commId, rank, -1, collective, 0, 0,
         payloadBytes, trafficBytes, uint64_t(nranks));
}

void ncclTelemetryRecordChannel(uint64_t collectiveId, uint64_t planId, int rank,
                                int channel, int collective, int algorithm, int protocol,
                                size_t payloadBytes, size_t trafficBytes) {
  std::lock_guard<std::mutex> lock(stateMutex);
  ChannelPlanKey dedupe{collectiveId, planId, uint32_t(rank), uint16_t(channel < 0 ? 0xffff : channel)};
  if (!channelPlans.insert(dedupe).second) return;
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

void ncclTelemetryRecordProxyProgress(uint64_t proxyId, uint64_t collectiveId,
                                      uint64_t planId, int rank, int channel, int peer,
                                      int direction, int transport, uint64_t expectedBytes,
                                      uint64_t progressedBytes, uint32_t phase,
                                      uint32_t status) {
  if (ncclTelemetryLevel() < NCCL_TELEM_EXECUTION) return;
  uint32_t reserved = (phase & 0xff) | ((status & 0xff) << 8);
  record(NCCL_TELEM_PROXY_PROGRESS, collectiveId, planId, rank, channel, direction, peer,
         transport, expectedBytes, progressedBytes, proxyId, reserved);
}

void ncclTelemetrySetNetRequestContext(uint64_t proxyId, uint64_t collectiveId,
                                       uint64_t planId, int rank, int channel, int peer,
                                       int direction) {
  netRequestContexts[0] = {proxyId, collectiveId, planId, rank, channel, peer, direction};
  netRequestContextCount = 1;
}

void ncclTelemetrySetNetRequestContexts(const ncclTelemetryNetRequestContext* contexts,
                                        size_t count) {
  if (contexts == nullptr) count = 0;
  netRequestContextCount = count < kMaxNetRequestContexts ? count : kMaxNetRequestContexts;
  for (size_t i = 0; i < netRequestContextCount; ++i) netRequestContexts[i] = contexts[i];
}

void ncclTelemetryClearNetRequestContext() {
  netRequestContextCount = 0;
}

bool ncclTelemetryGetNetRequestContext(ncclTelemetryNetRequestContext* context) {
  if (netRequestContextCount == 0 || context == nullptr) return false;
  *context = netRequestContexts[0];
  return true;
}

size_t ncclTelemetryGetNetRequestContexts(ncclTelemetryNetRequestContext* contexts,
                                          size_t capacity) {
  if (contexts == nullptr) return 0;
  size_t count = netRequestContextCount < capacity ? netRequestContextCount : capacity;
  for (size_t i = 0; i < count; ++i) contexts[i] = netRequestContexts[i];
  return count;
}

void ncclTelemetryRecordRdmaRequest(const ncclTelemetryNetRequestContext* context,
                                    uint64_t requestId, uint32_t qpNum, uint64_t wrId,
                                    uint32_t opcode, uint32_t status, uint64_t bytes,
                                    uint32_t phase, uint32_t ownerIndex,
                                    uint32_t ownerCount, bool completionExpected) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  // Event-specific layout: planId=full proxy id, algorithm=peer, protocol=QP
  // number, traffic=wr_id, value=request id with opcode in the high byte,
  // collective carries direction/status. Reserved carries phase and the
  // position in a grouped receive request. The original plan is recovered
  // through the proxy event.
  uint32_t reserved = (phase & 0xff) | ((ownerIndex & 0xff) << 8) |
                      ((ownerCount & 0xff) << 16) |
                      (uint32_t(completionExpected) << 24);
  uint64_t requestValue = (requestId & 0x00ffffffffffffffull) |
                          (uint64_t(opcode & 0xff) << 56);
  record(NCCL_TELEM_RDMA_REQUEST, context->collectiveId, context->proxyId, context->rank,
         context->channel, context->direction | ((status & 0xff) << 8), context->peer,
         qpNum, bytes, wrId, requestValue, reserved);
}

void ncclTelemetryRecordRdmaRequestState(const ncclTelemetryNetRequestContext* context,
                                         uint64_t requestId, uint32_t requestType,
                                         uint32_t state, uint64_t expectedBytes,
                                         uint64_t postedBytes, uint64_t completedBytes,
                                         uint32_t wrCount, uint32_t cqeCount) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  uint32_t reserved = (wrCount & 0xffff) | ((cqeCount & 0xffff) << 16);
  record(NCCL_TELEM_RDMA_REQUEST_STATE, requestId, context->proxyId,
         context->rank, context->channel, context->direction, requestType, state,
         expectedBytes, postedBytes, completedBytes, reserved);
}

void ncclTelemetryRecordRdmaWr(const ncclTelemetryNetRequestContext* context,
                               uint64_t requestId, uint64_t proxyId, uint32_t qpNum,
                               uint64_t wrId, uint32_t opcode, uint32_t sendFlags,
                               uint32_t numSge, uint64_t totalSgeBytes,
                               uint64_t remoteAddr, uint32_t rkey, uint32_t immData) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  uint32_t reserved = uint32_t(std::min<uint64_t>(totalSgeBytes, UINT32_MAX));
  uint32_t collective = uint32_t(context->direction) | ((opcode & 0xff) << 8) |
                        ((sendFlags & 0xff) << 16) | ((numSge & 0xff) << 24);
  uint64_t value = (uint64_t(rkey) << 32) | immData;
  record(NCCL_TELEM_RDMA_WR, requestId, proxyId, context->rank, context->channel,
         collective, context->peer, qpNum, remoteAddr, wrId, value, reserved);
}

void ncclTelemetryRecordRdmaSge(const ncclTelemetryNetRequestContext* context,
                                uint64_t requestId, uint64_t proxyId, uint32_t qpNum,
                                uint64_t wrId, uint32_t sgeIndex, uint64_t addr,
                                uint32_t length, uint32_t lkey) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  uint32_t collective = uint32_t(context->direction) | ((sgeIndex & 0xffffff) << 8);
  record(NCCL_TELEM_RDMA_SGE, requestId, proxyId, context->rank, context->channel,
         collective, context->peer, qpNum, addr, length, wrId, lkey);
}

void ncclTelemetryRecordRdmaQpDepth(const ncclTelemetryNetRequestContext* context,
                                    uint64_t requestId, uint64_t proxyId, uint32_t qpNum,
                                    uint32_t phase, uint32_t outstandingWrs,
                                    uint64_t postedBytes, uint64_t completedBytes,
                                    uint32_t postedWrs, uint32_t completedWrs) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  uint32_t reserved = (outstandingWrs & 0xffff) | ((phase & 0xff) << 16);
  uint64_t value = (uint64_t(postedWrs) << 32) | completedWrs;
  record(NCCL_TELEM_RDMA_QP_DEPTH, requestId, proxyId, context->rank, context->channel,
         context->direction, context->peer, qpNum, postedBytes, completedBytes, value, reserved);
}

void ncclTelemetryRecordRdmaCqPoll(const ncclTelemetryNetRequestContext* context,
                                   uint64_t requestId, uint64_t cqId, uint64_t windowStartNs,
                                   uint64_t pollCalls, uint64_t emptyPolls,
                                   uint64_t returnedCqes, uint64_t busyNs) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  record(NCCL_TELEM_RDMA_CQ_POLL, requestId, context->proxyId, context->rank,
         context->channel, context->direction, uint32_t(cqId), uint32_t(emptyPolls),
         pollCalls, returnedCqes, windowStartNs, uint32_t(busyNs));
}

void ncclTelemetryRecordRdmaCqeDetail(const ncclTelemetryNetRequestContext* context,
                                      uint64_t requestId, uint32_t qpNum, uint64_t wrId,
                                      uint32_t opcode, uint32_t status, uint32_t vendorErr,
                                      uint32_t srcQp, uint32_t wcFlags, uint32_t immData,
                                      uint32_t byteLen) {
  if (context == nullptr || ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  uint32_t reserved = (status & 0xff) | ((opcode & 0xff) << 8) |
                      ((wcFlags & 0xffff) << 16);
  uint32_t collective = uint32_t(context->direction) | ((vendorErr & 0xffffff) << 8);
  uint64_t value = (uint64_t(srcQp) << 32) | immData;
  record(NCCL_TELEM_RDMA_CQE_DETAIL, requestId, context->proxyId, context->rank,
         context->channel, collective, context->peer, qpNum, byteLen, wrId, value, reserved);
}

void ncclTelemetryRecordRingEdge(int rank, int channel, int prev, int next) {
  record(NCCL_TELEM_RING_EDGE, 0, 0, rank, channel, 0, prev, next, 0, 0, 0);
}

void ncclTelemetryRecordWork(uint64_t collectiveId, uint64_t planId, int rank, int channel,
                             uint64_t counter, uint64_t timestamp, bool end) {
  if (ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  record(end ? NCCL_TELEM_WORK_END : NCCL_TELEM_WORK_START, collectiveId, planId, rank, channel,
         0, 0, 0, 0, counter, timestamp);
}

void ncclTelemetryRecordWorkSnapshot(uint64_t collectiveId, uint64_t planId, int rank, int channel,
                                     uint64_t counter, uint64_t value) {
  record(NCCL_TELEM_WORK_SNAPSHOT, collectiveId, planId, rank, channel, 0, 0, 0, 0, counter, value);
}

void ncclTelemetryRecordStep(int rank, int channel, uint64_t step, bool last) {
  if (ncclTelemetryLevel() < NCCL_TELEM_EXECUTION) return;
  record(last ? NCCL_TELEM_LAST_STEP : NCCL_TELEM_FIRST_STEP, step, 0, rank, channel, 0, 0, 0, 0, 0, step);
}

void ncclTelemetryRecordTransfer(uint64_t collectiveId, uint64_t planId, uint64_t opCount,
                                 int rank, int channel, int peer, int direction, int transport,
                                 uint64_t step, size_t bytes) {
  // Per-step transfer events are diagnostic detail; keep them off the normal execution path.
  if (ncclTelemetryLevel() < NCCL_TELEM_DIAGNOSTIC) return;
  record(NCCL_TELEM_CHANNEL_TRANSFER, collectiveId, planId, rank, channel, direction, peer,
         transport, bytes, opCount, step);
}

void ncclTelemetryRecordChannelSummary(uint64_t collectiveId, uint64_t planId, int rank,
                                       int channel, uint64_t workCount, size_t bytes,
                                       uint64_t durationNs) {
  ncclTelemetryChannelSummary summary{collectiveId, planId, workCount, uint64_t(bytes),
                                      durationNs, uint32_t(rank), uint16_t(channel)};
  ncclTelemetryRecordChannelSummaries(&summary, 1);
}

void ncclTelemetryRecordChannelSummaries(const ncclTelemetryChannelSummary* summaries,
                                         size_t count) {
  constexpr size_t kBatchSize = 64;
  ncclTelemetryEvent events[kBatchSize];
  while (count != 0) {
    size_t batch = count < kBatchSize ? count : kBatchSize;
    uint64_t timestamp = nowNs();
    for (size_t i = 0; i < batch; ++i) {
      const ncclTelemetryChannelSummary& summary = summaries[i];
      ncclTelemetryEvent& event = events[i];
      memset(&event, 0, sizeof(event));
      event.timestampNs = timestamp;
      event.collectiveId = summary.collectiveId;
      event.planId = summary.planId;
      event.rank = summary.rank;
      event.channel = summary.channel;
      event.eventType = NCCL_TELEM_CHANNEL_SUMMARY;
      event.payloadBytes = summary.bytes;
      event.trafficBytes = summary.durationNs;
      event.value = summary.workCount;
    }
    recordBatch(events, batch);
    summaries += batch;
    count -= batch;
  }
}

void ncclTelemetryRecordTransport(int rank, int channel, int peer, int connIndex, int transport,
                                  int direction, int pathType) {
  record(NCCL_TELEM_TRANSPORT_CONNECT, 0, 0, rank, channel, direction, transport, connIndex,
         0, 0, uint64_t(uint32_t(peer)), uint32_t(pathType));
}

void ncclTelemetryRecordNetPath(int rank, int peer, int channel, int connIndex, int direction,
                                int backend, int netDev, int64_t netId, int proxyRank, int gdrMode,
                                int pathType, bool shared, bool sameDevice, uint64_t guid, int port,
                                int speed, int railId, int planeId) {
  uint32_t flags = uint32_t(gdrMode) & 0x3;
  flags |= uint32_t(shared) << 2;
  flags |= uint32_t(sameDevice) << 3;
  flags |= uint32_t(proxyRank != rank) << 4;
  flags |= (uint32_t(backend) & 0x7) << 5;
  flags |= uint32_t(uint16_t(port)) << 8;
  flags |= (uint32_t(pathType) & 0xff) << 24;
  uint64_t properties = uint64_t(uint32_t(speed));
  properties |= uint64_t(uint16_t(railId)) << 32;
  properties |= uint64_t(uint16_t(planeId)) << 48;
  record(NCCL_TELEM_NET_PATH, uint64_t(netId), guid, rank, channel, direction, netDev,
         connIndex, size_t(uint32_t(peer)), size_t(uint32_t(proxyRank)), properties, flags);
}
