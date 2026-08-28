/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_TELEMETRY_H_
#define NCCL_TELEMETRY_H_

#include <stddef.h>
#include <stdint.h>

enum ncclTelemetryEventType : uint16_t {
  NCCL_TELEM_COLLECTIVE_ENQUEUE = 1,
  NCCL_TELEM_CHANNEL_PLAN = 2,
  NCCL_TELEM_PLAN_LAUNCH = 3,
  NCCL_TELEM_PROXY_OP = 4,
  NCCL_TELEM_TRANSPORT_CONNECT = 5,
  NCCL_TELEM_RING_EDGE = 6,
  NCCL_TELEM_WORK_START = 7,
  NCCL_TELEM_WORK_END = 8,
  NCCL_TELEM_WORK_SNAPSHOT = 9,
  NCCL_TELEM_FIRST_STEP = 10,
  NCCL_TELEM_LAST_STEP = 11,
  NCCL_TELEM_CHANNEL_TRANSFER = 12,
  NCCL_TELEM_CHANNEL_SUMMARY = 13,
  NCCL_TELEM_NET_PATH = 14,
  NCCL_TELEM_PROXY_PROGRESS = 15,
  NCCL_TELEM_RDMA_REQUEST = 16
};

enum ncclTelemetryLevel : uint8_t {
  NCCL_TELEM_BASIC = 0,
  NCCL_TELEM_EXECUTION = 1,
  NCCL_TELEM_DIAGNOSTIC = 2
};

enum ncclTelemetryDirection : uint8_t {
  NCCL_TELEM_DIRECTION_RECV = 0,
  NCCL_TELEM_DIRECTION_SEND = 1,
  NCCL_TELEM_DIRECTION_UNKNOWN = 2
};

enum ncclTelemetryNetBackend : uint8_t {
  NCCL_TELEM_NET_BACKEND_UNKNOWN = 0,
  NCCL_TELEM_NET_BACKEND_SOCKET = 1,
  NCCL_TELEM_NET_BACKEND_IB = 2,
  NCCL_TELEM_NET_BACKEND_PLUGIN = 3
};

enum ncclTelemetryProxyPhase : uint8_t {
  NCCL_TELEM_PROXY_ENQUEUE = 0,
  NCCL_TELEM_PROXY_START = 1,
  NCCL_TELEM_PROXY_FIRST_PROGRESS = 2,
  NCCL_TELEM_PROXY_COMPLETE = 3,
  NCCL_TELEM_PROXY_ERROR = 4
};

enum ncclTelemetryRdmaPhase : uint8_t {
  NCCL_TELEM_RDMA_WQE_POST = 0,
  NCCL_TELEM_RDMA_CQE = 1,
  NCCL_TELEM_RDMA_REQUEST_COMPLETE = 2,
  NCCL_TELEM_RDMA_POLL_DELAY = 3
};

struct ncclTelemetryNetRequestContext {
  uint64_t proxyId;
  uint64_t collectiveId;
  uint64_t planId;
  int rank;
  int channel;
  int peer;
  int direction;
};

struct ncclTelemetryEvent {
  uint64_t timestampNs;
  uint64_t collectiveId;
  uint64_t planId;
  uint32_t rank;
  uint16_t channel;
  uint16_t eventType;
  uint32_t collective;
  uint32_t algorithm;
  uint32_t protocol;
  uint32_t reserved;
  uint64_t payloadBytes;
  uint64_t trafficBytes;
  uint64_t value;
};

struct ncclTelemetryChannelSummary {
  uint64_t collectiveId;
  uint64_t planId;
  uint64_t workCount;
  uint64_t bytes;
  uint64_t durationNs;
  uint32_t rank;
  uint16_t channel;
};

void ncclTelemetryRecordCollective(uint64_t collectiveId, uint64_t commId, int nranks,
                                   int rank, int collective, size_t payloadBytes,
                                   size_t trafficBytes);
uint64_t ncclTelemetryNextCollectiveId();
uint64_t ncclTelemetryNextProxyId();
void ncclTelemetryFlush();
int ncclTelemetryLevel();
void ncclTelemetryRecordChannel(uint64_t collectiveId, uint64_t planId, int rank,
                                int channel, int collective, int algorithm, int protocol,
                                size_t payloadBytes, size_t trafficBytes);
void ncclTelemetryRecordPlan(uint64_t planId, int rank, uint64_t channelMask, int nCollectives);
void ncclTelemetryRecordProxy(uint64_t planId, int rank, int channel, int pattern,
                              size_t bytes, uint64_t opCount);
void ncclTelemetryRecordProxyProgress(uint64_t proxyId, uint64_t collectiveId,
                                      uint64_t planId, int rank, int channel, int peer,
                                      int direction, int transport, uint64_t expectedBytes,
                                      uint64_t progressedBytes, uint32_t phase,
                                      uint32_t status);
void ncclTelemetrySetNetRequestContext(uint64_t proxyId, uint64_t collectiveId,
                                       uint64_t planId, int rank, int channel, int peer,
                                       int direction);
void ncclTelemetrySetNetRequestContexts(const ncclTelemetryNetRequestContext* contexts,
                                        size_t count);
void ncclTelemetryClearNetRequestContext();
bool ncclTelemetryGetNetRequestContext(ncclTelemetryNetRequestContext* context);
size_t ncclTelemetryGetNetRequestContexts(ncclTelemetryNetRequestContext* contexts,
                                          size_t capacity);
void ncclTelemetryRecordRdmaRequest(const ncclTelemetryNetRequestContext* context,
                                    uint64_t requestId, uint32_t qpNum, uint64_t wrId,
                                    uint32_t opcode, uint32_t status, uint64_t bytes,
                                    uint32_t phase, uint32_t ownerIndex,
                                    uint32_t ownerCount, bool completionExpected);
void ncclTelemetryRecordTransport(int rank, int channel, int peer, int connIndex, int transport,
                                  int direction, int pathType);
void ncclTelemetryRecordNetPath(int rank, int peer, int channel, int connIndex, int direction,
                                int backend, int netDev, int64_t netId, int proxyRank, int gdrMode,
                                int pathType, bool shared, bool sameDevice, uint64_t guid, int port,
                                int speed, int railId, int planeId);
void ncclTelemetryRecordRingEdge(int rank, int channel, int prev, int next);
void ncclTelemetryRecordWork(uint64_t collectiveId, uint64_t planId, int rank, int channel,
                             uint64_t counter, uint64_t timestamp, bool end);
void ncclTelemetryRecordWorkSnapshot(uint64_t collectiveId, uint64_t planId, int rank, int channel,
                                     uint64_t counter, uint64_t value);
void ncclTelemetryRecordStep(int rank, int channel, uint64_t step, bool last);
void ncclTelemetryRecordTransfer(uint64_t collectiveId, uint64_t planId, uint64_t opCount,
                                 int rank, int channel, int peer, int direction, int transport,
                                 uint64_t step, size_t bytes);
void ncclTelemetryRecordChannelSummary(uint64_t collectiveId, uint64_t planId, int rank,
                                       int channel, uint64_t workCount, size_t bytes,
                                       uint64_t durationNs);
void ncclTelemetryRecordChannelSummaries(const ncclTelemetryChannelSummary* summaries,
                                         size_t count);

#endif
