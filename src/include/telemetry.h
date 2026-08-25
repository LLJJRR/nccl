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
  NCCL_TELEM_CHANNEL_SUMMARY = 13
};

enum ncclTelemetryLevel : uint8_t {
  NCCL_TELEM_BASIC = 0,
  NCCL_TELEM_EXECUTION = 1,
  NCCL_TELEM_DIAGNOSTIC = 2
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

void ncclTelemetryRecordCollective(uint64_t collectiveId, int rank, int collective,
                                   size_t payloadBytes, size_t trafficBytes);
uint64_t ncclTelemetryNextCollectiveId();
void ncclTelemetryFlush();
int ncclTelemetryLevel();
void ncclTelemetryRecordChannel(uint64_t collectiveId, uint64_t planId, int rank,
                                int channel, int collective, int algorithm, int protocol,
                                size_t payloadBytes, size_t trafficBytes);
void ncclTelemetryRecordPlan(uint64_t planId, int rank, uint64_t channelMask, int nCollectives);
void ncclTelemetryRecordProxy(uint64_t planId, int rank, int channel, int pattern,
                              size_t bytes, uint64_t opCount);
void ncclTelemetryRecordTransport(int rank, int channel, int peer, int connIndex, int transport);
void ncclTelemetryRecordRingEdge(int rank, int channel, int prev, int next);
void ncclTelemetryRecordWork(int rank, int channel, uint64_t counter, uint64_t timestamp, bool end);
void ncclTelemetryRecordWorkSnapshot(int rank, int channel, uint64_t counter, uint64_t value);
void ncclTelemetryRecordStep(int rank, int channel, uint64_t step, bool last);
void ncclTelemetryRecordTransfer(uint64_t opCount, int rank, int channel, int peer,
                                 int transport, uint64_t step, size_t bytes);
void ncclTelemetryRecordChannelSummary(uint64_t collectiveId, uint64_t planId, int rank,
                                       int channel, uint64_t workCount, size_t bytes,
                                       uint64_t durationNs);

#endif
