/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"
#include "telemetry.h"

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to sub->nsteps to indicate that the profiler has started the event
// - transmitted: is set to sub->nsteps to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  ncclTelemetryChannelSummary summaries[NCCL_PROXY_MAX_SUBS];
  size_t summaryCount = 0;
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler *)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler *)sub->recvbuff;
      if (sub->posted < sub->nsteps && sub->base <= workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        sub->telemetryStart = workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp;
        ncclTelemetryRecordWork(sub->telemetryCollectiveId, sub->telemetryPlanId,
                                proxyState->comm->rank, sub->channelId, sub->base,
                                sub->telemetryStart, false);
        ncclProfilerStartKernelChEvent(args, s, workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->posted = sub->nsteps;
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps && sub->base <= workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        uint64_t telemetryEnd = workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp;
        ncclTelemetryRecordWork(sub->telemetryCollectiveId, sub->telemetryPlanId,
                                proxyState->comm->rank, sub->channelId, sub->base,
                                telemetryEnd, true);
        if (ncclTelemetryLevel() == NCCL_TELEM_EXECUTION && sub->telemetryCollectiveId != UINT64_MAX) {
          uint64_t duration = telemetryEnd >= sub->telemetryStart ? telemetryEnd - sub->telemetryStart : 0;
          summaries[summaryCount++] = ncclTelemetryChannelSummary{
            sub->telemetryCollectiveId, sub->telemetryPlanId, 1, sub->telemetryBytes, duration,
            uint32_t(proxyState->comm->rank), uint16_t(sub->channelId)};
        }
        if (ncclTelemetryLevel() >= NCCL_TELEM_DIAGNOSTIC)
          ncclTelemetryRecordWorkSnapshot(sub->telemetryCollectiveId, sub->telemetryPlanId,
            proxyState->comm->rank, sub->channelId, sub->base,
            workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter);
        ncclProfilerStopKernelChEvent(args, s, workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        // The profiler proxy emits one aggregate transfer for the completed channel work.
        // Collective sub->nbytes can describe only one protocol step, while telemetryBytes
        // carries the complete channel work size from the planner.
        size_t transferBytes = sub->telemetryCollectiveId != UINT64_MAX && sub->telemetryBytes > 0 ?
          size_t(sub->telemetryBytes) : (sub->nbytes > 0 ? size_t(sub->nbytes) : 0);
        ncclTelemetryRecordTransfer(sub->telemetryCollectiveId, sub->telemetryPlanId,
                                    args->opCount, proxyState->comm->rank, sub->channelId,
                                    sub->peer, NCCL_TELEM_DIRECTION_UNKNOWN, 0, sub->base,
                                    transferBytes);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    ncclTelemetryRecordChannelSummaries(summaries, summaryCount);
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {
  "Prof",
  NULL,
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }
};
