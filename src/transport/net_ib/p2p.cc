/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p.h"
#include "common.h"
#include "compiler.h"
#include "p2p_resiliency.h"
#include "telemetry.h"

#include <atomic>
#include <ctime>

NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", -2);
int64_t ncclIbArThreshold = 8192;

// By default, use ncclIbRequestMatchingScheme::BY_INDEX matching scheme.
NCCL_PARAM(IbReceiverSideMatchingScheme, "IB_RECEIVER_SIDE_MATCHING_SCHEME", -2);
NCCL_PARAM(TelemetryFaultIbPollDelayUs, "TELEMETRY_FAULT_IB_POLL_DELAY_US", 0);
NCCL_PARAM(TelemetryFaultIbPollDelayRank, "TELEMETRY_FAULT_IB_POLL_DELAY_RANK", -1);
NCCL_PARAM(TelemetryFaultIbPollDelayChannel, "TELEMETRY_FAULT_IB_POLL_DELAY_CHANNEL", -1);

const char* ncclIbReqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };
static std::atomic<bool> telemetryFaultIbPollDelayInjected{false};
static std::atomic<uint64_t> telemetryRequestGenerations{1};

static inline void ncclIbRecordRequestTelemetry(struct ncclIbRequest* request,
                                                uint32_t qpNum, uint64_t wrId,
                                                uint32_t opcode, uint32_t status,
                                                uint64_t bytes, uint32_t phase,
                                                bool completionExpected = false) {
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    ncclTelemetryRecordRdmaRequest(request->telemetryContexts + i, request->telemetryGeneration, qpNum,
                                   wrId, opcode, status, bytes, phase, i,
                                   request->telemetryContextCount, completionExpected);
  }
}

static inline uint64_t ncclIbTelemetryNowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

ncclResult_t ncclIbPostSendTelemetry(struct ncclIbRequest* request,
                                     struct ibv_qp* qp,
                                     struct ibv_send_wr* first) {
  struct ibv_send_wr* badWr = NULL;
  uint32_t attempted = 0;
  for (struct ibv_send_wr* wr = first; wr != NULL; wr = wr->next) attempted++;
  uint64_t begin = ncclIbTelemetryNowNs();
  ncclResult_t result = wrap_ibv_post_send(qp, first, &badWr);
  uint64_t end = ncclIbTelemetryNowNs();
  if (request != NULL && request->telemetryContextCount != 0) {
    uint32_t posted = result == ncclSuccess ? attempted : 0;
    if (badWr != NULL) {
      posted = 0;
      for (struct ibv_send_wr* wr = first; wr != NULL && wr != badWr; wr = wr->next) posted++;
    }
    for (int i = 0; i < request->telemetryContextCount; ++i)
      ncclTelemetryRecordRdmaPost(request->telemetryContexts + i,
        request->telemetryGeneration, qp ? qp->qp_num : 0, attempted, posted,
        begin, end, int(result), badWr ? badWr->wr_id : 0);
  }
  return result;
}

static inline bool ncclIbTelemetryDetailed(const struct ncclIbRequest* request) {
  return request->telemetryContextCount != 0 &&
         ncclTelemetryLevel() >= NCCL_TELEM_DIAGNOSTIC;
}

static inline void ncclIbRecordRequestState(struct ncclIbRequest* request, uint32_t state) {
  if (!ncclIbTelemetryDetailed(request)) return;
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    ncclTelemetryRecordRdmaRequestState(request->telemetryContexts + i, request->telemetryGeneration,
      request->type, state, request->telemetryExpectedBytes, request->telemetryPostedBytes,
      request->telemetryCompletedBytes, request->telemetryWrCount,
      request->telemetryCqeCount);
  }
}

static inline void ncclIbRecordWr(struct ncclIbRequest* request, uint32_t qpNum,
                                  const struct ibv_send_wr* wr) {
  if (!ncclIbTelemetryDetailed(request) || wr == NULL) return;
  uint64_t bytes = 0;
  if (wr->num_sge > 0 && wr->sg_list == NULL) return;
  for (int s = 0; s < wr->num_sge; ++s) bytes += wr->sg_list[s].length;
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    const ncclTelemetryNetRequestContext* context = request->telemetryContexts + i;
    ncclTelemetryRecordRdmaWr(context, request->telemetryGeneration, context->proxyId, qpNum, wr->wr_id,
      wr->opcode, wr->send_flags, wr->num_sge, bytes, wr->wr.rdma.remote_addr,
      wr->wr.rdma.rkey, wr->imm_data);
    for (int s = 0; s < wr->num_sge; ++s) {
      ncclTelemetryRecordRdmaSge(context, request->telemetryGeneration, context->proxyId, qpNum,
        wr->wr_id, s, wr->sg_list[s].addr, wr->sg_list[s].length,
        wr->sg_list[s].lkey);
    }
  }
  request->telemetryPostedBytes += bytes;
  request->telemetryWrCount++;
}

static inline void ncclIbRecordQpDepth(struct ncclIbRequest* request,
                                       struct ncclIbQp* qp, uint32_t phase,
                                       bool receiveQueue) {
  if (!ncclIbTelemetryDetailed(request) || qp == NULL || qp->qp == NULL) return;
  uint64_t postedWrs = receiveQueue ? qp->telemetryRqPostedWrs : qp->telemetrySqPostedWrs;
  uint64_t completedWrs = receiveQueue ? qp->telemetryRqCompletedWrs : qp->telemetrySqCompletedWrs;
  uint64_t postedBytes = receiveQueue ? qp->telemetryRqPostedBytes : qp->telemetrySqPostedBytes;
  uint64_t completedBytes = receiveQueue ? qp->telemetryRqCompletedBytes : qp->telemetrySqCompletedBytes;
  uint64_t outstandingWrs = postedWrs >= completedWrs ? postedWrs - completedWrs : 0;
  uint32_t outstanding = uint32_t(std::min<uint64_t>(outstandingWrs, UINT32_MAX));
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    const ncclTelemetryNetRequestContext* context = request->telemetryContexts + i;
    ncclTelemetryRecordRdmaQpDepth(context, request->telemetryGeneration, context->proxyId, qp->qp->qp_num,
      phase, outstanding, postedBytes, completedBytes, uint32_t(postedWrs),
      uint32_t(completedWrs));
  }
}

static inline void ncclIbTrackSqBatch(struct ncclIbRequest* request,
                                      struct ncclIbQp* qp, uint32_t wrs,
                                      uint64_t bytes) {
  if (!ncclIbTelemetryDetailed(request)) return;
  qp->telemetrySqPostedWrs += wrs;
  qp->telemetrySqPostedBytes += bytes;
  qp->telemetrySqPendingWrs += wrs;
  qp->telemetrySqPendingBytes += bytes;
  if (request->telemetryRetireCount < NCCL_IB_MAX_QPS) {
    int i = request->telemetryRetireCount++;
    request->telemetryRetireQpNum[i] = qp->qp->qp_num;
    request->telemetryRetireWrs[i] = uint32_t(qp->telemetrySqPendingWrs);
    request->telemetryRetireBytes[i] = qp->telemetrySqPendingBytes;
    qp->telemetrySqPendingWrs = 0;
    qp->telemetrySqPendingBytes = 0;
  }
  ncclIbRecordQpDepth(request, qp, NCCL_TELEM_RDMA_WQE_POST, false);
}

static inline void ncclIbFlushPollTelemetry(struct ncclIbRequest* request, int devIndex,
                                            struct ibv_cq* cq) {
  if (!ncclIbTelemetryDetailed(request) || request->telemetryPollCalls[devIndex] == 0) return;
  uint64_t cqId = uint64_t(uintptr_t(cq));
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    ncclTelemetryRecordRdmaCqPoll(request->telemetryContexts + i, request->telemetryGeneration, cqId,
      request->telemetryPollWindowStartNs[devIndex], request->telemetryPollCalls[devIndex],
      request->telemetryPollEmpty[devIndex], request->telemetryPollReturned[devIndex],
      request->telemetryPollBusyNs[devIndex]);
  }
  request->telemetryPollWindowStartNs[devIndex] = 0;
  request->telemetryPollBusyNs[devIndex] = 0;
  request->telemetryPollCalls[devIndex] = 0;
  request->telemetryPollEmpty[devIndex] = 0;
  request->telemetryPollReturned[devIndex] = 0;
}

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->telemetryGeneration = telemetryRequestGenerations.fetch_add(1, std::memory_order_relaxed);
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      r->telemetryContextCount = uint8_t(ncclTelemetryGetNetRequestContexts(
        r->telemetryContexts, NCCL_NET_IB_MAX_RECVS));
      if (r->telemetryContextCount != 0) {
        r->telemetryExpectedBytes = 0;
        r->telemetryPostedBytes = 0;
        r->telemetryCompletedBytes = 0;
        r->telemetryWrCount = 0;
        r->telemetryCqeCount = 0;
        r->telemetryFirstCqeNs = 0;
        r->telemetryLastCqeNs = 0;
        r->telemetryRetireCount = 0;
        memset(r->telemetryPollWindowStartNs, 0, sizeof(r->telemetryPollWindowStartNs));
        memset(r->telemetryPollBusyNs, 0, sizeof(r->telemetryPollBusyNs));
        memset(r->telemetryPollCalls, 0, sizeof(r->telemetryPollCalls));
        memset(r->telemetryPollEmpty, 0, sizeof(r->telemetryPollEmpty));
        memset(r->telemetryPollReturned, 0, sizeof(r->telemetryPollReturned));
        memset(r->telemetryPollSeen, 0, sizeof(r->telemetryPollSeen));
      }
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  struct ncclIbNetCommDevBase* base = ncclIbGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}


#ifdef ENABLE_TRACE
static ncclResult_t ncclIbPrintWr(struct ibv_send_wr* wr, char* wrStr) {
  if (wr == NULL) {
    sprintf(wrStr, "wr=NULL");
    return ncclSuccess;
  }
  const char* opcodeStr = ibvWrOpcodeStr(wr->opcode);
  switch (wr->opcode) {
    case IBV_WR_RDMA_WRITE:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ", rdma.rkey=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey);
      break;
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ",  rdma.rkey=0x%x, imm_data=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey,
        wr->imm_data);
      break;
    default:
      WARN("NET/IB: %s: No format specified for opcode=%d", __func__, wr->opcode);
      return ncclInternalError;
  }
  return ncclSuccess;
}
#endif // ENABLE_TRACE

// The alignment for IB writes that is required to make LL and LL128 protocols work
#define IB_WRITE_CHUNK_ALIGNMENT 128

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  for (int r = 0; r < nreqs; ++r) ncclIbRecordRequestState(reqs[r], NCCL_TELEM_RDMA_REQUEST_POST_BEGIN);

  TRACE(NCCL_NET, "NET/IB: %s: Posting a send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs);

  int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (uint64_t)(slot & 0xff) << (r*8);
    wr->wr_id = wr_id;
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif

    // Every request is chunked equally across all QPs that are used to transfer
    // the request (in case of a single QP, the chunk is the size of the request).
    // The chunk size of each request determined solely by the send size and the
    // number of QPs used to transfer the request. If the send size is not big
    // enough, starting from some QP there might be no data left to send and the
    // length will be zeroed.
    sge->length = DIVUP(DIVUP(reqs[r]->send.size, nqps), IB_WRITE_CHUNK_ALIGNMENT) * IB_WRITE_CHUNK_ALIGNMENT;
    wr->sg_list = sge;
    wr->num_sge = 1;
  }

  // For ID-based matching scheme, immData carries the request ID.
  // For index-based matching scheme, immData carries the send size:
  // - nreqs == 1
  //      It's the send size.
  // - nreqs > 1
  //      Send size is still sent but receiver ignores it since the sizes are
  //      written to directly to remote completion records array
  uint32_t immData = comm->base.recvMatchingScheme == BY_ID ? (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (!(comm->base.remOooRq && comm->base.localOooRq) && comm->ar && reqs[0]->send.size > ncclIbArThreshold)) {
    // When Adaptive Routing is enabled, send the bulk of the data first as an
    // RDMA Write.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes array
      lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*sizeof(struct ncclIbRequestCompletionRecord);
      lastWr->num_sge = 1;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(immData);
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, reqs[0]->id, i, &qp, &qpIndex));

    TRACE(NCCL_NET, "NET/IB: %s: Posting send (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld) on QP (qp_num=%u, devIndex=%d, qpIndex=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id, qp->qp->qp_num, qp->devIndex, qpIndex);

    // Selective retransmission
    if (comm->base.resiliency && reqs[0]->send.sentData[qpIndex] == true) {
      for (int r=0; r<nreqs; r++) {
        comm->wrs[r].sg_list->addr += comm->wrs[r].sg_list->length;
        comm->wrs[r].wr.rdma.remote_addr += comm->wrs[r].sg_list->length;
      }
      INFO(NCCL_NET, "NET/IB: %s: Skipping retransmission on QP index %d (req=%p, slot=%d) as it was already delivered.", __func__, qpIndex, reqs[0], slot);
      continue;
    }

    int devIndex = qp->devIndex;
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      // Check the data left to send. If the send is too small, it might be
      // that on the current QP there is no data left to be sent.
      comm->wrs[r].sg_list->length = std::min(reqs[r]->send.size-sendOffsets[r], comm->wrs[r].sg_list->length);
      if (comm->wrs[r].sg_list->length == 0) {
        comm->wrs[r].num_sge = 0;
      } else {
        comm->wrs[r].sg_list->lkey = reqs[r]->send.lkeys[devIndex];
      }
    }

    if (nreqs > 1) {
      // Populating the correct gather information based on the device and
      // slot used.
      // Note that the lkey is already correct from the initialization phase.
      lastWr->sg_list = &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs*sizeof(int);
      // Populate the correct RKey based on the device used
      lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store the qpIndex for this request
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
#ifdef ENABLE_TRACE
    for (int r = 0; r < nreqs; r++) {
      TRACE(NCCL_NET, "NET/IB: %s: Posting send work request on QP (qpn=%u, devIndex=%d, qpIndex=%d) (slot=%d, req[r=%d]=%p)", __func__, qp->qp->qp_num, qp->devIndex, qpIndex, slot, r, reqs[r]);
    }
    int wrIdx = 0;
    char wrStr[1024];
    struct ibv_send_wr* currWr = comm->wrs;
    while (currWr != NULL) {
      NCCLCHECK(ncclIbPrintWr(currWr, wrStr));
      TRACE(NCCL_NET, "NET/IB: %s: slot=%d, wrIdx[%d], %s", __func__, slot, wrIdx, wrStr);
      wrIdx++;
      currWr = currWr->next;
    }
#endif // ENABLE_TRACE
    NCCLCHECK(ncclIbPostSendTelemetry(nreqs > 0 ? reqs[0] : NULL, qp->qp, comm->wrs));
    uint32_t postedWrs = 0;
    uint64_t postedBytes = 0;
    for (int r = 0; r < nreqs; r++) {
      ncclIbRecordWr(reqs[r], qp->qp->qp_num, comm->wrs + r);
      postedWrs++;
      if (comm->wrs[r].sg_list) postedBytes += comm->wrs[r].sg_list->length;
      if (reqs[r]->telemetryContextCount != 0) {
        ncclIbRecordRequestTelemetry(reqs[r], qp->qp->qp_num, comm->wrs[r].wr_id,
          comm->wrs[r].opcode, 0,
          comm->wrs[r].sg_list ? comm->wrs[r].sg_list->length : 0,
          NCCL_TELEM_RDMA_WQE_POST,
          (comm->wrs[r].send_flags & IBV_SEND_SIGNALED) != 0);
      }
    }
    if (reqs[0]->telemetryContextCount != 0 && lastWr != comm->wrs + nreqs - 1) {
      ncclIbRecordWr(reqs[0], qp->qp->qp_num, lastWr);
      postedWrs++;
      if (lastWr->sg_list) postedBytes += lastWr->sg_list->length;
      ncclIbRecordRequestTelemetry(reqs[0], qp->qp->qp_num, lastWr->wr_id,
        lastWr->opcode, 0,
        lastWr->sg_list ? lastWr->sg_list->length : 0, NCCL_TELEM_RDMA_WQE_POST,
        (lastWr->send_flags & IBV_SEND_SIGNALED) != 0);
    }
    ncclIbTrackSqBatch(reqs[0], qp, postedWrs, postedBytes);

    // Update the send offset and addresses for the next QP according to the
    // actual data size that was sent on the current QP, for every request
    for (int r=0; r<nreqs; r++) {
      sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + comm->wrs[r].sg_list->length, reqs[r]->send.size);
      comm->wrs[r].sg_list->addr += comm->wrs[r].sg_list->length;
      comm->wrs[r].wr.rdma.remote_addr += comm->wrs[r].sg_list->length;
      TRACE(NCCL_NET, "NET/IB: %s: Send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, reqIdx=%d, wr_id=%ld) posted %d bytes on QP index %d (devIndex=%d, qp_num=%u), total posted %d/%d bytes", __func__, reqs[r], reqs[0]->base, reqs[r]->id, slot, nreqs, r, comm->wrs[r].wr_id, comm->wrs[r].sg_list->length, qpIndex, devIndex, qp->qp->qp_num, sendOffsets[r], reqs[r]->send.size);
      reqs[r]->send.sentData[qpIndex] = true;
    }
  }

  for (int r = 0; r < nreqs; ++r) ncclIbRecordRequestState(reqs[r], NCCL_TELEM_RDMA_REQUEST_POST_END);

  TRACE(NCCL_NET, "NET/IB: %s: Send request posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id);

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  std::atomic_thread_fence(std::memory_order_seq_cst); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->id = comm->base.fifoHead;
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    if (ncclIbTelemetryDetailed(req)) req->telemetryExpectedBytes = size;
    ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_CREATE);
    if (comm->base.resiliency) {
      memset(req->send.sentData, 0, sizeof(req->send.sentData));
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // Populate events
    int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &qpIndex));
      ncclIbAddEvent(req, qp->devIndex);
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }
    ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_READY);

    // In case the sender will write the size of the send directly to the
    // receiver's memory, prepare the source buffer which will hold the sizes
    // and be sent to the receiver.
    comm->remCmplsRecords.elems[slot][r] = req->send.size;

    TRACE(NCCL_NET, "NET/IB: %s: Send request created (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d, nreqs=%d, tag=%x, size=%ld, data=0x%016" PRIx64 ", mhandle=%p, size=%ld)", __func__, req, req->base, req->id, slot, r, nreqs, tag, size, (uint64_t)data, mhandle, size);

    *request = reqs[r] = req;

    comm->sendReqsCnt[slot]++;
    // If this is a multi-recv, send only when all requests have matched.
    if (comm->sendReqsCnt[slot] < nreqs) return ncclSuccess;

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot) {
  ncclIbQp* ctsQp = NULL;;
  NCCLCHECK(ncclIbRecvCommGetQpForCts(comm, req->id, &ctsQp));

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct rkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;

  // Populating the correct gather information based on the device and user
  // provided information
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = req->nreqs*sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to CTS FIFO slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each CTS posting QP gets drained
  if (slot == ctsQp->devIndex || comm->base.resiliency) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = slot;
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting a CTS (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(ncclIbPostSendTelemetry(req, ctsQp->qp, &wr));
  // CTS belongs to the receive request, but its occasionally-signaled SQ
  // batching can outlive that request. Record its WR/SGE shape without
  // claiming request-owned QP retirement.
  ncclIbRecordWr(req, ctsQp->qp->qp_num, &wr);

  TRACE(NCCL_NET, "NET/IB: %s: CTS posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->id = comm->base.fifoHead;
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
  if (ncclIbTelemetryDetailed(req))
    for (int i = 0; i < n; ++i) req->telemetryExpectedBytes += sizes[i];
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_CREATE);
  if (comm->base.resiliency) {
    // When resiliency is enabled, a recv request can be served by any device.
    for (int devIndex = 0; devIndex < comm->base.vProps.ndevs; devIndex++) {
      req->devBases[devIndex] = ncclIbGetNetCommDevBase(&comm->base, devIndex);
    }
  }
  TRACE(NCCL_NET, "NET/IB: %s: Recv request created (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, tag[0]=%x)", __func__, req, req->base, req->id, slot, n, tags[0]);

#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  // Store the request in a table for easy retrieval by ID.
  comm->recvReqs[req->id % NET_IB_MAX_REQUESTS] = req;

  TIME_START(1);
  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_READY);
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_POST_BEGIN);
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &qpIndex));
    ncclIbAddEvent(req, qp->devIndex);
    if (comm->prepostReceiveWorkRequests) {
      continue;
    }
    // Post receive work request on the QP
    comm->ibRecvWorkRequest.wr_id = slot;
    NCCLCHECK(ncclIbPostRecvWorkRequest(qp->qp, &comm->ibRecvWorkRequest));
    if (ncclIbTelemetryDetailed(req)) {
      qp->telemetryRqPostedWrs++;
      req->telemetryWrCount++;
      for (int c = 0; c < req->telemetryContextCount; ++c) {
        const ncclTelemetryNetRequestContext* context = req->telemetryContexts + c;
        ncclTelemetryRecordRdmaWr(context, req->id, context->proxyId, qp->qp->qp_num,
          comm->ibRecvWorkRequest.wr_id, 0xff, 0, comm->ibRecvWorkRequest.num_sge,
          0, 0, 0, 0);
      }
      ncclIbRecordQpDepth(req, qp, NCCL_TELEM_RDMA_WQE_POST, true);
    }
    if (req->telemetryContextCount != 0) {
      ncclIbRecordRequestTelemetry(req, qp->qp->qp_num, comm->ibRecvWorkRequest.wr_id,
        0, 0, 0, NCCL_TELEM_RDMA_WQE_POST, true);
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    // Start a QP event for every request in the multirecv and every qp
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = comm->ibRecvWorkRequest.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
  }
  TIME_STOP(1);

  req->recv.aggSize = 0;
  req->recv.cmplsRecords = &comm->cmplsRecords[slot];
  memset(req->recv.cmplsRecords->sizes, 0, sizeof(int)*n);
  memset(req->recv.cmplsRecords->completions, 0, sizeof(req->recv.cmplsRecords->completions));
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];
    // Send all applicable rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++) {
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
    }
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->base.fifoHead+1;
  }

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, req, slot));
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_POST_END);
  comm->base.fifoHead++;
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  // Keep the request id encoding unchanged: the flush wr_id decoder relies on
  // the request-pool index plus NCCL_IB_FLUSH_REQ_WR_ID_OFFSET.
  req->id = uint64_t(req - comm->base.reqs);
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  if (ncclIbTelemetryDetailed(req)) req->telemetryExpectedBytes = sizes[last];
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_CREATE);
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_READY);
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_POST_BEGIN);
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (req - comm->base.reqs) + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(ncclIbPostSendTelemetry(req, comm->devs[i].gpuFlush.qp.qp, &wr));
    TIME_STOP(4);

    ncclIbAddEvent(req, i);
    ncclIbRecordWr(req, comm->devs[i].gpuFlush.qp.qp->qp_num, &wr);
    ncclIbTrackSqBatch(req, &comm->devs[i].gpuFlush.qp, 1,
      wr.sg_list ? wr.sg_list->length : 0);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  }
  ncclIbRecordRequestState(req, NCCL_TELEM_RDMA_REQUEST_POST_END);

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

static inline ncclResult_t ncclIbRequestRetrieveFromCompletion(struct ncclIbNetCommBase* base, ibv_wc* wc, ncclIbRequest** req) {
  assert(req != NULL);
  assert(wc != NULL);

  // In case of a completion with error, there is no guarantee that all fields
  // of the completion are valid.
  assert(wc->status == IBV_WC_SUCCESS);

  TRACE(NCCL_NET, "NET/IB: %s: Retrieving a %s request (wr_id=%ld, opcode=%s)", __func__, base->isSend ? "send" : "recv", wc->wr_id, ibvWcOpcodeStr(wc->opcode));

  if (!base->isSend && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM && base->recvMatchingScheme == BY_ID) {
    TRACE(NCCL_NET, "NET/IB: %s: Retrieving a receive request (wr_id=%ld, opcode=%s, imm_data=%d, byte_len=%d)", __func__, wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len);
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    *req = recvComm->recvReqs[be32toh(wc->imm_data) % NET_IB_MAX_REQUESTS];
  } else if (!base->isSend && wc->opcode == IBV_WC_RDMA_READ) { // Flush request completion
    NCCLCHECK(ncclIbRequestRetrieveAsIndex(base->reqs, (wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET), req));
  } else if (!base->isSend) {
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    *req = recvComm->recvReqs[wc->wr_id];
  } else {
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)base;
    // On the sender side, the lower 8 bits of wr_id are used to retrieve the
    // request, since in multi-send case, multiple IDs are encoded in the same
    // wr_id.,
    *req = sendComm->sendReqs[wc->wr_id & 0xff][0];
  }
  TRACE(NCCL_NET, "NET/IB: %s: Retrieved a %s request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.opcode=%s, wc.imm_data=%d, wc.byte_len=%d, wc.qp_num=%u)", __func__, base->isSend ? "send" : "recv", *req, (*req)->base, (*req)->id, ncclIbReqTypeStr[(*req)->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len, wc->qp_num);
  return ncclSuccess;
}

// Error CQEs only guarantee wr_id/qp_num/status. Resolve the owning request
// without dereferencing optional opcode/imm_data fields.
static inline ncclIbRequest* ncclIbRequestFromErrorCompletion(struct ncclIbNetCommBase* base, const ibv_wc* wc) {
  if (base->isSend) {
    int slot = int(wc->wr_id & 0xff);
    if (slot < 0 || slot >= NET_IB_MAX_REQUESTS) return NULL;
    return ((ncclIbSendComm*)base)->sendReqs[slot][0];
  }
  uint64_t index = wc->wr_id;
  if (index >= NCCL_IB_FLUSH_REQ_WR_ID_OFFSET &&
      index < NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS) {
    return base->reqs[index - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET];
  }
  if (index >= NET_IB_MAX_REQUESTS) return NULL;
  return ((ncclIbRecvComm*)base)->recvReqs[index];
}

static inline bool ncclIbRequestIsComplete(struct ncclIbRequest *request) {
  bool complete = (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
  if (!complete && request->base->resiliency) {
    NCCLCHECK(ncclIbResiliencyRequestIsComplete(request, &complete));
  }
  return complete;
}

static inline void ncclIbRecordCqeDetail(struct ncclIbRequest* request,
                                         const struct ibv_wc* wc) {
  if (!ncclIbTelemetryDetailed(request)) return;
  uint64_t now = ncclIbTelemetryNowNs();
  if (request->telemetryFirstCqeNs == 0) request->telemetryFirstCqeNs = now;
  request->telemetryLastCqeNs = now;
  request->telemetryCqeCount++;
  for (int i = 0; i < request->telemetryContextCount; ++i) {
    // For error completions only wr_id, qp_num, and status are portable.
    uint32_t opcode = wc->status == IBV_WC_SUCCESS ? wc->opcode : 0;
    uint32_t byteLen = wc->status == IBV_WC_SUCCESS ? wc->byte_len : 0;
    bool hasReceiveMetadata = wc->status == IBV_WC_SUCCESS &&
      wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM;
    uint32_t srcQp = hasReceiveMetadata ? wc->src_qp : 0;
    uint32_t wcFlags = wc->status == IBV_WC_SUCCESS ? wc->wc_flags : 0;
    uint32_t immData = hasReceiveMetadata ? wc->imm_data : 0;
    ncclTelemetryRecordRdmaCqeDetail(request->telemetryContexts + i, request->telemetryGeneration,
      wc->qp_num, wc->wr_id, opcode, wc->status, wc->vendor_err, srcQp,
      wcFlags, immData, byteLen);
  }
}

static inline struct ncclIbQp* ncclIbFindTelemetryQp(struct ncclIbRequest* request,
                                                     uint32_t qpNum) {
  for (int i = 0; i < request->base->nqps; ++i) {
    struct ncclIbQp* qp = request->base->activeQps[i];
    if (qp && qp->qp && qp->qp->qp_num == qpNum) return qp;
  }
  if (!request->base->isSend && request->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)request->base;
    for (int i = 0; i < request->base->vProps.ndevs; ++i) {
      struct ncclIbQp* qp = &comm->devs[i].gpuFlush.qp;
      if (qp->qp && qp->qp->qp_num == qpNum) return qp;
    }
  }
  return NULL;
}

static inline void ncclIbRetireSqBatch(struct ncclIbRequest* request,
                                       const struct ibv_wc* wc) {
  if (!ncclIbTelemetryDetailed(request)) return;
  struct ncclIbQp* qp = ncclIbFindTelemetryQp(request, wc->qp_num);
  if (qp == NULL) return;
  for (int i = 0; i < request->telemetryRetireCount; ++i) {
    // A request can be retransmitted on the same QP. Skip batches already
    // retired so a later CQE reaches the next batch for that QP.
    if (request->telemetryRetireQpNum[i] != wc->qp_num ||
        request->telemetryRetireWrs[i] == 0) continue;
    qp->telemetrySqCompletedWrs += request->telemetryRetireWrs[i];
    qp->telemetrySqCompletedBytes += request->telemetryRetireBytes[i];
    request->telemetryCompletedBytes = std::min(request->telemetryExpectedBytes,
      request->telemetryCompletedBytes + request->telemetryRetireBytes[i]);
    request->telemetryRetireWrs[i] = 0;
    request->telemetryRetireBytes[i] = 0;
    ncclIbRecordQpDepth(request, qp, NCCL_TELEM_RDMA_CQE, false);
    break;
  }
}

static inline void ncclIbRetireRq(struct ncclIbRequest* request,
                                  const struct ibv_wc* wc) {
  if (!ncclIbTelemetryDetailed(request)) return;
  struct ncclIbQp* qp = ncclIbFindTelemetryQp(request, wc->qp_num);
  if (qp == NULL) return;
  qp->telemetryRqCompletedWrs++;
  qp->telemetryRqCompletedBytes += wc->byte_len;
  request->telemetryCompletedBytes = std::min(request->telemetryExpectedBytes,
    request->telemetryCompletedBytes + wc->byte_len);
  ncclIbRecordQpDepth(request, qp, NCCL_TELEM_RDMA_CQE, true);
}

static inline ncclResult_t ncclIbRequestComplete(struct ncclIbRequest* r, int* done, int* sizes) {
  TRACE(NCCL_NET, "NET/IB: %s: %s request completed (req=%p, comm=%p, id=%ld, type=%s)", __func__, r->base->isSend ? "Send" : "Recv", r, r->base, r->id, ncclIbReqTypeStr[r->type]);
  *done = 1;
  for (int i = 0; i < r->base->vProps.ndevs; ++i) {
    if (r->devBases[i]) ncclIbFlushPollTelemetry(r, i, r->devBases[i]->cq);
  }
  if (r->telemetryContextCount != 0) {
    // A send request is complete only after all of its QP events have drained.
    // Do not report it as fully completed when the first signaled CQE arrives.
    if (r->type == NCCL_NET_IB_REQ_SEND)
      r->telemetryCompletedBytes = r->telemetryExpectedBytes;
    ncclIbRecordRequestTelemetry(r, 0, 0, 0, 0,
      r->type == NCCL_NET_IB_REQ_SEND ? r->send.size : 0,
      NCCL_TELEM_RDMA_REQUEST_COMPLETE);
  }
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    TRACE(NCCL_NET, "NET/IB: %s: Recv request completed (req=%p, comm=%p, id=%ld, type=%s, nreqs=%d)", __func__, r, r->base, r->id, ncclIbReqTypeStr[r->type], r->nreqs);
    int *sizesToReport = (r->nreqs > 1 || r->recv.cmplsRecords->sizes[0] > 0) ? r->recv.cmplsRecords->sizes : &(r->recv.aggSize);
    if (ncclIbTelemetryDetailed(r)) {
      r->telemetryCompletedBytes = 0;
      for (int i = 0; i < r->nreqs; ++i)
        r->telemetryCompletedBytes += uint64_t(std::max(0, sizesToReport[i]));
    }
    for (int i=0; i<r->nreqs; i++) {
      sizes[i] = sizesToReport[i];
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
        NCCLCHECK(ncclProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
#endif
    }
  }
  ncclIbRecordRequestState(r, NCCL_TELEM_RDMA_STATE_COMPLETE);
  if (r->type == NCCL_NET_IB_REQ_SEND) {
    TRACE(NCCL_NET, "NET/IB: %s: Send request completed (req=%p, comm=%p, id=%ld)", __func__, r, r->base, r->id);
    if (sizes) {
      sizes[0] = r->send.size;
  #ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
        NCCLCHECK(ncclProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
  #endif
    }
    int slot = r->id % NET_IB_MAX_REQUESTS;
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)r->base;
    sendComm->sendReqsCnt[slot]--;
    if (sendComm->sendReqsCnt[slot] == 0) {
      // Only after completing the last send of a multi-recv, allow accepting
      // following send requests on the same slot.
      memset(&sendComm->sendReqs[slot], 0, sizeof(sendComm->sendReqs[slot]));
    }
  }
  // Stop all remaining Qp events for this event
  NCCLCHECK(ncclIbFreeRequest(r));
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t ncclIbLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(commBase, devIndex);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char* localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u %s%s%s%s hca %s",
      sockStr, ibvWcStatusStr(wc->status), wc->status,
      ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
      localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbCompletionEventProcess(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct ncclIbRequest* req = NULL;
  NCCLCHECK(ncclIbRequestRetrieveFromCompletion(commBase, wc, &req));
  if (req == NULL) {
    WARN("NET/IB: %s: %s comm could not retreive a request found for a successful completion (comm=%p, wc.wr_id=%ld, opcode=%d, qp_num=%u)", __func__, commBase->isSend ? "Send" : "Recv", commBase, wc->wr_id, wc->opcode, wc->qp_num);
    return ncclInternalError;
  }
  #ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, devIndex=%d",
    ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
  #endif

  if (commBase->isSend) {
    if (req->type != NCCL_NET_IB_REQ_SEND) {
      WARN("NET/IB: %s: Sender expected a 'send' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, ncclIbReqTypeStr[req->type], req, commBase, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
      return ncclInternalError;
    }
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)commBase;
    struct ncclIbRequest* sendReq = NULL;
    int slot = req->id % NET_IB_MAX_REQUESTS;
    for (int j = 0; j < req->nreqs; j++) {
      sendReq = sendComm->sendReqs[slot][j];
      if (!commBase->resiliency && (sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, devIndex=%d, reqIdx=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        return ncclInternalError;
      }
      sendReq->events[devIndex]--;
      if (sendReq->telemetryContextCount != 0) {
        ncclIbRecordRequestTelemetry(sendReq, wc->qp_num, wc->wr_id,
          wc->opcode, wc->status, wc->byte_len, NCCL_TELEM_RDMA_CQE);
        ncclIbRecordCqeDetail(sendReq, wc);
      }
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a send request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, sendReq, sendReq->base, sendReq->id, devIndex, wc->qp_num);
#ifdef NCCL_ENABLE_NET_PROFILING
      // Stop Qp event for sendReq
      int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
    }
    ncclIbRetireSqBatch(req, wc);
  } else {
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      if (req->type == NCCL_NET_IB_REQ_UNUSED && commBase->resiliency) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a data transfer but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.status=%s(%d), wc.wr_id=%ld, wc.imm_data=%d, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, req, commBase, req->id, ibvWcStatusStr(wc->status), wc->status, wc->wr_id, be32toh(wc->imm_data), ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclSuccess;
      }
      if (req->type != NCCL_NET_IB_REQ_RECV && !commBase->resiliency) {
        WARN("NET/IB: %s: Receiver expected a 'recv' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.status=%s(%d) wc.opcode=%s(%d), wc.qp_num=%u)", __func__, ncclIbReqTypeStr[req->type], req, req->base, req->id, wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclInternalError;
      }
      if (req->nreqs == 1) {
        if (commBase->recvMatchingScheme == BY_INDEX) {
          req->recv.cmplsRecords->sizes[0] = be32toh(wc->imm_data);
        } else if (req->recv.cmplsRecords->sizes[0] == 0) {
          req->recv.aggSize+= wc->byte_len;
        }
      }
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a recv request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      ncclIbRecordRequestTelemetry(req, wc->qp_num, wc->wr_id,
        wc->opcode, wc->status, wc->byte_len, NCCL_TELEM_RDMA_CQE);
      ncclIbRecordCqeDetail(req, wc);
      ncclIbRetireRq(req, wc);
      struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)commBase;
      if (recvComm->prepostReceiveWorkRequests) {
        // Post another receive work request on the QP
        ncclIbQp* qp = NULL;
        int qpIndex = -1;
        NCCLCHECK(ncclIbCommBaseGetQpByQpNum(commBase, devIndex, wc->qp_num, &qp, &qpIndex));
        req->recv.cmplsRecords->completions[qpIndex] = 1;
        ncclIbPostRecvWorkRequest(qp->qp, &recvComm->ibRecvWorkRequest);
        if (ncclIbTelemetryDetailed(req)) {
          qp->telemetryRqPostedWrs++;
          ncclIbRecordQpDepth(req, qp, NCCL_TELEM_RDMA_WQE_POST, true);
        }
      }
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_READ) {
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a flush request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      req->events[devIndex]--;
      ncclIbRecordRequestTelemetry(req, wc->qp_num, wc->wr_id,
        wc->opcode, wc->status, wc->byte_len, NCCL_TELEM_RDMA_CQE);
      ncclIbRecordCqeDetail(req, wc);
      ncclIbRetireSqBatch(req, wc);
    } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
      // This is a CTS completion
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a CTS (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      if (req->type == NCCL_NET_IB_REQ_UNUSED) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a CTS but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, req->base, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
        return ncclSuccess;
      }
    } else {
      WARN("NET/IB: %s: Unknown completion (req=%p, comm=%p, id=%ld, devIndex=%d, req->type=%s, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, commBase, req ? req->id : -1, devIndex, ncclIbReqTypeStr[req->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
      return ncclInternalError;
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  if (r->telemetryContextCount != 0 &&
      !telemetryFaultIbPollDelayInjected.load(std::memory_order_relaxed)) {
    int64_t delayUs = ncclParamTelemetryFaultIbPollDelayUs();
    int targetRank = int(ncclParamTelemetryFaultIbPollDelayRank());
    int targetChannel = int(ncclParamTelemetryFaultIbPollDelayChannel());
    bool matches = false;
    for (int i = 0; i < r->telemetryContextCount; ++i) {
      const ncclTelemetryNetRequestContext& context = r->telemetryContexts[i];
      if ((targetRank < 0 || context.rank == targetRank) &&
          (targetChannel < 0 || context.channel == targetChannel)) {
        matches = true;
        break;
      }
    }
    bool expected = false;
    if (matches && delayUs > 0 && telemetryFaultIbPollDelayInjected.compare_exchange_strong(
          expected, true, std::memory_order_relaxed)) {
      // This opt-in diagnostic fault delays CQ polling, not the network. Cap a
      // single injection so an accidental value cannot hang the proxy thread.
      delayUs = std::min<int64_t>(delayUs, 1000000);
      ncclIbRecordRequestTelemetry(r, 0, 0, 0, 0, delayUs,
        NCCL_TELEM_RDMA_POLL_DELAY);
      usleep(useconds_t(delayUs));
    }
  }

  if (r->base->resiliency && r->base->resiliency->inProgress) {
    NCCLCHECK(ncclIbResiliencyProgress(r->base->resiliency));
  }

  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (ncclIbRequestIsComplete(r)) {
      NCCLCHECK(ncclIbRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < r->base->vProps.ndevs; i++) {
      // Reasons to skip polling this device:
      // 1. When resiliency is enabled events counters might reach negative values.
      // 2. On the sender side, a request might not use all devices (e.g., upon
      //    submission of the send request, a device was not available)
      if (!r->devBases[i] || (r->events[i] == 0 && !r->base->resiliency)) {
        continue;
      }
      TIME_START(3);
      bool firstTelemetryPoll = false;
      uint64_t telemetryPollStart = 0;
      if (ncclIbTelemetryDetailed(r)) {
        telemetryPollStart = ncclIbTelemetryNowNs();
        if (r->telemetryPollWindowStartNs[i] == 0)
          r->telemetryPollWindowStartNs[i] = telemetryPollStart;
        firstTelemetryPoll = r->telemetryPollSeen[i] == 0;
        r->telemetryPollSeen[i] = 1;
      }
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (ncclIbTelemetryDetailed(r)) {
        r->telemetryPollBusyNs[i] += ncclIbTelemetryNowNs() - telemetryPollStart;
        r->telemetryPollCalls[i]++;
        if (wrDone == 0) r->telemetryPollEmpty[i]++;
        r->telemetryPollReturned[i] += wrDone;
        if (firstTelemetryPoll || wrDone > 0 || r->telemetryPollEmpty[i] >= 64)
          ncclIbFlushPollTelemetry(r, i, r->devBases[i]->cq);
      }
      if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
      if (wrDone == 0) continue;
      totalWrDone += wrDone;
      for (int w=0; w<wrDone; w++) {
        struct ibv_wc *wc = wcs+w;
        if (wc->status != IBV_WC_SUCCESS) {
          ncclIbRequest* errorReq = ncclIbRequestFromErrorCompletion(r->base, wc);
          if (errorReq != NULL && errorReq->telemetryContextCount != 0) {
            ncclIbRecordRequestTelemetry(errorReq, wc->qp_num, wc->wr_id,
              0, wc->status, 0, NCCL_TELEM_RDMA_CQE);
            ncclIbRecordCqeDetail(errorReq, wc);
          }
          if (r->base->resiliency == NULL) {
            WARN("NET/IB: %s: Got CQE with error (devIndex=%d, req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)", __func__, i, r, r->base, r->base->isSend ? "send" : "recv", wc->wr_id, wc->qp_num);
            ncclIbLogCompletionWithError(r->base, wc, i);
            // If resiliency is not enabled, we cannot recover from any error.
            return ncclRemoteError;
          }
          NCCLCHECK(ncclIbResiliencyHandleCompletionError(r->base->resiliency, wc, i));
        } else {
          TRACE(NCCL_NET, "NET/IB: %s: Processing a completion event (devIndex=%d, comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)", __func__, i, r->base, r->base->isSend ? "send" : "recv", r, wc->wr_id, wc->qp_num);
          NCCLCHECK(ncclIbCompletionEventProcess(r->base, wc, i));
        }
      }
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

  // If no (more) CQEs found on any device, return and come back later
  return ncclSuccess;
}
