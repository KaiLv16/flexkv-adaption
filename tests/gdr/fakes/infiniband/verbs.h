#pragma once

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

struct ibv_device {};
struct ibv_context {};
struct ibv_pd {};
struct ibv_gid { uint8_t raw[16]; };
struct ibv_port_attr {
  int active_mtu, link_layer, lid, state, gid_tbl_len;
  uint32_t max_msg_sz;
};
struct ibv_device_attr {
  uint64_t max_mr_size;
  int max_cqe, max_qp_wr, max_qp_rd_atom, max_qp_init_rd_atom, phys_port_cnt;
};
struct ibv_mr { uint32_t lkey, rkey; uintptr_t base; size_t size; };
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
  uint64_t wr_id;
  ibv_send_wr *next;
  ibv_sge *sg_list;
  int num_sge, opcode, send_flags;
  struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
};
struct ibv_wc { uint64_t wr_id; int status; uint32_t qp_num; };
struct FakeCopy { void *dst; const void *src; size_t bytes; };
struct FakeCompletion { uint64_t tail; uint32_t qp_num; std::vector<FakeCopy> copies; };
struct ibv_cq { std::deque<FakeCompletion> pending; };
struct ibv_qp { uint32_t qp_num; ibv_cq *cq; int port; };
struct ibv_qp_init_attr {
  ibv_cq *send_cq, *recv_cq;
  int qp_type;
  struct { unsigned max_send_wr, max_recv_wr, max_send_sge, max_recv_sge; } cap;
};
struct ibv_qp_attr {
  int qp_state, port_num, qp_access_flags, path_mtu, dest_qp_num;
  int max_dest_rd_atomic, min_rnr_timer, timeout, retry_cnt, rnr_retry, max_rd_atomic;
  struct {
    int port_num, dlid, is_global;
    struct { ibv_gid dgid; int sgid_index, hop_limit; } grh;
  } ah_attr;
};
enum {
  IBV_ACCESS_LOCAL_WRITE=1, IBV_ACCESS_REMOTE_READ=2, IBV_ACCESS_REMOTE_WRITE=4,
  IBV_QP_STATE=1, IBV_QP_PKEY_INDEX=2, IBV_QP_PORT=4, IBV_QP_ACCESS_FLAGS=8,
  IBV_QP_AV=16, IBV_QP_PATH_MTU=32, IBV_QP_DEST_QPN=64, IBV_QP_RQ_PSN=128,
  IBV_QP_MAX_DEST_RD_ATOMIC=256, IBV_QP_MIN_RNR_TIMER=512, IBV_QP_SQ_PSN=1024,
  IBV_QP_TIMEOUT=2048, IBV_QP_RETRY_CNT=4096, IBV_QP_RNR_RETRY=8192,
  IBV_QP_MAX_QP_RD_ATOMIC=16384, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS,
  IBV_QPT_RC, IBV_LINK_LAYER_ETHERNET, IBV_WR_RDMA_WRITE, IBV_WR_RDMA_READ,
  IBV_PORT_ACTIVE=4, IBV_SEND_SIGNALED=1, IBV_WC_SUCCESS=0
};

namespace fake {
static ibv_device device;
static ibv_device *devices[] = {&device};
static uint32_t next_key = 1, next_qp = 1;
static int live_qps = 0, reject_index = -1;
static uint32_t blocked_qp = 0;
static bool hold_cq = false, bad_sequence = false, cq_error = false;
static std::map<uint32_t, ibv_mr *> registrations;
}

inline const char *ibv_get_device_name(ibv_device *) { return "mock"; }
inline ibv_device **ibv_get_device_list(int *count) { *count = 1; return fake::devices; }
inline void ibv_free_device_list(ibv_device **) {}
inline int ibv_fork_init() { return 0; }
inline ibv_context *ibv_open_device(ibv_device *) { return new ibv_context; }
inline int ibv_close_device(ibv_context *ctx) { delete ctx; return 0; }
inline ibv_pd *ibv_alloc_pd(ibv_context *) { return new ibv_pd; }
inline int ibv_dealloc_pd(ibv_pd *pd) { delete pd; return 0; }
inline int ibv_query_device(ibv_context *, ibv_device_attr *attr) {
  *attr = {1u << 30, 16, 8, 4, 4, 2};
  return 0;
}
inline int ibv_query_port(ibv_context *, int port, ibv_port_attr *attr) {
  *attr = {1, 0, 0, port == 2 ? IBV_PORT_ACTIVE : 0, 0, 1u << 20};
  return 0;
}
inline int ibv_query_gid(ibv_context *, int, int, ibv_gid *) { return -1; }
inline ibv_cq *ibv_create_cq(ibv_context *, int, void *, void *, int) { return new ibv_cq; }
inline int ibv_destroy_cq(ibv_cq *cq) { delete cq; return 0; }
inline ibv_qp *ibv_create_qp(ibv_pd *, ibv_qp_init_attr *attr) {
  ++fake::live_qps;
  return new ibv_qp{fake::next_qp++, attr->send_cq, 0};
}
inline int ibv_destroy_qp(ibv_qp *qp) { --fake::live_qps; delete qp; return 0; }
inline int ibv_modify_qp(ibv_qp *qp, ibv_qp_attr *attr, int) {
  if (attr->qp_state == IBV_QPS_INIT) qp->port = attr->port_num;
  if (attr->qp_state == IBV_QPS_RTR) assert(qp->port == 2 && attr->ah_attr.port_num == 2);
  return 0;
}
inline ibv_mr *ibv_reg_mr(ibv_pd *, void *base, size_t bytes, int) {
  const auto key = fake::next_key++;
  auto *mr = new ibv_mr{key, key, reinterpret_cast<uintptr_t>(base), bytes};
  fake::registrations.emplace(key, mr);
  return mr;
}
inline int ibv_dereg_mr(ibv_mr *mr) {
  fake::registrations.erase(mr->lkey);
  delete mr;
  return 0;
}
inline void check_mr(uint32_t key, uint64_t address, size_t bytes) {
  const auto *mr = fake::registrations.at(key);
  assert(address >= mr->base && address - mr->base < mr->size);
  assert(bytes <= mr->size - (address - mr->base));
}
inline int ibv_post_send(ibv_qp *qp, ibv_send_wr *wr, ibv_send_wr **bad) {
  std::vector<FakeCopy> copies;
  for (int i = 0; wr; wr = wr->next, ++i) {
    if (i == fake::reject_index) { *bad = wr; return 1; }
    assert(wr->num_sge == 1);
    const auto &sge = *wr->sg_list;
    check_mr(sge.lkey, sge.addr, sge.length);
    check_mr(wr->wr.rdma.rkey, wr->wr.rdma.remote_addr, sge.length);
    auto *local = reinterpret_cast<void *>(sge.addr);
    auto *remote = reinterpret_cast<void *>(wr->wr.rdma.remote_addr);
    const bool read = wr->opcode == IBV_WR_RDMA_READ;
    copies.push_back({read ? local : remote, read ? remote : local, sge.length});
    if (wr->send_flags & IBV_SEND_SIGNALED) {
      qp->cq->pending.push_back({wr->wr_id, qp->qp_num, copies});
      copies.clear();
    }
  }
  assert(copies.empty());
  return 0;
}
inline int ibv_poll_cq(ibv_cq *cq, int, ibv_wc *wc) {
  if (fake::hold_cq) return 0;
  const auto done = std::find_if(cq->pending.begin(), cq->pending.end(),
      [](const FakeCompletion &entry) { return entry.qp_num != fake::blocked_qp; });
  if (done == cq->pending.end()) return 0;
  for (const auto &copy : done->copies) std::memcpy(copy.dst, copy.src, copy.bytes);
  *wc = {done->tail + (fake::bad_sequence ? 16 : 0),
         fake::cq_error ? 1 : IBV_WC_SUCCESS, done->qp_num};
  cq->pending.erase(done);
  return 1;
}
