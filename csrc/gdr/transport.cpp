// SPDX-License-Identifier: Apache-2.0
// Local RC endpoint setup is based on FlexKV PR #348.
#include "gdr/transport.h"
#include "gdr/cq_watchdog.h"
#include "gdr/topology.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unistd.h>

#include <cuda_runtime.h>
#ifndef FLEXKV_GDR_MAPPED
#include <cuda.h>
#endif
#include <infiniband/verbs.h>
#include "logging.h"

namespace flexkv {
namespace gdr {
namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(std::string("GDR: ") + message);
}
std::string canonical(const std::string &p) {
  const std::unique_ptr<char, decltype(&std::free)> path(realpath(p.c_str(), nullptr), &std::free);
  return path ? std::string(path.get()) : std::string();
}
struct Mr { void *ptr; size_t bytes; ibv_mr *mr; uintptr_t mapped; };
bool contains(uintptr_t base,size_t size,uintptr_t address,size_t bytes) {
  return address>=base && address-base<=size && bytes<=size-(address-base);
}
ibv_mr *cover(const std::vector<Mr> &mrs,void *p,size_t n) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  for (const auto &m:mrs) {
    const auto b=reinterpret_cast<uintptr_t>(m.ptr);
    if(a>=b && a-b<=m.bytes && n<=m.bytes-(a-b)) return m.mr;
  }
  return nullptr;
}
// Peer-memory registration may require host-page-aligned addresses and lengths.
// Packed KV tensors need not end on a host page boundary.
// Expand the verbs window to the pages that contain the tensor. The stored Mr
// keeps the caller's pointer and length, so exported gpu_address values stay
// on the original mapping.
struct PageWindow { void *addr; size_t bytes; };
PageWindow verbs_page_window(void *p,size_t bytes) {
  const long probed=sysconf(_SC_PAGESIZE);
  const size_t page=probed>0?static_cast<size_t>(probed):4096u;
  const auto start=reinterpret_cast<uintptr_t>(p);
  const auto end=start+bytes;
  const auto aligned=start-start%page;
  const auto rem=end%page;
  const auto aligned_end=rem==0?end:end+(page-rem);
  require(aligned_end>=end,"page-aligned GPU registration overflow");
  return {reinterpret_cast<void *>(aligned),static_cast<size_t>(aligned_end-aligned)};
}

int active_port(ibv_context *ctx, int requested, ibv_port_attr &port) {
  ibv_device_attr device{};
  if (ibv_query_device(ctx, &device) != 0 || requested > device.phys_port_cnt)
    return 0;
  const int first = requested ? requested : 1;
  const int last = requested ? requested : device.phys_port_cnt;
  for (int index = first; index <= last; ++index) {
    if (ibv_query_port(ctx, index, &port) == 0 && port.state == IBV_PORT_ACTIVE)
      return index;
  }
  return 0;
}

}  // namespace

struct Transport::Impl {
  struct Operation {
    uint64_t sequence;
    bool read;
    uint32_t bytes;
    int scratch;
    bool signaled;
  };

  static constexpr size_t kScratchStride = 64;
  static constexpr size_t kMaxFences = 16;
  static constexpr size_t kMaxSge = 16;
  static constexpr size_t kScratchBytes = kMaxFences * kScratchStride;
  static constexpr unsigned kQpIndexBits = 4;
  static_assert(kMaxQps == (1u << kQpIndexBits), "QP index must fit in WR id");

  struct Queue {
    ibv_qp *qp = nullptr;
#ifdef FLEXKV_GDR_WR_API
    ibv_qp_ex *qp_ex = nullptr;
#endif
    std::vector<ibv_send_wr> wrs;
    std::vector<ibv_sge> sges;
    std::vector<Operation> ledger;
    std::array<bool, kMaxFences> scratch_used{};
    Span last_write{};
    uint64_t posted = 0;
    uint64_t retired = 0;
    uint64_t last_checked = 0;
    uint64_t payload_bytes = 0;
    int reads = 0;
    int fences = 0;
    bool needs_readback = false;
    CqWatchdog watchdog{std::chrono::seconds(30)};
  };

  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  std::vector<Queue> queues;
  std::vector<ibv_qp *> targets;
  std::vector<Mr> hosts, devices;
  std::vector<RemoteRegion> remote_regions;
  ibv_port_attr port{};
  int port_num = 0;
  ibv_gid gid{};
  int gid_index = 0;
  int depth = 0;
  int read_depth = 0;
  int write_sge_limit = 1;
  int read_sge_limit = 1;
  size_t h2d_qps = 1;
  size_t d2h_qps = 1;
  bool use_wr_api = false;
  uint64_t max_mr_size = 0;
  bool sender = false;
  bool connected = false;
  bool unsafe = false;
  std::string name, nic_selector, endpoint;
  size_t host_hint = 0;
  size_t remote_hint = 0;

  std::vector<unsigned char> scratch_storage;
  unsigned char *scratch = nullptr;
  ibv_mr *scratch_mr = nullptr;
  int fence_depth = 0;
  Stats stats{};

  ~Impl() {
    for (const auto &queue : queues)
      if (queue.qp) ibv_destroy_qp(queue.qp);
    for (auto *target : targets) ibv_destroy_qp(target);
    if (scratch_mr) ibv_dereg_mr(scratch_mr);
    for (auto &mr : hosts) ibv_dereg_mr(mr.mr);
    for (auto &mr : devices) ibv_dereg_mr(mr.mr);
    if (cq) ibv_destroy_cq(cq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
  }

  [[noreturn]] void fail(const std::string &message) {
    unsafe = true;
    throw UnsafeTransfer("GDR: " + message);
  }

#ifdef FLEXKV_GDR_WR_API
  ibv_qp *create(int requested_send_sge = 1, int *actual_send_sge = nullptr,
                 ibv_qp_ex **out_qp_ex = nullptr) {
#else
  ibv_qp *create(int requested_send_sge = 1, int *actual_send_sge = nullptr) {
#endif
    int try_sge = std::max(1, requested_send_sge);
    while (true) {
      ibv_qp_init_attr attr{};
      attr.send_cq = cq;
      attr.recv_cq = cq;
      attr.qp_type = IBV_QPT_RC;
      attr.cap.max_send_wr = depth;
      attr.cap.max_recv_wr = 1;
      attr.cap.max_send_sge = static_cast<unsigned>(try_sge);
      attr.cap.max_recv_sge = 1;
#ifdef FLEXKV_GDR_WR_API
      if (out_qp_ex && use_wr_api) {
        ibv_qp_init_attr_ex ex{};
        ex.send_cq = cq;
        ex.recv_cq = cq;
        ex.qp_type = IBV_QPT_RC;
        ex.pd = pd;
        ex.cap = attr.cap;
        ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_READ;
        auto *q = ibv_create_qp_ex(ctx, &ex);
        if (q) {
          auto *qex = ibv_qp_to_qp_ex(q);
          if (qex && ex.cap.max_send_wr >= static_cast<unsigned int>(depth) &&
              ex.cap.max_send_sge >= static_cast<unsigned int>(try_sge)) {
            *out_qp_ex = qex;
            if (actual_send_sge) *actual_send_sge = try_sge;
            return q;
          }
          ibv_destroy_qp(q);
          if (!qex) {
            use_wr_api = false;
            try_sge = std::max(1, requested_send_sge);
            continue;
          }
        } else if (errno == EOPNOTSUPP || errno == ENOSYS) {
          use_wr_api = false;
          try_sge = std::max(1, requested_send_sge);
          continue;
        }
        if (try_sge > 1) {
          try_sge /= 2;
          continue;
        }
        use_wr_api = false;
        try_sge = std::max(1, requested_send_sge);
        continue;
      }
#endif
      auto *result = ibv_create_qp(pd, &attr);
      if (result && attr.cap.max_send_wr >= static_cast<unsigned int>(depth) &&
          attr.cap.max_send_sge >= static_cast<unsigned int>(try_sge)) {
        if (actual_send_sge) *actual_send_sge = try_sge;
        return result;
      }
      if (result) ibv_destroy_qp(result);
      if (try_sge == 1) {
        require(result != nullptr, "create_qp failed");
        throw std::runtime_error("GDR: provider reduced SQ capacity");
      }
      try_sge /= 2;
    }
  }

  void modify(ibv_qp *q,ibv_qp_attr &a,int mask,const char *stage) {
    const int rc=ibv_modify_qp(q,&a,mask);
    const int saved_errno=errno;
    if(!rc) return;
    std::ostringstream error;
    error << "GDR: QP " << stage << " failed: nic=" << name
          << " port=" << port_num << " gid_index=" << gid_index << " qpn=" << q->qp_num
          << " peer=" << a.dest_qp_num << " mtu=" << port.active_mtu
          << " link_layer=" << int(port.link_layer) << " read_depth=" << read_depth
          << " rc=" << rc << " (" << std::strerror(rc>0?rc:saved_errno) << ")"
          << " endpoint=" << endpoint;
    throw std::runtime_error(error.str());
  }
  void init_qp(ibv_qp *q) {
    // Port/P_Key/GID select a shared fabric endpoint, not exclusive resources.
    // PSNs start at zero on each new RC connection; they are scoped to its QP.
    ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.port_num=port_num;
    a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    modify(q,a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS,"INIT");
  }
  void connect(ibv_qp *q,uint32_t peer,bool initialized=false) {
    require(peer>0 && peer<=0xffffff,"invalid peer QPN");
    if(!initialized) init_qp(q);
    ibv_qp_attr a{};a.qp_state=IBV_QPS_RTR;a.path_mtu=port.active_mtu;a.dest_qp_num=peer;
    a.max_dest_rd_atomic=read_depth;a.min_rnr_timer=12;a.ah_attr.port_num=port_num;
    if(port.link_layer==IBV_LINK_LAYER_ETHERNET) {
      // Both QPs use this port and GID. The GRH describes a local peer.
      a.ah_attr.is_global=1;a.ah_attr.grh.dgid=gid;
      a.ah_attr.grh.sgid_index=gid_index;a.ah_attr.grh.hop_limit=1;
    } else a.ah_attr.dlid=port.lid;
    modify(q,a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER,"RTR");
    a={};a.qp_state=IBV_QPS_RTS;a.timeout=14;a.retry_cnt=7;a.rnr_retry=7;a.max_rd_atomic=read_depth;
    modify(q,a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC,"RTS");
  }

  template <typename Region, typename Address>
  static const Region &lookup(const std::vector<Region> &regions, size_t &hint,
                              uintptr_t address, Address base) {
    if (hint < regions.size() && contains(base(regions[hint]), regions[hint].bytes,
                                          address, 1)) {
      return regions[hint];
    }
    const auto next = std::upper_bound(regions.begin(), regions.end(), address,
        [&](uintptr_t key, const Region &region) { return key < base(region); });
    if (next != regions.begin()) {
      const auto index = static_cast<size_t>(next - regions.begin() - 1);
      if (contains(base(regions[index]), regions[index].bytes, address, 1)) {
        hint = index;
        return regions[index];
      }
    }
    throw std::invalid_argument("GDR: address outside the registered regions");
  }

  uint64_t wr_id(uint64_t sequence, size_t index) {
    if (sequence > (std::numeric_limits<uint64_t>::max() >> kQpIndexBits))
      fail("WR sequence overflow");
    return (sequence << kQpIndexBits) | index;
  }

  void post(Queue &queue, size_t count) {
#ifdef FLEXKV_GDR_WR_API
    if (queue.qp_ex) {
      ibv_wr_start(queue.qp_ex);
      for (size_t i = 0; i < count; ++i) {
        const auto &wr = queue.wrs[i];
        queue.qp_ex->wr_id = wr.wr_id;
        queue.qp_ex->wr_flags = static_cast<unsigned int>(wr.send_flags);
        if (wr.opcode == IBV_WR_RDMA_READ) {
          ibv_wr_rdma_read(queue.qp_ex, wr.wr.rdma.rkey, wr.wr.rdma.remote_addr);
        } else {
          ibv_wr_rdma_write(queue.qp_ex, wr.wr.rdma.rkey, wr.wr.rdma.remote_addr);
        }
        if (wr.num_sge == 1) {
          const auto &sge = wr.sg_list[0];
          ibv_wr_set_sge(queue.qp_ex, sge.lkey, sge.addr, sge.length);
        } else {
          ibv_wr_set_sge_list(queue.qp_ex, static_cast<size_t>(wr.num_sge), wr.sg_list);
        }
      }
      const int error = ibv_wr_complete(queue.qp_ex);
      if (error) {
        queue.posted += count;
        fail("ibv_wr_complete failed; count=" + std::to_string(count) +
             ", error=" + std::to_string(error));
      }
      queue.posted += count;
      ++stats.posts;
      return;
    }
#endif
    ibv_send_wr *bad = nullptr;
    const int error = ibv_post_send(queue.qp, queue.wrs.data(), &bad);
    if (error) {
      // A failed post may have accepted a prefix. No caller may destroy the
      // route, reuse memory, or replay until the session has been isolated.
      size_t accepted = 0;
      while (accepted < count && &queue.wrs[accepted] != bad) ++accepted;
      queue.posted += accepted;
      fail("post_send failed; possible accepted prefix=" + std::to_string(accepted) +
           ", error=" + std::to_string(error));
    }
    queue.posted += count;
    ++stats.posts;
  }

  bool post_readback(size_t index) {
    auto &queue = queues[index];
    if (queue.fences == fence_depth || queue.reads == read_depth ||
        queue.posted - queue.retired >= static_cast<size_t>(depth)) return false;
    const auto free = std::find(queue.scratch_used.begin(), queue.scratch_used.end(), false);
    const auto slot = static_cast<int>(free - queue.scratch_used.begin());
    const auto seq = queue.posted + 1;
    const auto &last = queue.last_write;
    const uint32_t rb_len = std::min(last.bytes, uint32_t(8));
    const uint64_t rb_remote = last.remote + (last.bytes - rb_len);
    queue.sges[0] = {reinterpret_cast<uint64_t>(scratch + index * kScratchBytes +
                                               slot * kScratchStride),
                     rb_len, scratch_mr->lkey};
    auto &wr = queue.wrs[0];
    wr = {};
    wr.wr_id = wr_id(seq, index);
    wr.sg_list = &queue.sges[0];
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = rb_remote;
    wr.wr.rdma.rkey = last.rkey;
    queue.ledger[seq % depth] = {seq, true, 0, slot, true};
    post(queue, 1);
    *free = true;
    ++queue.reads;
    ++queue.fences;
    ++stats.readbacks;
    return true;
  }
};

Transport::Transport(int device, bool sender, const std::string &nic,
                     const std::string &required_endpoint, int requested_depth,
                     int requested_fences, size_t qp_count, size_t h2d_qp_count,
                     size_t d2h_qp_count, int max_sge) : p_(new Impl) {
  auto &e = *p_;
  require(qp_count >= 1 && qp_count <= kMaxQps, "QP count must be 1..16");
  e.h2d_qps = h2d_qp_count ? h2d_qp_count : std::min<size_t>(qp_count, 4);
  e.d2h_qps = d2h_qp_count ? d2h_qp_count : qp_count;
  require(e.h2d_qps >= 1 && e.h2d_qps <= qp_count, "H2D QP count must be 1..qp_count");
  require(e.d2h_qps >= 1 && e.d2h_qps <= qp_count, "D2H QP count must be 1..qp_count");
  require(max_sge >= 0 && max_sge <= static_cast<int>(Impl::kMaxSge), "max_sge must be 0..16");
  e.sender = sender;
#ifdef FLEXKV_GDR_WR_API
  const char *wr_env = std::getenv("FLEXKV_GDR_WR_API");
  e.use_wr_api = sender && (!wr_env || std::strcmp(wr_env, "0") != 0);
#endif
  require(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice failed");
  // Workers spawn a fresh interpreter. An Owner may already have live MRs
  // from another transport, in which case fork protection cannot be enabled
  // retroactively. The local GDR session never uses inherited verbs objects.
  static std::once_flag fork_once;
  std::call_once(fork_once, [] {
    const int result = ibv_fork_init();
    require(result == 0 || result == EINVAL, "ibv_fork_init failed");
  });
  int n=0;ibv_device **raw=ibv_get_device_list(&n);
  std::unique_ptr<ibv_device*,decltype(&ibv_free_device_list)> list(raw,ibv_free_device_list);
  require(raw && n,"no verbs devices; check rdma-core providers and /dev/infiniband");
  std::string selected_nic=nic;
  int requested_port = 0;
  const auto colon = selected_nic.find(':');
  if (colon != std::string::npos) {
    const auto suffix = selected_nic.substr(colon + 1);
    require(colon != 0 && !suffix.empty() && suffix.size() <= 3 &&
            suffix.find_first_not_of("0123456789") == std::string::npos,
            "FLEXKV_GDR_NICS must use NIC or NIC:PORT");
    requested_port = std::stoi(suffix);
    require(requested_port > 0 && requested_port <= 255, "invalid RDMA port number");
    selected_nic.resize(colon);
  }
  char bus[32]{};
  const auto pci_status=cudaDeviceGetPCIBusId(bus,sizeof(bus),device);
  std::string bdf=pci_status==cudaSuccess ? bus : "";
  if(pci_status!=cudaSuccess && !nic.empty()) cudaGetLastError();
  std::transform(bdf.begin(),bdf.end(),bdf.begin(),[](unsigned char c){return std::tolower(c);});
  // The endpoint fingerprint must match whether the owner selected automatically
  // or the worker received its explicit NIC through the owner handshake.
  if(selected_nic.empty()) {
    require(pci_status==cudaSuccess,
            "cannot query logical device PCI bus; set FLEXKV_GDR_NICS explicitly");
    const auto device_path=canonical("/sys/bus/pci/devices/"+bdf);
    require(!bdf.empty() && !device_path.empty(),"logical device PCI path missing from sysfs; cannot auto-select NIC");
    std::vector<std::pair<std::string,std::string>> candidates;
    for(int i=0;i<n;++i) {
      const std::string name=ibv_get_device_name(raw[i]);
      auto ctx=ibv_open_device(raw[i]);if(!ctx) continue;
      ibv_port_attr port{};
      const bool active=active_port(ctx, 0, port) != 0;
      ibv_close_device(ctx);
      if(active) candidates.emplace_back(name,canonical("/sys/class/infiniband/"+name+"/device"));
    }
    selected_nic=candidates.at(select_local_nic(device_path,candidates)).first;
    FLEXKV_LOG_INFO("[GDR-NIC] logical_device=%d pci=%s nic=%s selection=pcie_topology",
                    device,bdf.c_str(),selected_nic.c_str());
  }
  for(int i=0;i<n;++i) {
    if(selected_nic!=ibv_get_device_name(raw[i])) continue;
    e.ctx=ibv_open_device(raw[i]);e.name=selected_nic;break;
  }
  require(e.ctx, "selected NIC is unavailable; check FLEXKV_GDR_NICS");
  e.port_num = active_port(e.ctx, requested_port, e.port);
  require(e.port_num, "selected NIC has no matching active port; check FLEXKV_GDR_NICS and RDMA port state");
  const auto port_string = std::to_string(e.port_num);
  e.nic_selector = e.name + ":" + port_string;
  struct Candidate {int priority,index;ibv_gid gid;};
  std::vector<Candidate> candidates;
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    for(int i=0;i<e.port.gid_tbl_len;++i) {
      std::ifstream f("/sys/class/infiniband/"+e.name+"/ports/"+port_string+"/gid_attrs/types/"+std::to_string(i));
      std::string type;std::getline(f,type);
      const bool v2=type.find("RoCE v2")!=std::string::npos;
      const bool v1=type.find("RoCE v1")!=std::string::npos;
      ibv_gid gid{};
      if((!v1 && !v2) || ibv_query_gid(e.ctx,e.port_num,i,&gid)) continue;
      bool nonzero=false;for(auto v:gid.raw) nonzero |= v!=0;
      const bool local=gid.raw[0]==0xfe && (gid.raw[1]&0xc0)==0x80;
      if(nonzero) candidates.push_back({v2?(local?1:0):2,i,gid});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate &a, const Candidate &b) { return a.priority < b.priority; });
  } else candidates.push_back({0,0,{}});
  require(!candidates.empty(),"no nonzero RoCE GID with a known v1/v2 type");

  ibv_device_attr attr{};
  require(ibv_query_device(e.ctx, &attr) == 0, "query_device failed");
  e.max_mr_size = attr.max_mr_size;
  const auto count = static_cast<int>(qp_count);
  e.depth = std::min({256, attr.max_qp_wr, attr.max_cqe / count});
  const int capacity = std::min({128, attr.max_qp_rd_atom, attr.max_qp_init_rd_atom});
  require(e.depth > 1 && capacity > 0 && e.port.max_msg_sz, "insufficient SQ/CQ or READ capacity");
  require(requested_depth >= 0 && requested_depth <= capacity, "invalid READ depth");
  e.read_depth = requested_depth ? requested_depth : std::min(capacity, 64);
  const int hw_write_sge = attr.max_sge > 0 ? attr.max_sge : 1;
  const int hw_read_sge = attr.max_sge_rd > 0 ? attr.max_sge_rd : hw_write_sge;
  int env_max_sge = max_sge;
  if (env_max_sge == 0) {
    if (const char *sge_env = std::getenv("FLEXKV_GDR_MAX_SGE")) {
      char *endptr = nullptr;
      const long parsed = std::strtol(sge_env, &endptr, 10);
      require(endptr != sge_env && *endptr == '\0' &&
                  parsed >= 0 && parsed <= static_cast<long>(Impl::kMaxSge),
              "FLEXKV_GDR_MAX_SGE must be 0..16");
      env_max_sge = static_cast<int>(parsed);
    }
  }
  const int sge_cap = env_max_sge > 0 ? env_max_sge : static_cast<int>(Impl::kMaxSge);
  const int desired_send_sge = sender
      ? std::max(1, std::min({hw_write_sge, sge_cap, static_cast<int>(Impl::kMaxSge)}))
      : 1;
#ifdef FLEXKV_GDR_MAPPED
  e.fence_depth = sender && requested_fences == 0
      ? std::min({4, e.depth - 1, e.read_depth}) : requested_fences;
  require(!sender || (e.fence_depth > 0 && e.fence_depth < e.depth &&
      e.fence_depth <= e.read_depth && e.fence_depth <= static_cast<int>(Impl::kMaxFences)),
      "invalid mapped read-back depth");
#else
  (void)requested_fences;
#endif
  e.pd = ibv_alloc_pd(e.ctx);
  require(e.pd, "alloc_pd failed");
  e.cq = ibv_create_cq(e.ctx, sender ? e.depth * count : e.depth, nullptr, nullptr, 0);
  require(e.cq, "create_cq failed");
  if (sender && e.fence_depth) {
    const auto scratch_bytes = qp_count * Impl::kScratchBytes;
    e.scratch_storage.resize(scratch_bytes + Impl::kScratchStride - 1);
    void *scratch = e.scratch_storage.data();
    size_t space = e.scratch_storage.size();
    e.scratch = static_cast<unsigned char *>(std::align(
        Impl::kScratchStride, scratch_bytes, scratch, space));
    require(e.scratch, "cannot align read-back scratch storage");
    e.scratch_mr = ibv_reg_mr(e.pd, e.scratch, scratch_bytes, IBV_ACCESS_LOCAL_WRITE);
    require(e.scratch_mr, "read-back scratch registration failed");
  }
  for (const auto &candidate : candidates) {
    e.gid_index = candidate.index;
    e.gid = candidate.gid;
#ifdef FLEXKV_GDR_MAPPED
    const char *registration = "mapped";
#else
    const char *registration = "native";
#endif
    e.endpoint = std::string(registration) + "|" + bdf + "|" + e.name + "|" + port_string + "|" + std::to_string(e.gid_index) + "|" +
                 std::to_string(e.port.active_mtu) + "|" + std::to_string(e.port.lid);
    for (auto byte : e.gid.raw) e.endpoint += "|" + std::to_string(byte);
    e.endpoint += "|" + std::to_string(qp_count);
    if (!required_endpoint.empty() && required_endpoint != e.endpoint) continue;
    if (sender) {
      e.queues.resize(qp_count);
      e.stats.qp_payload_wrs.resize(qp_count);
      int negotiated_sge = desired_send_sge;
      for (auto &queue : e.queues) {
        int actual_sge = 1;
#ifdef FLEXKV_GDR_WR_API
        queue.qp = e.create(negotiated_sge, &actual_sge, &queue.qp_ex);
#else
        queue.qp = e.create(negotiated_sge, &actual_sge);
#endif
        negotiated_sge = std::min(negotiated_sge, actual_sge);
        e.init_qp(queue.qp);
      }
      e.write_sge_limit = negotiated_sge;
      e.read_sge_limit = std::max(1, std::min(hw_read_sge, negotiated_sge));
      for (auto &queue : e.queues) {
        queue.wrs.resize(e.depth);
        queue.sges.resize(static_cast<size_t>(e.depth) * static_cast<size_t>(e.write_sge_limit));
        queue.ledger.resize(e.depth);
      }
    }
    return;
  }
  throw std::runtime_error("GDR: no matching local RC endpoint on " + e.name);
}

// Until device-level isolation is established, preserve verbs resources after
// any uncertainty. The owning process must also retain the backing memory.
Transport::~Transport() {
  if (p_ && (p_->unsafe || has_outstanding())) (void)p_.release();
}

void Transport::register_device(void *p,size_t bytes) {
  auto &e=*p_;require(!e.sender,"never register an imported CUDA IPC alias");
  require(p && bytes && bytes<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(p),"invalid GPU registration range");
  if(cover(e.devices,p,bytes)) return;
  void *mapped=p;
#ifdef FLEXKV_GDR_MAPPED
  // Mapped GDR uses the allocation's peer-memory address. This is not
  // host staging: both MR registration and WQEs must use the mapped address.
  cudaPointerAttributes attr{};
  require(cudaPointerGetAttributes(&attr,p)==cudaSuccess && attr.hostPointer,
          "mapped registration failed: check CUDA runtime peer-memory mapping support and the peer-memory driver");
  mapped=attr.hostPointer;
#else
  unsigned sync=1;
  require(cuPointerSetAttribute(&sync,CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,reinterpret_cast<CUdeviceptr>(p))==CUDA_SUCCESS,"GPU SYNC_MEMOPS failed");
#endif
  require(bytes<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(mapped),"mapped device range overflow");
  const auto window=verbs_page_window(mapped,bytes);
  const int access=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
  auto mr=ibv_reg_mr(e.pd,window.addr,window.bytes,access);
  if(!mr && (window.addr!=mapped || window.bytes!=bytes))
    mr=ibv_reg_mr(e.pd,mapped,bytes,access);
  if(!mr) {
    const int error=errno;
    throw std::runtime_error("GDR: GPU MR registration failed nic="+e.name+
        " bytes="+std::to_string(bytes)+" reg_bytes="+std::to_string(window.bytes)+
        " errno="+std::to_string(error)+
        " ("+std::strerror(error)+"); check the device peer-memory driver and memlock limit");
  }
  try { e.devices.push_back({p,bytes,mr,reinterpret_cast<uintptr_t>(mapped)}); }
  catch (...) { ibv_dereg_mr(mr); throw; }
}

void Transport::register_host(void *base, size_t bytes) {
  auto &e = *p_;
  require(e.sender && !e.connected && base && bytes &&
          bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(base), "invalid host MR");
  if (cover(e.hosts, base, bytes)) return;
  const long page = sysconf(_SC_PAGESIZE);
  require(page > 0, "cannot determine Host page size");
  const size_t chunk = std::min(e.max_mr_size, uint64_t(1) << 30) / page * page;
  require(chunk > 0, "NIC cannot register one Host page");
  auto *ptr = static_cast<unsigned char *>(base);
  while (bytes) {
    const auto length = std::min(chunk, bytes);
    auto *mr = ibv_reg_mr(e.pd, ptr, length, IBV_ACCESS_LOCAL_WRITE);
    require(mr, "host MR registration failed; check pinned allocation and memlock");
    try { e.hosts.push_back({ptr, length, mr, 0}); }
    catch (...) { ibv_dereg_mr(mr); throw; }
    bytes -= length;
    ptr += length;
  }
}

RemoteRegion Transport::export_region(void *base, size_t bytes) const {
  const auto address = reinterpret_cast<uintptr_t>(base);
  for (const auto &region : p_->devices) {
    const auto allocation = reinterpret_cast<uintptr_t>(region.ptr);
    if (contains(allocation, region.bytes, address, bytes))
      return {0, bytes, region.mapped + (address - allocation), region.mr->rkey};
  }
  throw std::invalid_argument("GDR: device view outside its owner MR");
}

const std::string &Transport::endpoint() const {
  auto &e=*p_;ibv_port_attr port{};
  require(ibv_query_port(e.ctx,e.port_num,&port)==0 && port.state==IBV_PORT_ACTIVE &&
      port.link_layer==e.port.link_layer && port.active_mtu==e.port.active_mtu &&
      port.lid==e.port.lid,"NIC port changed; restart endpoints");
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    ibv_gid current{};
    require(ibv_query_gid(e.ctx,e.port_num,e.gid_index,&current)==0 &&
        std::memcmp(current.raw,e.gid.raw,sizeof(current.raw))==0,
        "GID changed; restart endpoints");
  }
  return e.endpoint;
}

const std::string &Transport::nic() const { return p_->nic_selector; }
int Transport::read_depth() const { return p_->read_depth; }
size_t Transport::h2d_qp_count() const { return p_->h2d_qps; }
size_t Transport::d2h_qp_count() const { return p_->d2h_qps; }
int Transport::write_sge_limit() const { return p_->write_sge_limit; }
int Transport::read_sge_limit() const { return p_->read_sge_limit; }
bool Transport::uses_wr_api() const {
#ifdef FLEXKV_GDR_WR_API
  return !p_->queues.empty() &&
         std::all_of(p_->queues.begin(), p_->queues.end(),
                     [](const Impl::Queue &q) { return q.qp_ex != nullptr; });
#else
  return false;
#endif
}
std::vector<uint32_t> Transport::sender_qpns() const {
  std::vector<uint32_t> qpns;
  qpns.reserve(p_->queues.size());
  for (const auto &queue : p_->queues) qpns.push_back(queue.qp->qp_num);
  return qpns;
}
uint32_t Transport::accept(uint32_t sender) {
  require(!p_->sender, "unexpected target on sender route");
  auto *target = p_->create();
  try {
    p_->connect(target, sender);
    p_->targets.push_back(target);
  } catch (...) {
    ibv_destroy_qp(target);
    throw;
  }
  return target->qp_num;
}
void Transport::release(uint32_t qpn) {
  auto &targets = p_->targets;
  auto target = std::find_if(targets.begin(), targets.end(),
                             [qpn](const ibv_qp *qp) { return qp->qp_num == qpn; });
  require(target != targets.end(), "unknown target QP");
  require(ibv_destroy_qp(*target) == 0, "destroy target QP failed");
  targets.erase(target);
}

void Transport::connect(const std::vector<uint32_t> &targets, std::vector<RemoteRegion> regions) {
  auto &e = *p_;
  require(e.sender && !e.connected && !regions.empty() && !e.hosts.empty(),
          "invalid owner handshake");
  require(targets.size() == e.queues.size(), "Owner QP count does not match sender");
  for (const auto &region : regions) {
    require(region.local && region.remote && region.bytes &&
            region.bytes <= UINTPTR_MAX - region.local &&
            region.bytes <= UINT64_MAX - region.remote, "invalid owner view");
  }
  std::sort(e.hosts.begin(), e.hosts.end(), [](const Mr &a, const Mr &b) {
    return reinterpret_cast<uintptr_t>(a.ptr) < reinterpret_cast<uintptr_t>(b.ptr);
  });
  for (size_t i = 1; i < e.hosts.size(); ++i) {
    require(reinterpret_cast<uintptr_t>(e.hosts[i - 1].ptr) + e.hosts[i - 1].bytes <=
            reinterpret_cast<uintptr_t>(e.hosts[i].ptr), "overlapping Host registrations");
  }
  std::sort(regions.begin(), regions.end(), [](const RemoteRegion &a, const RemoteRegion &b) {
    return a.local < b.local || (a.local == b.local && a.bytes > b.bytes);
  });
  std::vector<RemoteRegion> merged;
  for (const auto &region : regions) {
    if (!merged.empty() && region.local < merged.back().local + merged.back().bytes) {
      auto &prev = merged.back();
      require(prev.key == region.key && prev.remote + (region.local - prev.local) == region.remote,
              "conflicting imported device views");
      prev.bytes = std::max(prev.bytes, region.local - prev.local + region.bytes);
    } else {
      merged.push_back(region);
    }
  }
  for (size_t i = 0; i < targets.size(); ++i) e.connect(e.queues[i].qp, targets[i], true);
  e.remote_regions = std::move(merged);
  e.connected = true;
}

size_t Transport::resolve(uintptr_t host, uintptr_t device, size_t bytes, bool h2d, Span &out) {
  auto &e = *p_;
  require(e.connected && bytes, "invalid transfer span");
  auto host_base = [](const Mr &region) { return reinterpret_cast<uintptr_t>(region.ptr); };
  const auto &hm = e.lookup(e.hosts, e.host_hint, host, host_base);
  const auto &gm = e.lookup(e.remote_regions, e.remote_hint, device,
                            [](const RemoteRegion &region) { return region.local; });
  const auto device_offset = device - gm.local;
  size_t length = std::min({bytes, hm.bytes - (host - host_base(hm)),
      gm.bytes - device_offset, max_payload()});
  const auto remote = gm.remote + device_offset;
  size_t index = 0;
  const size_t active_qps = h2d ? e.h2d_qps : e.d2h_qps;
  if (active_qps > 1) {
    // Assign by destination bytes. Overlapping writes must reach the same
    // SQ in request order, even when a later span starts at a different offset.
    constexpr size_t stripe_bytes = 64u << 10;
    const uint64_t destination = h2d ? remote : host;
    const auto stripe = destination / stripe_bytes;
    index = (stripe ^ (stripe >> 4) ^ (stripe >> 8)) % active_qps;
    length = std::min(length, static_cast<size_t>(stripe_bytes - destination % stripe_bytes));
  }
  out = {host, remote, static_cast<uint32_t>(length), hm.mr->lkey, gm.key,
         static_cast<uint32_t>(index)};
  return length;
}

size_t Transport::resolve_chunk(uintptr_t host, uintptr_t device, size_t bytes,
                                uint64_t route_key, Span &out, bool h2d) {
  auto &e = *p_;
  require(e.connected && bytes, "invalid transfer span");
  auto host_base = [](const Mr &region) { return reinterpret_cast<uintptr_t>(region.ptr); };
  const auto &hm = e.lookup(e.hosts, e.host_hint, host, host_base);
  const auto &gm = e.lookup(e.remote_regions, e.remote_hint, device,
                            [](const RemoteRegion &region) { return region.local; });
  const auto device_offset = device - gm.local;
  const size_t length = std::min({bytes, hm.bytes - (host - host_base(hm)),
      gm.bytes - device_offset, max_payload()});
  const auto remote = gm.remote + device_offset;
  size_t index = 0;
  const size_t active_qps = h2d ? e.h2d_qps : e.d2h_qps;
  if (active_qps > 1) {
    // Route by destination block/layer key so chunks stay unsplit across QPs
    // while overlapping writes to the same destination still share one SQ.
    const uint64_t mixed = route_key + (route_key >> 16) + (route_key >> 32);
    index = mixed % active_qps;
  }
  out = {host, remote, static_cast<uint32_t>(length), hm.mr->lkey, gm.key,
         static_cast<uint32_t>(index)};
  return length;
}

size_t Transport::post_payload(const Span *spans, size_t count, bool h2d) {
  auto &e = *p_;
  require(!e.unsafe && e.connected, "route not available");
  if (!h2d && e.read_depth >= 4 && count >= 4) {
    bool all_busy = true;
    for (size_t i = 0; i < e.d2h_qps; ++i) {
      if (e.queues[i].reads == 0) {
        all_busy = false;
        break;
      }
    }
    const int refill_batch = std::min(8, std::max(2, e.read_depth / 2));
    if (all_busy && e.queues[spans[0].qp_index].reads > e.read_depth - refill_batch) {
      return 0;
    }
  }
  const int sge_limit = h2d ? e.write_sge_limit : e.read_sge_limit;
  const size_t payload_limit = max_payload();
  std::array<size_t, kMaxQps> counts{};
  std::array<size_t, kMaxQps> sge_counts{};
  std::array<size_t, kMaxQps> bytes{};
  size_t actual = 0;
  for (; actual < count; ++actual) {
    const auto &span = spans[actual];
    const auto index = span.qp_index;
    require(index < e.queues.size() && span.bytes > 0 && span.bytes <= payload_limit,
            "invalid payload span");
    auto &queue = e.queues[index];
    if (queue.payload_bytes + bytes[index] + span.bytes > (16u << 20)) break;
    auto &slot = counts[index];
    if (slot > 0) {
      auto &tail_wr = queue.wrs[slot - 1];
      const uint64_t tail_seq = queue.posted + slot;
      auto &tail_op = queue.ledger[tail_seq % e.depth];
      if (tail_wr.wr.rdma.rkey == span.rkey &&
          tail_wr.wr.rdma.remote_addr + tail_op.bytes == span.remote &&
          span.bytes <= payload_limit - tail_op.bytes) {
        auto &last_sge = queue.sges[sge_counts[index] - 1];
        if (last_sge.lkey == span.lkey && last_sge.addr + last_sge.length == span.host) {
          last_sge.length += span.bytes;
          tail_op.bytes += span.bytes;
          bytes[index] += span.bytes;
          if (h2d && e.fence_depth) queue.last_write = span;
          continue;
        }
        if (tail_wr.num_sge < sge_limit) {
          queue.sges[sge_counts[index]++] = {span.host, span.bytes, span.lkey};
          ++tail_wr.num_sge;
          tail_op.bytes += span.bytes;
          bytes[index] += span.bytes;
          if (h2d && e.fence_depth) queue.last_write = span;
          continue;
        }
      }
    }
    const size_t reserved = h2d ? e.fence_depth - queue.fences : 0;
    const auto inflight = queue.posted - queue.retired;
    if (inflight + slot >= e.depth - reserved ||
        (!h2d && slot >= static_cast<size_t>(e.read_depth - queue.reads))) break;
    const uint64_t seq = queue.posted + slot + 1;
    auto *sg_start = &queue.sges[sge_counts[index]];
    queue.sges[sge_counts[index]++] = {span.host, span.bytes, span.lkey};
    auto &wr = queue.wrs[slot];
    wr = {};
    wr.wr_id = e.wr_id(seq, index);
    wr.sg_list = sg_start;
    wr.num_sge = 1;
    wr.opcode = h2d ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    wr.send_flags = h2d ? 0 : IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = span.remote;
    wr.wr.rdma.rkey = span.rkey;
    if (slot) queue.wrs[slot - 1].next = &wr;
    queue.ledger[seq % e.depth] = {seq, !h2d, span.bytes, -1, !h2d};
    bytes[index] += span.bytes;
    ++slot;
    if (h2d && e.fence_depth) queue.last_write = span;
  }
  for (size_t i = 0; i < e.queues.size(); ++i) {
    const auto n = counts[i];
    if (!n) continue;
    auto &queue = e.queues[i];
    // A tail CQE retires only the earlier WRs on this SQ.
    if (h2d) {
      queue.wrs[n - 1].send_flags = IBV_SEND_SIGNALED;
      queue.ledger[(queue.posted + n) % e.depth].signaled = true;
    }
    e.post(queue, n);
    queue.payload_bytes += bytes[i];
    if (!h2d) queue.reads += static_cast<int>(n);
    queue.needs_readback |= h2d && e.fence_depth;
    e.stats.payload_wrs += n;
    e.stats.qp_payload_wrs[i] += n;
    (h2d ? e.stats.h2d_bytes : e.stats.d2h_bytes) += bytes[i];
  }
  return actual;
}

bool Transport::finish_scope(TransferCheckpoint &checkpoint) {
  auto &e = *p_;
  bool ready = true;
  for (size_t i = 0; i < e.queues.size(); ++i) {
    auto &queue = e.queues[i];
    if (queue.needs_readback && e.post_readback(i)) queue.needs_readback = false;
    ready &= !queue.needs_readback;
    checkpoint[i] = queue.posted;
  }
  return ready;
}

bool Transport::is_complete(const TransferCheckpoint &checkpoint) const {
  const auto &queues = p_->queues;
  for (size_t i = 0; i < queues.size(); ++i)
    if (queues[i].retired < checkpoint[i]) return false;
  return true;
}

int Transport::poll() {
  auto &e = *p_;
  if (!has_outstanding()) return 0;
  ibv_wc wc[64];
  const int count = ibv_poll_cq(e.cq, 64, wc);
  if (count < 0) e.fail("poll_cq failed");
  for (int i = 0; i < count; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS)
      e.fail("CQE failed, WC status=" + std::to_string(wc[i].status));
    const auto index = wc[i].wr_id & (kMaxQps - 1);
    const auto seq = wc[i].wr_id >> Impl::kQpIndexBits;
    if (index >= e.queues.size() || wc[i].qp_num != e.queues[index].qp->qp_num)
      e.fail("invalid CQE QP");
    auto &queue = e.queues[index];
    if (seq <= queue.retired || seq > queue.posted || !queue.ledger[seq % e.depth].signaled)
      e.fail("invalid CQE sequence");
    while (queue.retired < seq) {
      const auto next = queue.retired + 1;
      const auto &op = queue.ledger[next % e.depth];
      if (op.sequence != next || op.signaled != (next == seq))
        e.fail("invalid CQE prefix");
      queue.payload_bytes -= op.bytes;
      if (op.read) --queue.reads;
      if (op.scratch >= 0) {
        queue.scratch_used[op.scratch] = false;
        --queue.fences;
      }
      queue.retired = next;
    }
  }
  e.stats.completions += count;
  return count;
}

bool Transport::has_outstanding() const {
  for (const auto &queue : p_->queues)
    if (queue.posted != queue.retired) return true;
  return false;
}

void Transport::check_progress() {
  auto &e = *p_;
  const auto now = CqWatchdog::Clock::now();
  for (auto &queue : e.queues) {
    if (queue.watchdog.expired(now, queue.posted != queue.retired,
                               queue.retired != queue.last_checked))
      e.fail("timed out waiting for completions on QP " + std::to_string(queue.qp->qp_num));
    queue.last_checked = queue.retired;
  }
}
size_t Transport::max_payload() const { return std::min(size_t(p_->port.max_msg_sz), size_t(1u << 20)); }
Stats Transport::stats() const { return p_->stats; }
void Transport::mark_unsafe() { p_->unsafe = true; }

void Transport::make_device_visible(int device) {
  require(cudaSetDevice(device) == cudaSuccess, "cannot select consumer device");
#if defined(FLEXKV_GDR_MAPPED)
  require(cudaDeviceSynchronize() == cudaSuccess, "mapped consumer device sync failed");
#elif CUDART_VERSION >= 11030
  int ordering = 0, options = 0;
  require(cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device) == cudaSuccess &&
          cudaDeviceGetAttribute(&options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device) == cudaSuccess,
          "cannot query consumer device visibility");
  if (ordering < cudaFlushGPUDirectRDMAWritesToAllDevices) {
    require(options & cudaFlushGPUDirectRDMAWritesOptionHost, "host-triggered flush is unavailable");
    require(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
            cudaFlushGPUDirectRDMAWritesToAllDevices) == cudaSuccess, "consumer visibility flush failed");
  }
#else
  throw std::runtime_error("GDR native H2D requires CUDA >= 11.3");
#endif
}
}  // namespace gdr
}  // namespace flexkv
