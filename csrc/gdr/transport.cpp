// SPDX-License-Identifier: Apache-2.0
// Local RC endpoint setup is based on FlexKV PR #348.
#include "gdr/transport.h"
#include "gdr/topology.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
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

namespace flexkv::gdr {
namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(std::string("GDR: ") + message);
}
std::string canonical(const std::string &p) {
  std::error_code ec;
  auto s=std::filesystem::canonical(p,ec);
  return ec ? std::string() : s.string();
}
struct Mr { void *ptr; size_t bytes; ibv_mr *mr; uintptr_t mapped=0; };
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

}  // namespace

struct Transport::Impl {
  struct Operation {
    uint64_t sequence = 0;
    bool read = false;
    uint32_t bytes = 0;
    int scratch = -1;
    uint64_t fence = 0;
  };

  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  std::vector<ibv_qp *> targets;
  std::vector<Mr> hosts, devices;
  std::vector<RemoteRegion> remote_regions;
  ibv_port_attr port{};
  ibv_gid gid{};
  int gid_index = 0;
  int depth = 0;
  int read_depth = 0;
  uint64_t max_mr_size = 0;
  bool sender = false;
  bool connected = false;
  bool unsafe = false;
  std::string name, endpoint;
  size_t host_hint = 0;
  size_t remote_hint = 0;

  static constexpr size_t kPostBatch = 16;
  static constexpr size_t kScratchStride = 64;
  static constexpr size_t kMaxFences = 16;
  std::array<ibv_send_wr, kPostBatch> wrs{};
  std::array<ibv_sge, kPostBatch> sges{};
  std::vector<Operation> ledger;
  std::array<bool, kMaxFences> scratch_used{};
  alignas(kScratchStride) std::array<unsigned char, kMaxFences * kScratchStride> scratch{};
  ibv_mr *scratch_mr = nullptr;
  int fence_depth = 0;
  uint64_t posted = 0;
  uint64_t retired = 0;
  uint64_t fence_retired = 0;
  uint64_t fence_posted = 0;
  uint64_t payload_bytes = 0;
  int reads = 0;
  int fences = 0;
  Stats stats{};

  ~Impl() {
    if (qp) ibv_destroy_qp(qp);
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

  ibv_qp *create() {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = depth;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = attr.cap.max_recv_sge = 1;
    auto *result = ibv_create_qp(pd, &attr);
    require(result, "create_qp failed");
    if (attr.cap.max_send_wr < static_cast<unsigned int>(depth)) {
      ibv_destroy_qp(result);
      throw std::runtime_error("GDR: provider reduced SQ capacity");
    }
    return result;
  }

  void modify(ibv_qp *q,ibv_qp_attr &a,int mask,const char *stage) {
    const int rc=ibv_modify_qp(q,&a,mask);
    const int saved_errno=errno;
    if(!rc) return;
    std::ostringstream error;
    error << "GDR: QP " << stage << " failed: nic=" << name
          << " port=1 gid_index=" << gid_index << " qpn=" << q->qp_num
          << " peer=" << a.dest_qp_num << " mtu=" << port.active_mtu
          << " link_layer=" << int(port.link_layer) << " read_depth=" << read_depth
          << " rc=" << rc << " (" << std::strerror(rc>0?rc:saved_errno) << ")"
          << " endpoint=" << endpoint;
    throw std::runtime_error(error.str());
  }
  void init_qp(ibv_qp *q) {
    // Port/P_Key/GID select a shared fabric endpoint, not exclusive resources.
    // PSNs start at zero on each new RC connection; they are scoped to its QP.
    ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.port_num=1;
    a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    modify(q,a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS,"INIT");
  }
  void connect(ibv_qp *q,uint32_t peer,bool initialized=false) {
    require(peer>0 && peer<=0xffffff,"invalid peer QPN");
    if(!initialized) init_qp(q);
    ibv_qp_attr a{};a.qp_state=IBV_QPS_RTR;a.path_mtu=port.active_mtu;a.dest_qp_num=peer;
    a.max_dest_rd_atomic=read_depth;a.min_rnr_timer=12;a.ah_attr.port_num=1;
    if(port.link_layer==IBV_LINK_LAYER_ETHERNET) {
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

  void post(size_t count) {
    if (posted > std::numeric_limits<uint64_t>::max() - count)
      fail("WR sequence overflow");
    ibv_send_wr *bad = nullptr;
    const int error = ibv_post_send(qp, wrs.data(), &bad);
    if (error) {
      // A failed post may have accepted a prefix. No caller may destroy the
      // route, reuse memory, or replay until the session has been isolated.
      size_t accepted = 0;
      while (accepted < count && &wrs[accepted] != bad) ++accepted;
      posted += accepted;
      fail("post_send failed; possible accepted prefix=" + std::to_string(accepted) +
           ", error=" + std::to_string(error));
    }
    posted += count;
    ++stats.posts;
  }
};

Transport::Transport(int device, bool sender, const std::string &nic,
                     const std::string &required_endpoint, int requested_depth,
                     int requested_fences) : p_(std::make_unique<Impl>()) {
  auto &e = *p_;
  e.sender = sender;
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
  char bus[32]{};
  const auto pci_status=cudaDeviceGetPCIBusId(bus,sizeof(bus),device);
  std::string bdf=pci_status==cudaSuccess ? bus : "";
  if(pci_status!=cudaSuccess && !nic.empty()) cudaGetLastError();
  std::transform(bdf.begin(),bdf.end(),bdf.begin(),[](unsigned char c){return std::tolower(c);});
  // The endpoint fingerprint must match whether the owner selected automatically
  // or the worker received its explicit NIC through the owner handshake.
  if(selected_nic.empty()) {
    require(pci_status==cudaSuccess,
            "cannot query logical device PCI bus; set gdr.nic_names explicitly");
    const auto device_path=canonical("/sys/bus/pci/devices/"+bdf);
    require(!bdf.empty() && !device_path.empty(),"logical device PCI path missing from sysfs; cannot auto-select NIC");
    std::vector<std::pair<std::string,std::string>> candidates;
    for(int i=0;i<n;++i) {
      const std::string name=ibv_get_device_name(raw[i]);
      auto ctx=ibv_open_device(raw[i]);if(!ctx) continue;
      ibv_port_attr port{};
      const bool active=ibv_query_port(ctx,1,&port)==0 && port.state==IBV_PORT_ACTIVE;
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
  require(e.ctx && ibv_query_port(e.ctx,1,&e.port)==0 && e.port.state==IBV_PORT_ACTIVE,
          "selected NIC port 1 is unavailable; check gdr.nic_names and RDMA port state");
  struct Candidate {int priority,index;ibv_gid gid;};
  std::vector<Candidate> candidates;
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    for(int i=0;i<e.port.gid_tbl_len;++i) {
      std::ifstream f("/sys/class/infiniband/"+e.name+"/ports/1/gid_attrs/types/"+std::to_string(i));
      std::string type;std::getline(f,type);
      const bool v2=type.find("RoCE v2")!=std::string::npos;
      const bool v1=type.find("RoCE v1")!=std::string::npos;
      ibv_gid gid{};
      if((!v1 && !v2) || ibv_query_gid(e.ctx,1,i,&gid)) continue;
      bool nonzero=false;for(auto v:gid.raw) nonzero |= v!=0;
      const bool local=gid.raw[0]==0xfe && (gid.raw[1]&0xc0)==0x80;
      if(nonzero) candidates.push_back({v2?(local?1:0):2,i,gid});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b) {return a.priority<b.priority;});
  } else candidates.push_back({0,0,{}});
  require(!candidates.empty(),"no nonzero RoCE GID with a known v1/v2 type");

  ibv_device_attr attr{};
  require(ibv_query_device(e.ctx, &attr) == 0, "query_device failed");
  e.max_mr_size = attr.max_mr_size;
  e.depth = std::min({256, attr.max_qp_wr, attr.max_cqe});
  const int capacity = std::min({128, attr.max_qp_rd_atom, attr.max_qp_init_rd_atom});
  require(e.depth > 1 && capacity > 0 && e.port.max_msg_sz, "insufficient SQ/CQ or READ capacity");
  require(requested_depth >= 0 && requested_depth <= capacity, "invalid READ depth");
  e.read_depth = requested_depth ? requested_depth : std::min(capacity, 16);
#ifdef FLEXKV_GDR_MAPPED
  e.fence_depth = requested_fences;
  require(!sender || (e.fence_depth > 0 && e.fence_depth < e.depth &&
      e.fence_depth <= e.read_depth && e.fence_depth <= static_cast<int>(Impl::kMaxFences)),
      "invalid mapped read-back depth");
#else
  (void)requested_fences;
#endif
  e.pd = ibv_alloc_pd(e.ctx);
  require(e.pd, "alloc_pd failed");
  e.cq = ibv_create_cq(e.ctx, e.depth, nullptr, nullptr, 0);
  require(e.cq, "create_cq failed");
  e.ledger.resize(e.depth);
  if (sender && e.fence_depth) {
    e.scratch_mr = ibv_reg_mr(e.pd, e.scratch.data(), e.scratch.size(), IBV_ACCESS_LOCAL_WRITE);
    require(e.scratch_mr, "read-back scratch registration failed");
  }
  for (const auto &candidate : candidates) {
    e.gid_index = candidate.index;
    e.gid = candidate.gid;
    e.endpoint = bdf + "|" + e.name + "|1|" + std::to_string(e.gid_index) + "|" +
                 std::to_string(e.port.active_mtu) + "|" + std::to_string(e.port.lid);
    for (auto byte : e.gid.raw) e.endpoint += "|" + std::to_string(byte);
    if (!required_endpoint.empty() && required_endpoint != e.endpoint) continue;
    if (sender) {
      e.qp = e.create();
      e.init_qp(e.qp);
    }
    return;
  }
  throw std::runtime_error("GDR: no matching local RC endpoint on " + e.name);
}

// Until device-level isolation is established, preserve verbs resources after
// any uncertainty. The owning process must also retain the backing memory.
Transport::~Transport() {
  if (p_ && (p_->unsafe || p_->posted != p_->retired)) (void)p_.release();
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
    try { e.hosts.push_back({ptr, length, mr}); }
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
  require(ibv_query_port(e.ctx,1,&port)==0 && port.state==IBV_PORT_ACTIVE &&
      port.link_layer==e.port.link_layer && port.active_mtu==e.port.active_mtu &&
      port.lid==e.port.lid,"NIC port changed; restart endpoints");
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    ibv_gid current{};
    require(ibv_query_gid(e.ctx,1,e.gid_index,&current)==0 &&
        std::memcmp(current.raw,e.gid.raw,sizeof(current.raw))==0,
        "GID changed; restart endpoints");
  }
  return e.endpoint;
}

const std::string &Transport::nic() const { return p_->name; }
int Transport::read_depth() const { return p_->read_depth; }
uint32_t Transport::sender_qpn() const {
  require(p_->qp, "sender QP does not exist");
  return p_->qp->qp_num;
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
                             [qpn](auto *qp) { return qp->qp_num == qpn; });
  require(target != targets.end(), "unknown target QP");
  require(ibv_destroy_qp(*target) == 0, "destroy target QP failed");
  targets.erase(target);
}

void Transport::connect(uint32_t target, std::vector<RemoteRegion> regions) {
  auto &e = *p_;
  require(e.sender && !e.connected && !regions.empty() && !e.hosts.empty(),
          "invalid owner handshake");
  for (const auto &region : regions) {
    require(region.local && region.remote && region.bytes &&
            region.bytes <= UINTPTR_MAX - region.local &&
            region.bytes <= UINT64_MAX - region.remote, "invalid owner view");
  }
  std::sort(e.hosts.begin(), e.hosts.end(), [](const auto &a, const auto &b) {
    return reinterpret_cast<uintptr_t>(a.ptr) < reinterpret_cast<uintptr_t>(b.ptr);
  });
  for (size_t i = 1; i < e.hosts.size(); ++i) {
    require(reinterpret_cast<uintptr_t>(e.hosts[i - 1].ptr) + e.hosts[i - 1].bytes <=
            reinterpret_cast<uintptr_t>(e.hosts[i].ptr), "overlapping Host registrations");
  }
  std::sort(regions.begin(), regions.end(), [](const auto &a, const auto &b) {
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
  e.connect(e.qp, target, true);
  e.remote_regions = std::move(merged);
  e.connected = true;
}

size_t Transport::resolve(uintptr_t host, uintptr_t device, size_t bytes, Span &out) {
  auto &e = *p_;
  require(e.connected && bytes, "invalid transfer span");
  auto host_base = [](const Mr &region) { return reinterpret_cast<uintptr_t>(region.ptr); };
  const auto &hm = e.lookup(e.hosts, e.host_hint, host, host_base);
  const auto &gm = e.lookup(e.remote_regions, e.remote_hint, device,
                            [](const auto &region) { return region.local; });
  const auto device_offset = device - gm.local;
  const size_t length = std::min({bytes, hm.bytes - (host - host_base(hm)),
      gm.bytes - device_offset, max_payload()});
  out = {host, gm.remote + device_offset, static_cast<uint32_t>(length), hm.mr->lkey, gm.key};
  return length;
}

size_t Transport::post_payload(const Span *spans, size_t count, bool h2d) {
  auto &e = *p_;
  require(!e.unsafe && e.connected, "route not available");
  const size_t depth = static_cast<size_t>(e.depth - (h2d ? e.fence_depth : 0));
  const size_t payload_inflight = e.posted - e.retired - e.fences;
  count = std::min({count, Impl::kPostBatch, depth - payload_inflight,
      h2d ? Impl::kPostBatch : static_cast<size_t>(e.read_depth - e.reads)});
  size_t bytes = 0;
  size_t actual = 0;
  for (; actual < count; ++actual) {
    const auto &span = spans[actual];
    if (e.payload_bytes + bytes + span.bytes > (16u << 20)) break;
    const uint64_t seq = e.posted + actual + 1;
    e.sges[actual] = {span.host, span.bytes, span.lkey};
    auto &wr = e.wrs[actual];
    wr = {};
    wr.wr_id = seq;
    wr.sg_list = &e.sges[actual];
    wr.num_sge = 1;
    wr.opcode = h2d ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = span.remote;
    wr.wr.rdma.rkey = span.rkey;
    if (actual) e.wrs[actual - 1].next = &wr;
    e.ledger[seq % e.depth] = {seq, !h2d, span.bytes, -1, 0};
    bytes += span.bytes;
  }
  if (!actual) return 0;
  e.post(actual);
  e.payload_bytes += bytes;
  if (!h2d) e.reads += static_cast<int>(actual);
  e.stats.payload_wrs += actual;
  (h2d ? e.stats.h2d_bytes : e.stats.d2h_bytes) += bytes;
  return actual;
}

bool Transport::post_readback(const Span &last_write, uint64_t fence) {
  auto &e = *p_;
  require(!e.unsafe && e.connected && e.fence_depth && fence > e.fence_posted,
          "invalid read-back ordering checkpoint");
  if (e.fences == e.fence_depth || e.reads == e.read_depth) return false;
  const auto free = std::find(e.scratch_used.begin(), e.scratch_used.end(), false);
  const int slot = static_cast<int>(free - e.scratch_used.begin());
  const uint64_t seq = e.posted + 1;
  e.sges[0] = {reinterpret_cast<uint64_t>(e.scratch.data() + slot * Impl::kScratchStride),
               std::min(last_write.bytes, uint32_t(8)), e.scratch_mr->lkey};
  auto &wr = e.wrs[0];
  wr = {};
  wr.wr_id = seq;
  wr.sg_list = &e.sges[0];
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = last_write.remote;
  wr.wr.rdma.rkey = last_write.rkey;
  e.ledger[seq % e.depth] = {seq, true, 0, slot, fence};
  e.post(1);
  *free = true;
  ++e.reads;
  ++e.fences;
  e.fence_posted = fence;
  ++e.stats.readbacks;
  return true;
}

int Transport::poll() {
  auto &e = *p_;
  if (e.posted == e.retired) return 0;
  ibv_wc wc[32];
  const int count = ibv_poll_cq(e.cq, 32, wc);
  if (count < 0) e.fail("poll_cq failed");
  for (int i = 0; i < count; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS)
      e.fail("CQE failed, WC status=" + std::to_string(wc[i].status));
    const auto seq = wc[i].wr_id;
    if (seq != e.retired + 1 || seq > e.posted || e.ledger[seq % e.depth].sequence != seq)
      e.fail("invalid CQE sequence");
    const auto &op = e.ledger[seq % e.depth];
    e.payload_bytes -= op.bytes;
    if (op.read) --e.reads;
    if (op.scratch >= 0) {
      e.scratch_used[op.scratch] = false;
      --e.fences;
      e.fence_retired = op.fence;
    }
    e.retired = seq;
  }
  e.stats.completions += count;
  return count;
}

uint64_t Transport::posted() const { return p_->posted; }
uint64_t Transport::retired() const { return p_->retired; }
uint64_t Transport::fence_retired() const { return p_->fence_retired; }
bool Transport::requires_readback() const { return p_->fence_depth != 0; }
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
}  // namespace flexkv::gdr
