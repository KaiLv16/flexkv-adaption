// SPDX-License-Identifier: Apache-2.0
#include "gdr/region_adapter.h"

#include "device_thread_pool.h"
#include "region_batch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace flexkv {
namespace gdr {
namespace {

constexpr size_t kWindow = 16;

struct BlockAddress {
  const GTensorHandler &handler;
  int layer;
  int kv;

  template <BackendType Type>
  uintptr_t operator()(TensorKindTag<Type>) const {
    return reinterpret_cast<uintptr_t>(ptr_at<Type>(handler, layer, kv, 0));
  }
};

// A prepared region owns no memory. RegionBatch keeps the torch objects
// alive until the combined executors return, including the error path.
class SpanCursor {
public:
  void reset(const RegionWork &work) {
    work_ = &work;
    layer_ = 0;
    kv_ = 0;
    block_ = 0;
    offset_ = 0;
  }

  bool empty() const {
    return work_->args.num_blocks == 0 || layer_ == work_->args.num_layers;
  }

  Span next(Transport &transport) {
    const auto &a = work_->args;
    const auto &gpu = a.gpu_tensor_handler;
    const auto layer = a.start_layer_id + layer_;
    const auto host_block = a.cpu_block_ids[block_];
    const auto device_block = a.gpu_block_ids[block_];
    if (host_block < 0 || host_block >= work_->cpu_num_blocks ||
        device_block < 0 || device_block >= work_->gpu_num_blocks)
      throw std::invalid_argument("GDR: block id outside its pool");
    auto host = add(reinterpret_cast<uintptr_t>(a.cpu_ptr), host_block,
                    a.cpu_block_stride_in_bytes);
    host = add(host, layer, a.cpu_layer_stride_in_bytes);
    host = add(host, kv_, a.cpu_kv_stride_in_bytes);
    host = add(host, 1, a.cpu_startoff_inside_chunks);
    auto device = with_tensor_kind(a.tensor_kind, BlockAddress{gpu, layer, kv_});
    if (!device) throw std::invalid_argument("GDR: missing device backing");
    device = add(device, device_block, a.gpu_block_stride_in_bytes);
    device = add(device, 1, a.gpu_startoff_inside_chunks);
    Span span;
    offset_ += transport.resolve(add(host, 1, offset_), add(device, 1, offset_),
                                 a.chunk_size_in_bytes - offset_, a.is_host_to_device, span);
    if (offset_ == static_cast<size_t>(a.chunk_size_in_bytes)) {
      offset_ = 0;
      if (++block_ == a.num_blocks) {
        block_ = 0;
        if (++kv_ == a.kv_dim) {
          kv_ = 0;
          ++layer_;
        }
      }
    }
    return span;
  }

private:
  static uintptr_t add(uintptr_t base, int64_t count, int64_t stride) {
    if (count < 0 || stride < 0 ||
        (stride && static_cast<uint64_t>(count) > (UINTPTR_MAX - base) / stride))
      throw std::invalid_argument("GDR: invalid block or byte offset");
    return base + static_cast<uint64_t>(count) * stride;
  }

  const RegionWork *work_ = nullptr;
  int layer_ = 0, kv_ = 0, block_ = 0;
  size_t offset_ = 0;
};

struct Checkpoint {
  TransferCheckpoint tails;
  int milestone;
  bool needs_visibility;
  bool sent;
};

bool send_command(int fd, char command) {
  ssize_t result;
  do {
    result = send(fd, &command, 1, MSG_DONTWAIT | MSG_NOSIGNAL);
  } while (result < 0 && errno == EINTR);
  if (result == 1) return true;
  if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return false;
  throw UnsafeTransfer("GDR: owner command channel failed");
}

bool receive_ack(int fd) {
  char response = 0;
  ssize_t result;
  do {
    result = recv(fd, &response, 1, MSG_DONTWAIT);
  } while (result < 0 && errno == EINTR);
  if (result == 1 && response == 'A') return true;
  if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return false;
  throw UnsafeTransfer("GDR: owner did not acknowledge ordering");
}

}  // namespace

struct RegionAdapter::Route {
  std::unique_ptr<Transport> sender;
  int control_fd = -1;

  ~Route() { if (control_fd >= 0) close(control_fd); }

  void run(const std::vector<RegionWork> &work, bool h2d,
           const std::atomic<bool> &stop, const std::atomic<bool> &stopped,
           const std::function<void(int)> &rank_visible) {
    if (work.empty()) return;
    auto &transport = *sender;
    bool producer_sent = false;
    bool producer_ready = h2d;
    size_t region = 0;
    SpanCursor cursor;
    cursor.reset(work.front());
    std::array<Span, kWindow> spans;
    size_t first = 0, last = 0;
    std::array<Checkpoint, kWindow> checkpoints;
    size_t head = 0, tail = 0, next_visibility = 0;
    bool had_payload = false;
    unsigned idle = 0;
    int backoff_us = 10;

    while (region < work.size() || head != tail) {
      if (stop.load(std::memory_order_acquire) || stopped.load(std::memory_order_acquire))
        throw UnsafeTransfer("GDR: local transfer stopped");
      bool progress = transport.poll() > 0;
      if (!producer_ready) {
        if (!producer_sent) {
          producer_sent = send_command(control_fd, 'R');
          progress |= producer_sent;
        }
        if (producer_sent) producer_ready = receive_ack(control_fd);
        progress |= producer_ready;
      }

      // Dispatch ordering to the Owner once the applicable transport evidence
      // exists. A slow ACK retains only a checkpoint, never a CQ or SQ credit.
      while (next_visibility < tail) {
        auto &done = checkpoints[next_visibility % kWindow];
        if (!transport.is_complete(done.tails)) break;
        if (done.needs_visibility && !send_command(control_fd, 'V')) break;
        done.sent = true;
        ++next_visibility;
        progress = true;
      }
      while (head < tail) {
        const auto &done = checkpoints[head % kWindow];
        if (!done.sent || (done.needs_visibility && !receive_ack(control_fd))) break;
        if (done.milestone >= 0) rank_visible(done.milestone);
        ++head;
        progress = true;
      }

      if (producer_ready && region < work.size() && tail - head < kWindow) {
        const auto layer = work[region].milestone;
        if (first == last) {
          first = last = 0;
          for (size_t planned = 0; planned < kWindow && !cursor.empty(); ++planned) {
            const auto next = cursor.next(transport);
            if (last > 0) {
              auto &prev = spans[last - 1];
              if (prev.qp_index == next.qp_index && prev.lkey == next.lkey && prev.rkey == next.rkey &&
                  prev.host + prev.bytes == next.host && prev.remote + prev.bytes == next.remote &&
                  next.bytes <= transport.max_payload() - prev.bytes) {
                prev.bytes += next.bytes;
                continue;
              }
            }
            spans[last++] = next;
          }
        }
        if (first != last) {
          const auto sent = transport.post_payload(spans.data() + first, last - first, h2d);
          first += sent;
          if (sent) {
            had_payload = true;
            progress = true;
          }
        }
        if (first == last && cursor.empty()) {
          const bool scope_end = region + 1 == work.size() || work[region + 1].milestone != layer;
          auto &checkpoint = checkpoints[tail % kWindow];
          if (!scope_end || transport.finish_scope(checkpoint.tails)) {
            if (scope_end) {
              checkpoint.milestone = layer;
              checkpoint.needs_visibility = h2d && had_payload;
              checkpoint.sent = false;
              ++tail;
              had_payload = false;
            }
            if (++region < work.size()) cursor.reset(work[region]);
            progress = true;
          }
        }
      }

      const bool outstanding = transport.has_outstanding();
      transport.check_progress();
      if (progress) {
        idle = 0;
        backoff_us = 10;
      } else if (++idle >= 64) {
        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
        backoff_us = std::min(backoff_us * 2, outstanding ? 50 : 1000);
      }
    }
  }
};

RegionAdapter::RegionAdapter(int ranks) : routes_(ranks) {}
RegionAdapter::~RegionAdapter() = default;

void RegionAdapter::install(int rank, std::unique_ptr<Transport> sender, int control_fd) {
  auto &route = routes_.at(rank);
  if (route) throw std::logic_error("GDR route already installed");
  std::unique_ptr<Route> candidate(new Route);
  candidate->control_fd = fcntl(control_fd, F_DUPFD_CLOEXEC, 0);
  if (candidate->control_fd < 0) throw std::runtime_error("GDR: cannot retain Owner channel");
  candidate->sender = std::move(sender);
  route = std::move(candidate);
}

bool RegionAdapter::ready() const {
  return !unsafe_ && std::all_of(routes_.begin(), routes_.end(),
                                [](const std::unique_ptr<Route> &route) { return bool(route); });
}

void RegionAdapter::run(DeviceThreadPool &threads,
                        const std::vector<std::vector<RegionWork>> &ranks,
                        bool h2d, const std::atomic<bool> &stopped,
                        const std::function<void(int)> &rank_visible) {
  if (!ready()) throw std::logic_error("GDR: rank group is not ready");
  std::atomic<bool> stop{false};
  std::vector<std::exception_ptr> failures(ranks.size());
  std::vector<std::future<void>> futures;
  futures.reserve(ranks.size());
  std::exception_ptr dispatch_failure;
  try {
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
      if (ranks[rank].empty()) continue;
      futures.push_back(threads.enqueue(rank, [&, rank] {
        try {
          routes_[rank]->run(ranks[rank], h2d, stop, stopped, rank_visible);
        } catch (...) {
          failures[rank] = std::current_exception();
          stop.store(true, std::memory_order_release);
        }
      }));
    }
  } catch (...) {
    dispatch_failure = std::current_exception();
    stop.store(true, std::memory_order_release);
  }
  for (auto &future : futures) future.get();

  if (!stop.load(std::memory_order_acquire)) return;
  unsafe_ = true;
  for (auto &route : routes_) route->sender->mark_unsafe();
  try {
    if (dispatch_failure) std::rethrow_exception(dispatch_failure);
    for (const auto &failure : failures) {
      if (failure) std::rethrow_exception(failure);
    }
  } catch (const std::exception &e) {
    throw UnsafeTransfer(e.what());
  }
  throw UnsafeTransfer("GDR: rank dispatch failed");
}

std::vector<Stats> RegionAdapter::statistics() const {
  std::vector<Stats> stats;
  for (const auto &route : routes_) {
    if (route) stats.push_back(route->sender->stats());
  }
  return stats;
}

}  // namespace gdr

void RegionBatchGroup::install_gdr_route(int rank, std::unique_ptr<gdr::Transport> sender,
                                         int fd) {
  if (!gdr_) gdr_.reset(new gdr::RegionAdapter(num_gpus()));
  gdr_->install(rank, std::move(sender), fd);
}

void RegionBatchGroup::close_gdr() {
  if (gdr_ && gdr_->unsafe()) throw gdr::UnsafeTransfer("GDR session remains quarantined");
  gdr_.reset();
}

std::vector<std::vector<uint64_t>> RegionBatchGroup::gdr_counters() const {
  std::vector<std::vector<uint64_t>> counters;
  if (gdr_) {
    for (const auto &route : gdr_->statistics()) {
      counters.push_back({route.h2d_bytes, route.d2h_bytes, route.payload_wrs,
                          route.posts, route.completions, route.readbacks});
      auto &row = counters.back();
      row.insert(row.end(), route.qp_payload_wrs.begin(), route.qp_payload_wrs.end());
    }
  }
  return counters;
}

bool RegionBatchGroup::try_gdr(const std::vector<RegionRequest> &requests,
                             const std::vector<int> *empty_layers, int counter_id) {
  if (gdr_stopped_.load(std::memory_order_acquire))
    throw gdr::UnsafeTransfer("GDR session is stopped");
  if (!gdr_ || !gdr_->ready() || requests.empty()) return false;
  const bool h2d = requests.front().is_host_to_device;
  const auto &table = notifier_.table();
  const int layers = empty_layers ? table.num_layers() : 0;
  if (empty_layers && (!h2d || !table.enabled() || counter_id < 0 ||
                       counter_id >= table.num_counters())) {
    throw std::invalid_argument("GDR: invalid layerwise consumer");
  }

  std::vector<const RegionRequest *> order;
  order.reserve(requests.size());
  for (const auto &request : requests) {
    if (request.backend != TransferBackendKind::AUTO ||
        request.is_host_to_device != h2d) return false;
    for (const auto *ids : {&request.cpu_block_id_tensor, &request.gpu_block_id_tensor}) {
      if (!ids->device().is_cpu() || ids->scalar_type() != torch::kInt64 ||
          ids->dim() != 1 || !ids->is_contiguous()) {
        throw std::invalid_argument("GDR: block ids must be contiguous CPU int64 values");
      }
    }
    order.push_back(&request);
  }
  if (empty_layers) {
    std::stable_sort(order.begin(), order.end(), [](const RegionRequest *a, const RegionRequest *b) {
      return a->milestone_layer < b->milestone_layer;
    });
  }
  int first_layer = -1;
  for (auto *req : order) {
    if (req->milestone_layer >= 0) {
      first_layer = req->milestone_layer;
      break;
    }
  }
  if (empty_layers && first_layer < 0) {
    throw std::invalid_argument("GDR: layerwise work has no destination milestone");
  }

  std::vector<std::vector<gdr::RegionWork>> work(num_gpus());
  for (auto &rank : work) rank.reserve(order.size());
  std::vector<std::atomic<int>> outstanding(layers);
  std::vector<bool> declared(layers, false);
  for (auto &count : outstanding) count.store(0, std::memory_order_relaxed);
  for (auto *req : order) {
    const int layer = !empty_layers ? -1 :
        (req->milestone_layer >= 0 ? req->milestone_layer : first_layer);
    if (empty_layers) {
      if (layer >= layers) throw std::invalid_argument("GDR: invalid milestone layer");
      declared[layer] = true;
    }
    for (int rank = 0; rank < num_gpus(); ++rank) {
      if (!rank_participates(*req, rank)) continue;
      auto args = build_args(*req, rank);
      const auto &geometry = regions_[req->region_index].desc;
      if (args.num_blocks < 0 || args.num_layers <= 0 || args.kv_dim <= 0 ||
          args.chunk_size_in_bytes <= 0 || geometry.cpu_num_blocks <= 0 ||
          geometry.gpu_num_blocks.size() != static_cast<size_t>(num_gpus()) ||
          geometry.gpu_num_blocks[rank] <= 0)
        throw std::invalid_argument("GDR: invalid region geometry");
      auto &items = work[rank];
      if (layer >= 0 && (items.empty() || items.back().milestone != layer))
        outstanding[layer].fetch_add(1, std::memory_order_relaxed);
      items.push_back({args, layer, geometry.cpu_num_blocks, geometry.gpu_num_blocks[rank]});
    }
  }

  if (empty_layers) {
    for (int layer : *empty_layers) {
      if (layer < 0 || layer >= layers || declared[layer])
        throw std::invalid_argument("GDR: duplicate or invalid empty milestone");
      declared[layer] = true;
    }
  }
  try {
    for (int layer = 0; layer < layers; ++layer) {
      if (declared[layer] && !outstanding[layer].load(std::memory_order_relaxed))
        table.post_checked(counter_id, layer);
    }
    gdr_->run(*pool_, work, h2d, gdr_stopped_, [&](int layer) {
      if (outstanding[layer].fetch_sub(1, std::memory_order_acq_rel) == 1 &&
          !gdr_stopped_.load(std::memory_order_acquire))
        table.post_checked(counter_id, layer);
    });
  } catch (const gdr::UnsafeTransfer &) {
    throw;
  } catch (const std::exception &e) {
    // An eventfd write can have succeeded for a subset of consumers.
    throw gdr::UnsafeTransfer(e.what());
  }
  return true;
}

}  // namespace flexkv
