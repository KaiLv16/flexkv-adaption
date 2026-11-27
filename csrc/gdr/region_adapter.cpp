// SPDX-License-Identifier: Apache-2.0
#include "gdr/region_adapter.h"

#include "device_thread_pool.h"
#include "region_batch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace flexkv {
namespace gdr {
namespace {

constexpr size_t kCheckpointWindow = 64;
constexpr size_t kSpanBatch = 1024;
constexpr uint64_t kDefaultSpinBudgetNs = 150000ULL;

uint64_t spin_budget_ns() noexcept {
  if (const char *env = std::getenv("FLEXKV_GDR_SPIN_US")) {
    char *end = nullptr;
    const long us = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && us >= 0) {
      return static_cast<uint64_t>(us) * 1000ULL;
    }
  }
  return kDefaultSpinBudgetNs;
}

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#endif
}

unsigned choose_block_group_shift(const RegionWork &work) noexcept {
  const auto n = work.args.num_blocks;
  const auto *gpu_ids = work.args.gpu_block_ids;
  if (n < 2 || !gpu_ids) return 0;
  const int64_t pairs = std::min<int64_t>(n - 1, 256);
  int64_t contig = 0;
  for (int64_t i = 0; i < pairs; ++i) {
    if (gpu_ids[i] >= 0 && gpu_ids[i + 1] == gpu_ids[i] + 1) ++contig;
  }
  if (contig * 2 < pairs) return 0;
  if (n >= 64) return 3;
  if (n >= 16) return 2;
  return 1;
}

bool d2h_can_route_by_device(const std::vector<RegionWork> &work) {
  std::vector<uint64_t> seen;
  for (size_t i = 0; i < work.size(); ++i) {
    const auto &ai = work[i].args;
    if (ai.num_blocks <= 0) continue;
    const int64_t pool = work[i].cpu_num_blocks;
    if (pool <= 0 || pool > (1 << 20) || !ai.cpu_block_ids) return false;
    const bool same_ids =
        i > 0 && ai.cpu_block_ids == work[i - 1].args.cpu_block_ids &&
        ai.num_blocks == work[i - 1].args.num_blocks &&
        pool == work[i - 1].cpu_num_blocks;
    if (!same_ids) {
      seen.assign(static_cast<size_t>((pool + 63) / 64), 0ULL);
      for (int64_t b = 0; b < ai.num_blocks; ++b) {
        const int64_t hb = ai.cpu_block_ids[b];
        if (hb < 0 || hb >= pool) return false;
        const uint64_t bit = 1ULL << (hb & 63);
        uint64_t &word = seen[static_cast<size_t>(hb >> 6)];
        if (word & bit) return false;
        word |= bit;
      }
    }
    for (size_t j = 0; j < i; ++j) {
      const auto &aj = work[j].args;
      if (aj.num_blocks <= 0 || ai.cpu_ptr != aj.cpu_ptr) continue;
      if (ai.cpu_startoff_inside_chunks != aj.cpu_startoff_inside_chunks) {
        const int64_t i_off = ai.cpu_startoff_inside_chunks;
        const int64_t j_off = aj.cpu_startoff_inside_chunks;
        if (i_off + ai.chunk_size_in_bytes <= j_off ||
            j_off + aj.chunk_size_in_bytes <= i_off) {
          continue;
        }
      }
      const int i_end = ai.start_layer_id + ai.num_layers;
      const int j_end = aj.start_layer_id + aj.num_layers;
      if (ai.start_layer_id < j_end && aj.start_layer_id < i_end) {
        return false;
      }
    }
  }
  return true;
}

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
  void reset(const RegionWork &work, unsigned group_shift, bool route_by_device) {
    work_ = &work;
    group_shift_ = group_shift;
    route_by_device_ = route_by_device;
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
    const auto route_block =
        (a.is_host_to_device || route_by_device_) ? device_block : host_block;
    const uint64_t block_group = static_cast<uint64_t>(route_block) >> group_shift_;
    const uint64_t kind =
        (a.is_host_to_device ? a.gpu_startoff_inside_chunks : a.cpu_startoff_inside_chunks) != 0
            ? 1ULL
            : 0ULL;
    const uint64_t route_key = block_group + (block_group >> 2) +
                               static_cast<uint64_t>(layer) +
                               static_cast<uint64_t>(kv_) * 3ULL + kind;
    Span span;
    offset_ += transport.resolve_chunk(add(host, 1, offset_), add(device, 1, offset_),
                                       a.chunk_size_in_bytes - offset_, route_key, span,
                                       a.is_host_to_device);
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
  unsigned group_shift_ = 0;
  bool route_by_device_ = true;
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
  RouteProfile profile{};
  std::vector<Span> spans_buf{kSpanBatch};
  uint64_t fenced_d2h_epoch = 0;

  ~Route() { if (control_fd >= 0) close(control_fd); }

  static uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
    if (to <= from) return 0;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
  }

  void run(const std::vector<RegionWork> &work, bool h2d,
           const std::atomic<bool> &stop, const std::atomic<bool> &stopped,
           const std::function<void(int)> &rank_visible,
           bool profile_on, Clock::time_point dispatch_start,
           uint64_t d2h_epoch) {
    if (work.empty()) return;
    auto &acc = h2d ? profile.h2d : profile.d2h;
    Clock::time_point route_start{};
    if (profile_on) {
      route_start = Clock::now();
      ++acc.calls;
      if (dispatch_start != Clock::time_point{}) {
        acc.dispatch_ns += elapsed_ns(dispatch_start, route_start);
      }
    }
    auto &transport = *sender;
    const bool route_by_device = h2d || d2h_can_route_by_device(work);
    const unsigned group_shift =
        route_by_device ? choose_block_group_shift(work.front()) : 0U;
    bool producer_sent = false;
    bool producer_ready = h2d || (d2h_epoch != 0 && fenced_d2h_epoch == d2h_epoch);
    size_t region = 0;
    SpanCursor cursor;
    cursor.reset(work.front(), group_shift, route_by_device);
    if (spans_buf.size() < kSpanBatch) spans_buf.resize(kSpanBatch);
    Span *const spans = spans_buf.data();
    size_t first = 0, last = 0;
    std::array<Checkpoint, kCheckpointWindow> checkpoints;
    size_t head = 0, tail = 0, next_visibility = 0;
    bool had_payload = false;
    bool first_milestone_seen = false;
    int owner_inflight = 0;
    Clock::time_point owner_wait_start{};
    unsigned idle = 0;
    Clock::time_point idle_since{};
    int backoff_us = 10;
    const uint64_t spin_budget = spin_budget_ns();

    while (region < work.size() || head != tail) {
      if (stop.load(std::memory_order_acquire) || stopped.load(std::memory_order_acquire))
        throw UnsafeTransfer("GDR: local transfer stopped");
      bool progress = false;
      if (profile_on) {
        const auto t0 = Clock::now();
        progress = transport.poll() > 0;
        acc.poll_ns += elapsed_ns(t0, Clock::now());
      } else {
        progress = transport.poll() > 0;
      }
      if (!producer_ready) {
        if (!producer_sent) {
          producer_sent = send_command(control_fd, 'R');
          if (producer_sent && profile_on) {
            if (owner_inflight++ == 0) owner_wait_start = Clock::now();
          }
          progress |= producer_sent;
        }
        if (producer_sent) {
          producer_ready = receive_ack(control_fd);
          if (producer_ready) {
            if (d2h_epoch != 0) fenced_d2h_epoch = d2h_epoch;
            if (profile_on && owner_inflight > 0) {
              if (--owner_inflight == 0) acc.owner_wait_ns += elapsed_ns(owner_wait_start, Clock::now());
            }
          }
        }
        progress |= producer_ready;
      }

      // Dispatch ordering to the Owner once the applicable transport evidence
      // exists. A slow ACK retains only a checkpoint, never a CQ or SQ credit.
      while (next_visibility < tail) {
        auto &done = checkpoints[next_visibility % kCheckpointWindow];
        if (!transport.is_complete(done.tails)) break;
        if (done.needs_visibility) {
          if (!send_command(control_fd, 'V')) break;
          if (profile_on && owner_inflight++ == 0) owner_wait_start = Clock::now();
        }
        done.sent = true;
        ++next_visibility;
        progress = true;
      }
      while (head < tail) {
        const auto &done = checkpoints[head % kCheckpointWindow];
        if (!done.sent) break;
        if (done.needs_visibility) {
          if (!receive_ack(control_fd)) break;
          if (profile_on && owner_inflight > 0) {
            if (--owner_inflight == 0) acc.owner_wait_ns += elapsed_ns(owner_wait_start, Clock::now());
          }
        }
        if (done.milestone >= 0) {
          if (profile_on) {
            const auto t0 = Clock::now();
            if (!first_milestone_seen) {
              first_milestone_seen = true;
              acc.first_milestone_ns += elapsed_ns(route_start, t0);
            }
            rank_visible(done.milestone);
            acc.notify_ns += elapsed_ns(t0, Clock::now());
          } else {
            rank_visible(done.milestone);
          }
        }
        ++head;
        progress = true;
      }

      if (producer_ready && region < work.size() && tail - head < kCheckpointWindow) {
        const auto post_t0 = profile_on ? Clock::now() : Clock::time_point{};
        if (first == last) {
          first = last = 0;
          while (last < kSpanBatch && region < work.size()) {
            while (last < kSpanBatch && !cursor.empty()) {
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
            if (!cursor.empty()) break;
            const bool scope_end =
                region + 1 == work.size() || work[region + 1].milestone != work[region].milestone;
            if (scope_end) break;
            ++region;
            cursor.reset(work[region], group_shift, route_by_device);
          }
        }
        if (first != last) {
          const auto sent = transport.post_payload(spans + first, last - first, h2d);
          first += sent;
          if (sent) {
            had_payload = true;
            progress = true;
          }
        }
        if (first == last && cursor.empty() && region < work.size()) {
          const auto layer = work[region].milestone;
          const bool scope_end = region + 1 == work.size() || work[region + 1].milestone != layer;
          auto &checkpoint = checkpoints[tail % kCheckpointWindow];
          if (!scope_end || transport.finish_scope(checkpoint.tails)) {
            if (scope_end) {
              checkpoint.milestone = layer;
              checkpoint.needs_visibility = h2d && had_payload;
              checkpoint.sent = false;
              ++tail;
              had_payload = false;
            }
            if (++region < work.size()) cursor.reset(work[region], group_shift, route_by_device);
            progress = true;
          }
        }
        if (profile_on) acc.post_ns += elapsed_ns(post_t0, Clock::now());
      }

      const bool waiting = transport.has_outstanding() || tail != head || !producer_ready;
      transport.check_progress();
      if (progress) {
        idle = 0;
        idle_since = Clock::time_point{};
        backoff_us = 10;
      } else if (waiting) {
        ++idle;
        cpu_relax();
        bool spin_expired = false;
        if ((idle & 15U) == 0U) {
          const auto now = Clock::now();
          if (idle_since == Clock::time_point{}) {
            idle_since = now;
          } else if (elapsed_ns(idle_since, now) >= spin_budget) {
            spin_expired = true;
          }
        }
        if (spin_expired) {
          if (profile_on) {
            const auto t0 = Clock::now();
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
            acc.sleep_ns += elapsed_ns(t0, Clock::now());
            ++acc.sleep_count;
          } else {
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
          }
          backoff_us = std::min(backoff_us * 2, 50);
        }
      } else if (++idle >= 64) {
        if (profile_on) {
          const auto t0 = Clock::now();
          std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
          acc.sleep_ns += elapsed_ns(t0, Clock::now());
          ++acc.sleep_count;
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
        }
        backoff_us = std::min(backoff_us * 2, 1000);
      }
    }
    if (profile_on) {
      acc.route_total_ns += elapsed_ns(route_start, Clock::now());
    }
  }
};

RegionAdapter::RegionAdapter(int ranks) : routes_(ranks) {
  const char *env = std::getenv("FLEXKV_GDR_PROFILE");
  profile_ = env && std::strcmp(env, "1") == 0;
  const char *fence_env = std::getenv("FLEXKV_GDR_BATCH_FENCE");
  batch_fence_enabled_ = !fence_env || std::strcmp(fence_env, "0") != 0;
}
RegionAdapter::~RegionAdapter() = default;

void RegionAdapter::begin_d2h_batch() noexcept {
  if (batch_fence_enabled_) {
    active_d2h_epoch_ = ++next_d2h_epoch_;
  }
}

void RegionAdapter::end_d2h_batch() noexcept {
  active_d2h_epoch_ = 0;
}

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
                        const std::function<void(int)> &rank_visible,
                        Clock::time_point dispatch_start) {
  if (!ready()) throw std::logic_error("GDR: rank group is not ready");
  std::atomic<bool> stop{false};
  std::vector<std::exception_ptr> failures(ranks.size());
  std::vector<std::future<void>> futures;
  futures.reserve(ranks.size());
  std::exception_ptr dispatch_failure;
  const bool profile_on = profile_;
  const uint64_t d2h_epoch = h2d ? 0ULL : active_d2h_epoch_;
  try {
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
      if (ranks[rank].empty()) continue;
      futures.push_back(threads.enqueue(rank, [&, rank] {
        try {
          routes_[rank]->run(ranks[rank], h2d, stop, stopped, rank_visible,
                             profile_on, dispatch_start, d2h_epoch);
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

std::vector<RouteProfile> RegionAdapter::profiles() const {
  std::vector<RouteProfile> result;
  if (!profile_) return result;
  for (const auto &route : routes_) {
    if (route) result.push_back(route->profile);
  }
  return result;
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

void RegionBatchGroup::begin_d2h_batch() noexcept {
  if (gdr_) gdr_->begin_d2h_batch();
}

void RegionBatchGroup::end_d2h_batch() noexcept {
  if (gdr_) gdr_->end_d2h_batch();
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

std::vector<std::vector<uint64_t>> RegionBatchGroup::gdr_profile() const {
  std::vector<std::vector<uint64_t>> rows;
  if (!gdr_ || !gdr_->profile_enabled()) return rows;
  for (const auto &route : gdr_->profiles()) {
    const auto &h = route.h2d;
    const auto &d = route.d2h;
    rows.push_back({
        h.calls, h.dispatch_ns, h.post_ns, h.poll_ns, h.owner_wait_ns,
        h.sleep_ns, h.sleep_count, h.notify_ns, h.first_milestone_ns, h.route_total_ns,
        d.calls, d.dispatch_ns, d.post_ns, d.poll_ns, d.owner_wait_ns,
        d.sleep_ns, d.sleep_count, d.notify_ns, d.first_milestone_ns, d.route_total_ns,
    });
  }
  return rows;
}

bool RegionBatchGroup::try_gdr(const std::vector<RegionRequest> &requests,
                             const std::vector<int> *empty_layers, int counter_id) {
  if (gdr_stopped_.load(std::memory_order_acquire))
    throw gdr::UnsafeTransfer("GDR session is stopped");
  if (!gdr_ || !gdr_->ready() || requests.empty()) return false;
  const auto dispatch_start = gdr_->profile_enabled()
      ? gdr::RegionAdapter::Clock::now()
      : gdr::RegionAdapter::Clock::time_point{};
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
    }, dispatch_start);
  } catch (const gdr::UnsafeTransfer &) {
    throw;
  } catch (const std::exception &e) {
    // An eventfd write can have succeeded for a subset of consumers.
    throw gdr::UnsafeTransfer(e.what());
  }
  return true;
}

}  // namespace flexkv
