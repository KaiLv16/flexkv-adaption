// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdr/transport.h"
#include "transfer_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace flexkv {
class DeviceThreadPool;

namespace gdr {

struct RegionWork {
  RegionTransferArgs args;
  int milestone;
  int64_t cpu_num_blocks;
  int64_t gpu_num_blocks;
};

struct DirectionProfile {
  uint64_t calls = 0;
  uint64_t dispatch_ns = 0;
  uint64_t post_ns = 0;
  uint64_t poll_ns = 0;
  uint64_t owner_wait_ns = 0;
  uint64_t sleep_ns = 0;
  uint64_t sleep_count = 0;
  uint64_t notify_ns = 0;
  uint64_t first_milestone_ns = 0;
  uint64_t route_total_ns = 0;
};

struct RouteProfile {
  DirectionProfile h2d;
  DirectionProfile d2h;
};

// Driver/lease setup is performed once by the Worker. All payload execution
// takes place on the rank's existing persistent thread, outside CUDA streams.
class RegionAdapter {
public:
  using Clock = std::chrono::steady_clock;

  explicit RegionAdapter(int ranks);
  ~RegionAdapter();
  RegionAdapter(const RegionAdapter &) = delete;
  RegionAdapter &operator=(const RegionAdapter &) = delete;

  void install(int rank, std::unique_ptr<Transport> sender, int control_fd);
  bool ready() const;
  bool profile_enabled() const { return profile_; }
  void begin_d2h_batch() noexcept;
  void end_d2h_batch() noexcept;
  void run(DeviceThreadPool &threads, const std::vector<std::vector<RegionWork>> &ranks,
           bool h2d, const std::atomic<bool> &stopped,
           const std::function<void(int)> &rank_visible,
           Clock::time_point dispatch_start = Clock::time_point{});
  std::vector<Stats> statistics() const;
  std::vector<RouteProfile> profiles() const;
  bool unsafe() const { return unsafe_; }

private:
  struct Route;
  std::vector<std::unique_ptr<Route>> routes_;
  uint64_t next_d2h_epoch_ = 0;
  uint64_t active_d2h_epoch_ = 0;
  bool batch_fence_enabled_ = true;
  bool unsafe_ = false;
  bool profile_ = false;
};

}  // namespace gdr
}  // namespace flexkv
