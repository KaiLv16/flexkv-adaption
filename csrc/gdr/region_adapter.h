// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdr/transport.h"
#include "transfer_backend.h"

#include <functional>
#include <atomic>

namespace flexkv {
class DeviceThreadPool;

namespace gdr {

struct RegionWork {
  RegionTransferArgs args;
  int milestone;
  int64_t cpu_num_blocks;
  int64_t gpu_num_blocks;
};

// Driver/lease setup is performed once by the Worker. All payload execution
// takes place on the rank's existing persistent thread, outside CUDA streams.
class RegionAdapter {
public:
  explicit RegionAdapter(int ranks);
  ~RegionAdapter();
  RegionAdapter(const RegionAdapter &) = delete;
  RegionAdapter &operator=(const RegionAdapter &) = delete;

  void install(int rank, std::unique_ptr<Transport> sender, int control_fd);
  bool ready() const;
  void run(DeviceThreadPool &threads, const std::vector<std::vector<RegionWork>> &ranks,
           bool h2d, const std::atomic<bool> &stopped,
           const std::function<void(int)> &rank_visible);
  std::vector<Stats> statistics() const;
  bool unsafe() const { return unsafe_; }

private:
  struct Route;
  std::vector<std::unique_ptr<Route>> routes_;
  bool unsafe_ = false;
};

}  // namespace gdr
}  // namespace flexkv
