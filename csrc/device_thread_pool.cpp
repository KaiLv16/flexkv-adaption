/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "device_thread_pool.h"
#include "cuda_stream_lifetime.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sched.h>
#endif

namespace flexkv {
namespace {

void bind_thread_to_device_numa(int device_id) noexcept {
#ifdef __linux__
  const char *enable = std::getenv("FLEXKV_ENABLE_NUMA_BIND");
  const char *disable = std::getenv("FLEXKV_DISABLE_NUMA_BIND");
  if (!enable || std::strcmp(enable, "1") != 0 ||
      (disable && std::strcmp(disable, "1") == 0)) {
    return;
  }
  char pci_bus_id[32] = {0};
  if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  std::string bdf(pci_bus_id);
  for (char &ch : bdf) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  std::ifstream cpulist_file("/sys/bus/pci/devices/" + bdf + "/local_cpulist");
  std::string cpulist;
  if (!cpulist_file.is_open() || !std::getline(cpulist_file, cpulist) || cpulist.empty()) {
    return;
  }

  cpu_set_t local_mask;
  CPU_ZERO(&local_mask);
  size_t pos = 0;
  while (pos < cpulist.size()) {
    size_t comma = cpulist.find(',', pos);
    std::string token = cpulist.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    pos = (comma == std::string::npos) ? cpulist.size() : comma + 1;
    if (token.empty()) {
      continue;
    }
    size_t dash = token.find('-');
    char *endptr = nullptr;
    long first = std::strtol(token.c_str(), &endptr, 10);
    if (endptr == token.c_str() || first < 0) {
      continue;
    }
    long last = first;
    if (dash != std::string::npos) {
      const char *rhs = token.c_str() + dash + 1;
      last = std::strtol(rhs, &endptr, 10);
      if (endptr == rhs || last < first) {
        continue;
      }
    }
    for (long cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
      CPU_SET(static_cast<int>(cpu), &local_mask);
    }
  }

  cpu_set_t allowed_mask;
  CPU_ZERO(&allowed_mask);
  if (sched_getaffinity(0, sizeof(allowed_mask), &allowed_mask) != 0) {
    return;
  }
  cpu_set_t target_mask;
  CPU_AND(&target_mask, &local_mask, &allowed_mask);
  if (CPU_COUNT(&target_mask) > 0) {
    sched_setaffinity(0, sizeof(target_mask), &target_mask);
  }
#else
  (void)device_id;
#endif
}

}  // namespace

DeviceThreadPool::DeviceThreadPool(const std::vector<int> &device_ids)
    : device_ids_(device_ids) {
  const int n = static_cast<int>(device_ids_.size());
  // Snapshot the driver-level current device: the cudaSetDevice calls below
  // are raw, so nothing else restores it for us.
  int prev_device = 0;
  cudaGetDevice(&prev_device);

  queues_.resize(n);
  mtxs_ = std::vector<std::mutex>(n);
  cvs_ = std::vector<std::condition_variable>(n);
  streams_.assign(n, nullptr);

  for (int i = 0; i < n; ++i) {
    cudaError_t err = cudaSetDevice(device_ids_[i]);
    if (err == cudaSuccess) {
      err = cudaStreamCreate(&streams_[i]);
    }
    if (err != cudaSuccess) {
      // Destroy what we already made: the destructor will not run for a ctor
      // that throws.
      for (int j = 0; j < i; ++j) {
        if (streams_[j] != nullptr &&
            cudaSetDevice(device_ids_[j]) == cudaSuccess) {
          cudaStreamDestroy(streams_[j]);
        }
      }
      cudaSetDevice(prev_device);
      cudaGetLastError();
      throw std::runtime_error(
          std::string("DeviceThreadPool: device setup failed for device ") +
          std::to_string(device_ids_[i]) + ": " + cudaGetErrorString(err));
    }
  }
  cudaSetDevice(prev_device);

  for (int i = 0; i < n; ++i) {
    threads_.emplace_back([this, i]() {
      // Once, for the life of the thread. Every task that runs here inherits
      // the right device and PCIe-local CPU affinity without touching it.
      cudaSetDevice(device_ids_[i]);
      bind_thread_to_device_numa(device_ids_[i]);
      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_ || !queues_[i].empty(); });
          if (stop_ && queues_[i].empty()) {
            return;
          }
          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();
      }
    });
  }
}

void DeviceThreadPool::shutdown() {
  if (stop_.exchange(true)) {
    return;
  }
  for (auto &cv : cvs_) {
    cv.notify_all();
  }
  for (auto &t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

DeviceThreadPool::~DeviceThreadPool() {
  int prev_device = 0;
  cudaGetDevice(&prev_device);

  // Join first: a live worker thread could still be enqueuing onto a stream
  // we are about to destroy.
  shutdown();

  for (size_t i = 0; i < streams_.size(); ++i) {
    if (streams_[i] == nullptr) {
      continue;
    }
    if (cudaSetDevice(device_ids_[i]) != cudaSuccess) {
      continue; // device gone (driver shutdown); nothing safe left to do
    }
    cudaStreamSynchronize(streams_[i]);
    cudaStreamDestroy(streams_[i]);
    streams_[i] = nullptr;
  }
  cudaSetDevice(prev_device);
  // Swallow teardown errors so the next cudaGetLastError() caller does not
  // mistake them for a transfer failure.
  cudaGetLastError();
}

std::future<void> DeviceThreadPool::enqueue(int idx, Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[idx]);
    queues_[idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[idx].notify_one();
  return fut;
}

void DeviceThreadPool::run_on_all(const std::function<void(int)> &body,
                                  const char *what) {
  std::mutex error_mtx;
  std::string error_msg;
  auto record = [&](const std::string &msg) {
    std::lock_guard<std::mutex> lk(error_mtx);
    if (error_msg.empty()) {
      error_msg = msg;
    }
  };

  std::vector<std::future<void>> futures;
  futures.reserve(device_ids_.size());
  for (int i = 0; i < size(); ++i) {
    futures.emplace_back(enqueue(i, [&, i]() {
      try {
        body(i);
      } catch (const std::exception &e) {
        record(std::string("rank ") + std::to_string(i) + ": " + e.what());
      } catch (...) {
        // A non-std::exception throw would otherwise escape the
        // packaged_task and resurface at f.get() as an opaque failure.
        record(std::string("rank ") + std::to_string(i) +
               ": unknown exception");
      }
    }));
  }
  for (auto &f : futures) {
    try {
      f.get();
    } catch (const std::exception &e) {
      record(std::string("future: ") + e.what());
    } catch (...) {
      record("future: unknown exception");
    }
  }
  if (!error_msg.empty()) {
    drain_all_streams_on_error();
    throw std::runtime_error(std::string(what) + " failed: " + error_msg);
  }
}

void DeviceThreadPool::drain_all_streams_on_error() {
  // Do not use run_on_all here: that function calls us on failure.
  std::vector<std::future<void>> futures;
  futures.reserve(device_ids_.size());
  for (int i = 0; i < size(); ++i) {
    futures.emplace_back(enqueue(i, []() {
      // The fallback per-group launcher may already have submitted an earlier
      // group on another pool's streams. Drain this device's whole context,
      // not only this pool, before its shared operation can release blocks.
      drain_device_or_abort("transfer error recovery");
    }));
  }
  for (auto &future : futures) {
    future.get();
  }
}

void DeviceThreadPool::sync_all_streams() {
  run_on_all(
      [this](int i) {
        cudaError_t err = cudaStreamSynchronize(streams_[i]);
        if (err != cudaSuccess) {
          throw std::runtime_error(cudaGetErrorString(err));
        }
      },
      "wait_all_streams");
}

} // namespace flexkv
