// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flexkv::gdr {

struct Span {
  uint64_t host;
  uint64_t remote;
  uint32_t bytes;
  uint32_t lkey;
  uint32_t rkey;
};
struct RemoteRegion {
  uintptr_t local;
  size_t bytes;
  uint64_t remote;
  uint32_t key;
};
struct Stats {
  uint64_t h2d_bytes = 0;
  uint64_t d2h_bytes = 0;
  uint64_t payload_wrs = 0;
  uint64_t posts = 0;
  uint64_t completions = 0;
  uint64_t readbacks = 0;
};
class UnsafeTransfer : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// After setup, a sender's SQ and CQ have one combined executor. The endpoint
// in the allocation owner never posts payload or polls the sender's CQ.
class Transport {
public:
  Transport(int device, bool sender, const std::string &nic,
            const std::string &required_endpoint, int read_depth, int readbacks);
  ~Transport();
  Transport(const Transport &) = delete;
  Transport &operator=(const Transport &) = delete;

  void register_device(void *base, size_t bytes);
  void register_host(void *base, size_t bytes);
  RemoteRegion export_region(void *base, size_t bytes) const;
  const std::string &endpoint() const;
  const std::string &nic() const;
  int read_depth() const;
  uint32_t sender_qpn() const;
  uint32_t accept(uint32_t sender);
  void release(uint32_t target);
  void connect(uint32_t target, std::vector<RemoteRegion> regions);

  size_t resolve(uintptr_t host, uintptr_t device, size_t bytes, Span &out);
  size_t post_payload(const Span *spans, size_t count, bool h2d);
  bool post_readback(const Span &last_write, uint64_t fence);
  int poll();
  uint64_t posted() const;
  uint64_t retired() const;
  uint64_t fence_retired() const;
  bool requires_readback() const;
  size_t max_payload() const;
  Stats stats() const;
  void mark_unsafe();

  static void make_device_visible(int device);

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace flexkv::gdr
