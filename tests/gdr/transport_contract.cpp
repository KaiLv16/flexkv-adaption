#include "gdr/cq_watchdog.h"
#include "gdr/transport.cpp"

#include <array>
#include <cassert>

using flexkv::gdr::CqWatchdog;
using flexkv::gdr::Span;
using flexkv::gdr::Transport;
using flexkv::gdr::TransferCheckpoint;
using flexkv::gdr::UnsafeTransfer;

void check_watchdog() {
  using namespace std::chrono;
  CqWatchdog watchdog(seconds(30));
  const CqWatchdog::Clock::time_point start{};
  // Waiting for the Owner, even after a drained chain, has no CQ deadline.
  assert(!watchdog.expired(start, false, false));
  assert(!watchdog.expired(start + seconds(180), false, false));
  assert(!watchdog.expired(start + seconds(181), true, false));
  for (int sec = 190; sec <= 400; sec += 10)
    assert(!watchdog.expired(start + seconds(sec), true, true));
  assert(watchdog.expired(start + seconds(430), true, false));
  assert(!watchdog.expired(start + seconds(431), false, true));
  assert(!watchdog.expired(start + seconds(900), true, false));
}

void check_transfer(const std::string &fault) {
  std::array<unsigned char, 128> host{}, owner_gpu{}, peer_memory{}, imported_view{};
  std::array<unsigned char, 128> &backing =
#ifdef FLEXKV_GDR_MAPPED
      peer_memory;
#else
      owner_gpu;
#endif
  fake::mapped_address = peer_memory.data() + 4;
  Transport owner(0, false, "mock", "", 4, 0);
  assert(owner.nic() == "mock:2");
  owner.register_device(owner_gpu.data() + 4, 100);
  auto remote = owner.export_region(owner_gpu.data() + 12, 72);
  assert(remote.remote == reinterpret_cast<uintptr_t>(backing.data() + 12));
  remote.local = reinterpret_cast<uintptr_t>(imported_view.data() + 8);

  uint32_t target = 0;
  {
    Transport sender(0, true, owner.nic(), owner.endpoint(), 4, 2);
    sender.register_host(host.data(), host.size());
    target = owner.accept(sender.sender_qpns().front());
    sender.connect({target}, {remote});
    const std::array<int, 4> host_offsets{{96, 0, 48, 24}};
    const std::array<int, 4> gpu_offsets{{64, 8, 40, 16}};
    std::array<Span, 4> spans;
    for (size_t i = 0; i < spans.size(); ++i) {
      std::fill_n(host.data() + host_offsets[i], 8, i + 1);
      const auto local = reinterpret_cast<uintptr_t>(host.data() + host_offsets[i]);
      assert(sender.resolve(local, remote.local + gpu_offsets[i], 16, true, spans[i]) ==
             (i == 0 ? 8 : 16));
      spans[i].bytes = 8;
      assert(spans[i].remote == remote.remote + gpu_offsets[i]);
    }
    try {
      Span span;
      sender.resolve(reinterpret_cast<uintptr_t>(host.data()), remote.local + remote.bytes, 1, true, span);
      assert(false && "address escaped the exported view");
    } catch (const std::invalid_argument &) {}

    if (fault == "partial") {
      fake::reject_index = 1;
      try {
        sender.post_payload(spans.data(), spans.size(), true);
        assert(false && "post accepted a partial chain silently");
      } catch (const UnsafeTransfer &) {
        assert(sender.has_outstanding());
      }
    } else {
      assert(sender.post_payload(spans.data(), spans.size(), true) == spans.size());
      fake::hold_cq = true;
      assert(sender.poll() == 0 && sender.has_outstanding() && backing[76] == 0);
      fake::hold_cq = false;
      fake::bad_sequence = fault == "sequence";
      fake::cq_error = fault == "wc";
      if (!fault.empty()) {
        try {
          sender.poll();
          assert(false && "bad CQE accepted");
        } catch (const UnsafeTransfer &) {
          assert(sender.has_outstanding());
        }
      } else {
        assert(sender.poll() == 1 && !sender.has_outstanding());
        for (size_t i = 0; i < spans.size(); ++i)
          assert(backing[12 + gpu_offsets[i]] == i + 1);
#ifdef FLEXKV_GDR_MAPPED
        assert(owner_gpu[76] == 0);
        TransferCheckpoint first{}, second{}, third{};
        assert(sender.finish_scope(first));
        assert(sender.post_payload(spans.data(), 1, true) == 1);
        assert(sender.finish_scope(second));
        assert(sender.post_payload(spans.data(), 1, true) == 1);
        assert(!sender.finish_scope(third));
        assert(sender.poll() == 1 && sender.is_complete(first) && !sender.is_complete(second));
        assert(sender.finish_scope(third));
        while (sender.has_outstanding()) assert(sender.poll() > 0);
        assert(sender.is_complete(second) && sender.is_complete(third));
#endif
        host.fill(0);
        assert(sender.post_payload(spans.data(), spans.size(), false) == spans.size());
        for (size_t i = 0; i < spans.size(); ++i) assert(sender.poll() == 1);
        for (size_t i = 0; i < spans.size(); ++i)
          assert(host[host_offsets[i]] == i + 1);
      }
    }
  }
  assert(fake::live_qps == (fault.empty() ? 1 : 2));
  if (fault.empty()) owner.release(target);
}

void check_parallel_queues() {
  constexpr size_t width = 64u << 10;
  constexpr size_t bytes = 4 * width;
  std::vector<unsigned char> host_storage(2 * bytes), gpu_storage(2 * bytes),
      peer_storage(2 * bytes), imported(bytes);
  auto align = [](std::vector<unsigned char> &storage) {
    void *ptr = storage.data();
    size_t space = storage.size();
    return static_cast<unsigned char *>(std::align(bytes, bytes, ptr, space));
  };
  auto *host = align(host_storage);
  auto *gpu = align(gpu_storage);
  auto *peer = align(peer_storage);
  assert(host && gpu && peer);
  fake::mapped_address = peer;
  Transport owner(0, false, "mock", "", 4, 0, 4);
  owner.register_device(gpu, bytes);
  auto remote = owner.export_region(gpu, bytes);
  remote.local = reinterpret_cast<uintptr_t>(imported.data());

  std::vector<uint32_t> targets;
  {
    Transport sender(0, true, owner.nic(), owner.endpoint(), 4, 2, 4);
    sender.register_host(host, bytes);
    const auto qpns = sender.sender_qpns();
    assert(qpns.size() == 4);
    for (auto qpn : qpns) targets.push_back(owner.accept(qpn));
    sender.connect(targets, {remote});
    auto *backing = reinterpret_cast<unsigned char *>(remote.remote);
    Span spans[5];
    unsigned used = 0;
    for (size_t i = 0; i < 4; ++i) {
      host[i * width] = static_cast<unsigned char>(i + 1);
      assert(sender.resolve(reinterpret_cast<uintptr_t>(host + i * width),
                            remote.local + i * width, 1, true, spans[i]) == 1);
      used |= 1u << spans[i].qp_index;
    }
    assert(used == 15);
    // The second write overlaps the first and must stay ordered on that SQ.
    host[7] = 99;
    assert(sender.resolve(reinterpret_cast<uintptr_t>(host + 7), remote.local,
                          1, true, spans[4]) == 1);
    assert(spans[4].qp_index == spans[0].qp_index);
    Span boundary;
    assert(sender.resolve(reinterpret_cast<uintptr_t>(host), remote.local + width - 1,
                          2, true, boundary) == 1);
    assert(sender.post_payload(spans, 5, true) == 5);
    TransferCheckpoint checkpoint{};
    assert(sender.finish_scope(checkpoint));
    fake::blocked_qp = qpns[spans[0].qp_index];
    while (sender.poll()) {}
    assert(!sender.is_complete(checkpoint));
    assert(backing[0] == 0 && backing[width] == 2 && backing[3 * width] == 4);
    fake::blocked_qp = 0;
    while (sender.has_outstanding()) assert(sender.poll() > 0);
    assert(sender.is_complete(checkpoint) && backing[0] == 99);
#ifdef FLEXKV_GDR_MAPPED
    assert(sender.stats().readbacks == 4);
#else
    assert(sender.stats().readbacks == 0);
#endif
    std::fill_n(host, bytes, 0);
    for (size_t i = 0; i < 4; ++i) {
      assert(sender.resolve(reinterpret_cast<uintptr_t>(host + i * width),
                            remote.local + i * width, 1, false, spans[i]) == 1);
    }
    assert(sender.post_payload(spans, 4, false) == 4);
    assert(sender.finish_scope(checkpoint));
    while (sender.has_outstanding()) assert(sender.poll() > 0);
    assert(sender.is_complete(checkpoint) && host[0] == 99 && host[3 * width] == 4);
    const auto stats = sender.stats();
    assert(stats.qp_payload_wrs.size() == 4);
    for (auto wrs : stats.qp_payload_wrs) assert(wrs >= 2);
  }
  assert(fake::live_qps == 4);
  for (auto target : targets) owner.release(target);
  assert(fake::live_qps == 0);
}

int main(int argc, char **argv) {
  check_watchdog();
  const std::string fault = argc > 1 ? argv[1] : "";
  check_transfer(fault);
  if (fault.empty()) check_parallel_queues();
}
