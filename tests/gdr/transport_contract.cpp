#include "gdr/cq_watchdog.h"
#include "gdr/transport.cpp"

#include <array>
#include <cassert>

using flexkv::gdr::CqWatchdog;
using flexkv::gdr::Span;
using flexkv::gdr::Transport;
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
    target = owner.accept(sender.sender_qpn());
    sender.connect(target, {remote});
    const std::array<int, 4> host_offsets{{96, 0, 48, 24}};
    const std::array<int, 4> gpu_offsets{{64, 8, 40, 16}};
    std::array<Span, 4> spans;
    for (size_t i = 0; i < spans.size(); ++i) {
      std::fill_n(host.data() + host_offsets[i], 8, i + 1);
      const auto local = reinterpret_cast<uintptr_t>(host.data() + host_offsets[i]);
      assert(sender.resolve(local, remote.local + gpu_offsets[i], 16, spans[i]) ==
             (i == 0 ? 8 : 16));
      spans[i].bytes = 8;
      assert(spans[i].remote == remote.remote + gpu_offsets[i]);
    }
    try {
      Span span;
      sender.resolve(reinterpret_cast<uintptr_t>(host.data()), remote.local + remote.bytes, 1, span);
      assert(false && "address escaped the exported view");
    } catch (const std::invalid_argument &) {}

    if (fault == "partial") {
      fake::reject_index = 1;
      try {
        sender.post_payload(spans.data(), spans.size(), true);
        assert(false && "post accepted a partial chain silently");
      } catch (const UnsafeTransfer &) {
        assert(sender.posted() == 1 && sender.retired() == 0);
      }
    } else {
      assert(sender.post_payload(spans.data(), spans.size(), true) == spans.size());
      fake::hold_cq = true;
      assert(sender.poll() == 0 && sender.retired() == 0 && backing[76] == 0);
      fake::hold_cq = false;
      fake::bad_sequence = fault == "sequence";
      fake::cq_error = fault == "wc";
      if (!fault.empty()) {
        try {
          sender.poll();
          assert(false && "bad CQE accepted");
        } catch (const UnsafeTransfer &) {
          assert(sender.retired() == 0);
        }
      } else {
        assert(sender.poll() == 1 && sender.retired() == spans.size());
        for (size_t i = 0; i < spans.size(); ++i)
          assert(backing[12 + gpu_offsets[i]] == i + 1);
#ifdef FLEXKV_GDR_MAPPED
        assert(owner_gpu[76] == 0);
        assert(sender.post_readback(spans.back(), 1));
        assert(sender.post_readback(spans.back(), 2));
        assert(!sender.post_readback(spans.back(), 3));
        assert(sender.poll() == 1 && sender.fence_retired() == 1);
        assert(sender.poll() == 1 && sender.fence_retired() == 2);
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

int main(int argc, char **argv) {
  check_watchdog();
  check_transfer(argc > 1 ? argv[1] : "");
}
