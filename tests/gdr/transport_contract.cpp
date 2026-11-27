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
    Span unsplit_chunk;
    assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host), remote.local + width - 1,
                                2, 42, unsplit_chunk) == 2);
    Span same_key_chunk;
    assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host), remote.local + width,
                                2, 42, same_key_chunk) == 2);
    assert(unsplit_chunk.qp_index == same_key_chunk.qp_index);
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

void check_multi_sge_and_decoupled_qps() {
  constexpr size_t block_stride = 1024;
  constexpr size_t chunk = 64;
  constexpr size_t num_blocks = 8;
  constexpr size_t host_bytes = num_blocks * block_stride;
  constexpr size_t gpu_bytes = num_blocks * chunk + 128;
  std::vector<unsigned char> host(host_bytes, 0), gpu(gpu_bytes, 0),
      peer(gpu_bytes, 0), imported(gpu_bytes, 0);
  fake::mapped_address = peer.data();

  // 1) Multi-SGE gather (H2D) + scatter (D2H) + decoupled QPs (qp_count=8 -> h2d=4, d2h=8)
  Transport owner(0, false, "mock", "", 4, 0, 8);
  owner.register_device(gpu.data(), gpu.size());
  auto remote = owner.export_region(gpu.data(), gpu.size());
  remote.local = reinterpret_cast<uintptr_t>(imported.data());

  std::vector<uint32_t> targets;
  {
    Transport sender(0, true, owner.nic(), owner.endpoint(), 4, 2, 8);
    assert(sender.h2d_qp_count() == 4);
    assert(sender.d2h_qp_count() == 8);
    assert(sender.write_sge_limit() == 16);
    assert(sender.read_sge_limit() == 16);
    sender.register_host(host.data(), host.size());
    for (auto qpn : sender.sender_qpns()) targets.push_back(owner.accept(qpn));
    sender.connect(targets, {remote});

    auto *backing = reinterpret_cast<unsigned char *>(remote.remote);
    std::array<Span, num_blocks> h2d_spans{};
    for (size_t i = 0; i < num_blocks; ++i) {
      for (size_t b = 0; b < chunk; ++b) {
        host[i * block_stride + b] = static_cast<unsigned char>((i * 17 + b + 3) & 0x7f);
      }
      assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data() + i * block_stride),
                                  remote.local + i * chunk, chunk, 7, h2d_spans[i], true) == chunk);
      assert(h2d_spans[i].qp_index < 4);
      if (i > 0) assert(h2d_spans[i].qp_index == h2d_spans[0].qp_index);
    }
    fake::last_max_sge_seen = 0;
    fake::wr_ex_posts = 0;
#ifdef FLEXKV_GDR_WR_API
    assert(sender.uses_wr_api());
#else
    assert(!sender.uses_wr_api());
#endif
    assert(sender.post_payload(h2d_spans.data(), h2d_spans.size(), true) == num_blocks);
    // All 8 discontiguous BLOCKFIRST host chunks merged into 1 single 8-SGE RDMA_WRITE WR!
    assert(fake::last_max_sge_seen == static_cast<int>(num_blocks));
    assert(sender.stats().payload_wrs == 1);
    assert(sender.stats().h2d_bytes == num_blocks * chunk);
#ifdef FLEXKV_GDR_WR_API
    assert(fake::wr_ex_posts == 1);
#else
    assert(fake::wr_ex_posts == 0);
#endif

    TransferCheckpoint cp{};
    assert(sender.finish_scope(cp));
    while (sender.has_outstanding()) assert(sender.poll() > 0);
    assert(sender.is_complete(cp));
    for (size_t i = 0; i < num_blocks; ++i) {
      for (size_t b = 0; b < chunk; ++b) {
        assert(backing[i * chunk + b] == static_cast<unsigned char>((i * 17 + b + 3) & 0x7f));
      }
    }

    // Verify overlapping write ordering is never violated by Multi-SGE coalescing:
    // Write A -> [0..64), Write B -> [0..64) (overwrites A!), Write C -> [64..128) (coalesces with B, NOT A).
    std::array<Span, 3> overlap_spans{};
    std::fill_n(host.data(), chunk, 11);
    std::fill_n(host.data() + block_stride, chunk, 22);
    std::fill_n(host.data() + 2 * block_stride, chunk, 33);
    assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data()),
                                remote.local, chunk, 7, overlap_spans[0], true) == chunk);
    assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data() + block_stride),
                                remote.local, chunk, 7, overlap_spans[1], true) == chunk);
    assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data() + 2 * block_stride),
                                remote.local + chunk, chunk, 7, overlap_spans[2], true) == chunk);
    assert(sender.post_payload(overlap_spans.data(), overlap_spans.size(), true) == 3);
    // Expected: 2 WRs (WR0=[A], WR1=[B, C]), never coalescing C into WR0 ahead of B.
    assert(sender.stats().payload_wrs == 3);
    assert(sender.finish_scope(cp));
    while (sender.has_outstanding()) assert(sender.poll() > 0);
    assert(backing[0] == 22 && backing[chunk - 1] == 22);
    assert(backing[chunk] == 33 && backing[2 * chunk - 1] == 33);

    // Verify D2H routes across all 8 QPs while H2D only touched QPs 0..3:
    unsigned h2d_mask = 0, d2h_mask = 0;
    for (uint64_t k = 0; k < 32; ++k) {
      Span hs{}, ds{};
      sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data()), remote.local, 8, k, hs, true);
      sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data()), remote.local, 8, k, ds, false);
      h2d_mask |= 1u << hs.qp_index;
      d2h_mask |= 1u << ds.qp_index;
    }
    assert(h2d_mask == 0x0fu);
    assert(d2h_mask == 0xffu);

    // Restore original pattern on backing and test D2H Multi-SGE scatter into discontiguous Host:
    for (size_t i = 0; i < num_blocks; ++i) {
      for (size_t b = 0; b < chunk; ++b) {
        backing[i * chunk + b] = static_cast<unsigned char>((i * 29 + b + 5) & 0x7f);
      }
    }
    std::fill(host.begin(), host.end(), 0);
    std::array<Span, num_blocks> d2h_spans{};
    for (size_t i = 0; i < num_blocks; ++i) {
      assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data() + i * block_stride),
                                  remote.local + i * chunk, chunk, 9, d2h_spans[i], false) == chunk);
    }
    const auto wrs_before = sender.stats().payload_wrs;
    assert(sender.post_payload(d2h_spans.data(), d2h_spans.size(), false) == num_blocks);
    // All 8 blocks coalesced into 1 single RDMA_READ WR consuming just 1 read_depth credit!
    assert(sender.stats().payload_wrs == wrs_before + 1);
    assert(sender.finish_scope(cp));
    while (sender.has_outstanding()) assert(sender.poll() > 0);
    for (size_t i = 0; i < num_blocks; ++i) {
      for (size_t b = 0; b < chunk; ++b) {
        assert(host[i * block_stride + b] == static_cast<unsigned char>((i * 29 + b + 5) & 0x7f));
      }
    }
  }
  for (auto t : targets) owner.release(t);
  targets.clear();

  // 2) Asymmetric max_sge_rd < max_sge and SQ-capacity fallback negotiation
  fake::max_sge = 16;
  fake::max_sge_for_full_sq = 8;
  fake::max_sge_rd = 3;
  {
    Transport asym_owner(0, false, "mock", "", 4, 0, 1);
    asym_owner.register_device(gpu.data(), gpu.size());
    auto asym_remote = asym_owner.export_region(gpu.data(), gpu.size());
    asym_remote.local = reinterpret_cast<uintptr_t>(imported.data());
    uint32_t t = 0;
    {
      Transport sender(0, true, asym_owner.nic(), asym_owner.endpoint(), 4, 2, 1);
      assert(sender.write_sge_limit() == 8);
      assert(sender.read_sge_limit() == 3);
      sender.register_host(host.data(), host.size());
      t = asym_owner.accept(sender.sender_qpns().front());
      sender.connect({t}, {asym_remote});

      std::array<Span, num_blocks> d2h_spans{};
      std::fill(host.begin(), host.end(), 0);
      for (size_t i = 0; i < num_blocks; ++i) {
        assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data() + i * block_stride),
                                    asym_remote.local + i * chunk, chunk, 0, d2h_spans[i], false) == chunk);
      }
      // 8 chunks with read_sge_limit=3 -> ceil(8/3) = 3 RDMA_READ WRs (<= read_depth=4)
      assert(sender.post_payload(d2h_spans.data(), d2h_spans.size(), false) == num_blocks);
      assert(sender.stats().payload_wrs == 3);
      while (sender.has_outstanding()) assert(sender.poll() > 0);
      auto *backing = reinterpret_cast<unsigned char *>(asym_remote.remote);
      for (size_t i = 0; i < num_blocks; ++i) {
        assert(std::memcmp(host.data() + i * block_stride, backing + i * chunk, chunk) == 0);
      }
    }
    asym_owner.release(t);
  }
  fake::max_sge = 16;
  fake::max_sge_for_full_sq = 16;
  fake::max_sge_rd = 16;

  // 3) Provider fallback when ibv_create_qp_ex is unsupported (EOPNOTSUPP)
  fake::disable_qp_ex = true;
  {
    Transport fb_owner(0, false, "mock", "", 4, 0, 2);
    fb_owner.register_device(gpu.data(), gpu.size());
    auto fb_remote = fb_owner.export_region(gpu.data(), gpu.size());
    fb_remote.local = reinterpret_cast<uintptr_t>(imported.data());
    std::vector<uint32_t> fb_targets;
    {
      Transport sender(0, true, fb_owner.nic(), fb_owner.endpoint(), 4, 2, 2);
      assert(!sender.uses_wr_api());
      assert(sender.write_sge_limit() == 16);
      sender.register_host(host.data(), host.size());
      for (auto qpn : sender.sender_qpns()) fb_targets.push_back(fb_owner.accept(qpn));
      sender.connect(fb_targets, {fb_remote});
      Span sp{};
      assert(sender.resolve_chunk(reinterpret_cast<uintptr_t>(host.data()),
                                  fb_remote.local, chunk, 0, sp, true) == chunk);
      fake::wr_ex_posts = 0;
      assert(sender.post_payload(&sp, 1, true) == 1);
      assert(fake::wr_ex_posts == 0);
      while (sender.has_outstanding()) assert(sender.poll() > 0);
    }
    for (auto t : fb_targets) fb_owner.release(t);
  }
  fake::disable_qp_ex = false;
  assert(fake::live_qps == 0);
}

int main(int argc, char **argv) {
  check_watchdog();
  const std::string fault = argc > 1 ? argv[1] : "";
  check_transfer(fault);
  if (fault.empty()) {
    check_parallel_queues();
    check_multi_sge_and_decoupled_qps();
  }
}
