#pragma once

#include <chrono>

namespace flexkv {
namespace gdr {

// Owner-side ordering may wait for GPU work for an unbounded time. Only
// outstanding WRs can make a missed CQ deadline a transport failure.
class CqWatchdog {
public:
  using Clock = std::chrono::steady_clock;

  explicit CqWatchdog(Clock::duration timeout) : timeout_(timeout) {}

  bool expired(Clock::time_point now, bool outstanding, bool completed) {
    if (!outstanding) {
      deadline_ = Clock::time_point::max();
      return false;
    }
    if (completed || deadline_ == Clock::time_point::max())
      deadline_ = now + timeout_;
    return now >= deadline_;
  }

private:
  Clock::duration timeout_;
  Clock::time_point deadline_ = Clock::time_point::max();
};

}  // namespace gdr
}  // namespace flexkv
