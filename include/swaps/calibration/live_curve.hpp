#pragma once
// Lock-free publish of the live calibrated curve: the async pricer/calibrator split (the "pricing
// branch"). One WRITER (the calibrator thread, running StreamingCalibrator) publishes the newest knot
// vector each tick; any number of READERS (pricer threads) take a consistent snapshot LOCK-FREE and price
// off it, never blocking the calibrator and never seeing a torn (half-updated) curve.
//
// Mechanism -- atomic pointer swap over a preallocated buffer ring (RCU-style), plus a per-slot seqlock:
//   * publish(x): the writer fills the NEXT slot in a small ring (round-robin, so a slot a reader just
//     grabbed is not overwritten for N-1 more publishes), bracketing the payload write with an
//     odd/even generation counter, then release-stores the slot index into `published_`.
//   * snapshot(out): the reader acquire-loads `published_`, copies that slot's payload, then re-checks
//     the slot generation AND that `published_` still points there. If a publish lapped it mid-copy
//     (rare -- a publish is per-tick, a copy is ~µs), it retries. No locks, no allocation after the ctor.
//
// The only shared mutable state is the atomics; the payload is written by ONE thread and copied out by
// readers, so there is no data race. Memory ordering: release on publish / acquire on read gives readers
// a happens-before view of the full payload once they observe the new generation (§5 determinism: a
// reader always prices off SOME wholly-published x, never a mix of two).

#include <Eigen/Core>

#include <atomic>
#include <cstdint>
#include <vector>

namespace swaps::calibration {

class LiveCurveFeed {
 public:
  // `n_knots` fixes the payload size; `ring` is the buffer count (>=3: writer's target, the published
  // one, and headroom for in-flight readers). All buffers are allocated HERE, once -- publish/snapshot
  // never allocate.
  explicit LiveCurveFeed(int n_knots, int ring = 4) : slots_(ring) {
    for (auto& s : slots_) s.x = Eigen::VectorXd::Zero(n_knots);
  }

  int n_knots() const { return static_cast<int>(slots_[0].x.size()); }
  // How many times publish() has run (monotone). Lets a reader detect it has a fresh curve since last read.
  std::uint64_t version() const { return version_.load(std::memory_order_acquire); }

  // WRITER ONLY (the calibrator thread). Lock-free: never waits on a reader.
  void publish(const Eigen::VectorXd& x) {
    write_idx_ = (write_idx_ + 1) % static_cast<int>(slots_.size());
    Slot& s = slots_[write_idx_];
    s.gen.fetch_add(1, std::memory_order_release);   // -> odd: slot is being written
    s.x = x;                                         // payload (fixed size -> no realloc)
    s.gen.fetch_add(1, std::memory_order_release);   // -> even: slot is stable
    published_.store(write_idx_, std::memory_order_release);
    version_.fetch_add(1, std::memory_order_release);
  }

  // READER(S) (pricer thread[s]). Copies the current published curve into `out` (sized to n_knots by the
  // caller ONCE, reused every call). Lock-free; spins only if a publish lands during the copy. Returns
  // the version copied. `out` is guaranteed to be a wholly-published curve, never torn.
  std::uint64_t snapshot(Eigen::VectorXd& out) const {
    for (;;) {
      const int i = published_.load(std::memory_order_acquire);
      const Slot& s = slots_[i];
      const std::uint64_t g1 = s.gen.load(std::memory_order_acquire);
      if (g1 & 1u) continue;                         // writer mid-write on this slot -> retry
      out = s.x;                                      // copy the payload
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t g2 = s.gen.load(std::memory_order_acquire);
      // Clean iff the slot was not touched during the copy AND it is still the published slot.
      if (g1 == g2 && published_.load(std::memory_order_acquire) == i)
        return version_.load(std::memory_order_acquire);
    }
  }

  // Non-blocking single attempt: fills `out` and returns true on a clean read, false if it should retry
  // (a publish was in flight). For a reader that would rather reuse its previous snapshot than spin.
  bool try_snapshot(Eigen::VectorXd& out, std::uint64_t& ver) const {
    const int i = published_.load(std::memory_order_acquire);
    const Slot& s = slots_[i];
    const std::uint64_t g1 = s.gen.load(std::memory_order_acquire);
    if (g1 & 1u) return false;
    out = s.x;
    std::atomic_thread_fence(std::memory_order_acquire);
    const std::uint64_t g2 = s.gen.load(std::memory_order_acquire);
    if (g1 == g2 && published_.load(std::memory_order_acquire) == i) {
      ver = version_.load(std::memory_order_acquire);
      return true;
    }
    return false;
  }

 private:
  struct Slot {
    Eigen::VectorXd x;
    std::atomic<std::uint64_t> gen{0};  // even = stable, odd = being written (per-slot seqlock)
  };
  std::vector<Slot> slots_;
  std::atomic<int> published_{0};        // index of the current published slot (the "pointer swap")
  std::atomic<std::uint64_t> version_{0};
  int write_idx_ = 0;                    // writer-thread-local round-robin cursor
};

}  // namespace swaps::calibration
