// E5 taxonomy: T4 hot-path invariant (allocation + factorisation + refresh-count pins on the streamer itself, reachable by tools/mutate.py)
// HOT-PATH GUARDS G1-G3 (2026-09-15 guard audit, A3). Every streaming stage that can allocate or refactorise now has a pin that a
// header-only TU reaches, so a mutation of streaming.hpp can be shown caught:
//   G1 the band walk (rescales, kink pins, KKT releases, rescale_row's degenerate-update fallback) -- its only allocation pin
//      disappeared with K5' (the requote pin runs at 0.1 bp and never reaches an edge);
//   G2 a 25 bp move on a square rung (no refresh, no allocation: the RefreshTick25bp metric's name overstates it), an in-tick refresh
//      (costs exactly its factorisation), and the regularised tick (the SDK / web default);
//   G3 the streamer-level twin of the session ladder pins (hotpath_shapes_test T4), with the break-even PINNED so the refresh
//      schedule is deterministic (a measured break-even moves with machine load: FLK3).
// Every count was measured 2026-09-15 on 38615f3 (scratchpad/guards/probe_guards). Pins may only DECREASE.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "malloc_count.hpp"  // bench/fixtures (an include dir of swaps_tests)
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;
using swaps::testing::AllocScope;

namespace {

const Shape& rung(const char* name) {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  for (const Shape& s : L)
    if (s.name == name) return s;
  throw std::invalid_argument(std::string("no rung ") + name);
}

bool has_band(const Shape& s) {
  for (const auto& in : s.prob.instruments)
    if (in.band_upper > in.band_lower) return true;
  return false;
}

SC::Options pinned() {
  SC::Options o;
  o.breakeven_steps = 64;  // a deterministic refresh schedule (the measured break-even moves with load)
  return o;
}

// The walk sequence: every HARD row (not banded, not an FX forward, not a turn) moves by amp * sin(0.8 k + 0.9 i); banded targets stay,
// so the curve drags banded model quotes across their edges and back.
std::vector<Eigen::VectorXd> walk_sequence(const Shape& s, double amp, int n) {
  std::vector<Eigen::VectorXd> seq;
  for (int k = 0; k < n; ++k) {
    Eigen::VectorXd q = s.q0;
    for (int i = 0; i < q.size(); ++i) {
      const auto& in = s.prob.instruments[static_cast<std::size_t>(i)];
      if (in.band_upper > in.band_lower || in.quote == cal::QuoteKind::FxForward || in.quote == cal::QuoteKind::TurnJump) continue;
      q[i] += amp * std::sin(0.8 * k + 0.9 * i);
    }
    seq.push_back(q);
  }
  return seq;
}

}  // namespace

// G1. desk under +-3 bp hard-row moves: 126 rescales, 28 kink pins, 28 releases, no refresh. Every release of a pin is a degenerate
// rank-one update (d = 1 + beta s cancels at the 1e3 pin weight) and falls back to a full factorisation -- all 28 of them, which is ALL
// of the 224 allocations (8 per desk factorisation). That is today's cost, pinned; driving it down is a speed item, not this guard.
TEST(StreamingWalkGuard, ABandWalkIsPinnedByAllocationsAndRefactorisations) {
  const Shape& s = rung("desk");
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, x, s.q0, pinned());
  const std::vector<Eigen::VectorXd> seq = walk_sequence(s, 3e-4, 26);
  for (int k = 0; k < 6; ++k) st.update(seq[static_cast<std::size_t>(k)]);
  int refreshes = 0, rescales = 0, failed = 0;
  const int f0 = st.factor_count(), p0 = st.pin_count(), r0 = st.release_count();
  unsigned long allocs = 0;
  {
    AllocScope a;
    for (int k = 6; k < 26; ++k) {
      const cal::StreamTick t = st.update(seq[static_cast<std::size_t>(k)]);
      refreshes += t.refreshes;
      rescales += t.rescales;
      failed += !t.converged;
    }
    allocs = a.allocs();
  }
  const int factors = st.factor_count() - f0, pins = st.pin_count() - p0, releases = st.release_count() - r0;
  std::printf("  [walk] desk +-3 bp: allocs/20 %lu rescales %d refreshes %d pins %d releases %d factorisations %d\n", allocs, rescales,
              refreshes, pins, releases, factors);
  ASSERT_EQ(failed, 0) << "premise: the walk sequence converges on desk";
  ASSERT_GT(rescales, 0) << "premise: the sequence crosses band edges";
  ASSERT_GT(pins, 0) << "premise: the walk reaches kink optima";
  ASSERT_GT(releases, 0) << "premise: the KKT check releases pins";
  EXPECT_EQ(refreshes, 0) << "a band walk is rescales, never a Jacobian refresh";
  // Every factorisation in a refresh-free walk is rescale_row's fallback. Pinned at 28, and never more than one per release: a crossing
  // (switch_row / track_bands) that refactorises instead of updating rank-one would push it past `releases`.
  EXPECT_LE(factors, 28) << "the walk refactorised more often than its pin";
  EXPECT_LE(factors, releases) << "a crossing fell back to a full factorisation (only a pin release may)";
  if (swaps::testing::alloc_counting_available()) EXPECT_LE(allocs, 224u) << "the band walk allocates more than its pin";
}

// G1b. A FAILED tick that refreshed restores its anchor at the last committed state and market. streaming_contract_test no longer reaches
// this (its +500 % move diverges before any refresh, and its +800 bp round trip now converges): deleting the restore survived mutation
// (2026-09-15). portfolio is non-square, so a 25 bp move takes the drift refresh first; a 2-step cap then fails the tick after it.
// Restored: the next tick at the committed market sees zero drift and needs no refresh. Not restored: the anchor still carries the
// failed market, so that tick reports 25 bp of drift and refreshes at it.
TEST(StreamingWalkGuard, AFailedTickRestoresItsAnchorSoTheNextTickNeedsNoRefresh) {
  const Shape& s = rung("portfolio");
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  SC::Options o = pinned();
  o.max_steps = 2;
  SC st(eng, s.prob, x, s.q0, o);
  const cal::StreamTick t1 = st.update(s.q_big);
  ASSERT_FALSE(t1.converged) << "premise: a 2-step cap fails the 25 bp tick";
  ASSERT_GT(t1.refreshes, 0) << "premise: the failed tick refreshed (drift refresh) before failing";
  EXPECT_EQ((st.current() - x).cwiseAbs().maxCoeff(), 0.0) << "a failed tick must not commit";
  const cal::StreamTick t2 = st.update(s.q0);
  std::printf("  [fail] portfolio: failed tick %s after %d refreshes; next tick drift %.3e refreshes %d\n", t1.reason(), t1.refreshes, t2.drift,
              t2.refreshes);
  EXPECT_TRUE(t2.converged) << t2.reason();
  EXPECT_EQ(t2.drift, 0.0) << "the anchor still carries the failed tick's market";
  EXPECT_EQ(t2.refreshes, 0) << "the next tick refreshed off a stale anchor";
  EXPECT_LT((st.current() - x).cwiseAbs().maxCoeff(), 1e-12);
}

// G2a. A 25 bp move each way on the square, unbanded rungs converges in frozen steps alone: no refresh, no factorisation, and no
// allocation (a compiled rung) -- the RefreshTick25bp metric on these rungs times frozen steps, not a refresh. mixed_scheme's AAD block
// allocates 58.5 per tick (the hybrid-eval item).
TEST(StreamingWalkGuard, ALargeMoveOnASquareRungNeitherRefreshesNorAllocates) {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  for (const Shape& s : L) {
    if (s.prob.n_residuals() != s.prob.n_knots() || has_band(s)) continue;
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    SC st(eng, s.prob, x, s.q0, pinned());
    bool f = false;
    for (int i = 0; i < 2; ++i) { f = !f; st.update(f ? s.q_big : s.q0); }
    int refreshes = 0, failed = 0;
    const int f0 = st.factor_count();
    unsigned long allocs = 0;
    {
      AllocScope a;
      for (int i = 0; i < 10; ++i) {
        f = !f;
        const cal::StreamTick t = st.update(f ? s.q_big : s.q0);
        refreshes += t.refreshes;
        failed += !t.converged;
      }
      allocs = a.allocs();
    }
    std::printf("  [big] %-20s allocs/10 %lu refreshes %d factorisations %d\n", s.name.c_str(), allocs, refreshes, st.factor_count() - f0);
    EXPECT_EQ(failed, 0) << s.name;
    EXPECT_EQ(refreshes, 0) << s.name << ": a 25 bp square tick refreshed";
    EXPECT_EQ(st.factor_count() - f0, 0) << s.name;
    if (swaps::testing::alloc_counting_available()) {
      const unsigned long pin = s.name == "mixed_scheme" ? 585u : 0u;
      EXPECT_LE(allocs, pin) << s.name << ": a 25 bp square tick allocates more than its pin";
    }
  }
}

// G2b. With refresh_drift = 0 every small tick takes the drift refresh and the post-convergence final refresh: exactly 2 per tick, each
// exactly one factorisation, and no allocation beyond the factorisations' own (3 each on a compiled rung, 8 on desk -- the C6a refresh
// pins). A new allocation around an in-tick refresh (not inside the refresh) fails here.
TEST(StreamingWalkGuard, AnInTickRefreshCostsOnlyItsFactorisation) {
  struct Case { const char* name; unsigned long per_refresh; };
  for (const Case c : {Case{"banded", 3}, Case{"turns", 3}, Case{"portfolio", 3}, Case{"desk", 8}}) {
    const Shape& s = rung(c.name);
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    SC::Options o = pinned();
    o.refresh_drift = 0.0;
    SC st(eng, s.prob, x, s.q0, o);
    bool f = false;
    for (int i = 0; i < 4; ++i) { f = !f; st.update(f ? s.q_small : s.q0); }
    int refreshes = 0, failed = 0;
    const int f0 = st.factor_count();
    unsigned long allocs = 0;
    {
      AllocScope a;
      for (int i = 0; i < 10; ++i) {
        f = !f;
        const cal::StreamTick t = st.update(f ? s.q_small : s.q0);
        refreshes += t.refreshes;
        failed += !t.converged;
      }
      allocs = a.allocs();
    }
    const int factors = st.factor_count() - f0;
    std::printf("  [in-tick] %-10s allocs/10 %lu refreshes %d factorisations %d\n", c.name, allocs, refreshes, factors);
    EXPECT_EQ(failed, 0) << c.name;
    EXPECT_EQ(refreshes, 20) << c.name << ": a drift refresh plus one final refresh per tick";
    EXPECT_EQ(factors, refreshes) << c.name << ": each refresh is exactly one factorisation";
    if (swaps::testing::alloc_counting_available())
      EXPECT_LE(allocs, c.per_refresh * static_cast<unsigned long>(refreshes)) << c.name << ": an in-tick refresh allocates beyond its factorisation";
  }
}

// G2c. The regularised small tick (tension -- the SDK / web default) allocates nothing on a compiled rung.
TEST(StreamingWalkGuard, ARegularisedTickIsAllocationFree) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  for (const char* name : {"ois_nolag", "desk"}) {
    const Shape& s = rung(name);
    std::vector<int> curves(s.prob.curves.size());
    for (std::size_t c = 0; c < curves.size(); ++c) curves[c] = static_cast<int>(c);
    SC::Options o = pinned();
    o.regularizer = cal::tension_energy_operator(s.prob, 0.02, 1.0, curves);
    ASSERT_GT(o.regularizer.rows(), 0) << "premise: a non-empty regulariser";
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    SC st(eng, s.prob, x, s.q0, o);
    bool f = false;
    for (int i = 0; i < 6; ++i) { f = !f; st.update(f ? s.q_small : s.q0); }
    int failed = 0;
    unsigned long allocs = 0;
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) { f = !f; failed += !st.update(f ? s.q_small : s.q0).converged; }
      allocs = a.allocs();
    }
    std::printf("  [reg] %-10s allocs/20 %lu\n", name, allocs);
    EXPECT_EQ(failed, 0) << name;
    EXPECT_EQ(allocs, 0u) << name << ": a regularised tick allocated";
  }
}

// G3. The session ladder pins (hotpath_shapes_test T4) on the streamer itself: 20 small ticks and 20 four-number requote ticks per rung.
// Compiled rungs 0; mixed_scheme 520 and desk_mixed 4530 (at the pinned break-even; the session's measured break-even gives 3380) are the
// hybrid-eval allocations.
TEST(StreamingWalkGuard, TheStreamerLadderTickPins) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  struct Pin { const char* name; unsigned long ticks; };
  static const Pin pins[] = {{"mixed_scheme", 520}, {"desk_mixed", 4070}};  // desk_mixed 4530 -> 4070 (S2, 2026-09-15)
  static const std::vector<Shape> L = swaps::shapes::ladder();
  for (const Shape& s : L) {
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    SC st(eng, s.prob, x, s.q0, pinned());
    bool f = false;
    for (int i = 0; i < 6; ++i) { f = !f; st.update(f ? s.q_small : s.q0); }
    unsigned long small = 0, requote = 0;
    int failed = 0;
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) { f = !f; failed += !st.update(f ? s.q_small : s.q0).converged; }
      small = a.allocs();
    }
    const Shape::Requote rs = s.requote(s.q_small), rb = s.requote(s.q0);
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) {
        f = !f;
        const Shape::Requote& r = f ? rs : rb;
        if (s.has_bands) {
          for (int j = 0; j < s.prob.n_residuals(); ++j) eng.set_quote(j, r.target[j], r.lower[j], r.upper[j], r.decay[j]);
          if (!st.set_bands(r.lower, r.upper, r.decay)) ++failed;  // a pure band move never re-anchors
        }
        failed += !st.update(r.target).converged;
      }
      requote = a.allocs();
    }
    unsigned long pin = 0;
    for (const Pin& p : pins)
      if (s.name == p.name) pin = p.ticks;
    std::printf("  [ladder] %-20s small allocs/20 %lu requote allocs/20 %lu (pin %lu)\n", s.name.c_str(), small, requote, pin);
    EXPECT_EQ(failed, 0) << s.name;
    EXPECT_LE(small, pin) << s.name << ": 20 small streamer ticks allocate more than the pin";
    EXPECT_LE(requote, pin) << s.name << ": 20 four-number requote ticks allocate more than the pin";
  }
}
