// E5 taxonomy: T4 hot-path invariant (allocation pin) | T6 regression (fails on the reverted bug)
// C6 REPRODUCTION (E3-C review 2026-09-09; measured 2026-09-15 on 8323778). A Jacobian REFRESH allocated its whole working set every
// time: the engine returned J by value and factor() built a fresh CompleteOrthogonalDecomposition (a copy of J plus its QR vectors).
// A warm refresh (StreamingCalibrator::resync) on the compiled ladder rungs cost 13 allocations (1 Jacobian + 12 decomposition), 18 on
// desk, 27 on mixed_scheme, 65 on desk_mixed, 24 / 29 regularised. The streamer now writes J in place and reuses one decomposition sized
// at construction. What remains per refresh is the solve's materialised identity, its Householder workspace and a permutation buffer
// (C6's next step), the regularised pseudo-inverse's own products, and the AAD block's curve rebuilds on the hybrid rungs. These pins
// may only DECREASE.
#include <cstdio>
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

namespace {
// Allocations of ONE warm refresh (the second resync at the same state: every workspace already sized).
unsigned long refresh_allocs(const Shape& s, const SC::Options& o) {
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, x, s.q0, o);
  st.resync(s.prob, st.current(), s.q0);
  swaps::testing::AllocScope a;
  st.resync(s.prob, st.current(), s.q0);
  return a.allocs();
}
}  // namespace

TEST(StreamingRefreshAllocRepro, ARefreshReusesItsJacobianAndDecompositionStorage) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  struct Pin { const char* name; unsigned long refresh; };
  // Measured 2026-09-15 after the change (was, on 8323778: 13 on every compiled rung, desk 18, mixed_scheme 27, desk_mixed 65).
  // desk_mixed 54 -> 31 on 2026-09-15 (S2): a refresh reads its band sides from the Jacobian pass instead of re-pricing every model quote.
  static const Pin pins[] = {{"desk", 8}, {"mixed_scheme", 16}, {"desk_mixed", 31}};
  SC::Options o;
  o.breakeven_steps = 64;
  for (const Shape& s : swaps::shapes::ladder()) {
    unsigned long pin = 3;  // a compiled rung below 48 knots: identity rhs + one Householder workspace + the permutation buffer
    for (const auto& p : pins)
      if (s.name == p.name) pin = p.refresh;
    const unsigned long n = refresh_allocs(s, o);
    std::printf("  [refresh] %-20s %3dx%-3d allocs %lu (pin %lu)\n", s.name.c_str(), s.prob.n_residuals(), s.prob.n_knots(), n, pin);
    EXPECT_LE(n, pin) << s.name << ": a refresh allocates more than its pin";
  }
}

TEST(StreamingRefreshAllocRepro, ARegularisedRefreshReusesItsStackedMatrix) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  // Measured 2026-09-15 after the change (was 24 / 29): the pseudo-inverse P and the G / B products remain.
  struct Pin { const char* name; unsigned long refresh; };
  static const Pin pins[] = {{"ois_nolag", 13}, {"desk", 18}};
  for (const Shape& s : swaps::shapes::ladder()) {
    for (const auto& p : pins) {
      if (s.name != p.name) continue;
      std::vector<int> curves(s.prob.curves.size());
      for (std::size_t c = 0; c < curves.size(); ++c) curves[c] = static_cast<int>(c);
      SC::Options o;
      o.breakeven_steps = 64;
      o.regularizer = cal::second_difference_operator(s.prob, 0.02, curves);
      ASSERT_GT(o.regularizer.rows(), 0) << "premise: a non-empty regulariser";
      const unsigned long n = refresh_allocs(s, o);
      std::printf("  [refresh] %-20s regularised allocs %lu (pin %lu)\n", s.name.c_str(), n, p.refresh);
      EXPECT_LE(n, p.refresh) << s.name << ": a regularised refresh allocates more than its pin";
    }
  }
}
