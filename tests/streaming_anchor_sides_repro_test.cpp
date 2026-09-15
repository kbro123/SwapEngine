// E5 taxonomy: T6 regression (repro-first: fails on the S3 commit, where a refresh re-priced every model quote for its band sides) | T4 allocation pin
// STEP 4 / S2 (hot-path audit A2, 2026-09-15). A refresh classifies each tracked band row -- inside, above or below its band -- to set the row's
// slope at the anchor. It did so by re-evaluating EVERY model quote after the Jacobian: on desk_mixed that rebuilt every AAD curve, 192 us and 23
// allocations of a 1.2 ms refresh (probe s4), repeating what the Jacobian pass had just computed. The Jacobian pass now also returns its residual
// values, and the side is read from them exactly as the tick's own side_of does.
#include <cstdio>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "malloc_count.hpp"  // bench/fixtures (an include dir of swaps_tests)
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

namespace {
// Allocations of ONE warm refresh (the second resync at the same state), break-even pinned: the refresh_allocs protocol.
unsigned long refresh_allocs(const Shape& s) {
  SC::Options o;
  o.breakeven_steps = 64;
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, x, s.q0, o);
  st.resync(s.prob, st.current(), s.q0);
  swaps::testing::AllocScope a;
  st.resync(s.prob, st.current(), s.q0);
  return a.allocs();
}
}  // namespace

TEST(StreamingAnchorSidesRepro, ARefreshReadsItsBandSidesWithoutRepricingEveryQuote) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  // desk_mixed: banded rows on the AAD block -- the re-evaluation rebuilt every AAD curve (54 allocations per refresh before, 31 after).
  const unsigned long dm = refresh_allocs(swaps::shapes::desk_mixed());
  // desk: banded, all rows compiled -- their re-pricing never allocated (8 before and after): the control.
  const unsigned long dk = refresh_allocs(swaps::shapes::desk());
  std::printf("  [anchor-sides] refresh allocations: desk_mixed %lu, desk %lu\n", dm, dk);
  EXPECT_LE(dm, 31u) << "desk_mixed: a refresh re-priced every model quote to read its band sides";
  EXPECT_LE(dk, 8u);
}
