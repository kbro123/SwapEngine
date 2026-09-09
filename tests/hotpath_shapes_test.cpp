// @regression-test — the SHAPE LADDER (bench/fixtures/shape_ladder.hpp): for EVERY instrument shape the compiled /
// hybrid hot path accepts, (T3) the hybrid engine's residual and analytic Jacobian match the templated kernel and
// the AAD Jacobian, and (T4) the streaming tick is allocation-free after warm-up on every W-cacheable shape, with
// the hybrid (FX/MtM) shapes and band-edge crossings pinned at their CURRENT counts so they can only go down.
// Until 2026-09-09 the gate's "0 allocations per tick" was proven on annual OIS only (E3 review A3/B3/C7).
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "malloc_count.hpp"
#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace cal = swaps::calibration;
namespace api = swaps::api;
using swaps::shapes::Shape;
using swaps::testing::AllocScope;

namespace {
double rel(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).cwiseAbs().maxCoeff() / (b.cwiseAbs().maxCoeff() + 1e-300);
}
}  // namespace

// T3: compiled/hybrid == templated == AAD on every rung, at x_true and off it.
TEST(ShapeLadder, HybridResidualAndJacobianMatchTemplatedAndAadOnEveryShape) {
  for (const Shape& s : swaps::shapes::ladder()) {
    const cal::HybridBundleResidual h(s.prob);
    double worst_r = 0, worst_j = 0;
    for (double bump : {0.0, 7e-4, -1.3e-3}) {
      Eigen::VectorXd x = s.x_true;
      for (int i = 0; i < x.size(); ++i) x[i] += bump * std::sin(0.9 * i + 0.2);
      worst_r = std::max(worst_r, (h.residuals(x) - s.prob.residuals<double>(x)).cwiseAbs().maxCoeff());
      worst_j = std::max(worst_j, rel(h.jacobian(x), cal::aad_jacobian(s.prob, x)));
    }
    std::cout << "  [ladder] " << s.name << ": " << s.prob.n_residuals() << " rows, " << s.prob.n_knots()
              << " knots, |dr|=" << worst_r << " |dJ|rel=" << worst_j << (s.expect_compiled ? "" : "  (hybrid route)") << "\n";
    EXPECT_LT(worst_r, 1e-12) << s.name;
    EXPECT_LT(worst_j, 1e-9) << s.name;
    EXPECT_LT(s.prob.residuals<double>(s.x_true).cwiseAbs().maxCoeff(), 1e-12) << s.name << ": markets must be the model quotes at x_true";
  }
}

// T4: after warm-up, a streaming tick allocates NOTHING on every W-cacheable shape. Hybrid shapes (FX/MtM) and
// band-edge crossings are pinned at their current counts (E3-B3: Hermite::build locals; E3-C7: a band re-scale
// is a full factor()) — they may only decrease; driving them to 0 is E4.D.
TEST(ShapeLadder, StreamingTickIsAllocationFreeOnEveryCompiledShape) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  for (const Shape& s : swaps::shapes::ladder()) {
    api::BundleSession sess(s.prob);
    sess.calibrate(s.x0);
    ASSERT_LT(sess.result().rms_residual, 1e-8) << s.name << ": cold calibrate must converge";
    sess.start_streaming();
    bool flip = false;
    for (int i = 0; i < 6; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
    unsigned long small = 0, cross = 0;
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
      small = a.allocs();
    }
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) { flip = !flip; sess.stream_update(flip ? s.q_cross : s.q0); }
      cross = a.allocs();
    }
    std::cout << "  [ladder] " << s.name << ": allocs over 20 small ticks = " << small << ", over 20 band-crossing ticks = " << cross << "\n";
    // PINS measured 2026-09-09 on engine 540abaf (20 ticks each). A compiled shape's small tick is 0 — that is the
    // invariant. The rest are the CURRENT costs of known E3 findings (B3 hybrid Hermite::build locals; C7 a band
    // re-scale is a full factor()); they may only DECREASE — lower a pin when you fix the cause, never raise one.
    struct Pin { const char* name; unsigned long small, cross; };
    //   averaged_leg (compiled, 18/tick = 6 per residual x 3 Newton steps): E3-A3, Eigen IndexedView copies on
    //   pv()'s general branch — the one compiled shape that is NOT allocation-free today.
    static const Pin pins[] = {{"averaged_leg", 360, 360}, {"banded", 0, 1800}, {"fx_xccy", 630, 630}, {"desk", 1120, 4000}};
    //   (banded / desk crossing pins carry a few % of slack: the count of refreshes 20 crossing ticks trigger moves with
    //    rounding when the kernel's summation order changes — 1733 sparse, 1757 segment.)
    Pin pin{s.name.c_str(), 0, 0};
    bool pinned = false;
    for (const auto& q : pins) if (s.name == q.name) { pin = q; pinned = true; }
    if (s.expect_compiled && !pinned) EXPECT_EQ(small, 0u) << s.name << ": the compiled tick must not allocate";
    else EXPECT_LE(small, pin.small) << s.name << ": tick allocations grew past the pinned count";
    EXPECT_LE(cross, pin.cross) << s.name << ": band-crossing / hybrid tick allocations grew past the pinned count";
  }
}

// The moment path vs the exact daily path on the SAME FF OIS legs: the documented ~5e-9 relative approximation
// (docs/bezier-and-moments.md Part B) must hold on real 1Y windows — the model par rates of the two rungs at the
// shared x_true agree to 2e-8 (0.0002 bp) while the moment rung ticks ~100x faster (shape_ladder_bench).
TEST(ShapeLadder, MomentPathAgreesWithTheExactDailyAverageOnFedFundsOis) {
  const Shape daily = swaps::shapes::averaged_leg(), moment = swaps::shapes::averaged_leg_moment();
  ASSERT_EQ(daily.prob.n_residuals(), moment.prob.n_residuals());
  double worst = 0.0;
  for (int i = 0; i < daily.prob.n_residuals(); ++i)
    worst = std::max(worst, std::abs(daily.q0[i] - moment.q0[i]));
  std::cout << "  [ladder] moment vs exact daily FF OIS par rates: max |dq| = " << worst << " (" << worst * 1e4 << " bp)\n";
  EXPECT_LT(worst, 2e-8);
  // And the moment rung is a pure W-cache bundle: the hybrid engine must not have routed any row to AAD.
  EXPECT_NO_THROW(cal::CompiledBundleResidual{moment.prob});
}
