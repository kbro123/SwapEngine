// E5 taxonomy: T3 cross-path parity (a refresh's band sides from the Jacobian pass's residuals vs from re-priced model quotes; bit-identical)
// STEP 4 / S2 (2026-09-15). Options::anchor_sides_from_jacobian = false keeps the reference -- every model quote re-priced after the Jacobian --
// and this test drives both through the same markets on every banded ladder rung: small ticks, the +-3 bp hard-row walk (band crossings, pins,
// releases) and 25 bp four-number requotes. The committed state must be BIT-IDENTICAL after every tick, with the same refreshes, rescales,
// stages and convergence: the residual inversion classifies exactly as the model quote does (probe s4: 0 mismatches over 30 ticks x 4 rungs).
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

namespace {
SC::Options opts(bool from_jacobian) {
  SC::Options o;
  o.breakeven_steps = 64;
  o.anchor_sides_from_jacobian = from_jacobian;
  return o;
}
}  // namespace

TEST(StreamingAnchorSidesParity, BandSidesFromTheJacobianPassAreBitIdenticalToRepricedQuotes) {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  int rungs = 0;
  for (const Shape& s : L) {
    bool banded = false;
    for (const auto& in : s.prob.instruments) banded = banded || in.band_upper > in.band_lower;
    if (!banded) continue;
    ++rungs;
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual e_ref(s.prob), e_new(s.prob);
    SC ref(e_ref, s.prob, x, s.q0, opts(false)), now(e_new, s.prob, x, s.q0, opts(true));
    int tick = 0;
    auto drive = [&](const char* what, const Eigen::VectorXd& q) {
      const cal::StreamTick a = ref.update(q), b = now.update(q);
      ++tick;
      EXPECT_EQ((ref.current() - now.current()).cwiseAbs().maxCoeff(), 0.0) << s.name << " " << what << " tick " << tick;
      EXPECT_EQ(a.refreshes, b.refreshes) << s.name << " " << what << " tick " << tick;
      EXPECT_EQ(a.rescales, b.rescales) << s.name << " " << what << " tick " << tick;
      EXPECT_EQ(a.stages, b.stages) << s.name << " " << what << " tick " << tick;
      EXPECT_EQ(a.converged, b.converged) << s.name << " " << what << " tick " << tick;
    };
    for (int i = 0; i < 6; ++i) drive("small", i % 2 == 0 ? s.q_small : s.q0);
    for (int k = 0; k < 20; ++k) {
      Eigen::VectorXd q = s.q0;
      for (int i = 0; i < q.size(); ++i) {
        const auto& in = s.prob.instruments[static_cast<std::size_t>(i)];
        if (in.band_upper > in.band_lower || in.quote == cal::QuoteKind::FxForward || in.quote == cal::QuoteKind::TurnJump) continue;
        q[i] += 3e-4 * std::sin(0.8 * k + 0.9 * i);
      }
      drive("walk", q);
    }
    const Shape::Requote big = s.requote(s.q_big), base = s.requote(s.q0);
    for (int i = 0; i < 4; ++i) {
      const Shape::Requote& r = i % 2 == 0 ? big : base;
      for (int j = 0; j < s.prob.n_residuals(); ++j) {
        e_ref.set_quote(j, r.target[j], r.lower[j], r.upper[j], r.decay[j]);
        e_new.set_quote(j, r.target[j], r.lower[j], r.upper[j], r.decay[j]);
      }
      ASSERT_TRUE(ref.set_bands(r.lower, r.upper, r.decay) && now.set_bands(r.lower, r.upper, r.decay)) << s.name;
      drive("requote25", r.target);
    }
  }
  EXPECT_EQ(rungs, 4) << "premise: banded, turns, desk and desk_mixed";
}

// The Jacobian pass's residuals ARE the engine's residuals_vs(x, q), on every ladder rung and engine route (compiled only; hybrid = cacheable rows +
// AAD rows; banded or not). r starts NaN, so a row the pass does not write fails. The side test above cannot see a row that stops being written: a
// streamer refresh then reads the previous refresh's residuals, which classify the same sides almost always (that mutation SURVIVED until this).
TEST(StreamingAnchorSidesParity, TheJacobianPassReturnsTheEnginesResidualsOnEveryRung) {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  for (const Shape& s : L) {
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    const Eigen::VectorXd xp = x.array() + 1e-6;  // a state no residual has been evaluated at
    cal::HybridBundleResidual eng(s.prob);
    Eigen::MatrixXd J, J3;
    Eigen::VectorXd r = Eigen::VectorXd::Constant(s.prob.n_residuals(), std::numeric_limits<double>::quiet_NaN());
    eng.jacobian_vs_into(xp, s.q_small, J, &r);
    ASSERT_TRUE(r.allFinite()) << s.name << ": a row's residual was not written by the Jacobian pass";
    const Eigen::VectorXd ref = eng.residuals_vs(xp, s.q_small);
    EXPECT_LT((r - ref).cwiseAbs().maxCoeff(), 1e-12) << s.name;
    eng.jacobian_vs_into(xp, s.q_small, J3);
    EXPECT_EQ((J - J3).cwiseAbs().maxCoeff(), 0.0) << s.name << ": returning the residuals changed the Jacobian";
  }
}
