// E5 taxonomy: T6 regression (repro-first: fails on 6b26213, where a tick that re-anchored to commit also took the final refresh) | T4 refresh count
// STEP 4 / S3 (hot-path audit A2, 2026-09-15). A tick that took the drift refresh re-anchors once more after converging (the FINAL refresh: J at
// the converged point, so the committed answer is exact to first order). A tick whose walk converged on a TRUNCATED operator re-anchors at
// full rank before committing (the C1 commit rule) -- also a Jacobian at a converged point of this tick's market. When both applied, the tick
// took them one after the other, at points one re-convergence apart: probe on 6b26213, the weak-direction strip with refresh_drift = 0,
// 3 refreshes (drift + commit + final) and 5 factorisations, where the commit re-anchor already gives the final refresh's guarantee.
#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace px = swaps::pricing;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

namespace {

cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}

Eigen::VectorXd model_quotes(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = cal::instrument_model_quote<double>(p.instruments[i], cof);
  return q;
}

struct Strip {
  cal::BundleProblem p, p1;
  Eigen::VectorXd x0, q, q1, xc;
};

// The C1 fixture: a square 1y/2y/3y strip, the 2y row banded +-5 bp with decay 1e-5, anchored OUTSIDE the band (knot 2 lifted 30 bp) and
// solved inside it, so the walk meets a weak direction the anchor did not have and truncates it.
Strip strip() {
  Strip s;
  s.p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3})});
  for (int T = 1; T <= 3; ++T) s.p.instruments.push_back(par_swap(T));
  Eigen::VectorXd xs(3);
  xs << 0.040, 0.042, 0.044;
  s.q = model_quotes(s.p, xs);
  for (int i = 0; i < 3; ++i) s.p.instruments[static_cast<std::size_t>(i)].market = s.q[i];
  s.p.instruments[1].band_lower = s.q[1] - 5e-4;
  s.p.instruments[1].band_upper = s.q[1] + 5e-4;
  s.p.instruments[1].band_decay = 1e-5;
  s.x0 = xs;
  s.x0[1] += 30e-4;
  s.q1 = s.q;
  s.q1[0] += 0.5e-4;
  s.q1[2] -= 0.5e-4;
  s.p1 = s.p;
  for (int i = 0; i < 3; ++i) s.p1.instruments[static_cast<std::size_t>(i)].market = s.q1[i];
  s.xc = cal::calibrate(s.p1, xs).x;
  return s;
}

bool reached(unsigned stages, cal::StreamStage st) { return stages & (1u << static_cast<unsigned>(st)); }

}  // namespace

TEST(StreamingCommitReanchorRepro, ACommitReanchorIsTheTicksFinalRefresh) {
  const Strip s = strip();
  SC::Options o;
  o.breakeven_steps = 64;
  o.refresh_drift = 0.0;  // the tick takes the drift refresh, so the final refresh applies
  SC st(s.p, s.x0, s.q, o);
  const int f0 = st.factor_count();
  const cal::StreamTick t = st.update(s.q1);
  ASSERT_TRUE(t.converged) << t.reason();
  ASSERT_TRUE(reached(t.stages, cal::StreamStage::DriftRefresh)) << "premise: the tick took the drift refresh";
  ASSERT_TRUE(reached(t.stages, cal::StreamStage::CommitReanchor)) << "premise: the walk truncated and the commit rule re-anchored";
  EXPECT_FALSE(reached(t.stages, cal::StreamStage::FinalRefresh)) << "the final refresh repeated the commit re-anchor";
  EXPECT_EQ(t.refreshes, 2) << "drift refresh + commit re-anchor (6b26213 took a third, the final refresh)";
  EXPECT_EQ(st.factor_count() - f0, 4) << "2 refreshes + the walk's 2 rescale fallbacks (6b26213: 5)";
  // Accuracy: the committed knots are 1.5e-11 from the cold solution -- exactly what the SAME tick commits without a drift refresh (the control
  // below; streaming_near_singular_repro_test pins it at 1e-9), inside step_tol (1e-9). The final refresh this removes had bought 1.5e-16:
  // extra accuracy below the streamer's contract, at the price of a Jacobian.
  EXPECT_LT((st.current() - s.xc).cwiseAbs().maxCoeff(), 1e-10) << "the committed knots are the cold solution to the step_tol contract";
  EXPECT_LT(std::abs(model_quotes(s.p1, st.current())[1] - s.q1[1]), 1e-10) << "the 2y quote is at its target";
}

// Without the drift refresh the tick never took the final refresh: unchanged (1 refresh, the commit re-anchor).
TEST(StreamingCommitReanchorRepro, WithoutADriftRefreshTheCommitReanchorIsTheOnlyRefresh) {
  const Strip s = strip();
  SC::Options o;
  o.breakeven_steps = 64;
  SC st(s.p, s.x0, s.q, o);
  const cal::StreamTick t = st.update(s.q1);
  ASSERT_TRUE(t.converged) << t.reason();
  EXPECT_EQ(t.refreshes, 1);
  EXPECT_LT((st.current() - s.xc).cwiseAbs().maxCoeff(), 1e-9);
}
