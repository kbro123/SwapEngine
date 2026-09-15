// E5 taxonomy: T6 regression (fails on the reverted bug)
// C8 REPRODUCTION (E3-C review 2026-09-09; reproduced 2026-09-15). StreamTick::drift -- and BundleSession::last_drift(), which
// copies it -- read 0 on exactly the ticks that drifted most: a tick whose market move exceeded Options::refresh_drift refreshed
// its Jacobian and then zeroed the drift it had just measured (streaming.hpp update_exact). The drift is the market move since the
// anchor the tick started from; StreamTick::refreshed says whether the tick re-anchored.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;

TEST(StreamingDriftReportRepro, ATickThatRefreshedOnItsDriftReportsThatDrift) {
  const Shape s = swaps::shapes::banded();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  const Shape::Requote r = s.requote(s.q_big);  // the ladder's 25 bp move, every band moved with its target
  for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, r.target[i], r.lower[i], r.upper[i], r.decay[i]);
  ASSERT_TRUE(st.set_bands(r.lower, r.upper, r.decay));
  const double moved = (r.target - s.q0).cwiseAbs().maxCoeff();
  ASSERT_GT(moved, cal::StreamingCalibrator<cal::BundleProblem>::Options{}.refresh_drift) << "premise: the move exceeds refresh_drift";

  const cal::StreamTick t = st.update(r.target);
  ASSERT_TRUE(t.converged) << t.reason();
  ASSERT_TRUE(t.refreshed) << "premise: the drift triggered the accuracy refresh";
  EXPECT_DOUBLE_EQ(t.drift, moved) << "a tick reports the market move it started from, also when that move refreshed it";

  // The next tick measures from the anchor the refresh set: a 0.01 bp move reads as 0.01 bp.
  Eigen::VectorXd q2 = r.target;
  q2[0] += 1e-6;
  const cal::StreamTick t2 = st.update(q2);
  ASSERT_TRUE(t2.converged) << t2.reason();
  EXPECT_FALSE(t2.refreshed);
  EXPECT_NEAR(t2.drift, 1e-6, 1e-12);
}
