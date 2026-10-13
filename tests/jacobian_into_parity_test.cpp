// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// C6 (2026-09-15): jacobian_vs_into(x, q, J) -- the streamer's in-place refresh -- writes EXACTLY jacobian_vs(x, q): every entry, on every
// ladder rung (compiled and hybrid), into a pre-sized matrix full of NaN (a path that skips zeroing or leaves a block untouched shows
// up as a NaN) and into a wrongly shaped one (it must resize).
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;

TEST(JacobianIntoParity, IntoANanFilledOrMisshapedMatrixEqualsTheByValueJacobianOnEveryRung) {
  for (const Shape& s : swaps::shapes::ladder()) {
    const cal::HybridBundleResidual eng(s.prob);
    for (const Eigen::VectorXd* q : {&s.q0, &s.q_big}) {
      const Eigen::MatrixXd ref = eng.jacobian_vs(s.x_true, *q);
      Eigen::MatrixXd J = Eigen::MatrixXd::Constant(ref.rows(), ref.cols(), std::nan(""));
      eng.jacobian_vs_into(s.x_true, *q, J);
      EXPECT_TRUE((J.array() == ref.array()).all()) << s.name << ": into a NaN-filled matrix";
      Eigen::MatrixXd K(1, 1);
      eng.jacobian_vs_into(s.x_true, *q, K);
      ASSERT_EQ(K.rows(), ref.rows()) << s.name;
      ASSERT_EQ(K.cols(), ref.cols()) << s.name;
      EXPECT_TRUE((K.array() == ref.array()).all()) << s.name << ": into a misshaped matrix";
    }
  }
}

// The regularised refresh factors the stacked [J; R] whose R rows are written ONCE at construction (C6). A regularised stream lands on the
// regularised least-squares solution: the tension-regularised ois_nolag rung, streamed q0 -> q_small -> q_big -> q0, against a cold LM on
// the same composed residual (instrument rows + R x) from the curve's seed.
TEST(JacobianIntoParity, ARegularisedStreamLandsOnTheRegularisedColdSolve) {
  for (const Shape& s : swaps::shapes::ladder()) {
    if (s.name != "ois_nolag") continue;
    std::vector<int> curves(s.prob.curves.size());
    for (std::size_t c = 0; c < curves.size(); ++c) curves[c] = static_cast<int>(c);
    // The regulariser is CONSTRAINT ROWS of the problem (2026-10-12): the streamer and the cold solve read the same rows.
    cal::BundleProblem pr = s.prob;
    cal::RegSpec reg;
    reg.lambda = 0.02;
    reg.curves = curves;
    cal::stamp_region_smoothing(pr, reg.lambda, reg.curves);  // smoothing is region data
  cal::add_smoothing_rows(pr);
    ASSERT_GT(pr.n_constraints(), 0) << "premise: a non-empty regulariser";
    const auto full = [&](const Eigen::VectorXd& q) {  // the streamer's market is every row's target: the quotes, then the rows' 0
      Eigen::VectorXd f = pr.market();
      f.head(q.size()) = q;
      return f;
    };
    const auto cold = [&](const Eigen::VectorXd& q) {
      cal::BundleProblem p = pr;
      for (int i = 0; i < q.size(); ++i) p.instruments[static_cast<std::size_t>(i)].market = q[i];
      const cal::HybridBundleResidual eng(p);
      return cal::calibrate_with(eng, p.n_knots(), p.n_residuals(), s.x0).x;
    };
    cal::StreamingCalibrator<cal::BundleProblem>::Options o;
    o.breakeven_steps = 64;
    cal::StreamingCalibrator<cal::BundleProblem> st(pr, cold(s.q0), full(s.q0), o);
    for (const Eigen::VectorXd* q : {&s.q_small, &s.q_big, &s.q0}) {
      const cal::StreamTick t = st.update(full(*q));
      ASSERT_TRUE(t.converged) << t.reason();
      EXPECT_LT((st.current() - cold(*q)).cwiseAbs().maxCoeff(), 1e-8) << "the regularised stream is not the regularised solution";
    }
  }
}
