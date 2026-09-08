// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference market; the
// engine is compared to ITSELF / hand formulas, not to a QuantLib number). DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Stage-2 gate: cached-Jacobian warm re-calibration must match the full LM re-solve for realistic
// market perturbations. The WarmCalibrator freezes J0 at the base solution and takes a couple of
// Gauss-Newton steps; here we prove that lands on the same curve as a from-scratch LM.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/warm.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

namespace {

// Apply a market perturbation dq to a problem. Instrument insertion order IS the residual order
// (avg futures, comp futures, swaps).
cal::CalibrationProblem bumped_market(cal::CalibrationProblem p, const Eigen::VectorXd& dq) {
  for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i) p.instruments[i].market += dq[i];
  return p;
}

}  // namespace

struct Warm : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x0;

  void SetUp() override {
    x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  }

  // Deterministic perturbation with a given peak size (bp), varied sign/shape across quotes.
  Eigen::VectorXd perturbation(double peak) const {
    Eigen::VectorXd dq(prob.n_residuals());
    for (int i = 0; i < dq.size(); ++i) dq[i] = peak * std::sin(0.7 * i + 1.0);
    return dq;
  }
};

TEST_F(Warm, AdaptiveMatchesFullLmAndDetectsEnvelope) {
  const cal::WarmCalibrator wc(prob, x0);

  // From 0.5bp (deep inside envelope) to 25bp (well outside).
  for (double peak : {0.00005, 0.0002, 0.0010, 0.0025}) {
    const Eigen::VectorXd dq = perturbation(peak);
    const Eigen::VectorXd x_full = cal::calibrate(bumped_market(prob, dq), x0, true).x;

    const cal::WarmResult r = wc.recalibrate(dq);
    std::cout << "  [warm] peak=" << peak * 1e4 << "bp  steps=" << r.steps
              << " refreshes=" << r.jacobian_refreshes << " converged=" << r.converged
              << " ||x-x_full||_inf=" << (r.x - x_full).cwiseAbs().maxCoeff() << "\n";

    // However big the move, the adaptive path lands on the full-LM solution.
    EXPECT_TRUE(r.converged) << " at peak " << peak;
    EXPECT_LT((r.x - x_full).cwiseAbs().maxCoeff(), 1e-7) << " at peak " << peak;
  }
}

TEST_F(Warm, SmallMovesStayInEnvelopeLargeMovesRefresh) {
  const cal::WarmCalibrator wc(prob, x0);
  // Small move: fast path, no Jacobian refresh.
  const auto small = wc.recalibrate(perturbation(0.00005));  // 0.5bp
  EXPECT_EQ(small.jacobian_refreshes, 0) << "a sub-bp move must stay on the cached-Jacobian fast path";
  // Large move: auto-detected as outside the envelope -> at least one refresh.
  const auto large = wc.recalibrate(perturbation(0.0025));   // 25bp
  EXPECT_GT(large.jacobian_refreshes, 0) << "a 25bp move must trigger an automatic Jacobian refresh";
  EXPECT_TRUE(large.converged);
}

TEST_F(Warm, LinearUpdateIsFirstOrderAccurate) {
  const cal::WarmCalibrator wc(prob, x0);
  // x = x0 + M*dq is the first frozen Newton step (r(x0) = -dq exactly), so its error vs the full LM
  // is O(|dq|^2): halving the move should quarter the error.
  double e_small = 0, e_big = 0;
  for (double peak : {0.00005, 0.0001}) {
    const Eigen::VectorXd dq = perturbation(peak);
    const Eigen::VectorXd xl = wc.recalibrate_linear(dq);
    const Eigen::VectorXd xf = cal::calibrate(bumped_market(prob, dq), x0, true).x;
    const double err = (xl - xf).cwiseAbs().maxCoeff();
    (peak < 0.00008 ? e_small : e_big) = err;
    EXPECT_LT(err, 1e-5) << " linear update should be ~1e-6 accurate for a " << peak * 1e4 << "bp move";
  }
  // quadratic scaling: doubling the move (0.5->1bp) roughly quadruples the error
  EXPECT_GT(e_big / e_small, 2.5);
  EXPECT_LT(e_big / e_small, 6.0);
  std::cout << "  [warm-linear] err(0.5bp)=" << e_small << " err(1bp)=" << e_big
            << " ratio=" << e_big / e_small << " (expect ~4, quadratic)\n";
}

TEST_F(Warm, ZeroPerturbationIsAFixedPoint) {
  const cal::WarmCalibrator wc(prob, x0);
  const cal::WarmResult r = wc.recalibrate(Eigen::VectorXd::Zero(prob.n_residuals()));
  EXPECT_LT((r.x - x0).cwiseAbs().maxCoeff(), 1e-12) << "no market move must return the base curve";
  EXPECT_EQ(r.jacobian_refreshes, 0);
  // and the fixed-step fast path agrees
  const Eigen::VectorXd xf = wc.recalibrate_fixed(Eigen::VectorXd::Zero(prob.n_residuals()), 2);
  EXPECT_LT((xf - x0).cwiseAbs().maxCoeff(), 1e-12);
}
