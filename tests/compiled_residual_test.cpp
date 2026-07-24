// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Stage-2 gate: the vectorized CompiledResidual must reproduce the scalar calibration residual
// (CalibrationProblem::residuals) exactly, for any knot forwards x -- it is the same math on the
// shared DF engine, so the only differences allowed are rounding.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include "reference_curve.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

struct Compiled : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
};

TEST_F(Compiled, ResidualMatchesScalarKernel) {
  const cal::CompiledResidual cr(prob);
  ASSERT_EQ(cr.n_residuals(), prob.n_residuals());

  std::vector<Eigen::VectorXd> curves;
  curves.push_back(Eigen::VectorXd::Constant(prob.n_knots(), 0.030));
  curves.push_back(Eigen::VectorXd::Constant(prob.n_knots(), 0.045));
  curves.push_back(cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x);
  Eigen::VectorXd tilt(prob.n_knots());
  for (int i = 0; i < tilt.size(); ++i) tilt[i] = 0.03 + 0.0004 * i;
  curves.push_back(tilt);

  double worst = 0.0;
  for (const auto& x : curves) {
    const Eigen::VectorXd scal = prob.residuals<double>(x);
    const Eigen::VectorXd vec = cr.residuals(x);
    worst = std::max(worst, (scal - vec).cwiseAbs().maxCoeff());
  }
  std::cout << "  [compiled residual] distinct times=" << cr.n_times()
            << " worst |vec - scalar| = " << worst << "\n";
  EXPECT_LT(worst, 1e-12);
}

TEST_F(Compiled, AnalyticJacobianMatchesAad) {
  const cal::CompiledResidual cr(prob);
  double worst_abs = 0, scale = 0;
  for (double lvl : {0.030, 0.040}) {
    const Eigen::VectorXd x = Eigen::VectorXd::Constant(prob.n_knots(), lvl);
    const Eigen::MatrixXd Ja = cr.jacobian(x);
    const Eigen::MatrixXd Jaad = cal::aad_jacobian(prob, x);
    ASSERT_EQ(Ja.rows(), prob.n_residuals());
    ASSERT_EQ(Ja.cols(), prob.n_knots());
    worst_abs = std::max(worst_abs, (Ja - Jaad).cwiseAbs().maxCoeff());
    scale = std::max(scale, Jaad.cwiseAbs().maxCoeff());
  }
  // also at the calibrated point
  const Eigen::VectorXd xs = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  worst_abs = std::max(worst_abs, (cr.jacobian(xs) - cal::aad_jacobian(prob, xs)).cwiseAbs().maxCoeff());

  std::cout << "  [analytic J] worst_abs=" << worst_abs << " / scale=" << scale
            << " = " << worst_abs / scale << "\n";
  EXPECT_LT(worst_abs / scale, 1e-9) << "analytic Jacobian must match AAD";
}
