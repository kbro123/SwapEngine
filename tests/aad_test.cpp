// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference market; the
// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// engine is compared to ITSELF / hand formulas, not to a QuantLib number). DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Phase 3 gate: the analytic AAD Jacobian.
//
//   (a) AAD Jacobian matches bump-and-reprice within tol::jacobian_rel (the spec check).
//   (b) LM with the AAD Jacobian reaches the same solution as the numerical-Jacobian LM.
//   (c) Linear-map cross-check: d(integral)/dx from AAD equals the weights w(t) built independently
//       by the unit-vector method, and integral(t) = w(t)*x, DF(t) = exp(-w*x). This validates the
//       CLAUDE.md §2 structure that Phase 5 caches.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>

#include "reference_curve.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

Eigen::VectorXd reference_x() {
  Eigen::VectorXd x(rm::n_knots);
  int i = 0;
  for (double f : rm::reference_front_forwards) x[i++] = f;
  for (double g : rm::reference_back_forwards) x[i++] = g;
  return x;
}

// Central-difference bump-and-reprice Jacobian (the thing AAD replaces).
Eigen::MatrixXd bump_jacobian(const cal::CalibrationProblem& p, const Eigen::VectorXd& x, double h) {
  Eigen::MatrixXd J(p.n_residuals(), p.n_knots());
  for (int k = 0; k < p.n_knots(); ++k) {
    Eigen::VectorXd xp = x, xm = x;
    xp[k] += h;
    xm[k] -= h;
    J.col(k) = (p.residuals<double>(xp) - p.residuals<double>(xm)) / (2 * h);
  }
  return J;
}

}  // namespace

struct Aad : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_problem(mk);
};

TEST_F(Aad, JacobianMatchesBumpAndReprice) {
  const Eigen::VectorXd x = reference_x();
  const Eigen::MatrixXd Jaad = cal::aad_jacobian(prob, x);
  const Eigen::MatrixXd Jbump = bump_jacobian(prob, x, 1e-6);

  ASSERT_EQ(Jaad.rows(), rm::n_instruments);
  ASSERT_EQ(Jaad.cols(), rm::n_knots);

  // Bump noise floors absolute accuracy at ~1e-9, so a per-entry relative test explodes on the
  // (many) structurally-zero entries. Compare on the matrix scale instead: worst absolute error
  // relative to the largest Jacobian entry. (A per-entry relative check on the SIGNIFICANT entries
  // is done separately below.)
  const double scale = Jbump.cwiseAbs().maxCoeff();
  const double worst_abs = (Jaad - Jbump).cwiseAbs().maxCoeff();
  double worst_rel_significant = 0.0;
  for (int i = 0; i < Jaad.rows(); ++i)
    for (int k = 0; k < Jaad.cols(); ++k)
      if (std::abs(Jbump(i, k)) > 1e-3 * scale)  // ignore near-zero entries (bump noise dominated)
        worst_rel_significant =
            std::max(worst_rel_significant, std::abs(Jaad(i, k) - Jbump(i, k)) / std::abs(Jbump(i, k)));

  std::cout << "  [aad vs bump] scale=" << scale << " worst_abs=" << worst_abs
            << " worst_abs/scale=" << worst_abs / scale
            << " worst_rel(significant)=" << worst_rel_significant << "\n";
  EXPECT_LT(worst_abs / scale, swaps::tol::jacobian_rel);
  EXPECT_LT(worst_rel_significant, swaps::tol::jacobian_rel);
}

TEST_F(Aad, AadCalibrationMatchesNumericalCalibration) {
  Eigen::VectorXd x0 = reference_x();
  for (int i = 0; i < x0.size(); ++i) x0[i] += (i % 2 ? 0.0020 : -0.0020);

  const cal::CalibrationResult a = cal::calibrate(prob, x0, /*use_aad=*/true);
  const cal::CalibrationResult n = cal::calibrate(prob, x0, /*use_aad=*/false);

  // Numerical LM stationarity, measured with the (exact) AAD Jacobian at its solution.
  const double n_stat =
      (cal::aad_jacobian(prob, n.x).transpose() * prob.residuals<double>(n.x)).cwiseAbs().maxCoeff();

  std::cout << "  [aad LM] iters=" << a.iterations << " rms=" << a.rms_residual
            << " stat=" << a.stationarity << "   [num LM] iters=" << n.iterations
            << " rms=" << n.rms_residual << " stat=" << n_stat
            << "   x-gap=" << (a.x - n.x).cwiseAbs().maxCoeff() << "\n";

  // The invariants: both solvers reach the SAME objective and both are first-order optimal. They do
  // NOT have to land on the same x -- the objective has a weakly-identified (near-flat) direction, so
  // the minimiser is ill-determined there (see the conditioning check below). AAD's own correctness
  // is established by JacobianMatchesBumpAndReprice, the linear-map test, and the recovery test.
  EXPECT_LT(std::abs(a.rms_residual - n.rms_residual), 1e-9) << "same objective value";
  EXPECT_LT(a.stationarity, 1e-8) << "AAD-driven LM is first-order optimal";
  EXPECT_LT(n_stat, 1e-6) << "numerical LM is also (approximately) stationary";
}

// Characterise the weak identifiability seen above: the residual Jacobian has a small trailing
// singular value, so some knot direction is barely constrained by the instruments. Documented here
// so it is a known property (a smoothness/Tikhonov regulariser is the future fix), not a surprise.
TEST_F(Aad, CalibrationConditioning) {
  const Eigen::MatrixXd J = cal::aad_jacobian(prob, reference_x());
  const Eigen::VectorXd sv = J.jacobiSvd().singularValues();
  const double cond = sv(0) / sv(sv.size() - 1);
  std::cout << "  [conditioning] sigma_max=" << sv(0) << " sigma_min=" << sv(sv.size() - 1)
            << " cond=" << cond << "\n";
  EXPECT_GT(sv(sv.size() - 1), 0.0) << "Jacobian must have full column rank (no exact null space)";
}

// The curve integral is linear in the knot forwards: integral(t) = w(t) . x, with w depending only
// on the knot times. Verify AAD recovers w, and that w reconstructs integral and discount.
TEST_F(Aad, IntegralIsLinearMapAndAadRecoversTheWeights) {
  const Eigen::VectorXd x = reference_x();
  const int m = rm::n_knots;
  const std::vector<double> test_t{0.05, 0.30, 0.69, 0.80, 1.5, 3.2, 7.0, 15.0, 29.0};

  // Independent weights: w_k(t) = integral(t) evaluated with forwards = e_k.
  auto weight_column = [&](int k) {
    Eigen::VectorXd ek = Eigen::VectorXd::Zero(m);
    ek[k] = 1.0;
    auto c = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(mk.meeting_times, mk.back_times));
    c.set_forwards(ek);
    Eigen::VectorXd col(test_t.size());
    for (std::size_t j = 0; j < test_t.size(); ++j) col[j] = c.integral(test_t[j]);
    return col;
  };
  Eigen::MatrixXd W(test_t.size(), m);  // W(j,k) = w_k(t_j)
  for (int k = 0; k < m; ++k) W.col(k) = weight_column(k);

  // AAD weights: d integral(t)/dx from the dual curve (independent of x since integral is linear).
  auto cd = swaps::curve::make_modular_curve<swaps::ad::Dual>(swaps::curve::flat_hermite(mk.meeting_times, mk.back_times));
  cd.set_forwards(swaps::ad::seed(x));

  auto c = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(mk.meeting_times, mk.back_times));
  c.set_forwards(x);

  double worst_w = 0, worst_recon = 0, worst_df = 0;
  for (std::size_t j = 0; j < test_t.size(); ++j) {
    const swaps::ad::Dual I = cd.integral(test_t[j]);
    for (int k = 0; k < m; ++k) worst_w = std::max(worst_w, std::abs(I.derivatives()[k] - W(j, k)));
    // reconstruction: integral(t) = w(t) . x
    const double recon = W.row(j).dot(x.transpose());
    worst_recon = std::max(worst_recon, std::abs(recon - c.integral(test_t[j])));
    worst_df = std::max(worst_df, std::abs(c.discount(test_t[j]) - std::exp(-recon)));
  }
  std::cout << "  [linear map] max|w_aad - w_ek|=" << worst_w << " max|recon-integral|=" << worst_recon
            << " max|DF-exp(-w.x)|=" << worst_df << "\n";
  EXPECT_LT(worst_w, 1e-12);
  EXPECT_LT(worst_recon, 1e-12);
  EXPECT_LT(worst_df, 1e-14);
}
