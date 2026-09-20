// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// THE ROUTER'S PARTITION (item 5 of the golden-source programme, 2026-09-10). A bundle whose back end
// interpolates with a VALUE-dependent scheme (MonotoneCubic) used to be non-cacheable AS A WHOLE: one such
// region anywhere sent EVERY row in the bundle to the AAD block, including short par swaps that never read
// a time past the linear front, and including rows on a completely different curve. The router now decides
// per row, against each curve's LINEAR HORIZON (the end of its maximal linear prefix, recursively capped by
// its base's), so the front stays on the W-cache and only the rows that actually reach the non-linear part
// pay for it.
//
// That is a BEHAVIOUR change -- rows moved from the slow path to the fast one -- so this file pins BOTH
// halves: WHERE each row was routed (compiled_row), and that the two routes still agree with the generic
// templated kernel to rounding in value AND derivative. Parity alone would be satisfied by a router that
// sent everything back to the slow path, which is exactly the bug this replaces.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {

// An annual par swap on curve `c`, maturing at T: it reads DFs at 1, 2, ... T and nothing beyond.
cal::Instrument par_inst(double T, int c) {
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon f;
    f.obs.sub_start = {prev};
    f.obs.sub_end = {u};
    f.obs.tau_index = u - prev;
    f.pay = u;
    f.tau_pay = u - prev;
    in.fwd.coupons.push_back(f);
    in.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  in.fwd.forecast = in.fwd.discount = in.fixed.discount = c;
  return in;
}

// The mixed bundle. Curve 0: Flat{0.25} + Linear{1,2,3} + MonotoneCubic{5,10} -- linear horizon 3.0.
// Curve 1: a SPREAD over curve 0, fully linear in its own right, so its horizon is inherited (3.0 too --
// logdf_weight recurses into the base at the same times). Curve 2: an independent, fully linear curve,
// horizon +inf, which the old bundle-wide veto punished for its neighbours' scheme.
cal::BundleProblem mixed_bundle() {
  cal::BundleProblem p;
  p.curves.resize(3);
  p.curves[0] = px::CurveStructure{.base = -1, .currency = 0,
                                   .regions = {cv::CurveModule{{0.25}, cv::Scheme::Flat},
                                               cv::CurveModule{{1, 2, 3}, cv::Scheme::Linear},
                                               cv::CurveModule{{5, 10}, cv::Scheme::MonotoneCubic}}};
  p.curves[1] = px::CurveStructure{.base = 0, .currency = 0,
                                   .regions = cv::flat_hermite({0.25}, {1, 2, 3, 5, 10})};
  p.curves[2] = px::CurveStructure{.base = -1, .currency = 1,
                                   .regions = cv::flat_hermite({0.25}, {1, 2, 3, 5, 10})};
  for (double T : {1.0, 2.0, 3.0, 5.0, 10.0}) p.instruments.push_back(par_inst(T, 0));
  for (double T : {1.0, 2.0, 3.0, 5.0, 10.0}) p.instruments.push_back(par_inst(T, 1));
  for (double T : {1.0, 2.0, 3.0, 5.0, 10.0}) p.instruments.push_back(par_inst(T, 2));
  return p;
}

// A state with a distinct level per curve, tilted so no two knots share a forward.
Eigen::VectorXd tilted_state(const cal::BundleProblem& p, double bump) {
  Eigen::VectorXd x(p.n_knots());
  for (int c = 0; c < static_cast<int>(p.curves.size()); ++c) {
    const int o = p.offset(c);
    for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i)
      x[o + i] = 0.02 + 0.01 * c + 0.0007 * i + bump;
  }
  return x;
}

// The GENERIC path's model quotes: BundleProblem exposes residuals only, and with no bands set (the
// default) model_rates = residuals + market, which is the identity the AadResidualEngine itself uses.
Eigen::VectorXd generic_rates(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  return p.residuals<double>(x) + p.market();
}

}  // namespace

// THE PARTITION, both routings. Since 2026-09-20 the piecewise-linear W tier is the DEFAULT: a MonotoneCubic
// region is piecewise-linear in its knots, so a row reading it still rides the compiled W-cache (W exact in
// the current Hyman branch cell, re-taken analytically when x crosses one) and NOTHING is pushed to the AAD
// block for interpolation reasons. The horizon partition below it -- rows within their curve's linear prefix
// compiled, rows past it on AAD -- is what SWAPS_EXP_PWL=0 restores, and it is still the fallback whenever a
// curve is not piecewise-linear-capable, so both are pinned here.
TEST(RouterPartition, ThePiecewiseLinearTierKeepsEveryRowOnTheWCache) {
  const cal::BundleProblem p = mixed_bundle();
  ASSERT_TRUE(cal::curves_are_noncacheable(p.curves)) << "pre-2026-09-10 this alone sent all 15 rows to AAD";

  const cal::HybridBundleResidual eng(p);  // default: the tier
  ASSERT_TRUE(eng.pwl_active()) << "the tier is the default routing for a piecewise-linear curve";
  for (int row = 0; row < 15; ++row)
    EXPECT_GE(eng.compiled_row(row), 0) << "row " << row << " should ride the W-cache under the tier";
  EXPECT_EQ(eng.n_compiled_rows(), 15);
}

// The FALLBACK routing (SWAPS_EXP_PWL=0, and any curve the tier cannot cover): per row, against each curve's
// LINEAR HORIZON -- the end of its maximal linear prefix, recursively capped by its base's. Before 2026-09-10
// one value-dependent region anywhere sent EVERY row to the AAD block, including short par swaps that never
// read past the linear front and rows on a completely different curve.
TEST(RouterPartition, WithoutTheTierOnlyRowsReachingTheNonLinearRegionLeaveTheWCache) {
  const cal::BundleProblem p = mixed_bundle();
  const std::vector<double> h = px::curve_linear_horizons(p.curves);
  EXPECT_DOUBLE_EQ(h[0], 3.0) << "curve 0's linear prefix ends at the last Linear knot";
  EXPECT_DOUBLE_EQ(h[1], 3.0) << "a linear SPREAD inherits its base's horizon, not +inf";
  EXPECT_EQ(h[2], std::numeric_limits<double>::infinity()) << "an unrelated linear curve is unrestricted";

  const cal::HybridBundleResidual eng(p, /*pwl=*/false);
  ASSERT_FALSE(eng.pwl_active());
  // curves 0 and 1: the 1y/2y/3y rows read nothing past 3.0 and stay compiled; 5y/10y do not.
  for (int c = 0; c < 2; ++c)
    for (int k = 0; k < 5; ++k) {
      const int row = 5 * c + k;
      if (k < 3) EXPECT_GE(eng.compiled_row(row), 0) << "curve " << c << " row " << k << " should be compiled";
      else EXPECT_EQ(eng.compiled_row(row), -1) << "curve " << c << " row " << k << " should be on AAD";
    }
  for (int k = 0; k < 5; ++k)
    EXPECT_GE(eng.compiled_row(10 + k), 0) << "curve 2 is fully linear; row " << k << " must stay compiled";
  EXPECT_EQ(eng.n_compiled_rows(), 11);
}

// And the two routings agree: same residuals, same Jacobian, to rounding. (Parity against the generic
// templated kernel is the test below; this one is tier-vs-fallback directly.)
TEST(RouterPartition, TheTwoRoutingsAgree) {
  const cal::BundleProblem p = mixed_bundle();
  const cal::HybridBundleResidual tier(p, true), fallback(p, false);
  const Eigen::VectorXd x = tilted_state(p, 0.0);
  EXPECT_LT((tier.residuals(x) - fallback.residuals(x)).cwiseAbs().maxCoeff(), 1e-14);
  const Eigen::MatrixXd Jf = fallback.jacobian(x);
  EXPECT_LT((tier.jacobian(x) - Jf).cwiseAbs().maxCoeff(), 1e-12 * Jf.cwiseAbs().maxCoeff());
}

// Parity: on that same mixed bundle both halves reproduce the generic templated kernel exactly, and the
// hybrid Jacobian matches a central difference of it across the seam.
TEST(RouterPartition, MixedBundleMatchesTheGenericKernelInValueAndDerivative) {
  const cal::BundleProblem p = mixed_bundle();
  const cal::HybridBundleResidual eng(p);

  for (double bump : {0.0, -0.004, 0.006}) {
    const Eigen::VectorXd x = tilted_state(p, bump);
    const Eigen::VectorXd got = eng.model_rates(x), want = generic_rates(p, x);
    ASSERT_EQ(got.size(), want.size());
    EXPECT_LT((got - want).cwiseAbs().maxCoeff(), 1e-13) << "bump " << bump;
  }

  const Eigen::VectorXd x = tilted_state(p, 0.0);
  const Eigen::MatrixXd J = eng.jacobian(x);
  const double eps = 1e-6;
  for (int j = 0; j < p.n_knots(); ++j) {
    Eigen::VectorXd up = x, dn = x;
    up[j] += eps;
    dn[j] -= eps;
    const Eigen::VectorXd fd = (generic_rates(p, up) - generic_rates(p, dn)) / (2 * eps);
    for (int r = 0; r < p.n_residuals(); ++r)
      EXPECT_NEAR(J(r, j), fd[r], 1e-7) << "row " << r << " (" << (eng.compiled_row(r) >= 0 ? "W-cache" : "AAD")
                                        << ") vs knot " << j;
  }
}

// And it still calibrates: the partitioned engine solves the mixed bundle back to its generating state.
TEST(RouterPartition, MixedBundleCalibratesThroughBothHalves) {
  cal::BundleProblem p = mixed_bundle();
  const Eigen::VectorXd q = generic_rates(p, tilted_state(p, 0.0));
  for (int i = 0; i < p.n_residuals(); ++i) p.instruments[static_cast<std::size_t>(i)].market = q[i];

  const cal::CalibrationResult res = cal::calibrate(p, tilted_state(p, 0.003));
  EXPECT_TRUE(res.converged) << res.status;
  EXPECT_LT(res.rms_residual, 1e-10);
  // NOT a state-recovery test: 18 knots carry 15 quotes, so the smoothed solve legitimately lands on a
  // different (smoother) state than the generating one. What must hold is that the state the PARTITIONED
  // engine converged on reprices every quote on the GENERIC path -- fast rows and slow rows alike.
  const Eigen::VectorXd back = generic_rates(p, res.x);
  EXPECT_LT((back - q).cwiseAbs().maxCoeff(), 1e-10) << "the hybrid's solution reprices on the generic kernel";
}
