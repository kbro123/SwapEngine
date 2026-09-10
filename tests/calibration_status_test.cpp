// E5 taxonomy: T6 regression (fails on the reverted bug)
// CalibrationResult.converged / status and the non-finite input guard (E3 finding C3, fixed 2026-09-10).
// Before: a NaN quote returned the SEED curve with info=4 ("cosine too small" -- LM's own convergence
// code) and rms=NaN, no throw; a rank-deficient solve reported info=0 ("improper input", the status of the
// first solve that refused m < n) although the anchored re-solve had converged (probe C-lm-status).
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}
cal::BundleProblem square() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  for (int T : {1, 2, 3, 5, 7, 10}) p.instruments.push_back(par_swap(T));
  Eigen::VectorXd xt(6); xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < 6; ++i) p.instruments[i].market = r[i];
  return p;
}
const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.03);
}  // namespace

TEST(CalibrationStatus, NonFiniteQuoteOrSeedThrows) {
  for (double bad : {std::nan(""), double(INFINITY), -double(INFINITY)}) {
    auto p = square();
    p.instruments[2].market = bad;
    EXPECT_THROW(cal::calibrate(p, x0), std::invalid_argument);
    EXPECT_THROW(cal::calibrate(p, x0, /*use_aad=*/false), std::invalid_argument);
  }
  Eigen::VectorXd xbad = x0; xbad[1] = std::nan("");
  EXPECT_THROW(cal::calibrate(square(), xbad), std::invalid_argument);
}

TEST(CalibrationStatus, SquareSolveReportsConvergedWithAStatusText) {
  const cal::CalibrationResult r = cal::calibrate(square(), x0);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(std::string(r.status), cal::lm_status_text(r.info));
  EXPECT_NE(std::string(r.status), "not started");
  EXPECT_LT(r.rms_residual, 1e-12);
  EXPECT_EQ(r.rank_deficiency, 0);
}

TEST(CalibrationStatus, UnderDeterminedSolveReportsItsOwnStatusNotTheRefusedFirstSolve) {
  auto p = square();
  p.instruments.erase(p.instruments.begin() + 4);  // 5 rows, 6 knots
  const cal::CalibrationResult r = cal::calibrate(p, x0);
  EXPECT_EQ(r.rank_deficiency, 1);
  EXPECT_NE(r.info, 0) << "info=0 is Eigen's 'improper input' from the refused m<n solve, not the anchored re-solve's";
  EXPECT_TRUE(r.converged) << r.status;
  EXPECT_LT(r.rms_residual, 1e-12);
}

TEST(CalibrationStatus, InconsistentDuplicateConvergesToALeastSquaresFit) {
  auto p = square();
  p.instruments.push_back(p.instruments[3]);
  p.instruments.back().market += 10e-4;
  const cal::CalibrationResult r = cal::calibrate(p, x0);
  EXPECT_TRUE(r.converged) << r.status;
  EXPECT_GT(r.rms_residual, 1e-4);  // the 10 bp inconsistency is split, not hidden
  EXPECT_LT(r.stationarity, 1e-12);
}
