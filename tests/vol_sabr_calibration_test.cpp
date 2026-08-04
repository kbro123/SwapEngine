// Gate for SABR strip calibration (vol/sabr_calibration.hpp): recover known params from a clean strip (proving
// the analytic ad::Dual Jacobian + LM work), fit a non-SABR strip within tolerance, and the arbitrage gate.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/vol/sabr_calibration.hpp"

namespace v = swaps::vol;

namespace {
std::vector<double> strip_strikes(double f, int lo_bp = -150, int hi_bp = 150, int step = 15) {
  std::vector<double> k;
  for (int bp = lo_bp; bp <= hi_bp; bp += step) k.push_back(f + bp / 1e4);
  return k;
}
}  // namespace

TEST(SabrCalib, RecoversKnownParamsExactly) {
  const double F = 0.032, T = 2.0;
  const v::SabrParams truth{0.0090, -0.30, 0.45};
  const std::vector<double> K = strip_strikes(F);
  std::vector<double> mv;
  for (double k : K) mv.push_back(v::sabr_normal_vol(F, k, T, truth));

  const v::SabrCalibResult r = v::sabr_calibrate(F, T, K, mv);
  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.rms, 1e-9);
  EXPECT_NEAR(r.params.alpha, truth.alpha, 1e-6);
  EXPECT_NEAR(r.params.rho, truth.rho, 1e-5);
  EXPECT_NEAR(r.params.nu, truth.nu, 1e-5);
}

TEST(SabrCalib, RecoversFromADifferentSeed) {
  const double F = 0.045, T = 5.0;
  const v::SabrParams truth{0.0075, 0.15, 0.80};  // positive skew, high vol-of-vol
  const std::vector<double> K = strip_strikes(F, -200, 200, 20);
  std::vector<double> mv;
  for (double k : K) mv.push_back(v::sabr_normal_vol(F, k, T, truth));

  const v::SabrCalibResult r = v::sabr_calibrate(F, T, K, mv, v::SabrParams{0.01, -0.5, 0.2});
  EXPECT_LT(r.rms, 1e-8);
  EXPECT_NEAR(r.params.alpha, truth.alpha, 1e-5);
  EXPECT_NEAR(r.params.rho, truth.rho, 1e-4);
  EXPECT_NEAR(r.params.nu, truth.nu, 1e-4);
}

TEST(SabrCalib, FitsNonSabrStripWithinTolerance) {
  // A hand-built smile with a kink that no SABR passes through exactly; expect a small-but-nonzero rms.
  const double F = 0.030, T = 1.0;
  const std::vector<double> K = strip_strikes(F);
  std::vector<double> mv;
  for (double k : K) {
    const double x = (k - F) * 1e4;                     // moneyness in bp
    mv.push_back(0.0090 + 2.0e-6 * x + 8.0e-9 * x * x + (x < 0 ? 1.0e-4 : 0.0));  // skew + a wing kink
  }
  const v::SabrCalibResult r = v::sabr_calibrate(F, T, K, mv);
  EXPECT_GT(r.rms, 0.0);
  EXPECT_LT(r.rms, 5e-4);  // within ~5bp rms of an imperfect market strip
}

TEST(SabrCalib, ArbitrageGate) {
  const double F = 0.030, T = 1.0;
  // A benign SABR smile is arbitrage-free — its butterfly stays >= -tol across the quoted range (only FD noise
  // in the far wing). Normal beta=0 SABR is arb-robust by construction, so this is the expected case.
  EXPECT_TRUE(v::sabr_arbitrage_free(F, T, v::SabrParams{0.0090, -0.30, 0.50}, F - 0.02, F + 0.02));
  EXPECT_GE(v::sabr_min_butterfly(F, T, v::SabrParams{0.0090, -0.30, 0.50}, F - 0.02, F + 0.02), -1e-8);

  // The general detector genuinely catches a non-convex call ladder (a butterfly arbitrage) — e.g. a
  // fitted/interpolated smile with a kink. A convex ladder passes; a dented one is flagged negative.
  const std::vector<double> convex{0.050, 0.030, 0.016, 0.008, 0.004};      // decreasing, convex -> density >= 0
  EXPECT_GE(v::min_price_convexity(convex, 1.0), 0.0);
  const std::vector<double> dented{0.050, 0.030, 0.022, 0.008, 0.004};      // middle price too high -> non-convex
  EXPECT_LT(v::min_price_convexity(dented, 1.0), 0.0);
}
