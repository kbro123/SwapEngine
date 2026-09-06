// Phase-2 gate for the FULL-beta SABR smile (vol/sabr.hpp). Verifies that generalising the backbone to a free
// β ∈ [0,1] leaves β=0 BYTE-IDENTICAL to the legacy normal-SABR code (bit-for-bit), that β=1 matches the
// Hagan lognormal (Black) benchmark, that the ATM K→F limit is continuous and singularity-free for several
// betas, and that calibration recovers known params with β fixed (β ∈ {0, 0.5, 1}) and with β freed.
// QuantLib-free, pure closed-form + calibrator; lives in swaps_tests.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr.hpp"
#include "swaps/vol/sabr_calibration.hpp"

namespace v = swaps::vol;

namespace {
constexpr double F = 0.030;
constexpr double T = 5.0;

// Verbatim copy of the LEGACY beta=0 normal-SABR formula, frozen here as the byte-identical oracle. If the
// shipped sabr_normal_vol(...,beta=0) ever drifts from this, the bit-exact test below fails.
double legacy_normal_vol(double fwd, double strike, double expiry, double alpha, double rho, double nu) {
  const double time_corr = 1.0 + ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu * expiry;
  const double dk = fwd - strike;
  if (nu <= 0.0) return alpha * time_corr;
  const double zeta = (nu / alpha) * dk;
  if (zeta > -1e-7 && zeta < 1e-7) return alpha * time_corr;
  const double xhat =
      std::log((std::sqrt(1.0 - 2.0 * rho * zeta + zeta * zeta) - rho + zeta) / (1.0 - rho));
  return alpha * (zeta / xhat) * time_corr;
}

bool bit_identical(double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; }
}  // namespace

// --- 1. BYTE-IDENTICAL default (beta=0) --------------------------------------------------------------------
TEST(SabrBeta, Beta0BitForBitLegacy) {
  const double alpha = 0.0091, rho = -0.32, nu = 0.47;
  const v::SabrParams p{alpha, rho, nu};  // beta defaults to 0.0
  EXPECT_EQ(p.beta, 0.0);
  for (double K : {0.005, 0.018, 0.030, 0.030 + 1e-9, 0.042, 0.060}) {
    const double got = v::sabr_normal_vol(F, K, T, p);
    const double oracle = legacy_normal_vol(F, K, T, alpha, rho, nu);
    EXPECT_TRUE(bit_identical(got, oracle)) << "K=" << K << " got=" << got << " oracle=" << oracle;
    // An explicit beta=0.0 argument takes the same verbatim branch.
    EXPECT_TRUE(bit_identical(v::sabr_normal_vol<double>(F, K, T, alpha, rho, nu, 0.0), oracle)) << "K=" << K;
  }
  // nu = 0 (flat) and the ATM point also byte-match.
  EXPECT_TRUE(bit_identical(v::sabr_normal_vol(F, 0.02, T, v::SabrParams{0.009, -0.3, 0.0}),
                            legacy_normal_vol(F, 0.02, T, 0.009, -0.3, 0.0)));
  EXPECT_TRUE(bit_identical(v::sabr_normal_vol(F, F, T, p), legacy_normal_vol(F, F, T, alpha, rho, nu)));
}

// --- 2. beta=1 lognormal (Black) benchmark -----------------------------------------------------------------
TEST(SabrBeta, Beta1LognormalBenchmark) {
  const double Fl = 0.030, alpha = 0.0090, rho = -0.25, nu = 0.40;  // note: alpha is a lognormal level here

  // (a) Known analytic ATM lognormal Black vol: σ_B = α·[1 + (ρνα/4 + (2−3ρ²)ν²/24)·T].
  const v::SabrParams p1{alpha, rho, nu, 1.0};
  const double atm_expected =
      alpha * (1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0) * T);
  EXPECT_NEAR(v::sabr_black_vol(Fl, Fl, T, p1), atm_expected, 1e-14);

  // (b) Independent literal evaluation of the Hagan B.65a lognormal formula at an off-ATM strike.
  const double K = 0.024;
  const double L = std::log(Fl / K), FK = Fl * K;
  const double FKhe = std::pow(FK, 0.0);  // (FK)^{(1-β)/2}, β=1 -> 1
  const double den_br = 1.0;              // (1-β)=0 -> bracket collapses to 1
  const double zeta = (nu / alpha) * FKhe * L;
  const double xhat =
      std::log((std::sqrt(1.0 - 2.0 * rho * zeta + zeta * zeta) - rho + zeta) / (1.0 - rho));
  const double tterm =
      (0.0 + rho * 1.0 * nu / 4.0 * alpha / FKhe + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * T;
  const double bench = alpha / (FKhe * den_br) * (zeta / xhat) * (1.0 + tterm);
  EXPECT_NEAR(v::sabr_black_vol(Fl, K, T, p1), bench, 1e-14);

  // (c) β=1, ν=0 is a flat Black smile at α (the textbook lognormal-with-no-vol-of-vol limit).
  const v::SabrParams flat{alpha, rho, 0.0, 1.0};
  for (double k : {0.015, 0.024, 0.030, 0.040, 0.055})
    EXPECT_NEAR(v::sabr_black_vol(Fl, k, T, flat), alpha, 1e-14) << "K=" << k;

  // (d) Normal/Black ATM consistency: σ_N ≈ F·σ_B·(1 − σ_B²T/24) to second order in T.
  const double sn = v::sabr_atm_normal_vol(Fl, T, p1);
  const double sb = v::sabr_black_vol(Fl, Fl, T, p1);
  EXPECT_NEAR(sn, Fl * sb * (1.0 - sb * sb * T / 24.0), 2e-6 * Fl);
}

// --- 3. ATM continuity / no divide-by-zero for several betas ----------------------------------------------
TEST(SabrBeta, AtmContinuousAllBetas) {
  const double alpha = 0.0088, rho = -0.30, nu = 0.50;
  for (double beta : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const v::SabrParams p{alpha, rho, nu, beta};
    const double atm = v::sabr_normal_vol(F, F, T, p);
    const double below = v::sabr_normal_vol(F, F - 1e-7, T, p);
    const double above = v::sabr_normal_vol(F, F + 1e-7, T, p);
    ASSERT_TRUE(std::isfinite(atm)) << "beta=" << beta;
    EXPECT_GT(atm, 0.0) << "beta=" << beta;
    EXPECT_NEAR(below, atm, 1e-7) << "beta=" << beta;   // continuous through K=F
    EXPECT_NEAR(above, atm, 1e-7) << "beta=" << beta;
    // The 3-arg ATM helper equals the smile evaluated exactly at the money.
    EXPECT_NEAR(v::sabr_atm_normal_vol(F, T, p), atm, 1e-15) << "beta=" << beta;
  }
  // The 2-arg legacy ATM helper is the β=0 closed form.
  const v::SabrParams p0{alpha, rho, nu};
  EXPECT_TRUE(bit_identical(v::sabr_atm_normal_vol(T, p0),
                            alpha * (1.0 + ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu * T)));
}

// --- 4. Fixed-beta calibration recovers known params for beta in {0, 0.5, 1} -------------------------------
TEST(SabrBeta, FixedBetaCalibrationRecovers) {
  const double Fc = 0.032, Tc = 2.0;
  std::vector<double> K;
  for (int bp = -150; bp <= 150; bp += 15) K.push_back(Fc + bp / 1e4);

  for (double beta : {0.0, 0.5, 1.0}) {
    const v::SabrParams truth{0.0090 * (beta == 0.0 ? 1.0 : std::pow(Fc, -beta)), -0.28, 0.42, beta};
    std::vector<double> mv;
    for (double k : K) mv.push_back(v::sabr_normal_vol(Fc, k, Tc, truth));

    // Calibrate with beta FIXED at the true value (the desk workflow), fitting {alpha, rho, nu}.
    v::SabrParams seed{};
    seed.beta = beta;
    const v::SabrCalibResult r = v::sabr_calibrate(Fc, Tc, K, mv, seed);
    EXPECT_TRUE(r.converged) << "beta=" << beta;
    EXPECT_LT(r.rms, 1e-9) << "beta=" << beta;
    EXPECT_EQ(r.params.beta, beta) << "beta=" << beta;  // beta is held, not fitted
    EXPECT_NEAR(r.params.alpha, truth.alpha, 1e-6 * truth.alpha + 1e-9) << "beta=" << beta;
    EXPECT_NEAR(r.params.rho, truth.rho, 1e-5) << "beta=" << beta;
    EXPECT_NEAR(r.params.nu, truth.nu, 1e-5) << "beta=" << beta;
  }
}

// --- 5. Free-beta calibration recovers the backbone too ----------------------------------------------------
TEST(SabrBeta, FreeBetaCalibrationRecovers) {
  const double Fc = 0.030, Tc = 3.0;
  const v::SabrParams truth{0.0090 * std::pow(Fc, -0.5), -0.30, 0.45, 0.5};  // β = 0.5 CEV backbone
  std::vector<double> K;
  for (int bp = -200; bp <= 200; bp += 10) K.push_back(Fc + bp / 1e4);
  std::vector<double> mv;
  for (double k : K) mv.push_back(v::sabr_normal_vol(Fc, k, Tc, truth));

  // Free all four params from a DIFFERENT beta seed; the fit must find the true backbone.
  v::SabrParams seed{0.02, 0.0, 0.20, 0.30};
  const v::SabrCalibResult r = v::sabr_calibrate_free_beta(Fc, Tc, K, mv, seed);
  EXPECT_LT(r.rms, 1e-7);
  EXPECT_NEAR(r.params.beta, truth.beta, 5e-3);
  EXPECT_NEAR(r.params.rho, truth.rho, 5e-3);
  EXPECT_NEAR(r.params.nu, truth.nu, 5e-3);
  // With β recovered, alpha follows the level.
  EXPECT_NEAR(r.params.alpha, truth.alpha, 1e-3 * truth.alpha);
}
