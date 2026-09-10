// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Phase-2 gate for the SABR normal-vol smile (beta=0). Pure closed-form limits — QuantLib-free, in
// swaps_tests. Checks the flat-smile limit (nu->0), the exact ATM closed form, smoothness/continuity through
// the ATM removable singularity, and that a positive vol-of-vol produces a genuine (non-flat) smile whose
// vols stay positive and feed cleanly into the Bachelier pricer.
#include <gtest/gtest.h>

#include <cmath>

#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr.hpp"

namespace v = swaps::vol;

namespace {
constexpr double F = 0.030;
constexpr double T = 5.0;
}  // namespace

TEST(Sabr, FlatWhenNuZero) {
  const v::SabrParams p{0.0090, -0.3, 0.0};  // nu = 0 -> a flat normal-vol smile at alpha
  for (double K : {0.005, 0.020, 0.030, 0.045, 0.060}) {
    EXPECT_NEAR(v::sabr_normal_vol(F, K, T, p), 0.0090, 1e-14) << "K=" << K;
  }
}

TEST(Sabr, AtmClosedForm) {
  const v::SabrParams p{0.0088, -0.25, 0.45};
  // ATM correction is (2-3rho^2)/24 * nu^2 * T above alpha.
  const double expected = 0.0088 * (1.0 + ((2.0 - 3.0 * 0.25 * 0.25) / 24.0) * 0.45 * 0.45 * T);
  EXPECT_NEAR(v::sabr_normal_vol(F, F, T, p), expected, 1e-13);
}

TEST(Sabr, ContinuousThroughAtm) {
  // The ζ/x̂(ζ) factor has a removable singularity at K=F; the smile must be smooth across it.
  const v::SabrParams p{0.0092, -0.35, 0.55};
  const double atm = v::sabr_normal_vol(F, F, T, p);
  const double just_below = v::sabr_normal_vol(F, F - 1e-6, T, p);
  const double just_above = v::sabr_normal_vol(F, F + 1e-6, T, p);
  EXPECT_NEAR(just_below, atm, 1e-6);
  EXPECT_NEAR(just_above, atm, 1e-6);
}

TEST(Sabr, PositiveSkewedSmileFeedsBachelier) {
  const v::SabrParams p{0.0090, -0.30, 0.50};  // negative rho -> receiver-side (low-strike) vols richer
  const double annuity = 4.2;
  double prev_low = -1.0;
  for (double K : {0.010, 0.020, 0.030, 0.040, 0.050}) {
    const double vol = v::sabr_normal_vol(F, K, T, p);
    EXPECT_GT(vol, 0.0) << "K=" << K;
    // negative rho: low strikes carry higher normal vol than ATM.
    if (K < F) EXPECT_GT(vol, v::sabr_normal_vol(F, F, T, p)) << "K=" << K;
    // the smile vol prices a valid (positive) swaption.
    const double px = v::bachelier_price<double>(F, K, vol, T, annuity, v::Payoff::Payer);
    EXPECT_GE(px, 0.0);
    prev_low = px;
  }
  EXPECT_GE(prev_low, 0.0);
}
