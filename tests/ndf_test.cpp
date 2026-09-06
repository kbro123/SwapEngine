// Gate for the NDF/NDS linear FX-forward layer (pricing/ndf.hpp). Pure closed-form forward/discount
// identities — no curve, no vol, no QuantLib — so this runs in the QuantLib-free swaps_tests binary.
// Invariants: (1) an NDF struck at the fair forward has PV ≈ 0; (2) covered-interest-parity outright;
// (3) PV sign flips with direction and scales linearly with notional; (4) delta_spot vs a central finite
// difference; (5) an NDS fair fixed rate reprices its strip to zero.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/pricing/ndf.hpp"

namespace p = swaps::pricing;

namespace {
constexpr double SPOT = 0.20;      // settlement (USD) per 1 ND unit, e.g. USD per 1 BRL
constexpr double R_SETTLE = 0.045; // settlement-ccy (USD) rate
constexpr double R_ND = 0.105;     // ND-ccy (BRL) rate — steep carry, typical for an EM NDF
constexpr double T = 1.5;          // maturity (years)
constexpr double N = 1.0e6;        // notional in ND-ccy units
}  // namespace

TEST(Ndf, StruckAtFairForwardIsZeroPv) {
  const double F = p::ndf_fair_forward<double>(SPOT, T, R_SETTLE, R_ND);
  const double pv = p::ndf_pv<double>(SPOT, F, T, R_SETTLE, R_ND, N, +1.0);
  EXPECT_NEAR(pv, 0.0, 1e-9);
  const p::NdfGreeks<double> g = p::ndf_greeks<double>(SPOT, F, T, R_SETTLE, R_ND, N, +1.0);
  EXPECT_NEAR(g.pv, 0.0, 1e-9);
}

TEST(Ndf, CoveredInterestParity) {
  // F = spot·e^{(r_settle − r_nd)T} to machine precision, at several maturities.
  for (double t : {0.25, 1.0, 2.0, 5.0}) {
    const double F = p::ndf_fair_forward<double>(SPOT, t, R_SETTLE, R_ND);
    EXPECT_NEAR(F, SPOT * std::exp((R_SETTLE - R_ND) * t), 1e-14) << "t=" << t;
  }
}

TEST(Ndf, DirectionFlipsAndNotionalScales) {
  const double K = 0.19;  // off-market strike so PV is nonzero
  const double pv_buy = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND, N, +1.0);
  const double pv_sell = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND, N, -1.0);
  EXPECT_NEAR(pv_buy, -pv_sell, 1e-9);        // sign flips with direction
  EXPECT_GT(std::abs(pv_buy), 1.0);           // materially nonzero
  const double pv_2x = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND, 2.0 * N, +1.0);
  EXPECT_NEAR(pv_2x, 2.0 * pv_buy, 1e-9);      // linear in notional
  const double pv_zero = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND, 0.0, +1.0);
  EXPECT_NEAR(pv_zero, 0.0, 1e-12);
}

TEST(Ndf, DeltaSpotVsFiniteDifference) {
  const double K = 0.185;
  const p::NdfGreeks<double> g = p::ndf_greeks<double>(SPOT, K, T, R_SETTLE, R_ND, N, +1.0);
  const double h = 1e-6;
  const double up = p::ndf_pv<double>(SPOT + h, K, T, R_SETTLE, R_ND, N, +1.0);
  const double dn = p::ndf_pv<double>(SPOT - h, K, T, R_SETTLE, R_ND, N, +1.0);
  const double fd = (up - dn) / (2.0 * h);
  EXPECT_NEAR(g.delta_spot, fd, 1e-4 * std::abs(fd) + 1e-6);

  // While here, pin the two rate Greeks vs central FD too (same closed forms the verb returns).
  const double up_rs = p::ndf_pv<double>(SPOT, K, T, R_SETTLE + h, R_ND, N, +1.0);
  const double dn_rs = p::ndf_pv<double>(SPOT, K, T, R_SETTLE - h, R_ND, N, +1.0);
  EXPECT_NEAR(g.dpv_dr_settle, (up_rs - dn_rs) / (2.0 * h), 1e-3 * std::abs(g.dpv_dr_settle) + 1e-3);
  const double up_rn = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND + h, N, +1.0);
  const double dn_rn = p::ndf_pv<double>(SPOT, K, T, R_SETTLE, R_ND - h, N, +1.0);
  EXPECT_NEAR(g.dpv_dr_nd, (up_rn - dn_rn) / (2.0 * h), 1e-3 * std::abs(g.dpv_dr_nd) + 1e-3);
}

TEST(Nds, FairRateRepricesStripToZero) {
  const std::vector<double> maturities{0.5, 1.0, 1.5, 2.0, 2.5};
  const std::vector<double> notionals{N, N, N, N, N};
  const double kstar = p::nds_fair_rate<double>(SPOT, R_SETTLE, R_ND, maturities, notionals);
  const double pv = p::nds_pv<double>(SPOT, kstar, R_SETTLE, R_ND, maturities, notionals, +1.0);
  EXPECT_NEAR(pv, 0.0, 1e-6);
  // The fair rate is a DF-weighted average of the per-period forwards, so it sits inside their range.
  const double f_first = p::ndf_fair_forward<double>(SPOT, maturities.front(), R_SETTLE, R_ND);
  const double f_last = p::ndf_fair_forward<double>(SPOT, maturities.back(), R_SETTLE, R_ND);
  EXPECT_LT(kstar, f_first);   // downward carry (r_nd > r_settle) => forwards decrease with T
  EXPECT_GT(kstar, f_last);

  // Empty/short notionals default to 1 per period and give the same rate (direction-independent).
  const double kstar_unit = p::nds_fair_rate<double>(SPOT, R_SETTLE, R_ND, maturities, {});
  EXPECT_NEAR(kstar, kstar_unit, 1e-14);
}
