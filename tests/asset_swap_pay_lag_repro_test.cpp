// E5 taxonomy: T6 regression (fails on the reverted bug) | T5 value pins (hand-derived dates, closed form)
// ASW PAY-LAG REPRODUCTION (owner decision 2026-09-14: "floating legs generically follow normal swap convention pay lags"):
// the same-currency asset swap's float leg is the currency's swap product leg and pays on THAT PRODUCT's payment_lag. build::asset_swap_float_leg (build/par_asset_swap.hpp:92) pays at the
// accrual END (pay = curve_time(value_date, p.second)) and ignores swc.pay_lag; par_asset_swap_spread then assumes the
// float leg is a par floater (PV = DF(settle) - DF(T)), which a lagged leg is not.
//
// Run against the unfixed library, predicted (record the printed misses in the commit):
//   AssetSwapPayLagRepro.FloatLegPaysOnTheProductPaymentLag   pays early by 2/365, 2/365, 5/365
//   AssetSwapPayLagRepro.ParSpreadCarriesTheLaggedFloatLeg    annuity by ~+3e-4 relative (too big), spread by ~-1e-5
//                                                             (too tight: ~ r*delta*(r+s), delta ~ 2.8/365)
// Header-only, in swaps_tests, so tools/mutate.py reaches it.
//
// Hand dates (USD-SOFR-OIS row, premises asserted): settle Wed 2026-09-16, maturity Thu 2029-02-15, annual boundaries rolled
// BACK on the 15th (short front stub): Mon 2027-02-15 is Washington's Birthday -> ModF Tue 2027-02-16; Tue 2028-02-15;
// Thu 2029-02-15. Pays +2 BD on the product calendar (SIFMA + New York since SC-CAL1): Thu 2027-02-18; Thu 2028-02-17;
// Tue 2029-02-20 (Mon 2029-02-19 is Washington's Birthday -- holiday-live).
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/par_asset_swap.hpp"
#include "swaps/build/schedule.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace cvd = swaps::conventions;
namespace tol = swaps::tol;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }

const char* const kStart[] = {"2026-09-16", "2027-02-16", "2028-02-15"};
const char* const kEnd[] = {"2027-02-16", "2028-02-15", "2029-02-15"};
const char* const kPay[] = {"2027-02-18", "2028-02-17", "2029-02-20"};
constexpr std::size_t kN = 3;

b::SwapConv usd_asset_swap_conv() {
  // The verb's own lookup: the bond convention's currency -> currencies[USD].default_swap_product.
  return b::swap_conv(std::string(cvd::require_bond("US-TREASURY").currency), "");
}

void assert_premises(const b::SwapConv& swc) {
  ASSERT_EQ(swc.product_id, "USD-SOFR-OIS") << "premise: currencies[USD].default_swap_product";
  ASSERT_EQ(swc.pay_lag, 2) << "premise: conventions.json USD-SOFR-OIS.payment_lag";
  ASSERT_EQ(swc.float_freq_tok, "1Y");
  ASSERT_EQ(swc.bdc, "ModifiedFollowing");
  ASSERT_EQ(swc.float_dc, "ACT/360");
  ASSERT_FALSE(b::is_business_day(swc.calendar, d("2027-02-15"))) << "premise: Washington's Birthday 2027";
  ASSERT_FALSE(b::is_business_day(swc.calendar, d("2029-02-19"))) << "premise: Washington's Birthday 2029";
  for (std::size_t i = 0; i < kN; ++i)
    ASSERT_EQ(b::advance_bd(swc.calendar, d(kEnd[i]), swc.pay_lag), d(kPay[i])) << "premise: hand pay date " << kPay[i];
}

struct FlatCurve {  // continuously-compounded flat curve on curve time: a closed-form reference, not an engine curve
  double r;
  double discount(double t) const { return std::exp(-r * t); }
};

}  // namespace

TEST(AssetSwapPayLagRepro, FloatLegPaysOnTheProductPaymentLag) {
  const b::SwapConv swc = usd_asset_swap_conv();
  assert_premises(swc);
  const b::Date vd = d("2026-09-15"), settle = d("2026-09-16"), maturity = d("2029-02-15");
  const b::FloatLegTimes leg = b::asset_swap_float_leg(swc, vd, settle, maturity);
  ASSERT_EQ(leg.pay.size(), kN);
  ASSERT_EQ(leg.tau.size(), kN);
  for (std::size_t i = 0; i < kN; ++i) {
    // The accrual is unchanged by the lag (passes today: pins that the fix moves ONLY the pay date).
    EXPECT_EQ(leg.tau[i], b::year_frac(swc.float_dc, d(kStart[i]), d(kEnd[i]))) << "period " << i;
    const double want = b::curve_time(vd, d(kPay[i]));
    EXPECT_EQ(leg.pay[i], want) << "period " << i << ": pays " << (want - leg.pay[i]) * 365.0
                                << " calendar days early (accrual end instead of end + payment_lag)";
  }
}

TEST(AssetSwapPayLagRepro, ParSpreadCarriesTheLaggedFloatLeg) {
  const b::SwapConv swc = usd_asset_swap_conv();
  assert_premises(swc);
  const b::Date vd = d("2026-09-15");
  b::AssetSwapBond bond;
  bond.convention = "US-TREASURY";
  bond.issue = d("2024-02-15");
  bond.settle = d("2026-09-16");
  bond.maturity = d("2029-02-15");  // a Thursday: the bond's unadjusted redemption == the float leg's adjusted end
  bond.coupon = 0.04;
  bond.clean = 0.985;
  const FlatCurve curve{0.04};
  const b::AssetSwapAnalytics a = b::asset_swap_analytics(bond, vd, curve);

  // Par package, notional 1, everything forward to settle (build/par_asset_swap.hpp's basis):
  //   s = (dirty_curve - dirty_purchase + E/DF(settle)) / A,   A = sum tau_i DF(p_i) / DF(settle),
  //   E = sum (DF(s_i)/DF(e_i) - 1) (DF(e_i) - DF(p_i))   -- the part of the par floater the lag defers past T.
  // The bond side (dirty_curve, accrued) does not depend on the float leg, so it is read from the output.
  const auto T = [&](const char* iso) { return b::curve_time(vd, d(iso)); };
  const double df_settle = curve.discount(b::curve_time(vd, bond.settle));
  double A = 0.0, E = 0.0, A_unlagged = 0.0;
  for (std::size_t i = 0; i < kN; ++i) {
    const double tau = b::year_frac(swc.float_dc, d(kStart[i]), d(kEnd[i]));
    const double ds = curve.discount(T(kStart[i])), de = curve.discount(T(kEnd[i])), dp = curve.discount(T(kPay[i]));
    A += tau * dp;
    A_unlagged += tau * de;
    E += (ds / de - 1.0) * (de - dp);
  }
  A /= df_settle;
  A_unlagged /= df_settle;
  const double dirty_purchase = *bond.clean + a.accrued;
  const double want = (a.dirty_curve - dirty_purchase + E / df_settle) / A;
  const double unlagged = (a.dirty_curve - dirty_purchase) / A_unlagged;  // today's formula on today's dates
  std::cout << "[O-X4 repro] asset swap: annuity got " << a.annuity << " want " << A << "  spread got " << a.asw_spread
            << " want " << want << "  predicted miss today (bp) " << (unlagged - want) * 1e4 << "\n";

  EXPECT_GT(std::abs(want - unlagged), 1e-7) << "fixture premise: the lag moves the spread visibly";
  EXPECT_NEAR(a.annuity, A, tol::literal * A) << "the float annuity must discount at end + payment_lag";
  EXPECT_NEAR(a.asw_spread, want, tol::literal) << "the par spread must carry the lagged float leg (not a par floater)";
}
