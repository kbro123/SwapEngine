// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// build::asset_swap_float_leg (E7 stage 3.5): the float leg the asset_swap verb receives. Every boundary is rolled
// BACKWARD from the maturity on the maturity's own day of month (a short front stub from settlement), so a leap-day
// or month-end maturity keeps its day in every year. Until 2026-09-13 a hand loop chained the roll from the previous
// date and a 2036-02-29 maturity got a 2028-02-28 boundary (tests/asset_swap_oracle_test.cpp shows it against
// QuantLib; this file pins the schedule itself, header-only, so tools/mutate.py reaches it).
// 2026-09-14: the leg ENDS on the maturity adjusted Following (ql::AssetSwap's requirement), not the product's Modified
// Following -- at a weekend month-end the latter ended the leg in the previous month, days before redemption.
#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "swaps/build/par_asset_swap.hpp"

namespace b = swaps::build;
namespace cvd = swaps::conventions;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }
}  // namespace

TEST(AssetSwapLeg, RollsEveryBoundaryFromTheMaturityAndKeepsTheLeapDay) {
  const b::Date vd = d("2026-09-15"), settle = d("2026-09-16"), maturity = d("2036-02-29");
  const b::SwapConv swc = b::swap_conv(std::string(cvd::require_bond("US-TREASURY").currency), "");
  const b::FloatLegTimes leg = b::asset_swap_float_leg(swc, vd, settle, maturity);
  // Boundaries are the ACCRUAL ends; the coupons pay payment_lag business days later (asset_swap_pay_lag_repro_test).
  const auto ends_on = [&](const b::Date& x) {
    const double t = b::curve_time(vd, b::adjust(swc.calendar, x, swc.bdc));
    return std::find(leg.accrual_end.begin(), leg.accrual_end.end(), t) != leg.accrual_end.end();
  };
  ASSERT_EQ(swc.float_freq_tok, "1Y") << "the fixture assumes USD's annual default swap product";
  EXPECT_TRUE(ends_on(d("2028-02-29"))) << "2028 keeps the maturity's 29th";
  EXPECT_FALSE(std::find(leg.accrual_end.begin(), leg.accrual_end.end(), b::curve_time(vd, d("2028-02-28"))) !=
               leg.accrual_end.end())
      << "a chained roll lands on 2028-02-28";
  EXPECT_TRUE(ends_on(d("2029-02-28"))) << "a non-leap year clamps to its last day";
  ASSERT_EQ(leg.accrual_end.size(), 10u);  // Feb 2027 .. Feb 2036
  ASSERT_EQ(leg.pay.size(), leg.accrual_end.size());
  ASSERT_EQ(leg.accrual_start.size(), leg.accrual_end.size());
  EXPECT_EQ(leg.accrual_end.back(), b::curve_time(vd, b::adjust(swc.calendar, maturity, swc.bdc)));
  EXPECT_EQ(leg.accrual_start.front(), b::curve_time(vd, settle));
  for (std::size_t i = 1; i < leg.accrual_end.size(); ++i)
    EXPECT_EQ(leg.accrual_start[i], leg.accrual_end[i - 1]) << "contiguous periods, boundary " << i;
  // A SHORT FRONT stub: the first accrual runs from settlement to the first rolled boundary.
  EXPECT_EQ(leg.tau.front(), b::year_frac(swc.float_dc, settle, b::adjust(swc.calendar, d("2027-02-28"), swc.bdc)));
}

TEST(AssetSwapLeg, WeekendMonthEndMaturityEndsOnTheFollowingAdjustedDate) {
  const b::Date vd = d("2026-09-15"), settle = d("2026-09-16"), maturity = d("2032-01-31");  // a Saturday
  const b::SwapConv swc = b::swap_conv(std::string(cvd::require_bond("US-TREASURY").currency), "");
  ASSERT_EQ(swc.float_freq_tok, "1Y") << "premise";
  ASSERT_EQ(swc.bdc, "ModifiedFollowing") << "premise";
  ASSERT_EQ(swc.pay_lag, 2) << "premise";
  ASSERT_FALSE(b::is_business_day(swc.calendar, maturity)) << "premise: a weekend maturity";
  for (const char* bd : {"2027-01-29", "2028-01-31", "2029-01-31", "2030-01-31", "2031-01-31", "2032-02-02", "2032-02-04"})
    ASSERT_TRUE(b::is_business_day(swc.calendar, d(bd))) << "premise: " << bd;
  ASSERT_FALSE(b::is_business_day(swc.calendar, d("2027-01-31"))) << "premise: 2027-01-31 is a Sunday";

  const b::FloatLegTimes leg = b::asset_swap_float_leg(swc, vd, settle, maturity);
  const char* want_end[] = {"2027-01-29", "2028-01-31", "2029-01-31", "2030-01-31", "2031-01-31", "2032-02-02"};
  ASSERT_EQ(leg.accrual_end.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_EQ(leg.accrual_end[i], b::curve_time(vd, d(want_end[i]))) << "boundary " << i << " want " << want_end[i];
  EXPECT_EQ(leg.accrual_start.front(), b::curve_time(vd, settle));
  EXPECT_EQ(leg.pay.back(), b::curve_time(vd, d("2032-02-04"))) << "the last coupon pays 2 business days after Mon 2 Feb";
  EXPECT_EQ(leg.tau.back(), b::year_frac(swc.float_dc, d("2031-01-31"), d("2032-02-02"), swc.calendar))
      << "the last accrual runs to the adjusted redemption date";
}

TEST(AssetSwapLeg, ControlABusinessDayMonthEndMaturityIsUnchanged) {
  // Following == Modified Following when the maturity is a business day: the leg ends ON it.
  const b::Date vd = d("2026-09-15"), settle = d("2026-09-16"), maturity = d("2031-01-31");  // a Friday
  const b::SwapConv swc = b::swap_conv(std::string(cvd::require_bond("US-TREASURY").currency), "");
  ASSERT_TRUE(b::is_business_day(swc.calendar, maturity)) << "premise";
  const b::FloatLegTimes leg = b::asset_swap_float_leg(swc, vd, settle, maturity);
  EXPECT_EQ(leg.accrual_end.back(), b::curve_time(vd, maturity));
}
