// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// build::asset_swap_float_leg (E7 stage 3.5): the float leg the asset_swap verb receives. Every boundary is rolled
// BACKWARD from the maturity on the maturity's own day of month (a short front stub from settlement), so a leap-day
// or month-end maturity keeps its day in every year. Until 2026-09-13 a hand loop chained the roll from the previous
// date and a 2036-02-29 maturity got a 2028-02-28 boundary (tests/asset_swap_oracle_test.cpp shows it against
// QuantLib; this file pins the schedule itself, header-only, so tools/mutate.py reaches it).
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
