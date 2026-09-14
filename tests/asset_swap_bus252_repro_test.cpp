// E5 taxonomy: T6 regression (fails on the reverted bug) | T5 value pin (hand business-day count)
// AS2 REPRODUCTION (2026-09-14). build::asset_swap_float_leg computes each float accrual with the 3-argument year_frac,
// which THROWS for BUS/252: that day count needs the product calendar (day_count.hpp refuses to guess one). No request
// reaches it today -- bond rows are USD only (O9) -- so it is pinned by a direct builder call on a BUS/252 conventions
// object, the case the first BRL bond row will hit.
//
// Expected, by hand: Brazilian business days d with 2026-09-14 <= d < 2026-12-14 (BUS/252's half-open count). The span
// is exactly 13 weeks = 65 weekdays, less the three national holidays that fall on weekdays inside it -- Our Lady of
// Aparecida (Mon 10-12), All Souls' Day (Mon 11-02), Black Awareness Day (Fri 11-20, national since 2024); Republic
// Proclamation Day (11-15) is a Sunday -- = 62, so tau = 62/252. The holiday premises are asserted.
#include <gtest/gtest.h>

#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/par_asset_swap.hpp"

namespace b = swaps::build;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }
}  // namespace

TEST(AssetSwapBus252Repro, FloatAccrualCountsBrazilianBusinessDays) {
  b::SwapConv swc;
  swc.product_id = "AS2-BUS252-FIXTURE";
  swc.calendar = "BRL";
  swc.bdc = "Following";
  swc.float_dc = "BUS/252";
  swc.fixed_dc = "BUS/252";
  swc.float_freq_tok = "3M";
  swc.fixed_freq_tok = "3M";
  swc.spot_lag = 0;
  swc.pay_lag = 0;
  const b::Date start = d("2026-09-14"), end = d("2026-12-14");
  ASSERT_TRUE(b::is_business_day(swc.calendar, start)) << "premise: the start is a B3 business day";
  ASSERT_TRUE(b::is_business_day(swc.calendar, end)) << "premise: the end needs no adjustment";
  for (const char* h : {"2026-10-12", "2026-11-02", "2026-11-20"})
    ASSERT_FALSE(b::is_business_day(swc.calendar, d(h))) << "premise: national holiday " << h;

  const b::FloatLegTimes leg = b::asset_swap_float_leg(swc, start, start, end);
  ASSERT_EQ(leg.tau.size(), 1u) << "one 3M period, no stub";
  EXPECT_DOUBLE_EQ(leg.tau[0], 62.0 / 252.0);
}
