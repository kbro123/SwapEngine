// E5 taxonomy: T6 regression (fails on the reverted bug)
// FS1 REPRODUCTION (2026-09-14, found by the matched-maturity multi-currency design): under a FRONT stub the backward
// roll's first interior boundary can sit strictly after spot and still business-day-adjust back ONTO spot. The
// extended ISDA path of build::swap_periods_between kept it, so the schedule opened with a ZERO-LENGTH period and,
// after it, a period running BACKWARDS. A zero tau is refused by the bundle validator and asserted against by the kernel;
// QuantLib's Schedule drops a boundary at or before the first date. Currency-independent: any month-end maturity whose
// spot is the last business day before a weekend month-end. build::asset_swap_float_leg takes the same path.
//
// The reference is the convention written out by hand: USD-SOFR's product row (asserted as premises), spot Fri
// 2026-10-30, maturity Tue 2028-10-31, annual boundaries rolled back on the 31st -- Sat 2026-10-31 adjusts (Modified
// Following) onto spot and is not a boundary; Sun 2027-10-31 adjusts to Fri 2027-10-29.
#include <vector>

#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/par_asset_swap.hpp"
#include "swaps/build/schedule.hpp"

namespace b = swaps::build;

namespace {

b::Date d(const char* iso) { return b::Date::from_iso(iso); }

}  // namespace

TEST(ScheduleFrontStubRepro, ARollDateThatAdjustsOntoSpotIsNotAZeroLengthPeriod) {
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  ASSERT_EQ(conv.bdc, "ModifiedFollowing") << "fixture premise";
  ASSERT_EQ(d("2026-10-31").weekday(), 5) << "fixture premise: a Saturday";
  ASSERT_EQ(b::adjust(conv.calendar, d("2026-10-31"), conv.bdc), d("2026-10-30"))
      << "fixture premise: the first roll date adjusts ONTO spot";
  ASSERT_EQ(b::adjust(conv.calendar, d("2027-10-31"), conv.bdc), d("2027-10-29")) << "fixture premise";
  ASSERT_EQ(b::adjust(conv.calendar, d("2028-10-31"), conv.bdc), d("2028-10-31")) << "fixture premise: a business day";

  b::ScheduleRule front;
  front.side = b::StubSide::Front;
  const std::vector<b::Period> periods =
      b::swap_periods_between(d("2026-10-30"), conv.calendar, d("2028-10-31"), "1Y", conv.bdc, front);
  ASSERT_EQ(periods.size(), 2u) << "spot .. 2027-10-29 .. 2028-10-31";
  EXPECT_EQ(periods[0].first, d("2026-10-30"));
  EXPECT_EQ(periods[0].second, d("2027-10-29"));
  EXPECT_EQ(periods[1].first, d("2027-10-29"));
  EXPECT_EQ(periods[1].second, d("2028-10-31"));
  for (const b::Period& p : periods) EXPECT_LT(p.first, p.second) << "every period runs forward, none is empty";
}

TEST(ScheduleFrontStubRepro, TheAssetSwapFloatLegHasNoZeroAccrual) {
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  const b::Date vd = d("2026-10-28"), settle = d("2026-10-30"), maturity = d("2028-10-31");
  const b::FloatLegTimes leg = b::asset_swap_float_leg(conv, vd, settle, maturity);
  const std::vector<b::Period> want = [&] {
    b::ScheduleRule front;
    front.side = b::StubSide::Front;
    front.roll_dom = 31;
    return b::swap_periods_between(settle, conv.calendar, maturity, conv.float_freq_tok, conv.bdc, front);
  }();
  ASSERT_FALSE(leg.tau.empty());
  ASSERT_EQ(leg.tau.size(), want.size());
  for (std::size_t i = 0; i < leg.tau.size(); ++i) {
    EXPECT_GT(leg.tau[i], 0.0) << "float period " << i << " has a zero or negative accrual";
    if (i > 0) EXPECT_GT(leg.pay[i], leg.pay[i - 1]) << "float payment " << i << " is not after the previous one";
  }
}
