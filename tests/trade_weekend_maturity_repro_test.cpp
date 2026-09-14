// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTION (2026-09-14, found while drafting portfolio_verb_oracle_test.cpp): a BOOKED trade whose maturity
// falls on a non-business day kept that raw date as its final accrual end. build::swap_periods_between ends the last
// period at the maturity it is given, and Trade::to_position hands it the booked date unadjusted, so a swap booked to
// its unadjusted anniversary (a Sunday) accrued to the Sunday and paid off it -- where the market (and QuantLib's
// Schedule, which adjusts the termination date by the convention) accrues to the business day the convention rolls
// to. The par_swap builders never hit it: their callers pass an already-adjusted maturity.
//
// The reference is the convention itself: booking the unadjusted Sunday must give EXACTLY the position booked on the
// Modified-Following Monday it rolls to, and the hand numbers below follow from the DB row (asserted as premises).
#include <vector>

#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/trade/trade.hpp"

namespace b = swaps::build;
namespace tr = swaps::trade;

namespace {

const b::Date kValue = b::Date::from_iso("2026-07-08");      // a Wednesday
const b::Date kEffective = b::Date::from_iso("2026-07-10");  // Friday, spot

swaps::portfolio::MultiCurveBook::Position booked(const char* maturity) {
  return tr::Trade::vanilla_swap("T", 1e6, tr::Pay::Fixed, 0.03, "USD", "USD-SOFR", kEffective,
                                 b::Date::from_iso(maturity), 0, 0)
      .to_position(kValue);
}

}  // namespace

TEST(TradeWeekendMaturityRepro, AMaturityOnASundayAccruesToTheBusinessDayItRollsTo) {
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  ASSERT_EQ(conv.fixed_dc, "ACT/360") << "fixture premise";
  ASSERT_EQ(conv.pay_lag, 2) << "fixture premise";
  ASSERT_EQ(conv.bdc, "ModifiedFollowing") << "fixture premise";
  ASSERT_EQ(b::Date::from_iso("2033-07-10").weekday(), 6) << "fixture premise: a Sunday";

  const auto sunday = booked("2033-07-10");
  const auto monday = booked("2033-07-11");  // where Modified Following rolls it
  ASSERT_EQ(sunday.fixed_coupons.size(), monday.fixed_coupons.size());
  ASSERT_EQ(sunday.float_coupons.size(), monday.float_coupons.size());
  // The last period [2032-07-12 (Sat 07-10 rolled), 2033-07-11]: 364 days ACT/360, paid two business days later.
  EXPECT_EQ(monday.fixed_coupons.back().tau, 364.0 / 360.0) << "the control: a business-day maturity";
  EXPECT_EQ(sunday.fixed_coupons.back().tau, 364.0 / 360.0) << "accrued to the raw Sunday";
  EXPECT_EQ(sunday.fixed_coupons.back().pay, b::curve_time(kValue, b::Date::from_iso("2033-07-13")));
  for (std::size_t i = 0; i < sunday.fixed_coupons.size(); ++i) {
    EXPECT_EQ(sunday.fixed_coupons[i].pay, monday.fixed_coupons[i].pay) << "fixed coupon " << i;
    EXPECT_EQ(sunday.fixed_coupons[i].tau, monday.fixed_coupons[i].tau) << "fixed coupon " << i;
  }
  for (std::size_t i = 0; i < sunday.float_coupons.size(); ++i) {
    EXPECT_EQ(sunday.float_coupons[i].pay, monday.float_coupons[i].pay) << "float coupon " << i;
    EXPECT_EQ(sunday.float_coupons[i].tau_pay, monday.float_coupons[i].tau_pay) << "float coupon " << i;
    EXPECT_EQ(sunday.float_coupons[i].obs.sub_end, monday.float_coupons[i].obs.sub_end) << "float coupon " << i;
  }
}

// The ISDA path's edge: with a back stub rolling on the 9th, the last unadjusted boundary (Saturday 2033-07-09) sits one
// day before a Sunday maturity and rolls onto the Monday end. It must not leave a zero-length (or, before the fix, a
// backwards) final period: the schedule is spot .. regular boundaries .. the rolled end, strictly ascending.
TEST(TradeWeekendMaturityRepro, AnIsdaBoundaryRollingOntoTheEndLeavesNoEmptyPeriod) {
  b::ScheduleRule rule;
  rule.roll_dom = 9;  // non-default: the ISDA path
  ASSERT_EQ(b::Date::from_iso("2033-07-09").weekday(), 5) << "fixture premise: a Saturday";
  const auto periods = b::swap_periods_between(b::Date::from_iso("2026-07-09"), "NONE", b::Date::from_iso("2033-07-10"),
                                               "1Y", "ModifiedFollowing", rule);
  ASSERT_EQ(periods.size(), 7u);
  EXPECT_EQ(periods.back().first, b::Date::from_iso("2032-07-09"));
  EXPECT_EQ(periods.back().second, b::Date::from_iso("2033-07-11"));
  for (const auto& [s, e] : periods) EXPECT_LT(s, e) << "every period has a positive span";
}
