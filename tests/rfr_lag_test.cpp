// Priority-2 gate: the OPTIONAL RFR observation-timing conventions (observation-shift / lookback / lockout)
// added to build::rfr_observation and threaded through ois_coupon / float_leg. The load-bearing guarantee is
// that an INACTIVE lag (the default) is byte-identical to the plain single telescoped bracket the OIS builders
// have always produced; the active paths move the observation window as expected. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include "swaps/build/instruments.hpp"

namespace b = swaps::build;

// Default (inactive) lag == the plain single telescoped bracket, byte-for-byte.
TEST(RfrLag, InactiveIsByteIdenticalToPlainBracket) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");
  const b::Date s = b::Date::from_iso("2026-08-03"), e = b::Date::from_iso("2026-11-03");

  const b::px::FloatCoupon def = b::ois_coupon(vd, conv, s, e, conv.float_dc);          // no lag arg
  const b::px::FloatCoupon none = b::ois_coupon(vd, conv, s, e, conv.float_dc, b::RfrLag{});
  ASSERT_EQ(def.obs.sub_start.size(), 1u);
  EXPECT_FALSE(def.obs.compounded);
  EXPECT_TRUE(def.obs.weight.empty());
  EXPECT_DOUBLE_EQ(def.obs.sub_start[0], b::curve_time(vd, s));
  EXPECT_DOUBLE_EQ(def.obs.sub_end[0], b::curve_time(vd, e));
  EXPECT_DOUBLE_EQ(def.obs.tau_index, b::year_frac(conv.float_dc, s, e));
  // The explicit-None overload matches the no-arg default bit-for-bit.
  EXPECT_EQ(none.obs.sub_start, def.obs.sub_start);
  EXPECT_EQ(none.obs.sub_end, def.obs.sub_end);
  EXPECT_DOUBLE_EQ(none.obs.tau_index, def.obs.tau_index);

  // The whole float leg is likewise untouched with no lag.
  const auto plain = b::float_leg(vd, conv, e, 0, 0, conv.float_freq_tok, conv.float_dc);
  for (const auto& c : plain.coupons) {
    EXPECT_FALSE(c.obs.compounded);
    EXPECT_EQ(c.obs.sub_start.size(), 1u);
  }
}

// Observation SHIFT: the single bracket telescopes still (compounded=false) but the whole window slides back
// `days` business days, so both endpoints move earlier by exactly that shift.
TEST(RfrLag, ShiftMovesWholeWindowAndStaysTelescoped) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");
  const b::Date s = b::Date::from_iso("2026-08-03"), e = b::Date::from_iso("2026-11-03");

  b::RfrLag lag{b::RfrStyle::Shift, 5, "USD-SOFR"};
  const b::px::FloatCoupon c = b::ois_coupon(vd, conv, s, e, conv.float_dc, lag);
  ASSERT_EQ(c.obs.sub_start.size(), 1u);  // still ONE telescoped bracket
  EXPECT_FALSE(c.obs.compounded);
  EXPECT_TRUE(c.obs.weight.empty());
  const b::Date ss = b::advance_obs_bd("USD-SOFR", s, -5), ee = b::advance_obs_bd("USD-SOFR", e, -5);
  EXPECT_DOUBLE_EQ(c.obs.sub_start[0], b::curve_time(vd, ss));
  EXPECT_DOUBLE_EQ(c.obs.sub_end[0], b::curve_time(vd, ee));
  EXPECT_LT(c.obs.sub_start[0], b::curve_time(vd, s));  // window moved EARLIER
  // Payment accrual is unchanged (actual [s,e] span); only the observation window shifted.
  EXPECT_DOUBLE_EQ(c.tau_pay, b::year_frac(conv.float_dc, s, e));
}

// Lookback (without shift): a genuine daily compounded product, one bracket per business day, each observation
// window looked back `days` BDs while the accrual weight keys off the ACTUAL day.
TEST(RfrLag, LookbackIsDailyCompounded) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");
  const b::Date s = b::Date::from_iso("2026-08-03"), e = b::Date::from_iso("2026-09-01");

  b::RfrLag lag{b::RfrStyle::Lookback, 5, "USD-SOFR"};
  const b::px::FloatCoupon c = b::ois_coupon(vd, conv, s, e, conv.float_dc, lag);
  EXPECT_TRUE(c.obs.compounded);
  const std::size_t nfix = b::business_days(s, e, "USD-SOFR").size();
  ASSERT_GT(nfix, 1u);
  EXPECT_EQ(c.obs.sub_start.size(), nfix);
  EXPECT_EQ(c.obs.weight.size(), nfix);
  // The first day's observation window is looked back 5 BDs from the first fixing day.
  const auto days = b::business_days(s, e, "USD-SOFR");
  const b::Date look0 = b::advance_obs_bd("USD-SOFR", days[0], -5);
  EXPECT_DOUBLE_EQ(c.obs.sub_start[0], b::curve_time(vd, look0));
  EXPECT_LT(c.obs.sub_start[0], b::curve_time(vd, days[0]));  // observed earlier than the accrual day
  EXPECT_DOUBLE_EQ(c.obs.tau_index, b::year_frac(conv.float_dc, s, e));
}

// Lockout: daily compounded, but the last `days` business days FREEZE to the lockout-start day's observation
// window (identical sub-period), while each still carries its own actual-accrual weight.
TEST(RfrLag, LockoutFreezesTailWindow) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");
  const b::Date s = b::Date::from_iso("2026-08-03"), e = b::Date::from_iso("2026-09-01");

  const int lock = 3;
  b::RfrLag lag{b::RfrStyle::Lockout, lock, "USD-SOFR"};
  const b::px::FloatCoupon c = b::ois_coupon(vd, conv, s, e, conv.float_dc, lag);
  EXPECT_TRUE(c.obs.compounded);
  const std::size_t n = c.obs.sub_start.size();
  ASSERT_GT(n, std::size_t(lock));
  // The final `lock` brackets share one frozen window.
  for (std::size_t i = n - lock; i < n; ++i) {
    EXPECT_DOUBLE_EQ(c.obs.sub_start[i], c.obs.sub_start[n - lock]) << "frozen start " << i;
    EXPECT_DOUBLE_EQ(c.obs.sub_end[i], c.obs.sub_end[n - lock]) << "frozen end " << i;
  }
  // A day BEFORE the lockout still observes its own (distinct) window.
  EXPECT_NE(c.obs.sub_start[n - lock - 1], c.obs.sub_start[n - lock]);
}
