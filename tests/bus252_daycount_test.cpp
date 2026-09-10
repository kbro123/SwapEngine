// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Gate for BUS/252 (Brazilian business/252) day count — the calendar-aware year_frac added so BRL-CDI
// swaps calibrate. Covers: (a) business_days_between on the BRL (B3/ANBIMA) calendar for a known span,
// (b) BUS/252 year_frac over a full calendar year == the exact business-day count / 252 (== 1.0 for 2025),
// (c) a BRL-CDI-style par-swap accrual sanity check through the fixed/float leg builders. Calendar ids come
// from the conventions DB accessors (not hard-coded), like the other build tests. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "swaps/build/instruments.hpp"

namespace b = swaps::build;

// The BRL calendar id, resolved from the conventions DB via the BRL-CDI index (not hard-coded).
static std::string brl_calendar() {
  const std::string cal = b::index_calendar("BRL-CDI");
  EXPECT_EQ(cal, "BRL");  // sanity: the DB really keys the Brazil calendar as "BRL"
  return cal;
}

// (a) Known span: January 2025 on the BRL calendar. Jan 2025 has 23 weekdays; only New Year's Day (Wed
// Jan 1) is a BRL holiday, so [2025-01-01, 2025-02-01) holds exactly 22 business days.
TEST(Bus252, BusinessDaysBetweenKnownSpan) {
  const std::string cal = brl_calendar();
  EXPECT_EQ(b::business_days_between(cal, b::Date::from_iso("2025-01-01"), b::Date::from_iso("2025-02-01")),
            22);
  // Half-open + sign-flip: reversing the span negates, and an empty span is 0.
  EXPECT_EQ(b::business_days_between(cal, b::Date::from_iso("2025-02-01"), b::Date::from_iso("2025-01-01")),
            -22);
  EXPECT_EQ(b::business_days_between(cal, b::Date::from_iso("2025-01-01"), b::Date::from_iso("2025-01-01")),
            0);
}

// (b) A full calendar year: 2025 has exactly 252 BRL business days (261 weekdays − 9 weekday holidays:
// New Year, Carnival Mon/Tue, Good Friday, Tiradentes, Labour Day, Corpus Christi, Black Awareness,
// Christmas), so BUS/252 over the year is exactly 1.0.
TEST(Bus252, YearFracFullYearIsOne) {
  const std::string cal = brl_calendar();
  const b::Date y0 = b::Date::from_iso("2025-01-01"), y1 = b::Date::from_iso("2026-01-01");
  EXPECT_EQ(b::business_days_between(cal, y0, y1), 252);
  EXPECT_NEAR(b::year_frac("BUS/252", y0, y1, cal), 1.0, 1e-12);
  // The 4-arg overload IS business_days_between / 252 for BUS/252 ...
  EXPECT_DOUBLE_EQ(b::year_frac("BUS/252", y0, y1, cal),
                   double(b::business_days_between(cal, y0, y1)) / 252.0);
  // ... and the 3-arg form refuses BUS/252 rather than silently guessing a calendar.
  EXPECT_THROW(b::year_frac("BUS/252", y0, y1), std::invalid_argument);
  // Non-BUS/252 day counts ignore the calendar entirely (byte-identical to the 3-arg form).
  EXPECT_DOUBLE_EQ(b::year_frac("ACT/360", y0, y1, cal), b::year_frac("ACT/360", y0, y1));
}

// (c) BRL-CDI DI×Pre swap: a ZERO-COUPON product (DB products[].zero_coupon) — one BUS/252 accrual spot->maturity on
// each leg, on the BRL calendar; every accrual is an exact integer number of business days / 252. (Its quote kind,
// closed form and compiled/AAD parity are proven in zero_coupon_swap_test.cpp.)
TEST(Bus252, BrlCdiZeroCouponSwapAccrual) {
  const b::Date vd = b::Date::from_iso("2025-01-02");
  const b::SwapConv conv = b::swap_conv("BRL", "BRL-CDI");
  ASSERT_EQ(conv.calendar, "BRL");
  ASSERT_EQ(conv.fixed_dc, "BUS/252");
  ASSERT_EQ(conv.float_dc, "BUS/252");
  ASSERT_TRUE(conv.zero_coupon);
  const b::Date mat = b::resolve("2y", vd, conv.calendar, conv.bdc, conv.spot_lag);
  const auto ins = b::par_swap(vd, conv, mat, 0, 0, 0.12);
  ASSERT_EQ(ins.fixed.coupons.size(), 1u);
  ASSERT_EQ(ins.fwd.coupons.size(), 1u);
  const double bd = ins.fixed.coupons.front().tau * 252.0;   // business-day count implied by the accrual
  EXPECT_NEAR(std::round(bd), bd, 1e-9);                     // BUS/252 => integer/252
  EXPECT_GT(ins.fixed.coupons.front().tau, 1.9);             // ~2 x 250 business days
  EXPECT_LT(ins.fixed.coupons.front().tau, 2.05);
  const double bdi = ins.fwd.coupons.front().obs.tau_index * 252.0;  // telescoped observation, same count
  EXPECT_NEAR(std::round(bdi), bdi, 1e-9);
  EXPECT_DOUBLE_EQ(ins.fwd.coupons.front().tau_pay, ins.fixed.coupons.front().tau);
}
