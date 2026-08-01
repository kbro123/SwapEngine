// Golden-value gate for the build/ observation + coupon builders (observations.hpp, instruments.hpp,
// conventions.hpp). Expected values are the output of the Python compiler (server/conventions.py observation()
// + compile.py _fixed_coupons/_float_leg), pinning C++/Python parity for the subtle averaging/compounding
// windows and the OIS swap schedule. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include "swaps/build/instruments.hpp"

namespace b = swaps::build;

TEST(BuildInstruments, ObservationWindowsMatchPython) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  // 1M averaged future — whole calendar month Aug-2026, SOFR calendar.
  const auto avg = b::observation(vd, b::Date::from_iso("2026-08-01"), b::Date::from_iso("2026-09-01"),
                                  "averaged", 0.0, "ACT/360", "USD-SOFR");
  EXPECT_EQ(avg.sub_start.size(), 21u);
  EXPECT_NEAR(avg.tau_index, 0.0861111111, 1e-10);
  ASSERT_EQ(avg.weight.size(), 21u);  // non-trivial weights (ACT/360 vs ACT/365F)
  EXPECT_NEAR(avg.weight.front(), 1.0138888889, 1e-10);
  EXPECT_NEAR(avg.weight.back(), 1.0138888889, 1e-10);
  EXPECT_NEAR(avg.sub_start.front(), 0.0712328767, 1e-10);
  EXPECT_NEAR(avg.sub_end.back(), 0.1506849315, 1e-10);

  // 3M compounded future — IMM U27..Z27, single telescoped bracket.
  const auto cmp = b::observation(vd, b::resolve("U27", vd), b::resolve("Z27", vd), "compounded", 0.0,
                                  "ACT/360", "USD-SOFR");
  ASSERT_EQ(cmp.sub_start.size(), 1u);
  EXPECT_TRUE(cmp.weight.empty());
  EXPECT_NEAR(cmp.sub_start[0], 1.1890410959, 1e-10);
  EXPECT_NEAR(cmp.sub_end[0], 1.4383561644, 1e-10);
  EXPECT_NEAR(cmp.tau_index, 0.2527777778, 1e-10);
}

TEST(BuildInstruments, OisSwapScheduleMatchesPython) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", 1.0, "USD-SOFR");
  const b::Date mat = b::resolve("5y", vd);

  const auto fixed = b::fixed_coupons(vd, conv, mat, 0);
  ASSERT_EQ(fixed.coupons.size(), 5u);
  EXPECT_NEAR(fixed.coupons.front().pay, 1.0164383562, 1e-10);
  EXPECT_NEAR(fixed.coupons.front().tau, 1.0194444444, 1e-10);
  EXPECT_NEAR(fixed.coupons.back().pay, 5.0082191781, 1e-10);
  EXPECT_NEAR(fixed.coupons.back().tau, 1.0083333333, 1e-10);

  const auto flt = b::float_leg(vd, conv, mat, 0, 0, conv.float_freq_tok, conv.float_dc);
  ASSERT_EQ(flt.coupons.size(), 5u);
  EXPECT_NEAR(flt.coupons.front().obs.sub_start[0], 0.0054794521, 1e-10);
  EXPECT_NEAR(flt.coupons.front().obs.sub_end[0], 1.0109589041, 1e-10);
  EXPECT_NEAR(flt.coupons.front().pay, 1.0164383562, 1e-10);
  EXPECT_NEAR(flt.coupons.back().pay, 5.0082191781, 1e-10);
  EXPECT_NEAR(flt.coupons.back().tau_pay, 1.0083333333, 1e-10);
}
