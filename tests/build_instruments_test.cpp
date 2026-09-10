// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
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
  // CORRECTED 2026-09-10 (item 17): a plain averaged leg observes each fixing over exactly the window it
  // accrues, so every weight is 1 and the vector is stored empty. It used to hold 21 copies of 1.0138888889
  // = 365/360 — the index day count divided by CURVE time instead of by the index year-fraction of the
  // observation window — which made every averaged overnight rate 1.389 % too high. That number was pinned
  // here as "C++/Python parity"; the web compiler still produces it (TASKS-API §A0.5).
  EXPECT_TRUE(avg.weight.empty()) << "observation window == accrual window => unit weights";
  EXPECT_NEAR(avg.sub_start.front(), 0.0712328767, 1e-10);
  EXPECT_NEAR(avg.sub_end.back(), 0.1506849315, 1e-10);

  // 3M compounded future — IMM U27..Z27, single telescoped bracket.
  const auto cmp = b::observation(vd, b::resolve("U27", vd, "NONE", "Following", 0), b::resolve("Z27", vd, "NONE", "Following", 0), "compounded", 0.0,
                                  "ACT/360", "USD-SOFR");
  ASSERT_EQ(cmp.sub_start.size(), 1u);
  EXPECT_TRUE(cmp.weight.empty());
  EXPECT_NEAR(cmp.sub_start[0], 1.1890410959, 1e-10);
  EXPECT_NEAR(cmp.sub_end[0], 1.4383561644, 1e-10);
  EXPECT_NEAR(cmp.tau_index, 0.2527777778, 1e-10);
}

TEST(BuildInstruments, OisSwapScheduleMatchesPython) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  const b::Date mat = b::resolve("5y", vd, "NONE", "Following", 0);

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

// Item 17 (E4.E L2, 2026-09-10): the float-leg builder reads products[].float_leg.compounding. Fed Funds is
// quoted BOTH ways off the SAME daily fixings — a fixed-vs-FF OIS COMPOUNDS them (ISDA OIS-COMPOUND) while the
// FF/SOFR basis leg is the H.15 ARITHMETIC AVERAGE — and until now every overnight leg the builders produced
// was compounded, so an averaged product priced as a compounded one. The two must differ by Jensen's gap and
// no more: over one 3M period on a 4 % flat-forward curve, compounding gains ~r²τ²/2 ≈ 0.1 bp of rate.
TEST(BuildInstruments, FloatLegHonoursTheProductsCompoundingConvention) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::Date s = b::Date::from_iso("2026-09-01"), e = b::Date::from_iso("2026-12-01");

  const b::SwapConv ois = b::swap_conv("USD", "USD-FEDFUNDS");          // USD-FEDFUNDS-OIS: compounded
  const b::SwapConv basis = b::conv_from_product(                        // USD-FF-SOFR-BASIS: averaged FF leg
      swaps::conventions::require_product("USD-FF-SOFR-BASIS"));
  ASSERT_EQ(ois.float_compounding, "compounded");
  ASSERT_EQ(basis.float_compounding, "averaged");

  const auto c_comp = b::ois_coupon(vd, ois, s, e, ois.float_dc);
  const auto c_avg = b::ois_coupon(vd, basis, s, e, basis.float_dc);

  // Shape: compounding telescopes to ONE bracket with no day-count moments; averaging carries the moments
  // that turn the bracket into the arithmetic mean (the W-cacheable moment path).
  EXPECT_EQ(c_comp.obs.sub_start.size(), 1u);
  EXPECT_EQ(c_comp.obs.fixing_step, 0.0);
  EXPECT_EQ(c_avg.obs.sub_start.size(), 1u);
  EXPECT_GT(c_avg.obs.fixing_step, 0.0) << "an averaged leg must carry the calendar's day-count moments";

  // Value: price both off one flat 4 % continuously-compounded forward curve.
  auto curve = swaps::curve::make_modular_curve<double>(
      {swaps::curve::CurveModule{{0.25, 1.0, 2.0, 5.0}, swaps::curve::Scheme::Flat}});
  curve.set_forwards(Eigen::VectorXd::Constant(4, 0.04));
  const double r_comp = swaps::pricing::rate<double>(c_comp.obs, curve);
  const double r_avg = swaps::pricing::rate<double>(c_avg.obs, curve);

  // The EXACT daily arithmetic average over the same window — what the moment expansion approximates.
  const auto daily = b::observation(vd, s, e, "averaged", 0.0, basis.float_dc, basis.calendar);
  const double r_daily = swaps::pricing::rate<double>(daily, curve);

  const double tau = c_comp.tau_pay;
  std::cout << "  [compounding] 3M FF: compounded " << r_comp * 1e4 << " bp, averaged (moment) " << r_avg * 1e4
            << " bp, averaged (exact daily) " << r_daily * 1e4 << " bp, Jensen gap "
            << (r_comp - r_avg) * 1e4 << " bp\n";

  EXPECT_NEAR(r_avg, r_daily, 5e-9 * r_daily) << "the moment path IS the arithmetic average (docs Part B)";
  EXPECT_GT(r_comp, r_avg) << "compounding the same fixings must beat averaging them (Jensen)";
  EXPECT_NEAR(r_comp - r_avg, 0.5 * 0.04 * 0.04 * tau, 0.15 * 0.5 * 0.04 * 0.04 * tau)
      << "and the gap is the r^2*tau/2 convexity of compounding, not an arbitrary difference";
}
