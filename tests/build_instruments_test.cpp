// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Golden-value gate for the build/ observation + coupon builders (observations.hpp, instruments.hpp,
// conventions.hpp). Expected values are the output of the Python compiler (server/conventions.py observation()
// + compile.py _fixed_coupons/_float_leg), pinning C++/Python parity for the subtle averaging/compounding
// windows and the OIS swap schedule. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include <cmath>

#include "swaps/build/calendar.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace b = swaps::build;

TEST(BuildInstruments, ObservationWindowsMatchPython) {
  const b::Date vd = b::Date::from_iso("2026-07-08");
  // 1M averaged future — whole calendar month Aug-2026, SOFR calendar.
  const auto avg = b::observation(vd, b::Date::from_iso("2026-08-01"), b::Date::from_iso("2026-09-01"),
                                  "averaged", 0.0, "ACT/360", "USD-SOFR");
  // CORRECTED AGAIN 2026-09-10 (E3): 22 fixings, not 21. 1 August 2026 is a SATURDAY, and the rate that
  // applies to it is the fixing published for Friday 31 July, whose overnight window runs 31 Jul -> 3 Aug
  // and earns this window 2 of its 3 days. Both compilers used to enumerate only the business days INSIDE
  // [1 Aug, 1 Sep), so 1-2 August contributed nothing to the sum while tau_index still spanned the whole
  // month: exactly 29/31 of the correct rate, -25 bp on a flat 4 % curve. A CME 30-day Fed funds future
  // references the CALENDAR month, so about a third of contract months start this way.
  EXPECT_EQ(avg.sub_start.size(), 22u);
  EXPECT_NEAR(avg.tau_index, 0.0861111111, 1e-10);
  // CORRECTED 2026-09-10 (item 17): a plain averaged leg observes each fixing over exactly the window it
  // accrues, so every weight is 1 and the vector is stored empty. It used to hold 21 copies of 1.0138888889
  // = 365/360 — the index day count divided by CURVE time instead of by the index year-fraction of the
  // observation window — which made every averaged overnight rate 1.389 % too high. That number was pinned
  // here as "C++/Python parity"; the web compiler still produces it (TASKS-API §A0.5).
  // ...so the weights are NOT all one here: the leading fixing is earned 2/3, every interior day in full.
  ASSERT_EQ(avg.weight.size(), 22u) << "a partly-earned leading fixing means the weights are kept";
  EXPECT_NEAR(avg.weight[0], 2.0 / 3.0, 1e-12) << "31 Jul's fixing runs to 3 Aug; the window earns 1-2 Aug";
  for (std::size_t i = 1; i < avg.weight.size(); ++i)
    EXPECT_NEAR(avg.weight[i], 1.0, 1e-12) << "interior day " << i << " earns its whole fixing";
  EXPECT_NEAR(avg.sub_start.front(), 0.0630136986, 1e-10) << "31 Jul (23/365), not 3 Aug (26/365)";
  EXPECT_NEAR(avg.sub_end.back(), 0.1506849315, 1e-10);

  // The same month with a BUSINESS-DAY start is untouched by the fix -- 21 fixings, all weights one. That is
  // what makes E3 a pure bug fix: every schedule-generated accrual period is business-day adjusted already.
  const auto adj = b::observation(vd, b::Date::from_iso("2026-08-03"), b::Date::from_iso("2026-09-01"),
                                  "averaged", 0.0, "ACT/360", "USD-SOFR");
  EXPECT_EQ(adj.sub_start.size(), 21u);
  EXPECT_TRUE(adj.weight.empty()) << "observation window == accrual window => unit weights";
  EXPECT_NEAR(adj.sub_start.front(), 0.0712328767, 1e-10);

  // 3M compounded future — IMM U27..Z27, single telescoped bracket.
  const auto cmp = b::observation(vd, b::resolve("U27", vd, "NONE", "Following", 0), b::resolve("Z27", vd, "NONE", "Following", 0), "compounded", 0.0,
                                  "ACT/360", "USD-SOFR");
  ASSERT_EQ(cmp.sub_start.size(), 1u);
  EXPECT_TRUE(cmp.weight.empty());
  EXPECT_NEAR(cmp.sub_start[0], 1.1890410959, 1e-10);
  EXPECT_NEAR(cmp.sub_end[0], 1.4383561644, 1e-10);
  EXPECT_NEAR(cmp.tau_index, 0.2527777778, 1e-10);
}

// E3 (2026-09-10), stated as the invariant rather than as pinned digits: on a FLAT curve the arithmetic
// average of the daily fixings is the same number whatever day the window opens on -- a weekend start does
// not make the month cheaper. Before the fix a Saturday-start month priced at 29/31 of the correct rate and
// a Sunday-start month at 29/30, because the leading non-business days were dropped from the sum while
// tau_index still spanned the whole window. Independent of QuantLib and of any pinned digit.
TEST(BuildInstruments, AnAveragedWindowEarnsEveryDayWhateverDayItOpensOn) {
  struct Flat {
    double f;
    double discount(double t) const { return std::exp(-f * t); }
    double forward(double t) const { (void)t; return f; }
    double integral(double t) const { return f * t; }
  };
  const Flat curve{0.04};
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const std::string cal = "USD-FED";
  // The REFERENCE is the contract definition, computed here day by day: a 30-day Fed funds future settles on
  // the arithmetic mean of the daily EFFR over the CALENDAR month, each calendar day carrying the fixing of
  // the business day on or before it (so a Friday fixing counts three times). Deriving it from the curve
  // independently of the builder is the point -- a closed form for the 1-day rate would be wrong, because a
  // weekend fixing compounds over three days and sits slightly above it.
  const auto h15_mean = [&](const b::Date& start, const b::Date& end) {
    double sum = 0.0;
    int n = 0;
    for (b::Date d = start; d < end; d = d.plus_days(1)) {
      b::Date f = d;
      while (!b::is_business_day(cal, f)) f = f.plus_days(-1);  // the fixing that applies on this day
      b::Date nxt = f.plus_days(1);
      while (!b::is_business_day(cal, nxt)) nxt = nxt.plus_days(1);
      const double g = curve.discount(b::curve_time(vd, f)) / curve.discount(b::curve_time(vd, nxt)) - 1.0;
      sum += g / b::year_frac("ACT/360", f, nxt, cal);  // that day's annualised fixing
      ++n;
    }
    return sum / n;
  };
  struct W { const char* s; const char* e; const char* opens; };
  const W windows[] = {
      {"2026-08-01", "2026-09-01", "Saturday"},   // the CME contract month; was 29/31 of `want`
      {"2026-11-01", "2026-12-01", "Sunday"},     // was 29/30
      {"2026-09-01", "2026-10-01", "Tuesday"},    // always correct
      {"2026-10-01", "2026-11-01", "Thursday"},   // always correct
      {"2026-08-03", "2026-09-01", "Monday"},     // the adjusted start: byte-identical before and after
  };
  for (const W& w : windows) {
    const auto o = b::observation(vd, b::Date::from_iso(w.s), b::Date::from_iso(w.e), "averaged", 0.0,
                                  "ACT/360", "USD-FED");
    const double got = swaps::pricing::rate<double>(o, curve);
    const double want = h15_mean(b::Date::from_iso(w.s), b::Date::from_iso(w.e));
    EXPECT_NEAR(got, want, 1e-12) << w.s << " opens on a " << w.opens << ": " << (got - want) * 1e4 << " bp off";
  }
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
