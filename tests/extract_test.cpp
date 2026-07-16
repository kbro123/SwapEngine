// Design §5 gate: the GENERIC extractors (swaps/ql/extract.hpp) vs QuantLib's own pricing.
//
// The extractor is the ONLY QuantLib-touching layer, and it dispatches on QuantLib COUPON TYPE
// alone. So the acceptance criterion is: for every coupon type, QuantLib pricing a leg off our curve
// and OUR kernel pricing the EXTRACTED data off the same curve must agree. Any gap is an extraction
// bug (the curve is identical by construction -- CurveTermStructure hands QuantLib our own DFs).
//
// Every index/currency/calendar/convention below lives HERE, in the test, and none of it appears in
// include/ (design §6). SOFR vs EURIBOR, compounded vs averaged, is nothing but different data.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace rm = swaps::refmkt;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;

namespace {

::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure()
                       << "got=" << got << " want=" << want << " rel=" << err << " > " << rel);
}

using TS = qlx::CurveTermStructure<rb::Curve>;

// Fixture: the reference market + curve, wired in as QuantLib's term structure, PLUS a second,
// deliberately different curve so forecast != discount is a real distinction (multi-curve).
struct Extract : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;      // discount (and single-curve forecast)
  RelinkableHandle<YieldTermStructure> hfwd;   // a separate forecast curve
  rb::Market mk = rb::build_market(h);
  rb::Curve curve = rb::reference_curve(mk);
  rb::Curve fwd_curve = rb::reference_curve(mk);
  ext::shared_ptr<TS> ts, ts_fwd;

  void SetUp() override {
    // Forecast curve = reference + 25bp on every knot forward: a genuine basis, so a multi-curve
    // test that accidentally used one curve for both roles would fail loudly.
    std::vector<double> x(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
    x.insert(x.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
    for (double& v : x) v += 0.0025;
    fwd_curve.set_forwards(x);

    ts = ext::make_shared<TS>(mk.today, mk.dc, &curve);
    ts->enableExtrapolation();
    ts_fwd = ext::make_shared<TS>(mk.today, mk.dc, &fwd_curve);
    ts_fwd->enableExtrapolation();
    h.linkTo(ts);
    hfwd.linkTo(ts_fwd);
    for (auto& s : mk.swaps) s->deepUpdate();
  }

  // QuantLib's own NPV of a leg off `d`, per unit notional (legs below are built with notional 1).
  double ql_leg_npv(const Leg& leg, const ext::shared_ptr<TS>& d) const {
    double npv = 0.0;
    for (const auto& cf : leg) npv += cf->amount() * d->discount(cf->date());
    return npv;
  }
};

}  // namespace

// ---- 1. Compounded overnight coupons: design §6.1, numbers must not move -----------------------
// The generic extractor must reproduce the legacy `extract_ois_swap` path EXACTLY on the existing
// reference market -- and both must equal QuantLib's fairRate.
TEST_F(Extract, OvernightCompoundedSwapMatchesQuantLibAndLegacy) {
  double worst = 0.0;
  for (const auto& swap : mk.swaps) {
    const double ql = swap->fairRate();
    const auto fl = qlx::extract_float_leg(swap->overnightLeg(), mk.today, mk.dc);
    const auto fx = qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
    const double ours = px::par_rate<double>(fl, fx, curve, curve);
    EXPECT_TRUE(close(ours, ql, swaps::tol::curve_rel)) << " maturity " << swap->maturityDate();

    // ... and the legacy shape is preserved: one unit-weight sub-period, nothing realized, no
    // spread, tau_pay == tau_index (=> k == 1, konst == 0: the compiled engine's fused fast path).
    for (const auto& c : fl) {
      ASSERT_EQ(c.obs.sub_start.size(), 1u);
      EXPECT_TRUE(c.obs.weight.empty());
      EXPECT_EQ(c.obs.realized, 0.0);
      EXPECT_EQ(c.spread, 0.0);
      EXPECT_EQ(c.tau_pay, c.obs.tau_index);
    }
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [ois generic] max |ours - QuantLib| = " << worst << "\n";
}

// ---- 2. Averaged overnight coupons: the per-business-day shape ---------------------------------
// RateAveraging::Simple, non-telescopic (QuantLib's EXACT arithmetic path; telescopic would select
// its Takada log-approximation, a different model -- see the extractor's note).
TEST_F(Extract, OvernightAveragedLegMatchesQuantLib) {
  Schedule sch = MakeSchedule()
                     .from(mk.today + 2)
                     .to(mk.today + 2 + Period(1, Years))
                     .withTenor(Period(3, Months))
                     .withCalendar(mk.sofr->fixingCalendar())
                     .withConvention(ModifiedFollowing)
                     .backwards();
  Leg leg = OvernightLeg(sch, mk.sofr)
                .withNotionals(1.0)
                .withAveragingMethod(RateAveraging::Simple)
                .withTelescopicValueDates(false);

  const auto fl = qlx::extract_float_leg(leg, mk.today, mk.dc);
  ASSERT_EQ(fl.size(), leg.size());
  EXPECT_GT(fl[0].obs.sub_start.size(), 50u);  // one sub-period per business day, not telescoped
  EXPECT_TRUE(close(px::float_leg_pv<double>(fl, curve, curve), ql_leg_npv(leg, ts),
                    swaps::tol::curve_rel));
}

// ---- 3./6. Vanilla IBOR swap, single-curve, 30/360 fixed vs ACT/360 float -----------------------
// MakeVanillaSwap's EUR conventions are 30/360 on the fixed leg and ACT/360 on the float leg, so a
// par rate matching to 1e-10 IS the mixed-day-count gate: the two legs' accruals come from their own
// day counters (never from the curve's) and swapping them would show up here immediately.
TEST_F(Extract, VanillaIborSwapSingleCurveMatchesQuantLib) {
  auto index = ext::make_shared<Euribor3M>(h);
  ext::shared_ptr<VanillaSwap> swap =
      MakeVanillaSwap(Period(7, Years), index, 0.03).withDiscountingTermStructure(h);

  const auto fl = qlx::extract_float_leg(swap->floatingLeg(), mk.today, mk.dc);
  const auto fx = qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);

  // The legs' accruals really are on different day counts (30/360 vs ACT/360) -- else this test
  // would prove nothing about the separation. A same-period ACT/360 30/360 pair cannot coincide.
  EXPECT_NE(fx[0].tau, fl[0].tau_pay);
  // Standard vanilla shape: the float coupon's payment accrual IS the index's spanning time, so
  // k == 1 and the coupon stays on the compiled engine's fused fast path.
  EXPECT_EQ(fl[0].tau_pay, fl[0].obs.tau_index);
  EXPECT_TRUE(close(px::par_rate<double>(fl, fx, curve, curve), swap->fairRate(),
                    swaps::tol::curve_rel));
}

// ---- 6. tau_pay != tau_index on ONE coupon (the k != 1 path) -----------------------------------
// The float leg above pays on the same day count its index observes, so tau_pay == tau_index and the
// distinction is invisible. Force them apart: an ACT/360 index paying on an ACT/365F accrual. The
// rate is still the index's (denominator = spanning time on ACT/360) but the amount accrues on
// ACT/365F -- exactly the separation the extractor must preserve, and ~1.4% of the coupon.
TEST_F(Extract, FloatCouponWithPaymentDayCountDifferentFromIndexMatchesQuantLib) {
  auto index = ext::make_shared<Euribor3M>(hfwd);
  const Date start = index->fixingCalendar().advance(mk.today, 2, Days);
  Schedule sch = MakeSchedule()
                     .from(start)
                     .to(start + Period(5, Years))
                     .withTenor(Period(3, Months))
                     .withCalendar(index->fixingCalendar())
                     .withConvention(ModifiedFollowing)
                     .backwards();
  Leg leg = IborLeg(sch, index).withNotionals(1.0).withPaymentDayCounter(Actual365Fixed());

  const auto fl = qlx::extract_float_leg(leg, mk.today, mk.dc);
  EXPECT_NE(fl[0].tau_pay, fl[0].obs.tau_index);  // ACT/365F payment vs ACT/360 index
  EXPECT_NEAR(fl[0].tau_pay / fl[0].obs.tau_index, 360.0 / 365.0, 1e-12);
  EXPECT_TRUE(close(px::float_leg_pv<double>(fl, fwd_curve, curve), ql_leg_npv(leg, ts),
                    swaps::tol::curve_rel));
}

TEST_F(Extract, VanillaIborSwapMultiCurveMatchesQuantLib) {
  auto index = ext::make_shared<Euribor3M>(hfwd);  // forecast on the +25bp curve
  ext::shared_ptr<VanillaSwap> swap =
      MakeVanillaSwap(Period(7, Years), index, 0.03).withDiscountingTermStructure(h);  // discount

  const auto fl = qlx::extract_float_leg(swap->floatingLeg(), mk.today, mk.dc);
  const auto fx = qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
  const double ours = px::par_rate<double>(fl, fx, fwd_curve, curve);
  EXPECT_TRUE(close(ours, swap->fairRate(), swaps::tol::curve_rel));

  // Negative control: pricing it single-curve must NOT accidentally agree.
  EXPECT_GT(std::abs(px::par_rate<double>(fl, fx, curve, curve) - ours), 1e-4);
}

// ---- 5. A float coupon WITH a spread ----------------------------------------------------------
TEST_F(Extract, IborLegWithSpreadAndGearingMatchesQuantLib) {
  auto index = ext::make_shared<Euribor3M>(hfwd);
  Schedule sch = MakeSchedule()
                     .from(index->fixingCalendar().advance(mk.today, 2, Days))
                     .to(index->fixingCalendar().advance(mk.today, 2, Days) + Period(5, Years))
                     .withTenor(Period(3, Months))
                     .withCalendar(index->fixingCalendar())
                     .withConvention(ModifiedFollowing)
                     .backwards();

  // Spread only: the standard shape, must stay on the unit-weight path.
  Leg spread_leg = IborLeg(sch, index).withNotionals(1.0).withSpreads(0.0035);
  const auto fs = qlx::extract_float_leg(spread_leg, mk.today, mk.dc);
  for (const auto& c : fs) {
    EXPECT_TRUE(c.obs.weight.empty());
    EXPECT_EQ(c.spread, 0.0035);
  }
  EXPECT_TRUE(close(px::float_leg_pv<double>(fs, fwd_curve, curve), ql_leg_npv(spread_leg, ts),
                    swaps::tol::curve_rel));

  // Gearing + spread: gearing folds into the observation WEIGHTS (there is no gearing field).
  Leg geared_leg = IborLeg(sch, index).withNotionals(1.0).withGearings(1.5).withSpreads(-0.0020);
  const auto fg = qlx::extract_float_leg(geared_leg, mk.today, mk.dc);
  for (const auto& c : fg) {
    ASSERT_EQ(c.obs.weight.size(), 1u);
    EXPECT_EQ(c.obs.weight[0], 1.5);
  }
  EXPECT_TRUE(close(px::float_leg_pv<double>(fg, fwd_curve, curve), ql_leg_npv(geared_leg, ts),
                    swaps::tol::curve_rel));
}

// ---- 9. Already-fixed / partially-fixed coupons (realized) -------------------------------------
// A leg that started in the past: its first IBOR coupon is fixed (a constant, no sub-periods) while
// its siblings are live. The overnight leg over the same window is PARTIALLY fixed -- the
// multiplicative already-fixed compound factor folds into weight = P, realized = P - 1.
TEST_F(Extract, FixedAndPartiallyFixedCouponsMatchQuantLib) {
  auto index = ext::make_shared<Euribor3M>(hfwd);
  const Date start = index->fixingCalendar().advance(mk.today, -1, Months);
  Schedule sch = MakeSchedule()
                     .from(start)
                     .to(start + Period(1, Years))
                     .withTenor(Period(3, Months))
                     .withCalendar(index->fixingCalendar())
                     .withConvention(ModifiedFollowing)
                     .backwards();
  Leg leg = IborLeg(sch, index).withNotionals(1.0);
  // The first coupon's fixing is in the past: publish it (a market fact, hence test-side).
  const Date fd = ext::dynamic_pointer_cast<IborCoupon>(leg.front())->fixingDate();
  ASSERT_LT(fd, mk.today);
  index->addFixing(fd, 0.0325);

  const auto fl = qlx::extract_float_leg(leg, mk.today, mk.dc);
  EXPECT_TRUE(fl[0].obs.sub_start.empty());  // fully fixed => a constant, zero derivative
  EXPECT_NE(fl[0].obs.realized, 0.0);
  EXPECT_FALSE(fl[1].obs.sub_start.empty());  // its live sibling
  EXPECT_NEAR(fl[0].obs.realized / fl[0].obs.tau_index, 0.0325, 1e-15);
  EXPECT_TRUE(close(px::float_leg_pv<double>(fl, fwd_curve, curve), ql_leg_npv(leg, ts),
                    swaps::tol::curve_rel));

  // Partially-fixed overnight coupon: ONE coupon straddling today, so its first days are fixed
  // (build_market seeded SOFR over the last 20 days) and the rest are still live.
  const Date ostart = mk.sofr->fixingCalendar().adjust(mk.today - 10);
  const Date oend = mk.sofr->fixingCalendar().advance(mk.today, 3, Months);
  ASSERT_LT(ostart, mk.today);
  Leg oleg{ext::make_shared<OvernightIndexedCoupon>(oend, 1.0, ostart, oend, mk.sofr)};
  const auto ofl = qlx::extract_float_leg(oleg, mk.today, mk.dc);
  ASSERT_EQ(ofl[0].obs.sub_start.size(), 1u);   // still telescoped: compounding is multiplicative
  ASSERT_EQ(ofl[0].obs.weight.size(), 1u);      // weight = the already-fixed compound factor P
  EXPECT_GT(ofl[0].obs.weight[0], 1.0);
  EXPECT_NEAR(ofl[0].obs.realized, ofl[0].obs.weight[0] - 1.0, 1e-15);  // realized == P - 1
  EXPECT_TRUE(close(px::float_leg_pv<double>(ofl, curve, curve), ql_leg_npv(oleg, ts),
                    swaps::tol::curve_rel));
}

// ---- 8. A generic (single-sub-period) future off caller-supplied dates --------------------------
// The engine never learns what a "future" is: the caller hands over sub-periods, the realized part,
// the denominator and a convexity NUMBER. Here that reproduces the legacy 3M compounded SOFR future.
TEST_F(Extract, GenericObservationReproducesCompoundedFuture) {
  for (const auto& f : mk.futures) {
    if (!f.quarterly) continue;
    OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Compound);
    const double ql_rate = 1.0 - qlf.NPV() / 100.0;
    const auto o = qlx::make_observation({{f.start, f.end}}, 0.0,
                                         mk.sofr->dayCounter().yearFraction(f.start, f.end),
                                         mk.today, mk.dc);
    EXPECT_TRUE(close(px::future_rate<double>(o, 0.0, curve), ql_rate, swaps::tol::curve_rel));
  }
}

// ---- 8. An IBOR future: the SAME path, a different index family --------------------------------
// The test above drives the generic future path with an OVERNIGHT contract. Design §1 lists the IBOR
// future as the other single-sub-period shape on that path, and it is a genuinely different market:
// the sub-period is the INDEX's fixing period (value -> maturity date), not a contract accrual, and
// the denominator is the index's own accrual over it.
//
// Oracle: QuantLib's `IborIndex::forecastFixing`, which computes the settlement rate off our curve
// through its own index machinery as (DF(d1)/DF(d2) - 1)/t. That is our generic formula with one
// sub-period and realized = 0, so agreement to curve_rel pins the mapping exactly. Convexity is an
// INPUT number added outside the index (design §3): the model that produced it lives in tests.
//
// NB the dates come from the INDEX (valueDate/maturityDate), not from a coupon: a future settles on
// the actual fixing, so the par-coupon approximation that governs `extract_ibor_obs` does not apply.
TEST_F(Extract, IborFutureMatchesQuantLibFixing) {
  auto index = ext::make_shared<Euribor3M>(hfwd);  // forecast off the +25bp curve
  const double conv = 2.7e-4;                      // an input NUMBER, never a model in include/

  for (int m : {3, 6, 12, 24}) {
    const Date fixing = index->fixingCalendar().adjust(mk.today + Period(m, Months));
    const Date d1 = index->valueDate(fixing);
    const Date d2 = index->maturityDate(d1);
    const auto o = qlx::make_observation({{d1, d2}}, 0.0,
                                         index->dayCounter().yearFraction(d1, d2), mk.today, mk.dc);

    // The IBOR-future shape design §1 specifies: ONE sub-period, nothing realized.
    ASSERT_EQ(o.sub_start.size(), 1u);
    EXPECT_EQ(o.realized, 0.0);
    EXPECT_TRUE(o.weight.empty());  // unit weight => stays on the compiled fused fast path

    const double ours = px::future_rate<double>(o, conv, fwd_curve);
    EXPECT_TRUE(close(ours, index->fixing(fixing) + conv, swaps::tol::curve_rel));

    // Negative control: the agreement above must come from the mapping, not from the two curves
    // being interchangeable. Forecasting off the DISCOUNT curve (25bp away) must not agree.
    EXPECT_GT(std::abs(px::future_rate<double>(o, conv, curve) - ours), 1e-4);
  }
}
