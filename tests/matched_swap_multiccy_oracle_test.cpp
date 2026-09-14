// @oracle-test — the swap_spread MATCHED-MATURITY swap (derive::spread_swap) PER CURRENCY vs QuantLib MakeOIS/MakeVanillaSwap
// DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// WHY A NEW FILE, not bond_rv_oracle_test.cpp: that file is USD-bound by construction -- kVd is a US holiday fixture
// (bond_rv_oracle_test.cpp:73-75), its QuantLib bond side reads US-TREASURY rows (:142-169) and its scope line says so
// (:31). The non-USD matched swaps cannot reach the VERB with a bond of their own currency: conventions.json has no
// non-USD bonds[] row. What is per-currency is the SWAP, and the swap is built by one
// bond-free seam, derive::spread_swap(request, product). So the verb-level USD case stays in bond_rv_oracle_test.cpp
// (swap_maturity / anchor / rows wiring) and this file oracles the seam in every currency. When a non-USD bond row lands,
// its verb-level case goes into bond_rv_oracle_test.cpp with that row's own bond oracle.
//
// PER CASE: QuantLib builds its swap from the DB ROWS on QuantLib's own schedule and QuantLib's own calendar -- settlement
// days = the product's spot_lag, calendar / bdc / leg frequencies / day counts / payment lag from the product row, the
// index (tenor, fixing lag, day count, calendar) from the index row, termination = the bond maturity, DateGeneration::
// Backward, termination adjusted by the bdc, EOM off (the engine rolls none; design O1). Its fairRate() is taken off OUR
// curve through swaps::qlx::CurveTermStructure. Nothing on the QuantLib side reads the engine's dates. QuantLib's calendars
// agree with the DB day by day through 2035 for every calendar used here (calendar_ql_oracle_test.cpp:36-58), so its spot
// arithmetic is an independent computation of the owner's definition, not a restatement of ours.
//
// ASSERTIONS                                                                  TOLERANCE
//   spot == QuantLib's schedule start; termination == its maturityDate()      exact dates
//   every coupon, both legs: pay time, accrual, accrual start/end             tol::literal (integer days / 360 / 365; 30E/360)
//   par rate on our curve vs fairRate()                                        tol::curve_rel (identical DFs through the adapter;
//                                                                              shipped par_swap vs MakeOIS measured 7.6e-17)
// NEGATIVE CONTROLS (per case; an oracle that cannot fail is not one)
//   S1 wrong spot LAG: another currency's spot_lag (USD's T+2 for GBP/AUD; GBP's T+0 for USD/EUR), all else equal
//   S2 wrong spot CALENDAR: the product's lag counted on another currency's calendar (withEffectiveDate), all else equal
//   S3 stub side: DateGeneration::Forward to the same termination (multi-period cases)
//   S4 termination roll: an Unadjusted termination (maturities that are holidays of the product calendar)
// Structural checks are exact; each rate gap is asserted above a stated threshold with the predicted size alongside.
//
// OUT OF SCOPE, stated: zero-coupon products (BRL-CDI-SWAP: no QuantLib DI zero swap, and conventions_ql has no BUS/252);
// day-frequency products (MXN-TIIE-28: refused by the engine); an IBOR swap valued on a non-business day (its first
// fixing falls before the value date -- a past fixing, not a spot question); two-curve swaps (the verb is single-curve); calendars whose QuantLib tabulation ends
// before the maturity (CNY 2024, INR/IDR 2025, SAR 2029, TRY 2034; RUB/ARS none).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"  // instrument_model_quote
#include "swaps/conventions_data.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/derive/asset_swap.hpp"  // spread_swap
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace bld = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace cvd = swaps::conventions;
namespace der = swaps::derive;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

ql::Date qd(const std::string& iso) {
  const bld::Date d = bld::Date::from_iso(iso);
  return ql::Date(ql::Day(d.day()), ql::Month(d.month()), ql::Year(d.year()));
}
std::string ql_iso(const ql::Date& d) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", static_cast<int>(d.year()), static_cast<int>(d.month()),
                static_cast<int>(d.dayOfMonth()));
  return buf;
}
::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure() << "got " << got << " want " << want << " rel " << err);
}

struct HandleCurve {
  const cal::CurveHandle<double>* h = nullptr;
  double discount(double t) const { return h->discount(t); }
};

// Control rows (identity only; lags and calendars are read from them).
const char* const kUsdIndex = "USD-SOFR";
const char* const kGbpIndex = "GBP-SONIA";
bld::SwapConv product_of(const std::string& index) { return bld::Index(index).par_convention().resolve(); }

struct Case {
  const char* index;
  const char* value_date;
  const char* maturity;  // the bond's UNADJUSTED maturity
  bool spot_discriminates;  // the value date makes S1 and S2 give a different spot (premise-checked)
  bool stub_live;           // the schedule has a genuine FRONT stub, so S3 (Forward) must differ
  const char* why;
  bool raw_value_date = false;  // the value date is NOT a business day: QuantLib gets the RAW spot as its effective
                                // date (O5), and S0 shows its settlement-days path is the adjust-first defect
};
const std::vector<Case>& cases() {
  static const std::vector<Case> c = {
      {"USD-SOFR", "2026-09-04", "2031-08-15", true, true, "T+2 over Labor Day (London open); 341-day stub"},
      {"EUR-ESTR", "2027-03-24", "2031-05-01", true, true, "T+2 over Good Friday + Easter Monday; TARGET Labour Day maturity"},
      {"GBP-SONIA", "2026-10-12", "2030-12-26", true, true, "T+0 on Columbus Day; Boxing Day maturity; Christmas substitutes"},
      {"AUD-AONIA", "2026-10-02", "2032-01-26", true, true, "T+1 over NSW Labour Day; Australia Day maturity"},
      {"AUD-BBSW-6M", "2026-10-02", "2029-04-25", true, true, "IBOR, semiannual both legs, 20-day front stub; ANZAC Day"},
      {"EUR-EURIBOR-6M", "2027-03-24", "2031-05-01", true, true, "IBOR, annual 30E/360 fixed vs 6M float: two front stubs"},
      // Forward and Backward coincide here (both give 2026-10-30 -> 2027-10-29 -> 2028-10-31), so S3 is not live.
      {"USD-SOFR", "2026-10-28", "2028-10-31", false, false, "month-end note: the spot-month roll adjusts ONTO spot (FS1)"},
      // O5: non-business value dates -- spot counts from the RAW date (build/schedule.hpp spot_date).
      {"USD-SOFR", "2026-09-07", "2031-08-15", false, true, "valued ON Labor Day: RAW T+2 = 09-09", true},
      {"USD-SOFR", "2026-09-05", "2031-08-15", false, true, "valued on a Saturday: RAW T+2 = 09-09", true},
      {"EUR-ESTR", "2027-03-29", "2031-05-01", false, true, "valued ON Easter Monday: RAW T+2 = 03-31", true},
      {"AUD-AONIA", "2026-10-05", "2032-01-26", false, true, "valued ON NSW Labour Day: RAW T+1 = 10-06", true},
  };
  return c;
}

// QuantLib's matched swap from the DB rows. `settlement_days` / `effective` / `rule` / `termination` are the knobs the
// controls turn; the reference passes the product's spot_lag, no effective date, Backward, the product bdc.
struct QlKnobs {
  ql::Natural settlement_days;
  ql::Date effective;  // null => spot from settlement_days on the swap calendar
  ql::DateGeneration::Rule rule;
  ql::BusinessDayConvention termination;
  ql::Calendar calendar = ql::Calendar();  // empty => the product calendar (S2 builds the swap on another currency's)
};
using QlSwap = ql::ext::shared_ptr<ql::FixedVsFloatingSwap>;

QlSwap ql_matched(const std::string& index_id, const bld::SwapConv& sc, const ql::Handle<ql::YieldTermStructure>& disc,
                  const ql::Date& maturity, const QlKnobs& k) {
  const cvd::IndexConv ix = qconv::index(index_id);
  const ql::Calendar pc = k.calendar.empty() ? qconv::calendar(sc.calendar) : k.calendar;
  const ql::BusinessDayConvention bdc = qconv::bdc(sc.bdc);
  if (ix.type == "overnight") {
    // A DB-built overnight index: QuantLib's named indices carry QuantLib's calendar choice (ql::Sonia fixes on
    // UnitedKingdom::Exchange, sonia.cpp:28-29; the DB's GBP is the Settlement calendar). Fixing days do not enter a
    // telescoped forecast; QuantLib's own Sofr / Estr / Sonia / Aonia all use 0.
    const auto on = ql::ext::make_shared<ql::OvernightIndex>(index_id + "-DB", 0, qconv::currency(ix.currency), pc,
                                                             qconv::day_counter(ix.day_count), disc);
    ql::MakeOIS m(ql::Period(1, ql::Years), on, 0.03);  // the tenor is unused once a termination date is set
    m.withTerminationDate(maturity)
        .withSettlementDays(k.settlement_days)
        .withRule(k.rule)
        .withCalendar(pc)
        .withConvention(bdc)
        .withTerminationDateConvention(k.termination)
        .withEndOfMonth(false)
        .withFixedLegDayCount(qconv::day_counter(sc.fixed_dc))
        .withFixedLegPaymentFrequency(qconv::period(sc.fixed_freq_tok).frequency())
        .withOvernightLegPaymentFrequency(qconv::period(sc.float_freq_tok).frequency())
        .withPaymentLag(sc.pay_lag)
        .withPaymentAdjustment(bdc)
        .withPaymentCalendar(pc)
        .withDiscountingTermStructure(disc);
    if (k.effective != ql::Date()) m.withEffectiveDate(k.effective);
    return ql::ext::shared_ptr<ql::OvernightIndexedSwap>(m);
  }
  // IBOR: a DB-built IborIndex (as portfolio_verb_oracle_test.cpp:231-235), at-par coupons (the engine forecasts each
  // period over its own accrual dates).
  // A family name of this file's own, so the fixings seeded below cannot reach another test's DB-built index.
  const auto ib = ql::ext::make_shared<ql::IborIndex>(index_id + "-MATCHED-ORACLE", qconv::period(ix.tenor),
                                                      static_cast<ql::Natural>(ix.fixing_lag), qconv::currency(ix.currency),
                                                      pc, bdc, /*endOfMonth=*/false, qconv::day_counter(ix.day_count), disc);
  // A CONTROL swap starting on the value date (another currency's T+0 lag) has its first fixing BEFORE the evaluation
  // date, where QuantLib wants a stored fixing. Seed it with QuantLib's own forecast on the same curve -- only for fixing
  // dates whose value date is not before the evaluation date, so no negative curve time is read. The reference swap
  // (the product's own spot) never needs one.
  const ql::Date eval = ql::Settings::instance().evaluationDate();
  for (ql::Date fd = eval - 10; fd < eval; ++fd)
    if (ib->isValidFixingDate(fd) && ib->valueDate(fd) >= eval) ib->addFixing(fd, ib->forecastFixing(fd), /*forceOverwrite=*/true);
  ql::MakeVanillaSwap m(ql::Period(1, ql::Years), ib, 0.03);
  m.withTerminationDate(maturity)
      .withSettlementDays(k.settlement_days)
      .withRule(k.rule)
      .withFixedLegTenor(qconv::period(sc.fixed_freq_tok))
      .withFloatingLegTenor(qconv::period(sc.float_freq_tok))
      .withFixedLegCalendar(pc)
      .withFloatingLegCalendar(pc)
      .withFixedLegConvention(bdc)
      .withFloatingLegConvention(bdc)
      .withFixedLegTerminationDateConvention(k.termination)
      .withFloatingLegTerminationDateConvention(k.termination)
      .withFixedLegEndOfMonth(false)
      .withFloatingLegEndOfMonth(false)
      .withFixedLegDayCount(qconv::day_counter(sc.fixed_dc))
      .withFloatingLegDayCount(qconv::day_counter(sc.float_dc))
      .withPaymentConvention(bdc)
      .withDiscountingTermStructure(disc)
      .withAtParCoupons(true);
  if (k.effective != ql::Date()) m.withEffectiveDate(k.effective);
  return ql::ext::shared_ptr<ql::VanillaSwap>(m);
}

}  // namespace

TEST(MatchedSwapMultiCurrencyOracle, EachCurrencysMatchedSwapIsQuantLibsSwapFromItsSpotToTheBondMaturity) {
  ql::SavedSettings saved;
  ql::IborCoupon::Settings::instance().createAtParCoupons();  // stated, not assumed

  const bld::SwapConv usd = product_of(kUsdIndex), gbp = product_of(kGbpIndex);
  double worst_par = 0.0;

  for (const Case& c : cases()) {
    SCOPED_TRACE(std::string(c.index) + " " + c.value_date + " -> " + c.maturity + ": " + c.why);
    const bld::SwapConv sc = product_of(c.index);
    const cvd::IndexConv ix = qconv::index(c.index);
    const ql::Date vd = qd(c.value_date), mat = qd(c.maturity);
    ql::Settings::instance().evaluationDate() = vd;

    // ---- premises (DB rows; QuantLib's calendars) -------------------------------------------------------------------
    ASSERT_FALSE(sc.zero_coupon) << "premise: zero-coupon products are out of scope";
    ASSERT_EQ(std::string(ix.calendar), sc.calendar) << "premise: the index fixes on the product calendar";
    ASSERT_EQ(std::string(ix.day_count), sc.float_dc) << "premise: the float leg accrues on the index day count";
    if (ix.type != "overnight") ASSERT_EQ(sc.pay_lag, 0) << "premise: VanillaSwap has no payment lag";
    const ql::Calendar pc = qconv::calendar(sc.calendar);
    ASSERT_EQ(pc.isBusinessDay(vd), !c.raw_value_date) << "premise: the case's value-date kind";
    // RAW (O5): Calendar::advance(n, Days) from a non-business date counts the next business day as day 1.
    const ql::Date spot = pc.advance(vd, sc.spot_lag, ql::Days);

    const int other_lag = usd.spot_lag != sc.spot_lag ? usd.spot_lag : gbp.spot_lag;
    const std::string other_cal_id = usd.calendar != sc.calendar ? usd.calendar : gbp.calendar;
    const ql::Calendar oc = qconv::calendar(other_cal_id);
    ASSERT_NE(other_lag, sc.spot_lag);
    const ql::Date spot_wrong_lag = pc.advance(vd, other_lag, ql::Days);
    const ql::Date spot_wrong_cal = oc.advance(oc.adjust(vd), sc.spot_lag, ql::Days);
    if (c.spot_discriminates) {
      ASSERT_NE(spot_wrong_lag, spot) << "premise: another spot lag gives another spot";
      ASSERT_NE(spot_wrong_cal, spot) << "premise: another calendar (" << other_cal_id << ") gives another spot";
    }

    // ---- our curve, to QuantLib through the adapter (the headline test's shape: upward forwards) ---------------------
    cal::BundleProblem p;
    p.curves.push_back({.base = -1, .regions = cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0})});
    Eigen::VectorXd x(p.n_knots());
    for (int i = 0; i < x.size(); ++i) x[i] = 0.0360 + 0.0008 * i;
    const auto C = cal::build_bundle_curves<double>(p.curves, [&](int ci, int i) { return x[p.offset(ci) + i]; });
    HandleCurve hc{C[0].get()};
    const ql::Handle<ql::YieldTermStructure> disc(
        ql::ext::make_shared<swaps::qlx::CurveTermStructure<HandleCurve>>(vd, ql::Actual365Fixed(), &hc));
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[static_cast<std::size_t>(i)]; };
    const auto t_of = [&](const ql::Date& d) { return ql::Actual365Fixed().yearFraction(vd, d); };

    // ---- the engine's matched swap (the seam swap_spread calls) --------------------------------------------------
    der::SwapSpreadRequest r;
    r.value_date = bld::Date::from_iso(c.value_date);
    r.type = der::SwapSpreadType::MatchedMaturity;
    r.index = c.index;
    r.bond.maturity = bld::Date::from_iso(c.maturity);
    r.swap_curve = 0;
    r.factor_curve = 1;
    const der::SpreadSwap m = der::spread_swap(r, sc);
    const cal::Instrument& swap = m.swap;
    ASSERT_EQ(swap.quote, cal::QuoteKind::ParRate);

    // ---- QuantLib's -----------------------------------------------------------------------------------------------
    // On a non-business value date the vendored QuantLib's settlement-days path adjusts the date first (fixed upstream in
    // PR #2653), so the reference is handed the RAW spot explicitly.
    const QlKnobs ref{static_cast<ql::Natural>(sc.spot_lag), c.raw_value_date ? spot : ql::Date(),
                      ql::DateGeneration::Backward, qconv::bdc(sc.bdc)};
    const QlSwap q = ql_matched(c.index, sc, disc, mat, ref);
    ASSERT_EQ(q->fixedSchedule().startDate(), spot) << "premise: QuantLib's start is the product spot";

    // (1) where it starts and ends
    ASSERT_FALSE(swap.fwd.coupons.empty());
    EXPECT_NEAR(swap.fwd.coupons.front().accrual_start, t_of(spot), tol::literal) << "spot " << ql_iso(spot);
    EXPECT_EQ(bld::iso(m.maturity), ql_iso(q->maturityDate())) << "termination";

    // (2) every coupon, both legs
    const ql::Leg& qfix = q->fixedLeg();
    ASSERT_EQ(swap.fixed.coupons.size(), qfix.size()) << "fixed coupon count";
    for (std::size_t k = 0; k < qfix.size(); ++k) {
      const auto cp = ql::ext::dynamic_pointer_cast<ql::Coupon>(qfix[k]);
      ASSERT_TRUE(cp) << "fixed coupon " << k;
      EXPECT_NEAR(swap.fixed.coupons[k].pay, t_of(cp->date()), tol::literal) << "fixed pay " << k;
      EXPECT_NEAR(swap.fixed.coupons[k].tau, cp->accrualPeriod(), tol::literal) << "fixed accrual " << k;
    }
    const ql::Leg& qflt = q->floatingLeg();
    ASSERT_EQ(swap.fwd.coupons.size(), qflt.size()) << "float coupon count";
    for (std::size_t k = 0; k < qflt.size(); ++k) {
      const auto cp = ql::ext::dynamic_pointer_cast<ql::Coupon>(qflt[k]);
      ASSERT_TRUE(cp) << "float coupon " << k;
      EXPECT_NEAR(swap.fwd.coupons[k].pay, t_of(cp->date()), tol::literal) << "float pay " << k;
      EXPECT_NEAR(swap.fwd.coupons[k].accrual_start, t_of(cp->accrualStartDate()), tol::literal) << "float start " << k;
      EXPECT_NEAR(swap.fwd.coupons[k].accrual_end, t_of(cp->accrualEndDate()), tol::literal) << "float end " << k;
      EXPECT_NEAR(swap.fwd.coupons[k].tau_pay, cp->accrualPeriod(), tol::literal) << "float accrual " << k;
    }

    // (3) its par rate on our curve
    const double par_eng = cal::instrument_model_quote<double>(swap, curve_of);
    const double fair_ql = q->fairRate();
    EXPECT_TRUE(close(par_eng, fair_ql, tol::curve_rel)) << "matched swap par rate";
    worst_par = std::max(worst_par, std::abs(par_eng - fair_ql));

    // ---- negative controls ------------------------------------------------------------------------------------------
    // Predicted rate gaps on this curve (forwards 3.60% at the front rising 8 bp per knot, par ~3.9%): a start moved by
    // δ business days shifts the float leg by DF·f_spot·δ/365 and the annuity by DF·δ/basis, so
    // Δpar ≈ (f_spot − par·365/basis)·(δ/365)/A ≈ 0.3% · δ/365 / 4 ≈ 2e-6 per day. Asserted > 1e-8 (100× curve_rel,
    // ~200× margin). The START-DATE check is exact and is the primary teeth.
    double gap_lag = 0.0, gap_cal = 0.0, gap_stub = 0.0, gap_roll = 0.0, gap_adj = 0.0;
    // S0 (O5) -- the adjust-first rule: QuantLib's own settlement-days path on a non-business value date starts one
    //    business day LATER than the RAW spot, and must not match the engine.
    if (c.raw_value_date) {
      const QlSwap q_adj = ql_matched(c.index, sc, disc, mat, {ref.settlement_days, ql::Date(), ref.rule, ref.termination});
      ASSERT_EQ(q_adj->fixedSchedule().startDate(), pc.advance(pc.adjust(vd, ql::Following), sc.spot_lag, ql::Days))
          << "premise: the vendored QuantLib adjusts the value date first";
      ASSERT_NE(q_adj->fixedSchedule().startDate(), spot) << "premise: adjust-first differs from RAW here";
      EXPECT_GT(std::abs(swap.fwd.coupons.front().accrual_start - t_of(q_adj->fixedSchedule().startDate())), 0.5 / 365.0)
          << "the engine must not start on the adjust-first spot";
      gap_adj = std::abs(par_eng - q_adj->fairRate());
      EXPECT_GT(gap_adj, 1e-8) << "an adjust-first swap must not match";
    }
    if (c.spot_discriminates) {
      // S1 -- another currency's spot LAG. For GBP / AUD this is "USD's T+2 applied to a GBP / AUD swap".
      const QlSwap q_lag =
          ql_matched(c.index, sc, disc, mat, {static_cast<ql::Natural>(other_lag), ql::Date(), ref.rule, ref.termination});
      ASSERT_EQ(q_lag->fixedSchedule().startDate(), spot_wrong_lag);
      EXPECT_GT(std::abs(swap.fwd.coupons.front().accrual_start - t_of(spot_wrong_lag)), 0.5 / 365.0)
          << "the start must not be the T+" << other_lag << " spot " << ql_iso(spot_wrong_lag);
      gap_lag = std::abs(par_eng - q_lag->fairRate());
      EXPECT_GT(gap_lag, 1e-8) << "a T+" << other_lag << " swap must not match a T+" << sc.spot_lag << " product";

      // S2 -- the product's lag counted on ANOTHER currency's calendar.
      //   (a) structural, exact: the engine's start is not that raw date. (QuantLib cannot be handed it as an effective
      //       date and still differ: Schedule business-day-adjusts the first date on the product calendar, and for EUR
      //       (Easter Monday 2027-03-29) and AUD (NSW Labour Day 2026-10-05) the wrong-calendar spot IS a product holiday
      //       that rolls straight back onto the true spot -- the control would collapse.)
      EXPECT_GT(std::abs(swap.fwd.coupons.front().accrual_start - t_of(spot_wrong_cal)), 0.5 / 365.0)
          << "the start must not be the spot counted on " << other_cal_id << " (" << ql_iso(spot_wrong_cal) << ")";
      //   (b) rate: QuantLib's swap built wholly on the other currency's calendar (spot, rolls, termination, payments).
      const QlSwap q_cal =
          ql_matched(c.index, sc, disc, mat, {ref.settlement_days, ql::Date(), ref.rule, ref.termination, oc});
      ASSERT_NE(q_cal->fixedSchedule().startDate(), spot) << "premise: the other calendar moves the start";
      gap_cal = std::abs(par_eng - q_cal->fairRate());
      EXPECT_GT(gap_cal, 1e-8) << "a swap on " << other_cal_id << " must not match";
    }

    // S3 -- stub side: Forward from spot to the same termination (a back stub). First accrual differs by months; the rate
    //       by ~1e-6..1e-5 (annuity timing). Asserted > 1e-7 (1000× curve_rel). If a first run measures a rate gap
    //       under 1e-7, DROP the rate line and keep the structural line; do not loosen it.
    if (c.stub_live) {
      ASSERT_GT(qfix.size(), 1u) << "premise: a multi-period swap";
      const QlSwap q_fwd =
          ql_matched(c.index, sc, disc, mat, {ref.settlement_days, ref.effective, ql::DateGeneration::Forward, ref.termination});
      const auto c0 = ql::ext::dynamic_pointer_cast<ql::Coupon>(q_fwd->fixedLeg().front());
      ASSERT_TRUE(c0);
      EXPECT_GT(std::abs(swap.fixed.coupons.front().tau - c0->accrualPeriod()), 10.0 / 365.0) << "the stub is FIRST";
      gap_stub = std::abs(par_eng - q_fwd->fairRate());
      EXPECT_GT(gap_stub, 1e-7) << "a forward-rolled swap must not match";
    }

    // S4 -- termination roll (a maturity that is closed on the product calendar): Unadjusted termination. Last accrual
    //       differs by >= 1 day exactly; rate ~1e-6 (Δpar ≈ DF·(par − f_end)·Δt/A), asserted > 1e-8. Same drop rule.
    if (!pc.isBusinessDay(mat)) {
      const QlSwap q_raw = ql_matched(c.index, sc, disc, mat, {ref.settlement_days, ref.effective, ref.rule, ql::Unadjusted});
      const auto cl = ql::ext::dynamic_pointer_cast<ql::Coupon>(q_raw->fixedLeg().back());
      ASSERT_TRUE(cl);
      EXPECT_GT(std::abs(swap.fixed.coupons.back().tau - cl->accrualPeriod()), 0.5 / 365.0) << "the termination rolls";
      gap_roll = std::abs(par_eng - q_raw->fairRate());
      EXPECT_GT(gap_roll, 1e-8) << "an unadjusted termination must not match";
    }

    // The seeded control fixings end with the case -- only this file's own index family, never another test's history.
    for (const std::string& name : ql::IndexManager::instance().histories())
      if (name.find("-MATCHED-ORACLE") != std::string::npos) ql::IndexManager::instance().clearHistory(name);
    std::cout << "  [matched " << c.index << " " << c.value_date << "->" << c.maturity << "] |eng-QL| par "
              << std::abs(par_eng - fair_ql) << "  S1 lag " << gap_lag << "  S2 calendar " << gap_cal << "  S3 stub "
              << gap_stub << "  S4 roll " << gap_roll << "  S0 adjust-first " << gap_adj << "\n";
  }
  std::cout << "  [matched] worst |eng-QL| par " << worst_par << "\n";
}

// FS1 in QuantLib's own words: the month-end case's spot-month roll date adjusts onto spot, and QuantLib's Schedule drops
// it (schedule.cpp "final safety checks"), leaving TWO periods. A premise for the engine-side guard, so the drop is
// QuantLib's behaviour and not an assumption.
TEST(MatchedSwapMultiCurrencyOracle, QuantLibDropsARollDateThatAdjustsOntoSpot) {
  ql::SavedSettings saved;
  // The month-end FS1 case, found by its value date (the RAW value-date cases follow it in the table).
  const auto fs1 = std::find_if(cases().begin(), cases().end(),
                                [](const Case& k) { return std::string(k.value_date) == "2026-10-28"; });
  ASSERT_NE(fs1, cases().end());
  const Case& c = *fs1;
  ASSERT_FALSE(c.spot_discriminates);
  const bld::SwapConv sc = product_of(c.index);
  const ql::Date vd = qd(c.value_date);
  ql::Settings::instance().evaluationDate() = vd;
  const ql::Calendar pc = qconv::calendar(sc.calendar);
  const ql::Date spot = pc.advance(vd, sc.spot_lag, ql::Days);
  const ql::Date roll = ql::Date(31, ql::October, 2026);
  ASSERT_GT(roll, spot);
  ASSERT_EQ(pc.adjust(roll, qconv::bdc(sc.bdc)), spot) << "premise: the roll date adjusts onto spot";
  const ql::Schedule s(spot, qd(c.maturity), qconv::period(sc.fixed_freq_tok), pc, qconv::bdc(sc.bdc), qconv::bdc(sc.bdc),
                       ql::DateGeneration::Backward, /*endOfMonth=*/false);
  ASSERT_EQ(s.size(), 3u) << "QuantLib keeps spot, 2027-10-29, 2028-10-31";
  EXPECT_EQ(s.dates()[1], ql::Date(29, ql::October, 2027));
}
