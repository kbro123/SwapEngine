// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// Pins the par-par asset-swap spread (swaps/build/par_asset_swap.hpp) against QuantLib::AssetSwap::fairSpread on
// the SAME discount curve and bond, across a range of market prices. The float leg is a 0-fixing-day index
// with the leg starting on the evaluation date, so the first fixing is FORECAST (no historical fixings needed)
// and QuantLib's asset swap is deterministic off the flat curve.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <vector>

#include "swaps/build/par_asset_swap.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/pricing/bond.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace px = swaps::pricing;
namespace bld = swaps::build;

namespace {

QuantLib::Date qd(const bld::Date& d) { return QuantLib::Date(int(d.day()), Month(int(d.month())), d.year()); }

struct QlDisc {
  Handle<YieldTermStructure> h;
  double discount(double t) const { return h->discount(t); }
};

// One (curve, bond) config: build QuantLib's par-par AssetSwap fairSpread and OUR spread at a par purchase
// (dirty = 1 + accrued), off the SAME flat curve + matched float schedule. QuantLib's fairSpread is the
// running par-par spread (price monetised in the upfront, so it's independent of the market price) — our
// formula at a par purchase reproduces it exactly, which pins the curve bond PV + the float annuity.
void check(double curve_rate, const bld::Date& value, const bld::Date& issue, const bld::Date& maturity,
           double coupon) {
  Settings::instance().evaluationDate() = qd(value);
  Handle<YieldTermStructure> disc(
      ext::make_shared<FlatForward>(qd(value), curve_rate, Actual365Fixed(), Continuous, Annual));
  QlDisc qc{disc};

  Schedule bsched(qd(issue), qd(maturity), Period(Semiannual), NullCalendar(), Unadjusted, Unadjusted,
                  DateGeneration::Backward, false);
  DayCounter bdc = ActualActual(ActualActual::ISMA, bsched);
  auto qbond = ext::make_shared<FixedRateBond>(/*settlementDays=*/0, /*face=*/100.0, bsched,
                                               std::vector<Rate>{coupon}, bdc, Following, 100.0, qd(issue));
  qbond->setPricingEngine(ext::make_shared<DiscountingBondEngine>(disc));
  const QuantLib::Date settle = qbond->settlementDate();

  // Float leg: quarterly, ACT/360, 0 fixing days, NullCalendar/Unadjusted (first fixing == settle == today,
  // so forecast not historical); deterministic off the flat curve.
  auto idx = ext::make_shared<IborIndex>("TestIbor", Period(3, Months), /*fixingDays=*/0, USDCurrency(),
                                         NullCalendar(), Unadjusted, false, Actual360(), disc);
  Schedule fsched(settle, qd(maturity), Period(3, Months), NullCalendar(), Unadjusted, Unadjusted,
                  DateGeneration::Backward, false);
  AssetSwap asw(/*parAssetSwap=*/true, qbond, /*cleanPrice=*/100.0, idx, /*spread=*/0.0, fsched, Actual360());
  asw.setPricingEngine(ext::make_shared<DiscountingSwapEngine>(disc));
  const double fair_ql = asw.fairSpread();

  bld::FixedBondTerms t;
  t.value_date = value; t.settle = value; t.issue = issue; t.maturity = maturity; t.coupon = coupon; t.freq = 2;
  const bld::BuiltBond bond = bld::fixed_rate_bond(t);
  std::vector<double> fpay, ftau;
  for (std::size_t i = 1; i < fsched.size(); ++i) {
    const QuantLib::Date a = fsched[i - 1], b = fsched[i];
    ftau.push_back(Actual360().yearFraction(a, b));
    fpay.push_back(bld::curve_time(value, bld::Date::ymd(b.year(), b.month(), b.dayOfMonth())));
  }
  const double ours = bld::par_asset_swap_spread(bond.curve, qc, 1.0 + bond.accrued, fpay, ftau);
  EXPECT_NEAR(ours, fair_ql, 5e-5) << "rate=" << curve_rate << " coupon=" << coupon
                                   << " ql=" << fair_ql << " ours=" << ours;

  // The market/proceeds spread widens as the bond cheapens (buy below par -> wider running spread).
  const double cheap = bld::par_asset_swap_spread(bond.curve, qc, 0.97 + bond.accrued, fpay, ftau);
  EXPECT_GT(cheap, ours);
}

TEST(BondAssetSwapOracle, ParParSpreadMatchesQuantLib) {
  const bld::Date value = bld::Date::ymd(2026, 8, 17), issue = bld::Date::ymd(2021, 2, 15);
  check(0.035, value, issue, bld::Date::ymd(2033, 2, 15), 0.04);   // premium coupon vs curve
  check(0.045, value, issue, bld::Date::ymd(2033, 2, 15), 0.04);   // discount coupon vs curve
  check(0.030, value, issue, bld::Date::ymd(2036, 2, 15), 0.05);   // longer, high coupon
  check(0.050, value, issue, bld::Date::ymd(2031, 8, 15), 0.03);   // shorter, low coupon
}

}  // namespace
