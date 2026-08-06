// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
//
// Pins the bond kernels (swaps/pricing/bond.hpp, swaps/build/bond.hpp, swaps/portfolio/bond_universe.hpp)
// penny-perfect against QuantLib::BondFunctions / FixedRateBond / DiscountingBondEngine:
//   (a) YIELD space  — accrued, clean/dirty from yield, yield from clean, modified duration, convexity;
//   (b) CURVE space  — dirty/clean off a discount curve vs a QuantLib DiscountingBondEngine on the SAME
//                      discount factors, and z-spread vs a QuantLib ZeroSpreadedTermStructure;
//   (c) SWEEP        — the batched BondUniverse yield solve matches BondFunctions::yield bond-for-bond.
//
// The bond is a semiannual, ActualActual(ISMA), Unadjusted, Backward-generated schedule — exactly what
// swaps::build::fixed_rate_bond assembles, so the two constructions are the same instrument.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/portfolio/bond_universe.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace px = swaps::pricing;
namespace bld = swaps::build;
namespace pf = swaps::portfolio;

namespace {

QuantLib::Date qd(const bld::Date& d) {
  return QuantLib::Date(int(d.day()), Month(int(d.month())), d.year());
}

// A QuantLib term structure exposed as our curve interface (Scalar discount(double t)) so the SAME
// discount factors feed both pricers -- any disagreement is then a pricing bug, not a curve mismatch.
struct QlDisc {
  Handle<YieldTermStructure> h;
  double discount(double t) const { return h->discount(t); }
};

// Build the matched pair: our BuiltBond and QuantLib's FixedRateBond, sharing eval/settle/schedule.
struct Pair {
  bld::BuiltBond ours;
  ext::shared_ptr<FixedRateBond> ql;
  DayCounter dc;
  QuantLib::Date settle;
};

Pair make_pair(const bld::Date& value, const bld::Date& issue, const bld::Date& maturity, double coupon) {
  Settings::instance().evaluationDate() = qd(value);
  Schedule sched(qd(issue), qd(maturity), Period(Semiannual), NullCalendar(), Unadjusted, Unadjusted,
                 DateGeneration::Backward, false);
  DayCounter dc = ActualActual(ActualActual::ISMA, sched);
  auto bond = ext::make_shared<FixedRateBond>(/*settlementDays=*/1, /*faceAmount=*/100.0, sched,
                                              std::vector<Rate>{coupon}, dc, Following, 100.0, qd(issue));
  const QuantLib::Date settle = bond->settlementDate();
  // Build OUR bond with settlement == QuantLib's settlement date, so the comparison is exact.
  bld::FixedBondTerms t;
  t.value_date = value;
  t.settle = bld::Date::ymd(settle.year(), settle.month(), settle.dayOfMonth());
  t.issue = issue;
  t.maturity = maturity;
  t.coupon = coupon;
  t.freq = 2;
  return {bld::fixed_rate_bond(t), bond, dc, settle};
}

::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure() << "got " << got << " want " << want << " rel " << err);
}

TEST(BondOracle, YieldSpaceMatchesBondFunctions) {
  const Pair p = make_pair(bld::Date::ymd(2024, 1, 15), bld::Date::ymd(2019, 8, 15),
                           bld::Date::ymd(2034, 2, 15), 0.05);
  // Accrued (QuantLib returns per 100 face; ours is per unit notional).
  EXPECT_TRUE(close(p.ours.accrued * 100.0, p.ql->accruedAmount(p.settle), swaps::tol::curve_rel));

  for (double y : {0.02, 0.035, 0.05, 0.07}) {
    const double clean_ql = BondFunctions::cleanPrice(*p.ql, y, p.dc, Compounded, Semiannual, p.settle);
    const double dirty_ql = BondFunctions::dirtyPrice(*p.ql, y, p.dc, Compounded, Semiannual, p.settle);
    EXPECT_TRUE(close(px::bond_clean_from_yield(p.ours.yield, y) * 100.0, clean_ql, swaps::tol::curve_rel))
        << "clean y=" << y;
    EXPECT_TRUE(close(px::bond_dirty_from_yield(p.ours.yield, y) * 100.0, dirty_ql, swaps::tol::curve_rel))
        << "dirty y=" << y;
    // Yield from clean (both directions).
    const double y_ours = px::bond_yield_from_clean(p.ours.yield, clean_ql / 100.0);
    const double y_ql = BondFunctions::yield(*p.ql, clean_ql, p.dc, Compounded, Semiannual, p.settle);
    EXPECT_TRUE(close(y_ours, y_ql, swaps::tol::curve_rel)) << "yield y=" << y;
    // Modified duration and convexity.
    const px::BondRisk r = px::bond_risk(p.ours.yield, y);
    const double md_ql =
        BondFunctions::duration(*p.ql, y, p.dc, Compounded, Semiannual, Duration::Modified, p.settle);
    const double cx_ql = BondFunctions::convexity(*p.ql, y, p.dc, Compounded, Semiannual, p.settle);
    EXPECT_TRUE(close(r.modified_duration, md_ql, swaps::tol::jacobian_rel)) << "mod-dur y=" << y;
    EXPECT_TRUE(close(r.convexity, cx_ql, swaps::tol::jacobian_rel)) << "convexity y=" << y;
  }
}

TEST(BondOracle, CurveSpaceMatchesDiscountingEngine) {
  const Pair p = make_pair(bld::Date::ymd(2024, 1, 15), bld::Date::ymd(2018, 5, 15),
                           bld::Date::ymd(2031, 5, 15), 0.0375);
  const QuantLib::Date ref = Settings::instance().evaluationDate();
  Handle<YieldTermStructure> disc(
      ext::make_shared<FlatForward>(ref, 0.042, Actual365Fixed(), Continuous, Annual));
  p.ql->setPricingEngine(ext::make_shared<DiscountingBondEngine>(disc));
  QlDisc qc{disc};
  // Dirty / clean off the SAME discount factors (per 100 face).
  EXPECT_TRUE(
      close(px::bond_dirty_price<double>(p.ours.curve, qc) * 100.0, p.ql->dirtyPrice(), swaps::tol::curve_rel));
  EXPECT_TRUE(
      close(px::bond_clean_price<double>(p.ours.curve, qc) * 100.0, p.ql->cleanPrice(), swaps::tol::curve_rel));

  // z-spread vs a QuantLib ZeroSpreadedTermStructure: price the bond off base+z (continuous, ACT/365F),
  // then our z-spread against the base curve must recover z.
  const double z = 0.0075;
  Handle<YieldTermStructure> spreaded(ext::make_shared<ZeroSpreadedTermStructure>(
      disc, Handle<Quote>(ext::make_shared<SimpleQuote>(z)), Continuous, Annual, Actual365Fixed()));
  QlDisc qs{spreaded};
  const double target_dirty = px::bond_dirty_price<double>(p.ours.curve, qs);
  EXPECT_TRUE(close(px::bond_z_spread(p.ours.curve, qc, target_dirty), z, swaps::tol::jacobian_rel));
}

TEST(BondOracle, UniverseSweepMatchesBondFunctions) {
  // A universe of distinct treasuries; solve every yield in one batched Newton and check bond-for-bond.
  std::vector<px::YieldBond> univ;
  std::vector<ext::shared_ptr<FixedRateBond>> qls;
  std::vector<DayCounter> dcs;
  std::vector<QuantLib::Date> settles;
  std::vector<double> targets_clean;
  for (int i = 0; i < 40; ++i) {
    const Pair p = make_pair(bld::Date::ymd(2024, 1, 15), bld::Date::ymd(2016, 5, 15),
                             bld::Date::ymd(2027 + i % 15, 5, 15), 0.01 + 0.001 * (i % 40));
    univ.push_back(p.ours.yield);
    qls.push_back(p.ql);
    dcs.push_back(p.dc);
    settles.push_back(p.settle);
    targets_clean.push_back(95.0 + 0.2 * i);  // a spread of quoted clean prices (per 100)
  }
  pf::BondUniverse bu;
  bu.set(univ);
  Eigen::VectorXd cl(univ.size());
  for (std::size_t i = 0; i < univ.size(); ++i) cl[i] = targets_clean[i] / 100.0;  // per unit notional
  const Eigen::VectorXd y_ours = bu.yields_from_clean(cl);
  for (std::size_t i = 0; i < univ.size(); ++i) {
    const double y_ql = BondFunctions::yield(*qls[i], targets_clean[i], dcs[i], Compounded, Semiannual,
                                             settles[i]);
    EXPECT_TRUE(close(y_ours[i], y_ql, swaps::tol::curve_rel)) << "bond " << i;
  }
}

}  // namespace
