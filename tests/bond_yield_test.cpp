// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T3 cross-path parity (two engine paths, same inputs)
// QuantLib-FREE self-consistency gate for the bond kernels (pricing/bond.hpp, portfolio/bond_universe.hpp,
// build/bond.hpp). These properties hold by construction and need no oracle: price<->yield round-trips,
// analytic duration/convexity match finite differences, the batched universe sweep equals the per-bond
// scalar path, and the curve-space kernel's PV/z-spread are internally consistent. The penny-perfect
// comparison to QuantLib::BondFunctions lives in the QL-linked bond_oracle_test.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/portfolio/bond_universe.hpp"

namespace px = swaps::pricing;
namespace bld = swaps::build;
namespace pf = swaps::portfolio;

namespace {

// Flat continuously-compounded discount curve (curve time) for the curve-space checks.
struct FlatCurve {
  double r;
  double discount(double t) const { return std::exp(-r * t); }
};

bld::BuiltBond make_bond(int mat_year, double coupon) {
  bld::FixedBondTerms t;
  t.value_date = bld::Date::ymd(2024, 1, 15);
  t.settle = bld::Date::ymd(2024, 1, 16);
  t.issue = bld::Date::ymd(2019, 8, 15);
  t.maturity = bld::Date::ymd(mat_year, 2, 15);
  t.coupon = coupon;
  t.freq = 2;
  return bld::fixed_rate_bond(t);
}

TEST(BondYield, PriceYieldRoundTrip) {
  const bld::BuiltBond b = make_bond(2034, 0.05);
  for (double y : {0.01, 0.03, 0.043, 0.06, 0.09}) {
    const double clean = px::bond_clean_from_yield(b.yield, y);
    const double y_back = px::bond_yield_from_clean(b.yield, clean);
    EXPECT_NEAR(y_back, y, 1e-12) << "y=" << y;
  }
}

TEST(BondYield, DirtyEqualsCleanPlusAccrued) {
  const bld::BuiltBond b = make_bond(2030, 0.04);
  const double y = 0.038;
  EXPECT_NEAR(px::bond_dirty_from_yield(b.yield, y), px::bond_clean_from_yield(b.yield, y) + b.accrued,
              1e-14);
  EXPECT_GT(b.accrued, 0.0);  // settlement is mid-period
}

TEST(BondYield, DurationConvexityVsFiniteDifference) {
  const bld::BuiltBond b = make_bond(2044, 0.045);
  const double y = 0.041, h = 1e-6;
  const double P = px::bond_dirty_from_yield(b.yield, y);
  const double Pp = px::bond_dirty_from_yield(b.yield, y + h);
  const double Pm = px::bond_dirty_from_yield(b.yield, y - h);
  const px::BondRisk r = px::bond_risk(b.yield, y);
  EXPECT_NEAR(r.modified_duration, -(Pp - Pm) / (2 * h) / P, 1e-6);
  EXPECT_NEAR(r.convexity, (Pp - 2 * P + Pm) / (h * h) / P, 1e-3 * r.convexity);  // 2nd-order FD is roundoff-limited (~eps/h^2); relative sanity bound (QL oracle pins the exact value)
  // Macaulay = Modified * (1 + y/f).
  EXPECT_NEAR(r.macaulay_duration, r.modified_duration * (1.0 + y / b.yield.conv.freq), 1e-14);
}

TEST(BondYield, UniverseBatchedEqualsScalar) {
  std::vector<px::YieldBond> univ;
  std::vector<double> ys;
  for (int i = 0; i < 200; ++i) {
    const bld::BuiltBond bi = make_bond(2026 + i % 25, 0.015 + 0.0005 * (i % 40));
    univ.push_back(bi.yield);
    ys.push_back(0.02 + 0.0003 * i);
  }
  pf::BondUniverse bu;
  bu.set(univ);
  EXPECT_TRUE(bu.is_regular());  // builder bonds are regular => the Horner (FMA) fast path is exercised
  Eigen::VectorXd yv(univ.size()), cl(univ.size());
  for (std::size_t i = 0; i < univ.size(); ++i) {
    yv[i] = ys[i];
    cl[i] = px::bond_clean_from_yield(univ[i], ys[i]);
  }
  // Batched yield-from-clean recovers the per-bond input.
  const Eigen::VectorXd y_solved = bu.yields_from_clean(cl);
  for (std::size_t i = 0; i < univ.size(); ++i) EXPECT_NEAR(y_solved[i], ys[i], 1e-11);
  // Batched dirty / duration / convexity equal the scalar kernel bond-for-bond.
  const Eigen::VectorXd dv = bu.dirty_prices(yv);
  const Eigen::VectorXd md = bu.modified_durations(yv);
  const Eigen::VectorXd cx = bu.convexities(yv);
  for (std::size_t i = 0; i < univ.size(); ++i) {
    EXPECT_NEAR(dv[i], px::bond_dirty_from_yield(univ[i], ys[i]), 1e-13);
    const px::BondRisk r = px::bond_risk(univ[i], ys[i]);
    EXPECT_NEAR(md[i], r.modified_duration, 1e-13);
    EXPECT_NEAR(cx[i], r.convexity, 1e-12 * std::abs(r.convexity));  // batched Horner vs scalar exp reassociate differently across platforms (FMA); relative bound
  }
}

TEST(BondYield, ReverseYieldToCleanAndAccrued) {
  // The REVERSE direction (yield -> clean price) and accrued must both be fast batched ops.
  std::vector<px::YieldBond> univ;
  std::vector<double> built_accrued, ys;
  for (int i = 0; i < 120; ++i) {
    const bld::BuiltBond bi = make_bond(2028 + i % 20, 0.02 + 0.0004 * i);
    univ.push_back(bi.yield);
    built_accrued.push_back(bi.accrued);
    ys.push_back(0.025 + 0.0004 * i);
  }
  pf::BondUniverse bu;
  bu.set(univ);
  Eigen::VectorXd yv(univ.size());
  for (std::size_t i = 0; i < univ.size(); ++i) yv[i] = ys[i];

  // accrued() is the structure-cached vector: equals the builder value bond-for-bond, O(1), no recompute.
  const Eigen::VectorXd accr = bu.accrued();
  for (std::size_t i = 0; i < univ.size(); ++i) EXPECT_NEAR(accr[i], built_accrued[i], 1e-15);

  // yield -> clean (batched) equals the scalar kernel, and clean == dirty - accrued elementwise.
  const Eigen::VectorXd clean = bu.clean_prices(yv);
  const Eigen::VectorXd dirty = bu.dirty_prices(yv);
  for (std::size_t i = 0; i < univ.size(); ++i) {
    EXPECT_NEAR(clean[i], px::bond_clean_from_yield(univ[i], ys[i]), 1e-13);
    EXPECT_NEAR(clean[i], dirty[i] - accr[i], 1e-15);
  }
  // Full round trip: yields -> clean -> yields recovers the input.
  const Eigen::VectorXd y_back = bu.yields_from_clean(clean);
  for (std::size_t i = 0; i < univ.size(); ++i) EXPECT_NEAR(y_back[i], ys[i], 1e-11);
}

TEST(BondAccrued, StandaloneRecomputesForRolledSettlement) {
  // Accrued is a pure schedule quantity, so it recomputes O(1) for any settlement in the current period
  // via accrued_interest() — no rebuild. Rolling settlement forward increases accrued linearly.
  bld::FixedBondTerms t;
  t.value_date = bld::Date::ymd(2024, 1, 15);
  t.settle = bld::Date::ymd(2024, 1, 16);
  t.issue = bld::Date::ymd(2019, 8, 15);
  t.maturity = bld::Date::ymd(2034, 2, 15);
  t.coupon = 0.05;
  t.freq = 2;
  const bld::BuiltBond b = bld::fixed_rate_bond(t);
  // Recompute accrued at the build settlement from the cached period == the built value.
  EXPECT_NEAR(bld::accrued_interest(t.coupon, t.freq, b.prev_coupon, b.next_coupon, t.settle), b.accrued,
              1e-15);
  // Roll settlement +10 days (still inside the period): accrued grows by 10 days of coupon.
  const bld::Date s2 = t.settle.plus_days(10);
  const double a2 = bld::accrued_interest(t.coupon, t.freq, b.prev_coupon, b.next_coupon, s2);
  const double period_days = double(b.next_coupon - b.prev_coupon);
  EXPECT_NEAR(a2 - b.accrued, b.coupon_per_period * 10.0 / period_days, 1e-15);
  EXPECT_GT(a2, b.accrued);
}

TEST(BondWhenIssued, NewIssueAndReopening) {
  // WI new issue: settles on the dated date with a SHORT first coupon => ZERO accrued, still regular.
  const bld::Date dated = bld::Date::ymd(2024, 6, 15), fcpn = bld::Date::ymd(2024, 11, 15),
                 mat = bld::Date::ymd(2034, 11, 15);
  const double coupon = 0.045;
  const bld::BuiltBond wi = bld::us_treasury_wi(bld::Date::ymd(2024, 6, 14), dated, fcpn, mat, coupon);
  EXPECT_NEAR(wi.accrued, 0.0, 1e-15);
  pf::BondUniverse bu;
  bu.set({wi.yield});
  EXPECT_TRUE(bu.is_regular());  // short first coupon changes only the first coefficient, not the spacing
  // price/yield still round-trips.
  const double y = 0.047, clean = px::bond_clean_from_yield(wi.yield, y);
  EXPECT_NEAR(px::bond_yield_from_clean(wi.yield, clean), y, 1e-12);

  // WI reopening: settle inside the first period => accrued from the ORIGINAL dated date, > 0.
  const bld::Date reopen = bld::Date::ymd(2024, 8, 15);
  const bld::BuiltBond ro =
      bld::when_issued_bond(bld::Date::ymd(2024, 8, 14), dated, fcpn, mat, coupon, 2, reopen);
  const double E = double(fcpn - bld::Date::ymd(2024, 5, 15));
  EXPECT_NEAR(ro.accrued, (coupon / 2.0) * double(reopen - dated) / E, 1e-13);
  EXPECT_GT(ro.accrued, 0.0);
}

TEST(BondWhenIssued, RejectsLongFirstCouponForNow) {
  // A LONG first coupon (dated before the prior quasi-coupon date) is a documented follow-up, rejected.
  EXPECT_THROW(bld::us_treasury_wi(bld::Date::ymd(2024, 1, 10), bld::Date::ymd(2024, 1, 10),
                                   bld::Date::ymd(2024, 11, 15), bld::Date::ymd(2034, 11, 15), 0.04),
               std::invalid_argument);
}

TEST(BondCurve, PvDirtyAndZSpread) {
  const bld::BuiltBond b = make_bond(2039, 0.05);
  FlatCurve c{0.04};
  double manual = 0;
  for (const auto& f : b.curve.flows) manual += f.amount * c.discount(f.pay);
  EXPECT_NEAR(px::bond_pv_today<double>(b.curve, c), manual, 1e-14);
  EXPECT_NEAR(px::bond_dirty_price<double>(b.curve, c), manual / c.discount(b.curve.settle), 1e-14);
  // z-spread recovers a known parallel shift of the (flat, continuous) curve.
  const double shift = 0.0025;
  FlatCurve shifted{0.04 + shift};
  const double target = px::bond_dirty_price<double>(b.curve, shifted);
  EXPECT_NEAR(px::bond_z_spread(b.curve, c, target), shift, 1e-10);
}

TEST(BondCurve, CompiledBookBatchedReprice) {
  std::vector<px::Bond> bonds;
  for (int i = 0; i < 60; ++i) bonds.push_back(make_bond(2027 + i % 12, 0.02 + 0.0006 * i).curve);
  std::vector<double> meeting;
  std::vector<double> back;
  for (int i = 1; i <= 40; ++i) back.push_back(0.25 * i);  // knots to 10y
  const Eigen::VectorXd x = Eigen::VectorXd::Constant((int)back.size(), 0.04);
  pf::CompiledBondBook book(meeting, back, bonds);
  EXPECT_EQ(book.n_bonds(), (int)bonds.size());
  const Eigen::VectorXd dirty = book.dirty_prices(x);
  // z-spread to the book's own price is ~0 for every bond.
  const Eigen::VectorXd z0 = book.z_spreads(x, dirty);
  EXPECT_LT(z0.cwiseAbs().maxCoeff(), 1e-10);
  // Pricing 10bp of extra discounting => a ~+10bp z-spread, positive for every bond.
  const Eigen::VectorXd dirty_lo = book.dirty_prices((x.array() + 0.0010).matrix());
  const Eigen::VectorXd zpos = book.z_spreads(x, dirty_lo);
  EXPECT_GT(zpos.minCoeff(), 0.0);
  EXPECT_LT(zpos.maxCoeff(), 0.0011);
}

}  // namespace
