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
  EXPECT_NEAR(r.convexity, (Pp - 2 * P + Pm) / (h * h) / P, 1e-4);
  // Macaulay = Modified * (1 + y/f).
  EXPECT_NEAR(r.macaulay_duration, r.modified_duration * (1.0 + y / b.yield.freq), 1e-14);
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
    EXPECT_NEAR(cx[i], r.convexity, 1e-13);
  }
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
