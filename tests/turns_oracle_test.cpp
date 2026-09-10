// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
// Turns as a calibration instrument, end-to-end oracle (docs/turns-calibration.md). A curve carrying a
// turn OVERLAY must be a usable, arbitrage-clean discount curve: wrap the TurnedCurve handle in
// CurveTermStructure, price OIS swaps off it with QuantLib, and confirm QuantLib's fairRate equals our
// kernel priced off the SAME turned curve. Because the discount factors are identical by construction,
// any gap is a pricing/curve bug — the real proof that a turned curve is drop-in wherever a curve is
// templated. A second case pins the exp(−δ·τ) discount-factor shift QuantLib itself sees across the turn.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/curve_spec.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace px = swaps::pricing;
namespace cal = swaps::calibration;
namespace qlx = swaps::qlx;

namespace {
px::CurveStructure turned_spec() {
  px::CurveStructure s;
  s.regions = swaps::curve::flat_hermite({0.25, 0.5}, {1, 2, 3, 5, 7, 10});
  s.base = -1;
  s.turns = {{0.98, 1.0}};  // a year-end-ish turn window (year fractions, resolved web-side)
  return s;
}
Eigen::VectorXd turned_state(double delta) {
  Eigen::VectorXd x(9);  // 2 front + 6 back interp + 1 δ
  x << 0.030, 0.032, 0.035, 0.037, 0.039, 0.041, 0.042, 0.043, delta;
  return x;
}
}  // namespace

TEST(TurnsOracle, QuantLibPricesOisOffTurnedDiscountFactors) {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);

  std::vector<px::CurveStructure> specs{turned_spec()};
  const Eigen::VectorXd x = turned_state(0.0030);  // +30 bp turn
  auto C = cal::build_bundle_curves<double>(specs, [&](int, int i) { return x[i]; });

  auto ts = ext::make_shared<qlx::CurveTermStructure<cal::CurveHandle<double>>>(mk.today, mk.dc, C[0].get());
  ts->enableExtrapolation();
  h.linkTo(ts);  // SOFR forecasts and discounts off the turned curve

  double worst = 0.0;
  for (int T : {2, 3, 5, 7, 10}) {
    auto swap = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(Period(T, Years), mk.sofr, 0.03).withDiscountingTermStructure(h));
    swap->deepUpdate();
    const double ql = swap->fairRate();
    const auto fl = qlx::extract_float_leg(swap->overnightLeg(), mk.today, mk.dc);
    const auto fx = qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
    const double ours = px::par_rate<double>(fl, fx, *C[0], *C[0]);
    worst = std::max(worst, std::abs(ours - ql) / std::max(1.0, std::abs(ql)));
  }
  std::cout << "  [turns-oracle] max rel |ours - QuantLib fairRate| = " << worst << "\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "turned curve must price OIS identically to QuantLib";
}

TEST(TurnsOracle, TurnShiftsQuantLibDiscountFactorsByExpMinusDeltaTau) {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);

  std::vector<px::CurveStructure> specs{turned_spec()};
  const double delta = 0.0030, a = 0.98, b = 1.0;
  auto C0 = cal::build_bundle_curves<double>(specs, [&, x = turned_state(0.0)](int, int i) { return x[i]; });
  auto C1 = cal::build_bundle_curves<double>(specs, [&, x = turned_state(delta)](int, int i) { return x[i]; });

  auto ts0 = ext::make_shared<qlx::CurveTermStructure<cal::CurveHandle<double>>>(mk.today, mk.dc, C0[0].get());
  auto ts1 = ext::make_shared<qlx::CurveTermStructure<cal::CurveHandle<double>>>(mk.today, mk.dc, C1[0].get());
  ts0->enableExtrapolation();
  ts1->enableExtrapolation();

  for (double t : {0.5, 0.99, 1.5, 3.0}) {
    const double ov = px::turn_overlap(t, {a, b});
    const double ratio = ts1->discount(t) / ts0->discount(t);
    EXPECT_NEAR(ratio, std::exp(-delta * ov), 1e-11) << "at t=" << t;
  }
}
