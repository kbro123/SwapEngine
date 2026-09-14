// E5 taxonomy: T6 regression (fails on the reverted bug)
// XB1 REPRODUCTION (2026-09-14). The MtM cross-currency basis quote (calibration/problem.hpp QuoteKind::XccyMtmBasis)
// stands the constant-notional leg's notional exchanges in with the telescoped self-forecast leg pv_self:
//     b = (pv_self - pv_foreign) / annuity + mtm / (fx_spot * annuity)
// pv_self = sum_i (DF(s_i)/DF(e_i) - 1) DF(p_i) equals the exchange pair DF(s_0) - DF(e_N) only when every coupon pays on
// its accrual end. The builder (build::xccy_mtm_basis) pays the coupons payment_lag business days LATER, while the
// notional exchanges settle ON the start and maturity dates (owner decision 2026-09-14 after sourced research: ARRC 2020,
// Bank of Canada / CARR term sheets, AFMA, Tokyo Tanshi, Clarus, rateslib / ORE / QuantLib defaults -- only coupons are
// delayed). So the quote is off by E / annuity with E = sum_i (DF(s_i)/DF(e_i) - 1)(DF(e_i) - DF(p_i)).
//
// The reference below keeps the engine's own foreign-leg and MtM-leg kernels (not under test) and writes the constant
// leg's exchange pair out explicitly from the instrument's own accrual dates. A control shows the two formulas agree
// exactly when the payment lag is 0.
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

// Curves: 0 = EUR discounted under USD collateral (the pinned self curve), 1 = ESTR forecast, 2 = USD SOFR.
struct World {
  cal::BundleProblem p;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> C;
  World() {
    const std::vector<double> knots{0.5, 1.0, 2.0, 3.0, 5.0};
    for (int c = 0; c < 3; ++c) p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite({}, knots)});
    const double level[] = {0.022, 0.020, 0.040}, slope[] = {0.0010, 0.0008, -0.0005};
    Eigen::VectorXd x(p.n_knots());
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i)
        x[p.offset(c) + i] = level[c] + slope[c] * i;
    C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  }
  const cal::CurveHandle<double>& curve(int i) const { return *C[static_cast<std::size_t>(i)]; }
};

double explicit_pair_quote(const cal::Instrument& ins, const World& w) {
  const auto& self = ins.fwd.coupons;
  const double s0 = self.front().accrual_start, eN = self.back().accrual_end;
  const cal::CurveHandle<double>& d = w.curve(ins.fwd.discount);
  const double ann = px::annuity<double>(ins.fixed.coupons, w.curve(ins.fixed.discount));
  const double pv_fx = px::float_leg_pv<double>(ins.bench.coupons, w.curve(ins.bench.forecast), w.curve(ins.bench.discount));
  const double mtm = px::xccy_mtm_leg_pv<double>(ins.mtm.coupons, ins.mtm.fx_spot, w.curve(ins.mtm.forecast),
                                                 w.curve(ins.mtm.discount), w.curve(ins.mtm.reset_num),
                                                 w.curve(ins.mtm.reset_den));
  return (d.discount(s0) - d.discount(eN) - pv_fx) / ann + mtm / (ins.mtm.fx_spot * ann);
}

double engine_quote(const cal::Instrument& ins, const World& w) {
  const auto curve_of = [&w](int i) -> const cal::CurveHandle<double>& { return w.curve(i); };
  return cal::instrument_model_quote<double>(ins, curve_of);
}

}  // namespace

TEST(XccyBasisExchangePairRepro, TheConstantLegsExchangesSettleOnTheAccrualDatesNotWithTheLaggedCoupons) {
  const World w;
  const b::XccyConv x = b::xccy_conv("EURUSD");
  ASSERT_EQ(x.pay_lag, 2) << "premise: the product lags its coupons";
  const b::Date vd = b::Date::from_iso("2026-09-15");
  for (const char* tenor : {"1Y", "2Y", "5Y"}) {
    SCOPED_TRACE(tenor);
    const b::Date mat = b::resolve(tenor, vd, x.calendar, x.bdc, x.spot_lag);
    const cal::Instrument ins = b::xccy_mtm_basis(vd, x, mat, /*ci=*/0, /*foreign=*/1, /*fund=*/2, /*fx_spot=*/1.10, 0.0);
    ASSERT_TRUE(ins.fwd.coupons.front().accrual_set && ins.fwd.coupons.back().accrual_set) << "premise: accrual dates";
    ASSERT_TRUE(ins.bench.coupons.back().accrual_set) << "premise";
    ASSERT_GT(ins.bench.coupons.back().pay, ins.bench.coupons.back().accrual_end) << "premise: the coupons pay late";
    const double want = explicit_pair_quote(ins, w);
    const double got = engine_quote(ins, w);
    EXPECT_NEAR(got, want, swaps::tol::literal) << "basis off by " << (got - want) * 1e4 << " bp";
  }
}

TEST(XccyBasisExchangePairRepro, ControlWithoutAPaymentLagTheTwoFormulasAgree) {
  const World w;
  b::XccyConv x = b::xccy_conv("EURUSD");
  x.pay_lag = 0;  // coupons pay on the accrual end: pv_self telescopes to the exchange pair exactly
  const b::Date vd = b::Date::from_iso("2026-09-15");
  const b::Date mat = b::resolve("2Y", vd, x.calendar, x.bdc, x.spot_lag);
  const cal::Instrument ins = b::xccy_mtm_basis(vd, x, mat, 0, 1, 2, 1.10, 0.0);
  EXPECT_NEAR(engine_quote(ins, w), explicit_pair_quote(ins, w), swaps::tol::literal);
}
