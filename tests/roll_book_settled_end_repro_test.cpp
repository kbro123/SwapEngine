// E5 taxonomy: T6 regression (fails on the reverted bug)
// PN2b REPRODUCTION (2026-09-14). calibration::roll_book keeps a coupon while pay > dt. A coupon whose accrual END is on or
// before dt but whose PAY is after it (the payment-lag window) survives the shift with accrual_end < 0: its interest is
// still owed, but its end-of-period notional exchange has SETTLED (exchanges sit on accrual dates -- O-X4, owner 2026-09-14).
// The pricers skip a settled INITIAL exchange (s < 0) but book the final one at any time:
//   (a) pricing::xccy_mtm_leg_pv adds +DF(e) for e < 0;
//   (b) MultiCurveBook::position_value's domestic constant-notional leg adds +DF(e_N) for e_N < 0;
//   (c) CompiledMultiCurveBook's domestic exchange row, which mirrors (b) term for term (T3 parity below).
// Expected values are written by hand from flat continuously-compounded curves: a settled exchange contributes nothing,
// an exchange dated today (t = 0) or later is still a flow (the same rule the initial exchange already follows).
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/pnl_explain.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

struct Flat {  // continuously-compounded flat curve: a closed-form reference
  double r;
  double integral(double t) const { return r * t; }
  double forward(double) const { return r; }
  double discount(double t) const { return std::exp(-r * t); }
};

// A fully-fixed MtM coupon (interest known: realized, no forecast window) with its FX notional fixed.
px::FloatCoupon fixed_mtm_coupon(double s, double e, double pay) {
  px::FloatCoupon c;
  c.obs.realized = 0.0105;
  c.obs.tau_index = 0.25;
  c.tau_pay = 0.25;
  c.pay = pay;
  c.accrual_set = true;
  c.accrual_start = s;
  c.accrual_end = e;
  c.reset_fx = 1.12;
  return c;
}

// A coupon on accrual (s, e) paying 2 days after e, with an already-fixed interest part (obs.realized) and a forecast
// window on the accrual period.
px::FloatCoupon booked_coupon(double s, double e) {
  px::FloatCoupon c;
  c.obs.sub_start = {s};
  c.obs.sub_end = {e};
  c.obs.realized = 0.01;
  c.obs.tau_index = e - s;
  c.tau_pay = e - s;
  c.pay = e + 2.0 / 365.0;
  c.accrual_set = true;
  c.accrual_start = s;
  c.accrual_end = e;
  return c;
}

// Three flat-hermite bundle curves at a sloped state (0 EUR-in-USD, 1 ESTR, 2 SOFR): what CompiledMultiCurveBook prices on.
struct World {
  cal::BundleProblem p;
  Eigen::VectorXd x;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> C;
  World() {
    const std::vector<double> knots{7.0 / 365.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    for (int c = 0; c < 3; ++c) p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, knots)});
    const double level[] = {0.022, 0.020, 0.040}, slope[] = {0.0010, 0.0008, -0.0005};
    x.resize(p.n_knots());
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) x[p.offset(c) + i] = level[c] + slope[c] * i;
    C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  }
  const cal::CurveHandle<double>& operator()(int i) const { return *C[static_cast<std::size_t>(i)]; }
};

}  // namespace

TEST(RollBookSettledEndRepro, MtmLegDoesNotBookAFinalExchangeThatSettledBeforeTheValuationDate) {
  const Flat fund{0.04}, num{0.02}, den{0.04};
  const px::FloatCoupon c = fixed_mtm_coupon(-0.26, -0.01, 0.004);  // accrual ended yesterday-ish, interest pays in 1.5 days
  ASSERT_TRUE((c).seasoned_mtm()) << "premise";
  const double want = 1.12 * (std::exp(-0.04 * 0.004) * 0.0105);  // interest only: both exchanges settled
  EXPECT_NEAR(px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{c}, 1.10, fund, fund, num, den), want, 1e-15);
}

TEST(RollBookSettledEndRepro, ControlAFinalExchangeDatedTodayOrLaterIsStillAFlow) {
  const Flat fund{0.04}, num{0.02}, den{0.04};
  for (double e : {0.0, 0.001}) {
    SCOPED_TRACE(e);
    const px::FloatCoupon c = fixed_mtm_coupon(-0.249, e, 0.004);
    const double want = 1.12 * (std::exp(-0.04 * 0.004) * 0.0105 + std::exp(-0.04 * e));
    EXPECT_NEAR(px::xccy_mtm_leg_pv<double>(std::vector<px::FloatCoupon>{c}, 1.10, fund, fund, num, den), want, 1e-15);
  }
}

TEST(RollBookSettledEndRepro, RolledIntoThePaymentLagWindowTheXccyPositionValuesInterestOnly) {
  pf::MultiCurveBook book;
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 1.0e6;
  p.fx_spot = 1.10;
  p.fwd_curve = 0;
  p.disc_curve = 1;
  p.mtm_fwd_curve = 2;
  p.mtm_disc_curve = 3;
  p.mtm_reset_num = 2;
  p.mtm_reset_den = 1;
  p.float_coupons = {booked_coupon(0.0, 0.5), booked_coupon(0.5, 1.0)};
  p.mtm_coupons = {booked_coupon(0.0, 0.5), booked_coupon(0.5, 1.0)};
  book.positions.push_back(p);

  const double dt = 1.002;  // after the last accrual end (1.0), before its pay (1.0 + 2/365 = 1.00548)
  const pf::MultiCurveBook rolled = cal::roll_book(book, dt, /*shift=*/true);
  ASSERT_EQ(rolled.positions.size(), 1u) << "premise: the last coupon's interest is still owed";
  pf::MultiCurveBook::Position q = rolled.positions[0];
  ASSERT_EQ(q.float_coupons.size(), 1u);
  ASSERT_EQ(q.mtm_coupons.size(), 1u);
  ASSERT_LT(q.float_coupons[0].accrual_end, 0.0) << "premise: the final exchange settled before the new valuation date";
  ASSERT_GT(q.float_coupons[0].pay, 0.0) << "premise: its interest pays after it";
  q.mtm_coupons[0].reset_fx = 1.12;  // the notional's FX fixed long ago: a known number

  const std::array<Flat, 4> curves{Flat{0.03}, Flat{0.04}, Flat{0.02}, Flat{0.025}};
  auto C = [&](int i) -> const Flat& { return curves[static_cast<std::size_t>(i)]; };
  const double pay = 1.0 + 2.0 / 365.0 - dt;
  // Each leg: DF(pay) * realized (the forecast window [0, 0] after the floor grows by exactly 0). No exchange on either leg.
  const double mtm = 1.12 * (std::exp(-0.025 * pay) * 0.01);
  const double dom = std::exp(-0.04 * pay) * 0.01;
  EXPECT_NEAR(pf::MultiCurveBook::position_value<double>(q, C), 1.0e6 * (mtm - dom), 1e-9);
}

// T3: the compiled book's domestic exchange row follows the templated rule. The MtM coupons are all future (unseasoned),
// so the position COMPILES; its one domestic coupon's accrual (-0.3, -0.002) settled both exchanges, its interest pays at
// 0.004. Booking the settled final exchange on either path would move the NPV by ~notional.
TEST(RollBookSettledEndRepro, CompiledXccyRowSkipsASettledDomesticFinalExchangeLikeTheTemplatedBook) {
  const World w;
  const b::XccyConv xc = b::xccy_conv("EURUSD");
  const b::Date vd = b::Date::from_iso("2026-09-15");
  const cal::Instrument legs = b::xccy_mtm_basis(vd, xc, b::resolve("2Y", vd, xc.calendar, xc.bdc, xc.spot_lag), 0, 1, 2, 1.10, 0.0);
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 1e6;
  px::FloatCoupon dom;
  dom.obs.sub_start = {0.0};
  dom.obs.sub_end = {0.0};  // the forecast window after roll_book's floor: grows by exactly 0
  dom.obs.realized = 0.01;
  dom.obs.tau_index = 0.25;
  dom.tau_pay = 0.25;
  dom.pay = 0.004;
  dom.accrual_set = true;
  dom.accrual_start = -0.3;
  dom.accrual_end = -0.002;
  p.float_coupons = {dom}; p.fwd_curve = 1; p.disc_curve = 0;
  p.mtm_coupons = legs.mtm.coupons; p.mtm_fwd_curve = 2; p.mtm_disc_curve = 2;
  p.mtm_reset_num = 0; p.mtm_reset_den = 2; p.fx_spot = 1.10;
  pf::MultiCurveBook book;
  book.positions = {p};
  const pf::CompiledMultiCurveBook cb(w.p.curves, book);
  ASSERT_EQ(cb.n_compiled(), 1) << "premise: the position compiles (its MtM coupons are unseasoned)";
  const double v = pf::MultiCurveBook::position_value<double>(p, w);
  const double dom_want = w(0).discount(0.004) * 0.01;  // interest only: no domestic exchange
  const double mtm = px::xccy_mtm_leg_pv<double>(p.mtm_coupons, 1.10, w(2), w(2), w(0), w(2));
  EXPECT_NEAR(v, 1e6 * (mtm - dom_want), 1e-9 * std::abs(v)) << "templated: the domestic leg is its interest only";
  EXPECT_NEAR(cb.npv(w.x), v, 1e-9 * std::abs(v)) << "compiled == templated";
}
