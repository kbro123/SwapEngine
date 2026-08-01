// Gate for the batched PORTFOLIO REPRICE kernel (api::BundleSession::price_portfolio) and its
// multi-curve + xccy book model (portfolio::MultiCurveBook). QuantLib-FREE: a small 3-curve bundle is
// hand-built as generic Instruments, made self-consistent from a known x_true and calibrated through a
// BundleSession, then a multi-curve + xccy book is repriced off the solved curves.
//
// Proves the correctness anchors the web reprice feature needs:
//   (a) an AT-PAR vanilla swap (fixed_rate == its model par rate) has NPV ~ 0;
//   (b) an OFF-PAR swap's NPV has the right sign AND the exact scale (-notional·Δrate·annuity);
//   (c) NPV is ADDITIVE over positions;
//   (d) price_us > 0 and the pricing pass is DETERMINISTIC (same NPV/PV01 across calls);
//   (e) an xccy position genuinely DEPENDS on the bundle's xccy/foreign curve (and a pure swap does not);
//   (f) the one-pass AAD PV01 matches a central finite-difference bump-and-reprice;
//   (g) the book JSON contract (book_from_json / price_portfolio_json) reprices identically to the struct.

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/capi.h"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

// 3-curve bundle topology (shared knots): 1 front meeting + 5 back knots => 6 knots per curve.
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 10.0};
constexpr int kNk = 6;
const std::vector<double> kSwapT{1.0, 2.0, 3.0, 5.0, 10.0};

// One OIS-style float coupon over [a,b]: a single telescoped sub-period (tau_pay == tau_index).
px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
px::FixedCoupon fixed_coupon(double a, double b) {
  px::FixedCoupon x;
  x.pay = b;
  x.tau = b - a;
  return x;
}
std::vector<px::FloatCoupon> annual_float(double T) {
  std::vector<px::FloatCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(ois_coupon(prev, t)); prev = t; }
  return v;
}
std::vector<px::FixedCoupon> annual_fixed(double T) {
  std::vector<px::FixedCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(fixed_coupon(prev, t)); prev = t; }
  return v;
}

// A par-rate swap instrument: float leg forecasts `fc` / discounts `dc`, annual fixed on `dc`.
cal::Instrument par_swap(double T, int fc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.fwd.coupons = annual_float(T);
  ins.fixed.discount = dc;
  ins.fixed.coupons = annual_fixed(T);
  return ins;
}
cal::Instrument front_rate(double a, double b, int fc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = fc;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// Build a well-determined 3-curve bundle (18 knots, 18 instruments) self-consistent at x_true.
//   curve 0 = domestic discount (ccy 0);  curve 1 = domestic forecast, discounts on 0 (multi-curve);
//   curve 2 = foreign discount (ccy 1) -- the "xccy" curve an xccy position must touch.
cal::BundleProblem build_bundle(Eigen::VectorXd& x_true) {
  cal::BundleProblem p;
  p.curves.push_back({kMeeting, kBack, /*base=*/-1, /*currency=*/0});
  p.curves.push_back({kMeeting, kBack, /*base=*/-1, /*currency=*/0});
  p.curves.push_back({kMeeting, kBack, /*base=*/-1, /*currency=*/1});
  // curve 0: self-discounting swaps + a front pin.
  p.instruments.push_back(front_rate(0.0, 0.5, 0));
  for (double T : kSwapT) p.instruments.push_back(par_swap(T, 0, 0));
  // curve 1: forecast curve, discounted on curve 0 (forecast != discount).
  p.instruments.push_back(front_rate(0.0, 0.5, 1));
  for (double T : kSwapT) p.instruments.push_back(par_swap(T, 1, 0));
  // curve 2: foreign self-discounting swaps.
  p.instruments.push_back(front_rate(0.0, 0.5, 2));
  for (double T : kSwapT) p.instruments.push_back(par_swap(T, 2, 2));

  x_true.resize(3 * kNk);
  for (int i = 0; i < kNk; ++i) {
    x_true[i] = 0.030 + 0.0010 * i;          // curve 0 ~3%
    x_true[kNk + i] = 0.033 + 0.0012 * i;    // curve 1 ~3.3% (forecast above discount)
    x_true[2 * kNk + i] = 0.020 + 0.0008 * i;  // curve 2 ~2% (foreign)
  }
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

// Build the double curve handles at a session's solved x (the same curves price_portfolio prices on).
std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves_at(const api::BundleSession& sess) {
  const cal::BundleProblem& p = sess.problem();
  const Eigen::VectorXd& x = sess.x();
  return cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
}

// A multi-curve vanilla swap position (payer-of-fixed): float forecasts `fc`, discounts `dc`; fixed on `dc`.
pf::MultiCurveBook::Position swap_position(double T, int fc, int dc, double rate, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notional;
  p.float_coupons = annual_float(T);
  p.fwd_curve = fc;
  p.disc_curve = dc;
  p.fixed_coupons = annual_fixed(T);
  p.fixed_curve = dc;
  p.fixed_rate = rate;
  return p;
}

// The model par rate of a `swap_position` off curve handles C: float_leg_pv(fc,dc) / annuity(dc).
double par_rate_of(const pf::MultiCurveBook::Position& p,
                   const std::vector<std::unique_ptr<cal::CurveHandle<double>>>& C) {
  return px::float_leg_pv<double>(p.float_coupons, *C[p.fwd_curve], *C[p.disc_curve]) /
         px::annuity<double>(p.fixed_coupons, *C[p.fixed_curve]);
}

// ---- helpers for the RISK-TRANSFORMATION primitives (jacobian / price_portfolio_risk) --------------

// The SAME state x_true both A and B calibrate to, so a portfolio's curve gradient g = dP/dx is identical
// across the two bundles (the whole point of the cross-bundle transform: same curves, different quotes).
void fill_x_true(Eigen::VectorXd& x) {
  x.resize(3 * kNk);
  for (int i = 0; i < kNk; ++i) {
    x[i] = 0.030 + 0.0010 * i;            // curve 0 ~3%
    x[kNk + i] = 0.033 + 0.0012 * i;      // curve 1 ~3.3% (forecast above discount)
    x[2 * kNk + i] = 0.020 + 0.0008 * i;  // curve 2 ~2% (foreign)
  }
}

// Coupon legs at an arbitrary step (1.0 = annual, 0.5 = semi-annual) so two bundles can quote the SAME
// curves with genuinely DIFFERENT instruments (different cashflow schedules => different Jacobian rows).
std::vector<px::FloatCoupon> float_leg(double T, double step) {
  std::vector<px::FloatCoupon> v;
  double prev = 0.0;
  for (double t = step; t <= T + 1e-9; t += step) { v.push_back(ois_coupon(prev, t)); prev = t; }
  return v;
}
std::vector<px::FixedCoupon> fixed_leg(double T, double step) {
  std::vector<px::FixedCoupon> v;
  double prev = 0.0;
  for (double t = step; t <= T + 1e-9; t += step) { v.push_back(fixed_coupon(prev, t)); prev = t; }
  return v;
}
cal::Instrument par_swap_step(double T, int fc, int dc, double step) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.fwd.coupons = float_leg(T, step);
  ins.fixed.discount = dc;
  ins.fixed.coupons = fixed_leg(T, step);
  return ins;
}

// The same 3-curve topology as build_bundle, but the swaps are quoted at `step` frequency and the whole
// bundle is made self-consistent at the SUPPLIED x_true. So build_bundle_freq(x, 1.0) and (x, 0.5)
// calibrate to the IDENTICAL curves through different instrument sets — the A/B pair the transform needs.
cal::BundleProblem build_bundle_freq(const Eigen::VectorXd& x_true, double step) {
  cal::BundleProblem p;
  p.curves.push_back({kMeeting, kBack, -1, 0});
  p.curves.push_back({kMeeting, kBack, -1, 0});
  p.curves.push_back({kMeeting, kBack, -1, 1});
  p.instruments.push_back(front_rate(0.0, 0.5, 0));
  for (double T : kSwapT) p.instruments.push_back(par_swap_step(T, 0, 0, step));
  p.instruments.push_back(front_rate(0.0, 0.5, 1));
  for (double T : kSwapT) p.instruments.push_back(par_swap_step(T, 1, 0, step));
  p.instruments.push_back(front_rate(0.0, 0.5, 2));
  for (double T : kSwapT) p.instruments.push_back(par_swap_step(T, 2, 2, step));
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

// Full model-quote vector q(x) of a bundle's instruments (for the finite-difference Jacobian check).
Eigen::VectorXd model_quotes_at(const cal::BundleProblem& prob, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(
      prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  Eigen::VectorXd q(prob.instruments.size());
  for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i)
    q[i] = cal::instrument_model_quote<double>(prob.instruments[i], cof);
  return q;
}

// A small multi-curve + xccy book that touches ALL THREE curves, so its curve gradient g = dP/dx and its
// delta ladder have nonzero mass on every curve's instruments.
pf::MultiCurveBook risk_book() {
  pf::MultiCurveBook book;
  book.positions.push_back(swap_position(10.0, /*fc=*/1, /*dc=*/0, /*rate=*/0.030, /*notional=*/1e7));
  book.positions.push_back(swap_position(5.0, /*fc=*/0, /*dc=*/0, /*rate=*/0.028, /*notional=*/-2e7));
  // xccy leg touching curve 2 (+ fx_spot), so the ladder has curve-2 sensitivity.
  pf::MultiCurveBook::Position xp;
  xp.kind = pf::MultiCurveBook::Kind::Xccy;
  xp.notional = 3e7;
  xp.float_coupons = annual_float(5.0);
  xp.fwd_curve = 0; xp.disc_curve = 0;
  xp.mtm_coupons = annual_float(5.0);
  for (auto& c : xp.mtm_coupons) c.spread = 0.005;
  xp.mtm_fwd_curve = 2; xp.mtm_disc_curve = 2; xp.mtm_reset_num = 2; xp.mtm_reset_den = 0;
  xp.fx_spot = 1.10;
  book.positions.push_back(xp);
  return book;
}

}  // namespace

// (a) at-par swap => NPV ~ 0, and (b) off-par sign + exact scale.
TEST(PortfolioReprice, ParSwapZeroNpvAndOffParScale) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const auto C = curves_at(sess);

  // A multi-curve swap: forecast curve 1, discount curve 0.
  auto pos = swap_position(10.0, /*fc=*/1, /*dc=*/0, /*rate=*/0.0, /*notional=*/1e7);
  const double par = par_rate_of(pos, C);
  const double annuity = px::annuity<double>(pos.fixed_coupons, *C[pos.disc_curve]);

  // AT PAR: NPV ~ 0 (to a few ulp of the notional).
  pos.fixed_rate = par;
  pf::MultiCurveBook at_par{{pos}};
  const api::PortfolioReprice r0 = sess.price_portfolio(at_par);
  EXPECT_EQ(r0.n, 1);
  EXPECT_LT(std::abs(r0.npv), 1e-6) << "an at-par swap must reprice to ~0 NPV";

  // OFF PAR by +1bp on the fixed leg: payer of fixed pays MORE -> NPV strictly negative, and EXACTLY
  // NPV = notional·(par·ann - (par+1bp)·ann) = -notional·1bp·annuity.
  const double bp = 1e-4;
  pos.fixed_rate = par + bp;
  pf::MultiCurveBook off_par{{pos}};
  const api::PortfolioReprice r1 = sess.price_portfolio(off_par);
  EXPECT_LT(r1.npv, 0.0) << "paying an above-par fixed rate is a loss to the payer";
  EXPECT_NEAR(r1.npv, -pos.notional * bp * annuity, 1e-6 * std::abs(pos.notional * bp * annuity));
}

// (c) NPV is additive over positions.
TEST(PortfolioReprice, NpvIsAdditive) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const auto C = curves_at(sess);

  auto a = swap_position(5.0, /*fc=*/1, /*dc=*/0, 0.0, 2.5e7);
  auto b = swap_position(10.0, /*fc=*/0, /*dc=*/0, 0.0, -1.5e7);  // a receiver (negative notional)
  a.fixed_rate = par_rate_of(a, C) + 5e-4;   // off par so NPV != 0
  b.fixed_rate = par_rate_of(b, C) - 3e-4;

  const double na = sess.price_portfolio(pf::MultiCurveBook{{a}}).npv;
  const double nb = sess.price_portfolio(pf::MultiCurveBook{{b}}).npv;
  const double nab = sess.price_portfolio(pf::MultiCurveBook{{a, b}}).npv;
  EXPECT_NEAR(nab, na + nb, 1e-6 * (std::abs(na) + std::abs(nb) + 1.0));
}

// (d) price_us > 0 and the pricing pass is deterministic (bit-identical NPV/PV01 across calls).
TEST(PortfolioReprice, PricingIsTimedAndDeterministic) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const auto C = curves_at(sess);

  pf::MultiCurveBook book;
  for (int i = 0; i < 200; ++i) {  // a book of 200 random-ish swaps across curves 0/1
    const double T = kSwapT[i % kSwapT.size()];
    auto p = swap_position(T, /*fc=*/(i % 2), /*dc=*/0, 0.0, 1e6 * (1 + (i % 7)));
    p.fixed_rate = par_rate_of(p, C) + 1e-4 * ((i % 5) - 2);  // spread around par
    book.positions.push_back(p);
  }
  const api::PortfolioReprice r1 = sess.price_portfolio(book);
  const api::PortfolioReprice r2 = sess.price_portfolio(book);
  EXPECT_EQ(r1.n, 200);
  EXPECT_GT(r1.price_us, 0.0) << "the engine must stamp a positive pricing time";
  EXPECT_GT(sess.last_price_us(), 0.0);
  EXPECT_EQ(r1.npv, r2.npv) << "the double pricing pass must be deterministic";
  EXPECT_EQ(r1.pv01, r2.pv01);
}

// (e) an xccy position genuinely depends on the bundle's xccy/foreign curve; a pure swap does not.
TEST(PortfolioReprice, XccyPositionTouchesTheForeignCurve) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem prob = build_bundle(x_true);

  // xccy position: receive a foreign resetting funding leg (curve 2) carrying a basis spread, pay a
  // domestic leg (curve 0). The funding leg's notional resets to fx_spot·DF[reset_num=2]/DF[reset_den=0]
  // and its coupons carry a 50bp basis, so its PV is nonzero and depends on curve 2 AND fx_spot. (A
  // basis-FREE MtM funding leg is a documented algebraic zero — cashflows.hpp xccy_mtm_leg_pv — so a
  // realistic xccy position quotes a basis spread on it.)
  pf::MultiCurveBook::Position xp;
  xp.kind = pf::MultiCurveBook::Kind::Xccy;
  xp.notional = 5e7;
  xp.float_coupons = annual_float(5.0);   // domestic leg
  xp.fwd_curve = 0; xp.disc_curve = 0;
  xp.mtm_coupons = annual_float(5.0);     // foreign resetting leg
  for (auto& c : xp.mtm_coupons) c.spread = 0.005;  // 50bp cross-currency basis
  xp.mtm_fwd_curve = 2; xp.mtm_disc_curve = 2; xp.mtm_reset_num = 2; xp.mtm_reset_den = 0;
  xp.fx_spot = 1.10;
  pf::MultiCurveBook xbook{{xp}};

  // A pure domestic swap (curves 0/1), which must NOT depend on curve 2.
  auto sp = swap_position(5.0, /*fc=*/1, /*dc=*/0, 0.03, 5e7);
  pf::MultiCurveBook sbook{{sp}};

  auto value_at = [&](const pf::MultiCurveBook& bk, const Eigen::VectorXd& x) {
    const auto C = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
    const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
    return bk.value<double>(cof);
  };
  // Bump ONLY curve 2's knot forwards by +10bp.
  Eigen::VectorXd x_bump = x_true;
  for (int i = 0; i < kNk; ++i) x_bump[prob.offset(2) + i] += 1e-3;

  const double xv0 = value_at(xbook, x_true), xv1 = value_at(xbook, x_bump);
  const double sv0 = value_at(sbook, x_true), sv1 = value_at(sbook, x_bump);
  EXPECT_GT(std::abs(xv1 - xv0), 1.0) << "an xccy position must move when the foreign/xccy curve moves";
  EXPECT_EQ(sv0, sv1) << "a pure domestic swap must be insensitive to the foreign curve";

  // Also sensitive to the FX spot.
  pf::MultiCurveBook::Position xp2 = xp; xp2.fx_spot = 1.20;
  EXPECT_NE(value_at(pf::MultiCurveBook{{xp2}}, x_true), xv0) << "xccy value must depend on fx_spot";
}

// (f) the one-pass AAD PV01 matches a central finite-difference parallel-bump reprice.
TEST(PortfolioReprice, Pv01MatchesFiniteDifference) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const auto C = curves_at(sess);

  pf::MultiCurveBook book;
  for (int i = 0; i < 25; ++i) {
    auto p = swap_position(kSwapT[i % kSwapT.size()], (i % 2), 0, 0.0, 1e6 * (1 + i));
    p.fixed_rate = par_rate_of(p, C) + 2e-4 * ((i % 3) - 1);
    book.positions.push_back(p);
  }
  const double pv01 = sess.price_portfolio(book).pv01;

  // Central FD of a +1bp PARALLEL shift of every fitted knot forward: (NPV(x+h) - NPV(x-h)) / 2.
  const cal::BundleProblem& prob = sess.problem();
  const double h = 1e-4;
  auto npv_at = [&](const Eigen::VectorXd& x) {
    const auto Cx = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
    const auto cof = [&Cx](int i) -> const cal::CurveHandle<double>& { return *Cx[i]; };
    return book.value<double>(cof);
  };
  Eigen::VectorXd xp = sess.x(), xm = sess.x();
  xp.array() += h; xm.array() -= h;
  const double fd = 0.5 * (npv_at(xp) - npv_at(xm));
  EXPECT_NEAR(pv01, fd, 1e-6 * std::abs(fd)) << "AAD PV01 must match a central-difference parallel bump";
  EXPECT_NE(pv01, 0.0);
}

// (g) the book JSON contract reprices identically to the struct-built book.
TEST(PortfolioReprice, BookJsonMatchesStruct) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const auto C = curves_at(sess);

  auto p = swap_position(5.0, /*fc=*/1, /*dc=*/0, 0.0, 1e7);
  p.fixed_rate = par_rate_of(p, C) + 7e-4;
  const double want = sess.price_portfolio(pf::MultiCurveBook{{p}}).npv;

  // Build the equivalent book JSON (reusing the coupon field shapes the instrument parser uses).
  json::array coupons, fixed;
  for (const auto& c : p.float_coupons) {
    json::object obs{{"sub_start", json::array{c.obs.sub_start[0]}}, {"sub_end", json::array{c.obs.sub_end[0]}},
                     {"tau_index", c.obs.tau_index}};
    coupons.push_back(json::object{{"obs", obs}, {"pay", c.pay}, {"tau_pay", c.tau_pay}});
  }
  for (const auto& c : p.fixed_coupons) fixed.push_back(json::object{{"pay", c.pay}, {"tau", c.tau}});
  json::object pos{{"kind", "swap"}, {"notional", p.notional}, {"fixed_rate", p.fixed_rate},
                   {"fwd_curve", p.fwd_curve}, {"disc_curve", p.disc_curve}, {"fixed_curve", p.fixed_curve},
                   {"float_coupons", coupons}, {"fixed_coupons", fixed}};
  json::object book{{"positions", json::array{pos}}};
  const std::string doc = json::serialize(json::value(book));

  const api::PortfolioReprice r = sess.price_portfolio_json(doc);
  EXPECT_EQ(r.n, 1);
  EXPECT_NEAR(r.npv, want, 1e-6 * (std::abs(want) + 1.0)) << "the JSON book path must match the struct path";
}

// =================================================================================================
// RISK-TRANSFORMATION primitives: jacobian() (J = dq/dx), price_portfolio_risk() (npv, curve_grad = dP/dx,
// ladder = dP/dq), and the cross-bundle transform assembled from J and M = risk_operator().
// =================================================================================================

// The calibration Jacobian J(i,j) = d(model_quote_i)/dx_j must match a central finite difference. For the
// linear ParRate/Rate instruments the calibration residual is q − market, so d(residual)/dx == dq/dx.
TEST(PortfolioRisk, JacobianMatchesFiniteDifference) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);
  const cal::BundleProblem& prob = sess.problem();
  const Eigen::VectorXd x = sess.x();

  const Eigen::MatrixXd J = sess.jacobian();  // n_res x n_knots
  ASSERT_EQ(J.rows(), prob.n_residuals());
  ASSERT_EQ(J.cols(), prob.n_knots());

  const double h = 1e-6;
  for (int j = 0; j < prob.n_knots(); ++j) {
    Eigen::VectorXd xp = x, xm = x;
    xp[j] += h; xm[j] -= h;
    const Eigen::VectorXd fd = (model_quotes_at(prob, xp) - model_quotes_at(prob, xm)) / (2 * h);
    for (int i = 0; i < prob.n_residuals(); ++i)
      EXPECT_NEAR(J(i, j), fd[i], 1e-6 * std::abs(fd[i]) + 1e-9)
          << "J(" << i << "," << j << ") must equal the central-difference dq/dx";
  }
}

// The native delta ladder equals curve_grad^T · M, AND satisfies PnL invariance: for any state move dx with
// its induced quote move dq = J·dx, dot(ladder, dq) == dot(curve_grad, dx).
TEST(PortfolioRisk, NativeLadderEqualsGradTimesM) {
  Eigen::VectorXd x_true;
  fill_x_true(x_true);
  api::BundleSession sess(build_bundle_freq(x_true, 1.0));
  sess.calibrate(x_true);

  const pf::MultiCurveBook book = risk_book();
  const api::PortfolioRisk risk = sess.price_portfolio_risk(book);
  const Eigen::MatrixXd M = sess.risk_operator();       // n_knots x n_res
  const Eigen::MatrixXd J = sess.jacobian();            // n_res x n_knots

  ASSERT_EQ(risk.curve_grad.size(), sess.problem().n_knots());
  ASSERT_EQ(risk.ladder.size(), sess.problem().n_residuals());

  // ladder == curve_grad^T · M  (i.e. M^T · curve_grad), the exact definition.
  const Eigen::VectorXd want = M.transpose() * risk.curve_grad;
  EXPECT_LT((risk.ladder - want).cwiseAbs().maxCoeff(), 1e-8 * (want.cwiseAbs().maxCoeff() + 1.0));

  // PnL invariance: dot(ladder, dq) == dot(curve_grad, dx) for a consistent perturbation dq = J·dx.
  Eigen::VectorXd dx(sess.problem().n_knots());
  for (int i = 0; i < dx.size(); ++i) dx[i] = 1e-4 * std::sin(0.7 * i + 1.0);  // arbitrary small move
  const Eigen::VectorXd dq = J * dx;
  const double lhs = risk.ladder.dot(dq), rhs = risk.curve_grad.dot(dx);
  EXPECT_NEAR(lhs, rhs, 1e-8 * (std::abs(rhs) + 1.0)) << "delta·dq must equal grad·dx (PnL invariance)";

  // Sanity: the NPV agrees with price_portfolio, and the ladder is nontrivial on every curve.
  EXPECT_NEAR(risk.npv, sess.price_portfolio(book).npv, 1e-6 * (std::abs(risk.npv) + 1.0));
  EXPECT_GT(risk.ladder.cwiseAbs().maxCoeff(), 0.0);
}

// THE KEY TEST: transform a risk ladder from bundle A's instruments to bundle B's via T = J_A · M_B.
// A and B calibrate the SAME curves (same x_true) with DIFFERENT instruments (annual vs semi-annual swaps),
// so g = dP/dx is shared and delta_B_transform = delta_A · (J_A · M_B) must recover delta_B_native = g · M_B.
TEST(PortfolioRisk, CrossBundleLadderTransform) {
  Eigen::VectorXd x_true;
  fill_x_true(x_true);
  api::BundleSession A(build_bundle_freq(x_true, 1.0));   // annual-swap quotes
  api::BundleSession B(build_bundle_freq(x_true, 0.5));   // semi-annual-swap quotes, SAME curves
  A.calibrate(x_true);
  B.calibrate(x_true);
  // Both must land on the shared state (else "same curves" is false and the transform is meaningless).
  ASSERT_LT((A.x() - x_true).cwiseAbs().maxCoeff(), 1e-9);
  ASSERT_LT((B.x() - x_true).cwiseAbs().maxCoeff(), 1e-9);

  const pf::MultiCurveBook book = risk_book();
  const api::PortfolioRisk rA = A.price_portfolio_risk(book);
  const api::PortfolioRisk rB = B.price_portfolio_risk(book);
  // g = dP/dx is identical across the two bundles (same curves, same book).
  ASSERT_LT((rA.curve_grad - rB.curve_grad).cwiseAbs().maxCoeff(), 1e-8);

  const Eigen::VectorXd g = rA.curve_grad;
  const Eigen::MatrixXd J_A = A.jacobian();          // n_res_A x n_knots
  const Eigen::MatrixXd M_A = A.risk_operator();     // n_knots x n_res_A
  const Eigen::MatrixXd M_B = B.risk_operator();     // n_knots x n_res_B

  const Eigen::VectorXd delta_A = rA.ladder;         // == M_A^T g
  const Eigen::VectorXd delta_B_native = rB.ladder;  // == M_B^T g

  // Cross-bundle transform: T = J_A · M_B (n_res_A x n_res_B); delta_B_transform = delta_A · T = T^T delta_A.
  const Eigen::MatrixXd T = J_A * M_B;
  const Eigen::VectorXd delta_B_transform = T.transpose() * delta_A;

  const double scaleB = delta_B_native.cwiseAbs().maxCoeff() + 1.0;
  const double abs_err = (delta_B_transform - delta_B_native).cwiseAbs().maxCoeff();
  EXPECT_LT(abs_err, 1e-6 * scaleB)
      << "A->B transformed ladder must match B's native ladder (max abs err " << abs_err << ")";

  // Self-transform A->A: delta_A · (J_A · M_A) must return delta_A.
  const Eigen::MatrixXd T_AA = J_A * M_A;
  const Eigen::VectorXd delta_A_self = T_AA.transpose() * delta_A;
  EXPECT_LT((delta_A_self - delta_A).cwiseAbs().maxCoeff(),
            1e-8 * (delta_A.cwiseAbs().maxCoeff() + 1.0))
      << "self-transform A->A must be the identity on the native ladder";

  // Recover the curve gradient from the native ladder: g == delta_A · J_A.
  const Eigen::VectorXd g_recovered = J_A.transpose() * delta_A;
  EXPECT_LT((g_recovered - g).cwiseAbs().maxCoeff(), 1e-8 * (g.cwiseAbs().maxCoeff() + 1.0))
      << "delta_A · J_A must recover the curve gradient g = dP/dx";
}

// NATIVE cross-bundle transform: B.transform_matrix(A.problem()) == the manual J_A · M_B, and applying it
// to A's ladder recovers B's native ladder. cross_jacobian(A) prices A's instruments on B's curve and (since
// A and B share x_true) equals A's own Jacobian. The JSON convenience matches the struct path, and the
// same_curve_set guard rejects an incompatible bundle.
TEST(PortfolioRisk, NativeTransformMatrixMatchesManualAndNative) {
  Eigen::VectorXd x_true;
  fill_x_true(x_true);
  api::BundleSession A(build_bundle_freq(x_true, 1.0));
  api::BundleSession B(build_bundle_freq(x_true, 0.5));
  A.calibrate(x_true);
  B.calibrate(x_true);
  const pf::MultiCurveBook book = risk_book();
  const Eigen::VectorXd delta_A = A.price_portfolio_risk(book).ladder;
  const Eigen::VectorXd delta_B_native = B.price_portfolio_risk(book).ladder;

  // Native analytic transform T = cross_jacobian_B(A) · M_B (n_res_A x n_res_B).
  const Eigen::MatrixXd T = B.transform_matrix(A.problem());
  ASSERT_EQ(T.rows(), A.problem().n_residuals());
  ASSERT_EQ(T.cols(), B.problem().n_residuals());
  const Eigen::MatrixXd T_manual = A.jacobian() * B.risk_operator();  // states coincide -> cross_jac == J_A
  EXPECT_LT((T - T_manual).cwiseAbs().maxCoeff(), 1e-9 * (T_manual.cwiseAbs().maxCoeff() + 1.0))
      << "native transform_matrix must equal the manual J_A · M_B";

  const Eigen::VectorXd delta_B_transform = T.transpose() * delta_A;
  EXPECT_LT((delta_B_transform - delta_B_native).cwiseAbs().maxCoeff(),
            1e-6 * (delta_B_native.cwiseAbs().maxCoeff() + 1.0))
      << "delta_A · T must reproduce B's native ladder";

  // cross_jacobian on the shared state equals A's own calibration Jacobian.
  const Eigen::MatrixXd Jx = B.cross_jacobian(A.problem());
  EXPECT_LT((Jx - A.jacobian()).cwiseAbs().maxCoeff(), 1e-9 * (A.jacobian().cwiseAbs().maxCoeff() + 1.0));

  // JSON convenience matches the struct path.
  const std::string a_json = json::serialize(api::bundle_to_json(A.problem()));
  const Eigen::MatrixXd Tj = B.transform_matrix_json(a_json);
  EXPECT_LT((Tj - T).cwiseAbs().maxCoeff(), 1e-12) << "transform_matrix_json must match the struct overload";

  // same_curve_set guard: a bundle with a different curve count is rejected (and cross_jacobian throws).
  cal::BundleProblem other = build_bundle_freq(x_true, 1.0);
  other.curves.pop_back();
  EXPECT_FALSE(B.same_curve_set(other));
  EXPECT_THROW(B.cross_jacobian(other), std::invalid_argument);
}

// REGULARIZED risk ladder: price_portfolio_risk(book, reg) damps the ladder's alternating-sign fan-out into a
// stabilised (localised) key-rate hedge, while preserving the parallel P&L EXACTLY. Proof of the latter: for a
// state move dx in R's null space (a constant/parallel forward shift, R·dx = 0), (JᵀJ + RᵀR)⁻¹JᵀJ·dx = dx, so
// delta_reg·(J·dx) == g·dx regardless of lambda -- the smoother never biases DV01, only the ladder's shape.
TEST(PortfolioRisk, RegularizedLadderPreservesParallelPnLAndDampsShape) {
  Eigen::VectorXd x_true;
  fill_x_true(x_true);
  api::BundleSession sess(build_bundle_freq(x_true, 1.0));
  sess.calibrate(x_true);
  const cal::BundleProblem& prob = sess.problem();
  const pf::MultiCurveBook book = risk_book();

  api::RegSpec reg;
  reg.lambda = 1e-3;
  reg.tension = false;  // discrete second-difference curvature penalty (the classic key-rate stabiliser)
  for (int c = 0; c < prob.n_curves(); ++c) reg.curves.push_back(c);

  const api::PortfolioRisk r0 = sess.price_portfolio_risk(book);        // raw M
  const api::PortfolioRisk rR = sess.price_portfolio_risk(book, reg);   // regularised M
  const Eigen::VectorXd g = r0.curve_grad;
  const Eigen::MatrixXd J = sess.jacobian();

  // reg changed the ladder's shape (it is a genuinely different, damped ladder).
  EXPECT_GT((rR.ladder - r0.ladder).cwiseAbs().maxCoeff(), 1e-6 * (r0.ladder.cwiseAbs().maxCoeff() + 1.0))
      << "a nonzero risk regulariser must change the ladder";

  // Parallel-shift P&L is preserved EXACTLY by the regularised ladder (DV01 unbiased by the smoother).
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(prob.n_knots());
  const Eigen::VectorXd dq = J * ones;
  const double pnl_true = g.dot(ones);
  EXPECT_NEAR(rR.ladder.dot(dq), pnl_true, 1e-8 * (std::abs(pnl_true) + 1.0))
      << "regularised ladder must preserve the parallel-shift P&L";
  EXPECT_NEAR(r0.ladder.dot(dq), pnl_true, 1e-8 * (std::abs(pnl_true) + 1.0))
      << "raw ladder preserves it too (sanity)";
}

// risk_us is engine-stamped positive, deterministic, and mirrored in last_risk_us().
TEST(PortfolioRisk, RiskIsTimedAndDeterministic) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);

  const pf::MultiCurveBook book = risk_book();
  const api::PortfolioRisk r1 = sess.price_portfolio_risk(book);
  const api::PortfolioRisk r2 = sess.price_portfolio_risk(book);
  EXPECT_EQ(r1.n, static_cast<int>(book.positions.size()));
  EXPECT_GT(r1.risk_us, 0.0) << "the engine must stamp a positive risk time";
  EXPECT_GT(sess.last_risk_us(), 0.0);
  // The numeric result is deterministic across calls (only the timing varies).
  EXPECT_EQ(r1.npv, r2.npv);
  EXPECT_EQ(r1.curve_grad, r2.curve_grad);
  EXPECT_EQ(r1.ladder, r2.ladder);
}

// The JSON risk path matches the struct path (same book_from_json schema as price_portfolio_json).
TEST(PortfolioRisk, RiskJsonMatchesStruct) {
  Eigen::VectorXd x_true;
  api::BundleSession sess(build_bundle(x_true));
  sess.calibrate(x_true);

  auto p = swap_position(5.0, /*fc=*/1, /*dc=*/0, /*rate=*/0.03, /*notional=*/1e7);
  const api::PortfolioRisk want = sess.price_portfolio_risk(pf::MultiCurveBook{{p}});

  json::array coupons, fixed;
  for (const auto& c : p.float_coupons) {
    json::object obs{{"sub_start", json::array{c.obs.sub_start[0]}}, {"sub_end", json::array{c.obs.sub_end[0]}},
                     {"tau_index", c.obs.tau_index}};
    coupons.push_back(json::object{{"obs", obs}, {"pay", c.pay}, {"tau_pay", c.tau_pay}});
  }
  for (const auto& c : p.fixed_coupons) fixed.push_back(json::object{{"pay", c.pay}, {"tau", c.tau}});
  json::object pos{{"kind", "swap"}, {"notional", p.notional}, {"fixed_rate", p.fixed_rate},
                   {"fwd_curve", p.fwd_curve}, {"disc_curve", p.disc_curve}, {"fixed_curve", p.fixed_curve},
                   {"float_coupons", coupons}, {"fixed_coupons", fixed}};
  const std::string doc = json::serialize(json::value(json::object{{"positions", json::array{pos}}}));

  const api::PortfolioRisk got = sess.price_portfolio_risk_json(doc);
  EXPECT_EQ(got.n, 1);
  EXPECT_NEAR(got.npv, want.npv, 1e-6 * (std::abs(want.npv) + 1.0));
  EXPECT_LT((got.ladder - want.ladder).cwiseAbs().maxCoeff(),
            1e-8 * (want.ladder.cwiseAbs().maxCoeff() + 1.0));
}

// The C ABI (capi.h swaps_run_json) drives calibrate + portfolio_risk + cross-bundle transform end to end
// through the JSON seam -- the exact path an Excel/.NET/ctypes host would take, no C++ Session object.
TEST(CApi, RunJsonPortfolioRiskAndTransform) {
  Eigen::VectorXd x_true;
  fill_x_true(x_true);
  const cal::BundleProblem A = build_bundle_freq(x_true, 1.0);  // primary (target) bundle
  const cal::BundleProblem B = build_bundle_freq(x_true, 0.5);  // source bundle (same curves)

  auto p = swap_position(5.0, /*fc=*/1, /*dc=*/0, /*rate=*/0.03, /*notional=*/1e7);
  json::array coupons, fixed;
  for (const auto& c : p.float_coupons) {
    json::object obs{{"sub_start", json::array{c.obs.sub_start[0]}}, {"sub_end", json::array{c.obs.sub_end[0]}},
                     {"tau_index", c.obs.tau_index}};
    coupons.push_back(json::object{{"obs", obs}, {"pay", c.pay}, {"tau_pay", c.tau_pay}});
  }
  for (const auto& c : p.fixed_coupons) fixed.push_back(json::object{{"pay", c.pay}, {"tau", c.tau}});
  json::object pos{{"kind", "swap"}, {"notional", p.notional}, {"fixed_rate", p.fixed_rate},
                   {"fwd_curve", p.fwd_curve}, {"disc_curve", p.disc_curve}, {"fixed_curve", p.fixed_curve},
                   {"float_coupons", coupons}, {"fixed_coupons", fixed}};
  json::object book{{"positions", json::array{pos}}};

  json::array x0;
  for (int i = 0; i < x_true.size(); ++i) x0.push_back(x_true[i]);
  json::object req{{"bundle", api::bundle_to_json(A)},
                   {"x0", x0},
                   {"portfolio_risk", book},
                   {"transform", json::object{{"source_bundle", api::bundle_to_json(B)}}}};
  const std::string reqs = json::serialize(json::value(std::move(req)));

  const char* resp = swaps_run_json(reqs.c_str());
  ASSERT_NE(resp, nullptr);
  const json::value rv = json::parse(resp);
  swaps_string_free(resp);
  const auto& ro = rv.as_object();
  ASSERT_FALSE(ro.contains("error")) << json::serialize(rv);

  ASSERT_TRUE(ro.contains("portfolio_risk"));
  const auto& pr = ro.at("portfolio_risk").as_object();
  EXPECT_EQ(pr.at("n").as_int64(), 1);
  EXPECT_EQ(pr.at("ladder").as_array().size(), static_cast<std::size_t>(A.n_residuals()));

  ASSERT_TRUE(ro.contains("transform"));
  const auto& T = ro.at("transform").as_array();
  EXPECT_EQ(T.size(), static_cast<std::size_t>(B.n_residuals()));                    // rows = source residuals
  EXPECT_EQ(T.at(0).as_array().size(), static_cast<std::size_t>(A.n_residuals()));   // cols = this residuals

  EXPECT_EQ(swaps_run_json(nullptr), nullptr);  // the one non-JSON contract: null in -> null out
}
