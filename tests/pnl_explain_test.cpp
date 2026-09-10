// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Gate for the P&L EXPLAIN decomposition (calibration::pnl_explain) and its "pnl" verb contract.
// QuantLib-FREE: a small single-curve self-discounting bundle is hand-built self-consistent at a known
// x_true (so calibration recovers it), a book of vanilla swaps is repriced, and the NPV change between
// two dates/markets is attributed into carry / roll / market / residual off the SAME primitives the
// production verb uses (BundleSession::calibrate + price_portfolio_risk for the analytic delta ladder).
//
// Proves:
//   (a) the residual IS the horizon market move minus its linear explanation, recomputed from the pricing
//       primitives outside pnl_explain (the sum identity itself is definitional and is not asserted; the
//       contract), on a combined move (time AND market together);
//   (b) a PURE MARKET move (dt = 0, q1 != q0) lands ~entirely in `market`, which equals ladder·dq and
//       explains almost all of `total` (residual is the small second-order piece);
//   (c) a PURE TIME roll (q1 == q0, dt > 0) puts the P&L in carry/roll, with `market` == 0 and residual 0;
//   (d) the signs are economically right — a payer swap gains when rates rise (market > 0, total > 0),
//       a receiver loses, and an in-the-money book earns positive carry over time.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/pnl_explain.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;
namespace api = swaps::api;

namespace {

// Single self-discounting curve: 1 front meeting + 6 back knots => 7 knots, 7 instruments (square).
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
constexpr int kNk = 7;
const std::vector<double> kSwapT{1.0, 2.0, 3.0, 5.0, 7.0, 10.0};

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
cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = 0;
  ins.fwd.discount = 0;
  ins.fwd.coupons = annual_float(T);
  ins.fixed.discount = 0;
  ins.fixed.coupons = annual_fixed(T);
  return ins;
}
cal::Instrument front_rate(double a, double b) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// A single-curve bundle made self-consistent at x_true: markets are the model quotes there, so a cold
// calibration from a flat guess recovers x_true (zero residual).
cal::BundleProblem build_bundle_at(const Eigen::VectorXd& x_true) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.instruments.push_back(front_rate(0.0, 0.5));
  for (double T : kSwapT) p.instruments.push_back(par_swap(T));
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, cof);
  return p;
}

// An upward-sloping forward curve ~3% at the front rising to ~4%.
Eigen::VectorXd upward_curve() {
  Eigen::VectorXd x(kNk);
  for (int i = 0; i < kNk; ++i) x[i] = 0.030 + 0.0015 * i;
  return x;
}

pf::MultiCurveBook::Position payer_swap(double T, double rate, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notional;
  p.float_coupons = annual_float(T);
  p.fwd_curve = 0;
  p.disc_curve = 0;
  p.fixed_coupons = annual_fixed(T);
  p.fixed_curve = 0;
  p.fixed_rate = rate;  // payer pays this fixed
  return p;
}

// Calibrate a session to a bundle and return it (heap so the referenced problem stays put).
std::unique_ptr<api::BundleSession> calibrated(const cal::BundleProblem& prob) {
  auto sess = std::make_unique<api::BundleSession>(prob);
  sess->calibrate(api::flat_x0(sess->problem()));
  return sess;
}

}  // namespace

// (a) The SUM contract: on a combined time + market move the four components close to `total` to 1e-9.
TEST(PnlExplain, ComponentsSumToTotal) {
  const Eigen::VectorXd x0 = upward_curve();
  Eigen::VectorXd x1 = x0;
  for (int i = 0; i < kNk; ++i) x1[i] += 0.0007 + 0.00005 * i;  // a non-parallel +7-12bp market move

  const cal::BundleProblem p0 = build_bundle_at(x0);
  const cal::BundleProblem p1 = build_bundle_at(x1);
  auto s0 = calibrated(p0);
  auto s1 = calibrated(p1);

  pf::MultiCurveBook book;
  book.positions.push_back(payer_swap(10.0, 0.030, 1e4));
  book.positions.push_back(payer_swap(5.0, 0.036, -2e4));

  const Eigen::VectorXd ladder = s0->price_portfolio_risk(book).ladder;
  const Eigen::VectorXd dq = s1->problem().market() - s0->problem().market();

  const cal::PnlExplain e =
      cal::pnl_explain(s0->problem(), book, s0->x(), s1->x(), /*dt=*/0.25, ladder, dq);

  // The residual is DEFINED in pnl_explain.hpp as total − carry − roll − market, so the sum identity cannot
  // fail (E5 2026-09-10: the old `carry+roll+market+residual == total` line was a tautology). What can fail
  // is the residual's MEANING -- the horizon market move minus its linear explanation -- so pin that against
  // the primitives composed OUTSIDE pnl_explain: residual == [NPV(x1, t1) − NPV(x0, t1)] − ladder·dq.
  const pf::MultiCurveBook rolled = cal::roll_book(book, 0.25, /*shift=*/true);
  const double npv1_t1 = cal::book_npv(s0->problem(), rolled, s1->x());
  const double npv0_t1 = cal::book_npv(s0->problem(), rolled, s0->x());
  EXPECT_NEAR(e.residual, (npv1_t1 - npv0_t1) - ladder.dot(dq), 1e-9);
  EXPECT_NEAR(e.total, npv1_t1 - cal::book_npv(s0->problem(), book, s0->x()), 1e-9);
  // per-instrument ladder sums to the aggregate market term.
  double lad_sum = 0.0;
  for (double v : e.market_ladder) lad_sum += v;
  EXPECT_NEAR(lad_sum, e.market, 1e-9);
  // both a time move and a market move contribute here.
  EXPECT_GT(std::abs(e.carry) + std::abs(e.roll), 0.0);
  EXPECT_GT(std::abs(e.market), 0.0);
}

// (b) A PURE market move (dt = 0): market == ladder·dq and explains ~all of total; carry/roll vanish.
TEST(PnlExplain, PureMarketMoveIsAllMarket) {
  const Eigen::VectorXd x0 = upward_curve();
  Eigen::VectorXd x1 = x0;
  for (int i = 0; i < kNk; ++i) x1[i] += 0.0005;  // +5bp parallel

  auto s0 = calibrated(build_bundle_at(x0));
  auto s1 = calibrated(build_bundle_at(x1));

  pf::MultiCurveBook book;
  book.positions.push_back(payer_swap(10.0, 0.030, 1e4));

  const Eigen::VectorXd ladder = s0->price_portfolio_risk(book).ladder;
  const Eigen::VectorXd dq = s1->problem().market() - s0->problem().market();

  const cal::PnlExplain e =
      cal::pnl_explain(s0->problem(), book, s0->x(), s1->x(), /*dt=*/0.0, ladder, dq);

  EXPECT_NEAR(e.carry, 0.0, 1e-12);
  EXPECT_NEAR(e.roll, 0.0, 1e-12);
  EXPECT_NEAR(e.market, ladder.dot(dq), 1e-12);
  // The linear market term explains the bulk of the total; the residual is the small 2nd-order piece.
  EXPECT_GT(std::abs(e.market), 0.0);
  EXPECT_LT(std::abs(e.residual), 0.01 * std::abs(e.total));
  EXPECT_NEAR(e.market, e.total, std::abs(e.total) * 0.01 + 1e-9);
}

// (c) A PURE time roll (q1 == q0, dt > 0): market == 0, P&L is in carry/roll, residual closes to 0.
TEST(PnlExplain, PureTimeRollHasNoMarket) {
  const Eigen::VectorXd x0 = upward_curve();

  auto s0 = calibrated(build_bundle_at(x0));
  auto s1 = calibrated(build_bundle_at(x0));  // identical market => x1 == x0, dq == 0

  pf::MultiCurveBook book;
  book.positions.push_back(payer_swap(10.0, 0.030, 1e4));

  const Eigen::VectorXd ladder = s0->price_portfolio_risk(book).ladder;
  const Eigen::VectorXd dq = s1->problem().market() - s0->problem().market();  // ~0
  EXPECT_NEAR(dq.norm(), 0.0, 1e-9);

  const cal::PnlExplain e =
      cal::pnl_explain(s0->problem(), book, s0->x(), s1->x(), /*dt=*/0.25, ladder, dq);

  EXPECT_NEAR(e.market, 0.0, 1e-12);
  EXPECT_NEAR(e.residual, 0.0, 1e-9);                 // x1 == x0 => the horizon reprices coincide
  EXPECT_NEAR(e.carry + e.roll, e.total, 1e-9);       // all P&L is time
  EXPECT_GT(std::abs(e.total), 1e-6);                 // and it is non-trivial
}

// (d) Signs: a payer swap gains when rates rise (market/total > 0), a receiver loses, and a positive-NPV
//     (in-the-money payer) book earns positive carry as time passes.
TEST(PnlExplain, EconomicSigns) {
  const Eigen::VectorXd x0 = upward_curve();
  Eigen::VectorXd x1 = x0;
  for (int i = 0; i < kNk; ++i) x1[i] += 0.0010;  // rates +10bp

  auto s0 = calibrated(build_bundle_at(x0));
  auto s1 = calibrated(build_bundle_at(x1));
  const Eigen::VectorXd dq = s1->problem().market() - s0->problem().market();

  // Payer swap struck at 3% on an upward curve (par ~3.4%): pays below-par fixed => in the money, NPV>0.
  pf::MultiCurveBook payer;
  payer.positions.push_back(payer_swap(10.0, 0.030, 1e4));
  const Eigen::VectorXd lad_p = s0->price_portfolio_risk(payer).ladder;
  const cal::PnlExplain ep =
      cal::pnl_explain(s0->problem(), payer, s0->x(), s1->x(), /*dt=*/0.0, lad_p, dq);
  EXPECT_GT(ep.npv_t0, 0.0);   // in-the-money payer
  EXPECT_GT(ep.market, 0.0);   // rates up => payer gains
  EXPECT_GT(ep.total, 0.0);

  // Receiver = the opposite position (negative notional payer): rates up => it loses.
  pf::MultiCurveBook receiver;
  receiver.positions.push_back(payer_swap(10.0, 0.030, -1e4));
  const Eigen::VectorXd lad_r = s0->price_portfolio_risk(receiver).ladder;
  const cal::PnlExplain er =
      cal::pnl_explain(s0->problem(), receiver, s0->x(), s1->x(), /*dt=*/0.0, lad_r, dq);
  EXPECT_LT(er.market, 0.0);
  EXPECT_LT(er.total, 0.0);

  // Carry: hold the in-the-money payer for dt with the market unchanged => positive financing carry.
  const Eigen::VectorXd dq0 = Eigen::VectorXd::Zero(dq.size());
  const cal::PnlExplain ec =
      cal::pnl_explain(s0->problem(), payer, s0->x(), s0->x(), /*dt=*/0.25, lad_p, dq0);
  EXPECT_GT(ec.carry, 0.0);          // positive MtM financed forward
  EXPECT_NEAR(ec.market, 0.0, 1e-12);
}
