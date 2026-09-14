// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTIONS (2026-09-14, found while drafting the portfolio / scenario oracles): a "parallel" move counted a
// SPREAD curve twice and moved TURN jumps. A spread curve's forward is its base's plus its own (SpreadHandle), and a
// turn's δ is a localized jump over its window, not a level (TurnedCurve). So moving EVERY state entry by 1 bp moves a
// spread curve by 2 bp and adds a turn-window jump:
//   * portfolio PV01 = 1e-4 · Σ_j ∂NPV/∂x_j over every state entry (BundleSession::price_portfolio, seed_directional);
//   * var's reval parallel_bp added to every curve's interpolation knots (api/var.cpp parse_move);
// and in the library, tests/scenario_spread_repro_test.cpp (scenario / scenario_grid). The reference here is the
// DEFINITION: +1 bp parallel = every curve's forward +1 bp, which is +1 bp on the OUTRIGHT curves' interpolation
// forwards only (a spread curve inherits it from its base; a turn δ is untouched).
#include <cmath>
#include <string>

#include <Eigen/Core>
#include <boost/json.hpp>
#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/var.hpp"
#include "swaps/calibration/bundle_state.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;
namespace json = boost::json;

namespace {

cal::Instrument rate(double a, double b, int fc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = fc;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// Curve 0 outright (optionally with a turn over [0.4, 0.6]), curve 1 a spread over it; Hermite back knots so the book
// has real curve shape. Quotes are model quotes at `x_true`, so calibrating from x_true is exact.
struct Bundle {
  cal::BundleProblem p;
  Eigen::VectorXd x_true;
};
Bundle outright_and_spread(bool with_turn) {
  Bundle b;
  const std::vector<double> back{1.0, 3.0, 7.0};
  cal::BundleCurveSpec outright;
  outright.regions = crv::flat_hermite({}, back);
  if (with_turn) outright.turns = {px::Turn{0.4, 0.6}};
  cal::BundleCurveSpec spread;
  spread.base = 0;
  spread.regions = crv::flat_hermite({}, back);
  b.p.curves = {outright, spread};
  for (int c : {0, 1})
    for (auto [a, e] : {std::pair{0.0, 1.0}, std::pair{1.0, 3.0}, std::pair{3.0, 7.0}}) b.p.instruments.push_back(rate(a, e, c));
  if (with_turn) b.p.instruments.push_back(rate(0.3, 0.7, 0));
  b.x_true = Eigen::VectorXd::Zero(b.p.n_knots());
  b.x_true.segment(b.p.offset(0), 3) << 0.030, 0.032, 0.035;
  if (with_turn) b.x_true[b.p.offset(0) + 3] = 0.004;
  b.x_true.segment(b.p.offset(1), 3) << 0.004, 0.005, 0.006;
  const auto C = cal::build_bundle_curves<double>(b.p.curves, [&](int c, int i) { return b.x_true[b.p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[static_cast<std::size_t>(i)]; };
  for (auto& ins : b.p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return b;
}

// A 6y payer swap forecasting on the SPREAD curve, discounted on the outright curve.
json::object book_json() {
  json::array fl, fx;
  for (int k = 0; k < 6; ++k) {
    const double a = k, e = k + 1.0;
    fl.push_back(json::object{{"obs", json::object{{"sub_start", json::array{a}}, {"sub_end", json::array{e}}, {"tau_index", 1.0}}},
                              {"pay", e},
                              {"tau_pay", 1.0}});
    fx.push_back(json::object{{"pay", e}, {"tau", 1.0}});
  }
  json::object pos{{"kind", "swap"},   {"notional", 1e6},   {"fixed_rate", 0.03},       {"fwd_curve", 1},
                   {"disc_curve", 0},  {"fixed_curve", 0},  {"float_coupons", fl},       {"fixed_coupons", fx}};
  return json::object{{"positions", json::array{pos}}};
}

// The state with `bp` added to the OUTRIGHT curve's interpolation forwards only: every curve's forward moves by bp.
Eigen::VectorXd parallel(const cal::BundleProblem& p, Eigen::VectorXd x, double bp) {
  x.segment(p.offset(0), p.curves[0].n_interp_knots()).array() += bp / 1e4;
  return x;
}

}  // namespace

TEST(SpreadPv01Repro, PortfolioPv01IsTheBooksMoveForOneBasisPointOnEveryCurveOnce) {
  const Bundle b = outright_and_spread(/*with_turn=*/false);
  api::BundleSession sess(b.p);
  sess.calibrate(b.x_true);
  const pf::MultiCurveBook book = api::book_from_json(book_json());
  const double pv01 = sess.price_portfolio(book).pv01;
  const double fd = (cal::book_value_at(book, b.p, parallel(b.p, sess.x(), 1.0)) -
                     cal::book_value_at(book, b.p, parallel(b.p, sess.x(), -1.0))) / 2.0;
  EXPECT_NEAR(pv01, fd, 1e-6 * std::abs(fd)) << "PV01 counted the spread curve's forward twice";
}

TEST(SpreadPv01Repro, PortfolioPv01DoesNotMoveATurnJump) {
  const Bundle b = outright_and_spread(/*with_turn=*/true);
  api::BundleSession sess(b.p);
  sess.calibrate(b.x_true);
  const pf::MultiCurveBook book = api::book_from_json(book_json());
  const double pv01 = sess.price_portfolio(book).pv01;
  const double fd = (cal::book_value_at(book, b.p, parallel(b.p, sess.x(), 1.0)) -
                     cal::book_value_at(book, b.p, parallel(b.p, sess.x(), -1.0))) / 2.0;
  EXPECT_NEAR(pv01, fd, 1e-6 * std::abs(fd)) << "PV01 moved the turn jump and counted the spread curve twice";
}

TEST(SpreadVarRepro, AParallelVarMoveMovesEveryCurveOnce) {
  const Bundle b = outright_and_spread(/*with_turn=*/false);
  json::array x0;
  for (Eigen::Index i = 0; i < b.x_true.size(); ++i) x0.push_back(b.x_true[i]);
  const json::object req{{"var", json::object{{"bundle", api::bundle_to_json(b.p)},
                                              {"x0", x0},
                                              {"book", book_json()},
                                              {"scenarios", json::array{json::object{{"parallel_bp", 25.0}}}}}}};
  const json::object out = json::parse(api::var_json(req)).as_object().at("var").as_object();
  const double pnl = out.at("pnl_sorted").as_array().at(0).to_number<double>();

  api::BundleSession sess(b.p);
  sess.calibrate(b.x_true);
  const pf::MultiCurveBook book = api::book_from_json(book_json());
  const double want = cal::book_value_at(book, b.p, parallel(b.p, sess.x(), 25.0)) - cal::book_value_at(book, b.p, sess.x());
  EXPECT_NEAR(pnl, want, 1e-6 * std::abs(want)) << "var's parallel move moved the spread curve twice";
}
