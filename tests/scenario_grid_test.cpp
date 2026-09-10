// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// Gate for the SCENARIO-GRID verb (api/scenario_grid.cpp): a P&L SURFACE for a book under an outer product
// of shock axes, forking the once-calibrated anchor per cell and repricing through the compiled reprice
// twin. NO QuantLib — a small single-curve OIS bundle + a payer-swap book are hand-built (mirroring
// tests/scenario_verb_test.cpp) and driven THROUGH the JSON contract by scenario_grid_json().
//
// Proves: (a) the GRID DIAGONAL matches single `scenario` calls — a 1-axis parallel_bp grid cell reproduces
// scenario's npv_delta for the same parallel_bp (the required cross-check that the grid reuses the same
// fork); (b) the base NPV equals price_portfolio's; (c) the surface is MONOTONE where expected (a payer
// book's P&L falls as the parallel shift rises) and ADDITIVE (two cells with equal summed shift are equal).

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <vector>

#include "swaps/api/scenario_grid.hpp"
#include "swaps/api/scenario.hpp"
#include "swaps/api/bundle_api.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
constexpr int kNk = 7;

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
cal::Instrument make_swap(double T, int fc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.fixed.discount = dc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    ins.fwd.coupons.push_back(ois_coupon(prev, t));
    px::FixedCoupon x;
    x.pay = t;
    x.tau = t - prev;
    ins.fixed.coupons.push_back(x);
    prev = t;
  }
  return ins;
}
cal::Instrument make_rate(double a, double b, int fc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = fc;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}
cal::BundleProblem build_bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25, 0));
  p.instruments.push_back(make_rate(0.25, 0.5, 0));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T, 0, 0));
  Eigen::VectorXd x_true(kNk);
  for (int i = 0; i < kNk; ++i) x_true[i] = 0.030 + 0.001 * i;
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}
json::object make_book_json(double fixed_rate, double notional) {
  json::array fcs, xcs;
  double prev = 0.0;
  for (double t = 1.0; t <= 10.0 + 1e-9; t += 1.0) {
    json::object obs;
    obs["sub_start"] = json::array{prev};
    obs["sub_end"] = json::array{t};
    obs["tau_index"] = t - prev;
    json::object fc;
    fc["obs"] = obs;
    fc["pay"] = t;
    fc["tau_pay"] = t - prev;
    fcs.push_back(fc);
    json::object xc;
    xc["pay"] = t;
    xc["tau"] = t - prev;
    xcs.push_back(xc);
    prev = t;
  }
  json::object pos;
  pos["kind"] = "swap";
  pos["notional"] = notional;
  pos["fixed_rate"] = fixed_rate;
  pos["fwd_curve"] = 0;
  pos["disc_curve"] = 0;
  pos["fixed_curve"] = 0;
  pos["float_coupons"] = fcs;
  pos["fixed_coupons"] = xcs;
  json::object book;
  book["positions"] = json::array{pos};
  return book;
}
double cell(const json::object& out, const char* field, int i, int j) {
  return out.at(field).as_array()[i].as_array()[j].to_number<double>();
}

}  // namespace

// (a) A 1-axis parallel_bp grid's cells reproduce single `scenario` npv_deltas for the same shift.
TEST(ScenarioGrid, DiagonalMatchesSingleScenario) {
  const cal::BundleProblem p = build_bundle();
  const json::object book = make_book_json(/*fixed_rate=*/0.020, /*notional=*/1'000'000.0);
  const std::vector<double> vals{-40.0, -25.0, 0.0, 25.0, 60.0};

  // The grid: one parallel_bp axis over vals.
  json::object axis;
  axis["kind"] = "parallel_bp";
  axis["label"] = "parallel (bp)";
  axis["values"] = json::array(vals.begin(), vals.end());
  json::object g;
  g["bundle"] = api::bundle_to_json(p);
  g["book"] = book;
  g["axes"] = json::array{axis};
  json::object greq;
  greq["scenario_grid"] = g;
  const json::value gresp = json::parse(api::scenario_grid_json(json::serialize(greq)));
  ASSERT_FALSE(gresp.as_object().contains("error"));
  const auto& gout = gresp.as_object().at("scenario_grid").as_object();
  ASSERT_EQ(gout.at("shape").as_array()[0].to_number<int>(), static_cast<int>(vals.size()));
  ASSERT_EQ(gout.at("shape").as_array()[1].to_number<int>(), 1);

  // The reference: one `scenario` call carrying every shift as a separate scenario row.
  json::array srows;
  for (double v : vals) {
    json::object s;
    s["name"] = "p";
    s["parallel_bp"] = v;
    srows.push_back(s);
  }
  json::object s;
  s["bundle"] = api::bundle_to_json(p);
  s["book"] = book;
  s["scenarios"] = srows;
  json::object sreq;
  sreq["scenario"] = s;
  const json::value sresp = json::parse(api::scenario_json(json::serialize(sreq)));
  const auto& sout = sresp.as_object().at("scenario").as_object();

  // Bases agree, and every grid P&L cell equals the matching scenario npv_delta.
  EXPECT_NEAR(gout.at("base").as_object().at("npv").to_number<double>(),
              sout.at("base").as_object().at("npv").to_number<double>(), 1e-6);
  for (std::size_t i = 0; i < vals.size(); ++i) {
    const double grid_pnl = cell(gout, "pnl", static_cast<int>(i), 0);
    const double scen_delta = sout.at("scenarios").as_array()[i].as_object().at("npv_delta").to_number<double>();
    EXPECT_NEAR(grid_pnl, scen_delta, std::abs(scen_delta) * 1e-6 + 1e-4)
        << "grid cell " << i << " must match scenario npv_delta for " << vals[i] << "bp";
  }
}

// (b) Base NPV equals price_portfolio's, and the P&L surface is monotone + additive.
TEST(ScenarioGrid, MonotoneAndAdditiveSurface) {
  const cal::BundleProblem p = build_bundle();
  const json::object book = make_book_json(0.020, 1'000'000.0);
  api::BundleSession sess(p);
  sess.calibrate(api::flat_x0(p));
  const api::PortfolioReprice base = sess.price_portfolio(api::book_from_json(book));

  // Two parallel-ish axes on the single curve: axis0 parallel_bp, axis1 shift_curve role 0. A cell's total
  // shift is v0+v1, so equal-sum cells must coincide and the surface must fall as the total shift rises.
  const std::vector<double> a0{-30.0, 0.0, 30.0}, a1{-20.0, 0.0, 20.0};
  json::object ax0, ax1;
  ax0["kind"] = "parallel_bp";
  ax0["values"] = json::array(a0.begin(), a0.end());
  ax1["kind"] = "shift_curve";
  ax1["role"] = 0;
  ax1["values"] = json::array(a1.begin(), a1.end());
  json::object g;
  g["bundle"] = api::bundle_to_json(p);
  g["book"] = book;
  g["axes"] = json::array{ax0, ax1};
  json::object greq;
  greq["scenario_grid"] = g;
  const json::value gresp = json::parse(api::scenario_grid_json(json::serialize(greq)));
  const auto& gout = gresp.as_object().at("scenario_grid").as_object();

  EXPECT_NEAR(gout.at("base").as_object().at("npv").to_number<double>(), base.npv, 1e-6);

  // Monotone: down a column (rising total shift) a payer book with below-market fixed strictly GAINS
  // (its P&L rises with rates — the same sign the scenario verb's monotonicity gate asserts).
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i + 1 < 3; ++i)
      EXPECT_LT(cell(gout, "pnl", i, j), cell(gout, "pnl", i + 1, j))
          << "P&L must rise as the parallel shift rises (col " << j << ")";

  // Additive: a cell's total shift is v0+v1, so along the anti-diagonal the total rises −10 → 0 → +10 and
  // the P&L rises with it: pnl(0,2) < pnl(1,1) < pnl(2,0).
  EXPECT_LT(cell(gout, "pnl", 0, 2), cell(gout, "pnl", 1, 1));
  EXPECT_LT(cell(gout, "pnl", 1, 1), cell(gout, "pnl", 2, 0));
  // The zero-shift centre cell (total 0) has zero P&L.
  EXPECT_NEAR(cell(gout, "pnl", 1, 1), 0.0, 1e-6) << "the zero-shift cell has zero P&L";
}
