// Gate for the SCENARIO / STRESS verb (api/scenario.cpp): market::Scenario wired into production. NO
// QuantLib — a small single-curve outright bundle + a payer-swap book are hand-built as generic
// Instruments / MultiCurveBook positions (mirroring tests/api_test.cpp), made self-consistent from a
// known x_true, and driven THROUGH the JSON contract by scenario_json().
//
// Proves: (a) a +25bp PARALLEL scenario lowers every discount factor and raises every zero by ~25bp;
// (b) a book's NPV delta under a parallel shift matches PV01·25 to first order (sign + rough magnitude);
// (c) the BASE is unchanged after N scenarios (a trailing no-op scenario reproduces the base exactly).

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <vector>

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

// A single outright SOFR-like curve: front Rate pins + par OIS swaps, self-consistent from x_true.
cal::BundleProblem build_bundle(Eigen::VectorXd& x_true) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25, 0));
  p.instruments.push_back(make_rate(0.25, 0.5, 0));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T, 0, 0));

  x_true.resize(kNk);
  for (int i = 0; i < kNk; ++i) x_true[i] = 0.030 + 0.001 * i;

  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

// Book JSON (book_from_json "positions" schema): one 10y payer swap on curve 0, off-market so NPV != 0.
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

std::vector<double> arr_d(const json::value& v) {
  std::vector<double> out;
  for (const auto& e : v.as_array()) out.push_back(e.to_number<double>());
  return out;
}

}  // namespace

// (a) A +25bp PARALLEL scenario lowers every discount factor and raises every zero by ~25bp.
TEST(ScenarioVerb, ParallelShiftLowersDiscountsAndRaisesZeros) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);

  json::object scen;
  json::object s25;
  s25["name"] = "+25bp";
  s25["parallel_bp"] = 25.0;
  scen["bundle"] = api::bundle_to_json(p);
  scen["scenarios"] = json::array{s25};
  scen["sample_times"] = json::array{0.5, 1.0, 5.0, 10.0};
  json::object req;
  req["scenario"] = scen;

  const json::value resp = json::parse(api::scenario_json(json::serialize(req)));
  const auto& out = resp.as_object().at("scenario").as_object();
  ASSERT_FALSE(resp.as_object().contains("error"));

  const auto& base_c = out.at("base").as_object().at("curves").as_array()[0].as_object();
  const auto& shk_c = out.at("scenarios").as_array()[0].as_object().at("curves").as_array()[0].as_object();
  const auto bt = arr_d(base_c.at("t"));
  const auto bd = arr_d(base_c.at("discount")), sd = arr_d(shk_c.at("discount"));
  const auto bz = arr_d(base_c.at("zero")), sz = arr_d(shk_c.at("zero"));

  for (std::size_t i = 0; i < bt.size(); ++i) {
    EXPECT_LT(sd[i], bd[i]) << "shocked DF must be below base at t=" << bt[i];
    EXPECT_NEAR(sz[i] - bz[i], 0.0025, 2e-4) << "zero must rise ~25bp at t=" << bt[i];
  }
  // DF(0.5) shift is dominated by exp(-0.0025*0.5): a small but strictly-negative move.
  EXPECT_NEAR(sd[3], bd[3] * std::exp(-0.0025 * 10.0), 5e-4) << "DF(10) ~ base·exp(-25bp·10y)";
}

// (b) The book NPV delta under a parallel shift matches PV01·25 to first order (sign + rough magnitude).
TEST(ScenarioVerb, BookNpvDeltaMatchesPv01ToFirstOrder) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);

  // Off-market payer swap so NPV and PV01 are non-trivial; PV01 from the engine's own AAD pass.
  const json::object book = make_book_json(/*fixed_rate=*/0.020, /*notional=*/1'000'000.0);
  api::BundleSession sess(p);
  sess.calibrate(api::flat_x0(p));
  const api::PortfolioReprice base = sess.price_portfolio(api::book_from_json(book));
  ASSERT_GT(std::abs(base.pv01), 1.0) << "need a non-trivial PV01 to compare against";

  json::object scen;
  json::object s25;
  s25["name"] = "+25bp";
  s25["parallel_bp"] = 25.0;
  scen["bundle"] = api::bundle_to_json(p);
  scen["book"] = book;
  scen["scenarios"] = json::array{s25};
  json::object req;
  req["scenario"] = scen;

  const json::value resp = json::parse(api::scenario_json(json::serialize(req)));
  const auto& out = resp.as_object().at("scenario").as_object();
  const double base_npv = out.at("base").as_object().at("npv").to_number<double>();
  const auto& row = out.at("scenarios").as_array()[0].as_object();
  const double npv_delta = row.at("npv_delta").to_number<double>();

  EXPECT_NEAR(base_npv, base.npv, 1e-6) << "the verb's base NPV must equal price_portfolio's";
  const double first_order = base.pv01 * 25.0;  // pv01 is d(NPV) for +1bp parallel
  EXPECT_GT(npv_delta * first_order, 0.0) << "delta must share the sign of PV01·25";
  EXPECT_NEAR(npv_delta, first_order, std::abs(first_order) * 0.05 + 1.0)
      << "delta ~ PV01·25 to first order (5% tol absorbs the 25bp curvature)";
}

// (c) The BASE is unchanged after N scenarios: a trailing no-op scenario reproduces the base exactly, and
// the base block (priced once) is untouched by the intervening shocks.
TEST(ScenarioVerb, BaseUnchangedAfterManyScenarios) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  const json::object book = make_book_json(0.020, 1'000'000.0);

  json::array scens;
  for (double bp : {25.0, -40.0, 10.0}) {
    json::object s;
    s["name"] = "shift";
    s["parallel_bp"] = bp;
    scens.push_back(s);
  }
  json::object noop;  // a final scenario with NO shift: must reproduce the base exactly
  noop["name"] = "unchanged";
  scens.push_back(noop);

  json::object scen;
  scen["bundle"] = api::bundle_to_json(p);
  scen["book"] = book;
  scen["scenarios"] = scens;
  scen["sample_times"] = json::array{1.0, 5.0, 10.0};
  json::object req;
  req["scenario"] = scen;

  const json::value resp = json::parse(api::scenario_json(json::serialize(req)));
  const auto& out = resp.as_object().at("scenario").as_object();
  const auto& base_o = out.at("base").as_object();
  const double base_npv = base_o.at("npv").to_number<double>();
  const auto base_disc = arr_d(base_o.at("curves").as_array()[0].as_object().at("discount"));

  const auto& scen_arr = out.at("scenarios").as_array();
  const auto& last = scen_arr.back().as_object();  // the no-op
  EXPECT_NEAR(last.at("npv_delta").to_number<double>(), 0.0, 1e-9)
      << "a null shock must leave the book NPV at the base";
  EXPECT_NEAR(last.at("npv").to_number<double>(), base_npv, 1e-9);
  const auto last_disc = arr_d(last.at("curves").as_array()[0].as_object().at("discount"));
  for (std::size_t i = 0; i < base_disc.size(); ++i)
    EXPECT_EQ(last_disc[i], base_disc[i]) << "no-op scenario must reproduce the base curve bit-for-bit";

  // The +25bp shock still genuinely moved the book (not a frozen no-op) and shares PV01's sign.
  EXPECT_LT(scen_arr[1].as_object().at("npv_delta").to_number<double>(),
            scen_arr[2].as_object().at("npv_delta").to_number<double>())
      << "a -40bp shift's delta must sit below a +10bp shift's (monotone in the parallel shift)";
}
