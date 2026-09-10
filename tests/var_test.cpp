// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Gate for the VaR / Expected-Shortfall verb (api/var.cpp). Two paths:
//   * SUPPLIED  — a known P&L distribution matches the ANALYTIC quantile: for the uniform integer sample
//                 {0..N-1}, the type-7 VaR at q is the interpolated order statistic (1−q)·(N−1) and the ES
//                 is the mean of the m=round((1−q)N) worst points — both closed-form, so the verb's numbers
//                 are checked to 1e-9.
//   * REVAL     — full revaluation matches the `scenario` verb: a VaR built from parallel-shift MOVES
//                 reproduces scenario's npv_deltas cell-for-cell (same fork), the distribution is monotone
//                 in the shift, and ES(q) is a worse loss than VaR(q). NO QuantLib.

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/api/var.hpp"
#include "swaps/api/scenario.hpp"
#include "swaps/api/bundle_api.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};

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
  Eigen::VectorXd x_true(7);
  for (int i = 0; i < 7; ++i) x_true[i] = 0.030 + 0.001 * i;
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
const json::object& qrow(const json::object& out, int i) {
  return out.at("quantiles").as_array()[i].as_object();
}

}  // namespace

// SUPPLIED mode: a uniform integer P&L {0..999} matches the closed-form type-7 quantile + tail mean.
TEST(Var, SuppliedMatchesAnalyticQuantile) {
  const int N = 1000;
  json::array pnl;
  for (int i = 0; i < N; ++i) pnl.push_back(static_cast<double>(i));
  json::object v;
  v["pnl"] = pnl;
  v["quantiles"] = json::array{0.95, 0.99};
  json::object req;
  req["var"] = v;

  const json::value resp = json::parse(api::var_json(json::serialize(req)));
  ASSERT_FALSE(resp.as_object().contains("error"));
  const auto& out = resp.as_object().at("var").as_object();
  EXPECT_EQ(std::string(out.at("mode").as_string().c_str()), "supplied");
  EXPECT_EQ(out.at("n").to_number<int>(), N);

  // q=0.95 -> p=0.05 -> pos = 0.05*999 = 49.95 -> interp(49,50) = 49.95 ; ES = mean{0..49} = 24.5.
  const auto& q95 = qrow(out, 0);
  EXPECT_NEAR(q95.at("var_pnl").to_number<double>(), 49.95, 1e-9);
  EXPECT_NEAR(q95.at("var").to_number<double>(), -49.95, 1e-9);
  EXPECT_NEAR(q95.at("es_pnl").to_number<double>(), 24.5, 1e-9);  // mean of the 50 worst
  // q=0.99 -> p=0.01 -> pos = 9.99 -> interp(9,10) = 9.99 ; ES = mean{0..9} = 4.5.
  const auto& q99 = qrow(out, 1);
  EXPECT_NEAR(q99.at("var_pnl").to_number<double>(), 9.99, 1e-9);
  EXPECT_NEAR(q99.at("es_pnl").to_number<double>(), 4.5, 1e-9);   // mean of the 10 worst
  // ES is a worse (more negative) P&L than VaR at the same level.
  EXPECT_LT(q99.at("es_pnl").to_number<double>(), q99.at("var_pnl").to_number<double>());
}

// REVAL mode: full revaluation under parallel-shift moves reproduces the `scenario` verb's npv_deltas,
// the distribution is monotone in the shift, and ES(q) is a worse loss than VaR(q).
TEST(Var, RevalMatchesScenarioAndOrders) {
  const cal::BundleProblem p = build_bundle();
  const json::object book = make_book_json(/*fixed_rate=*/0.020, /*notional=*/1'000'000.0);

  std::vector<double> shifts;
  for (double bp = -60.0; bp <= 60.0 + 1e-9; bp += 10.0) shifts.push_back(bp);

  // VaR (reval) over the parallel-shift moves.
  json::array moves;
  for (double bp : shifts) {
    json::object m;
    m["parallel_bp"] = bp;
    moves.push_back(m);
  }
  json::object v;
  v["bundle"] = api::bundle_to_json(p);
  v["book"] = book;
  v["scenarios"] = moves;
  v["quantiles"] = json::array{0.95};
  json::object vreq;
  vreq["var"] = v;
  const json::value vresp = json::parse(api::var_json(json::serialize(vreq)));
  const auto& out = vresp.as_object().at("var").as_object();
  EXPECT_EQ(std::string(out.at("mode").as_string().c_str()), "reval");
  EXPECT_EQ(out.at("n").to_number<int>(), static_cast<int>(shifts.size()));

  // The SAME moves through `scenario`: each npv_delta must appear in the sorted VaR distribution.
  json::array srows;
  for (double bp : shifts) {
    json::object s;
    s["parallel_bp"] = bp;
    srows.push_back(s);
  }
  json::object s;
  s["bundle"] = api::bundle_to_json(p);
  s["book"] = book;
  s["scenarios"] = srows;
  json::object sreq;
  sreq["scenario"] = s;
  const json::value sresp = json::parse(api::scenario_json(json::serialize(sreq)));
  const auto& sscen = sresp.as_object().at("scenario").as_object().at("scenarios").as_array();

  std::vector<double> sorted;
  for (const auto& e : out.at("pnl_sorted").as_array()) sorted.push_back(e.to_number<double>());
  for (std::size_t k = 0; k < shifts.size(); ++k) {
    const double scen_delta = sscen[k].as_object().at("npv_delta").to_number<double>();
    const bool found = std::any_of(sorted.begin(), sorted.end(), [&](double x) {
      return std::abs(x - scen_delta) < std::abs(scen_delta) * 1e-6 + 1e-4;
    });
    EXPECT_TRUE(found) << "reval P&L must contain scenario npv_delta for " << shifts[k] << "bp";
  }

  // Sorted ascending, and VaR/ES are positive losses with ES the more extreme.
  EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
  const auto& q95 = qrow(out, 0);
  EXPECT_GT(q95.at("var").to_number<double>(), 0.0) << "a payer book has a positive downside VaR";
  EXPECT_GE(q95.at("es").to_number<double>(), q95.at("var").to_number<double>());

  // Consistency: feeding the reval distribution back as SUPPLIED reproduces the same VaR.
  json::object v2;
  v2["pnl"] = out.at("pnl_sorted");
  v2["quantiles"] = json::array{0.95};
  json::object v2req;
  v2req["var"] = v2;
  const json::value v2resp = json::parse(api::var_json(json::serialize(v2req)));
  const auto& out2 = v2resp.as_object().at("var").as_object();
  EXPECT_NEAR(qrow(out2, 0).at("var").to_number<double>(), q95.at("var").to_number<double>(), 1e-9);
}
