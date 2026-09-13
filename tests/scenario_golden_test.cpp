// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// E7 stage 5.0 PINS, written BEFORE the scenario / scenario_grid lift (P11): each verb's WHOLE response on a rich
// request -- a 3-curve bundle (domestic discount, domestic forecast, foreign) with a turn, a book of two swaps and an
// xccy position, parallel / per-curve / override / FX / no-op scenarios, a parallel x fx grid and a shift_curve +
// parallel grid -- compared BYTE FOR BYTE with tests/golden/scenario/*.json, recorded from the verbs before the lift.
// The lift must keep every number bitwise. An owner-gated behaviour change (SC1 override-vs-add, SC2 multi-pair FX)
// re-records deliberately and says so in its commit:
//   SWAPS_WRITE_GOLDEN=1 ./build/api/swaps_api_tests --gtest_filter='ScenarioGolden.*'
// grid_us (wall-clock) is the one field cut before comparing, by string surgery: a parse/serialize round trip is not
// bit-exact (ASSUMPTIONS E1). Every shock size is one where bp/1e4 != bp*1e-4 in double: `scenario` samples the
// shocked curves, so its golden sees that one-ULP change (shown by hand, 2026-09-14); the grid emits only NPVs, where
// it can vanish, and its golden was shown catching the shift_curve axis turned from add to override. Beside each
// golden, a few properties show the request exercises what it claims.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <boost/json.hpp>
#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/scenario.hpp"
#include "swaps/api/scenario_grid.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace json = boost::json;

namespace {

const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
const std::vector<double> kSwapT{1.0, 2.0, 4.0, 7.0, 10.0};

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
cal::Instrument rate(double a, double b, int fc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = fc;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}
cal::Instrument par_swap(double T, int fc, int dc) {
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

// curve 0 domestic discount (with a turn over [0.10, 0.20]); curve 1 domestic forecast discounted on 0; curve 2
// foreign (currency 1). Quotes are made self-consistent at a known state; the verbs calibrate from the flat seed.
cal::BundleProblem bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.curves[0].turns = {px::Turn{0.10, 0.20}};
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.curves.push_back({.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  const int d = 0, f = 1, x = 2;
  for (int c : {d, f, x}) {
    p.instruments.push_back(rate(0.0, 0.25, c));
    p.instruments.push_back(rate(0.25, 0.5, c));
    for (double T : kSwapT) p.instruments.push_back(par_swap(T, c, c == f ? d : c));
  }
  p.instruments.push_back(rate(0.10, 0.20, d));  // pins the turn

  Eigen::VectorXd state = Eigen::VectorXd::Zero(p.n_knots());
  const double level[] = {0.030, 0.033, 0.020}, slope[] = {0.0010, 0.0012, 0.0008};
  for (int c = 0; c < p.n_curves(); ++c)
    for (int i = 0; i < p.curves[c].n_interp_knots(); ++i) state[p.offset(c) + i] = level[c] + slope[c] * i;
  state[p.offset(0) + p.curves[0].n_interp_knots()] = 0.002;  // the turn's jump
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return state[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

json::array annual_float(double T, double spread) {
  json::array out;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    out.push_back(json::object{
        {"obs", json::object{{"sub_start", json::array{prev}}, {"sub_end", json::array{t}}, {"tau_index", t - prev}}},
        {"pay", t},
        {"tau_pay", t - prev},
        {"spread", spread}});
    prev = t;
  }
  return out;
}
json::array annual_fixed(double T) {
  json::array out;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    out.push_back(json::object{{"pay", t}, {"tau", t - prev}});
    prev = t;
  }
  return out;
}
json::object swap(double T, int fc, int dc, double fixed_rate, double notional) {
  return json::object{{"kind", "swap"},          {"notional", notional},   {"fixed_rate", fixed_rate},
                      {"fwd_curve", fc},         {"disc_curve", dc},       {"fixed_curve", dc},
                      {"float_coupons", annual_float(T, 0.0)}, {"fixed_coupons", annual_fixed(T)}};
}
json::object xccy() {
  return json::object{{"kind", "xccy"},       {"notional", 3e7},     {"fwd_curve", 0},
                      {"disc_curve", 0},      {"float_coupons", annual_float(5.0, 0.0)},
                      {"mtm_coupons", annual_float(5.0, 0.005)},    {"mtm_fwd_curve", 2},
                      {"mtm_disc_curve", 2},  {"mtm_reset_num", 2},  {"mtm_reset_den", 0},
                      {"fx_spot", 1.10}};
}

json::object base_request() {
  return json::object{
      {"bundle", api::bundle_to_json(bundle())},
      {"book", json::object{{"positions", json::array{swap(10.0, 1, 0, 0.030, 1e7), swap(5.0, 0, 0, 0.028, -2e7), xccy()}}}},
      {"sample_times", json::array{0.05, 0.15, 0.5, 2.0, 9.0}}};
}

// Cut `"grid_us":<number>` and one adjoining comma out of a serialized response.
std::string without_timing(std::string s) {
  const std::string key = "\"grid_us\":";
  const std::size_t at = s.find(key);
  if (at == std::string::npos) return s;
  const std::size_t end = s.find_first_of(",}", at);
  if (s[end] == ',')
    s.erase(at, end + 1 - at);
  else {
    s.erase(at, end - at);
    if (s[at - 1] == ',') s.erase(at - 1, 1);
  }
  return s;
}

void expect_golden(const std::string& name, const std::string& response) {
  const std::string got = without_timing(response);
  const std::string path = std::string(SWAPS_GOLDEN_DIR) + "/scenario/" + name + ".json";
  if (std::getenv("SWAPS_WRITE_GOLDEN")) {
    std::ofstream(path) << got << "\n";
    return;
  }
  std::ifstream in(path);
  ASSERT_TRUE(in.good()) << "missing golden " << path << " (record with SWAPS_WRITE_GOLDEN=1)";
  std::stringstream ss;
  ss << in.rdbuf();
  std::string want = ss.str();
  if (!want.empty() && want.back() == '\n') want.pop_back();
  if (got == want) return;
  std::size_t k = 0;
  while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
  const std::size_t from = k < 80 ? 0 : k - 80;
  ADD_FAILURE() << name << ": first difference at byte " << k << "\n  got:  ..." << got.substr(from, 160)
                << "\n  want: ..." << want.substr(from, 160);
}

double at(const json::value& v, std::initializer_list<const char*> keys, int c, int k) {
  const json::value* node = &v;
  for (const char* key : keys) node = &node->as_object().at(key);
  return node->as_array()[c].as_object().at("zero").as_array()[k].to_number<double>();
}

}  // namespace

TEST(ScenarioGolden, ScenarioResponseIsBitwiseAndItsShocksAreTheOnesClaimed) {
  json::object req = base_request();
  req["scenarios"] = json::array{
      json::object{{"name", "parallel"}, {"parallel_bp", 37.5}},
      json::object{{"name", "override"}, {"parallel_bp", 34.0}, {"shift_curve", json::object{{"0", 14.5}}}},
      json::object{{"name", "foreign"}, {"shift_curve", json::object{{"2", -33.25}}}},
      json::object{{"name", "fx"},
                   {"bump_fx", json::array{json::object{{"base", "EUR"}, {"quote", "USD"}, {"rel", 0.02}},
                                           json::object{{"base", "GBP"}, {"quote", "USD"}, {"rel", -0.01}}}}},
      json::object{{"name", "noop"}}};
  const std::string response = api::scenario_json(json::object{{"scenario", req}});
  expect_golden("scenario", response);

  const json::value v = json::parse(response);
  const json::value& out = v.as_object().at("scenario");
  const json::array& rows = out.as_object().at("scenarios").as_array();
  ASSERT_EQ(rows.size(), 5u);
  const auto zero_move = [&](int row, int c) {
    const json::value& r = rows[static_cast<std::size_t>(row)];
    return at(r, {"curves"}, c, 4) - at(out, {"base", "curves"}, c, 4);  // at t = 9
  };
  EXPECT_NEAR(zero_move(1, 0), 14.5e-4, 1e-12) << "override: an explicit shift_curve replaces the parallel (SC1 today)";
  EXPECT_NEAR(zero_move(1, 1), 34e-4, 1e-12) << "override: an unkeyed curve takes the parallel";
  EXPECT_NEAR(zero_move(2, 2), -33.25e-4, 1e-12);
  EXPECT_EQ(zero_move(2, 0), 0.0);
  const double npv_fx = rows[3].as_object().at("npv").to_number<double>();
  const double npv_noop = rows[4].as_object().at("npv").to_number<double>();
  EXPECT_NE(npv_fx, npv_noop) << "the FX bumps reach the xccy position";
  EXPECT_EQ(npv_noop, out.as_object().at("base").as_object().at("npv").to_number<double>());
}

TEST(ScenarioGolden, GridResponsesAreBitwise) {
  json::object parallel_by_fx = base_request();
  parallel_by_fx["axes"] = json::array{
      json::object{{"label", "parallel"}, {"kind", "parallel_bp"}, {"values", json::array{-36.0, 0.0, 34.5}}},
      json::object{{"label", "eurusd"}, {"kind", "fx"}, {"base", "EUR"}, {"quote", "USD"},
                   {"values", json::array{-0.05, 0.0, 0.05}}}};
  const std::string a = api::scenario_grid_json(json::object{{"scenario_grid", parallel_by_fx}});
  expect_golden("grid_parallel_fx", a);

  json::object curve_plus_parallel = base_request();
  curve_plus_parallel["axes"] = json::array{  // parallel FIRST: the curve axis then adds onto a non-zero shift (SC1)
      json::object{{"label", "parallel"}, {"kind", "parallel_bp"}, {"values", json::array{42.0}}},
      json::object{{"label", "forecast"}, {"kind", "shift_curve"}, {"role", 1}, {"values", json::array{-31.75, 36.75}}}};
  const std::string b = api::scenario_grid_json(json::object{{"scenario_grid", curve_plus_parallel}});
  expect_golden("grid_curve_plus_parallel", b);

  const json::object g = json::parse(a).as_object().at("scenario_grid").as_object();
  const auto pnl = [&g](int i, int j) {
    return g.at("pnl").as_array()[static_cast<std::size_t>(i)].as_array()[static_cast<std::size_t>(j)].to_number<double>();
  };
  EXPECT_EQ(pnl(1, 1), 0.0) << "the zero cell is the base";
  EXPECT_NE(pnl(1, 0), pnl(1, 2)) << "the fx axis reaches the xccy position";
  EXPECT_NE(pnl(0, 1), pnl(2, 1));
}
