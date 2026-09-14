// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// E7 stage 5.0 PINS, written BEFORE the scenario / scenario_grid lift (P11): each verb's WHOLE response on a rich
// request -- a 3-curve bundle (domestic discount, domestic forecast, foreign) with a turn, a book of two swaps and an
// xccy position, parallel / per-curve / override / FX / no-op scenarios, a parallel x fx grid and a shift_curve +
// parallel grid, and `var`'s full-revaluation P&L over the same kinds of move -- compared BYTE FOR BYTE with tests/golden/scenario/*.json, recorded from the verbs before the lift.
// Before the var lift (E7, 2026-09-14) two more: var's SUPPLIED reduction and a seeded, smoothed reval, recorded from
// api/var.cpp as it stood -- a labelled regression freeze of the pieces var_reval.json does not reach.
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
#include "swaps/api/var.hpp"

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
  p.currency_codes = {"USD", "EUR"};  // tag 0 (curves 0, 1) USD, tag 1 (curve 2) EUR: the xccy position is EURUSD
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

// Cut each wall-clock field (`"grid_us":<number>`, `"reval_us":<number>`) and one adjoining comma out of a response.
std::string without_timing(std::string s) {
  for (const std::string key : {"\"grid_us\":", "\"reval_us\":"}) {
    const std::size_t at = s.find(key);
    if (at == std::string::npos) continue;
    const std::size_t end = s.find_first_of(",}", at);
    if (s[end] == ',')
      s.erase(at, end + 1 - at);
    else {
      s.erase(at, end - at);
      if (s[at - 1] == ',') s.erase(at - 1, 1);
    }
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
      json::object{{"name", "keyed_plus_parallel"}, {"parallel_bp", 34.0}, {"shift_curve", json::object{{"0", 14.5}}}},
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
  EXPECT_NEAR(zero_move(1, 0), 48.5e-4, 1e-12) << "an explicit shift_curve ADDS onto the parallel (SC1, owner 2026-09-14)";
  EXPECT_NEAR(zero_move(1, 1), 34e-4, 1e-12) << "an unkeyed curve takes the parallel alone";
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

// var (deferred, but it shares the fork and the compiled-book cache the stage-5 lift moves): the reval P&L
// distribution over moves of every kind, including a parallel with an explicit curve key (every verb ADDS them: SC1).
TEST(ScenarioGolden, VarRevaluationResponseIsBitwise) {
  json::object req = base_request();
  req.erase("sample_times");
  req["quantiles"] = json::array{0.9, 0.975};
  req["scenarios"] = json::array{
      json::object{{"parallel_bp", 37.5}},
      json::object{{"parallel_bp", -36.0}},
      json::object{{"parallel_bp", 34.0}, {"shift_curve", json::object{{"0", 14.5}}}},
      json::object{{"shift_curve", json::object{{"2", -33.25}}}},
      json::object{{"bump_fx", json::array{json::object{{"base", "EUR"}, {"quote", "USD"}, {"rel", 0.02}},
                                           json::object{{"base", "GBP"}, {"quote", "USD"}, {"rel", -0.01}}}}},
      json::object{{"parallel_bp", -31.75}, {"bump_fx", json::array{json::object{{"base", "EUR"}, {"quote", "USD"}, {"rel", -0.05}}}}},
      json::object{{"parallel_bp", 34.5}},
      json::object{}};
  const std::string response = api::var_json(json::object{{"var", req}});
  expect_golden("var_reval", response);

  const json::object out = json::parse(response).as_object().at("var").as_object();
  EXPECT_EQ(out.at("mode").as_string(), "reval");
  EXPECT_EQ(out.at("n").as_int64(), 8);
  const json::array& sorted = out.at("pnl_sorted").as_array();
  int zeros = 0;
  for (const auto& v : sorted) zeros += v.to_number<double>() == 0.0;
  EXPECT_EQ(zeros, 1) << "exactly the no-op move reprices to the base";
}

// SUPPLIED: every branch of the reduction -- interpolated position, m clamped to 1 (q = 0.99 on 7 points), p == 1 on the
// last point (q = 1e-17), duplicates -- and the key order of a supplied response. The bare body (no "var" envelope)
// answers with the same bytes.
TEST(ScenarioGolden, VarSuppliedResponseIsBitwise) {
  json::object req;
  req["pnl"] = json::array{2.5, -3.25, 5.0, -1.0, 7.75, -3.25, -12.5};
  req["quantiles"] = json::array{0.5, 0.9, 0.99, 1e-17};
  const std::string response = api::var_json(json::object{{"var", req}});
  expect_golden("var_supplied", response);
  EXPECT_EQ(api::var_json(req), response) << "the bare body is the same request";

  const json::object out = json::parse(response).as_object().at("var").as_object();
  EXPECT_EQ(out.at("mode").as_string(), "supplied");
  const json::array& q = out.at("quantiles").as_array();
  EXPECT_EQ(q[2].as_object().at("es_pnl").to_number<double>(), -12.5) << "q = 0.99 on 7 points: the single worst";
  EXPECT_EQ(q[3].as_object().at("var_pnl").to_number<double>(), 7.75) << "q = 1e-17: the last point";
}

// REVAL with an explicit seed, a regulariser and no quantiles (the [0.95, 0.99] default): the pieces the lift re-plumbs
// (seed_or_flat, RegSpec, the default) that var_reval.json does not exercise.
TEST(ScenarioGolden, VarRevalSeededAndSmoothedResponseIsBitwise) {
  json::object req = base_request();
  req.erase("sample_times");
  json::array x0;
  const cal::BundleProblem p = bundle();
  for (int i = 0; i < p.n_knots(); ++i) x0.push_back(0.025 + 0.0005 * i);  // not the flat seed
  req["x0"] = std::move(x0);
  json::array reg_curves;
  for (int c = 0; c < p.n_curves(); ++c) reg_curves.push_back(c);
  req["regularize"] = json::object{{"lambda", 1e-4}, {"tension", true}, {"curves", reg_curves}};  // on(): lambda AND curves
  req["scenarios"] = json::array{
      json::object{{"parallel_bp", 37.5}},
      json::object{{"parallel_bp", 34.0}, {"shift_curve", json::object{{"1", -31.75}}}},
      json::object{{"bump_fx", json::array{json::object{{"base", "EUR"}, {"quote", "USD"}, {"rel", 0.02}}}}},
      json::object{}};
  const std::string response = api::var_json(json::object{{"var", req}});
  expect_golden("var_reval_seeded", response);

  const json::object out = json::parse(response).as_object().at("var").as_object();
  const json::array& q = out.at("quantiles").as_array();
  ASSERT_EQ(q.size(), 2u);
  EXPECT_EQ(q[0].as_object().at("q").to_number<double>(), 0.95) << "absent quantiles take the default";
  // The regulariser reaches the calibration: the same request without it (same seed) prices another base.
  json::object plain = req;
  plain.erase("regularize");
  const double plain_base =
      json::parse(api::var_json(json::object{{"var", plain}})).as_object().at("var").as_object().at("base_npv").to_number<double>();
  EXPECT_NE(out.at("base_npv").to_number<double>(), plain_base) << "the regulariser must change the calibrated base";
  // The golden pins a CONVERGED, regularised base: the same bundle, seed and regulariser through the calibrate dispatch.
  json::object cal_req{{"bundle", req.at("bundle")}, {"x0", req.at("x0")}, {"regularize", req.at("regularize")}};
  const json::object cal_out = json::parse(api::run_json(cal_req)).as_object();
  ASSERT_TRUE(cal_out.contains("calibration")) << json::serialize(cal_out);
  EXPECT_TRUE(cal_out.at("calibration").at("converged").as_bool()) << json::serialize(cal_out.at("calibration"));
  EXPECT_TRUE(cal_out.at("calibration").at("regularize_applied").as_bool());
}
