// QuantLib head-to-head for the swaption VOL SURFACE reprice — the system-level speed claim behind the
// options hot path. Both sides reprice the SAME 37-point surface (a ±150bp 1Yx5Y smile + a 4x4 ATM grid) off
// the SAME discounting, starting from a curve that is already built (the streaming regime — reprice as the
// vol/market moves, no re-bootstrap):
//   * BM_VolSurface_Ours     — BundleSession::price_vol_cube: ONE curve sample over the union of all schedule
//                              times into a reused SoA, then a pure Bachelier pass. (Includes the cube-spec
//                              JSON parse each call — a handicap AGAINST us, so the ratio is conservative.)
//   * BM_VolSurface_QuantLib — the standard QuantLib analytic path: per cell, query the QL term structure's
//                              discount factor at the start + every pay date, form the forward swap rate +
//                              annuity, and price each strike with QuantLib::bachelierBlackFormula. This is
//                              CONSERVATIVE for QuantLib: the full Swaption + BachelierSwaptionEngine object
//                              path (schedules, instrument, engine per cell) is heavier still; here QuantLib
//                              only pays its real per-cell term-structure lookups + the same closed form we use.
//
// So the delta measured is exactly the SYSTEM difference: our batched single-sample + SoA reuse vs QuantLib's
// per-cell curve access. Same closed form on both sides (QuantLib's bachelierBlackFormula IS our oracle), so
// this is a throughput comparison, not a correctness one (that is tests/vol_bachelier_ql_oracle.cpp). The QL
// curve is built once from our calibrated fixture curve's discount factors, so both discount identically.
//
// QuantLib-linked; ours-vs-QuantLib pair feeds tools/check_perf.py (metric vol_cube_warm). Run quiesced.
#include <benchmark/benchmark.h>

#include <ql/quantlib.hpp>

#include <boost/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace QL = QuantLib;

namespace {

const std::string& bundle_json() {
  static const std::string js = [] {
    std::ifstream f(SOFR_BUNDLE_JSON);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }();
  return js;
}

const char* kExp[] = {"1Y", "2Y", "5Y", "10Y"};
const char* kTen[] = {"2Y", "5Y", "10Y", "30Y"};
constexpr double kVol = 0.008;   // flat normal vol (bp-scale), same on both sides (isolate the system cost)
constexpr int kSmileExpY = 1, kSmileTenY = 5;

// The cube spec ours prices: a 1Yx5Y ±150bp smile (21 strikes) + a 4x4 ATM grid — 37 points across 17 cells.
std::string cube_json() {
  std::string s = R"({"value_date":"2026-07-08","index":"USD-SOFR","cells":[)"
                  R"({"expiry":"1Y","tenor":"5Y","normal_vol":0.008,)"
                  R"("moneyness_bp":[-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,105,120,135,150]})";
  for (const char* e : kExp)
    for (const char* t : kTen) {
      s += R"(,{"expiry":")";
      s += e;
      s += R"(","tenor":")";
      s += t;
      s += R"(","normal_vol":0.008,"atm":true})";
    }
  s += "]}";
  return s;
}

// The SAME 37-point surface as a NATIVE VolCubeSpec (no JSON) — what the perf bench actually prices, so we
// compare our compute to QuantLib's compute, not our JSON marshalling to QuantLib's compute.
api::VolCubeSpec make_spec() {
  api::VolCubeSpec s;
  s.value_date = "2026-07-08";
  s.index = "USD-SOFR";
  s.currency = "USD";
  s.curve = 0;
  api::VolCubeCell smile;
  smile.expiry = "1Y";
  smile.tenor = "5Y";
  smile.normal_vol = kVol;
  for (int bp = -150; bp <= 150; bp += 15) smile.moneyness_bp.push_back(bp);
  s.cells.push_back(std::move(smile));
  for (const char* e : kExp)
    for (const char* t : kTen) {
      api::VolCubeCell c;
      c.expiry = e;
      c.tenor = t;
      c.normal_vol = kVol;
      c.atm = true;
      s.cells.push_back(std::move(c));
    }
  return s;
}

api::BundleSession calibrated_session() {
  api::BundleSession sess(api::bundle_from_json(boost::json::parse(bundle_json())));
  api::RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  reg.curves = {0};
  sess.calibrate(api::flat_x0(sess.problem()), reg);
  return sess;
}

// A cell's QuantLib-side schedule + strikes, curve-independent, built once.
struct QLCell {
  QL::Date start;
  std::vector<QL::Date> pay;
  std::vector<double> tau;
  double expiry_T;
  std::vector<double> moneyness_bp;  // strikes are forward + bp/1e4 (empty single ATM => {0})
};

// Build QuantLib's OWN discount curve (log-linear in DF) from our calibrated curve's discount factors, so
// QuantLib discounts identically to us but through its own term-structure machinery.
QL::ext::shared_ptr<QL::YieldTermStructure> ql_curve_from(const api::BundleSession& sess) {
  const QL::Date ref(8, QL::July, 2026);
  QL::Settings::instance().evaluationDate() = ref;
  std::vector<double> times;
  for (int i = 0; i <= 492; ++i) times.push_back(i / 12.0);  // monthly to 41y
  const std::vector<api::CurveSample> cs = sess.sample(times);
  const std::vector<double>& df = cs.at(0).discount;
  std::vector<QL::Date> dates;
  std::vector<QL::DiscountFactor> dfs;
  for (std::size_t i = 0; i < times.size(); ++i) {
    dates.push_back(ref + static_cast<QL::Integer>(std::lround(times[i] * 365.0)));
    dfs.push_back(i == 0 ? 1.0 : df[i]);
  }
  auto curve = QL::ext::make_shared<QL::DiscountCurve>(dates, dfs, QL::Actual365Fixed());
  curve->enableExtrapolation();
  return curve;
}

std::vector<QLCell> ql_cells() {
  const QL::Date ref(8, QL::July, 2026);
  std::vector<QLCell> cells;
  auto make = [&](int expY, int tenY, bool smile) {
    QLCell c;
    c.start = ref + static_cast<QL::Integer>(std::lround(expY * 365.0));
    for (int k = 1; k <= tenY; ++k) {
      c.pay.push_back(c.start + static_cast<QL::Integer>(std::lround(k * 365.0)));
      c.tau.push_back(1.0);
    }
    c.expiry_T = expY;
    if (smile)
      for (int bp = -150; bp <= 150; bp += 15) c.moneyness_bp.push_back(bp);
    else
      c.moneyness_bp.push_back(0.0);  // ATM
    return c;
  };
  cells.push_back(make(kSmileExpY, kSmileTenY, /*smile=*/true));
  const int expY[] = {1, 2, 5, 10}, tenY[] = {2, 5, 10, 30};
  for (int e : expY)
    for (int t : tenY) cells.push_back(make(e, t, /*smile=*/false));
  return cells;
}

}  // namespace

// Ours: the whole surface off one batched curve sample + SoA reprice, via the NATIVE entry point (no JSON) —
// the fair native-vs-native comparison against QuantLib's native loop.
static void BM_VolSurface_Ours(benchmark::State& state) {
  const api::BundleSession sess = calibrated_session();
  const api::VolCubeSpec spec = make_spec();
  for (auto _ : state) {
    api::VolCube r = sess.price_vol_cube(spec);
    benchmark::DoNotOptimize(r.price.data());
  }
}
BENCHMARK(BM_VolSurface_Ours);

// QuantLib: the standard analytic path — per-cell term-structure discount lookups + bachelierBlackFormula.
static void BM_VolSurface_QuantLib(benchmark::State& state) {
  const api::BundleSession sess = calibrated_session();
  const auto curve = ql_curve_from(sess);
  const std::vector<QLCell> cells = ql_cells();
  const double sqrt_scale = 1.0;  // vols already absolute
  for (auto _ : state) {
    double sink = 0.0;
    for (const QLCell& c : cells) {
      const double df_start = curve->discount(c.start);
      double annuity = 0.0;
      for (std::size_t k = 0; k < c.pay.size(); ++k) annuity += c.tau[k] * curve->discount(c.pay[k]);
      const double df_end = curve->discount(c.pay.back());
      const double fwd = (df_start - df_end) / annuity;
      const double stddev = kVol * std::sqrt(c.expiry_T) * sqrt_scale;
      for (double bp : c.moneyness_bp) {
        const double K = fwd + bp / 1e4;
        const QL::Option::Type ot = (K >= fwd) ? QL::Option::Call : QL::Option::Put;
        sink += QL::bachelierBlackFormula(ot, K, fwd, stddev, annuity);
      }
    }
    benchmark::DoNotOptimize(sink);
  }
}
BENCHMARK(BM_VolSurface_QuantLib);

// How much of "ours" is just parsing the cube-spec JSON string each call (the string-interface marshalling
// cost QuantLib's native loop never pays) — the residual after the schedule/forward caches.
static void BM_VolSurface_ParseOnly(benchmark::State& state) {
  const std::string cube = cube_json();
  for (auto _ : state) {
    boost::json::value v = boost::json::parse(cube);
    benchmark::DoNotOptimize(&v);
  }
}
BENCHMARK(BM_VolSurface_ParseOnly);

BENCHMARK_MAIN();
