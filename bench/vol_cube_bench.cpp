// Swaption VOL-CUBE reprice bench — the options hot path (BundleSession::price_vol_cube_json).
//
// Measures the two regimes the options redesign cares about, mirroring the swap engine's cold/warm split:
//   * BM_VolCube_Warm  — reprice a 17-cell (smile + 4x4 ATM grid) SABR surface off an ALREADY-CALIBRATED
//                        session. This is the streaming/interactive number: moving the SABR params or the
//                        market reprices the whole cube with NO recalibration, one curve sample over the
//                        union of schedule times + a pure Bachelier/SABR pass. What a live vol surface costs.
//   * BM_VolCube_Cold  — construct the session from the bundle JSON, calibrate, then price the cube. The
//                        one-shot verb cost (Excel / a fresh run_json call).
//
// QuantLib-free, ours-only (no BM_*_QuantLib pair): QuantLib has no normal-vol swaption oracle wired here, so
// this is a SCALING/REGRESSION PROBE like bundle_scale_bench, not a fingerprint speedup gate. The bundle is a
// realistic mid-2026 SOFR curve (bench/fixtures/sofr_bundle.json, compiled from server/compile.py's default).
//
// Run on a quiesced machine (CLAUDE.md §4).
#include <benchmark/benchmark.h>

#include <fstream>
#include <sstream>
#include <string>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;

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

// A 17-cell SABR surface: a ±150bp 1Yx5Y smile (21 strikes) + a 4x4 ATM grid, the /options page's whole cube.
std::string cube_json() {
  std::string s =
      R"({"value_date":"2026-07-08","index":"USD-SOFR","cells":[)"
      R"({"expiry":"1Y","tenor":"5Y","sabr":{"alpha":0.009,"rho":-0.25,"nu":0.45},)"
      R"("moneyness_bp":[-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,105,120,135,150]})";
  const char* exps[] = {"1Y", "2Y", "5Y", "10Y"};
  const char* tens[] = {"2Y", "5Y", "10Y", "30Y"};
  for (const char* e : exps)
    for (const char* t : tens) {
      s += R"(,{"expiry":")";
      s += e;
      s += R"(","tenor":")";
      s += t;
      s += R"(","sabr":{"alpha":0.009,"rho":-0.25,"nu":0.45},"atm":true})";
    }
  s += "]}";
  return s;
}

api::RegSpec sofr_reg() {
  api::RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  reg.curves = {0};
  return reg;
}

api::BundleSession calibrated_session() {
  api::BundleSession sess(api::bundle_from_json(boost::json::parse(bundle_json())));
  sess.calibrate(api::flat_x0(sess.problem()), sofr_reg());
  return sess;
}

}  // namespace

// Warm: reprice the whole cube off a fixed calibrated session (the streaming / slider-move hot path).
static void BM_VolCube_Warm(benchmark::State& state) {
  const api::BundleSession sess = calibrated_session();
  const std::string cube = cube_json();
  for (auto _ : state) {
    api::VolCube r = sess.price_vol_cube_json(cube);
    benchmark::DoNotOptimize(r.price.data());
  }
}
BENCHMARK(BM_VolCube_Warm);

// Cold: build the session, calibrate, price the cube (the one-shot verb / Excel call).
static void BM_VolCube_Cold(benchmark::State& state) {
  const std::string js = bundle_json();
  const std::string cube = cube_json();
  const api::RegSpec reg = sofr_reg();
  for (auto _ : state) {
    api::BundleSession sess(api::bundle_from_json(boost::json::parse(js)));
    sess.calibrate(api::flat_x0(sess.problem()), reg);
    api::VolCube r = sess.price_vol_cube_json(cube);
    benchmark::DoNotOptimize(r.price.data());
  }
}
BENCHMARK(BM_VolCube_Cold);

BENCHMARK_MAIN();
