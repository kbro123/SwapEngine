// The PRODUCT bundle timed end to end: SOFR (outright) + ESTR (outright) + EUR-in-USD xccy basis
// (spread over ESTR) — tests/reference_multicurrency.hpp's build_xccy_bundle(), real conventions via
// the QuantLib extraction path, self-consistent market. This is "the USD+EUR+xccy bundle" the web
// product calibrates; the desk-scale spread-chain benches (bundle_scale/session_warm) probe SCALING,
// this one reports the real thing. Ours-only probe (not fingerprint-gated).
#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include "reference_multicurrency.hpp"
#include "swaps/api/bundle_api.hpp"

namespace cal = swaps::calibration;
namespace rb = swaps::refbuild;
namespace api = swaps::api;

namespace {
const rb::MultiCcyBundle& fx() {
  static const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  return b;
}
api::RegSpec tension_reg(const rb::MultiCcyBundle& b) {  // (name kept: the bench's regularised variant)
  api::RegSpec r;                       // the web/SDK default smoothing shape: the curvature penalty
  r.lambda = 1e-3;
  for (int c = 0; c < b.n_curves(); ++c) r.curves.push_back(c);
  return r;
}
cal::BundleProblem perturbed(const rb::MultiCcyBundle& b) {
  cal::BundleProblem p = b.prob;
  for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i)
    p.instruments[i].market += 1e-4 * std::sin(0.7 * i + 0.3);  // ~1bp tick
  return p;
}
}  // namespace

// Cold: session construction + calibrate from the flat seed (what a fresh page load pays).
static void BM_XccyBundle_ColdCalibrate(benchmark::State& state) {
  const auto& b = fx();
  for (auto _ : state) {
    api::BundleSession sess(b.prob);
    sess.calibrate(b.x0);
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_XccyBundle_ColdCalibrate);

// Cold with the web's default tension smoothing (the shipped shape).
static void BM_XccyBundle_ColdCalibrateTension(benchmark::State& state) {
  const auto& b = fx();
  const api::RegSpec reg = tension_reg(b);
  for (auto _ : state) {
    api::BundleSession sess(b.prob);
    sess.calibrate(b.x0, reg);
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_XccyBundle_ColdCalibrateTension);

// Warm rebind on a ~1bp move (the SDK Model.requote path: cached engine, full quote RHS).
static void BM_XccyBundle_WarmRebind(benchmark::State& state) {
  const auto& b = fx();
  api::BundleSession sess(b.prob);
  sess.calibrate(b.x0);
  const cal::BundleProblem pert = perturbed(b);
  bool flip = false;
  for (auto _ : state) {
    sess.rebind(flip ? b.prob : pert);
    flip = !flip;
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_XccyBundle_WarmRebind);

// Streaming tick (the live-market path the web page runs).
static void BM_XccyBundle_StreamTick(benchmark::State& state) {
  const auto& b = fx();
  api::BundleSession sess(b.prob);
  sess.calibrate(b.x0);
  sess.start_streaming();
  const cal::BundleProblem pert = perturbed(b);
  Eigen::VectorXd q0 = b.prob.market(), q1 = pert.market();
  bool flip = false;
  for (auto _ : state) {
    const Eigen::VectorXd& x = sess.stream_update(flip ? q0 : q1);
    flip = !flip;
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_XccyBundle_StreamTick);
