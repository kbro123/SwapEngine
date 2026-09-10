// shape_ladder_bench — the per-SHAPE perf-gate metrics (bench/fixtures/shape_ladder.hpp): for every instrument shape
// the hot path accepts, the streaming tick (0.1 bp, frozen Jacobian), the refresh tick (25 bp move: Jacobian +
// factorisation), and the hybrid engine's analytic Jacobian. A kernel change is gated on EVERY rung, simplest to
// most complex, so an optimisation cannot be measured on annual OIS alone. Ours-only, QuantLib-free.
#include <benchmark/benchmark.h>

#include <stdexcept>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {
const std::vector<Shape>& ladder() { static const std::vector<Shape> L = swaps::shapes::ladder(); return L; }

void stream_tick(benchmark::State& state, const Shape& s) {
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int i = 0; i < 4; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_small : s.q0);
    benchmark::DoNotOptimize(x.data());
    if (!sess.last_converged()) throw std::runtime_error(s.name + " stream tick did not converge: " + sess.last_reason());
  }
}
void refresh_tick(benchmark::State& state, const Shape& s) {  // a 25 bp move each way: every tick refreshes J
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int i = 0; i < 2; ++i) { flip = !flip; sess.stream_update(flip ? s.q_big : s.q0); }
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_big : s.q0);
    benchmark::DoNotOptimize(x.data());
    // A metric must never time a FAILING tick (the fx_xccy refresh was 17 us of non-finite ticks until 2026-09-10).
    if (!sess.last_converged()) throw std::runtime_error(s.name + " refresh tick did not converge: " + sess.last_reason());
  }
}
// The active-set stress (C2, 2026-09-10): the banded rows' market oscillates 0.05 bp either side of their
// upper edge, so every tick the optimum sits ON a kink -- pins, multiplier checks and re-scales each tick.
void edge_osc_tick(benchmark::State& state, const Shape& s) {
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int i = 0; i < 4; ++i) { flip = !flip; sess.stream_update(flip ? s.q_edge_hi : s.q_edge_lo); }
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_edge_hi : s.q_edge_lo);
    benchmark::DoNotOptimize(x.data());
    if (!sess.last_converged()) throw std::runtime_error(s.name + " edge tick did not converge: " + sess.last_reason());
  }
}
void jacobian(benchmark::State& state, const Shape& s) {
  const cal::HybridBundleResidual h(s.prob);
  for (auto _ : state) {
    const Eigen::MatrixXd J = h.jacobian(s.x_true);
    benchmark::DoNotOptimize(J.data());
  }
}
}  // namespace

int main(int argc, char** argv) {
  for (const Shape& s : ladder()) {
    benchmark::RegisterBenchmark(("BM_Shape_" + s.name + "_StreamTick").c_str(), [&s](benchmark::State& st) { stream_tick(st, s); });
    benchmark::RegisterBenchmark(("BM_Shape_" + s.name + "_RefreshTick25bp").c_str(), [&s](benchmark::State& st) { refresh_tick(st, s); });
    benchmark::RegisterBenchmark(("BM_Shape_" + s.name + "_Jacobian").c_str(), [&s](benchmark::State& st) { jacobian(st, s); });
    if (s.has_bands)
      benchmark::RegisterBenchmark(("BM_Shape_" + s.name + "_EdgeOscTick").c_str(), [&s](benchmark::State& st) { edge_osc_tick(st, s); });
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
