// shape_ladder_bench — the per-SHAPE perf-gate metrics (bench/fixtures/shape_ladder.hpp): for every instrument shape
// the hot path accepts, the streaming tick (0.1 bp, frozen Jacobian), the refresh tick (25 bp move: Jacobian +
// factorisation; on a banded shape a FOUR-NUMBER requote whose bands move with their targets, K5'), the requote tick (banded
// shapes: every row's target AND band, 0.1 bp), and the hybrid engine's analytic Jacobian. A kernel change is gated on EVERY rung, simplest to
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

// G4 PREMISES (2026-09-15): every tick metric reports, per tick, what its timed ticks actually did -- Jacobian refreshes, band
// re-scales, frozen steps -- so tools/check_perf.py can refuse a number whose workload no longer matches its name (a square rung's
// RefreshTick25bp never refreshed; a requote tick that starts refreshing would still read as a requote).
struct Premise {
  double refreshes = 0, rescales = 0, steps = 0;
  void add(const api::BundleSession& s) {
    refreshes += s.last_refreshes();
    rescales += s.last_rescales();
    steps += s.last_newton_steps();
  }
  void report(benchmark::State& st) const {
    st.counters["refreshes"] = benchmark::Counter(refreshes, benchmark::Counter::kAvgIterations);
    st.counters["rescales"] = benchmark::Counter(rescales, benchmark::Counter::kAvgIterations);
    st.counters["steps"] = benchmark::Counter(steps, benchmark::Counter::kAvgIterations);
  }
};

void stream_tick(benchmark::State& state, const Shape& s) {
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int i = 0; i < 4; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
  Premise p;
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_small : s.q0);
    benchmark::DoNotOptimize(x.data());
    p.add(sess);
    if (!sess.last_converged()) throw std::runtime_error(s.name + " stream tick did not converge: " + sess.last_reason());
  }
  p.report(state);
}
// One tick to `r`: a banded shape moves every row's target and band (K5'), an unbanded one its targets.
const Eigen::VectorXd& tick(api::BundleSession& sess, const Shape& s, const Shape::Requote& r) {
  return s.has_bands ? sess.stream_update(r.target, r.lower, r.upper, r.decay) : sess.stream_update(r.target);
}
void refresh_tick(benchmark::State& state, const Shape& s) {  // a 25 bp move each way: every tick refreshes J
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  const Shape::Requote big = s.requote(s.q_big), base = s.requote(s.q0);
  bool flip = false;
  for (int i = 0; i < 2; ++i) { flip = !flip; tick(sess, s, flip ? big : base); }
  Premise p;
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = tick(sess, s, flip ? big : base);
    benchmark::DoNotOptimize(x.data());
    p.add(sess);
    // A metric must never time a FAILING tick (the fx_xccy refresh was 17 us of non-finite ticks until 2026-09-10).
    if (!sess.last_converged()) throw std::runtime_error(s.name + " refresh tick did not converge: " + sess.last_reason());
  }
  p.report(state);
}
// The FOUR-NUMBER requote tick (K5', owner 2026-09-14): every row's target AND band move 0.1 bp together each tick. The band
// update rides the streamer in place (a row re-scale if a slope changes), never a re-anchor. Replaces EdgeOscTick, which timed
// a banded target oscillating OUTSIDE its band -- a quote the engine now refuses.
void requote_tick(benchmark::State& state, const Shape& s) {
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  const Shape::Requote small = s.requote(s.q_small), base = s.requote(s.q0);
  bool flip = false;
  for (int i = 0; i < 4; ++i) { flip = !flip; tick(sess, s, flip ? small : base); }
  Premise p;
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = tick(sess, s, flip ? small : base);
    benchmark::DoNotOptimize(x.data());
    p.add(sess);
    if (!sess.last_converged()) throw std::runtime_error(s.name + " requote tick did not converge: " + sess.last_reason());
  }
  p.report(state);
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
      benchmark::RegisterBenchmark(("BM_Shape_" + s.name + "_RequoteTick").c_str(), [&s](benchmark::State& st) { requote_tick(st, s); });
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
