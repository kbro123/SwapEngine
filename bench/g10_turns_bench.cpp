// g10_turns_bench -- the G10 desk with and without year-end turns, side by side (bench/fixtures/g10_market.hpp).
//
// g10_desk_ql_bench times the desk against QuantLib and is therefore limited to par swaps. This bench is QuantLib-free
// and times the SAME desk twice -- plain, and with a year-end turn on each G4 curve (bracket observation + TurnJump
// pin) -- on the three numbers a desk pays for: the cold joint solve (cal::calibrate from the flat seed, the same
// definition as g10_8curve_cold_calibrate), the streamed 0.1 bp tick (frozen Jacobian), the 25 bp refresh tick and
// the hybrid engine's analytic Jacobian. The plain rung is the control: a difference between the two rungs is what
// the turns cost, nothing else varies.
#include <benchmark/benchmark.h>

#include <stdexcept>

#include <Eigen/Core>

#include "g10_market.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {
const Shape& plain() { static const Shape s = swaps::shapes::g10_desk(false); return s; }
const Shape& turned() { static const Shape s = swaps::shapes::g10_desk(true); return s; }

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

void cold(benchmark::State& state, const Shape& s) {
  for (auto _ : state) {
    const cal::CalibrationResult r = cal::calibrate(s.prob, s.x0);
    benchmark::DoNotOptimize(r.x.data());
    if (!r.converged) throw std::runtime_error(s.name + " cold calibrate did not converge: " + r.status);
  }
}
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
void refresh_tick(benchmark::State& state, const Shape& s) {
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int i = 0; i < 2; ++i) { flip = !flip; sess.stream_update(flip ? s.q_big : s.q0); }
  Premise p;
  for (auto _ : state) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_big : s.q0);
    benchmark::DoNotOptimize(x.data());
    p.add(sess);
    if (!sess.last_converged()) throw std::runtime_error(s.name + " refresh tick did not converge: " + sess.last_reason());
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

void BM_G10_noturns_Cold(benchmark::State& st) { cold(st, plain()); }
void BM_G10_g4turns_Cold(benchmark::State& st) { cold(st, turned()); }
void BM_G10_noturns_StreamTick(benchmark::State& st) { stream_tick(st, plain()); }
void BM_G10_g4turns_StreamTick(benchmark::State& st) { stream_tick(st, turned()); }
void BM_G10_noturns_RefreshTick25bp(benchmark::State& st) { refresh_tick(st, plain()); }
void BM_G10_g4turns_RefreshTick25bp(benchmark::State& st) { refresh_tick(st, turned()); }
void BM_G10_noturns_Jacobian(benchmark::State& st) { jacobian(st, plain()); }
void BM_G10_g4turns_Jacobian(benchmark::State& st) { jacobian(st, turned()); }
BENCHMARK(BM_G10_noturns_Cold);
BENCHMARK(BM_G10_g4turns_Cold);
BENCHMARK(BM_G10_noturns_StreamTick);
BENCHMARK(BM_G10_g4turns_StreamTick);
BENCHMARK(BM_G10_noturns_RefreshTick25bp);
BENCHMARK(BM_G10_g4turns_RefreshTick25bp);
BENCHMARK(BM_G10_noturns_Jacobian);
BENCHMARK(BM_G10_g4turns_Jacobian);
}  // namespace

int main(int argc, char** argv) {
  // PREMISE CHECK before any timing: both desks must calibrate from the flat seed to their generating state, and the
  // turned desk must actually carry four identified turn jumps (a turn nobody brackets would sit at the seed and the
  // metric would time a no-op).
  for (const Shape* s : {&plain(), &turned()}) {
    const cal::CalibrationResult r = cal::calibrate(s->prob, s->x0);
    if (!r.converged || r.rank_deficiency != 0) throw std::runtime_error(s->name + ": the desk does not calibrate cleanly: " + r.status);
    const double err = (r.x - s->x_true).cwiseAbs().maxCoeff();
    if (err > 1e-7) throw std::runtime_error(s->name + ": the cold solve does not recover x_true (max |dx| " + std::to_string(err) + ")");
  }
  int n_turns = 0;
  for (const auto& c : turned().prob.curves) n_turns += static_cast<int>(c.turns.size());
  if (n_turns != 4 || turned().prob.n_knots() != plain().prob.n_knots() + 4 || turned().prob.n_residuals() != plain().prob.n_residuals() + 8)
    throw std::runtime_error("g10_g4turns: expected 4 turns, +4 state and +8 rows over the plain desk");
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
