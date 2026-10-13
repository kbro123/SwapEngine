// g10_turns_bench -- the G10 desk with and without year-end turns, side by side (bench/fixtures/g10_market.hpp).
//
// g10_desk_ql_bench times the desk against QuantLib and is therefore limited to par swaps. This bench is QuantLib-free
// and times the SAME desk twice -- plain, and with a year-end turn on each G4 curve (bracket observation + TurnJump
// pin) -- on the numbers a desk pays for: the cold joint solve (cal::calibrate from the flat seed, the same
// definition as g10_8curve_cold_calibrate), the streamed 0.1 bp tick (frozen Jacobian), the 25 bp refresh tick and
// the hybrid engine's analytic Jacobian. BOTH rungs stream under Light smoothing (g10_market.hpp says why: a square
// control never refreshes and would charge the turns for leaving the square regime). The plain rung is the control: a
// difference between the two rungs is what the turns cost, nothing else varies.
#include <benchmark/benchmark.h>

#include <stdexcept>

#include <Eigen/Core>

#include "g10_market.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include <utility>

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {
const Shape& plain() { static const Shape s = swaps::shapes::g10_desk(false); return s; }
const Shape& turned() { static const Shape s = swaps::shapes::g10_desk(true); return s; }
// A session appends the regions' smoothing rows to ITS problem at construction (the client's bundle carries none), so the
// kernel-level rungs (cal::calibrate, the hybrid Jacobian) take the problem WITH its rows to time the same objective the
// session streams.
cal::BundleProblem with_rows(const Shape& s) {
  cal::BundleProblem p = s.prob;
  if (cal::add_smoothing_rows(p) == 0) throw std::runtime_error(s.name + ": no smoothing rows -- the timed pair must be the smoothed one");
  return p;
}
const cal::BundleProblem& plain_rows() { static const cal::BundleProblem p = with_rows(plain()); return p; }
const cal::BundleProblem& turned_rows() { static const cal::BundleProblem p = with_rows(turned()); return p; }

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

void cold(benchmark::State& state, const Shape& s, const cal::BundleProblem& prob) {
  for (auto _ : state) {
    const cal::CalibrationResult r = cal::calibrate(prob, s.x0);
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
void jacobian(benchmark::State& state, const Shape& s, const cal::BundleProblem& prob) {
  const cal::HybridBundleResidual h(prob);
  for (auto _ : state) {
    const Eigen::MatrixXd J = h.jacobian(s.x_true);
    benchmark::DoNotOptimize(J.data());
  }
}

void BM_G10_noturns_Cold(benchmark::State& st) { cold(st, plain(), plain_rows()); }
void BM_G10_g4turns_Cold(benchmark::State& st) { cold(st, turned(), turned_rows()); }
void BM_G10_noturns_StreamTick(benchmark::State& st) { stream_tick(st, plain()); }
void BM_G10_g4turns_StreamTick(benchmark::State& st) { stream_tick(st, turned()); }
void BM_G10_noturns_RefreshTick25bp(benchmark::State& st) { refresh_tick(st, plain()); }
void BM_G10_g4turns_RefreshTick25bp(benchmark::State& st) { refresh_tick(st, turned()); }
void BM_G10_noturns_Jacobian(benchmark::State& st) { jacobian(st, plain(), plain_rows()); }
void BM_G10_g4turns_Jacobian(benchmark::State& st) { jacobian(st, turned(), turned_rows()); }
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
  // PREMISE CHECK before any timing: the RAW pair (no smoothing) must calibrate from the flat seed back to its
  // generating state exactly with zero rank deficiency -- the turned desk's four jumps are identified, not sitting at
  // the seed -- and the timed (Light) pair must converge full-rank near it (a smoothed fit is not exact: the generating
  // forwards are a line in pillar INDEX, not in time, so the divided second difference pulls a few bp).
  for (bool turns : {false, true}) {
    const Shape raw = swaps::shapes::g10_desk(turns, /*light=*/false);
    const cal::CalibrationResult r = cal::calibrate(raw.prob, raw.x0);
    if (!r.converged || r.rank_deficiency != 0) throw std::runtime_error(raw.name + ": the raw desk does not calibrate cleanly: " + r.status);
    const double err = (r.x - raw.x_true).cwiseAbs().maxCoeff();
    if (err > 1e-7) throw std::runtime_error(raw.name + ": the raw cold solve does not recover x_true (max |dx| " + std::to_string(err) + ")");
  }
  for (const auto& [s, prob] : {std::pair{&plain(), &plain_rows()}, std::pair{&turned(), &turned_rows()}}) {
    const cal::CalibrationResult r = cal::calibrate(*prob, s->x0);
    if (!r.converged || r.rank_deficiency != 0) throw std::runtime_error(s->name + ": the smoothed desk does not calibrate cleanly: " + r.status);
    const double err = (r.x - s->x_true).cwiseAbs().maxCoeff();
    if (err > 5e-3) throw std::runtime_error(s->name + ": the smoothed solve is not near x_true (max |dx| " + std::to_string(err) + ")");
    api::BundleSession sess(s->prob);
    if (sess.problem().n_constraints() != prob->n_constraints()) throw std::runtime_error(s->name + ": the session's rows differ from the kernel rungs'");
  }
  int n_turns = 0;
  for (const auto& c : turned().prob.curves) n_turns += static_cast<int>(c.turns.size());
  if (n_turns != 4 || turned().prob.n_knots() != plain().prob.n_knots() + 4 || turned().prob.n_residuals() != plain().prob.n_residuals() + 8)
    throw std::runtime_error("g10_g4turns: expected 4 turns, +4 state and +8 rows over the plain desk");
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
