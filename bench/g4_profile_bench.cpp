// g4_profile_bench -- where a COLD / OFF-HOT-PATH re-calibration of the hardest desk bundle spends its time
// (bench/fixtures/g4_market.hpp: 4 currencies, 14 curves, meeting-flat RFR fronts with a value-dependent long end).
//
// The owner's target: a full re-calibration off the hot path (a cold solve, a refresh after a regime move, the LM the
// streamer falls back to) in under 50 ms on this bundle. Each LM iteration pays one Jacobian, one or more residual
// evaluations and one column-pivoted QR of the (rows x knots) Jacobian (Eigen's NonLinearOptimization LM), so the rungs
// here time each of those pieces on the calibrated state beside the whole cold solve, per variant of the long end:
//   hermite     every row W-cached (the linear control: analytic Jacobian, no AAD)
//   monocubic   the RFR long ends on MonotoneCubic -- value-dependent, piecewise-linear: the W tier, re-synced per branch
//   monoconvex  the RFR long ends on MonotoneConvex -- not piecewise-linear: the rows past each horizon on the AAD block
//   monoconvex_all  every curve on MonotoneConvex: the whole bundle on the AAD block (the worst case)
// Counters report the partition (compiled vs AAD rows, the block's touched width and whether it is pooled or on the
// heap dual) and the LM's iteration count, so a timing can be read as iterations x (pieces). Not gated yet: a probe.
#include <benchmark/benchmark.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "g4_market.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/normal_op.hpp"
#include "swaps/calibration/regularize.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
using swaps::shapes::Shape;

namespace {
struct Variant {
  const char* tag;
  swaps::shapes::g4::Opts opts;
};
const std::vector<Variant>& variants() {
  static const std::vector<Variant> V = {
      {"hermite", {cv::Scheme::Hermite, false, true, true}},
      {"monocubic", {cv::Scheme::MonotoneCubic, false, true, true}},
      {"monoconvex", {cv::Scheme::MonotoneConvex, false, true, true}},
      {"monoconvex_all", {cv::Scheme::MonotoneConvex, true, true, true}},
  };
  return V;
}
// One fixture per variant, built once: the shape (the session's problem) and the problem WITH its smoothing rows (the
// kernel rungs time the objective the session streams, as g10_turns_bench does).
struct Fixture {
  Shape shape;
  cal::BundleProblem rows;
  Eigen::VectorXd x_cal;  // the smoothed cold solve's answer: the state the pieces are timed at
  int lm_iterations = 0;
};
const Fixture& fixture(std::size_t i) {
  static std::vector<Fixture> F = [] {
    std::vector<Fixture> out;
    for (const Variant& v : variants()) {
      Fixture f;
      f.shape = swaps::shapes::g4_desk(v.opts);
      f.rows = f.shape.prob;
      if (cal::add_smoothing_rows(f.rows) == 0) throw std::runtime_error(f.shape.name + ": no smoothing rows");
      const cal::CalibrationResult r = cal::calibrate(f.rows, f.shape.x0);
      if (!r.converged || r.rank_deficiency != 0)
        throw std::runtime_error(f.shape.name + ": the smoothed cold solve did not converge cleanly: " + r.status);
      f.x_cal = r.x;
      f.lm_iterations = r.iterations;
      out.push_back(std::move(f));
    }
    return out;
  }();
  return F[i];
}
void partition_counters(benchmark::State& st, const Fixture& f, const cal::HybridBundleResidual& h) {
  st.counters["knots"] = static_cast<double>(f.rows.n_knots());
  st.counters["rows"] = static_cast<double>(f.rows.n_residuals());
  st.counters["compiled_rows"] = static_cast<double>(h.n_compiled_rows());
  st.counters["aad_rows"] = static_cast<double>(f.rows.n_residuals() - h.n_compiled_rows());
  for (const auto& e : h.engines())
    if (const auto* blk = dynamic_cast<const cal::AadBlock*>(e.get())) {
      st.counters["aad_touched"] = blk->touched_width();
      st.counters["aad_pooled"] = blk->pooled() ? 1.0 : 0.0;
    }
}

void cold(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  double iters = 0;
  for (auto _ : st) {
    const cal::CalibrationResult r = cal::calibrate(f.rows, f.shape.x0);
    benchmark::DoNotOptimize(r.x.data());
    if (!r.converged) throw std::runtime_error(f.shape.name + " cold calibrate did not converge: " + r.status);
    iters += r.iterations;
  }
  st.counters["lm_iterations"] = benchmark::Counter(iters, benchmark::Counter::kAvgIterations);
  const cal::HybridBundleResidual h(f.rows);
  partition_counters(st, f, h);
}
// A WARM re-solve: the full LM from the calibrated state after a 25 bp move (the off-hot-path refresh the owner
// means: not a frozen tick, a genuine re-calibration from a good seed).
void warm_lm(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  cal::BundleProblem moved = f.rows;
  for (int r = 0; r < f.shape.prob.n_residuals(); ++r) moved.instruments[static_cast<std::size_t>(r)].market = f.shape.q_big[r];
  double iters = 0;
  for (auto _ : st) {
    const cal::CalibrationResult r = cal::calibrate(moved, f.x_cal);
    benchmark::DoNotOptimize(r.x.data());
    if (!r.converged) throw std::runtime_error(f.shape.name + " warm LM did not converge: " + r.status);
    iters += r.iterations;
  }
  st.counters["lm_iterations"] = benchmark::Counter(iters, benchmark::Counter::kAvgIterations);
}
void engine_build(benchmark::State& st, std::size_t i) {  // what cal::calibrate pays before its first iteration
  const Fixture& f = fixture(i);
  for (auto _ : st) {
    const cal::HybridBundleResidual h(f.rows);
    benchmark::DoNotOptimize(&h);
  }
}
void residuals(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const cal::HybridBundleResidual h(f.rows);
  for (auto _ : st) {
    const Eigen::VectorXd& r = h.residuals(f.x_cal);
    benchmark::DoNotOptimize(r.data());
  }
  partition_counters(st, f, h);
}
void jacobian(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const cal::HybridBundleResidual h(f.rows);
  for (auto _ : st) {
    const Eigen::MatrixXd J = h.jacobian(f.x_cal);
    benchmark::DoNotOptimize(J.data());
  }
  partition_counters(st, f, h);
}
// The Jacobian split by tier: the compiled W-cache rows alone, the AAD block alone (each engine's jacobian_into).
void jacobian_tier(benchmark::State& st, std::size_t i, bool aad) {
  const Fixture& f = fixture(i);
  const cal::HybridBundleResidual h(f.rows);
  const cal::RowEngine* eng = nullptr;
  for (const auto& e : h.engines()) {
    const bool is_aad = dynamic_cast<const cal::AadBlock*>(e.get()) != nullptr;
    if (is_aad == aad) eng = e.get();
  }
  if (!eng) { st.SkipWithError("no such tier on this variant"); return; }
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(f.rows.n_residuals(), f.rows.n_knots());
  for (auto _ : st) {
    eng->jacobian_into(f.x_cal, J);
    benchmark::DoNotOptimize(J.data());
  }
  st.counters["tier_rows"] = static_cast<double>(eng->rows().size());
}
// The per-iteration factorisation Eigen's LM pays: a column-pivoted Householder QR of the dense Jacobian.
void lm_qr(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const cal::HybridBundleResidual h(f.rows);
  const Eigen::MatrixXd J = h.jacobian(f.x_cal);
  for (auto _ : st) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J);
    benchmark::DoNotOptimize(qr.matrixQR().data());
  }
}
// The pseudo-inverse the streamer / risk / calibrate's rank check pay once per refresh (NormalOp::decompose).
void normal_op(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const cal::HybridBundleResidual h(f.rows);
  const Eigen::MatrixXd J = h.jacobian(f.x_cal);
  cal::NormalOp op(f.rows.n_residuals(), f.rows.n_knots());
  for (auto _ : st) {
    op.decompose(J);
    benchmark::DoNotOptimize(op.rank());
  }
}
void stream_tick(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const Shape& s = f.shape;
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int k = 0; k < 4; ++k) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
  double refreshes = 0, steps = 0;
  for (auto _ : st) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_small : s.q0);
    benchmark::DoNotOptimize(x.data());
    refreshes += sess.last_refreshes();
    steps += sess.last_newton_steps();
    if (!sess.last_converged()) { st.SkipWithError((s.name + " stream tick did not converge: " + sess.last_reason()).c_str()); break; }
  }
  st.counters["refreshes"] = benchmark::Counter(refreshes, benchmark::Counter::kAvgIterations);
  st.counters["steps"] = benchmark::Counter(steps, benchmark::Counter::kAvgIterations);
}
void refresh_tick(benchmark::State& st, std::size_t i) {
  const Fixture& f = fixture(i);
  const Shape& s = f.shape;
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  bool flip = false;
  for (int k = 0; k < 2; ++k) { flip = !flip; sess.stream_update(flip ? s.q_big : s.q0); }
  double refreshes = 0, steps = 0;
  for (auto _ : st) {
    flip = !flip;
    const Eigen::VectorXd& x = sess.stream_update(flip ? s.q_big : s.q0);
    benchmark::DoNotOptimize(x.data());
    refreshes += sess.last_refreshes();
    steps += sess.last_newton_steps();
    if (!sess.last_converged()) { st.SkipWithError((s.name + " refresh tick did not converge: " + sess.last_reason()).c_str()); break; }
  }
  st.counters["refreshes"] = benchmark::Counter(refreshes, benchmark::Counter::kAvgIterations);
  st.counters["steps"] = benchmark::Counter(steps, benchmark::Counter::kAvgIterations);
}

#define G4_VARIANT(IDX, TAG)                                                                                  \
  void BM_G4_##TAG##_Cold(benchmark::State& st) { cold(st, IDX); }                                            \
  void BM_G4_##TAG##_WarmLM25bp(benchmark::State& st) { warm_lm(st, IDX); }                                   \
  void BM_G4_##TAG##_EngineBuild(benchmark::State& st) { engine_build(st, IDX); }                             \
  void BM_G4_##TAG##_Residuals(benchmark::State& st) { residuals(st, IDX); }                                  \
  void BM_G4_##TAG##_Jacobian(benchmark::State& st) { jacobian(st, IDX); }                                    \
  void BM_G4_##TAG##_JacobianCompiled(benchmark::State& st) { jacobian_tier(st, IDX, false); }                \
  void BM_G4_##TAG##_JacobianAad(benchmark::State& st) { jacobian_tier(st, IDX, true); }                      \
  void BM_G4_##TAG##_LmQR(benchmark::State& st) { lm_qr(st, IDX); }                                           \
  void BM_G4_##TAG##_NormalOp(benchmark::State& st) { normal_op(st, IDX); }                                   \
  void BM_G4_##TAG##_StreamTick(benchmark::State& st) { stream_tick(st, IDX); }                               \
  void BM_G4_##TAG##_RefreshTick25bp(benchmark::State& st) { refresh_tick(st, IDX); }                         \
  BENCHMARK(BM_G4_##TAG##_Cold)->Unit(benchmark::kMillisecond);                                               \
  BENCHMARK(BM_G4_##TAG##_WarmLM25bp)->Unit(benchmark::kMillisecond);                                         \
  BENCHMARK(BM_G4_##TAG##_EngineBuild)->Unit(benchmark::kMicrosecond);                                        \
  BENCHMARK(BM_G4_##TAG##_Residuals)->Unit(benchmark::kMicrosecond);                                          \
  BENCHMARK(BM_G4_##TAG##_Jacobian)->Unit(benchmark::kMicrosecond);                                           \
  BENCHMARK(BM_G4_##TAG##_JacobianCompiled)->Unit(benchmark::kMicrosecond);                                   \
  BENCHMARK(BM_G4_##TAG##_JacobianAad)->Unit(benchmark::kMicrosecond);                                        \
  BENCHMARK(BM_G4_##TAG##_LmQR)->Unit(benchmark::kMicrosecond);                                               \
  BENCHMARK(BM_G4_##TAG##_NormalOp)->Unit(benchmark::kMicrosecond);                                           \
  BENCHMARK(BM_G4_##TAG##_StreamTick)->Unit(benchmark::kMicrosecond);                                         \
  BENCHMARK(BM_G4_##TAG##_RefreshTick25bp)->Unit(benchmark::kMicrosecond);
G4_VARIANT(0, hermite)
G4_VARIANT(1, monocubic)
G4_VARIANT(2, monoconvex)
G4_VARIANT(3, monoconvex_all)
#undef G4_VARIANT
}  // namespace

int main(int argc, char** argv) {
  // PREMISE CHECK before any timing: every variant's RAW build (no smoothing) calibrates from the flat seed back to its
  // generating state with zero rank deficiency -- fourteen curves and four turns identified, nothing parked at the seed.
  for (const Variant& v : variants()) {
    swaps::shapes::g4::Opts raw = v.opts;
    raw.smoothing = false;
    const Shape s = swaps::shapes::g4_desk(raw);
    const cal::CalibrationResult r = cal::calibrate(s.prob, s.x0);
    if (!r.converged || r.rank_deficiency != 0) throw std::runtime_error(s.name + ": the raw desk does not calibrate cleanly: " + r.status);
    const double err = (r.x - s.x_true).cwiseAbs().maxCoeff();
    if (err > 1e-7) throw std::runtime_error(s.name + ": the raw cold solve does not recover x_true (max |dx| " + std::to_string(err) + ")");
  }
  for (std::size_t i = 0; i < variants().size(); ++i) {
    const Fixture& f = fixture(i);  // builds + the smoothed cold solve (throws if it does not converge)
    api::BundleSession sess(f.shape.prob);
    if (sess.problem().n_constraints() != f.rows.n_constraints()) throw std::runtime_error(f.shape.name + ": the session's rows differ from the kernel rungs'");
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
