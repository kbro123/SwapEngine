// FX/MtM streaming hot path -- the HYBRID engine on a bundle with genuinely NON-W-cacheable rows, i.e.
// the rows that ride the AadBlock's width-reduced forward-AAD sweep (R11 pooled-dual retype): 5 MtM xccy basis
// swaps whose funding coupons carry the compounded PRODUCT observation. The bundle itself is
// bench/fixtures/fx_stream_fixture.hpp (shared with tests/fixture_identifiability_test.cpp, which pins that every
// curve has at least as many rows as knots and no weak direction at the solution -- owner rule 2026-10-09).
// Measures HybridBundleResidual::residuals / ::jacobian (the AAD block dominating), the AadBlock sweep in
// isolation (pooled vs the force-heap fallback -- the before/after of the DualPooled retype in ONE
// binary), and a StreamingCalibrator tick.
//
// QuantLib-free and ours-only (no BM_*_QuantLib pair; not a fingerprint gate metric). Run on a quiesced
// machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include "fx_stream_fixture.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::fxstream::fixture;

namespace {
const swaps::fxstream::Fixture& fx() { return fixture(); }
}  // namespace

// Full hybrid residual: cacheable rows on the W-cache + the FX/MtM rows on the AAD block's double path.
static void BM_FxStream_HybridResiduals(benchmark::State& state) {
  const auto& f = fx();
  const cal::HybridBundleResidual hr(f.prob);
  for (auto _ : state) {
    const Eigen::VectorXd& r = hr.residuals(f.x_solved);
    benchmark::DoNotOptimize(r.data());
  }
}
BENCHMARK(BM_FxStream_HybridResiduals);

// Full hybrid Jacobian: the analytic W-cache rows + the width-reduced AAD sweep (the cost under test).
static void BM_FxStream_HybridJacobian(benchmark::State& state) {
  const auto& f = fx();
  const cal::HybridBundleResidual hr(f.prob);
  for (auto _ : state) {
    Eigen::MatrixXd J = hr.jacobian(f.x_solved);
    benchmark::DoNotOptimize(J.data());
  }
}
BENCHMARK(BM_FxStream_HybridJacobian);

// The AadBlock sweep in ISOLATION, pooled (the default for touched width <= kPooledMaxW).
static void BM_FxStream_AadBlockJacobian(benchmark::State& state) {
  const auto& f = fx();
  cal::AadBlock blk;
  blk.init(f.prob.curves, f.nc, f.nc_rows, f.prob.n_knots());
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(f.prob.n_residuals(), f.prob.n_knots());
  for (auto _ : state) {
    blk.jacobian_into(f.x_solved, J);
    benchmark::DoNotOptimize(J.data());
  }
}
BENCHMARK(BM_FxStream_AadBlockJacobian);

// The same sweep FORCED onto the heap-Dual fallback -- the pre-R11 path, so this pair is the pooled
// retype's before/after in one binary (and the fallback a >48-wide bundle takes).
static void BM_FxStream_AadBlockJacobianHeap(benchmark::State& state) {
  const auto& f = fx();
  cal::AadBlock blk;
  blk.init(f.prob.curves, f.nc, f.nc_rows, f.prob.n_knots(), /*force_heap=*/true);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(f.prob.n_residuals(), f.prob.n_knots());
  for (auto _ : state) {
    blk.jacobian_into(f.x_solved, J);
    benchmark::DoNotOptimize(J.data());
  }
}
BENCHMARK(BM_FxStream_AadBlockJacobianHeap);

// A StreamingCalibrator tick on the mixed bundle (frozen-Newton; the AAD sweep prices the FX/MtM rows'
// residuals every tick and their Jacobian on refresh).
static void BM_FxStream_StreamTick(benchmark::State& state) {
  const auto& f = fx();
  cal::StreamingCalibrator<cal::BundleProblem> sc(f.prob, f.x_solved, f.q0, {});
  bool flip = false;
  for (auto _ : state) {
    cal::StreamTick t = sc.update(flip ? f.q0 : f.q1);
    flip = !flip;
    benchmark::DoNotOptimize(&t);
  }
}
BENCHMARK(BM_FxStream_StreamTick);

BENCHMARK_MAIN();
