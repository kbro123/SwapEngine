// Profiling probe for the SPECULATIVE background-Jacobian idea (research): what does a Jacobian recompute
// actually cost on the streaming path, vs a normal frozen-Newton tick? These numbers set the ceiling on
// what a background-thread prefetch could hide. Ours-only (not a gate metric).
//
//   BM_*_TickFastPath   : a streaming tick that RIDES the cached M (the common case) -- the cost the
//                         prefetch does NOT touch.
//   BM_*_JacAAD         : the full AAD Jacobian recompute (the expensive refresh; the ONLY option for a
//                         non-linear curve or an un-compiled problem).
//   BM_*_JacAnalytic    : the analytic W-cache Jacobian refresh (compiled problems).
//   BM_*_Refresh        : a full set_anchor = Jacobian + factorize M = J^{-1} (what a refresh tick pays).
#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>

#include "reference_bundle.hpp"
#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

// ---- single-curve square problem (23 knots) ----
namespace {
struct Single {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  cal::CompiledResidual cr{prob};
  Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  Eigen::VectorXd q0 = cr.model_rates(x0);
};
}  // namespace

static void BM_Single_JacAAD(benchmark::State& s) {
  Single f;
  for (auto _ : s) { auto J = cal::aad_jacobian(f.prob, f.x0); benchmark::DoNotOptimize(J.data()); }
}
BENCHMARK(BM_Single_JacAAD)->Unit(benchmark::kMicrosecond);

static void BM_Single_JacAnalytic(benchmark::State& s) {
  Single f;
  for (auto _ : s) { auto J = f.cr.jacobian(f.x0); benchmark::DoNotOptimize(J.data()); }
}
BENCHMARK(BM_Single_JacAnalytic)->Unit(benchmark::kMicrosecond);

static void BM_Single_RefreshAnalytic(benchmark::State& s) {  // Jacobian + factorize M = J^{-1}
  Single f;
  const int n = f.prob.n_residuals();
  for (auto _ : s) {
    const Eigen::MatrixXd J = f.cr.jacobian(f.x0);
    Eigen::MatrixXd M = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(Eigen::MatrixXd::Identity(n, n));
    benchmark::DoNotOptimize(M.data());
  }
}
BENCHMARK(BM_Single_RefreshAnalytic)->Unit(benchmark::kMicrosecond);

static void BM_Single_TickFastPath(benchmark::State& s) {  // a streaming tick riding the cached M
  Single f;
  cal::StreamingCalibrator<> sc(f.prob, f.x0, f.q0, {});
  double t = 0;
  for (auto _ : s) {
    Eigen::VectorXd q = f.q0;
    t += 1.0;
    for (int i = 0; i < q.size(); ++i) q[i] += 0.3e-4 * std::sin(0.3 * t + 0.4 * i);  // sub-bp -> no refresh
    auto tick = sc.update(q);
    benchmark::DoNotOptimize(tick.newton_steps);
  }
}
BENCHMARK(BM_Single_TickFastPath)->Unit(benchmark::kMicrosecond);

// ---- realistic 8-curve bundle (chain) : the AAD refresh here is the big one ----
namespace {
struct Bundle {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::RealisticBundle b = rb::build_realistic_bundle(mk, h, 7, rb::BundleTopology::Chain);  // 8 curves
  cal::CompiledBundleResidual cr{b.prob};
  Eigen::VectorXd x0 = cal::calibrate(b.prob, b.x0, true).x;
};
}  // namespace

static void BM_Bundle_JacAAD(benchmark::State& s) {
  Bundle f;
  for (auto _ : s) { auto J = cal::aad_jacobian(f.b.prob, f.x0); benchmark::DoNotOptimize(J.data()); }
}
BENCHMARK(BM_Bundle_JacAAD)->Unit(benchmark::kMicrosecond);

static void BM_Bundle_JacAnalytic(benchmark::State& s) {
  Bundle f;
  for (auto _ : s) { auto J = f.cr.jacobian(f.x0); benchmark::DoNotOptimize(J.data()); }
}
BENCHMARK(BM_Bundle_JacAnalytic)->Unit(benchmark::kMicrosecond);

static void BM_Bundle_RefreshAAD(benchmark::State& s) {  // AAD Jacobian + factorize -- the worst refresh
  Bundle f;
  const int n = f.b.prob.n_residuals();
  for (auto _ : s) {
    const Eigen::MatrixXd J = cal::aad_jacobian(f.b.prob, f.x0);
    Eigen::MatrixXd M = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(Eigen::MatrixXd::Identity(n, n));
    benchmark::DoNotOptimize(M.data());
  }
}
BENCHMARK(BM_Bundle_RefreshAAD)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
