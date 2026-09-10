// ROUTING, BEFORE AND AFTER — over the WHOLE shape ladder (bench/fixtures/shape_ladder.hpp: 15 rungs,
// 8 quote kinds, 7 interpolation schemes).
//
// Until 2026-09-10 HybridBundleResidual asked ONE bundle-wide question -- curves_are_noncacheable(curves),
// "does any region anywhere have a value-dependent scheme?" -- and on a yes sent EVERY row to the AAD block.
// It now asks per row, against each curve's LINEAR HORIZON. This binary measures both answers on the same
// fixtures: the AFTER arm is the shipped engine; the BEFORE arm is an AadBlock holding ALL rows, which is
// exactly what the old veto produced.
//
// For a fully LINEAR rung the old router also compiled every row, so before == after by construction and the
// AAD column is only the cost of the fallback the old code would have used. The rungs where the two arms are a
// genuine before/after are the ones carrying a MonotoneCubic region: mixed_scheme and desk_mixed.
#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <cstdio>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {

struct Arms {
  cal::HybridBundleResidual routed;  // AFTER: per-row partition
  cal::AadBlock all_aad;             // BEFORE: the whole-bundle veto's outcome
  int compiled = 0, n = 0;
  explicit Arms(const Shape& s) : routed(s.prob), n(s.prob.n_residuals()) {
    for (int r = 0; r < n; ++r) compiled += routed.compiled_row(r) >= 0;
    std::vector<int> rows(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) rows[static_cast<std::size_t>(r)] = r;
    all_aad.init(s.prob.curves, s.prob.instruments, std::move(rows), s.prob.n_knots());
  }
};

void routed_residual(benchmark::State& st, const Shape& s) {
  Arms a(s);
  for (auto _ : st) benchmark::DoNotOptimize(a.routed.residuals(s.x_true).sum());
}
void aad_residual(benchmark::State& st, const Shape& s) {
  Arms a(s);
  Eigen::VectorXd r(s.prob.n_residuals());
  for (auto _ : st) { r.setZero(); a.all_aad.residuals_into(s.x_true, r); benchmark::DoNotOptimize(r.sum()); }
}
void routed_jacobian(benchmark::State& st, const Shape& s) {
  Arms a(s);
  for (auto _ : st) benchmark::DoNotOptimize(a.routed.jacobian(s.x_true).sum());
}
void aad_jacobian(benchmark::State& st, const Shape& s) {
  Arms a(s);
  Eigen::MatrixXd J(s.prob.n_residuals(), s.prob.n_knots());
  for (auto _ : st) { J.setZero(); a.all_aad.jacobian_into(s.x_true, J); benchmark::DoNotOptimize(J.sum()); }
}

}  // namespace

int main(int argc, char** argv) {
  static const std::vector<Shape> rungs = swaps::shapes::ladder();
  std::fprintf(stderr, "%-22s %5s %9s %9s\n", "rung", "rows", "compiled", "on AAD");
  for (const Shape& s : rungs) {
    const Arms a(s);
    std::fprintf(stderr, "%-22s %5d %9d %9d\n", s.name.c_str(), a.n, a.compiled, a.n - a.compiled);
  }
  for (const Shape& s : rungs) {
    benchmark::RegisterBenchmark(("BM_Route_" + s.name + "_Residual_routed").c_str(),
                                 [&s](benchmark::State& st) { routed_residual(st, s); });
    benchmark::RegisterBenchmark(("BM_Route_" + s.name + "_Residual_allAad").c_str(),
                                 [&s](benchmark::State& st) { aad_residual(st, s); });
    benchmark::RegisterBenchmark(("BM_Route_" + s.name + "_Jacobian_routed").c_str(),
                                 [&s](benchmark::State& st) { routed_jacobian(st, s); });
    benchmark::RegisterBenchmark(("BM_Route_" + s.name + "_Jacobian_allAad").c_str(),
                                 [&s](benchmark::State& st) { aad_jacobian(st, s); });
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
