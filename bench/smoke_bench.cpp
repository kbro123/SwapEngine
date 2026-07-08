// Phase 0 smoke benchmark: proves the Google Benchmark wiring builds & runs.
// Real benchmarks (curve build, AAD risk, batched analytics vs QuantLib) arrive in Phase 1+.
#include <benchmark/benchmark.h>
#include <Eigen/Dense>

static void BM_EigenMatVec(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  Eigen::MatrixXd M = Eigen::MatrixXd::Random(n, n);
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);
  for (auto _ : state) {
    Eigen::VectorXd r = M * v;   // stand-in for batched portfolio algebra
    benchmark::DoNotOptimize(r.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_EigenMatVec)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
