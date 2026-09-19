// EXPERIMENT (branch exp/piecewise-linear-w): is a one-pass "all coefficients = K·x" structure faster than
// ModularCurve::set_forwards' region-by-region forward substitution? And what does the per-evaluation work
// cost when the consumer only needs integrals (W·x)?
//
// K is (n_coeffs x n_knots) and structure-only; for timing we only need its SHAPE (dense), and a banded
// variant for the local Hermite back end. n_coeffs matches what set_forwards writes for flat_hermite:
// Flat: f, Icum (2 per knot); Hermite over N=n+1 nodes: ys, a, b, c, d, Is (~6 per node).
#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/compiled.hpp"

namespace cv = swaps::curve;

namespace {
// Reference-market-shaped knot times: 6 meeting knots, 17 back knots (8 futures + 9 swaps).
const std::vector<double> kFront = {0.0575, 0.1918, 0.3068, 0.4219, 0.5562, 0.6904};
const std::vector<double> kBack = {1.4384, 1.6877, 1.9562, 2.2055, 2.4548, 2.7041, 2.9534, 3.2027, 4.0082,
                                   5.0082, 7.0137, 10.0137, 12.0192, 15.0164, 20.0192, 25.0219, 30.0274};
Eigen::VectorXd x0() {
  Eigen::VectorXd x(23);
  x << 0.0428, 0.0415, 0.0400, 0.0385, 0.0372, 0.0360, 0.0345, 0.0342, 0.0340, 0.0339, 0.0338, 0.0338,
      0.0339, 0.0341, 0.0345, 0.0352, 0.0362, 0.0375, 0.0382, 0.0390, 0.0396, 0.0392, 0.0385;
  return x;
}
// ~60 cashflow times spread over the curve (a small book's worth of pay/accrual dates).
std::vector<double> cf_times() {
  std::vector<double> t;
  for (int i = 1; i <= 60; ++i) t.push_back(0.5 * i);
  return t;
}
constexpr int kCoeffs = 2 * 6 + 6 * 18;  // 120

void BM_SetForwards_double(benchmark::State& st) {
  auto c = cv::make_modular_curve<double>(cv::flat_hermite(kFront, kBack));
  Eigen::VectorXd x = x0();
  for (auto _ : st) {
    x[3] += 1e-12;  // defeat hoisting
    c.set_forwards(x);
    benchmark::DoNotOptimize(c.integral(29.0));
  }
}
BENCHMARK(BM_SetForwards_double);

void BM_KDense_double(benchmark::State& st) {
  std::mt19937 g(7);
  std::normal_distribution<double> n;
  Eigen::MatrixXd K(kCoeffs, 23);
  for (int i = 0; i < K.size(); ++i) K.data()[i] = n(g);
  Eigen::VectorXd x = x0(), c(kCoeffs);
  for (auto _ : st) {
    x[3] += 1e-12;
    c.noalias() = K * x;
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_KDense_double);

void BM_KBanded_double(benchmark::State& st) {  // Hermite is LOCAL: each coefficient reads <= ~5 knots
  std::vector<Eigen::Triplet<double>> tr;
  for (int i = 0; i < kCoeffs; ++i)
    for (int k = std::max(0, i / 5 - 2); k <= std::min(22, i / 5 + 2); ++k) tr.emplace_back(i, k, 0.1 * (i + k));
  Eigen::SparseMatrix<double, Eigen::RowMajor> K(kCoeffs, 23);
  K.setFromTriplets(tr.begin(), tr.end());
  Eigen::VectorXd x = x0(), c(kCoeffs);
  for (auto _ : st) {
    x[3] += 1e-12;
    c.noalias() = K * x;
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_KBanded_double);

// What a consumer that prices off the curve actually pays: rebuild + evaluate the integral at 60 times + exp.
void BM_Rebuild_plus_60_DF(benchmark::State& st) {
  auto c = cv::make_modular_curve<double>(cv::flat_hermite(kFront, kBack));
  const auto T = cf_times();
  Eigen::VectorXd x = x0(), df(60);
  for (auto _ : st) {
    x[3] += 1e-12;
    c.set_forwards(x);
    for (int i = 0; i < 60; ++i) df[i] = c.discount(T[i]);
    benchmark::DoNotOptimize(df.data());
  }
}
BENCHMARK(BM_Rebuild_plus_60_DF);

void BM_W_plus_60_DF(benchmark::State& st) {  // the same 60 DFs off the precomputed W
  const auto W = swaps::pricing::integral_weight_matrix(cv::flat_hermite(kFront, kBack), cf_times());
  Eigen::VectorXd x = x0(), L(60), df(60);
  for (auto _ : st) {
    x[3] += 1e-12;
    L.noalias() = W * x;
    df = (-L).array().exp();
    benchmark::DoNotOptimize(df.data());
  }
}
BENCHMARK(BM_W_plus_60_DF);

// The AAD block's Jacobian sweep does the rebuild on POOLED duals (width 23): that is the cost a one-pass
// K (or W) removes entirely, since for a linear map the derivatives ARE the constant matrix.
void BM_Rebuild_plus_60_DF_pooledDual(benchmark::State& st) {
  using D = swaps::ad::DualPooled<swaps::ad::kPooledMaxW>;
  auto c = cv::make_modular_curve<D>(cv::flat_hermite(kFront, kBack));
  const auto T = cf_times();
  Eigen::VectorXd x = x0();
  Eigen::Matrix<D, Eigen::Dynamic, 1> xd(23);
  for (int k = 0; k < 23; ++k) { xd[k].derivatives().setZero(23); xd[k].derivatives()[k] = 1.0; }
  for (auto _ : st) {
    for (int k = 0; k < 23; ++k) xd[k].value() = x[k];
    c.set_forwards(xd);
    D acc = c.discount(T[0]);
    for (int i = 1; i < 60; ++i) acc += c.discount(T[i]);
    benchmark::DoNotOptimize(acc.value());
  }
}
BENCHMARK(BM_Rebuild_plus_60_DF_pooledDual);

void BM_W_plus_60_DF_jacobian(benchmark::State& st) {  // analytic equivalent: dDF/dx = -diag(DF)·W
  const auto W = swaps::pricing::integral_weight_matrix(cv::flat_hermite(kFront, kBack), cf_times());
  Eigen::VectorXd x = x0(), L(60), df(60), g(23);
  for (auto _ : st) {
    x[3] += 1e-12;
    L.noalias() = W * x;
    df = (-L).array().exp();
    g.noalias() = -(W.transpose() * df);  // gradient of Σ DF
    benchmark::DoNotOptimize(g.data());
  }
}
BENCHMARK(BM_W_plus_60_DF_jacobian);
}  // namespace

BENCHMARK_MAIN();
