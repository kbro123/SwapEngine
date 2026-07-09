// Phase 3 speed check: analytic AAD Jacobian vs bump-and-reprice, and AAD-driven vs
// numerical-Jacobian LM calibration. Not a gate (that is Phase 6) -- it reports the speedup the
// AAD design buys. Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"

namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

// Built once; the QuantLib extraction is setup, not part of what we are timing.
struct Fixture {
  QuantLib::RelinkableHandle<QuantLib::YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_problem(mk);
  Eigen::VectorXd x = [] {
    Eigen::VectorXd v(rm::n_knots);
    int i = 0;
    for (double f : rm::reference_front_forwards) v[i++] = f;
    for (double g : rm::reference_back_forwards) v[i++] = g;
    return v;
  }();
};
const Fixture& fixture() {
  static const Fixture f;
  return f;
}

Eigen::MatrixXd bump_jacobian(const cal::CalibrationProblem& p, const Eigen::VectorXd& x, double bump) {
  Eigen::MatrixXd J(p.n_residuals(), p.n_knots());
  for (int k = 0; k < p.n_knots(); ++k) {
    Eigen::VectorXd xp = x, xm = x;
    xp[k] += bump;
    xm[k] -= bump;
    J.col(k) = (p.residuals<double>(xp) - p.residuals<double>(xm)) / (2 * bump);
  }
  return J;
}

}  // namespace

static void BM_Jacobian_AAD(benchmark::State& state) {
  const auto& f = fixture();
  for (auto _ : state) {
    Eigen::MatrixXd J = cal::aad_jacobian(f.prob, f.x);
    benchmark::DoNotOptimize(J.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Jacobian_AAD);

static void BM_Jacobian_BumpReprice(benchmark::State& state) {
  const auto& f = fixture();
  for (auto _ : state) {
    Eigen::MatrixXd J = bump_jacobian(f.prob, f.x, 1e-6);
    benchmark::DoNotOptimize(J.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Jacobian_BumpReprice);

static void BM_Calibrate_AAD(benchmark::State& state) {
  const auto& f = fixture();
  Eigen::VectorXd x0 = f.x;
  for (int i = 0; i < x0.size(); ++i) x0[i] += (i % 2 ? 0.002 : -0.002);
  for (auto _ : state) {
    auto r = cal::calibrate(f.prob, x0, /*use_aad=*/true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_Calibrate_AAD);

static void BM_Calibrate_Numerical(benchmark::State& state) {
  const auto& f = fixture();
  Eigen::VectorXd x0 = f.x;
  for (int i = 0; i < x0.size(); ++i) x0[i] += (i % 2 ? 0.002 : -0.002);
  for (auto _ : state) {
    auto r = cal::calibrate(f.prob, x0, /*use_aad=*/false);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_Calibrate_Numerical);

BENCHMARK_MAIN();
