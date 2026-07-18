// Branch-parallel staged bundle calibrate (CLAUDE.md §7b): serial vs thread-per-SCC on a STAR topology.
// Engine-only (no QuantLib): a base curve + K independent spread curves, so the dependency waves are
// [{base}, {K spreads}] and the K spread block-solves in wave 1 run concurrently. Ours-only probe (not a
// fingerprint gate metric) -- the point is the serial->parallel speedup, and that both give the identical
// solution (proven bit-for-bit in tests/bundle_test.cpp BundleParallel).
#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/lm.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

cal::Instrument annual_par_rate(double T, int fc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  std::vector<double> ends;  // annual, with a final stub to T (so T < 1 still yields ONE coupon [0,T])
  for (double u = 1.0; u < T - 1e-9; u += 1.0) ends.push_back(u);
  ends.push_back(T);
  double prev = 0.0;
  for (double u : ends) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev});
    prev = u;
  }
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.fixed.discount = dc;
  return ins;
}
cal::Instrument annual_basis(double T, int fwd_fc, int bench_fc, int dc) {
  cal::Instrument ins = annual_par_rate(T, fwd_fc, dc);
  ins.quote = cal::QuoteKind::ParSpread;
  ins.bench = ins.fwd;
  ins.bench.forecast = bench_fc;
  return ins;
}

// A star bundle: curve 0 outright base, curves 1..K each a spread straight off the base.
struct Star {
  cal::BundleProblem prob;
  Eigen::VectorXd x0;
  int nk = 0;
  explicit Star(int K) {
    const std::vector<double> meeting{0.5}, back{1, 2, 3, 4, 5, 7, 10, 15, 20, 30};
    prob.curves.resize(K + 1);
    prob.curves[0] = {meeting, back, -1};
    for (int c = 1; c <= K; ++c) prob.curves[c] = {meeting, back, 0};
    nk = prob.curves[0].n_knots();
    const std::vector<double> mats{0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30};
    for (double T : mats) prob.instruments.push_back(annual_par_rate(T, 0, 0));
    for (int c = 1; c <= K; ++c)
      for (double T : mats) prob.instruments.push_back(annual_basis(T, c, 0, 0));
    Eigen::VectorXd x_true((K + 1) * nk);
    for (int i = 0; i < nk; ++i) x_true[i] = 0.040 + 0.001 * i;
    for (int c = 1; c <= K; ++c)
      for (int i = 0; i < nk; ++i) x_true[c * nk + i] = 0.004 * c + 0.0003 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
    x0.resize((K + 1) * nk);
    x0.head(nk).setConstant(0.04);
    for (int c = 1; c <= K; ++c) x0.segment(c * nk, nk).setConstant(0.004 * c);
  }
};

}  // namespace

// K = 8 independent spread curves off one base -> wave 1 has 8 concurrent block solves.
static void BM_BundleStar8_StagedSerial(benchmark::State& s) {
  Star f(8);
  for (auto _ : s) {
    auto r = cal::calibrate_staged(f.prob, f.x0, true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleStar8_StagedSerial)->Unit(benchmark::kMicrosecond);

static void BM_BundleStar8_StagedParallel(benchmark::State& s) {
  Star f(8);
  for (auto _ : s) {
    auto r = cal::calibrate_staged_parallel(f.prob, f.x0, true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleStar8_StagedParallel)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
