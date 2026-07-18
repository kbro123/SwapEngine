// Large-scale bundle build -- a measurement bench for the compiled-engine hot-path optimizations
// (shared Jacobian gather, reused scratch buffers, DF memo). The 23-knot reference market is far too
// small to show them; here the gather runs over THOUSANDS of sub-periods per LM step, which is where
// removing its second pass (and the per-iteration allocations) actually pays.
//
// QuantLib-free and ours-only: this is NOT a fingerprint gate metric (no BM_*_QuantLib pair), it is a
// scaling probe. The bundle follows the desk convention: curve 0 is OUTRIGHT, every other curve is a
// SPREAD over the previous one (a spread chain), calibrated jointly and staged.
//
// Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/warm.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

constexpr int NC = 8;         // curves: 1 outright + 7 spread curves (a spread chain)
constexpr int NK = 26;        // knots per curve: 1 front + 25 back
constexpr double MAX_T = 30.0;  // longest tenor (years)

// An annual OIS to T as generic legs (one telescoped sub-period per coupon; tau cancels via k == 1).
struct Legs {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
Legs annual(double T) {
  Legs L;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    L.flt.push_back(c);
    L.fix.push_back({u, u - prev});
    prev = u;
  }
  return L;
}
cal::Instrument par_inst(double T, int fc, int dc) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {L.flt, fc, dc};
  in.fixed = {L.fix, dc};
  return in;
}
cal::Instrument basis_inst(double T, int fc, int bc, int dc) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParSpread;
  in.fwd = {L.flt, fc, dc};
  in.bench = {L.flt, bc, dc};
  in.fixed = {L.fix, dc};
  return in;
}

struct Fixture {
  cal::BundleProblem prob;
  Eigen::VectorXd x0, x_true, dq;
  cal::BundleProblem pert;                                    // perturbed market (warm cold-baseline)
  Eigen::VectorXd x_solved;
  std::unique_ptr<cal::WarmCalibrator<cal::BundleProblem>> wc;

  Fixture() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));  // back knots to 30y

    prob.curves.resize(NC);
    prob.curves[0] = {meeting, back, -1};                    // outright base
    for (int c = 1; c < NC; ++c) prob.curves[c] = {meeting, back, c - 1};  // spread over the previous

    std::vector<double> mats;
    for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);   // annual pillars 1y..30y

    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));   // curve 0 par swaps
    for (int c = 1; c < NC; ++c)
      for (double T : mats) prob.instruments.push_back(basis_inst(T, c, c - 1, 0));  // spread basis

    // x_true: curve 0 forward levels (~4%); spread curves ~20bp spreads. Self-consistent market.
    const int Ntot = NC * NK;
    x_true.resize(Ntot);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i)
        x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];

    x0.resize(Ntot);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;

    x_solved = cal::calibrate(prob, x0, true).x;
    dq = Eigen::VectorXd(prob.n_residuals());
    for (int i = 0; i < dq.size(); ++i) dq[i] = 1e-4 * std::sin(0.7 * i + 0.3);  // ~1bp tick
    pert = prob;
    for (int i = 0; i < static_cast<int>(pert.instruments.size()); ++i) pert.instruments[i].market += dq[i];
    wc = std::make_unique<cal::WarmCalibrator<cal::BundleProblem>>(prob, x_solved);
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

// Cold joint LM over the whole stacked bundle -- the compiled residual + analytic Jacobian per step,
// where the shared-gather / reused-scratch / DF-memo optimizations act on a large sub-period set.
static void BM_BundleScale_ColdJoint(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    auto r = cal::calibrate(f.prob, f.x0, true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleScale_ColdJoint);

// Staged (SCC-decomposed) solve: the spread chain is 8 singleton SCCs solved in dependency order.
static void BM_BundleScale_ColdStaged(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    auto r = cal::calibrate_staged(f.prob, f.x0, true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleScale_ColdStaged);

// Warm re-cal on a ~1bp tick: frozen-Jacobian Gauss-Newton, each step a large compiled residual eval.
static void BM_BundleScale_WarmRecal(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    auto r = f.wc->recalibrate(f.dq);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleScale_WarmRecal);

// ---- Profiling probes: isolate the pieces of one cold joint solve (measure before optimizing). ----
// EngineCtor builds W_all (currently one AAD pass per curve) + the compiled batches -- the one-time
// per-calibrate setup. OneResidual / OneJacobian are the per-LM-iteration costs. A full ColdJoint is
// ~ EngineCtor + n_iters*(OneResidual*trials + OneJacobian) + LM's own QR/step algebra.
static void BM_BundleScale_EngineCtor(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    cal::CompiledBundleResidual eng(f.prob);
    benchmark::DoNotOptimize(&eng);
  }
}
BENCHMARK(BM_BundleScale_EngineCtor);

static void BM_BundleScale_OneResidual(benchmark::State& state) {
  const auto& f = fx();
  const cal::CompiledBundleResidual eng(f.prob);
  for (auto _ : state) {
    Eigen::VectorXd r = eng.residuals(f.x_solved);
    benchmark::DoNotOptimize(r.data());
  }
}
BENCHMARK(BM_BundleScale_OneResidual);

static void BM_BundleScale_OneJacobian(benchmark::State& state) {
  const auto& f = fx();
  const cal::CompiledBundleResidual eng(f.prob);
  for (auto _ : state) {
    Eigen::MatrixXd J = eng.jacobian(f.x_solved);
    benchmark::DoNotOptimize(J.data());
  }
}
BENCHMARK(BM_BundleScale_OneJacobian);

BENCHMARK_MAIN();
