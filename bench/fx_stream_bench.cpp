// FX/MtM streaming hot path -- the HYBRID engine on a bundle with genuinely NON-W-cacheable rows, i.e.
// the rows that ride the AadBlock's width-reduced forward-AAD sweep (R11 pooled-dual retype):
//   * FX forwards NESTED inside Portfolio instruments (the compiled transforms don't compose in a Σ);
//   * MtM xccy basis swaps with a PAYMENT LAG on the funding leg (mtm_funding_term_negligible prices the
//     real rolled-out cashflows and correctly rejects them -> AAD).
// Measures HybridBundleResidual::residuals / ::jacobian (the AAD block dominating), the AadBlock sweep in
// isolation (pooled vs the force-heap fallback -- the before/after of the DualPooled retype in ONE
// binary), and a StreamingCalibrator tick.
//
// QuantLib-free and ours-only (no BM_*_QuantLib pair; not a fingerprint gate metric). Run on a quiesced
// machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

// ---- The synthetic 3-curve FX bundle: USD outright (0), EUR outright (1), EUR-in-USD spread (2). ----
constexpr int USD = 0, EUR = 1, EURUSD = 2;
constexpr double FX_SPOT = 1.10;
constexpr double PAY_LAG = 2.0 / 365.0;  // funding-leg payment lag -> the MtM funding term is NOT negligible

struct Legs {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
// An annual OIS to T as generic legs (one telescoped sub-period per coupon), optional payment lag.
Legs annual(double T, double lag = 0.0) {
  Legs L;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u + lag;
    c.tau_pay = u - prev;
    L.flt.push_back(c);
    L.fix.push_back({u + lag, u - prev});
    prev = u;
  }
  return L;
}

cal::Instrument par_inst(double T, int role) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {L.flt, role, role};
  in.fixed = {L.fix, role};
  return in;
}

// FX forward wrapped in a 1-component Portfolio: NON-cacheable by construction (an FX/MtM component
// forces the AAD block -- Instrument::noncacheable).
cal::Instrument fx_portfolio_inst(double T) {
  cal::Instrument fx;
  fx.quote = cal::QuoteKind::FxForward;
  fx.fx_num = EURUSD;
  fx.fx_den = USD;
  fx.fx_spot = FX_SPOT;
  fx.fx_time = T;
  cal::Instrument wrap;
  wrap.quote = cal::QuoteKind::Portfolio;
  wrap.combination = {{1.0, fx}};
  return wrap;
}

// MtM xccy basis whose USD funding leg pays with a LAG: the FX-reset funding bracket is genuinely
// nonzero, so mtm_funding_term_negligible rejects it and the row rides the AAD block.
cal::Instrument mtm_basis_inst(double T) {
  Legs Le = annual(T);                // EUR legs (self + benchmark share the schedule)
  Legs Lu = annual(T, PAY_LAG);       // USD funding leg, payment-lagged
  cal::Instrument in;
  in.quote = cal::QuoteKind::XccyMtmBasis;
  in.fwd = {Le.flt, EURUSD, EURUSD};  // self-forecast (primary = EUR-in-USD)
  in.bench = {Le.flt, EUR, EURUSD};   // EUR benchmark forecast
  in.fixed = {Le.fix, EURUSD};
  in.mtm = {Lu.flt, USD, USD};
  in.mtm.reset_num = EURUSD;
  in.mtm.reset_den = USD;
  in.mtm.fx_spot = FX_SPOT;
  return in;
}

struct Fixture {
  cal::BundleProblem prob;
  Eigen::VectorXd x_true, x0, x_solved, q0, q1;
  std::vector<cal::Instrument> nc;  // the non-cacheable instruments (for the AadBlock-only probes)
  std::vector<int> nc_rows;

  Fixture() {
    const std::vector<double> meet{0.25}, back{0.5, 1, 2, 3, 5, 7, 10};
    prob.curves.resize(3);
    prob.curves[USD] = px::CurveStructure{.base = -1, .currency = 0, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EUR] = px::CurveStructure{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EURUSD] = px::CurveStructure{.base = EUR, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};

    for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(par_inst(T, USD));
    for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(par_inst(T, EUR));
    for (double T : {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) prob.instruments.push_back(fx_portfolio_inst(T));
    for (double T : {2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(mtm_basis_inst(T));

    const int N = prob.n_knots();  // 24: 8 knots x 3 curves -> touched width 24 <= kPooledMaxW
    x_true.resize(N);
    auto fill = [&](int c, double lvl, double slope) {
      for (int i = 0; i < prob.curves[c].n_knots(); ++i) x_true[prob.offset(c) + i] = lvl + slope * i;
    };
    fill(USD, 0.0430, 0.0004);
    fill(EUR, 0.0300, 0.0004);
    fill(EURUSD, -0.0015, 0.00002);

    // Self-consistent market = the model quote at x_true (valid for every quote kind here; the FX rows
    // are Portfolio-wrapped, so their residual is the plain q - market, not the log transform).
    set_markets(x_true);

    x0.resize(N);
    x0.segment(prob.offset(USD), prob.curves[USD].n_knots()).setConstant(0.043);
    x0.segment(prob.offset(EUR), prob.curves[EUR].n_knots()).setConstant(0.030);
    x0.segment(prob.offset(EURUSD), prob.curves[EURUSD].n_knots()).setConstant(-0.0015);
    x_solved = cal::calibrate(prob, x0).x;

    // Live-feed pair for the streaming tick: q0 = the anchor market, q1 = a FEASIBLE ~5bp market move
    // (model quotes at a perturbed x, so the frozen-Newton fixed point exists for the FX/MtM couplings).
    q0 = prob.market();
    Eigen::VectorXd xp = x_true;
    for (int k = 0; k < N; ++k) xp[k] += 5e-4 * ((k % 2) ? 1.0 : -1.0);
    q1 = model_quotes(xp);

    // The non-cacheable partition (mirrors HybridBundleResidual's), for the AadBlock-only probes.
    for (int r = 0; r < prob.n_residuals(); ++r)
      if ((prob.instruments[r]).noncacheable()) {
        nc.push_back(prob.instruments[r]);
        nc_rows.push_back(r);
      }
  }

  Eigen::VectorXd model_quotes(const Eigen::VectorXd& x) const {
    const auto C = cal::build_bundle_curves<double>(
        prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
    Eigen::VectorXd q(prob.n_residuals());
    for (int i = 0; i < prob.n_residuals(); ++i)
      q[i] = cal::instrument_model_quote<double>(prob.instruments[i], curve_of);
    return q;
  }
  void set_markets(const Eigen::VectorXd& x) {
    const Eigen::VectorXd q = model_quotes(x);
    for (int i = 0; i < prob.n_residuals(); ++i) prob.instruments[i].market = q[i];
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

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
