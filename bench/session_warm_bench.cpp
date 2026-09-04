// BundleSession warm-path benches -- the SHIPPED entry points, not the inner kernels. The audit found the
// product surface leaving the fast path (rebind/recalibrate rebuilt the whole compiled engine per call;
// the tension-regularized default fell to a per-iteration AAD sweep). These benches measure the session
// calls a client actually makes, at desk scale (8-curve spread chain, 26 knots/curve, annual pillars to
// 30y), and two of them are FINGERPRINT-GATED (session_rebind, stream_tick in tools/check_perf.py) so the
// warm path can never silently regress back to a recompile.
//
// QuantLib-free, ours-only. Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

constexpr int NC = 8;           // 1 outright + 7 spread curves (a spread chain)
constexpr int NK = 26;          // knots per curve
constexpr double MAX_T = 30.0;  // longest tenor (years)

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
  Eigen::VectorXd x0;
  cal::BundleProblem pert;  // ~1bp-perturbed quotes (the rebind payload)
  api::RegSpec tension;     // the SDK/web default: tension-energy smoothing over every curve

  Fixture() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c)
      prob.curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};

    std::vector<double> mats;
    for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);
    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));
    for (int c = 1; c < NC; ++c)
      for (double T : mats) prob.instruments.push_back(basis_inst(T, c, c - 1, 0));

    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i)
        x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];

    x0.resize(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;

    pert = prob;
    for (int i = 0; i < static_cast<int>(pert.instruments.size()); ++i)
      pert.instruments[i].market += 1e-4 * std::sin(0.7 * i + 0.3);

    tension.lambda = 1e-3;
    tension.tension = true;
    tension.sigma = 0.5;
    for (int c = 0; c < NC; ++c) tension.curves.push_back(c);
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

// The GATED warm quote-RHS re-solve: session.rebind on a calibrated session, alternating between two
// nearby markets so every call genuinely re-solves. The engine must be REUSED across calls (set_quotes +
// warm LM); a regression to per-call engine construction shows up here as milliseconds, not microseconds.
static void BM_Session_RebindWarm(benchmark::State& state) {
  const auto& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  bool flip = false;
  for (auto _ : state) {
    sess.rebind(flip ? f.prob : f.pert);
    flip = !flip;
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_Session_RebindWarm);

// The SDK/web DEFAULT path: rebind under tension-energy smoothing. Rides the cached compiled engine
// composed with the constant R block -- previously a per-iteration AAD sweep over a wrapper problem.
static void BM_Session_RebindTensionWarm(benchmark::State& state) {
  const auto& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0, f.tension);
  bool flip = false;
  for (auto _ : state) {
    sess.rebind(flip ? f.prob : f.pert, f.tension);
    flip = !flip;
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_Session_RebindTensionWarm);

// The GATED bundle-scale streaming tick: frozen-Newton stream_update against an alternating ~1bp market.
// This is the headline "microsecond live curve" claim at desk scale -- previously unbenched, so it had no
// regression gate at all.
static void BM_Session_StreamTick(benchmark::State& state) {
  const auto& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  sess.start_streaming();
  Eigen::VectorXd q0(f.prob.n_residuals()), q1(f.prob.n_residuals());
  for (int i = 0; i < q0.size(); ++i) {
    q0[i] = f.prob.instruments[i].market;
    q1[i] = f.pert.instruments[i].market;
  }
  bool flip = false;
  for (auto _ : state) {
    const Eigen::VectorXd& x = sess.stream_update(flip ? q0 : q1);
    flip = !flip;
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_Session_StreamTick);

// The GATED shipped book reprice: session.price_portfolio over a 200-swap multi-curve book off the
// calibrated bundle. This is the productized MultiCurveBook path (templated virtual curves today) -- the
// audit's U2 target; gating it locks the baseline the compiled multi-curve book must later beat.
static void BM_Session_PricePortfolio(benchmark::State& state) {
  const auto& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  swaps::portfolio::MultiCurveBook book;
  for (int i = 0; i < 200; ++i) {
    const double T = 1.0 + (i % 30);
    Legs L = annual(T);
    swaps::portfolio::MultiCurveBook::Position p;
    p.kind = swaps::portfolio::MultiCurveBook::Kind::Swap;
    p.notional = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.01 * i);
    p.float_coupons = L.flt;
    p.fixed_coupons = L.fix;
    p.fwd_curve = i % NC;
    p.disc_curve = 0;
    p.fixed_curve = 0;
    p.fixed_rate = 0.04;
    book.positions.push_back(std::move(p));
  }
  for (auto _ : state) {
    const api::PortfolioReprice r = sess.price_portfolio(book);
    benchmark::DoNotOptimize(r.npv);
  }
}
BENCHMARK(BM_Session_PricePortfolio);

// Cold-construction reference: a full session build + cold calibrate (what a rebind used to approximate).
// Not gated -- context for the warm numbers above.
static void BM_Session_ColdBuildCalibrate(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    api::BundleSession sess(f.prob);
    sess.calibrate(f.x0);
    benchmark::DoNotOptimize(sess.x().data());
  }
}
BENCHMARK(BM_Session_ColdBuildCalibrate);
