// Curve-space GAMMA cost characterization (reverse-mode AAD tape) -- ours-only, QuantLib-free.
//
// The second-order cost against the first-order baselines, all on the SAME book/curve:
//   BM_CurveGradient  -- ONE reverse sweep: the full d(NPV)/dx (reverse-mode delta). O(one npv).
//   BM_BucketedDelta  -- risk.hpp's first-order market delta ladder (forward-AAD gradient + IFT solve).
//   BM_CurveGamma     -- the Hessian d^2(NPV)/dx^2 via n_knots Hessian-vector products (forward-over-reverse).
// Expectation: gamma ~ n_knots x the reverse-gradient cost (n HVPs). The ratio BM_CurveGamma/BM_BucketedDelta
// is the headline "what does second order cost over first order" number.

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <vector>

#include "research/gamma.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/risk.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace pr = swaps::pricing;
namespace pf_ns = swaps::portfolio;

namespace {

struct Legs {
  std::vector<pr::FloatCoupon> flt;
  std::vector<pr::FixedCoupon> fix;
};
Legs make_swap(double maturity) {
  Legs L;
  double a = 0.0;
  for (double b = 1.0; b <= maturity + 1e-9; b += 1.0) {
    const double tau = b - a;
    pr::FloatCoupon fc;
    fc.obs.sub_start = {a};
    fc.obs.sub_end = {b};
    fc.obs.tau_index = tau;
    fc.pay = b;
    fc.tau_pay = tau;
    L.flt.push_back(fc);
    pr::FixedCoupon xc;
    xc.pay = b;
    xc.tau = tau;
    L.fix.push_back(xc);
    a = b;
  }
  return L;
}

// 4 flat-front + 8 Hermite-back = 12 knots, with 12 ParRate swaps pinning them (J full column rank).
cal::CalibrationProblem build_problem() {
  cal::CalibrationProblem p;
  p.meeting_times = {0.25, 0.5, 0.75, 1.0};
  p.back_times = {2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0};
  const double mats[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 6.0, 8.0, 12.0};
  for (double mt : mats) {
    Legs L = make_swap(mt < 1.5 ? 1.0 : mt);  // ensure >=1 coupon
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd.coupons = L.flt;
    ins.fixed.coupons = L.fix;
    ins.market = 0.03;
    p.instruments.push_back(ins);
  }
  return p;
}

// A ~50-swap off-par book.
pf_ns::Portfolio build_book(int n) {
  pf_ns::Portfolio pf;
  const double mats[5] = {2.0, 3.0, 5.0, 7.0, 10.0};
  for (int i = 0; i < n; ++i) {
    Legs L = make_swap(mats[i % 5]);
    const double rate = 0.03 + ((i % 3) - 1) * 0.02;         // off par
    const double notl = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.1 * i);
    pf.positions.push_back({L.flt, L.fix, rate, notl});
  }
  return pf;
}

Eigen::VectorXd base_forwards(int n) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x[i] = 0.030 + 0.001 * i;
  return x;
}

const cal::CalibrationProblem g_prob = build_problem();
const pf_ns::Portfolio g_book = build_book(50);
const Eigen::VectorXd g_x = base_forwards(g_prob.n_knots());

}  // namespace

static void BM_CurveGradient(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::VectorXd g = cal::curve_gradient(g_prob, g_x, g_book);
    benchmark::DoNotOptimize(g.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_CurveGradient);

static void BM_BucketedDelta(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::VectorXd d = cal::bucketed_delta(g_prob, g_x, g_book);
    benchmark::DoNotOptimize(d.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_BucketedDelta);

static void BM_CurveGamma(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::MatrixXd H = cal::curve_gamma(g_prob, g_x, g_book);
    benchmark::DoNotOptimize(H.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_CurveGamma);

static void BM_MarketGammaGN(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::MatrixXd H = cal::market_gamma_gn(g_prob, g_x, g_book);
    benchmark::DoNotOptimize(H.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_MarketGammaGN);

BENCHMARK_MAIN();
