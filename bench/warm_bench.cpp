// Stage-2: warm re-calibration after a market tick — LIKE-FOR-LIKE on both sides (PRINCIPLES.md P6/P10).
//
//   BM_WarmRecal_QuantLib        QuantLib IterativeBootstrap re-bootstrap after a ±0.3 bp tick on EVERY quote
//   BM_WarmRecal_Ours            our cached-Jacobian warm calibrator after the SAME ±0.3 bp tick
//   BM_WarmRecal_QuantLib_10bp   the same pair at ±10 bp (a move that leaves our Jacobian envelope, so
//   BM_WarmRecal_Ours_10bp       ours pays a Jacobian refresh — the honest "big move" number)
//   BM_WarmRecal_OursFullLM      our from-scratch AAD LM, warm-started (Stage-1, for context)
//   BM_WarmRecal_OursLinear      the one-matvec first-order update (context; O(dq²) error)
//
// History: until 2026-09-08 the QuantLib side moved every quote ±1e-3 (10 bp on swaps, 0.01 bp on
// futures) while ours re-solved a fixed 0.3 bp in-envelope tick — apples to oranges. Both sides now see the
// identical market move each iteration (sign alternating so QuantLib's observer chain re-bootstraps every
// time and ours cannot short-circuit on an unchanged input). Futures quotes are PRICE points
// (100·(1−rate)), so a +dq rate move is a −100·dq price move.
//
// check_perf.py gates warm_recalibration / warm_recal_10bp on OURS (self-baseline + target); the QuantLib
// numbers are the informational reference. Run on a quiesced machine.

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/warm.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

constexpr int kFutures1m = 6;   // helper/residual order: 6 averaged 1m futures, 8 compounded 3m, then swaps
constexpr int kFutures = 14;

Eigen::VectorXd tick(int n, double bp) {
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i) dq[i] = bp * 1e-4 * ((i % 3) - 1);  // {-bp, 0, +bp} pattern in RATE units
  return dq;
}

struct Fixture {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  DayCounter dc = mk.dc;
  Date today = mk.today;

  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  cal::WarmCalibrator<> wc{prob, x0};
  Eigen::VectorXd dq03 = tick(prob.n_residuals(), 0.3);   // 0.3 bp live tick (inside the J envelope)
  Eigen::VectorXd dq10 = tick(prob.n_residuals(), 10.0);  // 10 bp move (outside: forces a J refresh)

  // QuantLib side (IterativeBootstrap)
  std::vector<ext::shared_ptr<SimpleQuote>> quotes;
  std::vector<ext::shared_ptr<RateHelper>> helpers = rb::build_square_ql(mk, quotes);

  double base_quote(std::size_t j) const {
    return (j < kFutures) ? (j < kFutures1m ? rm::futures_1m[j].price : rm::futures_3m[j - kFutures1m].price)
                          : rm::swaps[j - kFutures].par_rate;
  }
  // Apply the SAME rate move dq (residual order == helper order) to the QuantLib quotes, sign-flipped.
  void set_ql_market(const Eigen::VectorXd& dq, double sign) const {
    for (std::size_t j = 0; j < quotes.size(); ++j) {
      const double d = sign * dq[static_cast<int>(j)];
      quotes[j]->setValue(base_quote(j) + (j < kFutures ? -100.0 * d : d));
    }
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

void bm_ql(benchmark::State& state, const Eigen::VectorXd& dq) {
  const auto& f = fx();
  auto curve = ext::make_shared<PiecewiseYieldCurve<ForwardRate, BackwardFlat>>(f.today, f.helpers, f.dc);
  curve->enableExtrapolation();
  const Date far = f.today + 29 * Years;
  f.set_ql_market(dq, 0.0);
  benchmark::DoNotOptimize(curve->discount(far));  // initial build
  double sign = 1.0;
  for (auto _ : state) {
    f.set_ql_market(dq, sign);  // every quote moves -> observers invalidate -> re-bootstrap on discount()
    sign = -sign;
    benchmark::DoNotOptimize(curve->discount(far));
  }
}

void bm_ours(benchmark::State& state, const Eigen::VectorXd& dq) {
  const auto& f = fx();
  double sign = 1.0;
  for (auto _ : state) {
    cal::WarmResult r = f.wc.recalibrate(sign * dq);  // adaptive, exact (multi-step + envelope detection)
    sign = -sign;
    benchmark::DoNotOptimize(r.x.data());
  }
}

}  // namespace

static void BM_WarmRecal_QuantLib(benchmark::State& state) { bm_ql(state, fx().dq03); }
BENCHMARK(BM_WarmRecal_QuantLib);

static void BM_WarmRecal_Ours(benchmark::State& state) { bm_ours(state, fx().dq03); }
BENCHMARK(BM_WarmRecal_Ours);

static void BM_WarmRecal_QuantLib_10bp(benchmark::State& state) { bm_ql(state, fx().dq10); }
BENCHMARK(BM_WarmRecal_QuantLib_10bp);

static void BM_WarmRecal_Ours_10bp(benchmark::State& state) { bm_ours(state, fx().dq10); }
BENCHMARK(BM_WarmRecal_Ours_10bp);

static void BM_WarmRecal_OursFullLM(benchmark::State& state) {
  const auto& f = fx();
  cal::CalibrationProblem p = f.prob;  // apply dq to targets once (instrument order == residual order)
  for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i) p.instruments[i].market += f.dq03[i];
  for (auto _ : state) {
    auto r = cal::calibrate(p, f.x0, true);  // warm-started full LM
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_WarmRecal_OursFullLM);

static void BM_WarmRecal_OursLinear(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    Eigen::VectorXd x = f.wc.recalibrate_linear(f.dq03);  // first-order live-tick update: one matvec
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_WarmRecal_OursLinear);

BENCHMARK_MAIN();
