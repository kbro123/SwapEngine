// Stage-2: warm re-calibration under a small market perturbation. Three ways to re-solve the same
// curve after the market moves a little:
//   BM_WarmRecal_QuantLib   QuantLib IterativeBootstrap re-bootstrap (its FAST warm path)
//   BM_WarmRecal_OursFullLM  our from-scratch AAD LM (Stage-1)
//   BM_WarmRecal_Ours        our cached-Jacobian warm calibrator (Stage-2)
// The gate pairs _QuantLib with _Ours. Run on a quiesced machine (CLAUDE.md §4).

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

struct Fixture {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  DayCounter dc = mk.dc;
  Date today = mk.today;

  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  cal::WarmCalibrator<> wc{prob, x0};
  Eigen::VectorXd dq;  // a small (~1bp) in-envelope market move, residual order

  // QuantLib side (IterativeBootstrap)
  std::vector<ext::shared_ptr<SimpleQuote>> quotes;
  std::vector<ext::shared_ptr<RateHelper>> helpers = rb::build_square_ql(mk, quotes);

  Fixture() {
    dq = Eigen::VectorXd::Zero(prob.n_residuals());
    for (int i = 0; i < dq.size(); ++i) dq[i] = 3e-5 * ((i % 3) - 1);  // ~0.3bp live tick (in-envelope)
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

static void BM_WarmRecal_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  auto curve = ext::make_shared<PiecewiseYieldCurve<ForwardRate, BackwardFlat>>(f.today, f.helpers, f.dc);
  curve->enableExtrapolation();
  const Date far = f.today + 29 * Years;
  benchmark::DoNotOptimize(curve->discount(far));  // initial build
  int i = 0;
  for (auto _ : state) {
    // a small market move on every quote -> invalidates -> re-bootstraps on next discount()
    for (std::size_t j = 0; j < f.quotes.size(); ++j) {
      const double base = (j < 14) ? (j < 6 ? rm::futures_1m[j].price : rm::futures_3m[j - 6].price)
                                   : rm::swaps[j - 14].par_rate;
      f.quotes[j]->setValue(base + ((i % 2) ? 1e-3 : -1e-3));
    }
    ++i;
    benchmark::DoNotOptimize(curve->discount(far));
  }
}
BENCHMARK(BM_WarmRecal_QuantLib);

static void BM_WarmRecal_OursFullLM(benchmark::State& state) {
  const auto& f = fx();
  cal::CalibrationProblem p = f.prob;  // apply dq to targets once
  int i = 0;
  for (auto& a : p.avg_futs) a.market_rate += f.dq[i++];
  for (auto& c : p.comp_futs) c.market_rate += f.dq[i++];
  for (auto& s : p.swaps) s.market_rate += f.dq[i++];
  for (auto _ : state) {
    auto r = cal::calibrate(p, f.x0, true);  // warm-started full LM
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_WarmRecal_OursFullLM);

static void BM_WarmRecal_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    cal::WarmResult r = f.wc.recalibrate(f.dq);  // adaptive, exact (multi-step + envelope detection)
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_WarmRecal_Ours);

static void BM_WarmRecal_OursLinear(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    Eigen::VectorXd x = f.wc.recalibrate_linear(f.dq);  // first-order live-tick update: one matvec
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_WarmRecal_OursLinear);

BENCHMARK_MAIN();
