// Curve-build baseline: our AAD global-LM calibration vs QuantLib's OWN global LM
// (GlobalBootstrap) -- the honest same-algorithm-class comparison (CLAUDE.md §3).
//
// Square problem (23 instruments, 23 knots at the instrument pillars, zero convexity both sides) so
// GlobalBootstrap does not throw on a residual. This isolates our contribution: AAD Jacobian +
// templated pricing kernel vs GlobalBootstrap's numerical Jacobian + per-coupon object pricing.
//
// NOT compared against IterativeBootstrap: that is a different, cheaper algorithm (sequential 1-D)
// which cannot do the over-determined global fit or supply analytic risk. It is faster on the
// simple square case and that is stated openly in the perf notes -- it is not our target.
//
// Run on a quiesced machine (CLAUDE.md §4). BM_* names ending _QuantLib / _Ours let the Phase 6
// perf checker pair them into a speedup.

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/ql/extract.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

using GlobalPWC = PiecewiseYieldCurve<ForwardRate, BackwardFlat, GlobalBootstrap>;

// Built once: the same 23 instruments as QuantLib rate helpers AND as our extracted schedules,
// with knots at the shared instrument pillar dates.
struct SquareFixture {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  DayCounter dc = mk.dc;
  Date today = mk.today;
  Date far = mk.today + 29 * Years;

  std::vector<ext::shared_ptr<RateHelper>> helpers;             // QuantLib side
  cal::CalibrationProblem prob = rb::build_square_problem(mk);  // our side (generic Instruments)
  Eigen::VectorXd x0;

  SquareFixture() {
    // QuantLib rate helpers for the SAME 23 instruments; our side (`prob`) is built generically by
    // build_square_problem, so this benchmark now measures the ADOPTED (generic-instrument) path.
    for (int i = 0; i < 6; ++i) {
      const auto& q = rm::futures_1m[i];
      helpers.push_back(ext::make_shared<SofrFutureRateHelper>(
          Handle<Quote>(ext::make_shared<SimpleQuote>(q.price)), Month(q.ref_month), q.ref_year,
          Monthly));
    }
    for (int i = 0; i < 8; ++i) {
      const auto& q = rm::futures_3m[i];
      helpers.push_back(ext::make_shared<SofrFutureRateHelper>(
          Handle<Quote>(ext::make_shared<SimpleQuote>(q.price)), Month(q.ref_month), q.ref_year,
          Quarterly));
    }
    for (std::size_t i = 0; i < rm::swaps.size(); ++i)
      helpers.push_back(ext::make_shared<OISRateHelper>(
          2, Period(rm::swaps[i].tenor_years, Years),
          Handle<Quote>(ext::make_shared<SimpleQuote>(rm::swaps[i].par_rate)), mk.sofr));
    x0 = Eigen::VectorXd::Constant(prob.n_knots(), 0.035);
  }
};

const SquareFixture& fx() {
  static const SquareFixture f;
  return f;
}

}  // namespace

static void BM_CurveBuild_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    auto curve = ext::make_shared<GlobalPWC>(f.today, f.helpers, f.dc);
    curve->enableExtrapolation();
    benchmark::DoNotOptimize(curve->discount(f.far));  // triggers the global bootstrap
  }
}
BENCHMARK(BM_CurveBuild_QuantLib);

static void BM_CurveBuild_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    auto r = cal::calibrate(f.prob, f.x0, /*use_aad=*/true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_CurveBuild_Ours);

BENCHMARK_MAIN();
