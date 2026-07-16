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

  std::vector<ext::shared_ptr<RateHelper>> helpers;  // QuantLib side
  cal::CalibrationProblem prob;                      // our side
  Eigen::VectorXd x0;

  SquareFixture() {
    auto t = [&](const Date& d) { return dc.yearFraction(today, d); };
    std::vector<double> front, back;

    for (int i = 0; i < 6; ++i) {
      const auto& q = rm::futures_1m[i];
      helpers.push_back(ext::make_shared<SofrFutureRateHelper>(
          Handle<Quote>(ext::make_shared<SimpleQuote>(q.price)), Month(q.ref_month), q.ref_year,
          Monthly));
      const Date s = rb::sofr_start(Month(q.ref_month), q.ref_year, Monthly),
                 e = rb::sofr_end(Month(q.ref_month), q.ref_year, Monthly);
      prob.avg_futs.push_back(
          {swaps::qlx::extract_averaged_future(mk.sofr, s, e, today, dc), 0.0, 1.0 - q.price / 100.0});
      front.push_back(t(e));
    }
    for (int i = 0; i < 8; ++i) {
      const auto& q = rm::futures_3m[i];
      helpers.push_back(ext::make_shared<SofrFutureRateHelper>(
          Handle<Quote>(ext::make_shared<SimpleQuote>(q.price)), Month(q.ref_month), q.ref_year,
          Quarterly));
      const Date s = rb::sofr_start(Month(q.ref_month), q.ref_year, Quarterly),
                 e = rb::sofr_end(Month(q.ref_month), q.ref_year, Quarterly);
      prob.comp_futs.push_back(
          {swaps::qlx::extract_compounded_future(s, e, today, dc, mk.sofr->dayCounter()), 0.0, 1.0 - q.price / 100.0});
      back.push_back(t(e));
    }
    for (std::size_t i = 0; i < rm::swaps.size(); ++i) {
      helpers.push_back(ext::make_shared<OISRateHelper>(
          2, Period(rm::swaps[i].tenor_years, Years),
          Handle<Quote>(ext::make_shared<SimpleQuote>(rm::swaps[i].par_rate)), mk.sofr));
      prob.swaps.push_back(
          {swaps::qlx::extract_ois_swap(*mk.swaps[i], today, dc), rm::swaps[i].par_rate});
      back.push_back(t(mk.swaps[i]->maturityDate()));
    }
    std::sort(front.begin(), front.end());
    std::sort(back.begin(), back.end());
    prob.meeting_times = front;
    prob.back_times = back;
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
