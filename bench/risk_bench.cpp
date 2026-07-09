// Risk-ladder baseline: our analytic bucketed delta (AAD + implicit-function theorem) vs QuantLib
// bump-and-reprice. This is the largest expected win (CLAUDE.md §3, risk_full_jacobian >= 20x).
//
// Both start from an already-calibrated curve and compute the full delta ladder over all 23 market
// quotes for a swap portfolio:
//   QuantLib : bump each quote once, let the curve re-bootstrap, reprice the book, difference.
//              Uses IterativeBootstrap (QuantLib's FASTEST curve) and ONE-SIDED bumps -- both choices
//              favour QuantLib, so the measured speedup is conservative.
//   Ours     : one AAD gradient d(NPV)/dx + the calibration Jacobian J + one small solve.
//
// Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/risk.hpp"
#include "swaps/portfolio/portfolio.hpp"

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

  // ---- QuantLib side: IterativeBootstrap curve + swap book (priced off the same handle) ----
  std::vector<ext::shared_ptr<SimpleQuote>> quotes;
  std::vector<ext::shared_ptr<RateHelper>> helpers = rb::build_square_ql(mk, quotes);
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> book;

  // ---- our side ----
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  swaps::portfolio::Portfolio pf;
  Eigen::VectorXd xstar;

  Fixture() {
    auto curve = ext::make_shared<PiecewiseYieldCurve<ForwardRate, BackwardFlat>>(today, helpers, dc);
    curve->enableExtrapolation();
    h.linkTo(curve);

    for (std::size_t i = 0; i < mk.swaps.size(); ++i) {
      const double fixed = rm::swaps[i].par_rate + 0.005, notl = (i % 2 ? 1.0 : -1.0) * (1.0 + i);
      ext::shared_ptr<OvernightIndexedSwap> s =
          MakeOIS(Period(rm::swaps[i].tenor_years, Years), mk.sofr, fixed)
              .withNominal(notl)
              .withDiscountingTermStructure(h);
      book.push_back(s);
      pf.positions.push_back({swaps::qlx::extract_ois_swap(*s, today, dc), fixed, notl});
    }
    xstar = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  }

  double book_npv() const {
    double v = 0;
    for (const auto& s : book) v += s->NPV();
    return v;
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

static void BM_Risk_QuantLib_Bump(benchmark::State& state) {
  const auto& f = fx();
  const double eps = 1e-6;
  for (auto _ : state) {
    const double base = f.book_npv();
    for (std::size_t j = 0; j < f.quotes.size(); ++j) {
      const double q0 = f.quotes[j]->value();
      f.quotes[j]->setValue(q0 + eps);   // invalidates -> curve re-bootstraps on next NPV()
      const double d = (f.book_npv() - base) / eps;
      benchmark::DoNotOptimize(d);
      f.quotes[j]->setValue(q0);         // restore
    }
  }
}
BENCHMARK(BM_Risk_QuantLib_Bump);

static void BM_Risk_Ours_Analytic(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    Eigen::VectorXd ladder = cal::bucketed_delta(f.prob, f.xstar, f.pf);
    benchmark::DoNotOptimize(ladder.data());
  }
}
BENCHMARK(BM_Risk_Ours_Analytic);

BENCHMARK_MAIN();
