// Portfolio-analytics baseline: our vectorized CompiledPortfolio reprice vs QuantLib's per-swap
// NPV() loop, both off the SAME calibrated curve (CLAUDE.md §3, portfolio_analytics >= 10x).
//
// Repricing a large book as the curve moves is the real-time workload. QuantLib walks every swap's
// coupons through its object model each reprice; we do DF = exp(-Wx) then gathered elementwise
// coupon math + sparse reductions. To force QuantLib to actually re-price (not return cached NPVs)
// we relink the term-structure handle between two structures each iteration.
//
// Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

constexpr int kBook = 1000;

struct Fixture {
  mutable RelinkableHandle<YieldTermStructure> h;  // relinked during the QuantLib benchmark
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;

  swaps::curve::TwoRegionForwardCurve<double> curve{prob.meeting_times, prob.back_times};
  ext::shared_ptr<swaps::qlx::TwoRegionTermStructure> ts_a, ts_b;

  swaps::portfolio::Portfolio pf;
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> ql_book;
  std::unique_ptr<swaps::portfolio::CompiledPortfolio> cp;

  Fixture() {
    curve.set_forwards(x);
    ts_a = ext::make_shared<swaps::qlx::TwoRegionTermStructure>(mk.today, mk.dc, &curve);
    ts_b = ext::make_shared<swaps::qlx::TwoRegionTermStructure>(mk.today, mk.dc, &curve);
    ts_a->enableExtrapolation();
    ts_b->enableExtrapolation();
    h.linkTo(ts_a);

    for (int b = 0; b < kBook; ++b) {
      const auto& s = rm::swaps[b % rm::swaps.size()];
      const double fixed = s.par_rate + 0.0005 * ((b % 7) - 3);
      const double notl = ((b % 2) ? 1.0 : -1.0) * (1.0 + (b % 5));
      ext::shared_ptr<OvernightIndexedSwap> sw =
          MakeOIS(Period(s.tenor_years, Years), mk.sofr, fixed).withNominal(notl).withDiscountingTermStructure(h);
      ql_book.push_back(sw);
      pf.positions.push_back({swaps::qlx::extract_ois_swap(*sw, mk.today, mk.dc), fixed, notl});
    }
    cp = std::make_unique<swaps::portfolio::CompiledPortfolio>(prob.meeting_times, prob.back_times, pf);
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

static void BM_Portfolio_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  bool flip = false;
  for (auto _ : state) {
    f.h.linkTo(flip ? f.ts_a : f.ts_b);  // invalidate every swap's cached NPV
    flip = !flip;
    double total = 0;
    for (const auto& s : f.ql_book) total += s->NPV();
    benchmark::DoNotOptimize(total);
  }
}
BENCHMARK(BM_Portfolio_QuantLib);

static void BM_Portfolio_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    double total = f.cp->total_npv(f.x);
    benchmark::DoNotOptimize(total);
  }
}
BENCHMARK(BM_Portfolio_Ours);

BENCHMARK_MAIN();
