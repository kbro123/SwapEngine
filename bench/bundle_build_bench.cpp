// Curve-BUNDLE build: our single JOINT LM over all curves at once vs QuantLib building the same curves
// as a sequential multi-curve chain (SOFR self-discounted, then FF/PRIME/PRIME2 each SOFR-discounted),
// PiecewiseYieldCurve + GlobalBootstrap -- the same global-LM algorithm class as our per-curve gate.
//
// Both produce the identical 2- or 4-curve set; we time the whole build. QuantLib does N sequential
// global bootstraps; we do ONE joint AAD-Jacobian LM over the stacked parameter vector.

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/warm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"

using namespace QuantLib;
namespace cv = swaps::curve;
namespace cal = swaps::calibration;

namespace {

using GlobalPWC = PiecewiseYieldCurve<ForwardRate, BackwardFlat, GlobalBootstrap>;   // QuantLib global LM
using IterPWC = PiecewiseYieldCurve<ForwardRate, BackwardFlat>;                       // default IterativeBootstrap

struct BundleFixture {
  int NC;
  Date today = Date(15, July, 2026);
  DayCounter dc = Actual365Fixed();
  Calendar cal = UnitedStates(UnitedStates::GovernmentBond);
  Date far = today + 31 * Years;
  // Production swap grid (exact Periods -- no double->Period rounding): 1 front + 15 back knots/curve.
  std::vector<Period> pil = {3 * Months,  6 * Months,  9 * Months, 1 * Years,  18 * Months, 2 * Years,
                             3 * Years,    4 * Years,   5 * Years,  7 * Years,  10 * Years,  12 * Years,
                             15 * Years,   20 * Years,  25 * Years, 30 * Years};

  std::vector<ext::shared_ptr<OvernightIndex>> idx;
  std::vector<std::vector<ext::shared_ptr<SimpleQuote>>> quote;  // [curve][pillar] outright OIS par
  cal::BundleProblem prob;                                       // our side
  Eigen::VectorXd x0;

  explicit BundleFixture(int nc) : NC(nc) {
    Settings::instance().evaluationDate() = today;

    std::vector<RelinkableHandle<YieldTermStructure>> h(NC);
    idx.resize(NC);
    const char* names[] = {"SOFRx", "FFx", "PRIMEx", "PRIME2x"};
    for (int c = 0; c < NC; ++c) {
      h[c].linkTo(ext::make_shared<FlatForward>(today, 0.04, dc, Continuous));
      idx[c] = ext::make_shared<OvernightIndex>(names[c], 0, USDCurrency(), cal, Actual360(), h[c]);
    }

    std::vector<double> meeting, back;
    for (std::size_t i = 0; i < pil.size(); ++i) {
      const double tt = dc.yearFraction(today, today + pil[i]);
      (i == 0 ? meeting : back).push_back(tt);
    }
    // Desk convention: curve 0 OUTRIGHT, every other curve a SPREAD over the previous (a spread chain).
    prob.curves.resize(NC);
    prob.curves[0] = {meeting, back, -1};
    for (int c = 1; c < NC; ++c) prob.curves[c] = {meeting, back, c - 1};
    const int nk = prob.curves[0].n_knots();

    // x_true: curve 0 forward level (~4%); curves 1+ forward SPREADS to their base (~-8bp, +308bp,
    // +50bp), so the resulting forwards land ~4.00 / 3.92 / 7.00 / 7.50% as before.
    const double base[] = {0.0400, -0.0008, 0.0308, 0.0050};
    Eigen::VectorXd x_true(NC * nk);
    for (int c = 0; c < NC; ++c) {
      const double slope = (c == 0) ? 0.0004 : 0.0001;
      for (int i = 0; i < nk; ++i) x_true[c * nk + i] = base[c] + slope * i;
    }

    // Actual forward curves at x_true (SOFR outright; others base+spread) via the engine's spread-aware
    // handle, exposed to QuantLib to read fair rates.
    auto curve_handles = cal::build_bundle_curves<double>(
        prob.curves, [&](int c, int i) { return x_true[c * nk + i]; });
    std::vector<ext::shared_ptr<YieldTermStructure>> ts(NC);
    for (int c = 0; c < NC; ++c) {
      auto t = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
          today, dc, curve_handles[c].get());
      t->enableExtrapolation();
      ts[c] = t;
      h[c].linkTo(t);
    }

    quote.assign(NC, {});
    for (int c = 0; c < NC; ++c)
      for (const Period& T : pil) {
        auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(T, idx[c], 0.03).withDiscountingTermStructure(h[0]));
        o->deepUpdate();
        quote[c].push_back(ext::make_shared<SimpleQuote>(o->fairRate()));  // outright SOFR-discounted par
      }

    // Our bundle as generic Instruments: curve 0 by single-curve OIS (ParRate), curve c>0 by a basis
    // over c-1 (ParSpread, SOFR-discounted).
    auto par_inst = [&](const OvernightIndexedSwap& o, int fc, int disc, double mkt) {
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::ParRate;
      ins.fwd = {swaps::qlx::extract_float_leg(o.overnightLeg(), today, dc), fc, disc};
      ins.fixed = {swaps::qlx::extract_fixed_leg(o.fixedLeg(), today, dc), disc};
      ins.market = mkt;
      return ins;
    };
    auto basis_inst = [&](const OvernightIndexedSwap& o, int fwd_fc, int bench_fc, int disc, double mkt) {
      const auto fl = swaps::qlx::extract_float_leg(o.overnightLeg(), today, dc);
      const auto fx = swaps::qlx::extract_fixed_leg(o.fixedLeg(), today, dc);
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::ParSpread;
      ins.fwd = {fl, fwd_fc, disc};
      ins.bench = {fl, bench_fc, disc};
      ins.fixed = {fx, disc};
      ins.market = mkt;
      return ins;
    };
    for (std::size_t i = 0; i < pil.size(); ++i) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(pil[i], idx[0], 0.03).withDiscountingTermStructure(h[0]));
      prob.instruments.push_back(par_inst(*o, 0, 0, quote[0][i]->value()));
    }
    for (int c = 1; c < NC; ++c)
      for (std::size_t i = 0; i < pil.size(); ++i) {
        auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(pil[i], idx[c], 0.03).withDiscountingTermStructure(h[0]));
        prob.instruments.push_back(basis_inst(*o, c, c - 1, 0, quote[c - 1][i]->value() - quote[c][i]->value()));
      }
    // Per-curve flat start: curve 0 near its forward level, spread curves near a small spread.
    x0 = Eigen::VectorXd::Zero(prob.n_knots());
    for (int c = 0; c < NC; ++c)
      x0.segment(prob.offset(c), prob.curves[c].n_knots()).setConstant(c == 0 ? 0.04 : 0.005);
  }

  // QuantLib multi-curve chain: SOFR self-disc, then each curve SOFR-discounted; N sequential bootstraps.
  template <class PWC>
  void ql_build() const {
    RelinkableHandle<YieldTermStructure> hS;
    std::vector<ext::shared_ptr<YieldTermStructure>> built;
    for (int c = 0; c < NC; ++c) {
      std::vector<ext::shared_ptr<RateHelper>> hs;
      for (std::size_t i = 0; i < pil.size(); ++i)
        hs.push_back(ext::make_shared<OISRateHelper>(2, pil[i], Handle<Quote>(quote[c][i]), idx[c],
                                                     c == 0 ? Handle<YieldTermStructure>() : hS));
      auto curve = ext::make_shared<PWC>(today, hs, dc);
      curve->enableExtrapolation();
      benchmark::DoNotOptimize(curve->discount(far));  // triggers the bootstrap
      built.push_back(curve);
      if (c == 0) hS.linkTo(curve);  // SOFR is the discount curve for all others
    }
  }
};

const BundleFixture& fx2() { static const BundleFixture f(2); return f; }
const BundleFixture& fx4() { static const BundleFixture f(4); return f; }

// Warm re-calibration on a MINOR market perturbation of the solved 4-curve bundle. Everything
// expensive (the base solve + the base Jacobian factorization inside WarmCalibrator) is set up ONCE;
// the benchmarks time only the per-tick re-cal, contrasted with a full cold LM re-solve.
struct BundleWarmFixture {
  cal::BundleProblem prob = fx4().prob;  // own copy (we perturb its markets for the cold baseline)
  Eigen::VectorXd x_solved = cal::calibrate(prob, fx4().x0, true).x;
  Eigen::VectorXd dq;
  cal::BundleProblem pert = prob;                             // perturbed market (cold-resolve target)
  cal::WarmCalibrator<cal::BundleProblem> wc{prob, x_solved};  // caches J0 + M at the base solution
  BundleWarmFixture() {
    dq = Eigen::VectorXd(prob.n_residuals());
    for (int i = 0; i < dq.size(); ++i) dq[i] = 1e-4 * std::sin(0.7 * i + 0.3);  // ~1bp, residual order
    for (int i = 0; i < static_cast<int>(pert.instruments.size()); ++i) pert.instruments[i].market += dq[i];
  }
};
const BundleWarmFixture& warm4() { static const BundleWarmFixture f; return f; }

}  // namespace

// 4-curve bundle re-cal on a ~1bp tick: full cold LM re-solve vs warm frozen-Jacobian re-cal (exact,
// auto envelope detection) vs the one-matvec linear update. Not part of the fingerprint perf gate
// (an ours-vs-ours warm/cold contrast); measures the microsecond re-cal the streaming path rides.
static void BM_BundleWarm4_ColdLM(benchmark::State& s) {
  const auto& f = warm4();
  for (auto _ : s) {
    auto r = cal::calibrate(f.pert, f.x_solved, true);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleWarm4_ColdLM);
static void BM_BundleWarm4_Warm(benchmark::State& s) {
  const auto& f = warm4();
  for (auto _ : s) {
    auto r = f.wc.recalibrate(f.dq);
    benchmark::DoNotOptimize(r.x.data());
  }
}
BENCHMARK(BM_BundleWarm4_Warm);
static void BM_BundleWarm4_Linear(benchmark::State& s) {
  const auto& f = warm4();
  for (auto _ : s) {
    auto x = f.wc.recalibrate_linear(f.dq);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_BundleWarm4_Linear);

// Four ways per bundle size: QuantLib GlobalBootstrap, QuantLib IterativeBootstrap (sequential 1-D,
// the like-for-like for staged), our joint LM, our staged (auto local/global by dependency SCC).
#define BUNDLE_BENCH(NN, FX)                                                                          \
  static void BM_BundleBuild##NN##_QuantLibGlobal(benchmark::State& s) {                              \
    for (auto _ : s) FX().ql_build<GlobalPWC>();                                                      \
  }                                                                                                   \
  BENCHMARK(BM_BundleBuild##NN##_QuantLibGlobal);                                                     \
  static void BM_BundleBuild##NN##_QuantLibIterative(benchmark::State& s) {                           \
    for (auto _ : s) FX().ql_build<IterPWC>();                                                        \
  }                                                                                                   \
  BENCHMARK(BM_BundleBuild##NN##_QuantLibIterative);                                                  \
  static void BM_BundleBuild##NN##_OursJoint(benchmark::State& s) {                                   \
    const auto& f = FX();                                                                             \
    for (auto _ : s) { auto r = cal::calibrate(f.prob, f.x0, true); benchmark::DoNotOptimize(r.x.data()); } \
  }                                                                                                   \
  BENCHMARK(BM_BundleBuild##NN##_OursJoint);                                                          \
  static void BM_BundleBuild##NN##_OursStaged(benchmark::State& s) {                                  \
    const auto& f = FX();                                                                             \
    for (auto _ : s) { auto r = cal::calibrate_staged(f.prob, f.x0, true); benchmark::DoNotOptimize(r.x.data()); } \
  }                                                                                                   \
  BENCHMARK(BM_BundleBuild##NN##_OursStaged);

BUNDLE_BENCH(2, fx2)
BUNDLE_BENCH(4, fx4)

BENCHMARK_MAIN();
