// Curve-BUNDLE build: our single JOINT LM over all curves at once vs QuantLib building the same curves
// as a sequential multi-curve chain (SOFR self-discounted, then FF/PRIME/PRIME2 each SOFR-discounted),
// PiecewiseYieldCurve + GlobalBootstrap -- the same global-LM algorithm class as our per-curve gate.
//
// Both produce the identical 2- or 4-curve set; we time the whole build. QuantLib does N sequential
// global bootstraps; we do ONE joint AAD-Jacobian LM over the stacked parameter vector.

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/calibration_curve.hpp"
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
    cal::BundleProblem::CurveSpec spec{meeting, back};
    prob.curves.assign(NC, spec);
    const int nk = spec.n_knots();

    const double base[] = {0.0400, 0.0392, 0.0700, 0.0750};
    Eigen::VectorXd x_true(NC * nk);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < nk; ++i) x_true[c * nk + i] = base[c] + 0.0004 * i;

    // Build the OIS at x_true (link handles to the real curves) to read fair rates.
    std::vector<cv::CalibrationCurve<double>> curves;
    curves.reserve(NC);
    for (int c = 0; c < NC; ++c) curves.push_back(cv::make_calibration_curve<double>(meeting, back));
    for (int c = 0; c < NC; ++c) {
      Eigen::VectorXd xi = x_true.segment(c * nk, nk);
      curves[c].set_forwards(xi);
    }
    std::vector<ext::shared_ptr<YieldTermStructure>> ts(NC);
    for (int c = 0; c < NC; ++c) {
      auto t = ext::make_shared<swaps::qlx::CurveTermStructure<cv::CalibrationCurve<double>>>(today, dc, &curves[c]);
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

    // Our bundle: curve 0 by single-curve OIS, curve c>0 by a basis over c-1 (SOFR-discounted).
    for (std::size_t i = 0; i < pil.size(); ++i) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(pil[i], idx[0], 0.03).withDiscountingTermStructure(h[0]));
      prob.swaps.push_back({0, 0, swaps::qlx::extract_ois_swap(*o, today, dc), quote[0][i]->value()});
    }
    for (int c = 1; c < NC; ++c)
      for (std::size_t i = 0; i < pil.size(); ++i) {
        auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(pil[i], idx[c], 0.03).withDiscountingTermStructure(h[0]));
        prob.bases.push_back({c, c - 1, 0, swaps::qlx::extract_ois_swap(*o, today, dc),
                              quote[c - 1][i]->value() - quote[c][i]->value()});
      }
    x0 = Eigen::VectorXd::Constant(prob.n_knots(), 0.05);
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

}  // namespace

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
