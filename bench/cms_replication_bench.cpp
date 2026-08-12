// CMS static-replication hot-path benchmark (R7). A CMS-heavy book reprices many convexity-adjusted CMS
// forwards; each one runs a Simpson sweep whose per-node cost is dominated by the Hagan G-function
// derivatives (G', G'') — a fan of std::pow calls sharing the same base a = 1 + x/q — plus a Bachelier
// smile eval. This measures cms_replicated_forward at a desk-realistic node count. Ours-only, native C++
// (never through JSON, per the perf-test rule).
#include <benchmark/benchmark.h>

#include <cmath>

#include "swaps/vol/cms_replication.hpp"

namespace v = swaps::vol;

namespace {
// A cheap, smooth normal-vol smile proxy (skewed parabola around the forward) so the integrand exercises
// a non-constant vol_at without dragging a full SABR solve into the micro-benchmark's inner loop.
struct SmileProxy {
  double atm, skew, curv, f0;
  double operator()(double k) const {
    const double m = k - f0;
    return std::max(atm + skew * m + curv * m * m, 1e-6);
  }
};
}  // namespace

static void BM_CmsReplicatedForward(benchmark::State& s) {
  const int steps = static_cast<int>(s.range(0));
  const double F0 = 0.03, expiry = 5.0;
  const v::GFunctionStandard G{1.0, 0.0, 10.0};  // q=1 (annual), no pay-lag, 10y tenor
  const SmileProxy vol{0.008, -0.02, 0.5, F0};   // ~80bp ATM normal vol, mild skew/curvature
  for (auto _ : s) {
    const double cms = v::cms_replicated_forward(F0, expiry, G, vol, 6.0, steps);
    benchmark::DoNotOptimize(cms);
  }
}
BENCHMARK(BM_CmsReplicatedForward)->Arg(200)->Arg(1000)->Unit(benchmark::kMicrosecond);

// The raw G-derivative fan in isolation (the pow-heavy part R7 targets), one G'/G'' pair per node.
static void BM_GFunctionDerivs(benchmark::State& s) {
  const v::GFunctionStandard G{1.0, 0.0, 10.0};
  double acc = 0.0;
  for (auto _ : s) {
    for (int i = 1; i <= 400; ++i) {
      const double x = 0.001 * i;  // 0.1%..40%
      acc += G.d1(x) + G.d2(x);
    }
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_GFunctionDerivs)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
