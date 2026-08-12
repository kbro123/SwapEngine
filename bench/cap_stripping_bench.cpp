// Caplet-vol stripping hot path (R9). Bootstrapping per-caplet vols from a term structure of flat cap vols
// is O(n²) as written: each bucket rebuilds a `whole`-cap vector copy and re-sums the already-stripped
// prefix from scratch. A desk strips a full 30y quarterly cap curve (~120 caplets, ~15 quoted maturities)
// on every surface rebuild. Ours-only, native C++ (never through JSON, per the perf-test rule).
#include <benchmark/benchmark.h>

#include <vector>

#include "swaps/vol/cap_stripping.hpp"

namespace v = swaps::vol;

namespace {
struct CapCurve {
  std::vector<v::CapletLeg> legs;
  std::vector<int> last;
  std::vector<double> flat;
};

// A quarterly cap curve out to `years`, quoted at every `quote_every`-th caplet (a humped flat-vol term
// structure). Discount ~ flat 3% cont-comp; forwards ~ 3% with a gentle slope.
CapCurve make_curve(int years, int quote_every) {
  CapCurve c;
  const int n = years * 4;
  for (int i = 1; i <= n; ++i) {
    const double t = 0.25 * i;
    c.legs.push_back({0.028 + 0.0005 * i / n * 100.0 * 0.0, 0.25, std::exp(-0.03 * t), t - 0.25});
    c.legs.back().forward = 0.03 + 0.0004 * (i - n / 2.0) / n;  // mild slope through ATM
  }
  for (int i = quote_every - 1; i < n; i += quote_every) {
    c.last.push_back(i);
    const double m = 0.25 * (i + 1);
    c.flat.push_back(0.006 + 0.0025 * std::exp(-0.15 * (m - 3.0) * (m - 3.0)));  // humped ~60-85bp
  }
  return c;
}
}  // namespace

static void BM_StripCapletVols(benchmark::State& s) {
  const CapCurve c = make_curve(static_cast<int>(s.range(0)), static_cast<int>(s.range(1)));
  const double K = 0.03;
  for (auto _ : s) {
    auto vols = v::strip_caplet_vols(c.legs, K, c.last, c.flat);
    benchmark::DoNotOptimize(vols.data());
  }
}
// {years, quote_every}: 30y quarterly quoted every year (120 caplets, 30 caps) and a denser 40y case.
BENCHMARK(BM_StripCapletVols)->Args({30, 4})->Args({40, 2})->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
