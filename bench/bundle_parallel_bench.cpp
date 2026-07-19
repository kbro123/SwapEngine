// Branch-parallel staged bundle calibrate (CLAUDE.md §7b): serial vs thread-per-SCC on a REALISTIC STAR.
// The bundle is the canonical realistic model (tests/reference_bundle.hpp): a SOFR base with the full
// reference structure (6 FOMC meetings + 12x1M then 8x3M futures + par swaps) and K basis curves each
// spread straight off SOFR (12x1M futures + basis swaps). The star's dependency waves are [{SOFR}, {K
// basis}], so wave 1 is K independent, realistically-sized block solves run concurrently. Ours-only probe
// (not a fingerprint gate metric); parallel == serial bit-for-bit is proven in tests/bundle_test.cpp.
#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include "reference_bundle.hpp"
#include "reference_curve.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/parallel/thread_pool.hpp"

namespace cal = swaps::calibration;
namespace rb = swaps::refbuild;

namespace {
// Build the realistic star ONCE (K basis curves off SOFR); benchmark the cold staged solve on it.
struct StarFixture {
  QuantLib::RelinkableHandle<QuantLib::YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::RealisticBundle bundle;
  explicit StarFixture(int k) : bundle(rb::build_realistic_bundle(mk, h, k, rb::BundleTopology::Star)) {}
};
}  // namespace

#define STAR_BENCH(K)                                                                          \
  static void BM_BundleStar##K##_StagedSerial(benchmark::State& s) {                           \
    StarFixture f(K);                                                                          \
    for (auto _ : s) {                                                                         \
      auto r = cal::calibrate_staged(f.bundle.prob, f.bundle.x0, true);                        \
      benchmark::DoNotOptimize(r.x.data());                                                    \
    }                                                                                          \
  }                                                                                            \
  BENCHMARK(BM_BundleStar##K##_StagedSerial)->Unit(benchmark::kMicrosecond);                   \
  static void BM_BundleStar##K##_StagedParallelAsync(benchmark::State& s) {                    \
    StarFixture f(K);                                                                          \
    for (auto _ : s) {                                                                         \
      auto r = cal::calibrate_staged_parallel(f.bundle.prob, f.bundle.x0, true);               \
      benchmark::DoNotOptimize(r.x.data());                                                    \
    }                                                                                          \
  }                                                                                            \
  BENCHMARK(BM_BundleStar##K##_StagedParallelAsync)->Unit(benchmark::kMicrosecond);            \
  static void BM_BundleStar##K##_StagedParallelPool(benchmark::State& s) {                     \
    StarFixture f(K);                                                                          \
    swaps::parallel::ThreadPool pool(K);                                                       \
    for (auto _ : s) {                                                                         \
      auto r = cal::calibrate_staged_parallel(f.bundle.prob, f.bundle.x0, true, &pool);        \
      benchmark::DoNotOptimize(r.x.data());                                                    \
    }                                                                                          \
  }                                                                                            \
  BENCHMARK(BM_BundleStar##K##_StagedParallelPool)->Unit(benchmark::kMicrosecond);

STAR_BENCH(8)   // 8 basis curves off SOFR -> wave 1 is 8 concurrent realistic solves

BENCHMARK_MAIN();
