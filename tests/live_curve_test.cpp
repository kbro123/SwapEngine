// Lock-free publish of the live curve (the async pricer/calibrator split). Engine-only: hammers the
// LiveCurveFeed from one writer + several reader threads and proves a reader NEVER observes a torn
// (half-updated) curve, and always sees a monotone-nondecreasing version. The writer publishes distinct
// CONSTANT vectors (all entries == tick number k), so a clean snapshot has minCoeff == maxCoeff; any tear
// (a mix of two ticks) is caught as minCoeff != maxCoeff.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "swaps/calibration/live_curve.hpp"

namespace cal = swaps::calibration;

TEST(LiveCurve, ConcurrentReadsAreNeverTornAndVersionsAreMonotone) {
  const int nk = 96;      // a realistic knot count; a wide payload widens the tear window if the seqlock is wrong
  const long M = 300000;  // publishes
  cal::LiveCurveFeed feed(nk);

  std::atomic<bool> stop{false};
  std::atomic<long> total_reads{0}, torn{0}, regressions{0};

  auto reader = [&] {
    Eigen::VectorXd out(nk);
    std::uint64_t last_ver = 0;
    long reads = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const std::uint64_t ver = feed.snapshot(out);
      if (out.minCoeff() != out.maxCoeff()) torn.fetch_add(1, std::memory_order_relaxed);  // TORN
      if (ver < last_ver) regressions.fetch_add(1, std::memory_order_relaxed);              // went backwards
      last_ver = ver;
      ++reads;
    }
    total_reads.fetch_add(reads, std::memory_order_relaxed);
  };

  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) readers.emplace_back(reader);

  // Writer: publish distinct constant curves as fast as possible (the calibrator-thread role).
  for (long k = 1; k <= M; ++k) feed.publish(Eigen::VectorXd::Constant(nk, static_cast<double>(k)));
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : readers) t.join();

  std::cout << "  [live-curve] publishes=" << M << " reads=" << total_reads.load()
            << " torn=" << torn.load() << " version-regressions=" << regressions.load()
            << " final-version=" << feed.version() << "\n";
  EXPECT_EQ(torn.load(), 0) << "a reader observed a TORN curve -- the lock-free publish is unsafe";
  EXPECT_EQ(regressions.load(), 0) << "a reader saw the version go backwards -- publish ordering is wrong";
  EXPECT_GT(total_reads.load(), 0) << "readers must have actually run";
  EXPECT_EQ(feed.version(), static_cast<std::uint64_t>(M));
}

TEST(LiveCurve, SnapshotAlwaysEqualsAPublishedCurveUnderChurn) {
  // A reader's snapshot must be EXACTLY some curve the writer published (never a blend). Here the writer
  // cycles a small set of distinct non-constant vectors; the reader asserts every snapshot equals one of
  // them element-for-element.
  const int nk = 32;
  cal::LiveCurveFeed feed(nk);
  std::vector<Eigen::VectorXd> known;
  for (int p = 0; p < 5; ++p) {
    Eigen::VectorXd v(nk);
    for (int i = 0; i < nk; ++i) v[i] = 0.01 * p + 0.001 * i;  // distinct, non-constant
    known.push_back(v);
  }
  std::atomic<bool> stop{false};
  std::atomic<long> mismatches{0}, reads{0};
  std::thread reader([&] {
    Eigen::VectorXd out(nk);
    while (!stop.load(std::memory_order_relaxed)) {
      feed.snapshot(out);
      bool ok = false;
      for (const auto& v : known)
        if ((out - v).cwiseAbs().maxCoeff() == 0.0) { ok = true; break; }
      if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  });
  for (long k = 0; k < 400000; ++k) feed.publish(known[k % known.size()]);
  stop.store(true, std::memory_order_relaxed);
  reader.join();
  std::cout << "  [live-curve] churn reads=" << reads.load() << " mismatches=" << mismatches.load() << "\n";
  EXPECT_EQ(mismatches.load(), 0) << "every snapshot must be exactly one wholly-published curve";
}
