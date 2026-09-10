// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference market; the
// E5 taxonomy: T2 calibration (optimum / stationarity / recovery) | T3 cross-path parity (two engine paths, same inputs)
// engine is compared to ITSELF / hand formulas, not to a QuantLib number). DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Stage-2 streaming gate. The EXACT (frozen-Newton) streaming path is what a live pricer runs, so it
// must, on EVERY tick:
//   (a) ROUND-TRIP: reprice the calibration instruments back to the input quotes -- ||model_rates(x) - q||
//       at the Newton tolerance, so we never show an arbitrageable (stale) price;
//   (b) EXACTNESS: land on an independent from-scratch exact solve;
//   (c) reuse the cached Jacobian -- recomputing it only on genuine staleness, never per envelope crossing.
//
// The square problem is exactly solvable, and the market here is generated from a real curve, so the
// round-trip residual is genuinely ~0 (this is NOT the over-determined case where residuals are non-zero
// by construction -- see calibration_test for that).

#include <gtest/gtest.h>

#include <cstdlib>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/parallel/live_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

struct Streaming : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  cal::CompiledResidual cr{prob};
  Eigen::VectorXd x0, q0;

  void SetUp() override {
    x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
    q0 = cr.model_rates(x0);
  }

  // Independent exact solve at market q (full Newton with a fresh AAD Jacobian each step).
  Eigen::VectorXd exact(const Eigen::VectorXd& q, Eigen::VectorXd x) const {
    for (int k = 0; k < 12; ++k) {
      const Eigen::VectorXd r = cr.model_rates(x) - q;
      const Eigen::MatrixXd J = cal::aad_jacobian(prob, x);
      const Eigen::VectorXd dx = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(r);
      x -= dx;
      if (dx.cwiseAbs().maxCoeff() < 1e-13) break;
    }
    return x;
  }
};

TEST_F(Streaming, ExactPathRoundTripsAndMatchesExactSolve) {
  cal::StreamingCalibrator<>::Options opt;  // exact=true by default
  cal::StreamingCalibrator sc(prob, x0, q0, opt);

  double worst_rt = 0, worst_ex = 0;
  // 400 chained correlated moves ranging up to ~25bp from base (level sweep + twist).
  for (int t = 1; t <= 400; ++t) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i)
      q[i] += 25e-4 * std::sin(0.09 * t) * std::sin(0.6 * i + 1.0) +
              10e-4 * std::sin(0.05 * t + 1.0) * (i / double(q.size()) - 0.5);
    sc.update(q);
    const Eigen::VectorXd& x = sc.current();
    worst_rt = std::max(worst_rt, (cr.model_rates(x) - q).cwiseAbs().maxCoeff());
    worst_ex = std::max(worst_ex, (x - exact(q, x)).cwiseAbs().maxCoeff());
  }
  std::cout << "  [streaming exact] worst round-trip=" << worst_rt << " worst vs exact=" << worst_ex
            << " recalcs=" << sc.refresh_count() - 1 << "\n";
  EXPECT_LT(worst_rt, 1e-8) << "every tick must reprice the instruments back to the input quotes";
  EXPECT_LT(worst_ex, 1e-8) << "streamed curve must equal an independent exact solve";
}

TEST_F(Streaming, AsyncPricerThreadPricesLatestCurveLockFree) {
  // The pricing branch: a CALIBRATOR thread streams solutions into a LiveCurveFeed while a separate
  // PRICER thread continuously snapshots the feed lock-free and prices off it (model_rates = the par
  // rates). The pricer must, on every read, get EXACTLY the curve published at that version -- never a
  // torn/blended one -- proving the async split is correct with the real streaming engine.
  const int M = 400;
  std::vector<Eigen::VectorXd> published(static_cast<std::size_t>(M) + 2);
  swaps::parallel::LiveCurveFeed feed(prob.n_knots());
  std::atomic<bool> done{false};
  std::atomic<long> checks{0}, bad{0}, priced{0};

  std::thread pricer([&] {
    cal::CompiledResidual pricer_cr{prob};  // OWN scratch => safe to run concurrently with the calibrator
    Eigen::VectorXd out(prob.n_knots());
    while (!done.load(std::memory_order_acquire)) {
      const std::uint64_t ver = feed.snapshot(out);
      if (ver == 0) continue;                          // nothing published yet
      const Eigen::VectorXd rates = pricer_cr.model_rates(out);  // REAL pricing work off the snapshot
      const Eigen::VectorXd& px = published[ver];
      // The snapshot must be exactly the curve published at this version (no tear), and pricing it must
      // reproduce a synchronous reprice of that same curve.
      if (px.size() != out.size() || (out - px).cwiseAbs().maxCoeff() != 0.0 ||
          (rates - pricer_cr.model_rates(px)).cwiseAbs().maxCoeff() != 0.0)
        bad.fetch_add(1, std::memory_order_relaxed);
      priced.fetch_add(1, std::memory_order_relaxed);
      checks.fetch_add(1, std::memory_order_relaxed);
    }
  });

  cal::StreamingCalibrator<> sc(prob, x0, q0, {});
  for (int t = 1; t <= M; ++t) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i)
      q[i] += 20e-4 * std::sin(0.08 * t) * std::sin(0.5 * i + 1.0);
    sc.update(q);
    published[feed.version() + 1] = sc.current();  // written BEFORE publish -> visible via the release
    feed.publish(sc.current());
  }
  done.store(true, std::memory_order_release);
  pricer.join();

  std::cout << "  [async-pricer] published=" << M << " priced=" << priced.load()
            << " bad=" << bad.load() << "\n";
  EXPECT_EQ(bad.load(), 0) << "the pricer must always price EXACTLY the curve published at that version";
  // Scheduler-dependent (the pricer thread may never be scheduled under a loaded ctest -j); the exactness
  // assertion above is the contract. Timing/scheduling claims run only with SWAPS_TIMING_ASSERTS=1 (nightly).
  if (std::getenv("SWAPS_TIMING_ASSERTS"))
    EXPECT_GT(priced.load(), 0) << "the pricer thread must have actually priced";
  // The last curve the feed published IS the calibrator's final solution (x vs x, bit-identical).
  EXPECT_EQ((sc.current() - published[M]).cwiseAbs().maxCoeff(), 0.0);
}

TEST_F(Streaming, PrefetchIsExactMatchesSyncAndFires) {
  // The speculative background Jacobian must (a) keep every tick EXACT (round-trip to the quotes),
  // (b) converge to the SAME solution as the synchronous path (frozen-Newton's fixed point is r=0 for
  // any invertible M), and (c) actually SERVE refreshes from the worker on a feed that drifts enough.
  cal::StreamingCalibrator<>::Options popt;
  popt.prefetch = true;
  popt.prefetch_drift = 3e-4;
  cal::StreamingCalibrator scp(prob, x0, q0, popt);
  cal::StreamingCalibrator scs(prob, x0, q0, cal::StreamingCalibrator<>::Options{});  // sync baseline

  double worst_rt = 0, worst_diff = 0;
  for (int t = 1; t <= 500; ++t) {  // a trending feed with large swings -> crosses the staleness envelope
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i)
      q[i] += 60e-4 * std::sin(0.04 * t) + 20e-4 * std::sin(0.11 * t + 0.5 * i);
    scp.update(q);
    scs.update(q);
    worst_rt = std::max(worst_rt, (cr.model_rates(scp.current()) - q).cwiseAbs().maxCoeff());
    worst_diff = std::max(worst_diff, (scp.current() - scs.current()).cwiseAbs().maxCoeff());
  }
  std::cout << "  [streaming prefetch] worst round-trip=" << worst_rt << " worst |pref - sync|=" << worst_diff
            << " prefetch_hits=" << scp.prefetch_hits() << " refreshes=" << scp.refresh_count() - 1 << "\n";
  EXPECT_LT(worst_rt, 1e-8) << "prefetch path must still reprice the instruments exactly every tick";
  EXPECT_LT(worst_diff, 1e-7) << "prefetch and synchronous paths converge to the same exact solution";
  // Scheduler-dependent (worker-thread timing); timing claims live in bench/ (PRINCIPLES.md P6/P9).
  if (std::getenv("SWAPS_TIMING_ASSERTS"))
    EXPECT_GT(scp.prefetch_hits(), 0) << "the background Jacobian must have served at least one refresh";
}

TEST_F(Streaming, SubBpMovesReuseTheCachedJacobian) {
  cal::StreamingCalibrator<>::Options opt;
  cal::StreamingCalibrator sc(prob, x0, q0, opt);
  for (int t = 1; t <= 200; ++t) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i) q[i] += 0.5e-4 * std::sin(0.3 * t + 0.4 * i);
    sc.update(q);
    EXPECT_LT((cr.model_rates(sc.current()) - q).cwiseAbs().maxCoeff(), 1e-8);
  }
  EXPECT_EQ(sc.refresh_count() - 1, 0) << "sub-bp moves must ride the cached Jacobian with zero recalcs";
}
