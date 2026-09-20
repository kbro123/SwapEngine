// E5 taxonomy: T3 cross-path parity (the piecewise-linear W tier vs a fresh engine and vs the shipped router)
// EXPERIMENT (exp/piecewise-linear-w): the piecewise-linear W tier with its ANALYTIC re-take.
//
// The tier keeps rows that read a MonotoneCubic region on the compiled W-cache: W is exact per Hyman branch
// cell, and a cell change is applied as a rank-k update ΔW = B·ΔM (or a full reset W = L + B·M). These tests
// pin it against a FRESH engine (whose first sync is a full reset at x) and against the SHIPPED router.
#include <gtest/gtest.h>

#include <random>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;

// One long-lived engine jumping between random states (0.01 bp .. 50 bp): after every jump its residuals AND
// Jacobian equal a fresh engine's at the same x. Every re-take must have been analytic (no AAD fallback).
TEST(ExpPwl, AnalyticReTakeMatchesAFreshEngineAcrossRandomJumps) {
  for (const auto& s : {swaps::shapes::mixed_scheme(), swaps::shapes::desk_mixed()}) {
    const cal::HybridBundleResidual live(s.prob, /*pwl=*/true);
    ASSERT_TRUE(live.pwl_active()) << s.name;
    std::mt19937_64 g(7);
    std::normal_distribution<double> n;
    const double sizes[] = {1e-6, 1e-5, 1e-4, 1e-3, 5e-3};
    Eigen::VectorXd x = s.x_true;
    for (int step = 0; step < 100; ++step) {
      for (int i = 0; i < x.size(); ++i) x[i] = s.x_true[i] + sizes[step % 5] * n(g);
      const cal::HybridBundleResidual fresh(s.prob, true);
      EXPECT_LT((live.residuals(x) - fresh.residuals(x)).cwiseAbs().maxCoeff(), 1e-14) << s.name << " step " << step;
      const Eigen::MatrixXd Jf = fresh.jacobian(x);
      EXPECT_LT((live.jacobian(x) - Jf).cwiseAbs().maxCoeff(), 1e-12 * Jf.cwiseAbs().maxCoeff()) << s.name << " step " << step;
    }
    EXPECT_GT(live.pwl_rebuilds(), 10) << s.name << ": the jumps must actually change cells";
    EXPECT_EQ(live.pwl_analytic(), live.pwl_rebuilds() - 1) << s.name << ": all but the first sync analytic";
  }
}

// And against the SHIPPED router (value-dependent rows on AAD): the same residuals and Jacobian at a state.
TEST(ExpPwl, MatchesTheShippedRouter) {
  for (const auto& s : {swaps::shapes::mixed_scheme(), swaps::shapes::desk_mixed()}) {
    const cal::HybridBundleResidual base(s.prob, false), pwl(s.prob, true);
    EXPECT_GT(pwl.n_compiled_rows(), base.n_compiled_rows()) << s.name;
    EXPECT_LT((pwl.residuals(s.x_true) - base.residuals(s.x_true)).cwiseAbs().maxCoeff(), 1e-14) << s.name;
    const Eigen::MatrixXd Jb = base.jacobian(s.x_true);
    EXPECT_LT((pwl.jacobian(s.x_true) - Jb).cwiseAbs().maxCoeff(), 1e-12 * Jb.cwiseAbs().maxCoeff()) << s.name;
  }
}

// REGRESSION (found 2026-09-19): a FLAT cold seed is a degenerate cell point (every secant and raw tangent 0).
// An AAD re-take through the filter there follows the `correction != m` VALUE test and takes the NEIGHBOURING
// cell's derivative, disagreeing with the recorded pattern -- every later rank-k update then built on a wrong
// W and the cold calibration "converged" to |r| ~ 7e-5 on a square rung. The reset is now W = L + B·M.
TEST(ExpPwl, AFlatColdSeedCalibratesExactly) {
  const auto s = swaps::shapes::mixed_scheme();
  const cal::HybridBundleResidual live(s.prob, true);
  (void)live.residuals(s.x0);  // first sync AT the degenerate state
  const cal::HybridBundleResidual fresh(s.prob, true);
  EXPECT_LT((live.residuals(s.x_true) - fresh.residuals(s.x_true)).cwiseAbs().maxCoeff(), 1e-14);
  const Eigen::MatrixXd Jf = fresh.jacobian(s.x_true);
  EXPECT_LT((live.jacobian(s.x_true) - Jf).cwiseAbs().maxCoeff(), 1e-12 * Jf.cwiseAbs().maxCoeff());
}
