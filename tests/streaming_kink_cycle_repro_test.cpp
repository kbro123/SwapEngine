// E5 taxonomy: T6 regression (fails on the reverted bug) | T3 cross-path parity (two engine paths, same inputs)
// FLK2 REPRODUCTION (2026-09-14). ShapeLadder.EveryRungConvergesOnTheGateTicks / AMixedBundleStreamsToTheSameAnswerAsAColdSolve
// failed under ctest -j 8 with "desk_mixed: refresh cap hit after 14 steps". Cause, traced: on the tick after the 25 bp move the
// frozen-Newton corrector lands on a 2-CYCLE across a kink of the SOFR long end's MonotoneCubic (Hyman filter) region -- knot 11
// alternates 0.0341513 / 0.0342819, dx = -/+1.306e-4, |r|inf 3.32e-6 unchanged -- and a Jacobian refresh at either state
// reproduces it, so the tick burns max_refresh (6) and FAILS. Whether the tick lands on the cycle depends on WHEN the adaptive
// stall refreshes, i.e. on StreamingCalibrator::breakeven_steps_, a construction-time WALL-CLOCK ratio: desk_mixed failed at a
// break-even <= 6.5 and converged at >= 6.75, and measured ~7.1 on a quiet machine -- load pushed it under the line.
// (2026-09-14, K5': the ladder's moves became four-number requotes -- bands move with their targets.)
// This pins the break-even (Options::breakeven_steps) so the failure is deterministic, replays the ladder test's tick sequence on
// the desk_mixed rung with the session's streamer recipe (hybrid engine, anchored at the mids, default Options), and requires
// every tick to converge at any break-even and to the same state.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;

namespace {

struct StreamRun {
  int failed = 0;
  int max_refreshes = 0;
  double breakeven = 0.0;
  Eigen::VectorXd x;
};

const swaps::shapes::Shape& desk_mixed() {
  static const swaps::shapes::Shape s = swaps::shapes::desk_mixed();
  return s;
}

// The ladder's ORIGINAL 25 bp move (each row 25 bp x sin(0.7 i + 0.3), xccy basis rows a tenth, FX forwards unchanged): the tick
// sequence FLK2 was found on. The fixture's q_big became a realistic parallel-plus-tilt move on 2026-09-15; the 2-cycle stays pinned
// on the move that produced it.
const Eigen::VectorXd& legacy_big() {
  static const Eigen::VectorXd q = [] {
    const swaps::shapes::Shape& s = desk_mixed();
    Eigen::VectorXd v = s.q0;
    for (int i = 0; i < v.size(); ++i) {
      const auto k = s.prob.instruments[static_cast<std::size_t>(i)].quote;
      if (k == cal::QuoteKind::FxForward) continue;
      v[i] += (k == cal::QuoteKind::XccyMtmBasis ? 2.5e-4 : 25e-4) * std::sin(0.7 * i + 0.3);
    }
    return v;
  }();
  return q;
}

const Eigen::VectorXd& calibrated() {
  static const Eigen::VectorXd x = cal::calibrate(desk_mixed().prob, desk_mixed().x0).x;
  return x;
}

StreamRun stream(double breakeven) {
  const swaps::shapes::Shape& s = desk_mixed();
  cal::HybridBundleResidual eng(s.prob);  // non-const: a requote writes its bands
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  opt.breakeven_steps = breakeven;
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, calibrated(), s.prob.market(), opt);
  StreamRun r;
  r.breakeven = st.breakeven_steps();
  for (int rep = 0; rep < 3; ++rep)
    for (const Eigen::VectorXd* qv : {&legacy_big(), &s.q0, &s.q_small, &s.q0}) {
      // K5': each move is a four-number requote -- the bands move with their targets, on the engine and the streamer.
      const swaps::shapes::Shape::Requote rq = s.requote(*qv);
      for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, rq.target[i], rq.lower[i], rq.upper[i], rq.decay[i]);
      if (!st.set_bands(rq.lower, rq.upper, rq.decay)) throw std::logic_error("a requote of the ladder never changes which rows are banded");
      const cal::StreamTick t = st.update(rq.target);
      if (!t.converged) ++r.failed;
      r.max_refreshes = std::max(r.max_refreshes, t.refreshes);
    }
  r.x = st.current();
  return r;
}

}  // namespace

TEST(StreamingKinkCycleRepro, DeskMixedConvergesEveryTickAtABreakEvenOfSixAndAHalf) {
  const StreamRun r = stream(6.5);
  ASSERT_EQ(r.breakeven, 6.5) << "premise: the pinned break-even is used, not the wall-clock measurement";
  EXPECT_EQ(r.failed, 0) << "ticks that hit the refresh cap on a MonotoneCubic kink 2-cycle";
  EXPECT_LT(r.max_refreshes, 6) << "a tick used the whole refresh budget";
}

TEST(StreamingKinkCycleRepro, ControlABreakEvenOfEightConverges) {
  const StreamRun r = stream(8.0);
  ASSERT_EQ(r.breakeven, 8.0) << "premise";
  EXPECT_EQ(r.failed, 0);
}

// T3: the break-even is a speed heuristic; the committed state is the fixed point whatever the refresh schedule. step_tol is
// 1e-9 (predictive convergence commits within ~step_tol/10), so two schedules agree well inside 1e-8.
TEST(StreamingKinkCycleRepro, TheStreamedStateDoesNotDependOnTheBreakEven) {
  const StreamRun lo = stream(2.0), mid = stream(6.5), hi = stream(64.0);
  ASSERT_EQ(lo.failed + mid.failed + hi.failed, 0) << "every schedule must converge every tick";
  EXPECT_LT((lo.x - hi.x).cwiseAbs().maxCoeff(), 1e-8);
  EXPECT_LT((mid.x - hi.x).cwiseAbs().maxCoeff(), 1e-8);
}
