// E5 taxonomy: T6 regression (fails on the reverted bug) | T3 cross-path parity (two engine paths, same inputs)
// STREAMER NEAR-SINGULAR DIRECTION (reproduction, 2026-09-15). On desk_mixed (a MonotoneCubic SOFR long end) a large four-number
// requote -- each instrument moved 25 bp x sin(0.7 i + 0.3) with its band, the ladder's old q_big -- is a pathological market whose
// best fit puts SOFR's 30y knot negative. Streaming back from it, the Hyman filter leaves the 30y knot nearly unidentified: after a
// mid-tick refresh the Jacobian's sigma_min / sigma_max is ~1.7e-5 (cond 5.8e4), above the shared rank threshold (1e-10), so the
// undamped corrector stepped ALONG it -- knot 11 moved +0.16, +0.23, -0.51, +1.8, -61 -- and the tick diverged (the session then
// paid an LM fallback). The streamer's operator now WALKS dropping directions below 1e-4 x sigma_max -- and, because a direction
// that weak can be genuine at the solution, a tick whose operator truncated anything the shared 1e-10 threshold keeps re-converges
// at the shared threshold before it commits (WeakDirection... below: a plain 1e-4 cut left a near-collinear knot 25 % wrong while
// reporting convergence).
// This replays that sequence {big, q0, small, q0} x 3 through the streamer's resync (every band shifted with its target), requires
// every tick to converge, and requires every streamed point to be the requoted problem's optimum: a cold LM polished from it moves
// no knot by more than 1e-5 (measured 1.2e-6 on desk_mixed, 4.6e-6 on desk -- the LM's own stall on the band kinks).
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace px = swaps::pricing;
using swaps::shapes::Shape;

namespace {

// The pathological move: each non-FX row by 25 bp x sin(0.7 i + 0.3) (xccy basis rows a tenth), bands shifted with their targets.
Eigen::VectorXd sine_move(const Shape& s) {
  Eigen::VectorXd q = s.q0;
  for (int i = 0; i < q.size(); ++i) {
    const auto k = s.prob.instruments[static_cast<std::size_t>(i)].quote;
    if (k == cal::QuoteKind::FxForward) continue;
    q[i] += (k == cal::QuoteKind::XccyMtmBasis ? 2.5e-4 : 25e-4) * std::sin(0.7 * i + 0.3);
  }
  return q;
}

cal::BundleProblem requoted(const Shape& s, const Eigen::VectorXd& q) {
  cal::BundleProblem p = s.prob;
  for (int i = 0; i < p.n_residuals(); ++i) {
    auto& in = p.instruments[static_cast<std::size_t>(i)];
    const double d = in.band_upper > in.band_lower ? q[i] - s.q0[i] : 0.0;
    in.market = q[i];
    in.band_lower += d;
    in.band_upper += d;
  }
  return p;
}

struct Outcome {
  int ticks = 0, converged = 0;
  double worst_knot_gap = 0.0;
  std::string first_failure;
};

Outcome replay(const Shape& s) {
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  const Eigen::VectorXd big = sine_move(s);
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  Outcome o;
  const Eigen::VectorXd* seq[] = {&big, &s.q0, &s.q_small, &s.q0};
  for (int rep = 0; rep < 3; ++rep)
    for (int j = 0; j < 4; ++j) {
      const Eigen::VectorXd& q = *seq[j];
      const cal::BundleProblem pq = requoted(s, q);
      eng.set_quotes(pq);
      st.resync(pq, st.current(), q);
      const cal::StreamTick t = st.update(q);
      ++o.ticks;
      if (!t.converged) {
        if (o.first_failure.empty()) o.first_failure = "tick " + std::to_string(o.ticks - 1) + ": " + t.reason();
        continue;
      }
      ++o.converged;
      const Eigen::VectorXd xc = cal::calibrate(pq, st.current()).x;
      o.worst_knot_gap = std::max(o.worst_knot_gap, (st.current() - xc).cwiseAbs().maxCoeff());
    }
  return o;
}

// A square OIS curve whose last two knots (10y, 10+gap y) are pinned by a 10y and a 10+gap y swap: nearly collinear, exact.
cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = 0;
  ins.fwd.discount = 0;
  ins.fixed.discount = 0;
  std::vector<double> ends;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) ends.push_back(u);
  ends.push_back(T);
  double prev = 0.0;
  for (double u : ends) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}
cal::BundleProblem near_collinear(double gap) {
  cal::BundleProblem p;
  const std::vector<double> knots{1, 2, 3, 5, 7, 10, 10.0 + gap};
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, knots)});
  for (double T : knots) p.instruments.push_back(par_swap(T));
  Eigen::VectorXd xt(7);
  xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047, 0.047;
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < 7; ++i) p.instruments[static_cast<std::size_t>(i)].market = r[i];
  return p;
}

// The banded 10x6 bundle of tests/streaming_band_test.cpp: 6 Hermite knots, 10 annual par swaps, the odd ones banded (decay 0.1),
// quotes made inconsistent by `noise_bp` so rows sit on their band edges.
cal::Instrument banded_par_swap(double T, int fc, int disc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = fc;
  ins.fwd.discount = disc;
  ins.fixed.discount = disc;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}

Eigen::VectorXd model_quotes(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = cal::instrument_model_quote<double>(p.instruments[i], cof);
  return q;
}

// The banded, inconsistent 10x6 bundle. `half_bp` = band half-width, `noise_bp` = quote inconsistency.
cal::BundleProblem banded_bundle(double half_bp, double noise_bp, unsigned seed = 7, int burn = 0) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, noise_bp * 1e-4);
  for (int i = 0; i < burn; ++i) (void)noise(rng);
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  Eigen::VectorXd xt(6);
  xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(banded_par_swap(T, 0, 0));
  const Eigen::VectorXd q = model_quotes(p, xt);
  for (int i = 0; i < 10; ++i) {
    auto& ins = p.instruments[i];
    ins.market = q[i] + noise(rng);
    if (i % 2 == 1) {
      ins.band_lower = ins.market - half_bp * 1e-4;
      ins.band_upper = ins.market + half_bp * 1e-4;
      ins.band_decay = 0.1;
    }
  }
  return p;
}


Eigen::VectorXd strip_knots() {
  Eigen::VectorXd x(3);
  x << 0.040, 0.042, 0.044;
  return x;
}

// A square 3-knot bundle (1y / 2y / 3y par swaps) whose 2y row is banded +-5 bp with a tiny decay `d`: inside its band that row's Jacobian
// row is scaled by d, so the Jacobian's weak direction depends on WHERE the curve is (outside the band the row has slope 1).
cal::BundleProblem tiny_decay_strip(double d) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3})});
  for (int T = 1; T <= 3; ++T) p.instruments.push_back(banded_par_swap(T, 0, 0));
  const Eigen::VectorXd q = model_quotes(p, strip_knots());
  for (int i = 0; i < 3; ++i) p.instruments[static_cast<std::size_t>(i)].market = q[i];
  p.instruments[1].band_lower = q[1] - 5e-4;
  p.instruments[1].band_upper = q[1] + 5e-4;
  p.instruments[1].band_decay = d;
  return p;
}

}  // namespace

TEST(StreamingNearSingularRepro, DeskMixedStreamsBackFromAPathologicalRequoteWithoutDiverging) {
  const Outcome o = replay(swaps::shapes::desk_mixed());
  EXPECT_EQ(o.converged, o.ticks) << o.first_failure;
  EXPECT_LT(o.worst_knot_gap, 1e-5) << "a streamed tick is not the requoted problem's optimum";
}

TEST(StreamingNearSingularRepro, ControlDeskStreamsTheSameSequenceToTheSameOptimum) {
  const Outcome o = replay(swaps::shapes::desk());
  EXPECT_EQ(o.converged, o.ticks) << o.first_failure;
  EXPECT_LT(o.worst_knot_gap, 1e-5);
}

// T3: a direction that weak can be GENUINE at the solution. Moving the 10+gap y quote 1 bp moves that knot by tens of percent (gap
// 0.005: 4.70 -> 29.80 %; 0.002: -> 68.06 %), with sigma_min/sigma_max 7.7e-6 / 1.2e-6 -- below the 1e-4 walk threshold. The
// streamed fit must still be the exact solution: the commit rule re-converges at the shared threshold.
TEST(StreamingNearSingularRepro, AGenuinelyWeakDirectionIsStillSolvedExactly) {
  for (double gap : {0.005, 0.002}) {
    const cal::BundleProblem p = near_collinear(gap);
    const Eigen::VectorXd q0 = p.market();
    const Eigen::VectorXd x0 = cal::calibrate(p, Eigen::VectorXd::Constant(7, 0.03)).x;
    Eigen::VectorXd q = q0;
    q[6] += 1e-4;
    cal::BundleProblem pq = p;
    pq.instruments[6].market = q[6];
    const Eigen::VectorXd xc = cal::calibrate(pq, x0).x;
    ASSERT_GT(std::abs(xc[6] - x0[6]), 0.2) << "premise: the 1 bp quote move is a large move along the weak direction (gap " << gap << ")";
    const cal::HybridBundleResidual eng(p);
    cal::StreamingCalibrator<cal::BundleProblem> st(eng, p, x0, q0, {});
    const cal::StreamTick t = st.update(q);
    ASSERT_TRUE(t.converged) << t.reason();
    EXPECT_LT((st.current() - xc).cwiseAbs().maxCoeff(), 1e-9) << "gap " << gap << ": the streamed knot is not the solution";
    EXPECT_LT(eng.residuals_vs(st.current(), q).cwiseAbs().maxCoeff(), 1e-12) << "gap " << gap;
  }
}

// A pinned band row carries kPinWeight (1e3) and sets the operator's largest pivot; a walk cut RELATIVE to it dropped a genuine direction
// of the unpinned operator and forced a full-rank re-anchor (a wasted Jacobian) on an ordinary in-band tick. Every target stays at the
// centre of its band (four-number requotes); quote noise 0.1 bp puts rows on their edges. Found 2026-09-15 (2 spurious re-anchors in 60).
TEST(StreamingNearSingularRepro, APinnedBandRowDoesNotTruncateAGenuineDirection) {
  const cal::BundleProblem p = banded_bundle(0.25, 0.1, 11);
  const Eigen::VectorXd q0 = p.market();
  const Eigen::VectorXd x0 = cal::calibrate(p, Eigen::VectorXd::Constant(6, 0.03)).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, {});
  int failed = 0, refreshes = 0, rescales = 0;
  for (int k = 0; k < 60; ++k) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < 10; ++i) q[i] += 0.3e-4 * std::sin(0.37 * k + 1.3 * i);
    cal::BundleProblem pq = p;
    for (int i = 0; i < 10; ++i) {
      auto& ins = pq.instruments[static_cast<std::size_t>(i)];
      ins.market = q[i];
      if (i % 2 == 1) { ins.band_lower = q[i] - 0.25e-4; ins.band_upper = q[i] + 0.25e-4; }
    }
    sc.resync(pq, sc.current(), q);
    const cal::StreamTick t = sc.update(q);
    failed += !t.converged; refreshes += t.refreshes; rescales += t.rescales;
  }
  ASSERT_GT(rescales, 100) << "premise: the ticks walk band edges (rows re-scaled and pinned)";
  EXPECT_EQ(failed, 0);
  EXPECT_EQ(refreshes, 0) << "a pinned row's weight truncated a genuine direction and forced a full-rank re-anchor";
}

// The COMMIT RULE. The anchor is OUTSIDE the tiny-decay band (knot 2 lifted 30 bp: well conditioned, no weak direction); the solution is
// inside it, where the 2y row's slope is 1e-5. Walking in, the band re-scale re-factorises with a weak direction the anchor did not
// have, so the walk operator drops it. Committing that operator's fixed point was silently wrong (found 2026-09-15): 'converged' with
// knot 2 40 bp from the solution and the 2y quote 5 bp from its target. Re-converging at the shared threshold commits the solution.
TEST(StreamingNearSingularRepro, AWeakDirectionThatAppearsMidWalkIsCommittedExactly) {
  const cal::BundleProblem p = tiny_decay_strip(1e-5);
  const Eigen::VectorXd q = p.market();
  Eigen::VectorXd x0 = strip_knots();
  x0[1] += 30e-4;
  ASSERT_GT(model_quotes(p, x0)[1], p.instruments[1].band_upper) << "premise: the anchor is outside the band";
  Eigen::VectorXd q1 = q;
  q1[0] += 0.5e-4;
  q1[2] -= 0.5e-4;
  cal::BundleProblem p1 = p;
  for (int i = 0; i < 3; ++i) p1.instruments[static_cast<std::size_t>(i)].market = q1[i];
  const Eigen::VectorXd xc = cal::calibrate(p1, strip_knots()).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q, {});
  const cal::StreamTick t = sc.update(q1);
  ASSERT_TRUE(t.converged) << t.reason();
  EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-9) << "the committed knots are not the solution";
  EXPECT_LT(std::abs(model_quotes(p1, sc.current())[1] - q1[1]), 1e-12) << "the 2y quote is not at its target";
}

// A STRUCTURAL weak direction costs nothing. The anchor is inside the tiny-decay band, so the committed state already has the weak
// direction; with refresh_drift forcing a Jacobian refresh every tick, a walk that truncated it re-anchored every tick (18 refreshes in 6
// ticks against 12) -- the USD/EUR xccy bundle's stream tick went 53 -> 106 us that way.
TEST(StreamingNearSingularRepro, AStructuralWeakDirectionCostsNoExtraRefresh) {
  const cal::BundleProblem p = tiny_decay_strip(1e-5);
  const Eigen::VectorXd q = p.market();
  cal::StreamingCalibrator<cal::BundleProblem>::Options o;
  o.breakeven_steps = 64;
  o.refresh_drift = 1e-6;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, strip_knots(), q, o);
  int refreshes = 0;
  for (int k = 0; k < 6; ++k) {
    Eigen::VectorXd q1 = q;
    q1[0] += 0.5e-4 * std::sin(0.9 * k + 0.2);
    q1[1] += 0.1e-4 * std::sin(1.3 * k);
    q1[2] -= 0.5e-4 * std::cos(0.7 * k);
    cal::BundleProblem p1 = p;
    for (int i = 0; i < 3; ++i) p1.instruments[static_cast<std::size_t>(i)].market = q1[i];
    const Eigen::VectorXd xc = cal::calibrate(p1, strip_knots()).x;
    const cal::StreamTick t = sc.update(q1);
    ASSERT_TRUE(t.converged) << "tick " << k << ": " << t.reason();
    EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-9) << "tick " << k;
    refreshes += t.refreshes;
  }
  EXPECT_EQ(refreshes, 12) << "two refreshes per tick (drift + accuracy re-anchor); more means the structural weak direction was truncated";
}
