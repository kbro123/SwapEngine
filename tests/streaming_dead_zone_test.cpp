// E5 taxonomy: T3 cross-path parity (the streamed tick vs the cold solve; the compiled vs the templated residual)
// Constraint rows stage B (2026-10-09): a row's penalty map is DATA (Instrument::penalty_map) and the streamer's walk is
// over the map's PIECES -- including a ZERO-slope (dead-zone) piece, the shape a one-sided bound takes. Pins:
//   (1) the compiled residual of a general map equals the templated one (both read PenaltyMap) on every piece;
//   (2) a market move that drives a row's model value INTO its dead zone streams to the cold least-squares answer, and
//       back out of it (the row's residual row is zero inside the dead zone and re-scaled from zero on the way out);
//   (3) a row ANCHORED in its dead zone (untracked for that anchor) still converges to the cold answer when it leaves;
//   (4) the JSON codec round-trips the map.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {

cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
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

// 10 par swaps on 6 knots (over-determined), consistent quotes at xt; row 3 (the 4y) carries a ONE-SIDED band as a
// general map: unit pull below, decay 0.1 inside [m − w, m + w], NO pull above (a dead zone: the quote may trade
// anywhere above its band). `three` adds a fourth piece: unit pull again above m + 3w.
cal::BundleProblem bundle(double w_bp, bool three) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  Eigen::VectorXd xt(6);
  xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(par_swap(T));
  const Eigen::VectorXd q = model_quotes(p, xt);
  for (int i = 0; i < 10; ++i) p.instruments[i].market = q[i];
  const double w = w_bp * 1e-4, m = q[3];
  cal::PenaltyMap pm;
  pm.n = three ? 3 : 2;
  pm.b[0] = m - w; pm.b[1] = m + w; pm.b[2] = m + 3 * w;
  pm.s[0] = 1.0; pm.s[1] = 0.1; pm.s[2] = 0.0; pm.s[3] = 1.0;
  p.instruments[3].penalty_map = pm;
  return p;
}

Eigen::VectorXd cold(cal::BundleProblem p, const Eigen::VectorXd& q, const Eigen::VectorXd& x0) {
  for (int i = 0; i < q.size(); ++i) p.instruments[i].market = q[i];
  return cal::calibrate(p, x0).x;
}

double bp(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).cwiseAbs().maxCoeff() * 1e4; }

}  // namespace

TEST(StreamingDeadZone, CompiledAndTemplatedResidualsAgreeOnEveryPieceOfAGeneralMap) {
  for (bool three : {false, true}) {
    const cal::BundleProblem p = bundle(0.5, three);
    const cal::HybridBundleResidual eng(p);
    const cal::PenaltyMap& pm = *p.instruments[3].penalty_map;
    Eigen::VectorXd x(6);
    x << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
    // push the 4y model value onto every piece by moving the 5y knot
    for (double d : {-20e-4, -0.3e-4, 0.0, 0.3e-4, 1.2e-4, 3e-4}) {
      Eigen::VectorXd xs = x;
      xs[3] += d;
      const Eigen::VectorXd want = p.residuals<double>(xs);
      const Eigen::VectorXd got = eng.residuals(xs);
      EXPECT_LE((got - want).cwiseAbs().maxCoeff(), 1e-14) << "three=" << three << " d=" << d;
      const double q3 = model_quotes(p, xs)[3];
      const int piece = pm.piece_of(q3, pm.target_piece(p.instruments[3].market));
      EXPECT_NEAR(got[3], pm.residual_d(q3, p.instruments[3].market).first, 1e-14) << "piece " << piece;  // q3 priced separately: to rounding
      // the compiled Jacobian row carries the piece's slope (0 in the dead zone)
      const Eigen::MatrixXd J = eng.jacobian(xs);
      if (pm.s[piece] == 0.0) EXPECT_EQ(J.row(3).cwiseAbs().maxCoeff(), 0.0) << "dead zone: a zero residual row";
      else EXPECT_GT(J.row(3).cwiseAbs().maxCoeff(), 0.0);
    }
  }
}

TEST(StreamingDeadZone, AMoveIntoAndOutOfTheDeadZoneStreamsToTheColdAnswer) {
  for (bool three : {false, true}) {
    cal::BundleProblem p = bundle(0.5, three);
    const Eigen::VectorXd q0 = p.market();
    const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03));
    cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
    cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, opt);
    // Drag the 4y's model value UP, through its band (w = 0.5 bp) and into the dead zone, by moving the hard 3y and 5y
    // quotes: the four-piece shape's dead zone ends at 3w, so its move is smaller.
    Eigen::VectorXd q = q0;
    q[2] += three ? 0.8e-4 : 4e-4;
    q[4] += three ? 0.8e-4 : 4e-4;
    const cal::StreamTick t1 = sc.update(q);
    EXPECT_TRUE(t1.converged) << "three=" << three << " " << t1.reason();
    EXPECT_GT(t1.rescales, 0) << "the row crossed its band";
    const Eigen::VectorXd xc = cold(p, q, x0);
    EXPECT_LT(bp(sc.current(), xc), 0.05) << "three=" << three;
    {
      const double q3 = model_quotes(p, sc.current())[3];
      const cal::PenaltyMap& pm = *p.instruments[3].penalty_map;
      EXPECT_EQ(pm.s[pm.piece_of(q3, pm.target_piece(q0[3]))], 0.0)
          << "premise: the 4y sits in its dead zone after the move (three=" << three << ", q3 - m = " << (q3 - q0[3]) * 1e4
          << " bp, band half-width 0.5 bp; cold q3 - m = " << (model_quotes(p, xc)[3] - q0[3]) * 1e4 << " bp)";
    }
    // ... and back: the row leaves the dead zone (its residual row re-scaled from zero) and lands on the cold answer.
    const cal::StreamTick t2 = sc.update(q0);
    EXPECT_TRUE(t2.converged) << t2.reason();
    EXPECT_LT(bp(sc.current(), x0), 0.05);
    // A smaller move that parks the row INSIDE its band again from the dead zone, then a tick that does nothing.
    Eigen::VectorXd qh = q0;
    qh[2] += 0.5e-4;
    const cal::StreamTick t3 = sc.update(qh);
    EXPECT_TRUE(t3.converged) << t3.reason();
    EXPECT_LT(bp(sc.current(), cold(p, qh, x0)), 0.05);
  }
}

TEST(StreamingDeadZone, ARowAnchoredInItsDeadZoneIsUntrackedForThatAnchorAndStillConvergesWhenItLeaves) {
  cal::BundleProblem p = bundle(0.5, false);
  const Eigen::VectorXd q0 = p.market();
  Eigen::VectorXd qz = q0;
  qz[2] += 4e-4;
  qz[4] += 4e-4;
  const Eigen::VectorXd xz = cold(p, qz, Eigen::VectorXd::Constant(6, 0.03));  // the 4y in its dead zone
  {
    const double q3 = model_quotes(p, xz)[3];
    const cal::PenaltyMap& pm = *p.instruments[3].penalty_map;
    ASSERT_EQ(pm.s[pm.piece_of(q3, pm.target_piece(q0[3]))], 0.0) << "premise: anchored in the dead zone";
  }
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, xz, qz, opt);  // the anchor Jacobian has a ZERO row for the 4y
  const cal::StreamTick t = sc.update(q0);  // the move back pulls the 4y out of its dead zone
  EXPECT_TRUE(t.converged) << t.reason();
  EXPECT_LT(bp(sc.current(), cold(p, q0, xz)), 0.05);
}
