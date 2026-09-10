// Streaming through BAND-EDGE crossings: the frozen-Newton streamer must track the cold least-squares
// optimum on a banded, over-determined bundle -- the case where the old Gaussian band ramp produced (a) two
// stationary fits 110 bp apart in a knot (seed-dependent) and (b) streamed curves up to 147 bp from the cold
// answer with no refresh triggered. With the Huber band (problem.hpp) the objective has one minimum and the
// Jacobian is piecewise constant, so a crossing is a cheap frozen-row re-scale (StreamTick::rescales), not a
// moving target.
//
// Fixture: 6 Hermite knots, 10 annual par swaps (1y..10y), 5 of them banded, quotes made mutually
// INCONSISTENT by noise (the reason bands exist: overlapping instruments that cannot all be hit exactly).
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {

cal::Instrument par_swap(double T, int fc, int disc) {
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
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(par_swap(T, 0, 0));
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

// Cold least-squares at market q, optionally regularised, from seed x0.
Eigen::VectorXd cold(cal::BundleProblem p, const Eigen::VectorXd& q, const Eigen::VectorXd& x0,
                     const Eigen::MatrixXd* R) {
  for (int i = 0; i < q.size(); ++i) p.instruments[i].market = q[i];
  if (!R) return cal::calibrate(p, x0).x;
  const cal::HybridBundleResidual eng(p);
  const cal::RegularizedEngine<cal::HybridBundleResidual> composed(eng, *R);
  return cal::calibrate_with(composed, p.n_knots(), p.n_residuals() + static_cast<int>(R->rows()), x0).x;
}

Eigen::VectorXd drifted(const Eigen::VectorXd& q0, double bp) {
  Eigen::VectorXd q = q0;
  for (int i = 0; i < q.size(); ++i) q[i] += bp * 1e-4 * (1.0 + 0.5 * std::sin(0.7 * i));  // level + twist
  return q;
}

}  // namespace

// One minimum: cold LM from four seeds lands on the same curve (the old ramp gave two, 110 bp apart, on this
// exact noise realisation -- wide bands, 3 bp inconsistency).
TEST(StreamingBand, ColdLeastSquaresHasOneMinimum) {
  for (const auto& [half, noise, burn] : {std::tuple{2.0, 3.0, 10}, std::tuple{0.25, 0.5, 0}}) {
    const cal::BundleProblem p = banded_bundle(half, noise, 7, burn);
    const Eigen::VectorXd q = drifted(p.market(), 5.0);
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(6, 1e9), hi = Eigen::VectorXd::Constant(6, -1e9);
    for (double seed : {0.02, 0.03, 0.05, 0.08}) {
      const Eigen::VectorXd x = cold(p, q, Eigen::VectorXd::Constant(6, seed), nullptr);
      lo = lo.cwiseMin(x);
      hi = hi.cwiseMax(x);
    }
    EXPECT_LT((hi - lo).maxCoeff() * 1e4, 0.01) << "band " << half << "bp noise " << noise << "bp";
  }
}

// Objective 0.5·|r|² (+ 0.5·|Rx|²) at x for market q -- the quantity both solvers minimise.
double objective(cal::BundleProblem p, const Eigen::VectorXd& q, const Eigen::VectorXd& x, const Eigen::MatrixXd* R) {
  for (int i = 0; i < q.size(); ++i) p.instruments[i].market = q[i];
  const cal::HybridBundleResidual eng(p);
  double f = 0.5 * eng.residuals_vs(x, q).squaredNorm();
  if (R) f += 0.5 * (*R * x).squaredNorm();
  return f;
}

// The streamer tracks the least-squares optimum through band-edge crossings, unregularised and with the
// web's light tension penalty, at narrow (±0.25 bp) and wide (±2 bp) bands, up to 60 bp of drift: every
// tick converged, and its point is an optimum -- a cold LM re-seeded AT the streamed point cannot improve
// the objective by more than 1e-6 relative nor move away by more than 0.02 bp, and the streamed objective
// is no worse than the cold-from-anchor LM's. (LM itself stalls slightly short of stationarity on the
// kinks of the piecewise-quadratic band objective, so it is a bound, not a 1e-9 reference.)
TEST(StreamingBand, TracksLeastSquaresOptimumThroughEdgeCrossings) {
  for (const auto& [half, noise, burn] : {std::tuple{0.25, 0.5, 0}, std::tuple{2.0, 3.0, 10}, std::tuple{0.25, 3.0, 0}}) {
    for (double lambda : {0.0, 0.02}) {
      const cal::BundleProblem p = banded_bundle(half, noise, 7, burn);
      const Eigen::VectorXd q0 = p.market();
      Eigen::MatrixXd R;
      const Eigen::MatrixXd* Rp = nullptr;
      if (lambda > 0.0) {
        R = cal::tension_energy_operator(p, lambda, 0.0, {0});
        Rp = &R;
      }
      const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), Rp);
      cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
      if (Rp) opt.regularizer = R;
      cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, opt);
      for (double bp : {2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0}) {
        const Eigen::VectorXd q = drifted(q0, bp);
        const cal::StreamTick t = sc.update(q);
        const std::string tag = "band " + std::to_string(half) + " noise " + std::to_string(noise) + " lambda " +
                                std::to_string(lambda) + " drift " + std::to_string(bp) + " steps " +
                                std::to_string(t.newton_steps) + " refreshes " + std::to_string(t.refreshes);
        EXPECT_TRUE(t.converged) << tag;
        const double f_stream = objective(p, q, sc.current(), Rp);
        const double f_cold = objective(p, q, cold(p, q, x0, Rp), Rp);
        const Eigen::VectorXd x_polish = cold(p, q, sc.current(), Rp);
        const double f_polish = objective(p, q, x_polish, Rp);
        EXPECT_LE(f_stream, f_cold * (1.0 + 1e-6)) << tag;
        EXPECT_LE(f_stream, f_polish * (1.0 + 1e-6)) << tag;
        EXPECT_LT((sc.current() - x_polish).cwiseAbs().maxCoeff() * 1e4, 0.02) << tag;
      }
    }
  }
}

// A crossing is handled by the cheap re-scale, not a Jacobian refresh: drive a banded quote from inside its
// band to outside and back; the tick reports rescales > 0 and refreshes == 0.
TEST(StreamingBand, EdgeCrossingIsARescaleNotARefresh) {
  cal::BundleProblem p = banded_bundle(0.25, 0.0);  // consistent quotes: everything starts at its mid
  const Eigen::VectorXd q0 = p.market();
  const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), nullptr);
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, opt);
  // Move ONE hard-pinned neighbour by 3 bp: the banded 4y swap's model quote is dragged out of its ±0.25 bp
  // band (its market is unchanged), so its row must switch from decay-slope to unit-slope.
  Eigen::VectorXd q = q0;
  q[2] += 3e-4;  // the 3y swap
  const cal::StreamTick t1 = sc.update(q);
  EXPECT_TRUE(t1.converged);
  EXPECT_GT(t1.rescales, 0);
  EXPECT_EQ(t1.refreshes, 0);
  EXPECT_LT((sc.current() - cold(p, q, x0, nullptr)).cwiseAbs().maxCoeff() * 1e4, 0.05);
  // Back inside: another re-scale, still exact.
  const cal::StreamTick t2 = sc.update(q0);
  EXPECT_TRUE(t2.converged);
  EXPECT_GT(t2.rescales, 0);
  EXPECT_LT((sc.current() - x0).cwiseAbs().maxCoeff() * 1e4, 0.05);
}

// The same crossing with refreshes FORBIDDEN (max_refresh = 0) and the stall cap out of reach: a wrong
// re-scale can no longer be rescued by a full Jacobian refresh (E5.2 2026-09-10: the 2026-09-08 audit's
// mutation M2 -- the re-scale's anchor normalisation dropped -- was caught ONLY by the `refreshes == 0`
// counter above; here it would fail the accuracy assertion itself).
TEST(StreamingBand, EdgeCrossingIsExactWhenRefreshesAreForbidden) {
  cal::BundleProblem p = banded_bundle(0.25, 0.0);
  const Eigen::VectorXd q0 = p.market();
  const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), nullptr);
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  opt.max_refresh = 0;
  opt.max_frozen = 1000;
  opt.adaptive_stall = false;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, opt);
  const Eigen::MatrixXd M0 = sc.sensitivity();
  Eigen::VectorXd q = q0;
  q[2] += 3e-4;
  const cal::StreamTick t1 = sc.update(q);
  EXPECT_TRUE(t1.converged) << t1.reason();
  EXPECT_GT(t1.rescales, 0);
  EXPECT_EQ(t1.refreshes, 0);
  EXPECT_GT((sc.sensitivity() - M0).cwiseAbs().maxCoeff(), 0.0) << "the re-scale must change the operator";
  EXPECT_LT((sc.current() - cold(p, q, x0, nullptr)).cwiseAbs().maxCoeff() * 1e4, 0.05);
  const cal::StreamTick t2 = sc.update(q0);
  EXPECT_TRUE(t2.converged) << t2.reason();
  EXPECT_EQ(t2.refreshes, 0);
  EXPECT_LT((sc.current() - x0).cwiseAbs().maxCoeff() * 1e4, 0.05);
}

// ---- C2 (2026-09-10): the active set on sub-bp moves and on a quote that lives on its edge -------------------
// Before: a 0.6 bp move at decay 0.1 burned 54 steps / 6 refreshes / 21 re-scales and gave up 470 % above the
// optimum (a pin released at s = decay - 9e-4 could never be re-pinned and cycled); a target oscillating
// 0.05 bp either side of its upper edge cost 673 re-scales + 96 refreshes with 16 failed ticks over 200 ticks.
// Now pins are stiff equality rows with the multiplier read off the converged gap, releases are budgeted, and
// the re-scale budget ends the tick honestly instead of switching the breakpoint search off.
namespace {
// First-order optimality in EVERY direction: F(x + d) >= F(x) for 240 random d at |d|_inf = 1e-7. Valid AT a
// kink, where an LM polish is not (LM cannot step along a kink). Second-order terms are ~1e-14 relative.
double worst_descent(const cal::BundleProblem& p, const Eigen::VectorXd& q, const Eigen::VectorXd& x) {
  std::mt19937 rng(11);
  std::normal_distribution<double> g;
  const double f0 = objective(p, q, x, nullptr);
  double worst = 0.0;
  for (int k = 0; k < 240; ++k) {
    Eigen::VectorXd d(x.size());
    for (int i = 0; i < d.size(); ++i) d[i] = g(rng);
    d *= 1e-7 / d.cwiseAbs().maxCoeff();
    worst = std::min(worst, (objective(p, q, x + d, nullptr) - f0) / f0);
  }
  return worst;
}
}  // namespace

TEST(StreamingBand, SubBpMovesAtSmallDecayConvergeToAFirstOrderOptimum) {
  for (double decay : {0.1, 0.5}) {
    cal::BundleProblem p = banded_bundle(0.25, 0.0);  // consistent quotes: everything starts at its mid
    for (int i = 1; i < 10; i += 2) p.instruments[i].band_decay = decay;
    const Eigen::VectorXd q0 = p.market();
    const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), nullptr);
    cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, {});
    for (double bp : {0.2, 0.6, 3.0, 12.0}) {
      Eigen::VectorXd q = q0;
      for (int i = 0; i < 10; ++i) q[i] += bp * 1e-4 * (i % 2 == 1 ? 1.0 : -0.3);
      const cal::StreamTick t = sc.update(q);
      const std::string tag = "decay " + std::to_string(decay) + " move " + std::to_string(bp) + " bp: " + t.reason() +
                              " steps " + std::to_string(t.newton_steps) + " rescales " + std::to_string(t.rescales);
      EXPECT_TRUE(t.converged) << tag;
      EXPECT_LE(t.newton_steps, 40) << tag;
      if (bp * 1e-4 < 10e-4)  // below Options::refresh_drift: a crossing is a re-scale, never a Jacobian refresh
        EXPECT_EQ(t.refreshes, 0) << tag;
      const double wd = worst_descent(p, q, sc.current());
      EXPECT_GT(wd, -1e-7) << tag << ": a direction of descent exists (relative " << wd << ")";
      const Eigen::VectorXd xp = cold(p, q, sc.current(), nullptr);
      EXPECT_LE(objective(p, q, sc.current(), nullptr), objective(p, q, xp, nullptr) * (1.0 + 1e-6)) << tag;
      if (bp * 1e-4 <= 0.25e-4)  // the market still inside its band: the objective is convex, cold LM is a bound
        EXPECT_LE(objective(p, q, sc.current(), nullptr), objective(p, q, cold(p, q, x0, nullptr), nullptr) * (1.0 + 1e-6)) << tag;
      std::cout << "  [band] " << tag << " worst_descent " << wd << "\n";
      const cal::StreamTick back = sc.update(q0);
      EXPECT_TRUE(back.converged) << tag << " (back to q0: " << back.reason() << ")";
    }
  }
}

TEST(StreamingBand, AQuoteOscillatingAroundItsEdgeConvergesEveryTickWithBoundedRescales) {
  cal::BundleProblem p = banded_bundle(0.25, 0.0);  // decay 0.1
  const Eigen::VectorXd q0 = p.market();
  const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), nullptr);
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, x0, q0, {});
  int rescales = 0, refreshes = 0, failed = 0, worst_steps = 0;
  double worst_wd = 0.0;
  Eigen::VectorXd q = q0;
  for (int k = 0; k < 200; ++k) {
    q = q0;
    q[3] = p.instruments[3].band_upper + (k % 2 ? +0.05e-4 : -0.05e-4) * (1.0 + 0.5 * std::sin(0.3 * k));
    q[2] = q0[2] + 0.5e-4 * std::sin(0.1 * k);
    const cal::StreamTick t = sc.update(q);
    rescales += t.rescales; refreshes += t.refreshes; failed += !t.converged; worst_steps = std::max(worst_steps, t.newton_steps);
    if (k % 40 == 39) worst_wd = std::min(worst_wd, worst_descent(p, q, sc.current()));
  }
  std::cout << "  [band] 200 edge-oscillation ticks: rescales " << rescales << " refreshes " << refreshes << " failed " << failed
            << " worst steps " << worst_steps << " worst_descent " << worst_wd << "\n";
  EXPECT_EQ(failed, 0);
  EXPECT_EQ(refreshes, 0);
  EXPECT_LE(rescales, 240);   // measured 151 (was 673 with 16 failures)
  EXPECT_LE(worst_steps, 12);  // measured 9 (was 56)
  EXPECT_GT(worst_wd, -1e-7);
}

// A band re-scale updates the frozen operator in place (rank-one Sherman–Morrison on (JᵀJ+RᵀR)⁺, E3-C7,
// 2026-09-10) instead of re-factorising. Reference: the SAME calibrator with Options::rescale_update off, which
// re-factorises the identically re-scaled frozen Jacobian on every re-scale; the two operators must agree to
// rounding after every tick of a crossing sequence, pins and releases included.
TEST(StreamingBand, RescaledOperatorEqualsARefactorisation) {
  for (double lambda : {0.0, 0.02}) {
    for (double decay : {0.1, 0.5}) {
      cal::BundleProblem p = banded_bundle(0.25, 2.0);  // inconsistent quotes: pins and releases happen
      for (int i = 1; i < 10; i += 2) p.instruments[i].band_decay = decay;
      const Eigen::VectorXd q0 = p.market();
      Eigen::MatrixXd R;
      cal::StreamingCalibrator<cal::BundleProblem>::Options upd, ref;
      const Eigen::MatrixXd* Rp = nullptr;
      if (lambda > 0.0) { R = cal::tension_energy_operator(p, lambda, 0.0, {0}); upd.regularizer = R; ref.regularizer = R; Rp = &R; }
      ref.rescale_update = false;
      const Eigen::VectorXd x0 = cold(p, q0, Eigen::VectorXd::Constant(6, 0.03), Rp);
      cal::StreamingCalibrator<cal::BundleProblem> a(p, x0, q0, upd), b(p, x0, q0, ref);
      int rescales = 0;
      double worst_m = 0.0, worst_x = 0.0;
      for (double bp : {0.2, 0.6, 3.0, -1.0, 1.5, 0.0}) {
        Eigen::VectorXd q = q0;
        for (int i = 0; i < 10; ++i) q[i] += bp * 1e-4 * (i % 2 == 1 ? 1.0 : -0.3);
        const cal::StreamTick ta = a.update(q), tb = b.update(q);
        ASSERT_TRUE(ta.converged && tb.converged) << ta.reason() << " / " << tb.reason();
        rescales += ta.rescales;
        const double scale = b.sensitivity().cwiseAbs().maxCoeff();
        worst_m = std::max(worst_m, (a.sensitivity() - b.sensitivity()).cwiseAbs().maxCoeff() / scale);
        worst_x = std::max(worst_x, (a.current() - b.current()).cwiseAbs().maxCoeff());
      }
      std::cout << "  [band] lambda " << lambda << " decay " << decay << ": " << rescales << " rescales, |M_upd - M_ref|/|M| = " << worst_m
                << ", |x_upd - x_ref| = " << worst_x << "\n";
      EXPECT_GT(rescales, 0);
      EXPECT_LT(worst_m, 1e-9) << "lambda " << lambda << " decay " << decay;
      EXPECT_LT(worst_x, 1e-10) << "lambda " << lambda << " decay " << decay;
    }
  }
}
