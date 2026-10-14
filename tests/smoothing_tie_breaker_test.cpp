// E5 taxonomy: T5 properties (every shape round-trips its quotes under smoothing; the chooser acts only where the quotes leave freedom)
// SMOOTHING IS A TIE-BREAKER, NEVER A TRADE AGAINST THE QUOTES (owner's rule, 2026-10-13, ASSUMPTIONS K48). The
// curvature rows choose the minimum-curvature curve among those the quotes allow: a hard quote reprices to its market,
// a banded quote stays inside its band, and the rows decide only the directions the quotes leave free -- an
// under-determined bundle's unreached knot, the interior of a bid/offer band quoted as a bound. The weight is the fixed
// regularize.hpp kSmoothingTieBreak (its O(w^2) leak measured below the tolerance on every shape); a region's
// reg_lambda is a switch whose magnitude is not read.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <string>
#include <vector>

#include "g10_market.hpp"
#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"

namespace cal = swaps::calibration;
namespace api = swaps::api;
using swaps::shapes::Shape;

namespace {

constexpr double kQuoteTol = 1e-8;  // 1e-4 bp in rate: the round-trip tolerance a desk curve must meet

struct Fit {
  double hard = 0, band = 0;  // worst hard-quote misfit, worst band violation (rate)
  int rank = 0;
  bool converged = false;
  Eigen::VectorXd x;
};

// Calibrate `p` (whatever rows it carries) from `x0` and read the fit back in QUOTE space off the model quotes.
Fit fit_of(const cal::BundleProblem& p, const Eigen::VectorXd& x0) {
  const cal::CalibrationResult r = cal::calibrate(p, x0);
  Fit f;
  f.converged = r.converged;
  f.rank = r.rank_deficiency;
  f.x = r.x;
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return r.x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (int i = 0; i < p.n_quotes(); ++i) {
    const auto& ins = p.instruments[static_cast<std::size_t>(i)];
    if (ins.quote == cal::QuoteKind::FxForward) continue;  // its residual is a log form; the rate rows carry the test
    const double m = cal::instrument_model_quote<double>(ins, curve_of);
    if (ins.band_upper > ins.band_lower) f.band = std::max(f.band, std::max(ins.band_lower - m, m - ins.band_upper));
    else f.hard = std::max(f.hard, std::abs(m - ins.market));
  }
  return f;
}

cal::BundleProblem smoothed(const cal::BundleProblem& p, double stamp = 1.0) {
  cal::BundleProblem q = p;
  cal::stamp_region_smoothing(q, stamp);
  cal::add_smoothing_rows(q);
  return q;
}

double zigzag(const cal::BundleProblem& p, const Eigen::VectorXd& x, int c = 0) {
  int nk = 0;
  for (const auto& reg : p.curves[static_cast<std::size_t>(c)].regions) nk += static_cast<int>(reg.knots.size());
  const int off = p.offset(c);
  double z = 0;
  for (int i = 1; i + 1 < nk; ++i) z = std::max(z, std::abs(x[off + i + 1] - 2.0 * x[off + i] + x[off + i - 1]));
  return z;
}

}  // namespace

// (1) Every shape, smoothed on every region: hard quotes round-trip to < 1e-4 bp, banded quotes stay in band.
TEST(SmoothingTieBreaker, EveryShapeRoundTripsItsQuotesUnderSmoothing) {
  std::vector<Shape> shapes = swaps::shapes::ladder();
  shapes.push_back(swaps::shapes::g10_desk(false, /*light=*/false));
  shapes.push_back(swaps::shapes::g10_desk(true, /*light=*/false));
  for (const Shape& s : shapes) {
    const cal::BundleProblem p = smoothed(s.prob);
    ASSERT_GT(p.n_constraints(), 0) << s.name;
    const Fit f = fit_of(p, s.x0);
    EXPECT_TRUE(f.converged) << s.name;
    EXPECT_EQ(f.rank, 0) << s.name;
    EXPECT_LT(f.hard, kQuoteTol) << s.name << ": a hard quote was traded for smoothness (" << f.hard * 1e4 << " bp)";
    EXPECT_LE(f.band, kQuoteTol) << s.name << ": a banded quote left its band (" << f.band * 1e4 << " bp)";
  }
}

// (2) The switch: a region's reg_lambda magnitude is not read -- 0.5, 5 and 1e-3 give the same rows and the same curve.
TEST(SmoothingTieBreaker, TheMagnitudeOfTheSwitchIsNotRead) {
  const Shape s = swaps::shapes::ois_lag();
  const cal::BundleProblem a = smoothed(s.prob, 0.5), b = smoothed(s.prob, 5.0), c = smoothed(s.prob, 1e-3);
  ASSERT_EQ(a.n_constraints(), b.n_constraints());
  for (int i = 0; i < a.n_constraints(); ++i)
    for (std::size_t k = 0; k < a.constraints[static_cast<std::size_t>(i)].state.size(); ++k) {
      EXPECT_EQ(a.constraints[static_cast<std::size_t>(i)].state[k].coeff, b.constraints[static_cast<std::size_t>(i)].state[k].coeff);
      EXPECT_EQ(a.constraints[static_cast<std::size_t>(i)].state[k].coeff, c.constraints[static_cast<std::size_t>(i)].state[k].coeff);
    }
  EXPECT_EQ((fit_of(a, s.x0).x - fit_of(b, s.x0).x).cwiseAbs().maxCoeff(), 0.0);
  // and the rows ARE at the tie-breaker weight: the dense view at kSmoothingTieBreak over curve 0
  const Eigen::MatrixXd R = cal::state_rows_matrix(a, a.constraints);
  const Eigen::MatrixXd Rw = cal::second_difference_operator(a, cal::kSmoothingTieBreak, {0});
  EXPECT_LT((R - Rw).cwiseAbs().maxCoeff(), 1e-18);
}

// (3) A direction the quotes do not see is set by curvature: an unreached tail knot (rank-deficient unsmoothed, parked
// at the seed) becomes the minimum-curvature continuation -- its divided second difference is zero -- and the quotes
// are untouched.
TEST(SmoothingTieBreaker, AnUnreachedKnotIsSetByMinimumCurvatureAndTheQuotesAreUntouched) {
  Shape s = swaps::shapes::ois_lag();
  cal::BundleProblem p = s.prob;
  auto& back = p.curves[0].regions.back();
  const double t_last = back.knots.back();
  back.knots.push_back(t_last + 5.0);  // a 35y knot with no instrument reaching it
  Eigen::VectorXd x0(p.n_knots());
  x0 << s.x0, 0.030;
  const Fit raw = fit_of(p, x0);
  ASSERT_TRUE(raw.converged);
  ASSERT_EQ(raw.rank, 1) << "premise: the tail knot is unreached";
  const cal::BundleProblem ps = smoothed(p);
  const Fit sm = fit_of(ps, x0);
  EXPECT_TRUE(sm.converged);
  EXPECT_EQ(sm.rank, 0) << "the curvature row determines it";
  EXPECT_LT(sm.hard, kQuoteTol);
  // the last three knots: divided second difference zero to solver tolerance (the row's own residual, at the tie-breaker
  // weight a 1e-7 curvature is a 1e-11 residual), against the raw fit's, whose parked knot bends the continuation
  const int n = p.n_knots();
  const auto& K = back.knots;
  const std::size_t m = K.size();
  const double h0 = K[m - 2] - K[m - 3], h1 = K[m - 1] - K[m - 2];
  const auto d2_of = [&](const Eigen::VectorXd& x) { return 2.0 * ((x[n - 1] - x[n - 2]) / h1 - (x[n - 2] - x[n - 3]) / h0) / (h0 + h1); };
  EXPECT_LT(std::abs(d2_of(sm.x)), 1e-7) << "the unreached knot continues the curve with zero curvature";
  EXPECT_LT(std::abs(d2_of(sm.x)), 1e-3 * std::abs(d2_of(raw.x))) << "where the raw fit's seed-parked knot left a kink";
  EXPECT_NE(sm.x[n - 1], x0[n - 1]) << "it no longer sits at the seed";
  // The QUOTED knots move too -- the Hermite tangent at 30y reads the 35y knot, so the forward under the 30y swap changes
  // with it -- but the quotes do not: both fits reprice the market (sm.hard above); that is the whole rule.
  EXPECT_LT(raw.hard, kQuoteTol);
}

// (4) MINIMUM CURVATURE WITHIN BANDS: the long end's par quotes are integrals, so 0.03 bp of alternating jitter on
// hard quotes forces an alternating pattern of the knot forwards (the zigzag). Quoted as BOUNDS (a bid/offer band
// with no pull inside, decay 0) wider than the jitter, the smoothing chooses the minimum-curvature curve inside the
// bands: every quote stays inside its band and the zigzag collapses toward the generating curve's.
TEST(SmoothingTieBreaker, MinimumCurvatureWithinBandsRemovesTheJitterZigzagWithoutLeavingTheBands) {
  const Shape s = swaps::shapes::ois_lag();
  // hard quotes with alternating jitter: the exact fit amplifies it into the knots
  cal::BundleProblem hard = s.prob;
  for (int i = 0; i < hard.n_quotes(); ++i) hard.instruments[static_cast<std::size_t>(i)].market += (i % 2 ? 3e-6 : -3e-6);
  const Fit fh = fit_of(hard, s.x0);
  ASSERT_TRUE(fh.converged);
  const double z_true = zigzag(s.prob, s.x_true), z_hard = zigzag(hard, fh.x);
  ASSERT_GT(z_hard, 3.0 * z_true + 1e-6) << "premise: hard jittered quotes zigzag the knots";
  // the same jittered quotes as bounds +-0.1 bp with no pull inside, smoothed
  cal::BundleProblem banded = hard;
  for (auto& ins : banded.instruments) { ins.band_lower = ins.market - 1e-5; ins.band_upper = ins.market + 1e-5; ins.band_decay = 0.0; }
  const cal::BundleProblem bs = smoothed(banded);
  const Fit fb = fit_of(bs, s.x0);
  EXPECT_TRUE(fb.converged);
  EXPECT_LE(fb.band, kQuoteTol) << "a quote left its band";
  const double z_band = zigzag(bs, fb.x);
  EXPECT_LT(z_band, 0.25 * z_hard) << "the chooser did not remove the zigzag: " << z_band * 1e4 << " vs " << z_hard * 1e4 << " bp";
  EXPECT_LT((fb.x - s.x_true).cwiseAbs().maxCoeff(), 0.5 * (fh.x - s.x_true).cwiseAbs().maxCoeff()) << "and lands nearer the generating curve";
}

// (5) A square quote set plus its tie-breaker rows streams a 25 bp move EXACTLY on the frozen operator, with no
// drift refresh (the square-problem argument holds to the tie-breaker's O(w^2) leak): the streamed curve equals the
// cold solve of the same smoothed problem at the moved market.
TEST(SmoothingTieBreaker, SquareQuotesPlusTieBreakersStreamABigMoveExactlyWithoutRefresh) {
  const Shape s = swaps::shapes::ois_lag();
  ASSERT_EQ(s.prob.n_quotes(), s.prob.n_knots()) << "premise: square quotes";
  cal::BundleProblem p = s.prob;
  cal::stamp_region_smoothing(p, 1.0);
  api::BundleSession sess(p);
  ASSERT_GT(sess.problem().n_constraints(), 0);
  sess.calibrate(s.x0);
  sess.start_streaming();
  const Eigen::VectorXd& x = sess.stream_update(s.q_big);
  ASSERT_TRUE(sess.last_converged()) << sess.last_reason();
  EXPECT_EQ(sess.last_refreshes(), 0) << "square quotes + tie-breakers: exempt from the drift refresh";
  cal::BundleProblem pc = p;
  for (int i = 0; i < pc.n_quotes(); ++i) pc.instruments[static_cast<std::size_t>(i)].market = s.q_big[i];
  cal::add_smoothing_rows(pc);
  const Fit cold = fit_of(pc, s.x0);
  EXPECT_LT((x - cold.x).cwiseAbs().maxCoeff(), 1e-7) << "streamed vs cold on the same smoothed objective";
  EXPECT_LT(cold.hard, kQuoteTol);
}
