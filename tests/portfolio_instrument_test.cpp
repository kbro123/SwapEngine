// Portfolio (linear-combination) instruments + bid/offer band residuals.
//
// Engine-only (no QuantLib): a Portfolio's components are ordinary instruments whose pricing is already
// QuantLib-oracle-validated elsewhere, so here we pin only the NEW logic — that the combined quote is the
// exact weighted sum, that a butterfly actually constrains the mid leg, that the band re-weights the
// residual as specified, and (the subtle one) that the compiled W-cache path's analytic band Jacobian
// matches AAD term-for-term.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

// One outright OIS curve, Hermite over {1,2,5,10}. Self-discounting (forecast = discount = 0).
std::vector<cal::BundleCurveSpec> specs() {
  return {{.base = -1, .regions = swaps::curve::flat_hermite({}, {1, 2, 5, 10})}};
}
Eigen::VectorXd knots() {
  Eigen::VectorXd x(4);
  x << 0.030, 0.035, 0.040, 0.045;
  return x;
}

// Annual OIS legs -> a ParRate instrument to maturity T (self-forecast/discount on curve 0).
cal::Instrument par_swap(double T, double market = 0.0) {
  std::vector<double> t;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) t.push_back(u);
  t.push_back(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u : t) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev});
    prev = u;
  }
  ins.market = market;
  return ins;
}

// Model quote of a single instrument at x (a 1-instrument bundle; residual with market 0 IS the quote).
double quote_at(const cal::Instrument& ins, const Eigen::VectorXd& x) {
  cal::Instrument q = ins;
  q.market = 0.0;
  cal::BundleProblem p;
  p.curves = specs();
  p.instruments = {q};
  return p.residuals<double>(x)[0];
}

cal::Instrument butterfly(double wing_lo, double belly, double wing_hi) {
  cal::Instrument b;
  b.quote = cal::QuoteKind::Portfolio;
  b.combination = {{-1.0, par_swap(wing_lo)}, {2.0, par_swap(belly)}, {-1.0, par_swap(wing_hi)}};
  return b;
}

}  // namespace

// The combined quote is EXACTLY the weighted sum of the component model quotes -- no leg outright needed.
TEST(PortfolioInstrument, ButterflyQuoteIsTheWeightedSum) {
  const Eigen::VectorXd x = knots();
  const double q2 = quote_at(par_swap(2), x), q5 = quote_at(par_swap(5), x), q10 = quote_at(par_swap(10), x);
  const double qb = quote_at(butterfly(2, 5, 10), x);
  EXPECT_NEAR(qb, -q2 + 2.0 * q5 - q10, 1e-13);

  // Nested: a portfolio whose sole component is the butterfly (weight 1) prices identically.
  cal::Instrument nested;
  nested.quote = cal::QuoteKind::Portfolio;
  nested.combination = {{1.0, butterfly(2, 5, 10)}};
  EXPECT_NEAR(quote_at(nested, x), qb, 1e-14);
}

// A butterfly genuinely CONSTRAINS the belly: calibrate 4 knots from {1y, 2y, 10y outrights + a 2-5-10
// butterfly}, all quoted off a known curve, and recover that curve. The 5y knot is pinned ONLY through
// the butterfly, so recovery proves the combination carries real information.
TEST(PortfolioInstrument, ButterflyCalibratesTheBellyKnot) {
  const Eigen::VectorXd truth = knots();
  cal::BundleProblem prob;
  prob.curves = specs();
  cal::Instrument s1 = par_swap(1, quote_at(par_swap(1), truth));
  cal::Instrument s2 = par_swap(2, quote_at(par_swap(2), truth));
  cal::Instrument s10 = par_swap(10, quote_at(par_swap(10), truth));
  cal::Instrument bfly = butterfly(2, 5, 10);
  bfly.market = quote_at(bfly, truth);
  prob.instruments = {s1, s2, s10, bfly};

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(4, 0.02));  // compiled W-cache path
  EXPECT_LT(res.rms_residual, 1e-10);
  EXPECT_LT((res.x - truth).cwiseAbs().maxCoeff(), 1e-8) << "the butterfly pins the 5y knot";
}

// The portfolio is now on the compiled W-cache path (its components register as weighted batch entries
// that SUM into one row). So the compiled residual AND analytic Jacobian must match the templated + AAD
// path term-for-term -- this pins the weighted-accumulation scatter.
TEST(PortfolioInstrument, CompiledResidualAndJacobianMatchAad) {
  const Eigen::VectorXd truth = knots();
  cal::Instrument bfly = butterfly(2, 5, 10);
  bfly.market = quote_at(bfly, truth);
  cal::BundleProblem prob;
  prob.curves = specs();
  prob.instruments = {par_swap(1, quote_at(par_swap(1), truth)), par_swap(2, quote_at(par_swap(2), truth)),
                      bfly, par_swap(10, quote_at(par_swap(10), truth))};

  Eigen::VectorXd x(4);
  x << 0.033, 0.039, 0.047, 0.041;  // away from the solution
  const Eigen::VectorXd r_aad = prob.residuals<double>(x);
  const Eigen::MatrixXd J_aad = cal::aad_jacobian(prob, x);
  cal::CompiledBundleResidual eng(prob);  // no longer throws on the portfolio -- it is W-cacheable
  EXPECT_LT((eng.residuals(x) - r_aad).cwiseAbs().maxCoeff(), 1e-11) << "compiled portfolio residual == AAD";
  EXPECT_LT((eng.jacobian(x) - J_aad).cwiseAbs().maxCoeff(), 1e-7) << "compiled portfolio Jacobian == AAD";
}

// The Huber band residual: decay-slope pull to the mid inside [lower, upper], unit slope outside, continuous
// at both edges, monotone with a single zero (so the squared residual is convex), and the double (r, dr/dq)
// form agrees with the templated one.
TEST(BandResidual, HuberProfile) {
  const double lo = 0.030, hi = 0.032, m = 0.031, decay = 0.1;
  EXPECT_NEAR(cal::band_residual<double>(m, m, lo, hi, decay), 0.0, 1e-15) << "zero at the mid";
  EXPECT_NEAR(cal::band_slope(m, lo, hi, decay), decay, 1e-15) << "decay slope inside";
  EXPECT_NEAR(cal::band_slope(hi + 1e-6, lo, hi, decay), 1.0, 1e-15) << "unit slope outside";
  // continuity at the edges (the residual has no jump; only its slope does)
  EXPECT_NEAR(cal::band_residual<double>(hi + 1e-12, m, lo, hi, decay), cal::band_residual<double>(hi, m, lo, hi, decay), 1e-11);
  EXPECT_NEAR(cal::band_residual<double>(lo - 1e-12, m, lo, hi, decay), cal::band_residual<double>(lo, m, lo, hi, decay), 1e-11);
  // monotone, and strictly steeper outside than inside
  double prev = cal::band_residual<double>(0.020, m, lo, hi, decay);
  for (double q = 0.020; q <= 0.042; q += 0.0001) {
    const double r = cal::band_residual<double>(q, m, lo, hi, decay);
    EXPECT_GE(r, prev - 1e-15);
    const auto rd = cal::band_residual_d(q, m, lo, hi, decay);
    EXPECT_NEAR(rd.first, r, 1e-15) << "double form == templated form at q=" << q;
    EXPECT_NEAR(rd.second, cal::band_slope(q, lo, hi, decay), 1e-15);
    prev = r;
  }
  // far outside: full-slope pull to the EDGE plus the decay pull to the mid (not full-slope pull to the mid)
  EXPECT_NEAR(cal::band_residual<double>(0.040, m, lo, hi, decay), decay * (hi - m) + (0.040 - hi), 1e-15);
  // no band (upper <= lower) -> the plain residual q − m
  EXPECT_NEAR(cal::band_residual<double>(0.031, 0.030, 0.0, 0.0, 1.0), 0.001, 1e-15);
  EXPECT_EQ(cal::band_slope(0.031, 0.0, 0.0, 1.0), 1.0);
}

// THE subtle one: with a banded instrument, the compiled W-cache residual AND its analytic Jacobian must
// match the templated + AAD path term-for-term (the compiled band chain rule vs AAD through band_weight).
TEST(BandResidual, CompiledMatchesAadWithBand) {
  const Eigen::VectorXd truth = knots();
  const double q5 = quote_at(par_swap(5), truth);
  cal::Instrument s5 = par_swap(5, q5);
  s5.band_lower = q5 - 0.0010;  // ±10bp band around mid
  s5.band_upper = q5 + 0.0010;
  s5.band_decay = 0.1;

  cal::BundleProblem prob;
  prob.curves = specs();
  prob.instruments = {par_swap(2, quote_at(par_swap(2), truth)), s5,
                      par_swap(10, quote_at(par_swap(10), truth))};

  cal::CompiledBundleResidual eng(prob);
  // The same problem WITHOUT the band: inside the band the banded row must be exactly decay x the plain row
  // (the chain rule's non-trivial branch, slope = decay); outside it the slope is 1 and the rows coincide.
  cal::BundleProblem plain = prob;
  plain.instruments[1].band_lower = plain.instruments[1].band_upper = 0.0;
  plain.instruments[1].band_decay = 1.0;
  cal::CompiledBundleResidual eng_plain(plain);

  // (E5.2 2026-09-10: the 2026-09-08 audit showed this test evaluated ONLY outside the band -- slope 1, the
  // scaling a no-op -- so deleting the compiled chain-rule scaling passed it. Now three points: at truth (the
  // quote at its mid, slope = decay), inside but off-centre, and outside.)
  struct Point { const char* where; Eigen::VectorXd x; double slope; };
  Eigen::VectorXd inside = truth; inside[2] += 0.00015;  // moves the 5y quote ~1 bp: still inside +-10 bp
  Eigen::VectorXd outside(4); outside << 0.033, 0.039, 0.047, 0.041;
  const Point pts[] = {{"at truth (mid)", truth, 0.1}, {"inside the band", inside, 0.1}, {"outside the band", outside, 1.0}};
  for (const Point& pt : pts) {
    const Eigen::VectorXd r_aad = prob.residuals<double>(pt.x);
    const Eigen::MatrixXd J_aad = cal::aad_jacobian(prob, pt.x);
    const Eigen::VectorXd r_c = eng.residuals(pt.x);
    const Eigen::MatrixXd J_c = eng.jacobian(pt.x);
    EXPECT_LT((r_c - r_aad).cwiseAbs().maxCoeff(), 1e-11) << "compiled banded residual == AAD " << pt.where;
    EXPECT_LT((J_c - J_aad).cwiseAbs().maxCoeff(), 1e-7) << "compiled banded Jacobian == AAD " << pt.where;
    // the banded row is (slope x the unbanded row) -- the chain rule stated explicitly
    const Eigen::MatrixXd J_p = eng_plain.jacobian(pt.x);
    EXPECT_LT((J_c.row(1) - pt.slope * J_p.row(1)).cwiseAbs().maxCoeff(), 1e-12) << "band chain rule " << pt.where;
    EXPECT_LT((J_c.row(0) - J_p.row(0)).cwiseAbs().maxCoeff(), 1e-12) << "unbanded rows untouched " << pt.where;
  }
}

// A banded bundle STREAMS on the frozen-Newton fast path (residuals_vs drives the soft residual, not an
// exact reprice), and its per-tick solution equals a full cold recalibrate at the same market — i.e. the
// streamer solves the soft LEAST-SQUARES, not a reprice. This is what keeps bands at µs, not ms.
TEST(BandResidual, FrozenNewtonStreamsTheSoftLeastSquares) {
  const Eigen::VectorXd truth = knots();
  const double q5 = quote_at(par_swap(5), truth);
  cal::Instrument s5 = par_swap(5, q5);
  s5.band_lower = q5 - 0.0010;
  s5.band_upper = q5 + 0.0010;
  s5.band_decay = 0.1;
  cal::BundleProblem prob;  // 4 instruments for 4 knots -> determined (unique solution to compare)
  prob.curves = specs();
  prob.instruments = {par_swap(1, quote_at(par_swap(1), truth)), par_swap(2, quote_at(par_swap(2), truth)),
                      s5, par_swap(10, quote_at(par_swap(10), truth))};

  const Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(4, 0.02)).x;
  const Eigen::VectorXd q_anchor = prob.market();
  cal::StreamingCalibrator<cal::BundleProblem> sc(prob, x0, q_anchor, {});

  Eigen::VectorXd q_new = q_anchor;  // a market move
  q_new[0] += 0.0005;
  q_new[3] -= 0.0004;
  sc.update(q_new);

  cal::BundleProblem prob2 = prob;  // reference: cold recalibrate at the new market
  for (int i = 0; i < prob2.n_residuals(); ++i) prob2.instruments[i].market = q_new[i];
  const Eigen::VectorXd x_ref = cal::calibrate(prob2, x0).x;
  EXPECT_LT((sc.current() - x_ref).cwiseAbs().maxCoeff(), 1e-9)
      << "frozen-Newton streaming == cold recalibrate for a banded (soft) bundle";
}

// Behaviourally: inside the band the residual is down-weighted (softer) vs the same miss unbanded.
TEST(BandResidual, InsideBandIsSofterThanHard) {
  const Eigen::VectorXd truth = knots();
  const double q5 = quote_at(par_swap(5), truth);
  cal::Instrument hard = par_swap(5, q5);
  cal::Instrument soft = hard;
  soft.band_lower = q5 - 0.0010;
  soft.band_upper = q5 + 0.0010;
  soft.band_decay = 0.1;

  // A curve where the 5y quote lands ~5bp above mid but still inside a 10bp band.
  Eigen::VectorXd x = truth;
  x[2] += 0.0006;  // nudge the 5y-ish knot
  cal::BundleProblem ph;
  ph.curves = specs();
  ph.instruments = {hard};
  cal::BundleProblem psft;
  psft.curves = specs();
  psft.instruments = {soft};
  const double rh = std::abs(ph.residuals<double>(x)[0]);
  const double rs = std::abs(psft.residuals<double>(x)[0]);
  ASSERT_LT(std::abs(quote_at(hard, x) - q5), 0.0010) << "test setup: quote is inside the band";
  EXPECT_LT(rs, rh * 0.5) << "inside the band the residual is materially down-weighted";
}
