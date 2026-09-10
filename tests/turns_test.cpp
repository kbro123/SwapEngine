// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T2 calibration (optimum / stationarity / recovery)
// Turns as a calibration instrument (docs/turns-calibration.md, Mode 2). QuantLib-free unit tests for
// the turn OVERLAY math and the banded state-pin turn instrument:
//   * the compiled W-cache turned discount equals the templated TurnedCurve discount to machine epsilon;
//   * a jump δ shifts DFs by exactly exp(−δ·overlap) — the full width after the window, zero before it;
//   * the interpolation stays a linear map (a turn is an overlay, not a region);
//   * the banded TurnJump state-pin: its analytic Jacobian column matches AAD and bump-and-reprice, and
//     the calibrated fit is first-order optimal (‖Jᵀr‖∞ ≈ 0, never "reprices exactly" — CLAUDE.md §2);
//   * BEHAVIORAL: a POSITIVE turn lowers the surrounding smooth forwards vs a no-turn baseline, and lets
//     disagreeing futures (spanning vs not spanning the date) reconcile instead of wiggling the curve.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/pricing/compiled_book.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace px = swaps::pricing;
namespace cal = swaps::calibration;

namespace {

px::CurveStructure make_spec(std::vector<px::Turn> turns) {
  px::CurveStructure s;
  s.regions = swaps::curve::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0});
  s.base = -1;
  s.turns = std::move(turns);
  return s;
}

// A synthetic single-coupon OIS as a ParRate instrument: float_leg_pv telescopes to 1 − DF(T),
// annuity to T·DF(T), so the par rate is (1 − DF(T))/(T·DF(T)). Enough to pin the back knots.
cal::Instrument ois(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  px::RateObservation o;
  o.sub_start = {0.0};
  o.sub_end = {T};
  o.tau_index = T;
  px::FloatCoupon fc;
  fc.obs = o;
  fc.pay = T;
  fc.tau_pay = T;
  ins.fwd.coupons = {fc};
  ins.fixed.coupons = {px::FixedCoupon{T, T, 1.0}};
  return ins;
}

// A synthetic overnight future over [s,e] as a Rate instrument (one telescoped sub-period).
cal::Instrument fut(double s, double e) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  px::RateObservation o;
  o.sub_start = {s};
  o.sub_end = {e};
  o.tau_index = e - s;
  ins.obs = o;
  return ins;
}

cal::Instrument turn_pin(int turn_index, double target, double lo, double hi, double decay) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::TurnJump;
  ins.turn_curve = 0;
  ins.turn_index = turn_index;
  ins.market = target;
  ins.band_lower = lo;
  ins.band_upper = hi;
  ins.band_decay = decay;
  return ins;
}

}  // namespace

TEST(Turns, CompiledMatchesTemplatedTurnedDiscount) {
  const px::CurveStructure spec = make_spec({{0.98, 1.0}});
  std::vector<px::CurveStructure> specs{spec};
  ASSERT_EQ(spec.n_interp_knots(), 6);
  ASSERT_EQ(spec.n_knots(), 7);  // 6 interp + 1 δ

  Eigen::VectorXd x(7);
  x << 0.030, 0.032, 0.035, 0.037, 0.039, 0.041, 0.0025;  // δ = 25 bp

  px::CompiledCurveSet cs;
  cs.init(specs);
  std::vector<double> times{0.1, 0.3, 0.6, 0.97, 0.99, 1.5, 3.0, 5.0};
  std::vector<int> idx;
  for (double t : times) idx.push_back(cs.reg(0, t));
  cs.finalize();
  const Eigen::VectorXd DF = cs.df(x);

  auto C = cal::build_bundle_curves<double>(specs, [&](int, int i) { return x[i]; });
  double worst = 0.0;
  for (std::size_t k = 0; k < times.size(); ++k) {
    const double d_templated = C[0]->discount(times[k]);
    worst = std::max(worst, std::abs(DF[idx[k]] - d_templated));
  }
  EXPECT_LT(worst, 1e-14) << "compiled turned DF must equal the TurnedCurve templated discount";
}

TEST(Turns, DeltaShiftsDiscountByExpMinusDeltaOverlap) {
  const double a = 0.98, b = 1.0, delta = 0.0025;
  std::vector<px::CurveStructure> specs{make_spec({{a, b}})};
  px::CompiledCurveSet cs;
  cs.init(specs);
  std::vector<double> times{0.5, 0.97, 0.99, 1.0, 2.5};  // before / before / inside / at end / after
  std::vector<int> idx;
  for (double t : times) idx.push_back(cs.reg(0, t));
  cs.finalize();

  Eigen::VectorXd base(7);
  base << 0.030, 0.032, 0.035, 0.037, 0.039, 0.041, 0.0;
  Eigen::VectorXd bumped = base;
  bumped[6] = delta;
  const Eigen::VectorXd DF0 = cs.df(base), DF1 = cs.df(bumped);

  for (std::size_t k = 0; k < times.size(); ++k) {
    const double ov = px::turn_overlap(times[k], {a, b});
    EXPECT_NEAR(DF1[idx[k]] / DF0[idx[k]], std::exp(-delta * ov), 1e-12)
        << "at t=" << times[k] << " (overlap " << ov << ")";
  }
  // Explicit endpoints: exactly 1 before the window, exactly exp(−δ(b−a)) after it.
  EXPECT_NEAR(DF1[idx[0]] / DF0[idx[0]], 1.0, 1e-14);                  // t=0.5 < a
  EXPECT_NEAR(DF1[idx[4]] / DF0[idx[4]], std::exp(-delta * (b - a)), 1e-12);  // t=2.5 > b
}

TEST(Turns, InterpolationRemainsLinearMapAndTurnIsNotARegion) {
  const px::CurveStructure spec = make_spec({{0.98, 1.0}});
  EXPECT_EQ(spec.modules().size(), 2u);  // Flat front + Hermite back; the turn is NOT a region
  auto c = swaps::curve::make_modular_curve<double>(spec.modules());
  EXPECT_EQ(c.n_knots(), spec.n_interp_knots());  // the interpolator sees only interp knots
  EXPECT_TRUE(c.is_linear_map());
}

// Build a small bundle whose market is GENERATED from a known x_true (a smooth curve + one positive
// turn), so x_true is a stationary point. Returns the problem + x_true + the turn's global state index.
struct TurnProblem {
  cal::BundleProblem prob;
  Eigen::VectorXd x_true;
  int state_index;
};
static TurnProblem build_turn_problem(bool include_turn_pin) {
  const double a = 0.98, b = 1.02, delta_true = 0.0030;
  TurnProblem tp;
  tp.prob.curves = {make_spec({{a, b}})};
  const int ni = tp.prob.curves[0].n_interp_knots();  // 6
  tp.state_index = ni;                                 // δ sits right after the interp knots
  // Futures that DO and DON'T span the turn, plus pillar OIS.
  tp.prob.instruments = {fut(0.80, 0.95), fut(0.95, 1.05), fut(0.90, 1.10), fut(1.05, 1.20),
                         ois(0.5),        ois(1.0),        ois(2.0),        ois(3.0),
                         ois(5.0)};
  if (include_turn_pin)
    tp.prob.instruments.push_back(turn_pin(0, delta_true, delta_true - 0.001, delta_true + 0.001, 0.05));

  tp.x_true.resize(tp.prob.n_knots());
  tp.x_true << 0.030, 0.032, 0.035, 0.037, 0.039, 0.041, delta_true;

  // Market := model quotes at x_true, so the residual is 0 there (banded rows too: q == target).
  cal::CompiledBundleResidual seed(tp.prob);
  const Eigen::VectorXd mq = seed.model_rates(tp.x_true);
  for (int i = 0; i < tp.prob.n_residuals(); ++i) tp.prob.instruments[i].market = mq[i];
  return tp;
}

TEST(Turns, BandedTurnJumpJacobianMatchesAadAndBump) {
  const TurnProblem tp = build_turn_problem(/*include_turn_pin=*/true);
  cal::CompiledBundleResidual cr(tp.prob);

  // At x_true the residual is ~0 (market was generated there).
  EXPECT_LT(cr.residuals(tp.x_true).cwiseAbs().maxCoeff(), 1e-12);

  const Eigen::MatrixXd J = cr.jacobian(tp.x_true);
  const Eigen::MatrixXd Jaad = cal::aad_jacobian(tp.prob, tp.x_true);
  double worst = 0.0;
  for (int i = 0; i < J.rows(); ++i)
    for (int j = 0; j < J.cols(); ++j)
      worst = std::max(worst, std::abs(J(i, j) - Jaad(i, j)) / std::max(1.0, std::abs(Jaad(i, j))));
  EXPECT_LT(worst, 1e-9) << "analytic block Jacobian (incl. the turn column) must match AAD";

  // Turn column vs central bump-and-reprice on δ (rel <= 1e-6, bump noise dominates — CLAUDE.md §3c).
  const double h = 1e-6;
  Eigen::VectorXd xp = tp.x_true, xm = tp.x_true;
  xp[tp.state_index] += h;
  xm[tp.state_index] -= h;
  const Eigen::VectorXd rp = tp.prob.residuals<double>(xp);
  const Eigen::VectorXd rm = tp.prob.residuals<double>(xm);
  const Eigen::VectorXd col_fd = (rp - rm) / (2.0 * h);
  double worst_col = 0.0;
  for (int i = 0; i < col_fd.size(); ++i)
    worst_col = std::max(worst_col,
                         std::abs(J(i, tp.state_index) - col_fd[i]) / std::max(1.0, std::abs(col_fd[i])));
  EXPECT_LT(worst_col, 1e-6) << "turn Jacobian column must match bump-and-reprice on δ";
}

TEST(Turns, CalibrationWithBandedTurnIsFirstOrderOptimal) {
  const TurnProblem tp = build_turn_problem(/*include_turn_pin=*/true);
  Eigen::VectorXd x0 = tp.x_true;
  x0.array() += 0.002;   // perturb every state (incl. δ) off the truth
  x0[tp.state_index] = 0.0;  // and start the turn from no jump at all

  const cal::CalibrationResult res = cal::calibrate(tp.prob, x0);
  // Over-determined + banded: assert first-order optimality, NOT exact repricing (CLAUDE.md §2).
  EXPECT_LT(res.stationarity, 1e-7) << "‖Jᵀr‖∞ must vanish at the banded-turn optimum";
  // The turn should be recovered close to its target (the band + spanning futures identify it).
  EXPECT_NEAR(res.x[tp.state_index], tp.x_true[tp.state_index], 5e-4);
}

TEST(Turns, PositiveTurnLowersSurroundingForwardsAndReconcilesFutures) {
  // Market carries a real +30bp turn. A no-turn curve must distort (raise) the smooth forwards around
  // the date to fit the spanning futures; adding the turn instrument absorbs the spike into δ so the
  // smooth forward drops back down — the desk intuition the design encodes (§1).
  const TurnProblem base = build_turn_problem(/*include_turn_pin=*/false);   // no δ state, no pin
  // Rebuild the baseline WITHOUT the turn overlay in the curve at all, using the SAME market quotes.
  cal::BundleProblem baseline;
  baseline.curves = {make_spec({})};  // no turns
  baseline.instruments.assign(base.prob.instruments.begin(), base.prob.instruments.end());

  const TurnProblem turned = build_turn_problem(/*include_turn_pin=*/true);

  Eigen::VectorXd xb0 = Eigen::VectorXd::Constant(baseline.n_knots(), 0.035);
  Eigen::VectorXd xt0 = Eigen::VectorXd::Constant(turned.prob.n_knots(), 0.035);
  const cal::CalibrationResult rb = cal::calibrate(baseline, xb0);
  const cal::CalibrationResult rt = cal::calibrate(turned.prob, xt0);

  // The turned fit reconciles the disagreeing futures; the baseline cannot.
  EXPECT_LT(rt.rms_residual, rb.rms_residual * 0.5)
      << "a turn lets spanning/non-spanning futures reconcile: baseline rms=" << rb.rms_residual
      << " turned rms=" << rt.rms_residual;

  // Probe the SMOOTH forward just outside the window (overlay contributes nothing there, so this is the
  // underlying smooth level for both). The positive turn pulls it DOWN vs the distorted baseline.
  auto Cb = cal::build_bundle_curves<double>(baseline.curves, [&](int, int i) { return rb.x[i]; });
  auto Ct = cal::build_bundle_curves<double>(turned.prob.curves, [&](int, int i) { return rt.x[i]; });
  const double probe = 0.95;  // < a = 0.98, outside the window
  const double f_baseline = Cb[0]->forward(probe);
  const double f_turned = Ct[0]->forward(probe);
  EXPECT_LT(f_turned, f_baseline)
      << "a positive turn must lower the surrounding smooth forward: baseline f=" << f_baseline
      << " turned f=" << f_turned;
}
