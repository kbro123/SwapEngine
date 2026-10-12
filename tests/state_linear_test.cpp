// E5 taxonomy: T1 unit + T5 properties (hand values, AAD vs compiled, the previous stacked form)
// Constraint rows, the last part (2026-10-12): the regulariser is ROWS OF THE PROBLEM. A StateLinear instrument
// (problem.hpp) is Σ coeff·x[state] -- a curvature row, a bound on a knot forward, a turn pin are its instances --
// and BundleProblem::constraints is where rows without a market feed live; the ONE compiled engine carries them as
// Src::State terms. Pins:
//   (1) a StateLinear row's value / residual / Jacobian: compiled == templated == the hand sum, bit for bit;
//   (2) smoothing_rows ARE the dense second_difference_operator (the dense view is derived from them), and a
//       session's regularised calibrate and risk operator land where the previous stacked [J; R] forms landed;
//   (3) the session installs the smoothing as its problem's tail constraint rows: the client's structure (count,
//       stamp, rebind) is unchanged by them, a reg change replaces the tail, off removes it;
//   (4) a bound on a knot forward as a StateLinear quote row with a one-sided penalty map calibrates and streams
//       like any mapped row (the engine's rows are the frozen system; nothing is stacked);
//   (5) a TurnJump is a one-coefficient StateLinear row (identical rows on the compiled engine).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/normal_op.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace api = swaps::api;

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
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = cal::instrument_model_quote<double>(p.row(i), cof);
  return q;
}

// 10 par swaps on 6 Hermite knots (over-determined), consistent quotes at xt.
cal::BundleProblem bundle(const Eigen::VectorXd& xt) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(par_swap(T));
  const Eigen::VectorXd q = model_quotes(p, xt);
  for (int i = 0; i < 10; ++i) p.instruments[i].market = q[i];
  return p;
}

Eigen::VectorXd smooth_true() {
  Eigen::VectorXd xt(6);
  xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  return xt;
}

cal::Instrument state_row(std::vector<cal::StateRef> refs, double market = 0.0) {
  cal::Instrument r;
  r.quote = cal::QuoteKind::StateLinear;
  r.state = std::move(refs);
  r.market = market;
  return r;
}

}  // namespace

// (1) value, residual and Jacobian of a StateLinear row: compiled == templated == the hand sum.
TEST(StateLinear, CompiledEqualsTemplatedEqualsTheHandSum) {
  cal::BundleProblem p = bundle(smooth_true());
  // a curvature-shaped row as a CONSTRAINT, a single-coefficient row as a QUOTE with a target
  p.constraints.push_back(state_row({{0, 1, 0.5}, {0, 2, -1.25}, {0, 3, 0.75}}));
  p.instruments.push_back(state_row({{0, 4, 1.0}}, 0.0455));
  cal::validate_problem(p, "test");
  Eigen::VectorXd x(6);
  x << 0.0401, 0.0413, 0.0419, 0.0442, 0.0458, 0.0471;
  const cal::CompiledBundleResidual eng(p);
  const Eigen::VectorXd rc = eng.residuals(x);
  const Eigen::VectorXd rt = p.residuals<double, Eigen::VectorXd>(x);
  ASSERT_EQ(rc.size(), 12);
  ASSERT_EQ(rt.size(), 12);
  const double hand_c = 0.5 * x[1] + (-1.25) * x[2] + 0.75 * x[3];  // the same left-to-right accumulation from 0
  EXPECT_EQ(rc[11], hand_c);
  EXPECT_EQ(rt[11], hand_c);
  EXPECT_EQ(rc[10], x[4] - 0.0455);
  EXPECT_EQ(rt[10], x[4] - 0.0455);
  for (int i = 0; i < 10; ++i) EXPECT_NEAR(rc[i], rt[i], 1e-14) << "row " << i;  // the par rows: compiled vs templated to rounding
  // the Jacobian row IS the coefficient pattern; AAD agrees
  const Eigen::MatrixXd Jc = eng.jacobian(x);
  const Eigen::MatrixXd Ja = cal::aad_jacobian(p, x);
  ASSERT_EQ(Jc.rows(), 12);
  EXPECT_EQ(Jc(11, 1), 0.5);
  EXPECT_EQ(Jc(11, 2), -1.25);
  EXPECT_EQ(Jc(11, 3), 0.75);
  EXPECT_EQ(Jc(11, 0), 0.0);
  EXPECT_EQ(Jc(10, 4), 1.0);
  EXPECT_LT((Jc - Ja).cwiseAbs().maxCoeff(), 1e-14);
  // the model values the joint evaluation reports are the raw sums
  Eigen::VectorXd m;
  const Eigen::VectorXd q = p.market();
  const cal::HybridBundleResidual hyb(p);
  (void)hyb.residuals_vs(x, q, &m);
  EXPECT_EQ(m[11], hand_c);
  EXPECT_EQ(m[10], x[4]);
  // the directional derivative along a direction is the coefficient dot product
  Eigen::VectorXd dir = Eigen::VectorXd::LinSpaced(6, 1.0, 6.0), out;
  eng.directional_into(x, dir, out);
  EXPECT_NEAR(out[11], 0.5 * dir[1] - 1.25 * dir[2] + 0.75 * dir[3], 1e-15);
  EXPECT_EQ(out[10], dir[4]);
  // the hybrid router keeps a StateLinear row on the compiled engine
  EXPECT_GE(hyb.compiled_row(11), 0);
  EXPECT_GE(hyb.compiled_row(10), 0);
  // validation: a row with no refs, a ref outside the curve's state
  cal::Instrument bad = state_row({});
  EXPECT_THROW(cal::validate_instrument(bad, "test"), std::invalid_argument);
  cal::BundleProblem pb = p;
  pb.constraints.push_back(state_row({{0, 6, 1.0}}));
  EXPECT_THROW(cal::validate_problem(pb, "test"), std::invalid_argument);
}

// (2) the smoothing rows are the operator, and a session under `reg` lands where the stacked forms landed.
TEST(StateLinear, SmoothingRowsAreTheOperatorAndTheSessionLandsWhereTheStackedFormsLanded) {
  const cal::BundleProblem p = bundle(smooth_true());
  cal::RegSpec reg;
  reg.lambda = 0.02;
  reg.curves = {0};
  const std::vector<cal::Instrument> rows = cal::smoothing_rows(p, reg.lambda, reg.curves);
  ASSERT_EQ(rows.size(), 4u) << "6 Hermite knots: interior rows 1..4";
  const Eigen::MatrixXd R = cal::second_difference_operator(p, reg.lambda, reg.curves);
  ASSERT_EQ(R.rows(), 4);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    ASSERT_EQ(rows[r].state.size(), 3u);
    for (const auto& s : rows[r].state) EXPECT_EQ(R(static_cast<int>(r), s.index), s.coeff);
    EXPECT_EQ(rows[r].market, 0.0);
    EXPECT_TRUE(rows[r].penalty().is_plain());
  }
  // a row's value on the engine is R·x to rounding (three products accumulated from 0 vs a dense dot product)
  cal::BundleProblem pr = p;
  cal::add_smoothing_rows(pr, reg);
  ASSERT_EQ(pr.n_constraints(), 4);
  ASSERT_EQ(pr.n_quotes(), 10);
  ASSERT_EQ(pr.n_residuals(), 14);
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.03);
  Eigen::VectorXd xt = smooth_true();
  xt[2] += 7e-4;  // a kink, so R·x is not ~0
  const cal::CompiledBundleResidual eng(pr);
  const Eigen::VectorXd rc = eng.residuals(xt);
  const Eigen::VectorXd Rx = R * xt;
  for (int r = 0; r < 4; ++r) EXPECT_NEAR(rc[10 + r], Rx[r], 1e-18 + 4e-16 * std::abs(Rx[r])) << r;

  // the session's regularised calibrate == the previous composition RegularizedEngine(engine without rows, R)
  api::BundleSession sess(p);
  const Eigen::VectorXd x_rows = sess.calibrate(x0, reg).x;
  const cal::HybridBundleResidual bare(p);
  const cal::RegularizedEngine<cal::HybridBundleResidual> stacked(bare, R);
  const Eigen::VectorXd x_stack = cal::calibrate_with(stacked, p.n_knots(), stacked.n_residuals(), x0).x;
  EXPECT_LT((x_rows - x_stack).cwiseAbs().maxCoeff(), 1e-10) << "the rows and the stacked block are the same least squares";
  // and the unregularised solve is NOT the same point (the regulariser does something on this kinked fit)
  EXPECT_GT((cal::calibrate(p, x0).x - x_rows).cwiseAbs().maxCoeff(), 1e-7);

  // the risk operator == the pseudo-inverse of the hand-stacked [J; R], quote columns
  const Eigen::MatrixXd M_rows = sess.risk_operator(reg);
  const Eigen::MatrixXd J = sess.jacobian(reg);
  ASSERT_EQ(J.rows(), 10) << "jacobian() is the QUOTE block";
  ASSERT_EQ(M_rows.rows(), 6);
  ASSERT_EQ(M_rows.cols(), 10);
  Eigen::MatrixXd S(14, 6);
  S.topRows(10) = J;
  S.bottomRows(4) = R;
  cal::NormalOp op(14, 6);
  op.factor(S);
  const Eigen::MatrixXd M_stack = op.quote_sensitivity(sess.residual_market_scale(), 10);
  EXPECT_LT((M_rows - M_stack).cwiseAbs().maxCoeff(), 1e-12 * M_stack.cwiseAbs().maxCoeff());
}

// (3) the session installs the smoothing as its problem's tail constraint rows; the client's structure is untouched.
TEST(StateLinear, ASessionInstallsSmoothingAsItsOwnTailRowsAndTheClientStructureIsUnchanged) {
  const cal::BundleProblem p = bundle(smooth_true());
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.03);
  cal::RegSpec light;
  light.lambda = 0.5;
  light.curves = {0};
  api::BundleSession sess(p);
  EXPECT_EQ(sess.problem().n_constraints(), 0);
  sess.calibrate(x0, light);
  EXPECT_EQ(sess.problem().n_constraints(), 4);
  EXPECT_EQ(sess.problem().n_quotes(), 10);
  EXPECT_EQ(sess.structural_stamp(), cal::structural_stamp(p)) << "the installed rows are the session's, not the client's structure";
  // the client requotes with ITS bundle (no smoothing rows): both rebind paths accept it
  cal::BundleProblem p2 = p;
  for (auto& ins : p2.instruments) ins.market += 3e-4;
  EXPECT_NO_THROW(sess.rebind(p2, light));
  EXPECT_NO_THROW(sess.rebind(cal::make_stamped(p2), light));
  EXPECT_EQ(sess.problem().n_constraints(), 4);
  // set_market / stream_update take the QUOTE vector
  Eigen::VectorXd q = p2.market();
  q.array() += 1e-4;
  EXPECT_NO_THROW(sess.set_market(q));
  EXPECT_NO_THROW(sess.resolve(light));
  sess.start_streaming(light);
  q.array() += 1e-4;
  EXPECT_NO_THROW(sess.stream_update(q));
  EXPECT_TRUE(sess.last_converged());
  // the streamed curve is the regularised cold answer for that market (rows ARE the frozen system)
  cal::BundleProblem pc = p;
  for (int i = 0; i < 10; ++i) pc.instruments[static_cast<std::size_t>(i)].market = q[i];
  cal::add_smoothing_rows(pc, light);
  const Eigen::VectorXd xc = cal::calibrate(pc, x0).x;
  EXPECT_LT((sess.x() - xc).cwiseAbs().maxCoeff(), 1e-7);
  // a different strength replaces the tail (same count, scaled coefficients); off removes it
  cal::RegSpec strong = light;
  strong.lambda = 5.0;
  sess.calibrate(x0, strong);
  ASSERT_EQ(sess.problem().n_constraints(), 4);
  EXPECT_NEAR(sess.problem().constraints[0].state[0].coeff, 10.0 * cal::smoothing_rows(p, light.lambda, light.curves)[0].state[0].coeff, 1e-15);
  sess.calibrate(x0, cal::RegSpec{});
  EXPECT_EQ(sess.problem().n_constraints(), 0);
  cal::BundleProblem pnow = p;  // the session's quotes moved above: compare against ITS current market
  for (int i = 0; i < 10; ++i) pnow.instruments[static_cast<std::size_t>(i)].market = sess.problem().instruments[static_cast<std::size_t>(i)].market;
  EXPECT_LT((sess.x() - cal::calibrate(pnow, x0).x).cwiseAbs().maxCoeff(), 1e-10);
  // a client bundle that carries its OWN constraint row keeps it beside the installed ones
  cal::BundleProblem pb = p;
  pb.constraints.push_back(state_row({{0, 5, 1.0}}, 0.047));
  api::BundleSession sb(pb);
  sb.calibrate(x0, light);
  EXPECT_EQ(sb.problem().n_constraints(), 5);
  EXPECT_EQ(sb.problem().constraints[0].state[0].index, 5) << "the client's row first, the smoothing tail after";
  cal::BundleProblem pb2 = pb;
  pb2.constraints[0].market = 0.0472;  // the client re-targets its own row through rebind
  EXPECT_NO_THROW(sb.rebind(pb2, light));
  EXPECT_EQ(sb.problem().constraints[0].market, 0.0472);
  cal::BundleProblem pb3 = p;  // without the client's row: a different structure
  EXPECT_THROW(sb.rebind(pb3, light), std::runtime_error);
}

// (4) a bound on a knot forward: a StateLinear quote row with a one-sided map, calibrated and streamed.
TEST(StateLinear, ABoundOnAKnotForwardIsAMappedRowThatCalibratesAndStreams) {
  Eigen::VectorXd xt = smooth_true();
  xt[2] = -0.0020;  // the market implies a negative 3y forward
  cal::BundleProblem p = bundle(xt);
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.03);
  const Eigen::VectorXd x_free = cal::calibrate(p, x0).x;
  ASSERT_LT(x_free[2], -1e-3) << "premise: unconstrained, the knot goes negative";
  // the bound: target 0, unit pull below 0, no pull above (a dead zone) -- heavily weighted so it dominates the quotes
  cal::Instrument bnd = state_row({{0, 2, 1.0}}, 0.0);
  cal::PenaltyMap pm;
  pm.n = 1;
  pm.b[0] = 0.0;
  pm.s[0] = 50.0;
  pm.s[1] = 0.0;
  bnd.penalty_map = pm;
  p.instruments.push_back(bnd);
  cal::validate_problem(p, "test");
  const Eigen::VectorXd xb = cal::calibrate(p, x0).x;
  EXPECT_GT(xb[2], -5e-6) << "the bound holds the knot at its edge";
  // the compiled and templated residuals agree on the mapped row on both sides of the edge
  const cal::CompiledBundleResidual eng(p);
  for (double v : {-0.001, 0.0, 0.002}) {
    Eigen::VectorXd x = xb;
    x[2] = v;
    const Eigen::VectorXd rc = eng.residuals(x), rt = p.residuals<double, Eigen::VectorXd>(x);
    EXPECT_EQ(rc[10], rt[10]) << v;
    EXPECT_EQ(rc[10], v < 0.0 ? 50.0 * v : 0.0) << v;
  }
  // streamed: push the market so the bound goes slack, then binding again; every tick is the cold answer
  const Eigen::VectorXd q0 = p.market();
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, xb, q0, opt);
  for (double shift_bp : {30.0, 60.0, 0.0, -10.0}) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < 10; ++i) q[i] += shift_bp * 1e-4;  // the bound's target stays 0
    const cal::StreamTick t = sc.update(q);
    ASSERT_TRUE(t.converged) << t.reason();
    cal::BundleProblem pc = p;
    for (int i = 0; i < 11; ++i) pc.instruments[static_cast<std::size_t>(i)].market = q[i];
    const Eigen::VectorXd xc = cal::calibrate(pc, x0).x;
    EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff() * 1e4, 0.05) << "shift " << shift_bp;
  }
}

// (5) a TurnJump is a one-coefficient StateLinear row.
TEST(StateLinear, ATurnJumpIsAOneCoefficientStateLinearRow) {
  cal::BundleProblem p;
  px::CurveStructure s;
  s.regions = cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0});
  s.base = -1;
  s.turns = {px::Turn{0.95, 1.05}};
  p.curves.push_back(s);
  for (int T = 1; T <= 5; ++T) p.instruments.push_back(par_swap(T));
  const int ni = p.curves[0].n_interp_knots();
  cal::Instrument tj;
  tj.quote = cal::QuoteKind::TurnJump;
  tj.turn_curve = 0;
  tj.turn_index = 0;
  tj.market = 2e-4;
  p.instruments.push_back(tj);
  p.instruments.push_back(state_row({{0, ni, 1.0}}, 2e-4));
  cal::validate_problem(p, "test");
  Eigen::VectorXd x = Eigen::VectorXd::Constant(p.n_knots(), 0.03);
  x[ni] = 3e-4;
  const cal::CompiledBundleResidual eng(p);
  const Eigen::VectorXd r = eng.residuals(x);
  const Eigen::VectorXd rt = p.residuals<double, Eigen::VectorXd>(x);
  EXPECT_EQ(r[5], r[6]);
  EXPECT_EQ(rt[5], rt[6]);
  EXPECT_EQ(r[6], 3e-4 - 2e-4);
  const Eigen::MatrixXd J = eng.jacobian(x);
  EXPECT_EQ(J.row(5), J.row(6));
  EXPECT_EQ(J(6, ni), 1.0);
}
