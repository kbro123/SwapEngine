// E5 taxonomy: T2 calibration (optimum / stationarity / recovery) | T6 regression (fails on the reverted bug)
// The FAILED-TICK CONTRACT of the frozen-Newton streamer (E3 finding C1, fixed 2026-09-10).
// A tick that does not converge is (a) reported with a reason, (b) never committed, and (c) leaves the
// streamer in a VALID state: the next tick cannot report a stale curve as converged. Before the fix a
// refresh inside the failed tick left the anchor Jacobian at the divergent iterate; the next tick's
// frozen-Newton step was M·r == 0 there, so every later tick at that market reported converged=true with
// a 1289 bp stale curve until the caller recalibrated (probe C-divergent-repeat).
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {
// A 4-curve basis chain (OIS + 3 ParSpread basis curves), 26 knots each, annual par swaps 1y..30y -- the
// chain8x26 gate fixture at half width. Markets generated from a known state x_true.
constexpr int NC = 4, NK = 26;
constexpr double MAX_T = 30.0;
struct Legs { std::vector<px::FloatCoupon> flt; std::vector<px::FixedCoupon> fix; };
Legs annual(double T) {
  Legs L; double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c; c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    L.flt.push_back(c); L.fix.push_back({u, u - prev}); prev = u;
  }
  return L;
}
cal::Instrument par_inst(double T, int fc, int dc) {
  Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParRate; in.fwd = {L.flt, fc, dc}; in.fixed = {L.fix, dc}; return in;
}
cal::Instrument basis_inst(double T, int fc, int bc, int dc) {
  Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParSpread; in.fwd = {L.flt, fc, dc}; in.bench = {L.flt, bc, dc}; in.fixed = {L.fix, dc}; return in;
}
struct Chain {
  cal::BundleProblem prob;
  Eigen::VectorXd x0, q0;
  Chain() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c) prob.curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};
    std::vector<double> mats; for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);
    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));
    for (int c = 1; c < NC; ++c) for (double T : mats) prob.instruments.push_back(basis_inst(T, c, c - 1, 0));
    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < int(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
    x0.resize(NC * NK); for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;
    q0 = prob.market();
  }
};
}  // namespace

// The probe scenario: +800 bp in one tick (converges), then back to q0 in one tick (an intrinsically hard
// reverse step -- it may fail). Whatever happens, the contract: a failed tick is reported, never committed,
// and NO later tick at q0 reports converged with a curve away from the cold answer.
TEST(StreamingContract, AFailedTickIsReportedNotCommittedAndDoesNotPoisonLaterTicks) {
  const Chain f;
  const Eigen::VectorXd xc = cal::calibrate(f.prob, f.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(f.prob, xc, f.q0, {});
  const Eigen::VectorXd q8 = f.q0.array() + 0.08;
  const cal::StreamTick up = sc.update(q8);
  ASSERT_TRUE(up.converged) << up.reason();
  const Eigen::VectorXd x8 = sc.current();
  int failures = 0;
  for (int k = 0; k < 3; ++k) {
    const cal::StreamTick t = sc.update(f.q0);
    const double err = (sc.current() - xc).cwiseAbs().maxCoeff();
    if (t.converged) {
      EXPECT_EQ(t.status, cal::StreamStatus::Converged);
      EXPECT_LT(err, 1e-9) << "tick " << k << " reported converged with a curve " << err << " from the cold answer (the C1 lie)";
    } else {
      ++failures;
      EXPECT_NE(t.status, cal::StreamStatus::Converged);
      EXPECT_STRNE(t.reason(), "converged");
      EXPECT_EQ((sc.current() - x8).cwiseAbs().maxCoeff(), 0.0) << "a failed tick must not commit";
    }
  }
  std::cout << "  [contract] +800bp then q0 x3: " << failures << " failed ticks (honestly reported)\n";
  // A staircase down from wherever the streamer is converges to the cold answer.
  for (double bp : {0.06, 0.04, 0.02, 0.0}) {
    const Eigen::VectorXd q = f.q0.array() + bp;
    const cal::StreamTick t = sc.update(q);
    EXPECT_TRUE(t.converged) << "staircase " << bp << ": " << t.reason();
  }
  EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-9);
}

// A forced failure (step cap 1, no refresh) on a 30 bp move: reported as StepCap, not committed, and the very
// next tick at the anchor market converges in ONE step from the untouched anchor (M·r == 0 there).
TEST(StreamingContract, StepCapIsReportedAndTheAnchorStaysValid) {
  const Chain f;
  const Eigen::VectorXd xc = cal::calibrate(f.prob, f.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  opt.max_steps = 1;
  opt.max_refresh = 0;
  opt.max_frozen = 100;
  cal::StreamingCalibrator<cal::BundleProblem> sc(f.prob, xc, f.q0, opt);
  const cal::StreamTick t1 = sc.update(f.q0.array() + 30e-4);
  EXPECT_FALSE(t1.converged);
  EXPECT_EQ(t1.status, cal::StreamStatus::StepCap);
  EXPECT_STREQ(t1.reason(), "step cap hit");
  EXPECT_EQ((sc.current() - xc).cwiseAbs().maxCoeff(), 0.0);
  const cal::StreamTick t2 = sc.update(f.q0);
  EXPECT_TRUE(t2.converged) << t2.reason();
  EXPECT_EQ(t2.newton_steps, 1);
  EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-12);  // one zero-ish step from the untouched anchor
}

// An absurd move (+500 %) diverges: reported (Diverged or a cap), not committed, and the next tick at q0
// converges from the restored anchor.
TEST(StreamingContract, ADivergentTickIsReportedAndTheAnchorIsRestored) {
  const Chain f;
  const Eigen::VectorXd xc = cal::calibrate(f.prob, f.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(f.prob, xc, f.q0, {});
  const cal::StreamTick t1 = sc.update(f.q0.array() + 5.0);
  EXPECT_FALSE(t1.converged);
  std::cout << "  [contract] +500% tick: " << t1.reason() << " after " << t1.newton_steps << " steps, " << t1.refreshes << " refreshes\n";
  EXPECT_EQ((sc.current() - xc).cwiseAbs().maxCoeff(), 0.0);
  const cal::StreamTick t2 = sc.update(f.q0);
  EXPECT_TRUE(t2.converged) << t2.reason();
  EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(StreamingContract, NonFiniteInputsThrow) {
  const Chain f;
  const Eigen::VectorXd xc = cal::calibrate(f.prob, f.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(f.prob, xc, f.q0, {});
  Eigen::VectorXd q = f.q0;
  q[3] = std::nan("");
  EXPECT_THROW(sc.update(q), std::invalid_argument);
  q[3] = INFINITY;
  EXPECT_THROW(sc.update(q), std::invalid_argument);
  EXPECT_THROW(sc.update(f.q0.head(5)), std::invalid_argument);
  Eigen::VectorXd xbad = xc; xbad[0] = std::nan("");
  EXPECT_THROW((cal::StreamingCalibrator<cal::BundleProblem>(f.prob, xbad, f.q0, {})), std::invalid_argument);
  EXPECT_THROW((cal::StreamingCalibrator<cal::BundleProblem>(f.prob, xc, q, {})), std::invalid_argument);
  // the streamer is untouched by the refused inputs
  const cal::StreamTick t = sc.update(f.q0);
  EXPECT_TRUE(t.converged);
  EXPECT_LT((sc.current() - xc).cwiseAbs().maxCoeff(), 1e-12);
}

// Predictive convergence (Options::predict_convergence, E4.D 2026-09-10): the tick stops one step early when the
// observed contraction bounds the next step below step_tol / 10. The committed curve must stay within that bound of
// the fully iterated (non-predictive) reference on every tick, and the saving must be real (fewer steps).
TEST(StreamingContract, PredictiveConvergenceMatchesTheFullyIteratedReference) {
  const Chain f;
  const Eigen::VectorXd xc = cal::calibrate(f.prob, f.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem>::Options ref_opt;
  ref_opt.predict_convergence = false;
  cal::StreamingCalibrator<cal::BundleProblem> pred(f.prob, xc, f.q0, {});
  cal::StreamingCalibrator<cal::BundleProblem> ref(f.prob, xc, f.q0, ref_opt);
  const cal::HybridBundleResidual eng(f.prob);
  long steps_pred = 0, steps_ref = 0;
  double worst = 0.0, worst_rt = 0.0;
  for (int k = 1; k <= 200; ++k) {
    Eigen::VectorXd q = f.q0;
    for (int i = 0; i < q.size(); ++i) q[i] += 1e-5 * std::sin(0.05 * k + 0.3 * i) + 3e-4 * std::sin(0.01 * k);  // 0.1 bp jitter on a slow 3 bp swing
    const cal::StreamTick tp = pred.update(q), tr = ref.update(q);
    ASSERT_TRUE(tp.converged && tr.converged) << k << ": " << tp.reason() << " / " << tr.reason();
    steps_pred += tp.newton_steps;
    steps_ref += tr.newton_steps;
    worst = std::max(worst, (pred.current() - ref.current()).cwiseAbs().maxCoeff());
    worst_rt = std::max(worst_rt, eng.residuals_vs(pred.current(), q).cwiseAbs().maxCoeff());
  }
  std::cout << "  [predict] steps predictive " << steps_pred << " vs reference " << steps_ref << ", worst |x_pred - x_ref| "
            << worst << ", worst round-trip " << worst_rt << "\n";
  EXPECT_LT(worst, 1e-10);   // the skipped step: within step_tol / 10 of the fully iterated curve (measured ~6e-11)
  EXPECT_LT(worst_rt, 1e-9);  // the reprice stays at the step_tol contract
  EXPECT_LT(steps_pred, steps_ref) << "the prediction must actually skip steps";
}
