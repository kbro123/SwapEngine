// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T2 calibration (optimum / stationarity / recovery)
// Design §3 gate: the generic instrument / leg / role / quote model on the calibration problems.
//
// §2 gave the generic coupon and §4 the compiled batch; this file pins the PROBLEM-level wiring —
// that legs carry their own curve roles, that the three quote transforms are right, that the residual
// ORDER is what BundleProblem documents, and that the compiled W-cache path (values AND the analytic
// block Jacobian) agrees with the templated kernel and AAD on all of it.
//
// QuantLib-free by design: schedules are hand-built plain data, so no index/currency/calendar appears
// even in the test fixture. The QuantLib oracle for the extracted coupon SHAPES lives in
// extract_test.cpp; what is unproven until here is the problem-level assembly.
//
// Specifically covered, because each is a place a wiring bug hides while every other test stays green:
//   - ParRate / ParSpread / Rate model quotes vs their hand-written definitions
//   - the legacy `swaps`/`bases`/`comp_futs` groups re-expressed as generic Instruments -> the SAME
//     residuals (design §2's backward-compatibility invariant, at the instrument level)
//   - roles on LEGS: forecast != discount, and a basis whose two legs differ in FREQUENCY (design §6.7)
//   - residual order: legacy groups first, then instruments in INSERTION order with mixed quote kinds
//   - the compiled engine's row scatter, empty-`neg`-leg ParRate path and generic analytic Jacobian
//   - a mixed-frequency multi-curve CALIBRATION driven end-to-end by generic instruments only

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <iostream>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"

namespace px = swaps::pricing;
namespace cal = swaps::calibration;

namespace {

const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 10.0};
constexpr int kNk = 6;  // per curve: 1 front + 5 back

// Curve 0 = outright (the discount curve); curve 1 = an independent outright forecast curve.
std::vector<cal::BundleCurveSpec> two_outright() {
  const auto m = swaps::curve::flat_hermite(kMeeting, kBack);
  return {{.base = -1, .regions = m}, {.base = -1, .regions = m}};
}

Eigen::VectorXd stacked(double base0, double base1) {
  Eigen::VectorXd x(2 * kNk);
  for (int i = 0; i < kNk; ++i) {
    x[i] = base0 + 0.0009 * i;
    x[kNk + i] = base1 + 0.0006 * i;
  }
  return x;
}

// Period end times of an `n_per_year` schedule out to T, with a final stub if T is not a whole number
// of periods (T < one period => a single stub coupon). Every leg below is built from THIS, so a leg and
// its legacy OisSwap counterpart accumulate their times identically and can be compared bit for bit.
std::vector<double> period_ends(double T, double n_per_year) {
  std::vector<double> t;
  const double dt = 1.0 / n_per_year;
  for (double u = dt; u < T - 1e-9; u += dt) t.push_back(u);
  t.push_back(T);
  return t;
}

// A float leg of `n_per_year` compounded coupons out to T, forecasting `fc`, discounting `dc`.
// tau_pay == tau_index == 1 and the coupon amount is the DF ratio minus one -- the shape the legacy
// OisSwap encodes (the accrual is already inside the compounded growth factor). `spread` is
// contractual, additive and scaled by the period so it reads as an annualised rate.
cal::FloatLeg float_leg(double T, double n_per_year, int fc, int dc, double spread = 0.0) {
  cal::FloatLeg leg;
  leg.forecast = fc;
  leg.discount = dc;
  double prev = 0.0;
  for (double u : period_ends(T, n_per_year)) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = 1.0;
    c.pay = u;
    c.tau_pay = 1.0;
    c.spread = spread * (u - prev);
    leg.coupons.push_back(c);
    prev = u;
  }
  return leg;
}

cal::FixedLeg fixed_leg(double T, double n_per_year, int dc) {
  cal::FixedLeg leg;
  leg.discount = dc;
  double prev = 0.0;
  for (double u : period_ends(T, n_per_year)) {
    leg.coupons.push_back({u, u - prev});
    prev = u;
  }
  return leg;
}

cal::Instrument par_rate_inst(double T, double n, int fc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = float_leg(T, n, fc, dc);
  ins.fixed = fixed_leg(T, n, dc);
  return ins;
}

// A basis quote: `fwd` (the quoted leg, forecasting `fc`) against `bench` (forecasting `bc`), both
// discounting `dc`, on DIFFERENT frequencies -- exactly what the legacy Basis (one shared schedule)
// cannot express.
cal::Instrument par_spread_inst(double T, double n_fwd, double n_bench, int fc, int bc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParSpread;
  ins.fwd = float_leg(T, n_fwd, fc, dc);
  ins.bench = float_leg(T, n_bench, bc, dc);
  ins.fixed = fixed_leg(T, n_fwd, dc);
  return ins;
}

cal::Instrument rate_inst(double s, double e, double tau, int fc, double conv) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.obs.sub_start = {s};
  ins.obs.sub_end = {e};
  ins.obs.tau_index = tau;
  ins.forecast = fc;
  ins.convexity = conv;
  return ins;
}

}  // namespace

// ---- Quote transforms ------------------------------------------------------------------------
// Each transform vs its hand-written definition, so a wrong leg (or a swapped ParSpread sign) cannot
// hide behind the shared plumbing.
TEST(GenericInstrument, QuoteTransformsMatchTheirDefinitions) {
  const Eigen::VectorXd x = stacked(0.040, 0.045);
  const auto C = cal::build_bundle_curves<double>(two_outright(), [&](int c, int i) { return x[c * kNk + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };

  // ParRate: forecast curve 1, discount curve 0 (a genuinely multi-curve leg).
  const auto pr = par_rate_inst(5.0, 2.0, /*fc=*/1, /*dc=*/0);
  const double want_pr = px::float_leg_pv<double>(pr.fwd.coupons, *C[1], *C[0]) /
                         px::annuity<double>(pr.fixed.coupons, *C[0]);
  EXPECT_LT(std::abs(cal::instrument_model_quote<double>(pr, curve_of) - want_pr), 1e-16);

  // ParSpread: s = (pv_bench - pv_fwd)/annuity. Semi-annual quoted leg vs an ANNUAL benchmark leg.
  const auto ps = par_spread_inst(5.0, /*n_fwd=*/2.0, /*n_bench=*/1.0, /*fc=*/1, /*bc=*/0, /*dc=*/0);
  const double want_ps = (px::float_leg_pv<double>(ps.bench.coupons, *C[0], *C[0]) -
                          px::float_leg_pv<double>(ps.fwd.coupons, *C[1], *C[0])) /
                         px::annuity<double>(ps.fixed.coupons, *C[0]);
  EXPECT_LT(std::abs(cal::instrument_model_quote<double>(ps, curve_of) - want_ps), 1e-16);
  // Sign sanity: the quoted leg forecasts the HIGHER curve, so the benchmark pays less -> spread < 0.
  EXPECT_LT(want_ps, 0.0) << "ParSpread must be (bench - fwd), not (fwd - bench)";

  // Rate: rate(obs) + convexity, with convexity a pure INPUT number.
  const auto ri = rate_inst(1.0, 1.25, 0.2528, /*fc=*/1, /*conv=*/3.4e-4);
  const double want_ri = px::rate<double>(ri.obs, *C[1]) + 3.4e-4;
  EXPECT_LT(std::abs(cal::instrument_model_quote<double>(ri, curve_of) - want_ri), 1e-16);
  EXPECT_LT(std::abs(cal::instrument_model_quote<double>(ri, curve_of) -
                     (px::rate<double>(ri.obs, *C[1]) + ri.convexity)),
            1e-16);
}

// ---- Residual order --------------------------------------------------------------------------
// The order is a published contract (Jacobian rows, W-cache batches, market(), warm/streaming index
// off it): the instruments' INSERTION order, even with mixed quote kinds.
TEST(GenericInstrument, ResidualOrderIsInsertionOrder) {
  cal::BundleProblem p;
  p.curves = two_outright();
  // Deliberately interleaved kinds: Rate, ParRate, ParSpread, ParRate.
  p.instruments.push_back(rate_inst(2.0, 2.25, 0.2528, 1, 1e-4));
  p.instruments.push_back(par_rate_inst(3.0, 1.0, 1, 0));
  p.instruments.push_back(par_spread_inst(3.0, 2.0, 1.0, 1, 0, 0));
  p.instruments.push_back(par_rate_inst(10.0, 2.0, 1, 0));
  for (int i = 0; i < 4; ++i) p.instruments[i].market = 0.001 * (i + 1);  // distinct, so a swap shows

  const Eigen::VectorXd x = stacked(0.040, 0.045);
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[c * kNk + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  const Eigen::VectorXd r = p.residuals<double>(x);
  ASSERT_EQ(r.size(), 4);

  // Rows 0..3 must be the four instruments, in the order they were pushed.
  for (int i = 0; i < 4; ++i)
    EXPECT_LT(std::abs(r[i] - cal::instrument_residual<double>(p.instruments[i], curve_of)), 1e-16)
        << "generic residual row " << i << " must be instruments[" << i << "]";

  // market() must agree with the SAME order (it is what recovers model_rates = residuals + market).
  const Eigen::VectorXd m = p.market();
  ASSERT_EQ(m.size(), r.size());
  for (int i = 0; i < 4; ++i) EXPECT_EQ(m[i], p.instruments[i].market);
  const double dm = (cal::CompiledBundleResidual(p).model_rates(x) - (r + m)).cwiseAbs().maxCoeff();
  std::cout << "  [generic-inst] |compiled model_rates - (r + market)| = " << dm << "\n";
  EXPECT_LT(dm, 1e-14) << "compiled model_rates must fill the same rows as residuals() + market()";
}

namespace {
// A bundle exercising every generic degree of freedom at once, on top of the legacy groups: mixed
// quote kinds, mixed frequencies, forecast != discount, a contractual spread, and a Rate row.
cal::BundleProblem rich_bundle() {
  cal::BundleProblem p;
  p.curves = two_outright();
  p.instruments.push_back(par_rate_inst(7.0, 1.0, /*fc=*/1, /*dc=*/0));   // a plain annual OIS swap
  p.instruments.push_back(par_rate_inst(2.0, 4.0, /*fc=*/1, /*dc=*/0));   // quarterly, multi-curve
  p.instruments.push_back(rate_inst(0.75, 1.0, 0.2528, /*fc=*/1, 2.1e-4));
  p.instruments.push_back(par_spread_inst(5.0, 2.0, 1.0, /*fc=*/1, /*bc=*/0, /*dc=*/0));  // 2/y vs 1/y
  p.instruments.push_back(par_rate_inst(10.0, 1.0, /*fc=*/0, /*dc=*/0));   // self-discounting
  // A ParRate whose float leg carries a contractual spread and a 30/360-style payment basis
  // (tau_pay != tau_index), i.e. k != 1 on every coupon.
  cal::Instrument sp = par_rate_inst(3.0, 2.0, /*fc=*/1, /*dc=*/0);
  for (auto& c : sp.fwd.coupons) {
    c.spread = 25e-4;
    c.obs.tau_index = 0.5069444;  // ACT/360-flavoured index accrual
    c.tau_pay = 0.5;              // 30/360-flavoured payment accrual
  }
  p.instruments.push_back(sp);
  for (std::size_t i = 0; i < p.instruments.size(); ++i) p.instruments[i].market = 0.0005 * (i + 1);
  return p;
}
}  // namespace

// ---- Compiled engine vs the templated kernel and AAD -------------------------------------------
TEST(GenericInstrument, CompiledResidualAndJacobianMatchTheKernelAndAad) {
  const cal::BundleProblem p = rich_bundle();
  const Eigen::VectorXd x = stacked(0.040, 0.045);
  const cal::CompiledBundleResidual cr(p);
  ASSERT_EQ(cr.n_residuals(), p.n_residuals());

  const double dr = (cr.residuals(x) - p.residuals<double>(x)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(x) - cal::aad_jacobian(p, x)).cwiseAbs().maxCoeff();
  std::cout << "  [generic-inst] compiled |residual - kernel| = " << dr << "  |Jacobian - AAD| = " << dj
            << "\n";
  EXPECT_LT(dr, 1e-14) << "the W-cache path must agree with the templated kernel on generic instruments";
  EXPECT_LT(dj, 1e-9) << "the analytic block Jacobian must match AAD on every generic quote transform";

  // Block structure: instruments[4] is a self-discounting ParRate on curve 0 (residual row 4), so it
  // must have an identically-zero derivative w.r.t. curve 1's knots. This is the property the
  // staged/bundle solvers rely on.
  const Eigen::MatrixXd J = cr.jacobian(x);
  ASSERT_EQ(p.instruments[4].primary_curve(), 0);
  EXPECT_EQ(J.row(4).tail(kNk).cwiseAbs().maxCoeff(), 0.0)
      << "an instrument touching only curve 0 must not move curve 1's block";
}

// A ParRate-only bundle takes the compiled path where EVERY subtracted leg is empty -- its own edge
// case (a float batch with no coupons at all).
TEST(GenericInstrument, CompiledHandlesAParRateOnlyBundle) {
  cal::BundleProblem p;
  p.curves = two_outright();
  for (double T : {1.0, 2.0, 5.0}) p.instruments.push_back(par_rate_inst(T, 2.0, 1, 0));
  const Eigen::VectorXd x = stacked(0.040, 0.045);
  const cal::CompiledBundleResidual cr(p);
  const double dr = (cr.residuals(x) - p.residuals<double>(x)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(x) - cal::aad_jacobian(p, x)).cwiseAbs().maxCoeff();
  std::cout << "  [generic-inst] par-rate-only |residual - kernel| = " << dr << " |J - AAD| = " << dj
            << "\n";
  EXPECT_LT(dr, 1e-14);
  EXPECT_LT(dj, 1e-9);
}

// ---- End-to-end: a calibration driven by generic instruments only -------------------------------
// Design §6.7: a semi-annual basis + an annual outright in ONE calibration. Curve 0 is pinned by its
// own annual par rates; curve 1 by semi-annual basis quotes against curve 0, discounting curve 0.
TEST(GenericInstrument, MixedFrequencyMultiCurveCalibrationRecoversTheCurves) {
  cal::BundleProblem p;
  p.curves = two_outright();
  const std::vector<double> mats{0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
  for (double T : mats) p.instruments.push_back(par_rate_inst(T, 1.0, /*fc=*/0, /*dc=*/0));
  for (double T : mats) p.instruments.push_back(par_spread_inst(T, 2.0, 1.0, /*fc=*/1, /*bc=*/0, /*dc=*/0));

  // Make x_true the exact solution by absorbing the model quotes into the market.
  const Eigen::VectorXd x_true = stacked(0.040, 0.046);
  const Eigen::VectorXd r0 = p.residuals<double>(x_true);
  for (std::size_t i = 0; i < p.instruments.size(); ++i) p.instruments[i].market += r0[static_cast<int>(i)];
  ASSERT_LT(p.residuals<double>(x_true).cwiseAbs().maxCoeff(), 1e-14);

  Eigen::VectorXd x0(2 * kNk);
  x0.head(kNk).setConstant(0.04);
  x0.tail(kNk).setConstant(0.04);
  const auto joint = cal::calibrate(p, x0);
  const auto staged = cal::calibrate_staged(p, x0);
  const double dj = (joint.x - x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - x_true).cwiseAbs().maxCoeff();
  std::cout << "  [generic-inst] joint ||x*-xtrue||=" << dj << " iters=" << joint.iterations
            << " stat=" << joint.stationarity << "  staged ||x*-xtrue||=" << ds
            << " iters=" << staged.iterations << "\n";
  EXPECT_LT(dj, 1e-8) << "joint solve must recover both curves from generic instruments alone";
  EXPECT_LT(ds, 1e-8) << "staged solve must see the generic instruments (not silently drop them)";
}

// ---- The single-curve problem carries the generic model too --------------------------------------
TEST(GenericInstrument, SingleCurveProblemPricesGenericInstruments) {
  cal::CalibrationProblem p;
  p.meeting_times = kMeeting;
  p.back_times = kBack;
  p.instruments.push_back(par_rate_inst(5.0, 1.0, 0, 0));  // a plain annual OIS swap (single curve)
  for (double T : {2.0, 7.0}) p.instruments.push_back(par_rate_inst(T, 2.0, 0, 0));
  p.instruments.push_back(rate_inst(1.0, 1.25, 0.2528, 0, 3.4e-4));
  p.instruments[0].market = 0.002;

  Eigen::VectorXd x(kNk);
  for (int i = 0; i < kNk; ++i) x[i] = 0.040 + 0.0009 * i;
  ASSERT_EQ(p.n_residuals(), 4);

  // The compiled single-curve delegate must agree with the templated path -- including forcing every
  // generic leg role onto the one curve.
  const cal::CompiledResidual cr(p);
  const double dr = (cr.residuals(x) - p.residuals<double>(x)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(x) - cal::aad_jacobian(p, x)).cwiseAbs().maxCoeff();
  std::cout << "  [generic-inst] single-curve |residual - kernel| = " << dr << " |J - AAD| = " << dj
            << "\n";
  EXPECT_LT(dr, 1e-14);
  EXPECT_LT(dj, 1e-9);
}
