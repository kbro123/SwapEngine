// E5 taxonomy: T6 regression (fails on the reverted bug)
// KERNEL EDGES (E4.C, 2026-09-10): malformed inputs are refused with a message instead of crashing, pricing
// NaN or silently corrupting the compiled tables; and two rows that could not ride the hybrid engine now do.
//   E1  "regions": [] built an empty curve and segfaulted inside run_json;
//   E2  a knot at t <= 0 built silently and corrupted the integral / W (~49 bp of log-DF) while calibration
//       reported success;
//   A1  a moment-path future (fixing_step > 0) made calibrate() throw (regression pin: the compiled moment path
//       prices it since 9f71e87);
//   B11 a TurnJump row could not ride the AAD block (a MonotoneCubic region sends every row there) -- the
//       block's curve wrappers did not forward turn_jump and the turn's curve was not in the touched set;
//   B12 an instrument with an empty leg priced to NaN on both routes without a throw.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
cal::Instrument par_inst(double T, int c) {
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon f; f.obs.sub_start = {prev}; f.obs.sub_end = {u}; f.obs.tau_index = u - prev; f.pay = u; f.tau_pay = u - prev;
    in.fwd.coupons.push_back(f);
    in.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  in.fwd.forecast = in.fwd.discount = in.fixed.discount = c;
  return in;
}
cal::BundleProblem square(std::vector<double> knots = {1, 2, 3, 5}) {
  cal::BundleProblem p;
  p.curves = {px::CurveStructure{.base = -1, .regions = cv::flat_hermite({}, knots)}};
  for (double T : {1.0, 2.0, 3.0, 5.0}) p.instruments.push_back(par_inst(T, 0));
  const Eigen::VectorXd x = Eigen::VectorXd::Constant(p.n_knots(), 0.03);
  const Eigen::VectorXd r = p.residuals<double>(x);
  for (int i = 0; i < p.n_residuals(); ++i) p.instruments[i].market += r[i];
  return p;
}
}  // namespace

TEST(KernelEdges, EmptyRegionsAreRejectedNotBuilt) {
  EXPECT_THROW(cv::make_modular_curve<double>({}), std::invalid_argument);
  EXPECT_THROW(cv::make_modular_curve<double>({cv::CurveModule{{}, cv::Scheme::Hermite}}), std::invalid_argument);
  cal::BundleProblem p = square();
  p.curves[0].regions.clear();
  EXPECT_THROW(cal::validate_problem(p), std::invalid_argument);
  EXPECT_THROW((cal::HybridBundleResidual(p)), std::invalid_argument);
  EXPECT_THROW(cal::calibrate(p, Eigen::VectorXd::Zero(0)), std::invalid_argument);
}

TEST(KernelEdges, KnotAtOrBelowZeroIsRejected) {
  for (double t0 : {0.0, -0.5}) {
    EXPECT_THROW(cv::make_modular_curve<double>({cv::CurveModule{{t0, 1.0, 2.0}, cv::Scheme::Hermite}}), std::invalid_argument) << t0;
    EXPECT_THROW(cv::make_modular_curve<double>({cv::CurveModule{{t0}, cv::Scheme::Flat}, cv::CurveModule{{1.0, 2.0}, cv::Scheme::Hermite}}), std::invalid_argument) << t0;
    cal::BundleProblem p = square();
    p.curves[0].regions.front().knots.front() = t0;
    EXPECT_THROW(cal::validate_problem(p), std::invalid_argument) << t0;
    EXPECT_THROW((cal::HybridBundleResidual(p)), std::invalid_argument) << t0;
  }
  // the control still builds and calibrates
  const cal::BundleProblem ok = square();
  EXPECT_NO_THROW(cal::validate_problem(ok));
  EXPECT_LT(cal::calibrate(ok, Eigen::VectorXd::Constant(4, 0.02)).rms_residual, 1e-12);
}

// A1 regression: the moment-path future (fixing_step > 0) rides the hybrid engine (compiled moment path).
TEST(KernelEdges, MomentPathFutureCalibratesOnTheHybridEngine) {
  cal::BundleProblem p;
  p.curves = {px::CurveStructure{.base = -1, .regions = cv::flat_hermite({0.25, 0.5}, {1, 2, 3, 5})}};
  for (double T : {1.0, 2.0, 3.0, 5.0}) p.instruments.push_back(par_inst(T, 0));
  cal::Instrument fut; fut.quote = cal::QuoteKind::Rate; fut.forecast = 0;
  fut.obs.sub_start = {0.25}; fut.obs.sub_end = {0.5}; fut.obs.tau_index = 0.25; fut.obs.fixing_step = 1.0 / 360.0;
  p.instruments.push_back(fut);
  const Eigen::VectorXd x = Eigen::VectorXd::Constant(p.n_knots(), 0.03);
  const Eigen::VectorXd r = p.residuals<double>(x);
  for (int i = 0; i < p.n_residuals(); ++i) p.instruments[i].market += r[i];
  cal::HybridBundleResidual eng(p);
  EXPECT_LT(std::abs(eng.residuals(x)[4]), 1e-12) << "compiled moment path == templated at the generating state";
  const cal::CalibrationResult res = cal::calibrate(p, Eigen::VectorXd::Constant(p.n_knots(), 0.02));
  EXPECT_TRUE(res.converged) << res.status;
  EXPECT_LT(res.rms_residual, 1e-12);
}

// B11: a MonotoneCubic region routes EVERY row to the AAD block; a TurnJump row must still price (turn_jump is
// forwarded by the block's curve wrappers) and carry its delta column (the turn's curve is in the touched set).
TEST(KernelEdges, TurnJumpRidesTheAadBlock) {
  const std::vector<double> back{0.5, 1, 2, 3, 5};
  cal::BundleProblem p;
  p.curves.resize(2);
  p.curves[0] = px::CurveStructure{.base = -1, .currency = 0,
                                   .regions = {cv::CurveModule{{0.25}, cv::Scheme::Flat}, cv::CurveModule{back, cv::Scheme::MonotoneCubic}}};
  p.curves[1] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({0.25}, back)};
  p.curves[1].turns = {px::Turn{0.95, 1.05}};
  for (double T : {1.0, 2.0, 3.0, 5.0}) p.instruments.push_back(par_inst(T, 0));
  for (double T : {1.0, 2.0, 3.0, 5.0}) p.instruments.push_back(par_inst(T, 1));
  cal::Instrument tj; tj.quote = cal::QuoteKind::TurnJump; tj.turn_curve = 1; tj.turn_index = 0; tj.market = 0.001;
  p.instruments.push_back(tj);
  ASSERT_TRUE(cal::curves_are_noncacheable(p.curves));
  Eigen::VectorXd xt(p.n_knots());
  for (int i = 0; i < p.curves[0].n_knots(); ++i) xt[i] = 0.04 + 0.001 * i;
  const int o1 = p.offset(1), ni1 = p.curves[1].n_interp_knots();
  for (int i = 0; i < ni1; ++i) xt[o1 + i] = 0.03 + 0.001 * i;
  xt[o1 + ni1] = 0.0004;  // the turn δ
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < p.n_residuals(); ++i) p.instruments[i].market += r[i];
  ASSERT_NO_THROW((cal::HybridBundleResidual(p)));
  const cal::HybridBundleResidual eng(p);
  const Eigen::MatrixXd J = eng.jacobian(xt);
  const int row = p.n_residuals() - 1, delta_col = o1 + ni1;
  EXPECT_NEAR(J(row, delta_col), 1.0, 1e-12);
  Eigen::VectorXd rest = J.row(row).transpose(); rest[delta_col] = 0.0;
  EXPECT_LT(rest.cwiseAbs().maxCoeff(), 1e-12) << "the TurnJump row depends on its delta only";
  EXPECT_LT(eng.residuals(xt).cwiseAbs().maxCoeff(), 1e-12);
  Eigen::VectorXd x0 = xt; x0.array() += 0.002; x0[delta_col] = 0.0;
  const cal::CalibrationResult res = cal::calibrate(p, x0);
  EXPECT_TRUE(res.converged) << res.status;
  EXPECT_NEAR(res.x[delta_col], 0.0004, 1e-9) << "the turn δ is recovered through the AAD block";

  // B11 proper, router-independent: put the TurnJump row on the AAD block by hand and demand the same
  // residual and delta column from it.
  cal::AadBlock blk;
  blk.init(p.curves, {p.instruments[static_cast<std::size_t>(row)]}, {row}, p.n_knots());
  Eigen::VectorXd rb = Eigen::VectorXd::Zero(p.n_residuals());
  blk.residuals_into(xt, rb);
  EXPECT_LT(std::abs(rb[row]), 1e-12) << "the block prices the turn jump, not NaN";
  Eigen::MatrixXd Jb = Eigen::MatrixXd::Zero(p.n_residuals(), p.n_knots());
  blk.jacobian_into(xt, Jb);
  EXPECT_NEAR(Jb(row, delta_col), 1.0, 1e-12) << "the turn's curve is in the block's touched set";
  Eigen::VectorXd rest_b = Jb.row(row).transpose();
  rest_b[delta_col] = 0.0;
  EXPECT_LT(rest_b.cwiseAbs().maxCoeff(), 1e-12);
}

// B12: an instrument with an empty leg / bad shape is refused at engine construction, never priced to NaN.
TEST(KernelEdges, EmptyLegsAndBadShapesAreRejected) {
  const auto expect_bad = [](cal::BundleProblem p, const char* what) {
    EXPECT_THROW(cal::validate_problem(p), std::invalid_argument) << what;
    EXPECT_THROW((cal::HybridBundleResidual(p)), std::invalid_argument) << what;
  };
  { cal::BundleProblem p = square(); p.instruments[1].fwd.coupons.clear(); expect_bad(p, "empty float leg"); }
  { cal::BundleProblem p = square(); p.instruments[1].fixed.coupons.clear(); expect_bad(p, "empty fixed leg"); }
  { cal::BundleProblem p = square(); p.instruments[1].fwd.coupons[0].obs.tau_index = 0.0; expect_bad(p, "tau_index 0"); }
  { cal::BundleProblem p = square(); p.instruments[1].fwd.forecast = 3; expect_bad(p, "forecast curve out of range"); }
  { cal::BundleProblem p = square(); p.curves[0].base = 0; expect_bad(p, "self-referential base"); }
  { cal::BundleProblem p = square(); cal::Instrument pf; pf.quote = cal::QuoteKind::Portfolio; p.instruments.push_back(pf); expect_bad(p, "empty portfolio"); }
  { cal::BundleProblem p = square(); cal::Instrument sp = p.instruments[0]; sp.quote = cal::QuoteKind::ParSpread; p.instruments.push_back(sp); expect_bad(p, "ParSpread with no benchmark leg"); }
  { cal::BundleProblem p = square(); cal::Instrument fx; fx.quote = cal::QuoteKind::FxForward; fx.fx_num = 0; fx.fx_den = 0; fx.fx_spot = 1.1; fx.fx_time = 0.0; p.instruments.push_back(fx); expect_bad(p, "FX forward at t = 0"); }
  { cal::BundleProblem p = square(); cal::Instrument tj; tj.quote = cal::QuoteKind::TurnJump; tj.turn_curve = 0; tj.turn_index = 0; p.instruments.push_back(tj); expect_bad(p, "TurnJump on a curve with no turns"); }
  { cal::BundleProblem p = square(); cal::Instrument r; r.quote = cal::QuoteKind::Rate; r.forecast = 0; r.obs.sub_start = {0.1}; r.obs.sub_end = {0.35}; r.obs.tau_index = 0.0; p.instruments.push_back(r); expect_bad(p, "Rate with tau_index 0"); }
  // the pre-fix symptom, for the record: the templated residual of a par rate with no fixed leg is NaN (0/0)
  cal::BundleProblem p = square(); p.instruments[1].fwd.coupons.clear(); p.instruments[1].fixed.coupons.clear();
  EXPECT_TRUE(std::isnan(p.residuals<double>(Eigen::VectorXd::Constant(4, 0.03))[1]));
}
