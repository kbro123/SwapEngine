// Validation for the clamped cubic B-spline region (docs/bezier-and-moments.md, Part A).
// Engine-only (no QuantLib): checks the region contract and the B-spline's defining properties --
// C0 clamp to the boundary, partition-of-unity (constant control points -> flat forward), the
// convex-hull property, linearity in the control points, and integral() vs a dense quadrature.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled.hpp"

using swaps::curve::BSpline;
using swaps::curve::Boundary;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
const std::vector<double> kKnots{2, 3, 5, 7, 10};  // n = 5 -> 6 control points, 3 segments
constexpr double kT0 = 1.0, kV0 = 0.032, kI0 = 0.05;

BSpline<double> make(const Eigen::VectorXd& free_cp, double v0 = kV0) {
  BSpline<double> b(kKnots);
  Boundary<double> in{kT0, v0, 0.0, kI0};
  b.build(free_cp, 0, static_cast<int>(free_cp.size()), in);
  return b;
}
Eigen::VectorXd cp() {
  Eigen::VectorXd c(5);
  c << 0.030, 0.037, 0.041, 0.036, 0.044;
  return c;
}
// Dense trapezoidal integral of forward from t0 to t (reference for integral()).
double quad(const BSpline<double>& b, double t) {
  const int N = 200000;
  const double h = (t - kT0) / N;
  double s = 0.5 * (b.forward(kT0) + b.forward(t));
  for (int i = 1; i < N; ++i) s += b.forward(kT0 + i * h);
  return kI0 + s * h;
}
}  // namespace

TEST(BSpline, ClampedStartPinsBoundary) {
  const auto b = make(cp());
  EXPECT_NEAR(b.forward(kT0), kV0, 1e-12) << "clamped B-spline must interpolate the pinned boundary value";
  EXPECT_NEAR(b.integral(kT0), kI0, 1e-15) << "integral at the join must equal the incoming integral";
}

TEST(BSpline, ConstantControlPointsAreFlat) {
  // All control points (incl. the pinned P0) equal c => partition of unity gives f == c, and the
  // integral is exactly c*(t - t0). This pins down de Boor + the clamped knot vector.
  const double c = 0.039;
  const auto b = make(Eigen::VectorXd::Constant(5, c), /*v0=*/c);
  for (double t : {1.0, 1.7, 3.3, 5.0, 8.1, 10.0, 12.0}) {
    EXPECT_NEAR(b.forward(t), c, 1e-12) << " at t=" << t;
    EXPECT_NEAR(b.integral(t), kI0 + c * (t - kT0), 1e-12) << " at t=" << t;
  }
}

TEST(BSpline, IntegralMatchesDenseQuadrature) {
  const auto b = make(cp());
  double worst = 0;
  for (double t : {1.5, 2.0, 4.0, 6.5, 9.0, 10.0}) worst = std::max(worst, std::abs(b.integral(t) - quad(b, t)));
  std::cout << "  [bspline] max |integral - dense quadrature| = " << worst << "\n";
  EXPECT_LT(worst, 1e-7) << "2-pt Gauss is exact for the cubic; residual is the reference's trapezoid error";
}

TEST(BSpline, ConvexHull) {
  const Eigen::VectorXd c = cp();
  const auto b = make(c);
  const double lo = std::min(kV0, c.minCoeff()), hi = std::max(kV0, c.maxCoeff());
  double fmin = 1e9, fmax = -1e9;
  for (int i = 0; i <= 900; ++i) {
    const double f = b.forward(kT0 + (10.0 - kT0) * i / 900.0);
    fmin = std::min(fmin, f);
    fmax = std::max(fmax, f);
  }
  std::cout << "  [bspline] forward range [" << fmin << ", " << fmax << "] within hull [" << lo << ", " << hi << "]\n";
  EXPECT_GE(fmin, lo - 1e-12) << "convex-hull property: forward cannot dip below the min control point";
  EXPECT_LE(fmax, hi + 1e-12) << "convex-hull property: forward cannot exceed the max control point";
}

TEST(BSpline, IntegralIsLinearInControlPoints) {
  // integral(t) is affine in the free control points (fixed boundary), so it preserves affine
  // combinations -- this is exactly the is_linear_map promise the W-cache relies on.
  const Eigen::VectorXd a = cp(), d = cp().reverse();
  const double alpha = 0.3;
  const auto ba = make(a), bd = make(d), bm = make((alpha * a + (1 - alpha) * d).eval());
  double worst = 0;
  for (double t : {1.5, 3.0, 5.0, 7.5, 10.0})
    worst = std::max(worst, std::abs(bm.integral(t) - (alpha * ba.integral(t) + (1 - alpha) * bd.integral(t))));
  std::cout << "  [bspline] max linearity residual = " << worst << "\n";
  EXPECT_LT(worst, 1e-13);
}

TEST(BSpline, ComposesIntoAValidCurve) {
  // MultiRegionCurve<Flat, BSpline>: flat meeting-date front + control-point B-spline back. Must be a
  // valid discount curve -- DF(0)=1, positive & strictly decreasing (positive forwards), C0 across the
  // Flat->BSpline join -- and stay on the linear-map fast path.
  static_assert(swaps::curve::BSplineCurve<double>::is_linear_map, "B-spline curve must be a linear map");
  const std::vector<double> meeting{0.25, 0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = swaps::curve::make_bspline_curve<double>(meeting, back);
  Eigen::VectorXd x(8);  // 2 front forwards + 6 back control points
  x << 0.030, 0.033, 0.036, 0.040, 0.038, 0.042, 0.041, 0.045;
  c.set_forwards(x);

  EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0);
  double prev = 1.0;
  for (double t = 0.1; t <= 12.0; t += 0.1) {
    const double df = c.discount(t);
    EXPECT_GT(df, 0.0) << " DF must stay positive at t=" << t;
    EXPECT_LT(df, prev + 1e-15) << " DF must not increase (positive forward) at t=" << t;
    prev = df;
  }
  // C0 join: forward just left/right of the last meeting date agrees.
  const double h = 1e-6;
  EXPECT_NEAR(c.forward(0.5 - h), c.forward(0.5 + h), 1e-3) << "forward continuous across the front/back join";
  std::cout << "  [bspline-curve] DF(1y)=" << c.discount(1.0) << " DF(10y)=" << c.discount(10.0) << "\n";
}

TEST(BSpline, SecondMomentMatchesQuadrature) {
  // integral2(a,b) = int_a^b f^2 -- the moment scheme's convexity term. 4-pt Gauss is exact for f^2
  // (degree 6); the residual vs a dense quadrature reference is the reference's own error.
  const auto b = make(cp());
  auto quad2 = [&](double a, double t) {
    const int N = 200000;
    const double h = (t - a) / N;
    double s = 0.5 * (b.forward(a) * b.forward(a) + b.forward(t) * b.forward(t));
    for (int i = 1; i < N; ++i) {
      const double f = b.forward(a + i * h);
      s += f * f;
    }
    return s * h;
  };
  double worst = 0;
  for (auto ab : {std::pair{1.5, 4.0}, std::pair{2.0, 9.0}, std::pair{3.3, 7.7}, std::pair{1.0, 10.0}})
    worst = std::max(worst, std::abs(b.integral2(ab.first, ab.second) - quad2(ab.first, ab.second)));
  std::cout << "  [bspline] max |integral2 - dense quadrature| = " << worst << "\n";
  EXPECT_LT(worst, 1e-8) << "4-pt Gauss is exact for the per-segment f^2 (degree 6)";
}

// A calibration problem whose curve is a B-SPLINE (control-point). Duck-types CalibrationProblem
// (residuals<Scalar> / n_knots / n_residuals), so the generic LM + AAD Jacobian drive it unchanged --
// the same trick spread/bundle problems use. This is the real curve-FIT proof (vs the pricing-only
// oracle): the calibrated B-spline curve must reprice the market it was fit to.
namespace {
struct BSplineProblem {
  cal::CalibrationProblem inst;  // instruments + knot times (front meetings, back control-point knots)
  int n_knots() const { return inst.n_knots(); }
  int n_residuals() const { return inst.n_residuals(); }
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = cv::make_bspline_curve<Scalar>(inst.meeting_times, inst.back_times);
    c.set_forwards(x);
    return inst.price_residuals<Scalar>(c);
  }
};
// An annual OIS as a generic ParRate instrument (single coupon if T < 1), self-discounting.
cal::Instrument make_ois(double T) {
  std::vector<double> ends;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) ends.push_back(u);
  ends.push_back(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u : ends) {
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
  return ins;
}
}  // namespace

TEST(BSpline, CalibratesToMarketAndReprices) {
  BSplineProblem prob;
  prob.inst.meeting_times = {0.5};
  prob.inst.back_times = {1, 2, 3, 4, 5, 7, 10};  // 7 back control points + 1 front = 8 free vars
  const std::vector<double> mats{0.5, 1, 2, 3, 4, 5, 7, 10};  // 8 instruments (square, consistent)
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  // Self-consistent market generated from a KNOWN B-spline curve, so x_true is the exact solution.
  Eigen::VectorXd xt(8);
  xt << 0.030, 0.033, 0.036, 0.039, 0.041, 0.043, 0.044, 0.046;
  auto ct = cv::make_bspline_curve<double>(prob.inst.meeting_times, prob.inst.back_times);
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-13) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(8, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [bspline-calib] reprice=" << reprice << " ||x*-xt||=" << (res.x - xt).cwiseAbs().maxCoeff()
            << " stat=" << res.stationarity << " iters=" << res.iterations << "\n";
  EXPECT_LT(reprice, 1e-9) << "calibrated B-spline curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-7) << "first-order optimality";
}

TEST(BSpline, MomentAverageRateMatchesExactDailySum) {
  // The moment-integrated averaging rate must match the EXACT day-by-day arithmetic sum to the gate,
  // with NO per-day work -- the whole point of Part B. Uniform daily fixings so the moment coefficient
  // is exact (fixing_step = window/ndays); a real calendar's weekend day-count moment is an additive
  // refinement (see the header). Validated on a B-spline curve.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_bspline_curve<double>(meeting, back);
  Eigen::VectorXd cpv(7);
  cpv << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  c.set_forwards(cpv);

  double worst = 0;
  for (auto win : {std::pair{1.0, 1.0 + 1.0 / 12}, std::pair{2.0, 2.25}, std::pair{3.0, 4.0}}) {
    const double a = win.first, b = win.second, T = b - a;
    const int nd = std::max(1, static_cast<int>(std::round(T * 360)));
    const double step = T / nd;
    double exact_num = 0;
    for (int d = 0; d < nd; ++d) {
      const double t0 = a + d * step, t1 = a + (d + 1) * step;
      exact_num += c.discount(t0) / c.discount(t1) - 1.0;
    }
    const double exact = exact_num / T;
    const double moment = px::moment_average_rate<double>(c, a, b, step, T);
    worst = std::max(worst, std::abs(moment - exact) / std::max(1.0, std::abs(exact)));
  }
  std::cout << "  [bspline-moment] max rel |moment avg - exact daily| = " << worst << "\n";
  EXPECT_LT(worst, 1e-9) << "2-moment averaging must match the exact daily arithmetic sum to the gate";
}

TEST(BSpline, MomentCouponPricesThroughFloatCouponPv) {
  // The moment path wired into the coupon model: a FloatCoupon with fixing_step>0 must price (through
  // float_coupon_pv) equal to an exact daily-averaged coupon -- proving obs_numerator routing works
  // end-to-end, not just the standalone primitive. tau_pay == tau_index, no spread => pv = DF(pay)*num.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_bspline_curve<double>(meeting, back);
  Eigen::VectorXd cpv(7);
  cpv << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  c.set_forwards(cpv);

  const double a = 2.0, b = 3.0, T = b - a;
  const int nd = static_cast<int>(std::round(T * 360));
  px::FloatCoupon mc;
  mc.pay = b;
  mc.tau_pay = T;
  mc.obs.sub_start = {a};
  mc.obs.sub_end = {b};
  mc.obs.tau_index = T;
  mc.obs.fixing_step = T / nd;  // selects the moment path

  const double ours = px::float_coupon_pv<double>(mc, c, c);
  double exact_num = 0;
  for (int d = 0; d < nd; ++d)
    exact_num += c.discount(a + d * (T / nd)) / c.discount(a + (d + 1) * (T / nd)) - 1.0;
  const double exact_pv = c.discount(b) * exact_num;  // avg_rate*tau_pay discounted; tau_pay==tau_index
  std::cout << "  [bspline-moment-coupon] rel err = " << std::abs(ours - exact_pv) / std::max(1.0, std::abs(exact_pv)) << "\n";
  EXPECT_LT(std::abs(ours - exact_pv) / std::max(1.0, std::abs(exact_pv)), 1e-9);
}

TEST(BSpline, WCacheReproducesDiscounts) {
  // The B-spline curve on the compiled fast path: integral_weight_matrix(scheme=BSpline) builds W once
  // (control points are the free vars), then DF = exp(-W x) must equal the direct curve.discount() --
  // proving a B-spline curve reprices through the W-cache with NO curve rebuild, the whole point of
  // fast B-spline calibration.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_bspline_curve<double>(meeting, back);
  Eigen::VectorXd x(7);
  x << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  c.set_forwards(x);
  const std::vector<double> times{0.3, 0.8, 1.5, 3.0, 6.0, 9.5, 10.0};
  const Eigen::MatrixXd W = px::integral_weight_matrix(meeting, back, times, px::BackScheme::BSpline);
  const Eigen::VectorXd DF = (-(W * x).array()).exp();
  double worst = 0;
  for (std::size_t i = 0; i < times.size(); ++i) worst = std::max(worst, std::abs(DF[i] - c.discount(times[i])));
  std::cout << "  [bspline-wcache] max |exp(-Wx) - curve.discount| = " << worst << "\n";
  EXPECT_LT(worst, 1e-13) << "B-spline curve must reprice exactly through the W-cache";
}

TEST(BSpline, CollocationMapsControlPointsToForwardsAndIsInvertible) {
  // The risk transform: B x = forward-at-knot values (control-point basis -> forward basis). Must map
  // correctly AND be invertible so a control-point risk ladder can be presented as forward-at-knot deltas.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  const Eigen::MatrixXd B = px::bspline_collocation(meeting, back);
  const Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
  ASSERT_EQ(lu.rank(), B.rows()) << "collocation must be invertible to convert risk between bases";

  Eigen::VectorXd x(7);
  x << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  auto c = cv::make_bspline_curve<double>(meeting, back);
  c.set_forwards(x);
  const Eigen::VectorXd fwd = B * x;
  const std::vector<double> knots{0.5, 1, 2, 3, 5, 7, 10};
  double worst = 0;
  for (std::size_t i = 0; i < knots.size(); ++i) worst = std::max(worst, std::abs(fwd[i] - c.forward(knots[i])));
  std::cout << "  [bspline-collocation] max |Bx - forward@knot| = " << worst << " rank=" << lu.rank() << "\n";
  EXPECT_LT(worst, 1e-12) << "B x must equal the forward at each knot";
  // Round-trip: recover control points from forward values via B^{-1}.
  EXPECT_LT((lu.solve(fwd) - x).cwiseAbs().maxCoeff(), 1e-10) << "B^{-1} (B x) == x";
}

TEST(BSpline, ForwardIsC1) {
  const auto b = make(cp());
  const double h = 1e-5;
  for (double bp : {3.0, 5.0}) {  // interior breakpoints of this knot set
    const double dl = (b.forward(bp) - b.forward(bp - h)) / h;
    const double dr = (b.forward(bp + h) - b.forward(bp)) / h;
    EXPECT_NEAR(dl, dr, 1e-3) << "forward slope must be continuous across breakpoint " << bp;
  }
}
