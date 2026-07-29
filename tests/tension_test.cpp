// Validation for the spline-under-tension region (docs/tension-spline-research.md).
// Engine-only (no QuantLib): checks the region contract and the tension spline's defining properties --
// the sigma->0 reduction to the natural cubic, the sigma->inf taut/piecewise-linear limit, exact knot
// interpolation, linearity in the knot forwards (the is_linear_map promise), integral() vs quadrature,
// composition into a valid discount curve, W-cache reproduction, and an end-to-end curve fit.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled.hpp"

using swaps::curve::Boundary;
using swaps::curve::NaturalCubic;
using swaps::curve::Tension;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
const std::vector<double> kBack{1, 2, 3, 5, 7, 10, 15, 20, 30};
constexpr double kT0 = 0.5, kV0 = 0.030, kI0 = 0.015;

Eigen::VectorXd fwds() {
  Eigen::VectorXd y(9);
  y << 0.031, 0.034, 0.036, 0.039, 0.041, 0.043, 0.044, 0.045, 0.046;
  return y;
}
Tension<double> makeT(const Eigen::VectorXd& y, double sigma) {
  Tension<double> t(kBack, sigma);
  Boundary<double> in{kT0, kV0, 0.0, kI0};
  t.build(y, 0, static_cast<int>(y.size()), in);
  return t;
}
NaturalCubic<double> makeNC(const Eigen::VectorXd& y) {
  NaturalCubic<double> c(kBack);
  Boundary<double> in{kT0, kV0, 0.0, kI0};
  c.build(y, 0, static_cast<int>(y.size()), in);
  return c;
}
}  // namespace

TEST(Tension, SigmaToZeroReproducesNaturalCubic) {
  // The defining limit: as sigma -> 0 the hyperbolic pieces become cubics and the tension tridiagonal
  // reduces to NaturalCubic's, so forward AND integral converge to the natural cubic at rate O(sigma^2).
  const Eigen::VectorXd y = fwds();
  const auto nc = makeNC(y);
  const auto t = makeT(y, 1e-5);
  double wf = 0, wi = 0;
  for (double p = kT0 + 1e-3; p <= 30.0; p += 0.137) {
    wf = std::max(wf, std::abs(t.forward(p) - nc.forward(p)));
    wi = std::max(wi, std::abs(t.integral(p) - nc.integral(p)));
  }
  std::cout << "  [tension] sigma=1e-5 vs natural cubic: max|df|=" << wf << " max|dI|=" << wi << "\n";
  EXPECT_LT(wf, 1e-9) << "tension forward must reduce to the natural cubic as sigma->0";
  EXPECT_LT(wi, 1e-9) << "tension integral must reduce to the natural cubic as sigma->0";
}

TEST(Tension, InterpolatesKnotsAtEverySigma) {
  const Eigen::VectorXd y = fwds();
  double worst = 0;
  for (double sig : {0.05, 0.3, 1.0, 3.0, 20.0}) {
    const auto t = makeT(y, sig);
    worst = std::max(worst, std::abs(t.forward(kT0) - kV0));  // pinned near join (C0)
    for (int i = 0; i < y.size(); ++i) worst = std::max(worst, std::abs(t.forward(kBack[i]) - y[i]));
  }
  EXPECT_LT(worst, 1e-12) << "tension spline must interpolate the knot forwards exactly at any sigma";
}

TEST(Tension, IntegralIsLinearInForwards) {
  // integral(t) is affine in the knot forwards (fixed boundary) -- exactly the is_linear_map promise the
  // W-cache relies on. The hyperbolic node functions are fixed weights; only ys/z are the (linear) values.
  const Eigen::VectorXd a = fwds(), d = fwds().reverse();
  const double alpha = 0.3;
  double worst = 0;
  for (double sig : {0.1, 1.0, 5.0}) {
    const auto ta = makeT(a, sig), td = makeT(d, sig), tm = makeT((alpha * a + (1 - alpha) * d).eval(), sig);
    for (double p = kT0 + 1e-3; p <= 30.0; p += 0.29)
      worst = std::max(worst, std::abs(tm.integral(p) - (alpha * ta.integral(p) + (1 - alpha) * td.integral(p))));
  }
  std::cout << "  [tension] max linearity residual = " << worst << "\n";
  EXPECT_LT(worst, 1e-13);
}

TEST(Tension, IntegralMatchesDenseQuadrature) {
  const Eigen::VectorXd y = fwds();
  double worst = 0;
  for (double sig : {0.1, 1.0, 5.0, 20.0}) {
    const auto t = makeT(y, sig);
    for (double p : {2.0, 5.0, 12.0, 30.0}) {
      const int N = 400000;
      const double dt = (p - kT0) / N;
      double s = 0.5 * (t.forward(kT0) + t.forward(p));
      for (int k = 1; k < N; ++k) s += t.forward(kT0 + k * dt);
      worst = std::max(worst, std::abs(t.integral(p) - (kI0 + s * dt)));
    }
  }
  std::cout << "  [tension] max |integral - dense quadrature| = " << worst << "\n";
  EXPECT_LT(worst, 1e-8) << "adaptive 7-pt Gauss is ~exact for the smooth hyperbolic f";
}

TEST(Tension, SigmaToInfinityIsPiecewiseLinear) {
  // High tension pulls the forward taut to piecewise-linear between knots -- the overshoot-free limit.
  const Eigen::VectorXd y = fwds();
  const auto t = makeT(y, 400.0);
  auto lin = [&](double p) -> double {
    if (p <= kBack[0]) return kV0 + (y[0] - kV0) * (p - kT0) / (kBack[0] - kT0);
    for (int i = 0; i + 1 < y.size(); ++i)
      if (p <= kBack[i + 1]) return y[i] + (y[i + 1] - y[i]) * (p - kBack[i]) / (kBack[i + 1] - kBack[i]);
    return y[y.size() - 1];
  };
  double worst = 0;
  for (double p = kT0 + 1e-3; p <= 30.0; p += 0.137) worst = std::max(worst, std::abs(t.forward(p) - lin(p)));
  std::cout << "  [tension] sigma=400 max |forward - piecewise-linear| = " << worst << "\n";
  EXPECT_LT(worst, 1e-3) << "sigma->inf forward must approach the taut piecewise-linear interpolant";
}

TEST(Tension, ComposesIntoAValidCurve) {
  // flat_tension: flat meeting-date front + tension back. Must be a valid discount curve -- DF(0)=1,
  // positive & strictly decreasing (positive forwards), C0 across the join -- and stay on the fast path.
  const std::vector<double> meeting{0.25, 0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_modular_curve<double>(cv::flat_tension(meeting, back, 1.5));
  EXPECT_TRUE(c.is_linear_map()) << "fixed-tension curve must be a linear map (W-cache fast path)";
  Eigen::VectorXd x(8);
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
  const double h = 1e-6;
  EXPECT_NEAR(c.forward(0.5 - h), c.forward(0.5 + h), 1e-3) << "forward continuous across the front/back join";
  std::cout << "  [tension-curve] DF(1y)=" << c.discount(1.0) << " DF(10y)=" << c.discount(10.0) << "\n";
}

TEST(Tension, WCacheReproducesDiscounts) {
  // The tension curve on the compiled fast path: integral_weight_matrix builds W once (by differentiating
  // integral() with AAD -- which also proves the region is AAD-safe), then DF = exp(-W x) must equal the
  // direct curve.discount(). The whole point: a tension curve reprices through the W-cache, no rebuild.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  const auto modules = cv::flat_tension(meeting, back, 2.0);
  auto c = cv::make_modular_curve<double>(modules);
  Eigen::VectorXd x(7);
  x << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  c.set_forwards(x);
  const std::vector<double> times{0.3, 0.8, 1.5, 3.0, 6.0, 9.5, 10.0};
  const Eigen::MatrixXd W = px::integral_weight_matrix(modules, times);
  const Eigen::VectorXd DF = (-(W * x).array()).exp();
  double worst = 0;
  for (std::size_t i = 0; i < times.size(); ++i) worst = std::max(worst, std::abs(DF[i] - c.discount(times[i])));
  std::cout << "  [tension-wcache] max |exp(-Wx) - curve.discount| = " << worst << "\n";
  EXPECT_LT(worst, 1e-13) << "tension curve must reprice exactly through the W-cache";
}

// A calibration problem whose curve is a fixed-tension spline. Duck-types CalibrationProblem
// (residuals<Scalar> / n_knots / n_residuals), so the generic LM + AAD Jacobian drive it unchanged.
namespace {
struct TensionProblem {
  cal::CalibrationProblem inst;
  double sigma = 1.5;
  int n_knots() const { return inst.n_knots(); }
  int n_residuals() const { return inst.n_residuals(); }
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = cv::make_modular_curve<Scalar>(cv::flat_tension(inst.meeting_times, inst.back_times, sigma));
    c.set_forwards(x);
    return inst.price_residuals<Scalar>(c);
  }
};
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

TEST(Tension, CalibratesToMarketAndReprices) {
  TensionProblem prob;
  prob.inst.meeting_times = {0.5};
  prob.inst.back_times = {1, 2, 3, 4, 5, 7, 10};  // 7 back forwards + 1 front = 8 free vars
  const std::vector<double> mats{0.5, 1, 2, 3, 4, 5, 7, 10};
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  // Self-consistent market generated from a KNOWN tension curve, so x_true is the exact solution.
  Eigen::VectorXd xt(8);
  xt << 0.030, 0.033, 0.036, 0.039, 0.041, 0.043, 0.044, 0.046;
  auto ct = cv::make_modular_curve<double>(cv::flat_tension(prob.inst.meeting_times, prob.inst.back_times, prob.sigma));
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-13) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(8, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [tension-calib] reprice=" << reprice << " ||x*-xt||=" << (res.x - xt).cwiseAbs().maxCoeff()
            << " stat=" << res.stationarity << " iters=" << res.iterations << "\n";
  EXPECT_LT(reprice, 1e-9) << "calibrated tension curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-7) << "first-order optimality";
}
