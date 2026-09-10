// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Validation for the spline-under-tension region (research note §1,§3,§6). Engine-only (no QuantLib):
// checks the region contract and the tension spline's defining properties -- C0 clamp to the boundary,
// the σ→0 limit reproducing NaturalCubic, integral() vs a high-order quadrature, C¹/C² continuity at
// interior knots, linearity in the knot values, and the linear-map W-cache fast path.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/compiled.hpp"

using swaps::curve::Boundary;
using swaps::curve::NaturalCubic;
using swaps::curve::Tension;
namespace cv = swaps::curve;
namespace px = swaps::pricing;

namespace {
const std::vector<double> kKnots{2, 3, 5, 7, 10};  // n = 5 back knots -> 6 spline points, 5 segments
constexpr double kT0 = 1.0, kV0 = 0.032, kI0 = 0.05;

Eigen::VectorXd vals() {
  Eigen::VectorXd y(5);
  y << 0.030, 0.037, 0.041, 0.036, 0.044;
  return y;
}
Tension<double> makeT(double sigma, const Eigen::VectorXd& y, double v0 = kV0) {
  Tension<double> t(kKnots, sigma);
  Boundary<double> in{kT0, v0, 0.0, kI0, /*has_predecessor=*/true};
  t.build(y, 0, static_cast<int>(y.size()), in);
  return t;
}
NaturalCubic<double> makeNC(const Eigen::VectorXd& y, double v0 = kV0) {
  NaturalCubic<double> c(kKnots);
  Boundary<double> in{kT0, v0, 0.0, kI0, /*has_predecessor=*/true};
  c.build(y, 0, static_cast<int>(y.size()), in);
  return c;
}

// High-order composite Gauss-Legendre (5-pt/panel) reference for ∫ forward from t0 to t. For a smooth
// analytic f this is ~machine precision, so it can gate integral() at 1e-12 (a trapezoid cannot).
double quad(const Tension<double>& t, double a, double b) {
  static const double gx[5] = {0.0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640,
                               -0.9061798459386640};
  static const double gw[5] = {0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                               0.2369268850561891, 0.2369268850561891};
  const int panels = 4000;
  const double dx = (b - a) / panels;
  double acc = 0.0;
  for (int p = 0; p < panels; ++p) {
    const double lo = a + p * dx, c = lo + 0.5 * dx, hh = 0.5 * dx;
    for (int i = 0; i < 5; ++i) acc += gw[i] * t.forward(c + gx[i] * hh) * hh;
  }
  return acc;
}
}  // namespace

TEST(Tension, ClampedStartPinsBoundary) {
  const auto t = makeT(2.0, vals());
  EXPECT_NEAR(t.forward(kT0), kV0, 1e-12) << "tension spline must interpolate the pinned boundary value";
  EXPECT_NEAR(t.integral(kT0), kI0, 1e-15) << "integral at the join must equal the incoming integral";
  // Interpolates its knot values (it is an interpolating, not approximating, spline).
  const Eigen::VectorXd y = vals();
  for (int i = 0; i < y.size(); ++i)
    EXPECT_NEAR(t.forward(kKnots[i]), y[i], 1e-11) << "tension spline must pass through knot " << i;
}

TEST(Tension, ConstantValuesAreFlat) {
  // All values (incl. the pinned leading one) equal c => zero divided differences => z=0 => f == c.
  const double c = 0.039;
  const auto t = makeT(3.0, Eigen::VectorXd::Constant(5, c), /*v0=*/c);
  for (double tt : {1.0, 1.7, 3.3, 5.0, 8.1, 10.0, 12.0}) {
    EXPECT_NEAR(t.forward(tt), c, 1e-12) << " at t=" << tt;
    EXPECT_NEAR(t.integral(tt), kI0 + c * (tt - kT0), 1e-12) << " at t=" << tt;
  }
}

TEST(Tension, SigmaToZeroReproducesNaturalCubic) {
  // σ→0: the tension basis {1,t,sinh σt,cosh σt} degenerates to {1,t,t²,t³}. With a tiny σ the region
  // must agree with NaturalCubic to ~1e-12 (both forward AND integral), the continuous-limit gate.
  const Eigen::VectorXd y = vals();
  const double sigma = 1e-8;  // σ²h² ~ 1e-14 over these spacings -> limit error well under 1e-12
  const auto t = makeT(sigma, y);
  const auto c = makeNC(y);
  double wf = 0, wi = 0;
  for (double tt = 1.0; tt <= 11.0; tt += 0.05) {
    wf = std::max(wf, std::abs(t.forward(tt) - c.forward(tt)));
    wi = std::max(wi, std::abs(t.integral(tt) - c.integral(tt)));
  }
  std::cout << "  [tension] σ→0 max |forward - NaturalCubic| = " << wf << ", |integral - NaturalCubic| = "
            << wi << "\n";
  EXPECT_LT(wf, 1e-12) << "σ→0 forward must reproduce the natural cubic";
  EXPECT_LT(wi, 1e-12) << "σ→0 integral must reproduce the natural cubic";
}

TEST(Tension, IntegralMatchesQuadrature) {
  // integral(t) (closed form via ∫sinh = cosh/σ) must match a high-order quadrature of forward(t) to
  // ~1e-12 at a MODERATE σ (the exact hyperbolic path, no series fallback).
  for (double sigma : {0.5, 2.0, 8.0}) {
    const auto t = makeT(sigma, vals());
    double worst = 0;
    for (double tt : {1.5, 2.0, 4.0, 6.5, 9.0, 10.0})
      worst = std::max(worst, std::abs(t.integral(tt) - (kI0 + quad(t, kT0, tt))));
    std::cout << "  [tension] σ=" << sigma << " max |integral - quadrature| = " << worst << "\n";
    EXPECT_LT(worst, 1e-12) << "closed-form integral must match the quadrature of forward (σ=" << sigma << ")";
  }
}

TEST(Tension, ForwardIsC2) {
  // A tension spline is C² across interior knots. Finite-difference the forward's first and second
  // derivative from each side and require agreement.
  const auto t = makeT(4.0, vals());
  const double h = 1e-3;
  for (double bp : {3.0, 5.0, 7.0}) {  // interior knots of kKnots
    const double d1l = (t.forward(bp) - t.forward(bp - h)) / h;
    const double d1r = (t.forward(bp + h) - t.forward(bp)) / h;
    EXPECT_NEAR(d1l, d1r, 1e-4) << "forward slope (C¹) must be continuous across knot " << bp;
    const double d2l = (t.forward(bp) - 2 * t.forward(bp - h) + t.forward(bp - 2 * h)) / (h * h);
    const double d2r = (t.forward(bp + 2 * h) - 2 * t.forward(bp + h) + t.forward(bp)) / (h * h);
    EXPECT_NEAR(d2l, d2r, 5e-3) << "forward curvature (C²) must be continuous across knot " << bp;
  }
}

TEST(Tension, IntegralIsLinearInValues) {
  // integral(t) is affine in the free knot values (fixed boundary & σ), so it preserves affine
  // combinations -- exactly the is_linear_map promise the W-cache relies on.
  const Eigen::VectorXd a = vals(), d = vals().reverse();
  const double alpha = 0.3;
  const auto ta = makeT(2.0, a), td = makeT(2.0, d), tm = makeT(2.0, (alpha * a + (1 - alpha) * d).eval());
  double worst = 0;
  for (double tt : {1.5, 3.0, 5.0, 7.5, 10.0})
    worst = std::max(worst, std::abs(tm.integral(tt) - (alpha * ta.integral(tt) + (1 - alpha) * td.integral(tt))));
  std::cout << "  [tension] max linearity residual = " << worst << "\n";
  EXPECT_LT(worst, 1e-13);
}

TEST(Tension, ComposesIntoAValidCurve) {
  // flat_tension: flat meeting-date front + tension-spline back. Must be a valid discount curve --
  // DF(0)=1, positive & strictly decreasing (positive forwards) -- and stay on the linear-map fast path.
  const std::vector<double> meeting{0.25, 0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_modular_curve<double>(cv::flat_tension(meeting, back, 3.0));
  EXPECT_TRUE(c.is_linear_map()) << "fixed-σ tension curve must be a linear map";
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
  std::cout << "  [tension-curve] DF(1y)=" << c.discount(1.0) << " DF(10y)=" << c.discount(10.0) << "\n";
}

TEST(Tension, WCacheReproducesDiscounts) {
  // The tension curve on the compiled fast path: integral_weight_matrix builds W once (σ fixed => linear
  // map), then DF = exp(-W x) must equal the direct curve.discount() -- proving a tension curve reprices
  // through the W-cache with NO curve rebuild, the whole point of a fixed-tension region.
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 7, 10};
  auto c = cv::make_modular_curve<double>(cv::flat_tension(meeting, back, 5.0));
  Eigen::VectorXd x(7);
  x << 0.031, 0.034, 0.037, 0.040, 0.042, 0.044, 0.046;
  c.set_forwards(x);
  const std::vector<double> times{0.3, 0.8, 1.5, 3.0, 6.0, 9.5, 10.0};
  const Eigen::MatrixXd W = px::integral_weight_matrix(cv::flat_tension(meeting, back, 5.0), times);
  const Eigen::VectorXd DF = (-(W * x).array()).exp();
  double worst = 0;
  for (std::size_t i = 0; i < times.size(); ++i) worst = std::max(worst, std::abs(DF[i] - c.discount(times[i])));
  std::cout << "  [tension-wcache] max |exp(-Wx) - curve.discount| = " << worst << "\n";
  EXPECT_LT(worst, 1e-13) << "tension curve must reprice exactly through the W-cache";
}

TEST(Tension, LargeSigmaApproachesLinear) {
  // As σ→∞ the tension spline is pulled taut -> piecewise linear between knots, so its midpoint sits
  // near the average of the bracketing knot values (a plain cubic would overshoot). Sanity of the taut
  // limit and of the large-σh scaled-exponential asymptotics (no overflow/NaN).
  const Eigen::VectorXd y = vals();
  const auto t = makeT(200.0, y);  // σh up to 600 -> exercises the asymptotic branch
  for (int i = 0; i + 1 < y.size(); ++i) {
    const double mid = 0.5 * (kKnots[i] + kKnots[i + 1]);
    const double f = t.forward(mid);
    ASSERT_TRUE(std::isfinite(f)) << "large-σ evaluation must not overflow/NaN at t=" << mid;
    EXPECT_NEAR(f, 0.5 * (y[i] + y[i + 1]), 5e-3) << "taut spline midpoint ≈ linear interpolant, knot " << i;
  }
}
