// Validation for the clamped cubic B-spline region (docs/bezier-and-moments.md, Part A).
// Engine-only (no QuantLib): checks the region contract and the B-spline's defining properties --
// C0 clamp to the boundary, partition-of-unity (constant control points -> flat forward), the
// convex-hull property, linearity in the control points, and integral() vs a dense quadrature.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"

using swaps::curve::BSpline;
using swaps::curve::Boundary;

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

TEST(BSpline, ForwardIsC1) {
  const auto b = make(cp());
  const double h = 1e-5;
  for (double bp : {3.0, 5.0}) {  // interior breakpoints of this knot set
    const double dl = (b.forward(bp) - b.forward(bp - h)) / h;
    const double dr = (b.forward(bp + h) - b.forward(bp)) / h;
    EXPECT_NEAR(dl, dr, 1e-3) << "forward slope must be continuous across breakpoint " << bp;
  }
}
