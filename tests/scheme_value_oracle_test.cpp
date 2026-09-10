// @oracle-test — validates against QuantLib's interpolations VALUE-for-value. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// The Linear, NaturalCubic and Hermite (Bessel/parabolic tangents) interpolants vs QuantLib's
// LinearInterpolation, CubicNaturalSpline and CubicInterpolation(Parabolic): the same knots and values,
// ~200 evaluation points each, the VALUE and the PRIMITIVE (our integral() minus the flat pre-segment a
// leading region carries before its first knot). E5.3 (2026-09-10): until now the interpolation MATH had no
// external oracle -- review-2026-09-08/kernel-tests.md finding 3: a Hermite with its Bessel weights swapped
// passed every test that used Hermite. The hand-computed 3-knot pins live in kernel_pins_test.cpp.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace cv = swaps::curve;

TEST(SchemeValueOracle, LinearNaturalCubicAndHermiteMatchQuantLibValueAndPrimitive) {
  const std::vector<double> t{1.0, 1.5, 2.25, 3.0, 5.0, 7.5, 10.0};
  const std::vector<double> y{0.030, 0.034, 0.031, 0.037, 0.040, 0.036, 0.041};  // wiggly: tangents matter
  Eigen::VectorXd x(static_cast<int>(y.size()));
  for (int i = 0; i < x.size(); ++i) x[i] = y[i];

  struct Case { cv::Scheme scheme; std::string name; Interpolation ql; };
  std::vector<Case> cases;
  cases.push_back({cv::Scheme::Linear, "Linear", LinearInterpolation(t.begin(), t.end(), y.begin())});
  cases.push_back({cv::Scheme::NaturalCubic, "NaturalCubic", CubicNaturalSpline(t.begin(), t.end(), y.begin())});
  cases.push_back({cv::Scheme::Hermite, "Hermite",
                   CubicInterpolation(t.begin(), t.end(), y.begin(), CubicInterpolation::Parabolic, false,
                                      CubicInterpolation::SecondDerivative, 0.0,
                                      CubicInterpolation::SecondDerivative, 0.0)});
  for (Case& cs : cases) {
    cs.ql.update();
    auto c = cv::make_modular_curve<double>({cv::CurveModule{t, cs.scheme}});
    c.set_forwards(x);
    double worst_v = 0.0, worst_p = 0.0;
    for (double u = t.front(); u <= t.back() + 1e-12; u += 0.045) {
      const double v_ours = c.forward(u), v_ql = cs.ql(u, true);
      const double p_ours = c.integral(u) - y.front() * t.front(), p_ql = cs.ql.primitive(u, true);
      worst_v = std::max(worst_v, std::abs(v_ours - v_ql));
      worst_p = std::max(worst_p, std::abs(p_ours - p_ql));
      EXPECT_NEAR(v_ours, v_ql, swaps::tol::literal) << cs.name << " value at t=" << u;
      EXPECT_NEAR(p_ours, p_ql, swaps::tol::literal) << cs.name << " primitive at t=" << u;
    }
    std::cout << "  [scheme-oracle] " << cs.name << " vs QuantLib: max |dvalue| = " << worst_v
              << ", max |dprimitive| = " << worst_p << "\n";
  }
}
