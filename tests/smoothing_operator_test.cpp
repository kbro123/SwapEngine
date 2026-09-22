// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Gates for the SMOOTHING OPERATOR (swaps::calibration::second_difference_operator -- the one regulariser since the
// tension-energy operator was retired on 2026-09-22) and for the region derivatives it once consumed:
//   RegionSmoothing.*   the divided-difference curvature row: per-region lambda, unit spacing == the classic stencil,
//                       no row on a Flat knot (policy steps are never smoothed) nor reaching into a trailing Flat
//                       region, and ZERO on a straight line in time over non-uniform pillars (the property the old
//                       spacing-blind stencil lacked -- it biased every fit on 1y..5y/7y/10y grids).
//   RegionDerivatives.* every region's exact forward_d1 / forward_d2 against central finite differences, and
//                       pieces() reporting the true analytic breakpoints (a B-spline's de Boor-averaged knots).
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace curve = swaps::curve;

// A curve's two regions carry different curvature lambda: the operator weights each row by the lambda of
// its centre knot's region. Front region lambda=0 (unpenalised), back region lambda=5.
TEST(RegionSmoothing, SecondDifferenceIsPerRegion) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{{1.0, 2.0}, curve::Scheme::Flat, 0.0, 0.0});             // reg_lambda 0
  spec.regions.push_back(curve::CurveModule{{3.0, 4.0, 5.0, 6.0}, curve::Scheme::Hermite, 0.0, 5.0}); // reg_lambda 5
  p.curves.push_back(spec);

  const Eigen::MatrixXd R = cal::second_difference_operator(p, /*default lambda*/ 1.0, {0});
  ASSERT_EQ(R.rows(), 4);  // 6 interp knots -> interior rows i=1..4
  ASSERT_EQ(R.cols(), 6);
  EXPECT_NEAR(R.row(0).cwiseAbs().sum(), 0.0, 1e-15);  // knot i=1 in the lambda=0 front region: zero row
  for (int r = 1; r <= 3; ++r) {                        // knots i=2,3,4 in the lambda=5 back region
    const int i = r + 1;
    EXPECT_NEAR(R(r, i - 1), 5.0, 1e-12);
    EXPECT_NEAR(R(r, i), -10.0, 1e-12);
    EXPECT_NEAR(R(r, i + 1), 5.0, 1e-12);
  }
}

// With no region overriding (reg_lambda<0) every SMOOTH knot's row carries the single global lambda. At unit spacing the
// divided second difference with its trapezoid weight IS the plain stencil (a, b, c, w) = (1, -2, 1, 1), so the smooth
// rows read exactly as they did before 2026-09-21. The rows centred on the Flat region's knots (i = 1, 2) are ZERO: a Flat
// knot is a step level, not a point on a smooth forward -- the same rule the tension operator's zero-energy Flat pieces
// give (ASSUMPTIONS.md D18) -- so meeting-date policy steps are never smoothed by the default operator either. Knot 3
// (the Hermite region's first) IS smoothed, against the join value at knot 2: that join row is what pins a single Flat
// front knot on a basis-only curve.
TEST(RegionSmoothing, UniformLambdaOnSmoothKnotsAndNoRowOnFlatOnes) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{{1.0, 2.0, 3.0}, curve::Scheme::Flat});     // reg_lambda default (-1)
  spec.regions.push_back(curve::CurveModule{{4.0, 5.0, 6.0}, curve::Scheme::Hermite});  // reg_lambda default (-1)
  p.curves.push_back(spec);

  const Eigen::MatrixXd R = cal::second_difference_operator(p, /*global lambda*/ 0.7, {0});
  ASSERT_EQ(R.rows(), 4);
  EXPECT_NEAR(R.row(0).cwiseAbs().sum(), 0.0, 1e-15) << "knot 1 is a Flat step level";
  EXPECT_NEAR(R.row(1).cwiseAbs().sum(), 0.0, 1e-15) << "knot 2 is a Flat step level";
  for (int r = 2; r < 4; ++r) {
    const int i = r + 1;
    EXPECT_NEAR(R(r, i - 1), 0.7, 1e-12);
    EXPECT_NEAR(R(r, i), -1.4, 1e-12);
    EXPECT_NEAR(R(r, i + 1), 0.7, 1e-12);
  }
}

// A trailing single-knot Flat region (the ConsistentRisk unreached-knot fixture): no row reaches it, so an instrument-free
// tail knot stays genuinely unreached under the default operator -- the same as under tension.
TEST(RegionSmoothing, ATrailingFlatKnotHasNoRowReachingIt) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{{1.0, 2.0, 3.0, 4.0, 5.0}, curve::Scheme::Hermite});
  spec.regions.push_back(curve::CurveModule{{15.0}, curve::Scheme::Flat});
  p.curves.push_back(spec);
  const Eigen::MatrixXd R = cal::second_difference_operator(p, 0.5, {0});
  ASSERT_EQ(R.rows(), 4);
  EXPECT_NEAR(R.col(5).cwiseAbs().sum(), 0.0, 1e-15) << "nothing couples the Flat tail knot";
  EXPECT_NEAR(R.row(3).cwiseAbs().sum(), 0.0, 1e-15) << "the last Hermite knot's row would reach into the Flat tail: dropped";
  for (int r = 0; r < 3; ++r) EXPECT_GT(R.row(r).cwiseAbs().sum(), 0.0);
}

// NON-UNIFORM spacing: the row is the divided second difference scaled by the trapezoid weight, so an AFFINE-in-time
// forward has exactly zero penalty on any pillar grid (1y..5y, 7y, 10y: the case the spacing-blind stencil got wrong,
// pulling a square bundle's exact linear solution off the line under Light).
TEST(RegionSmoothing, DividedSecondDifferenceIsZeroOnALineOverNonUniformPillars) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  const std::vector<double> knots{1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0};
  spec.regions.push_back(curve::CurveModule{knots, curve::Scheme::Hermite});
  p.curves.push_back(spec);
  const Eigen::MatrixXd R = cal::second_difference_operator(p, 0.5, {0});
  ASSERT_EQ(R.rows(), 10);
  Eigen::VectorXd line(12), zig(12);
  for (int i = 0; i < 12; ++i) { line[i] = 0.03 + 4e-4 * knots[static_cast<std::size_t>(i)]; zig[i] = line[i] + (i % 2 ? -1e-3 : 1e-3); }
  EXPECT_LT((R * line).squaredNorm(), 1e-24) << "a line in TIME costs nothing";
  // hand-computed at knot 7y (h0 = 2, h1 = 3): weight sqrt(2.5) * 0.5 * 2/(5) * [(z8-z7)/3 - (z7-z6)/2]
  const double f2 = 2.0 / 5.0 * ((zig[6] - zig[5]) / 3.0 - (zig[5] - zig[4]) / 2.0);
  EXPECT_NEAR((R * zig)[4], 0.5 * std::sqrt(2.5) * f2, 1e-15);
  EXPECT_GT((R * zig).squaredNorm(), 1e-8) << "a zigzag costs";
}


namespace {
struct SchemeCase { const char* name; swaps::curve::Scheme scheme; double sigma; double tol_energy; };
const std::vector<SchemeCase>& scheme_cases() {
  using swaps::curve::Scheme;
  static const std::vector<SchemeCase> c = {{"Linear", Scheme::Linear, 0.0, 1e-10}, {"NaturalCubic", Scheme::NaturalCubic, 0.0, 1e-10},
      {"Hermite", Scheme::Hermite, 0.0, 1e-10}, {"BSpline", Scheme::BSpline, 0.0, 1e-10},
      {"Tension sigma=0.1", Scheme::Tension, 0.1, 1e-9}, {"Tension sigma=1", Scheme::Tension, 1.0, 1e-9},
      {"Tension sigma=5", Scheme::Tension, 5.0, 1e-8}, {"Tension sigma=25", Scheme::Tension, 25.0, 1e-7}};
  return c;
}
std::vector<swaps::curve::CurveModule> mods_for(const SchemeCase& c) {
  std::vector<swaps::curve::CurveModule> mods{{{0.25, 0.5}, swaps::curve::Scheme::Flat}, {{1, 2, 3, 5, 7, 10, 15, 20, 30}, c.scheme}};
  mods[1].sigma = c.sigma;
  return mods;
}
Eigen::VectorXd wiggly(int n) { Eigen::VectorXd x(n); for (int i = 0; i < n; ++i) x[i] = 0.03 + 0.004 * std::sin(1.3 * i) + 0.0005 * i; return x; }
}  // namespace

TEST(RegionDerivatives, AnalyticForwardDerivativesMatchFiniteDifferencesOnEveryScheme) {
  for (const auto& c : scheme_cases()) {
    auto crv = swaps::curve::make_modular_curve<double>(mods_for(c));
    crv.set_forwards(wiggly(11));
    double worst1 = 0.0, worst2 = 0.0, scale1 = 0.0, scale2 = 0.0;
    const double eps = 1e-5;
    for (int k = 0; k < 400; ++k) {
      const double t = 0.55 + 29.4 * (k + 0.5) / 400.0;  // interior of the back region, away from the nodes
      bool near_node = false;
      for (double kn : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0}) if (std::abs(t - kn) < 3 * eps) near_node = true;
      if (near_node) continue;
      const double f0 = crv.forward(t), fp = crv.forward(t + eps), fm = crv.forward(t - eps), fpp = crv.forward(t + 2 * eps), fmm = crv.forward(t - 2 * eps);
      const double d1 = (-fpp + 8 * fp - 8 * fm + fmm) / (12 * eps), d2 = (-fpp + 16 * fp - 30 * f0 + 16 * fm - fmm) / (12 * eps * eps);
      worst1 = std::max(worst1, std::abs(crv.forward_d1(t) - d1)); scale1 = std::max(scale1, std::abs(d1));
      worst2 = std::max(worst2, std::abs(crv.forward_d2(t) - d2)); scale2 = std::max(scale2, std::abs(d2));
    }
    std::cout << "  [derivatives] " << c.name << ": |d1-fd|/scale " << worst1 / scale1 << "  |d2-fd|/scale " << worst2 / scale2 << "\n";
    EXPECT_LT(worst1, 1e-7 * scale1 + 1e-12) << c.name;
    // second differences carry ~1e-16*|f|/eps^2 ~ 3e-8 absolute roundoff noise (Linear has d2 == 0 exactly)
    EXPECT_LT(worst2, 5e-4 * scale2 + 1e-6) << c.name;
  }
}


// pieces() reports the TRUE analytic breakpoints: a B-spline's de Boor-averaged interior knots, not the
// input knots; cubics and Tension break at their nodes.
TEST(RegionDerivatives, PiecesAreTheTrueAnalyticBreakpoints) {
  using swaps::curve::Scheme;
  const std::vector<double> back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  auto bs = swaps::curve::make_modular_curve<double>({{back, Scheme::BSpline}});
  bs.set_forwards(Eigen::VectorXd::Zero(static_cast<int>(back.size())));  // pieces are fixed at build
  const auto pb = bs.pieces();
  EXPECT_EQ(static_cast<int>(pb.size()), static_cast<int>(back.size()) - 1);  // n-2 segments => n-1 breakpoints
  for (std::size_t k = 1; k + 1 < pb.size(); ++k) {  // interior = de Boor average of three consecutive knots
    const double expect = (back[k - 1 + 0] + back[k] + back[k + 1]) / 3.0;
    EXPECT_NEAR(pb[k], expect, 1e-12) << k;
  }
  auto hm = swaps::curve::make_modular_curve<double>({{back, Scheme::Hermite}});
  EXPECT_THROW(hm.pieces(), std::logic_error);  // unbuilt: refused, never a partial list
  hm.set_forwards(Eigen::VectorXd::Zero(static_cast<int>(back.size())));
  EXPECT_EQ(hm.pieces(), back);
  swaps::curve::CurveModule tm{back, Scheme::Tension}; tm.sigma = 2.0;
  auto tn = swaps::curve::make_modular_curve<double>({tm});
  tn.set_forwards(Eigen::VectorXd::Zero(static_cast<int>(back.size())));
  EXPECT_EQ(tn.pieces(), back);
}

