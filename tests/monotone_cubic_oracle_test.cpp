// QuantLib oracle for the MonotoneCubic region: it is transcribed to match QuantLib's
// MonotonicCubicNaturalSpline (CubicInterpolation with da=Spline, monotonic=true, SecondDerivative=0 at
// both ends) EXACTLY. We build QuantLib's interpolation on the same nodes and require our region's
// forward(t) and integral(t) to agree to ~1e-11 -- on a monotone dataset AND a wiggly one where the Hyman
// filter actively clamps (so the match validates the filter, not just an unfiltered spline).
#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <vector>

#include "swaps/curve/regions.hpp"

using swaps::curve::Boundary;
using swaps::curve::MonotoneCubic;

namespace {
const std::vector<double> kBack{1, 2, 3, 5, 7, 10};
constexpr double kT0 = 0.5, kV0 = 0.030, kI0 = 0.015;

// Build our region and the QuantLib oracle on the SAME (node, value) set: [kT0, kBack...] / [kV0, vals...].
void compare(const std::vector<double>& vals, const char* label) {
  Eigen::VectorXd v(static_cast<int>(vals.size()));
  for (int i = 0; i < v.size(); ++i) v[i] = vals[i];
  MonotoneCubic<double> mc(kBack);
  mc.build(v, 0, static_cast<int>(v.size()), Boundary<double>{kT0, kV0, 0.0, kI0});

  std::vector<double> xs{kT0}, ys{kV0};
  for (double b : kBack) xs.push_back(b);
  for (double y : vals) ys.push_back(y);
  QuantLib::MonotonicCubicNaturalSpline ql(xs.begin(), xs.end(), ys.begin());

  double worst_f = 0, worst_I = 0;
  for (double t = kT0; t <= kBack.back() + 1e-12; t += 0.01) {
    worst_f = std::max(worst_f, std::abs(mc.forward(t) - ql(t)));
    // QuantLib primitive(t) = integral from the first node (kT0); our integral(t) adds the incoming kI0.
    worst_I = std::max(worst_I, std::abs((mc.integral(t) - kI0) - ql.primitive(t)));
  }
  std::cout << "  [monotone-oracle:" << label << "] max|forward-QL|=" << worst_f
            << " max|integral-QL.primitive|=" << worst_I << "\n";
  EXPECT_LT(worst_f, 1e-11) << label << ": forward must match QuantLib MonotonicCubicNaturalSpline";
  EXPECT_LT(worst_I, 1e-11) << label << ": integral must match QuantLib's primitive";
}
}  // namespace

TEST(MonotoneCubicOracle, MatchesQuantLibMonotoneData) {
  compare({0.033, 0.036, 0.038, 0.041, 0.043, 0.045}, "monotone");
}

TEST(MonotoneCubicOracle, MatchesQuantLibWigglyDataFilterFires) {
  // Up/down data: an unfiltered natural spline overshoots here; QuantLib's Hyman filter clamps the
  // tangents, and our transcription must clamp identically.
  compare({0.045, 0.030, 0.048, 0.032, 0.050, 0.031}, "wiggly");
}
