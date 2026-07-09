// Correctness gate for the two-region forward curve.
//
// The ORACLE files (interp_flat.csv, interp_cubic.csv) come from QuantLib's own interpolation
// objects — value(t) and primitive(t), both exact. They pin down BOTH our interpolators and
// their analytic integrals, with no curve, instrument, or finite-difference machinery involved.
//
// We deliberately never compare against YieldTermStructure::forwardRate(t,t,...): that is a
// finite difference on discount factors and is only ~1e-6 accurate, and it straddles the
// forward jumps at meeting dates.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "golden_loader.hpp"
#include "reference_market.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"
#include "tolerances.hpp"

using swaps::curve::TwoRegionForwardCurve;
namespace rm = swaps::refmkt;
namespace gd = swaps::golden;

namespace {

// Single source of truth — the same forwards gen_golden used. No local copies to drift.
const std::vector<double> kFront(rm::reference_front_forwards.begin(),
                                 rm::reference_front_forwards.end());
const std::vector<double> kBack(rm::reference_back_forwards.begin(),
                                rm::reference_back_forwards.end());

std::string golden(const std::string& f) { return std::string(SWAPS_GOLDEN_DIR) + "/" + f; }

// Knot times are recovered from the committed regression lock so the test never has to
// re-derive QuantLib's calendar arithmetic.
struct Knots {
  std::vector<double> front, back;
};

Knots load_knots() {
  Knots k;
  for (const auto& r : gd::read_csv(golden("composite_curve.csv"))) {
    if (r[0] != "knot") continue;
    (r[1] == "front" ? k.front : k.back).push_back(gd::as_double(r[2]));
  }
  return k;
}

TwoRegionForwardCurve<double> build() {
  const Knots k = load_knots();
  TwoRegionForwardCurve<double> c(k.front, k.back);
  std::vector<double> x = kFront;
  x.insert(x.end(), kBack.begin(), kBack.end());
  c.set_forwards(x);
  return c;
}

::testing::AssertionResult close(double got, double want, double rel) {
  const double denom = std::max(1.0, std::abs(want));
  const double err = std::abs(got - want) / denom;
  if (err <= rel) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "got=" << got << " want=" << want << " rel_err=" << err << " > " << rel;
}

}  // namespace

// ---- Structure ------------------------------------------------------------------------------

TEST(Curve, KnotCountsMatchTheReferenceMarket) {
  const auto c = build();
  EXPECT_EQ(c.n_front(), rm::n_front_knots);
  EXPECT_EQ(c.n_back(), rm::n_back_knots);
  EXPECT_EQ(c.n_knots(), rm::n_knots);
}

TEST(Curve, RejectsIllFormedKnots) {
  EXPECT_THROW(TwoRegionForwardCurve<double>({}, {1.0}), std::invalid_argument);
  EXPECT_THROW(TwoRegionForwardCurve<double>({1.0}, {}), std::invalid_argument);
  EXPECT_THROW(TwoRegionForwardCurve<double>({0.0}, {1.0}), std::invalid_argument);   // t<=0
  EXPECT_THROW(TwoRegionForwardCurve<double>({1.0}, {0.5}), std::invalid_argument);   // back before join
  EXPECT_THROW(TwoRegionForwardCurve<double>({2.0, 1.0}, {3.0}), std::invalid_argument);  // unsorted
}

// ---- (a) Front end vs the QuantLib BackwardFlat oracle ----------------------------------------

TEST(Curve, FrontMatchesQuantLibBackwardFlat) {
  const auto c = build();
  double worst_v = 0, worst_p = 0;
  int n = 0;
  for (const auto& r : gd::read_csv(golden("interp_flat.csv"))) {
    const double t = gd::as_double(r[0]), v = gd::as_double(r[1]), p = gd::as_double(r[2]);
    EXPECT_TRUE(close(c.forward(t), v, swaps::tol::curve_rel)) << " at t=" << t;
    EXPECT_TRUE(close(c.integral(t), p, swaps::tol::curve_rel)) << " integral at t=" << t;
    worst_v = std::max(worst_v, std::abs(c.forward(t) - v));
    worst_p = std::max(worst_p, std::abs(c.integral(t) - p));
    ++n;
  }
  ASSERT_GT(n, 100) << "oracle file nearly empty — the comparison would be vacuous";
  std::cout << "  [front] n=" << n << " max |df|=" << worst_v << "  max |dintegral|=" << worst_p
            << "\n";
}

// ---- (b) Back end vs the QuantLib natural-cubic oracle -----------------------------------------

TEST(Curve, BackMatchesQuantLibNaturalCubicSpline) {
  const auto c = build();
  const double Tj = c.join_time();
  const double I0 = c.integral(Tj);
  double worst_v = 0, worst_p = 0;
  int n = 0;
  for (const auto& r : gd::read_csv(golden("interp_cubic.csv"))) {
    const double t = gd::as_double(r[0]), v = gd::as_double(r[1]), p = gd::as_double(r[2]);
    EXPECT_TRUE(close(c.forward(t), v, swaps::tol::curve_rel)) << " at t=" << t;
    EXPECT_TRUE(close(c.integral(t) - I0, p, swaps::tol::curve_rel)) << " integral at t=" << t;
    worst_v = std::max(worst_v, std::abs(c.forward(t) - v));
    worst_p = std::max(worst_p, std::abs(c.integral(t) - I0 - p));
    ++n;
  }
  ASSERT_GT(n, 100) << "oracle file nearly empty — the comparison would be vacuous";
  std::cout << "  [back]  n=" << n << " max |df|=" << worst_v << "  max |dintegral|=" << worst_p
            << "\n";
}

// NEGATIVE CONTROL. The agreement above is ~1e-16 against a 1e-10 tolerance, which is good
// enough to be worth distrusting. Perturb one knot forward by 1 basis point and prove the
// SAME comparison fails. Without this, a bug that made the loop body never execute (empty
// file, wrong column) would leave every oracle test passing for the wrong reason.
TEST(Curve, NegativeControl_PerturbedCurveFailsTheOracle) {
  const Knots k = load_knots();
  TwoRegionForwardCurve<double> bad(k.front, k.back);
  std::vector<double> x = kFront;
  x.insert(x.end(), kBack.begin(), kBack.end());
  x[rm::n_front_knots] += 1e-4;  // +1bp on the first back knot
  bad.set_forwards(x);

  const double I0 = bad.integral(bad.join_time());
  bool any_mismatch = false;
  for (const auto& r : gd::read_csv(golden("interp_cubic.csv"))) {
    const double t = gd::as_double(r[0]), v = gd::as_double(r[1]);
    if (!close(bad.forward(t), v, swaps::tol::curve_rel)) any_mismatch = true;
  }
  EXPECT_TRUE(any_mismatch)
      << "a 1bp curve error slipped through the oracle comparison — the test has no teeth";

  // and the front, which the perturbation does NOT touch, must still match exactly
  for (const auto& r : gd::read_csv(golden("interp_flat.csv"))) {
    const double t = gd::as_double(r[0]), v = gd::as_double(r[1]);
    EXPECT_TRUE(close(bad.forward(t), v, swaps::tol::curve_rel))
        << "front must be unaffected by a back-knot perturbation, t=" << t;
  }
  (void)I0;
}

// ---- (c) Model properties that no oracle covers ------------------------------------------------

TEST(Curve, ForwardIsFlatBetweenMeetingsAndJumpsOnlyAtMeetings) {
  const auto c = build();
  const Knots k = load_knots();
  double prev = 0.0;
  for (std::size_t i = 0; i < k.front.size(); ++i) {
    const double lo = prev, hi = k.front[i];
    // sample strictly inside the segment; every sample must equal the segment's flat forward
    for (int j = 1; j <= 9; ++j) {
      const double t = lo + (hi - lo) * j / 10.0;
      EXPECT_TRUE(close(c.forward(t), kFront[i], 1e-15)) << " segment " << i << " t=" << t;
    }
    prev = hi;
  }
}

TEST(Curve, ForwardIsLevelContinuousAtTheJoin) {
  const auto c = build();
  const double Tj = c.join_time();
  const double eps = 1e-9;
  // last front forward, approached from both sides
  EXPECT_TRUE(close(c.forward(Tj - eps), kFront.back(), 1e-12));
  EXPECT_TRUE(close(c.forward(Tj + eps), kFront.back(), 1e-7))
      << "spline must start at the last front forward (the join constraint)";
  EXPECT_TRUE(close(c.forward(Tj), kFront.back(), 1e-15));
}

TEST(Curve, SplineIsC1AndC2AcrossInteriorBackKnots) {
  const auto c = build();
  const Knots k = load_knots();
  const double hstep = 1e-5;
  auto d1 = [&](double t) { return (c.forward(t + hstep) - c.forward(t - hstep)) / (2 * hstep); };
  auto d2 = [&](double t) {
    return (c.forward(t + hstep) - 2 * c.forward(t) + c.forward(t - hstep)) / (hstep * hstep);
  };
  // interior knots only (not the join, not the last knot)
  for (std::size_t i = 0; i + 1 < k.back.size(); ++i) {
    const double t = k.back[i];
    EXPECT_NEAR(d1(t - hstep), d1(t + hstep), 1e-4) << "C1 broken at back knot " << i;
    EXPECT_NEAR(d2(t - 10 * hstep), d2(t + 10 * hstep), 1e-2) << "C2 broken at back knot " << i;
  }
}

TEST(Curve, DiscountFactorsAreConsistentAndMonotone) {
  const auto c = build();
  EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0);
  double prev = 1.0;
  for (int i = 1; i <= 300; ++i) {
    const double t = c.max_time() * i / 300.0;
    const double df = c.discount(t);
    EXPECT_LT(df, prev) << "DF must strictly decrease for positive forwards, t=" << t;
    EXPECT_TRUE(close(df, std::exp(-c.integral(t)), 1e-15));
    prev = df;
  }
}

// ---- (d) Regression lock on our own composite output -------------------------------------------

TEST(Curve, ReproducesCommittedCompositeGolden) {
  const auto c = build();
  int n = 0;
  for (const auto& r : gd::read_csv(golden("composite_curve.csv"))) {
    if (r[0] != "grid") continue;
    const double t = gd::as_double(r[1]);
    EXPECT_TRUE(close(c.forward(t), gd::as_double(r[2]), swaps::tol::curve_rel)) << " fwd t=" << t;
    EXPECT_TRUE(close(c.integral(t), gd::as_double(r[3]), swaps::tol::curve_rel)) << " int t=" << t;
    EXPECT_TRUE(close(c.discount(t), gd::as_double(r[4]), swaps::tol::curve_rel)) << " df t=" << t;
    ++n;
  }
  EXPECT_GT(n, 500) << "golden grid unexpectedly small";
}
