// Verifies our transcribed Hull-White convexity matches QuantLib's HullWhite::convexityBias
// exactly, so the engine/tests can compute convexity without linking QuantLib.
//
// This target links the QuantLib oracle (unlike swaps_tests). It is a *self-check on the
// reference market*, not part of the shipped engine's own correctness gate.

#include <gtest/gtest.h>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>

#include <cmath>
#include <vector>

#include "reference_market.hpp"

namespace rm = swaps::refmkt;

TEST(Convexity, MatchesQuantLibHullWhiteBias) {
  // A spread of (price, t1, t2) triples covering the strip.
  struct Case { double price, t1, t2; };
  const std::vector<Case> cases{
      {95.90, 0.07, 0.15}, {96.25, 0.40, 0.49}, {96.56, 0.90, 0.98},
      {95.95, 0.19, 0.44}, {96.44, 0.69, 0.94}, {96.62, 1.44, 1.69},
      {96.66, 2.19, 2.44}, {96.60, 2.94, 3.20},
  };
  double worst = 0.0;
  for (const auto& c : cases) {
    const double ours = rm::hull_white_convexity(c.price, c.t1, c.t2);
    const double ql = QuantLib::HullWhite::convexityBias(c.price, c.t1, c.t2,
                                                         rm::convexity_sigma,
                                                         rm::convexity_mean_reversion);
    EXPECT_NEAR(ours, ql, 1e-15) << "price=" << c.price << " t1=" << c.t1 << " t2=" << c.t2;
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [convexity] max |ours - QuantLib| = " << worst << "\n";
}

TEST(Convexity, IsPositiveAndIncreasingWithinAFamily) {
  // Within a fixed accrual length, convexity must strictly increase with start time.
  const double dt = 0.25;  // ~3M
  double prev = -1.0;
  for (double t1 = 0.25; t1 <= 3.0; t1 += 0.25) {
    const double c = rm::hull_white_convexity(96.5, t1, t1 + dt);
    EXPECT_GT(c, 0.0);
    EXPECT_GT(c, prev) << "convexity must increase with start time, t1=" << t1;
    prev = c;
  }
}

TEST(Convexity, MeanReversionCapsGrowthVersusHoLee) {
  // Ho-Lee (a->0 limit) grows like t^2; Hull-White with a>0 must fall below it at long horizons.
  auto ho_lee = [](double t1, double t2) { return 0.5 * rm::convexity_sigma * rm::convexity_sigma * t1 * t2; };
  const double t1 = 3.0, t2 = 3.25;
  const double hw = rm::hull_white_convexity(96.5, t1, t2);
  const double hl = ho_lee(t1, t2);
  EXPECT_LT(hw, hl) << "mean reversion should damp convexity vs Ho-Lee at 3y (hw=" << hw
                    << " hl=" << hl << ")";
  std::cout << "  [convexity] at 3y: Hull-White=" << hw << "  Ho-Lee=" << hl
            << "  damping=" << (1.0 - hw / hl) * 100.0 << "%\n";
}
