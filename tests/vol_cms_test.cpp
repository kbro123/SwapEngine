// Phase-3b gate for the CMS convexity layer (vol/cms.hpp), linear-TSR normal model. Robust limits only —
// the ones that hold regardless of the exact TSR slope: no convexity at theta=0 or vol=0, positive and
// monotone convexity for theta>0, and the caplet reducing to a plain Bachelier / intrinsic in the limits.
#include <gtest/gtest.h>

#include <cmath>

#include "swaps/vol/cms.hpp"

namespace v = swaps::vol;

namespace {
constexpr double S0 = 0.034, VOL = 0.0090, T = 5.0;
}

TEST(Cms, NoConvexityAtZeroSlopeOrZeroVol) {
  EXPECT_NEAR(v::cms_forward<double>(S0, VOL, T, 0.0).rate, S0, 1e-15);      // theta = 0
  EXPECT_NEAR(v::cms_forward<double>(S0, VOL, T, 0.0).convexity, 0.0, 1e-15);
  EXPECT_NEAR(v::cms_forward<double>(S0, 0.0, T, 0.8).rate, S0, 1e-15);      // sigma = 0
}

TEST(Cms, ConvexityPositiveAndMonotone) {
  const double a = v::cms_forward<double>(S0, VOL, T, 0.5).convexity;
  const double b = v::cms_forward<double>(S0, VOL, T, 1.0).convexity;
  EXPECT_GT(a, 0.0);
  EXPECT_GT(b, a);                                    // monotone increasing in the slope
  EXPECT_NEAR(b, 1.0 * VOL * VOL * T, 1e-15);         // exact closed form theta*sigma^2*T
  // and CMS > forward
  EXPECT_GT(v::cms_forward<double>(S0, VOL, T, 1.0).rate, S0);
}

TEST(Cms, CapletReducesToBachelierAndIntrinsic) {
  const double K = 0.036, kvol = 0.0095;
  // (E5 2026-09-10: the theta=0 `cms_optionlet == bachelier_price` line was deleted -- cms.hpp makes that
  // exact call at theta=0; it could not fail.)
  // strike vol -> 0, theta -> 0: intrinsic max(S0-K,0) for a cap.
  EXPECT_NEAR(v::cms_optionlet<double>(S0, VOL, T, 0.0, K, 0.0, true), std::max(S0 - K, 0.0), 1e-14);
  EXPECT_NEAR(v::cms_optionlet<double>(S0, VOL, T, 0.0, 0.030, 0.0, true), std::max(S0 - 0.030, 0.0), 1e-14);
  // convexity lifts a cap's value (higher adjusted forward).
  EXPECT_GT(v::cms_optionlet<double>(S0, VOL, T, 0.8, K, kvol, true),
            v::cms_optionlet<double>(S0, VOL, T, 0.0, K, kvol, true));
}
