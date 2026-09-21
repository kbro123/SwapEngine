// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// The named smoothing strengths (calibration/regularize.hpp smoothing_preset, E7 stage 3): ONE table, read by the
// composer's compile_reg_spec and by every verb that needs a well-posed default calibration. Until 2026-09-13 the
// "light" value was written out in five verbs and the table again in compile_reg_spec. The values are the web
// composer's off / light / strong presets, pinned here so changing one is a decision, not a drift.
#include <vector>

#include <gtest/gtest.h>

#include "swaps/calibration/regularize.hpp"

namespace cal = swaps::calibration;

TEST(SmoothingPreset, TheComposerTableIsPinned) {
  // Continuous tension-energy operator (opt-in; the default until 2026-09-21).
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Off, true), 0.0);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Light, true), 0.02);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Strong, true), 0.2);
  // Discrete second-difference operator.
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Off, false), 0.0);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Light, false), 0.5);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Strong, false), 5.0);
}

TEST(SmoothingPreset, APresetCoversEveryCurveAndOffIsNoPenalty) {
  // THE DEFAULT OPERATOR is the discrete second difference (owner's decision 2026-09-21: the tension presets are ~1/h^3
  // weaker at knot spacing h and left a MonotoneCubic long end failing ticks where second-difference Light streams it).
  const cal::RegSpec dflt = cal::smoothing_preset(cal::Smoothing::Light, 2);
  EXPECT_FALSE(dflt.tension);
  EXPECT_EQ(dflt.lambda, 0.5);
  EXPECT_EQ(dflt.sigma, 0.0);

  const cal::RegSpec light = cal::smoothing_preset(cal::Smoothing::Light, 3, true, 0.7);
  EXPECT_TRUE(light.on());
  EXPECT_EQ(light.lambda, 0.02);
  EXPECT_EQ(light.curves, (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(light.tension);
  EXPECT_EQ(light.sigma, 0.7);

  const cal::RegSpec strong_sd = cal::smoothing_preset(cal::Smoothing::Strong, 2, false, 0.7);
  EXPECT_EQ(strong_sd.lambda, 5.0);
  EXPECT_EQ(strong_sd.curves, (std::vector<int>{0, 1}));
  EXPECT_FALSE(strong_sd.tension);
  EXPECT_EQ(strong_sd.sigma, 0.0) << "sigma parameterises the tension operator only";

  const cal::RegSpec off = cal::smoothing_preset(cal::Smoothing::Off, 4, true, 0.7);
  const cal::RegSpec none;
  EXPECT_FALSE(off.on());
  EXPECT_EQ(off.lambda, none.lambda);
  EXPECT_EQ(off.curves, none.curves);
  EXPECT_EQ(off.tension, none.tension);
  EXPECT_EQ(off.sigma, none.sigma);
}
