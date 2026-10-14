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
  // ONE operator (the curvature / second-difference penalty) and, since 2026-10-13 (K48), ONE switch: smoothing is a
  // tie-breaker among the curves the quotes allow, so "light" and "strong" -- strengths from when it traded against
  // the quotes (0.5 / 5.0) -- are both "on" (1) and the weight is regularize.hpp kSmoothingTieBreak, not this value.
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Off), 0.0);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Light), 1.0);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::Strong), 1.0);
  EXPECT_EQ(cal::smoothing_lambda(cal::Smoothing::On), 1.0);
  EXPECT_EQ(cal::kSmoothingTieBreak, 1e-4);
}

TEST(SmoothingPreset, APresetCoversEveryCurveAndOffIsNoPenalty) {
  const cal::RegSpec light = cal::smoothing_preset(cal::Smoothing::Light, 3);
  EXPECT_TRUE(light.on());
  EXPECT_EQ(light.lambda, 1.0);
  EXPECT_EQ(light.curves, (std::vector<int>{0, 1, 2}));

  const cal::RegSpec strong = cal::smoothing_preset(cal::Smoothing::Strong, 2);
  EXPECT_EQ(strong.lambda, 1.0);
  EXPECT_EQ(strong.curves, (std::vector<int>{0, 1}));

  const cal::RegSpec off = cal::smoothing_preset(cal::Smoothing::Off, 4);
  const cal::RegSpec none;
  EXPECT_FALSE(off.on());
  EXPECT_EQ(off.lambda, none.lambda);
  EXPECT_EQ(off.curves, none.curves);
}
