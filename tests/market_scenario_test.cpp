// market::Scenario — the declarative "fork over the market" primitive at the BUILD-INPUT level. Because
// Market and ModularCurve are move-only and a built curve does not expose its (modules, x) inputs, a Scenario
// shocks the INPUTS — the curve forward vectors (x) and the FxMatrix — and the caller rebuilds a shocked
// Market. This proves the two transforms in isolation. Value semantics; QuantLib-free.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "swaps/market/fx.hpp"
#include "swaps/market/scenario.hpp"

namespace mkt = swaps::market;

TEST(Scenario, ShiftsKeyedCurveAndLeavesOthersUnchanged) {
  Eigen::VectorXd x(3);
  x << 0.043, 0.041, 0.038;

  // +25bp on SOFR == +0.0025 on every forward.
  mkt::Scenario s;
  s.shift_curve("SOFR", 25.0);

  const Eigen::VectorXd sofr = s.shocked_forwards("SOFR", x);
  ASSERT_EQ(sofr.size(), 3);
  EXPECT_NEAR(sofr[0], 0.0455, 1e-9);
  EXPECT_NEAR(sofr[1], 0.0435, 1e-9);
  EXPECT_NEAR(sofr[2], 0.0405, 1e-9);

  // An unshocked curve is returned unchanged.
  const Eigen::VectorXd estr = s.shocked_forwards("ESTR", x);
  ASSERT_EQ(estr.size(), 3);
  EXPECT_NEAR(estr[0], 0.043, 1e-9);
  EXPECT_NEAR(estr[1], 0.041, 1e-9);
  EXPECT_NEAR(estr[2], 0.038, 1e-9);
}

TEST(Scenario, BumpsFxPairRelativelyOnACopy) {
  mkt::FxMatrix fx;
  fx.add("EUR", "USD", 1.09);

  // +1% on EURUSD.
  mkt::Scenario s;
  s.bump_fx("EUR", "USD", 0.01);

  const mkt::FxMatrix shocked = s.shocked_fx(fx);
  EXPECT_NEAR(shocked.rate("EUR", "USD"), 1.09 * 1.01, 1e-9);

  // The base matrix is untouched (shocked on a copy).
  EXPECT_NEAR(fx.rate("EUR", "USD"), 1.09, 1e-9);
}

TEST(Scenario, GlobalParallelShiftsAllUnkeyedCurves) {
  Eigen::VectorXd x(3);
  x << 0.043, 0.041, 0.038;

  // A global "shift ALL curves" default, with an explicit override for one curve.
  mkt::Scenario s = mkt::Scenario::parallel(10.0);  // +10bp everywhere
  s.shift_curve("SOFR", 25.0);                      // SOFR overridden to +25bp

  // Any unkeyed curve gets the global +10bp (+0.0010).
  const Eigen::VectorXd estr = s.shocked_forwards("ESTR", x);
  EXPECT_NEAR(estr[0], 0.044, 1e-9);
  EXPECT_NEAR(estr[1], 0.042, 1e-9);
  EXPECT_NEAR(estr[2], 0.039, 1e-9);

  // The explicitly keyed curve uses its own +25bp, not the global.
  const Eigen::VectorXd sofr = s.shocked_forwards("SOFR", x);
  EXPECT_NEAR(sofr[0], 0.0455, 1e-9);
}
