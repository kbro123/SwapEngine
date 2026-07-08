// Phase 0 smoke test: proves the engine headers, Eigen, and GTest wiring all build & run.
// Real correctness tests (curve vs QuantLib oracle) arrive in Phase 1+.
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "swaps/version.hpp"
#include "tolerances.hpp"

TEST(Smoke, EngineHeaderLoads) {
  EXPECT_EQ(swaps::version_major, 0);
  EXPECT_STREQ(swaps::version_string(), "0.0.0-phase0");
}

TEST(Smoke, EigenLinearAlgebraWorks) {
  // A tiny linear solve — the machinery the LM calibration will lean on.
  Eigen::Matrix2d A;
  A << 2.0, 0.0,
       0.0, 4.0;
  Eigen::Vector2d b(2.0, 8.0);
  Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);
  EXPECT_NEAR(x(0), 1.0, swaps::tol::curve_rel);
  EXPECT_NEAR(x(1), 2.0, swaps::tol::curve_rel);
}
