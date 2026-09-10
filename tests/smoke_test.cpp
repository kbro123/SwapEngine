// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Phase 0 smoke test: proves the engine headers, Eigen, and GTest wiring all build & run.
// Real correctness tests (curve vs QuantLib oracle) arrive in Phase 1+.
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <iostream>

#include "swaps/version.hpp"
#include "swaps/simd.hpp"
#include "tolerances.hpp"

TEST(Smoke, EngineHeaderLoads) {
  EXPECT_EQ(swaps::version_major, 0);
  EXPECT_STREQ(swaps::version_string(), "0.0.0-phase0");
}

// The ISA detection must agree with Eigen, and must be a sane power of two.
// (simd.hpp also static_asserts this; here we surface it at runtime for the log.)
TEST(Smoke, SimdWidthDetectedConsistently) {
  constexpr int w = swaps::simd::packet_size<double>;
  EXPECT_EQ(w, swaps::simd::detected::double_packet_size);
  EXPECT_TRUE(w == 1 || w == 2 || w == 4 || w == 8) << "unexpected packet width " << w;

  std::cout << "  [isa] " << swaps::simd::detected::isa_name
            << "  flags='" << swaps::simd::detected::arch_flags << "'"
            << "  doubles/register=" << w << "\n";
}

// padded_count must round up to a whole number of lanes, and never shrink n.
TEST(Smoke, PaddedCountRoundsUpToLanes) {
  constexpr int w = swaps::simd::packet_size<double>;
  for (int n : {0, 1, 3, 7, 8, 100, 1023}) {
    const int p = swaps::simd::padded_count<double>(n);
    EXPECT_GE(p, n);
    EXPECT_EQ(p % w, 0) << "n=" << n << " padded=" << p << " not a multiple of " << w;
    EXPECT_LT(p - n, w) << "over-padded n=" << n;
  }
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
