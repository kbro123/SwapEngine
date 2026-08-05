// Gate for no-arbitrage vol-cube interpolation (vol/vol_cube_interp.hpp): the calendar (total-variance)
// condition + the arb-preserving interpolator.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/vol/vol_cube_interp.hpp"

namespace v = swaps::vol;

TEST(VolCubeInterp, CalendarArbitrageDetection) {
  const std::vector<double> Ts{0.5, 1.0, 2.0, 5.0, 10.0};
  // Vols that fall with expiry but whose TOTAL VARIANCE still rises -> calendar arb-free.
  const std::vector<double> ok{0.0110, 0.0100, 0.0092, 0.0085, 0.0080};
  EXPECT_TRUE(v::calendar_arbitrage_free(Ts, ok));
  // A vol that collapses so hard the total variance drops between 5y and 10y -> calendar arbitrage.
  std::vector<double> bad = ok;
  bad[4] = 0.0030;  // w(10) = 9e-5 < w(5) = 3.6e-4
  EXPECT_FALSE(v::calendar_arbitrage_free(Ts, bad));
}

TEST(VolCubeInterp, TotalVarianceInterpolationHitsNodesAndIsArbFree) {
  const std::vector<double> Ts{1.0, 2.0, 5.0, 10.0};
  const std::vector<double> vols{0.0100, 0.0094, 0.0088, 0.0083};
  const v::NoArbVolCurve c(Ts, vols);
  EXPECT_TRUE(c.arbitrage_free());
  // Exact at the nodes.
  for (std::size_t i = 0; i < Ts.size(); ++i) EXPECT_NEAR(c.vol(Ts[i]), vols[i], 1e-14);
  // Interpolated total variance is monotone non-decreasing on a fine grid (calendar no-arb preserved).
  double prev_w = -1.0;
  for (int i = 0; i <= 200; ++i) {
    const double T = 0.5 + (12.0 - 0.5) * i / 200.0;
    const double w = v::total_variance(c.vol(T), T);
    EXPECT_GE(w, prev_w - 1e-12) << "T=" << T;
    prev_w = w;
  }
  // Between nodes, total variance is the linear blend (not the vol).
  const double wmid = 0.5 * (v::total_variance(vols[0], 1.0) + v::total_variance(vols[1], 2.0));
  EXPECT_NEAR(v::total_variance(c.vol(1.5), 1.5), wmid, 1e-14);
}

TEST(VolCubeInterp, FlatExtrapolationHoldsTotalVariance) {
  const std::vector<double> Ts{2.0, 5.0}, vols{0.0090, 0.0085};
  const v::NoArbVolCurve c(Ts, vols);
  // Past the last node the total variance is held flat (w never decreases -> no calendar arb).
  EXPECT_NEAR(v::total_variance(c.vol(8.0), 8.0), v::total_variance(vols[1], 5.0), 1e-14);
  EXPECT_NEAR(v::total_variance(c.vol(1.0), 1.0), v::total_variance(vols[0], 2.0), 1e-14);
}
