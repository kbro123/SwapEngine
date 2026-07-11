// The generalized MultiRegionCurve<Flat, NaturalCubic> must reproduce the hand-written
// TwoRegionForwardCurve to machine precision -- same interpolation, same knots, same forwards. This
// is the de-risking step before anything migrates onto the multi-region engine.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"

using swaps::curve::Flat;
using swaps::curve::Linear;
using swaps::curve::MultiRegionCurve;
using swaps::curve::NaturalCubic;
using swaps::curve::TwoRegionForwardCurve;

namespace {
const std::vector<double> kMeeting{0.08, 0.25, 0.45, 0.70};
const std::vector<double> kBack{1.5, 3.0, 5.0, 10.0, 20.0, 30.0};
const std::vector<double> kX{0.043, 0.041, 0.039, 0.036,          // front (flat)
                             0.035, 0.037, 0.040, 0.043, 0.041, 0.038};  // back (spline)
}  // namespace

TEST(MultiRegion, ReproducesTwoRegionForwardCurve) {
  TwoRegionForwardCurve<double> ref(kMeeting, kBack);
  ref.set_forwards(kX);

  MultiRegionCurve<double, Flat, NaturalCubic> mr{Flat<double>(kMeeting), NaturalCubic<double>(kBack)};
  ASSERT_EQ(mr.n_knots(), static_cast<int>(kX.size()));
  mr.set_forwards(kX);

  double wf = 0, wi = 0, wd = 0;
  // dense grid + exact boundary points (meeting dates, join, back knots, beyond max)
  std::vector<double> ts;
  for (double t = 0.0; t <= 31.0; t += 0.013) ts.push_back(t);
  for (double m : kMeeting) ts.push_back(m);
  for (double s : kBack) ts.push_back(s);
  for (double t : ts) {
    wf = std::max(wf, std::abs(mr.forward(t) - ref.forward(t)));
    wi = std::max(wi, std::abs(mr.integral(t) - ref.integral(t)));
    wd = std::max(wd, std::abs(mr.discount(t) - ref.discount(t)));
  }
  std::cout << "  [multi-region] max |dfwd|=" << wf << " |dint|=" << wi << " |ddf|=" << wd << "\n";
  EXPECT_LT(wf, 1e-15);
  EXPECT_LT(wi, 1e-15);
  EXPECT_LT(wd, 1e-15);
}

TEST(MultiRegion, LinearMapTraitAndComposition) {
  // The trait is the AND over regions; all-linear configs enable the fast path.
  static_assert(MultiRegionCurve<double, Flat, NaturalCubic>::is_linear_map);
  static_assert(MultiRegionCurve<double, Flat, Linear, NaturalCubic>::is_linear_map);

  // A three-region curve builds and is continuous in the discount factor across every join.
  const std::vector<double> m{0.1, 0.3}, lin{0.6, 1.0}, cub{2.0, 5.0, 10.0};
  MultiRegionCurve<double, Flat, Linear, NaturalCubic> c{Flat<double>(m), Linear<double>(lin),
                                                         NaturalCubic<double>(cub)};
  std::vector<double> x{0.04, 0.038, 0.037, 0.036, 0.035, 0.037, 0.039};  // 2 + 2 + 3
  ASSERT_EQ(c.n_knots(), static_cast<int>(x.size()));
  c.set_forwards(x);

  EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0);
  // discount factor is continuous across the region joins (integral is C0)
  for (double j : {0.3, 1.0}) {
    const double eps = 1e-9;
    EXPECT_NEAR(c.discount(j - eps), c.discount(j + eps), 1e-8);
  }
  // strictly decreasing for positive forwards
  double prev = 1.0;
  for (double t = 0.2; t <= 10.0; t += 0.2) {
    EXPECT_LT(c.discount(t), prev);
    prev = c.discount(t);
  }
}
