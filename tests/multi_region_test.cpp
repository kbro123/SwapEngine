// The generalized MultiRegionCurve<Flat, NaturalCubic> must reproduce the hand-written
// CalibrationCurve to machine precision -- same interpolation, same knots, same forwards. This
// is the de-risking step before anything migrates onto the multi-region engine.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/curve/calibration_curve.hpp"

using swaps::ad::Dual;
using swaps::curve::Flat;
using swaps::curve::Hermite;
using swaps::curve::Linear;
using swaps::curve::MultiRegionCurve;
using swaps::curve::NaturalCubic;
using swaps::curve::CalibrationCurve;

namespace {
const std::vector<double> kMeeting{0.08, 0.25, 0.45, 0.70};
const std::vector<double> kBack{1.5, 3.0, 5.0, 10.0, 20.0, 30.0};
const std::vector<double> kX{0.043, 0.041, 0.039, 0.036,          // front (flat)
                             0.035, 0.037, 0.040, 0.043, 0.041, 0.038};  // back (spline)
}  // namespace

TEST(MultiRegion, ReproducesCalibrationCurve) {
  auto ref = swaps::curve::make_calibration_curve<double>(kMeeting, kBack);
  ref.set_forwards(kX);

  MultiRegionCurve<double, Flat, Hermite> mr{Flat<double>(kMeeting), Hermite<double>(kBack)};
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

TEST(MultiRegion, HermiteInterpolatesKnotsIsC1AndLocal) {
  const std::vector<double> meeting{0.1, 0.3};
  const std::vector<double> back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  const std::vector<double> xv{0.040, 0.038,                                                   // front
                               0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.039, 0.037};  // back
  static_assert(MultiRegionCurve<double, Flat, Hermite>::is_linear_map);

  // Interpolates its back knots exactly, and is C1 (slope continuous) across them.
  MultiRegionCurve<double, Flat, Hermite> hc{Flat<double>(meeting), Hermite<double>(back)};
  hc.set_forwards(xv);
  for (std::size_t i = 0; i < back.size(); ++i) EXPECT_NEAR(hc.forward(back[i]), xv[2 + i], 1e-12);
  const double e = 1e-6;
  for (std::size_t i = 1; i + 1 < back.size(); ++i) {  // interior back knots
    const double left = (hc.forward(back[i]) - hc.forward(back[i] - e)) / e;
    const double right = (hc.forward(back[i] + e) - hc.forward(back[i])) / e;
    EXPECT_NEAR(left, right, 1e-3) << "C1 broken at back knot " << i;
  }

  // Locality: d forward(t)/dx for a mid-back time. Hermite depends on only a few nearby knots
  // (banded); the natural cubic depends on ~all of them (global).
  Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<int>(xv.size()));
  const double tmid = 6.0;
  auto nonzeros = [&](auto& curve) {
    curve.set_forwards(swaps::ad::seed(x));
    const auto d = curve.forward(tmid).derivatives();
    int nz = 0;
    for (int k = 0; k < d.size(); ++k)
      if (std::abs(d[k]) > 1e-9) ++nz;
    return nz;
  };
  MultiRegionCurve<Dual, Flat, Hermite> hd{Flat<Dual>(meeting), Hermite<Dual>(back)};
  MultiRegionCurve<Dual, Flat, NaturalCubic> cd{Flat<Dual>(meeting), NaturalCubic<Dual>(back)};
  const int herm = nonzeros(hd), cub = nonzeros(cd);
  std::cout << "  [locality] d forward(6y)/dx nonzeros: Hermite=" << herm << " NaturalCubic=" << cub
            << " (of " << xv.size() << " knots)\n";
  EXPECT_LE(herm, 6) << "Hermite forward should depend on only a few nearby knots";
  EXPECT_GT(cub, herm) << "natural cubic is global, so depends on more knots";
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

// Duplicate/unsorted knots must be rejected at construction (they cause a zero-length segment -> NaN).
TEST(RegionKnots, RejectsDuplicateAndUnsortedKnots) {
  EXPECT_THROW(Hermite<double>({1.0, 2.0, 2.0, 3.0}), std::invalid_argument);   // duplicate
  EXPECT_THROW(Hermite<double>({1.0, 3.0, 2.0}), std::invalid_argument);        // unsorted
  EXPECT_THROW(Flat<double>({0.5, 0.5}), std::invalid_argument);                // duplicate
  EXPECT_THROW(Flat<double>({}), std::invalid_argument);                        // empty
  EXPECT_NO_THROW(Hermite<double>({1.0, 2.0, 3.0}));                            // strictly increasing OK
  // Cross-region join: first back knot must exceed the last front knot.
  EXPECT_THROW(swaps::curve::make_calibration_curve<double>({0.5, 1.0}, {1.0, 2.0}), std::invalid_argument);
  EXPECT_NO_THROW(swaps::curve::make_calibration_curve<double>({0.5, 1.0}, {1.5, 2.0}));
}
