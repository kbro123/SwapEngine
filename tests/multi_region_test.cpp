// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Multi-region composition: a curve stitched from an arbitrary sequence of region policies must
// reproduce the shipped flat_hermite layout to machine precision -- same interpolation, same knots,
// same forwards -- and report linearity correctly. Every curve in the engine is built this way, so
// this is the region-stitching contract itself, not a migration check.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"

using swaps::ad::Dual;
using swaps::curve::Flat;
using swaps::curve::Hermite;
using swaps::curve::Linear;
using swaps::curve::make_modular_curve;
using swaps::curve::Scheme;
using swaps::curve::NaturalCubic;

namespace {
const std::vector<double> kMeeting{0.08, 0.25, 0.45, 0.70};
const std::vector<double> kBack{1.5, 3.0, 5.0, 10.0, 20.0, 30.0};
const std::vector<double> kX{0.043, 0.041, 0.039, 0.036,          // front (flat)
                             0.035, 0.037, 0.040, 0.043, 0.041, 0.038};  // back (spline)
}  // namespace

// The shipped `flat_hermite` helper is a NAME for a module list, nothing more. Pin that: it must be
// structurally and numerically identical to writing the two modules out by hand, so a caller can always
// substitute an explicit region list for the shipped layout (which is exactly what the web composer
// does when a user defines their own regions).
TEST(MultiRegion, ShippedLayoutIsJustAModuleList) {
  const auto shipped = swaps::curve::flat_hermite(kMeeting, kBack);
  ASSERT_EQ(shipped.size(), 2u);
  EXPECT_EQ(shipped[0].knots, kMeeting);
  EXPECT_EQ(shipped[0].scheme, Scheme::Flat);
  EXPECT_EQ(shipped[1].knots, kBack);
  EXPECT_EQ(shipped[1].scheme, Scheme::Hermite);

  auto ref = swaps::curve::make_modular_curve<double>(shipped);
  ref.set_forwards(kX);

  auto mr = make_modular_curve<double>({{kMeeting, Scheme::Flat}, {kBack, Scheme::Hermite}});
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

  // Interpolates its back knots exactly, and is C1 (slope continuous) across them.
  auto hc = make_modular_curve<double>({{meeting, Scheme::Flat}, {back, Scheme::Hermite}});
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
  auto hd = make_modular_curve<Dual>({{meeting, Scheme::Flat}, {back, Scheme::Hermite}});
  auto cd = make_modular_curve<Dual>({{meeting, Scheme::Flat}, {back, Scheme::NaturalCubic}});
  const int herm = nonzeros(hd), cub = nonzeros(cd);
  std::cout << "  [locality] d forward(6y)/dx nonzeros: Hermite=" << herm << " NaturalCubic=" << cub
            << " (of " << xv.size() << " knots)\n";
  EXPECT_LE(herm, 6) << "Hermite forward should depend on only a few nearby knots";
  EXPECT_GT(cub, herm) << "natural cubic is global, so depends on more knots";
}

TEST(MultiRegion, LinearMapTraitAndComposition) {
  // is_linear_map() is the AND over regions; an all-linear composition enables the W-cache fast path.
  const std::vector<double> m{0.1, 0.3}, lin{0.6, 1.0}, cub{2.0, 5.0, 10.0};
  auto c = make_modular_curve<double>({{m, Scheme::Flat}, {lin, Scheme::Linear}, {cub, Scheme::NaturalCubic}});
  EXPECT_TRUE(c.is_linear_map()) << "Flat+Linear+NaturalCubic are all linear maps";
  EXPECT_TRUE(make_modular_curve<double>(swaps::curve::flat_hermite(m, cub)).is_linear_map());
  EXPECT_FALSE(make_modular_curve<double>(swaps::curve::flat_monotone(m, cub)).is_linear_map())
      << "one non-linear region makes the whole composition non-linear";

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
  EXPECT_THROW(swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite({0.5, 1.0}, {1.0, 2.0})), std::invalid_argument);
  EXPECT_NO_THROW(swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite({0.5, 1.0}, {1.5, 2.0})));
}
