// The structure fingerprint is the switch behind the OO / hot-path collapse: it must be STABLE when only
// the market moves (so a Model warm-ticks over its existing W-cache) and CHANGE on any topology edit (so W
// is recompiled). QuantLib-free — pure hashing over hand-built problems.
#include <gtest/gtest.h>

#include "swaps/build/instruments.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
struct P { const char* t; double r; };

cal::BundleProblem base_problem() {
  const b::Date vd = b::Date::from_iso("2026-09-01");
  const b::SwapConv conv = b::Index("USD-SOFR").ois_convention().resolve();
  auto ct = [&](const char* t) { return b::curve_time(vd, b::resolve(t, vd)); };

  cal::BundleProblem p;
  px::CurveStructure c;
  c.base = -1;
  c.regions = {cv::CurveModule{{ct("1y")}, cv::Scheme::Flat},
               cv::CurveModule{{ct("2y"), ct("5y"), ct("10y"), ct("30y")}, cv::Scheme::Hermite}};
  p.curves = {c};
  for (const P& q : {P{"1y", 0.0433}, P{"2y", 0.0372}, P{"5y", 0.0368}, P{"10y", 0.0385}, P{"30y", 0.0405}})
    p.instruments.push_back(b::par_swap(vd, conv, b::resolve(q.t, vd), /*fc=*/0, /*disc=*/0, q.r));
  return p;
}
}  // namespace

TEST(StructureFingerprint, StableUnderMarketChange) {
  const cal::BundleProblem base = base_problem();
  const std::uint64_t f0 = cal::structure_fingerprint(base);

  // Re-quoting every instrument leaves the topology (and thus W) untouched -> identical fingerprint.
  cal::BundleProblem requoted = base;
  for (auto& ins : requoted.instruments) ins.market += 1e-3;
  EXPECT_EQ(cal::structure_fingerprint(requoted), f0);

  // A soft band is also pure RHS -> identical fingerprint.
  cal::BundleProblem banded = base;
  banded.instruments[0].band_lower = 0.010;
  banded.instruments[0].band_upper = 0.020;
  banded.instruments[0].band_decay = 0.1;
  EXPECT_EQ(cal::structure_fingerprint(banded), f0);
}

TEST(StructureFingerprint, ChangesWithStructure) {
  const cal::BundleProblem base = base_problem();
  const std::uint64_t f0 = cal::structure_fingerprint(base);
  const b::Date vd = b::Date::from_iso("2026-09-01");
  const b::SwapConv conv = b::Index("USD-SOFR").ois_convention().resolve();

  auto scheme = base;
  scheme.curves[0].regions[1].scheme = cv::Scheme::MonotoneCubic;
  EXPECT_NE(cal::structure_fingerprint(scheme), f0) << "changing a region scheme must recompile";

  auto knot = base;
  knot.curves[0].regions[1].knots[0] += 0.01;
  EXPECT_NE(cal::structure_fingerprint(knot), f0) << "moving a knot must recompile";

  auto role = base;
  role.instruments[2].fwd.forecast = 1;
  EXPECT_NE(cal::structure_fingerprint(role), f0) << "re-binding a forecast role must recompile";

  auto added = base;
  added.instruments.push_back(b::par_swap(vd, conv, b::resolve("20y", vd), 0, 0, 0.04));
  EXPECT_NE(cal::structure_fingerprint(added), f0) << "adding an instrument must recompile";

  auto smoothing = base;
  smoothing.curves[0].regions[1].reg_sigma = 5.0;  // per-region tension-energy (Phase 2) is baked into R
  EXPECT_NE(cal::structure_fingerprint(smoothing), f0) << "changing per-region smoothing must recompile";
}
