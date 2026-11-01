// E5 taxonomy: T2 calibration (PRINCIPLES P15: every curve of every QuantLib-free fixture has a front anchor)
// THE FRONT ANCHOR RULE over every QuantLib-free fixture: the shape ladder's rungs, the G10 desk (with and without turns)
// and the G4 desk's variants. calibration::front_anchors names, per curve, the row that reads its forward over one window
// from the front; a curve with none is a fixture defect (its first knot is a term pillar's value extrapolated to day one).
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "g10_market.hpp"
#include "g4_market.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/diagnostics.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
using swaps::shapes::Shape;

namespace {
void expect_anchored(const Shape& s) {
  const auto anchors = cal::front_anchors(s.prob);
  for (const cal::FrontAnchor& a : anchors) {
    if (a.first_knot < 0.0) continue;  // a curve with no interpolation knots has no front
    EXPECT_GE(a.row, 0) << s.name << ": curve " << a.curve << " (first knot " << a.first_knot
                        << "y) has no front anchor -- no row reads its forward over one window from the front";
    std::printf("  %-22s curve %2d first knot %7.4fy  anchor row %3d  window [%.4f, %.4f]\n", s.name.c_str(), a.curve, a.first_knot,
                a.row, a.start, a.end);
  }
}
}  // namespace

TEST(FrontAnchorRule, EveryLadderRungIsAnchored) {
  for (const Shape& s : swaps::shapes::ladder()) expect_anchored(s);
}

TEST(FrontAnchorRule, TheG10DeskIsAnchored) {
  expect_anchored(swaps::shapes::g10_desk(false));
  expect_anchored(swaps::shapes::g10_desk(true));
}

TEST(FrontAnchorRule, TheG4DeskIsAnchored) {
  for (cv::Scheme sch : {cv::Scheme::Hermite, cv::Scheme::MonotoneCubic, cv::Scheme::MonotoneConvex}) {
    swaps::shapes::g4::Opts o;
    o.back = sch;
    expect_anchored(swaps::shapes::g4_desk(o));
  }
}
