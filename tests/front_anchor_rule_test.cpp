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
#include "swaps/curve/curve_module.hpp"

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

// A leading Flat region's first segment is ONE value, so any window reading it anchors the curve: a 1W deposit under a
// first meeting six weeks out (the web composer's ESTR seed: its deposit pins the [spot, first-meeting] stub directly,
// SVD-verified there) is an anchor exactly as the reference market's 1M future reaching the FOMC knot is; a window that
// starts past front_tol (the SECOND 1M future) is not, and a window entirely before the value date reads nothing.
TEST(FrontAnchorRule, AnyWindowReadOfALeadingFlatSegmentAnchorsIt) {
  const auto problem = [](double s, double e) {
    cal::BundleProblem p;
    cal::BundleCurveSpec c;
    cv::CurveModule front, back;
    front.scheme = cv::Scheme::Flat;
    front.knots = {0.12, 0.25};  // the first meeting six weeks out
    back.scheme = cv::Scheme::Hermite;
    back.knots = {1.0, 2.0, 5.0};
    c.regions = {front, back};
    p.curves = {c};
    cal::Instrument depo;
    depo.quote = cal::QuoteKind::Rate;
    depo.forecast = 0;
    depo.obs.sub_start = {s};
    depo.obs.sub_end = {e};
    depo.obs.tau_index = e - s;
    depo.market = 0.04;
    p.instruments = {depo};
    return p;
  };
  EXPECT_EQ(cal::front_anchors(problem(0.0, 0.02)).front().row, 0) << "a 1W deposit inside the first flat segment anchors it";
  EXPECT_EQ(cal::front_anchors(problem(-0.02, 0.06)).front().row, 0) << "the straddling 1M future (window opened before the value date) anchors it";
  EXPECT_EQ(cal::front_anchors(problem(0.0, 0.14)).front().row, 0) << "a window reaching past the first knot still anchors a Flat front";
  EXPECT_EQ(cal::front_anchors(problem(0.06, 0.14)).front().row, -1) << "a window starting past front_tol (the second 1M future) is not an anchor";
  EXPECT_EQ(cal::front_anchors(problem(-0.05, -0.01)).front().row, -1) << "a window entirely before the value date reads nothing";
}

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
