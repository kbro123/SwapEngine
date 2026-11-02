// E5 taxonomy: T2 calibration (PRINCIPLES P15: every curve of every QuantLib-free fixture has a front anchor)
// THE FRONT ANCHOR RULE over every QuantLib-free fixture: the shape ladder's rungs, the G10 desk (with and without turns)
// and the G4 desk's variants. calibration::front_anchors names, per curve, the row that reads its forward over one window
// from the front; a curve with none is a fixture defect (its first knot is a term pillar's value extrapolated to day one).
#include <gtest/gtest.h>

#include <cstdio>
#include <utility>
#include <string>
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
    std::printf("  %-22s curve %2d first knot %7.4fy  anchor row %3d  window [%.4f, %.4f]  tenor %7.4f\n", s.name.c_str(), a.curve, a.first_knot,
                a.row, a.start, a.end, a.tenor);
  }
}
}  // namespace

// THE ANCHOR IS THE CURVE'S OWN FIXING (owner's rule, 2026-11-02). A curve stating its index tenor is anchored only by
// a window of that length from the front: on an overnight curve the 1D fixing (over a four-day weekend too), never a
// 1W deposit, the straddling 1M future or a 1Y OIS; under a leading Flat region the fixing reads the first segment,
// under a Hermite front it IS the first knot. A curve stating no tenor keeps the 2026-11-01 single-window rule: the
// window ends at the first knot (the reference market's 1M future reaching the FOMC knot), a window short of it does not.
TEST(FrontAnchorRule, TheAnchorIsTheCurvesOwnFixing) {
  const auto problem = [](double tenor, std::vector<cv::CurveModule> regions, double s, double e) {
    cal::BundleProblem p;
    cal::BundleCurveSpec c;
    c.regions = std::move(regions);
    c.fixing_tenor = tenor;
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
  const auto row = [&](double tenor, std::vector<cv::CurveModule> regions, double s, double e) {
    return cal::front_anchors(problem(tenor, std::move(regions), s, e)).front().row;
  };
  cv::CurveModule flat, back, herm;
  flat.scheme = cv::Scheme::Flat;
  flat.knots = {0.12, 0.25};  // the first meeting six weeks out
  back.scheme = cv::Scheme::Hermite;
  back.knots = {1.0, 2.0, 5.0};
  const std::vector<cv::CurveModule> meeting_front = {flat, back};
  herm.scheme = cv::Scheme::Hermite;
  herm.knots = {1.0 / 365.0, 1.0, 2.0, 5.0};  // the 1D fixing is the first knot
  const std::vector<cv::CurveModule> hermite_front = {herm};
  const double on = 1.0 / 365.0;
  EXPECT_EQ(row(on, meeting_front, 0.0, 1.0 / 365.0), 0) << "the 1D fixing under a meeting front anchors it";
  EXPECT_EQ(row(on, meeting_front, 0.0, 4.0 / 360.0), 0) << "the 1D fixing over a four-day weekend is the 1D fixing";
  EXPECT_EQ(row(on, meeting_front, 0.0, 7.0 / 365.0), -1) << "a 1W deposit is not the overnight curve's fixing";
  EXPECT_EQ(row(on, meeting_front, -0.02, 0.06), -1) << "the straddling 1M future is not the fixing";
  EXPECT_EQ(row(on, meeting_front, 0.0, 1.0), -1) << "a 1Y OIS (one coupon) is not the fixing";
  EXPECT_EQ(row(on, hermite_front, 0.0, 1.0 / 365.0), 0) << "under a Hermite front the 1D fixing is the first knot";
  EXPECT_EQ(row(on, hermite_front, 0.06, 0.06 + 1.0 / 365.0), -1) << "a 1D window starting past front_tol is not the front";
  cv::CurveModule h3;
  h3.scheme = cv::Scheme::Hermite;
  h3.knots = {0.2528, 1.0, 2.0, 5.0};  // the 3M fixing's end (ACT/360 stub) is the first knot
  EXPECT_EQ(row(0.25, {h3}, 0.0055, 0.2528), 0) << "a 3M tenor curve is anchored by its 3M fixing";
  EXPECT_EQ(row(0.25, {h3}, 0.0055, 0.0055 + 1.0 / 365.0), -1) << "a 1D fixing is not a 3M curve's fixing";
  EXPECT_EQ(row(0.25, {h3}, 0.0055, 0.5), -1) << "a 6M window is not a 3M curve's fixing";
  // no stated tenor: the 2026-11-01 single-window rule, the window ending at the first knot
  EXPECT_EQ(row(-1.0, meeting_front, -0.02, 0.12), 0) << "unstated: the straddling 1M future reaching the FOMC knot anchors";
  EXPECT_EQ(row(-1.0, meeting_front, 0.0, 7.0 / 365.0), -1) << "unstated: a window short of the first knot does not";
  EXPECT_EQ(row(-1.0, meeting_front, 0.06, 0.14), -1) << "unstated: a window starting past front_tol (the second 1M future) is not an anchor";
  EXPECT_EQ(row(-1.0, meeting_front, -0.05, -0.01), -1) << "a window entirely before the value date reads nothing";
  EXPECT_EQ(cal::fixing_tenor_label(1.0 / 365.0), "1D");
  EXPECT_EQ(cal::fixing_tenor_label(7.0 / 365.0), "1W");
  EXPECT_EQ(cal::fixing_tenor_label(0.25), "3M");
  EXPECT_EQ(cal::fixing_tenor_label(0.5), "6M");
  EXPECT_EQ(cal::fixing_tenor_label(1.0), "1Y");
  EXPECT_EQ(cal::fixing_tenor_label(28.0 / 365.0), "4W");
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
