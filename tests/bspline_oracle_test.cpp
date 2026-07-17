// B-spline curve, end-to-end oracle (docs/bezier-and-moments.md Part A). The control-point B-spline
// must be a usable discount curve: expose it to QuantLib via CurveTermStructure, price OIS swaps off
// it, and confirm QuantLib's fairRate equals our kernel priced off the SAME curve. Because the DFs are
// identical by construction, any gap is a pricing/curve bug -- the real correctness proof that a
// B-spline curve is arbitrage-clean and drop-in wherever a curve type is templated.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;
namespace cv = swaps::curve;

TEST(BSplineOracle, PricesOisConsistentlyWithQuantLib) {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);  // today, day count, SOFR index, calendar (all test-side)

  const std::vector<double> meeting{0.25, 0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  cv::BSplineCurve<double> curve = cv::make_bspline_curve<double>(meeting, back);
  Eigen::VectorXd cp(meeting.size() + back.size());
  cp << 0.030, 0.032,                                              // front forwards
      0.035, 0.037, 0.039, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046;  // back control points
  curve.set_forwards(cp);

  auto ts = ext::make_shared<qlx::CurveTermStructure<cv::BSplineCurve<double>>>(mk.today, mk.dc, &curve);
  ts->enableExtrapolation();
  h.linkTo(ts);  // SOFR forecasts and discounts off the B-spline curve

  double worst = 0.0;
  for (int T : {2, 3, 5, 7, 10, 15, 20, 30}) {
    auto swap = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(Period(T, Years), mk.sofr, 0.03).withDiscountingTermStructure(h));
    swap->deepUpdate();
    const double ql = swap->fairRate();
    const auto sched = qlx::extract_ois_swap(*swap, mk.today, mk.dc);
    const double ours = px::ois_par_rate<double>(sched, curve);
    worst = std::max(worst, std::abs(ours - ql) / std::max(1.0, std::abs(ql)));
  }
  std::cout << "  [bspline-oracle] max rel |ours - QuantLib fairRate| = " << worst << "\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "B-spline curve must price OIS identically to QuantLib";
}
