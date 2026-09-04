// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Tension-spline curve, end-to-end oracle (research note §3,§6). A fixed-tension spline is a linear map
// of the knot forwards, so it must be a usable discount curve: expose it to QuantLib via
// CurveTermStructure, price OIS swaps off it, and confirm QuantLib's fairRate equals our kernel priced
// off the SAME curve. Because the DFs are identical by construction, any gap is a pricing/curve bug --
// the correctness proof that a tension curve is arbitrage-clean and drop-in wherever a curve is templated.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;
namespace cv = swaps::curve;

TEST(TensionOracle, PricesOisConsistentlyWithQuantLib) {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);  // today, day count, SOFR index, calendar (all test-side)

  const std::vector<double> meeting{0.25, 0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  // Sweep a range of tension strengths, including the near-cubic (small σ) and taut (large σ) ends.
  for (double sigma : {0.05, 1.0, 5.0, 25.0}) {
    cv::ModularCurve<double> curve = cv::make_modular_curve<double>(cv::flat_tension(meeting, back, sigma));
    ASSERT_TRUE(curve.is_linear_map()) << "fixed-σ tension curve must stay a linear map";
    Eigen::VectorXd fw(meeting.size() + back.size());
    fw << 0.030, 0.032,                                                  // front forwards
        0.035, 0.037, 0.039, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046;  // back knot forwards
    curve.set_forwards(fw);

    auto ts = ext::make_shared<qlx::CurveTermStructure<cv::ModularCurve<double>>>(mk.today, mk.dc, &curve);
    ts->enableExtrapolation();
    h.linkTo(ts);  // SOFR forecasts and discounts off the tension curve

    double worst = 0.0;
    for (int T : {2, 3, 5, 7, 10, 15, 20, 30}) {
      auto swap = ext::shared_ptr<OvernightIndexedSwap>(
          MakeOIS(Period(T, Years), mk.sofr, 0.03).withDiscountingTermStructure(h));
      swap->deepUpdate();
      const double ql = swap->fairRate();
      const auto fl = qlx::extract_float_leg(swap->overnightLeg(), mk.today, mk.dc);
      const auto fx = qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
      const double ours = px::par_rate<double>(fl, fx, curve, curve);
      worst = std::max(worst, std::abs(ours - ql) / std::max(1.0, std::abs(ql)));
    }
    std::cout << "  [tension-oracle] σ=" << sigma << " max rel |ours - QuantLib fairRate| = " << worst << "\n";
    EXPECT_LT(worst, swaps::tol::curve_rel) << "tension curve must price OIS identically to QuantLib (σ=" << sigma << ")";
  }
}
