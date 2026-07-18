// Phase 5 (batched analytics) gate: the vectorized CompiledPortfolio must reproduce the scalar
// pricing kernel exactly, for any knot forwards x -- not just at the calibrated point.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

struct PortfolioVec : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  swaps::portfolio::Portfolio pf;

  void SetUp() override {
    // Repeat the 9 swaps several times with varied fixed rates/notionals -> 45-position book.
    for (int rep = 0; rep < 5; ++rep)
      for (std::size_t i = 0; i < mk.swaps.size(); ++i)
        pf.positions.push_back(
            {swaps::qlx::extract_float_leg(mk.swaps[i]->overnightLeg(), mk.today, mk.dc),
             swaps::qlx::extract_fixed_leg(mk.swaps[i]->fixedLeg(), mk.today, mk.dc),
             rm::swaps[i].par_rate + 0.001 * (rep - 2), (i % 2 ? 1.0 : -1.0) * (1.0 + rep)});
  }

  double scalar_total(const Eigen::VectorXd& x) const {
    auto c = swaps::curve::make_calibration_curve<double>(prob.meeting_times, prob.back_times);
    c.set_forwards(x);
    return pf.npv<double>(c);
  }
};

TEST_F(PortfolioVec, MatchesScalarKernelAtManyCurves) {
  swaps::portfolio::CompiledPortfolio cp(prob.meeting_times, prob.back_times, pf);
  ASSERT_EQ(cp.n_swaps(), static_cast<int>(pf.positions.size()));

  // A spread of curves: flat levels + a calibrated one + tilts.
  std::vector<Eigen::VectorXd> curves;
  curves.push_back(Eigen::VectorXd::Constant(prob.n_knots(), 0.030));
  curves.push_back(Eigen::VectorXd::Constant(prob.n_knots(), 0.045));
  curves.push_back(cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x);
  Eigen::VectorXd tilt(prob.n_knots());
  for (int i = 0; i < tilt.size(); ++i) tilt[i] = 0.03 + 0.0005 * i;
  curves.push_back(tilt);

  double worst = 0.0;
  for (const auto& x : curves) {
    // per-swap agreement
    const Eigen::VectorXd vec = cp.npv(x);
    auto c = swaps::curve::make_calibration_curve<double>(prob.meeting_times, prob.back_times);
    c.set_forwards(x);
    for (int p = 0; p < cp.n_swaps(); ++p) {
      const double scal =
          pf.positions[p].notional *
          swaps::portfolio::Portfolio::position_npv<double>(pf.positions[p], c);
      worst = std::max(worst, std::abs(vec[p] - scal));
    }
    // total agreement
    EXPECT_NEAR(cp.total_npv(x), scalar_total(x), 1e-10 * std::max(1.0, std::abs(scalar_total(x))));
  }
  std::cout << "  [compiled] book=" << cp.n_swaps() << " times=" << cp.n_times()
            << " worst |vec-scalar| per swap=" << worst << "\n";
  EXPECT_LT(worst, 1e-11);
}
