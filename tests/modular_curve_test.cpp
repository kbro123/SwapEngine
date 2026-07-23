// Two guarantees for the curve-composition layer:
//
//  (a) The RUNTIME builder reproduces the compile-time one. A ModularCurve assembled from
//      {Flat(meeting), Hermite(back)} modules is bit-identical to the compile-time
//      the shipped flat_hermite layout -- same policies, same C0 stitching.
//
//  (b) The pole star points at the SHIPPED curve. We calibrate the Hermite curve, wrap it as a
//      QuantLib::YieldTermStructure via the generalized CurveTermStructure adapter, and let QuantLib
//      price the instruments off it -- matching our own templated kernel to 1e-10. This closes the
//      §1/§3a loop for the interpolation we actually deploy (previously only the natural-cubic curve
//      was validated through the QuantLib YTS oracle).

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;

namespace {
::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure()
                       << "got=" << got << " want=" << want << " rel=" << err << " > " << rel);
}
}  // namespace

TEST(ModularCurve, RuntimeBuilderMatchesCompileTimeCurve) {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);

  // Same region composition, one compile-time and one assembled at runtime from CurveModule pieces.
  auto ref = cv::make_modular_curve<double>(cv::flat_hermite(prob.meeting_times, prob.back_times));
  auto mod = cv::make_modular_curve<double>(
      {{prob.meeting_times, cv::Scheme::Flat}, {prob.back_times, cv::Scheme::Hermite}});
  ASSERT_EQ(mod.n_knots(), ref.n_knots());
  EXPECT_TRUE(mod.is_linear_map());

  Eigen::VectorXd x(prob.n_knots());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.030 + 0.0006 * i;  // an arbitrary tilted curve
  ref.set_forwards(x);
  mod.set_forwards(x);

  double worst = 0.0;
  for (double t = 0.02; t <= 30.0; t += 0.05) {
    worst = std::max(worst, std::abs(ref.forward(t) - mod.forward(t)));
    worst = std::max(worst, std::abs(ref.discount(t) - mod.discount(t)));
  }
  std::cout << "  [modular] worst |runtime - compile-time| = " << worst << "\n";
  EXPECT_LT(worst, 1e-14) << "runtime ModularCurve must reproduce the compile-time curve";
}

// Fixture: calibrate the Hermite curve, expose it to QuantLib as a YieldTermStructure.
struct HermiteOracle : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  cv::ModularCurve<double> curve = cv::make_modular_curve<double>(cv::flat_hermite(prob.meeting_times, prob.back_times));
  ext::shared_ptr<swaps::qlx::CurveTermStructure<cv::ModularCurve<double>>> ts;

  void SetUp() override {
    const Eigen::VectorXd x = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.04), true).x;
    curve.set_forwards(x);
    ts = ext::make_shared<swaps::qlx::CurveTermStructure<cv::ModularCurve<double>>>(mk.today, mk.dc, &curve);
    ts->enableExtrapolation();
    h.linkTo(ts);
    for (auto& s : mk.swaps) s->deepUpdate();
  }
};

TEST_F(HermiteOracle, QuantLibPricesSwapsOffOurHermiteCurve) {
  double worst = 0.0;
  for (const auto& swap : mk.swaps) {
    const double ql = swap->fairRate();  // QuantLib, off our Hermite YieldTermStructure
    const auto fl = swaps::qlx::extract_float_leg(swap->overnightLeg(), mk.today, mk.dc);
    const auto fx = swaps::qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
    const double ours = swaps::pricing::par_rate<double>(fl, fx, curve, curve);  // our kernel, same curve
    EXPECT_TRUE(close(ours, ql, swaps::tol::curve_rel)) << " swap maturity " << swap->maturityDate();
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [hermite YTS · swaps] max |ours - QuantLib| = " << worst << "\n";
}

TEST_F(HermiteOracle, QuantLibPricesFuturesOffOurHermiteCurve) {
  double worst = 0.0;
  int nc = 0, na = 0;
  for (const auto& f : mk.futures) {
    if (f.quarterly) {
      OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Compound);
      const double ql = 1.0 - qlf.NPV() / 100.0;
      const auto obs = swaps::qlx::make_observation(
          {{f.start, f.end}}, 0.0, mk.sofr->dayCounter().yearFraction(f.start, f.end), mk.today, mk.dc);
      const double ours = swaps::pricing::rate<double>(obs, curve);
      EXPECT_TRUE(close(ours, ql, swaps::tol::curve_rel)) << " 3M future " << f.start << ".." << f.end;
      worst = std::max(worst, std::abs(ours - ql));
      ++nc;
    } else {
      OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Simple);
      const double ql = 1.0 - qlf.NPV() / 100.0;
      const auto obs = rb::avg_future_obs(mk, f);
      const double ours = swaps::pricing::rate<double>(obs, curve);
      EXPECT_TRUE(close(ours, ql, swaps::tol::curve_rel)) << " 1M future " << f.start << ".." << f.end;
      worst = std::max(worst, std::abs(ours - ql));
      ++na;
    }
  }
  ASSERT_EQ(nc, 8);
  ASSERT_EQ(na, 12);
  std::cout << "  [hermite YTS · futures] max |ours - QuantLib| = " << worst << "\n";
}
