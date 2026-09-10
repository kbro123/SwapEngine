// Credit gate: the hazard-rate (survival) curve + par-CDS instruments (credit layer).
//
// Covered, each a place a wiring bug would hide:
//   1. a flat-hazard curve reprices its OWN par spread exactly — build a CDS at the model par spread, the
//      residual is 0 to machine precision (the model_quote and the market use the same discretisation);
//   2. survival is monotone DECREASING in t with Q(0) = 1 (the defining property of a survival curve);
//   3. calibrating the hazard curve to a STRIP of par CDS spreads recovers the input spreads
//      (rank_deficiency 0), through the standard LM+AAD path;
//   4. for the SAME hazard curve, a HIGHER recovery R gives a LOWER par spread (par spread ∝ (1−R): the
//      protection leg scales with the loss-given-default, the risky annuity does not).
//
//      [Direction note: the coordinator's brief phrased (4) as "higher recovery ⇒ higher par spread", but
//       that is inverted — par spread = protectionPV/riskyAnnuity and protectionPV = (1−R)·∫DF·(−dQ), so
//       s* ∝ (1−R) and a higher R LOWERS the spread. Economically identical: less loss on default ⇒ less
//       spread. The parenthetical "protection leg scales with 1−R" is the correct fact; the test asserts
//       the correct monotonicity it implies.]
//
// QuantLib-free; header-only engine. Everything runs through the standard curve + LM+AAD machinery. Pinned
// against closed forms only (no QuantLib oracle).

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/build/credit_instruments.hpp"
#include "swaps/calibration/credit_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"

namespace curve = swaps::curve;
namespace cal = swaps::calibration;
namespace b = swaps::build;

namespace {

// A globally-flat hazard curve: instantaneous forward hazard == h everywhere, so Q(t) = exp(−h·t).
curve::ModularCurve<double> flat_hazard(double h, const std::vector<double>& knots) {
  auto c = curve::make_modular_curve<double>({{knots, curve::Scheme::Flat}});
  c.set_forwards(Eigen::VectorXd::Constant(static_cast<int>(knots.size()), h));
  return c;
}

// A flat continuously-compounded discount curve DF(t) = exp(−r·t) as a plain callable.
struct FlatDf {
  double r;
  double operator()(double t) const { return std::exp(-r * t); }
};

const std::vector<double> kKnots{1.0, 2.0, 3.0, 5.0, 10.0};

}  // namespace

// 1. A flat-hazard curve reprices its own par spread exactly (residual == 0).
TEST(Credit, FlatHazardRepricesParSpreadExactly) {
  const double h = 0.03, r = 0.02, R = 0.40, T = 5.0;
  const auto hz = flat_hazard(h, kKnots);
  const curve::SurvivalCurve<double> surv{&hz};

  // Sanity: Q(t) = exp(−h·t) and the par spread sits near the credit-triangle value (1−R)·h.
  EXPECT_NEAR(surv.survival(T), std::exp(-h * T), 1e-12);
  const auto probe = b::make_cds(T, /*spread=*/0.0, R, FlatDf{r}, 4, 4, 365.0 / 360.0);
  const double s_model = probe.model_quote<double>(surv);
  // The quarterly protection/premium discretisation of a flat-hazard CDS sits a few bp off the continuous
  // credit triangle (1−R)·h = 180 bp: the closed form is a sanity band only; the VALUE pin is the QuantLib
  // MidPointCdsEngine oracle (swaps_oracle_tests, E5 T1).
  EXPECT_NEAR(s_model, (1.0 - R) * h, 5e-4);
  EXPECT_GT(s_model, 0.0);

  // Build the instrument AT its own par spread => residual is exactly zero.
  const auto cds = b::make_cds(T, s_model, R, FlatDf{r}, 4, 4, 365.0 / 360.0);
  EXPECT_NEAR(cds.model_quote<double>(surv), s_model, 1e-14);
  EXPECT_NEAR(cds.residual<double>(surv), 0.0, 1e-14);
}

// 2. Survival is monotone decreasing with Q(0) = 1.
TEST(Credit, SurvivalMonotoneDecreasingFromOne) {
  const auto hz = flat_hazard(0.05, kKnots);
  const curve::SurvivalCurve<double> surv{&hz};

  EXPECT_NEAR(surv.survival(0.0), 1.0, 1e-14);
  double prev = surv.survival(0.0);
  for (double t = 0.25; t <= 10.0 + 1e-9; t += 0.25) {
    const double q = surv.survival(t);
    EXPECT_LT(q, prev);              // strictly decreasing
    EXPECT_GT(q, 0.0);              // a probability
    EXPECT_GT(surv.default_density(t), 0.0);  // −dQ/dt = h·Q > 0
    prev = q;
  }
}

// 3. Calibrating a multi-point CDS strip recovers the input par spreads (rank_deficiency 0).
TEST(Credit, CalibrateCdsStripRecoversSpreads) {
  const double r = 0.02, R = 0.40;
  const std::vector<double> mats{1.0, 3.0, 5.0, 7.0, 10.0};
  const std::vector<double> tgt{0.0080, 0.0110, 0.0150, 0.0175, 0.0200};

  cal::CreditProblem prob;
  prob.back_times = mats;
  for (std::size_t i = 0; i < mats.size(); ++i)
    prob.instruments.push_back(b::make_cds(mats[i], tgt[i], R, FlatDf{r}, 4, 4, 365.0 / 360.0));

  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), tgt.front() / (1.0 - R));
  const cal::CalibrationResult res = cal::calibrate(prob, x0);
  EXPECT_LT(res.stationarity, 1e-8);
  EXPECT_EQ(res.rank_deficiency, 0);

  auto hz = curve::make_modular_curve<double>(prob.hazard_layout());  // the layout the residual used
  hz.set_forwards(res.x);
  const curve::SurvivalCurve<double> surv{&hz};
  for (std::size_t i = 0; i < mats.size(); ++i)
    EXPECT_NEAR(prob.instruments[i].model_quote<double>(surv), tgt[i], 1e-8);
}

// 4. Higher recovery => lower par spread for the same hazard (protection leg scales with 1−R).
TEST(Credit, HigherRecoveryLowersParSpread) {
  const double h = 0.04, r = 0.02, T = 5.0;
  const auto hz = flat_hazard(h, kKnots);
  const curve::SurvivalCurve<double> surv{&hz};

  const double s_lowR = b::make_cds(T, 0.0, /*R=*/0.20, FlatDf{r}, 4, 4, 365.0 / 360.0).model_quote<double>(surv);
  const double s_midR = b::make_cds(T, 0.0, /*R=*/0.40, FlatDf{r}, 4, 4, 365.0 / 360.0).model_quote<double>(surv);
  const double s_hiR = b::make_cds(T, 0.0, /*R=*/0.60, FlatDf{r}, 4, 4, 365.0 / 360.0).model_quote<double>(surv);

  EXPECT_GT(s_lowR, s_midR);
  EXPECT_GT(s_midR, s_hiR);
  // Par spread ∝ (1−R): the ratio of spreads equals the ratio of (1−R). s(0.20)/s(0.60) = 0.8/0.4 = 2.
  EXPECT_NEAR(s_lowR / s_hiR, (1.0 - 0.20) / (1.0 - 0.60), 1e-9);
  EXPECT_NEAR(s_midR / s_hiR, (1.0 - 0.40) / (1.0 - 0.60), 1e-9);
}
