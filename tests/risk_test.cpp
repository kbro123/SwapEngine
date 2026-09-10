// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference market; the
// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// engine is compared to ITSELF / hand formulas, not to a QuantLib number). DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Phase 5 (risk) gate: the analytic bucketed delta ladder.
//
// The AAD + implicit-function-theorem ladder must equal a bump-and-RE-CALIBRATE ladder computed on
// OUR OWN curve (perturb each market quote, re-solve, reprice the portfolio, finite-difference).
// Comparing on the same curve isolates the risk method; the QuantLib-bump SPEED comparison lives in
// bench/risk_bench.cpp.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/risk.hpp"
#include "swaps/portfolio/portfolio.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

// Perturb the market quote of instrument j. Instrument insertion order IS the residual order
// (avg futures, comp futures, swaps), so residual row j is instrument j.
cal::CalibrationProblem bump_quote(cal::CalibrationProblem p, int j, double eps) {
  p.instruments[j].market += eps;
  return p;
}

double portfolio_npv_at(const cal::CalibrationProblem& prob, const Eigen::VectorXd& x,
                        const swaps::portfolio::Portfolio& pf) {
  auto c = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(prob.meeting_times, prob.back_times));
  c.set_forwards(x);
  return pf.npv<double>(c);
}

}  // namespace

struct Risk : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  swaps::portfolio::Portfolio pf;
  Eigen::VectorXd xstar;

  void SetUp() override {
    // A non-trivial book: the 9 swaps at 50bp off market, alternating pay/receive, varied notional.
    for (std::size_t i = 0; i < mk.swaps.size(); ++i)
      pf.positions.push_back(
          {swaps::qlx::extract_float_leg(mk.swaps[i]->overnightLeg(), mk.today, mk.dc),
           swaps::qlx::extract_fixed_leg(mk.swaps[i]->fixedLeg(), mk.today, mk.dc),
           rm::swaps[i].par_rate + 0.005, (i % 2 ? 1.0 : -1.0) * (1.0 + i)});
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), 0.035);
    xstar = cal::calibrate(prob, x0, true).x;
  }
};

TEST_F(Risk, AnalyticLadderMatchesBumpAndRecalibrate) {
  const Eigen::VectorXd analytic = cal::bucketed_delta(prob, xstar, pf);
  ASSERT_EQ(analytic.size(), prob.n_residuals());

  const double eps = 1e-6;
  Eigen::VectorXd bump(prob.n_residuals());
  for (int j = 0; j < prob.n_residuals(); ++j) {
    const auto rp = cal::calibrate(bump_quote(prob, j, eps), xstar, true);   // warm from x*
    const auto rm_ = cal::calibrate(bump_quote(prob, j, -eps), xstar, true);
    bump[j] = (portfolio_npv_at(prob, rp.x, pf) - portfolio_npv_at(prob, rm_.x, pf)) / (2 * eps);
  }

  // Scale-relative comparison (bump noise floors absolute accuracy; some buckets are ~0).
  const double scale = bump.cwiseAbs().maxCoeff();
  const double worst_abs = (analytic - bump).cwiseAbs().maxCoeff();
  double worst_rel_sig = 0.0;
  for (int j = 0; j < bump.size(); ++j)
    if (std::abs(bump[j]) > 1e-3 * scale)
      worst_rel_sig = std::max(worst_rel_sig, std::abs(analytic[j] - bump[j]) / std::abs(bump[j]));

  std::cout << "  [risk ladder] buckets=" << analytic.size() << " scale=" << scale
            << " worst_abs/scale=" << worst_abs / scale << " worst_rel(significant)=" << worst_rel_sig
            << "\n";
  EXPECT_LT(worst_abs / scale, 1e-4);   // re-calibration bump noise dominates
  EXPECT_LT(worst_rel_sig, 1e-4);
}

TEST_F(Risk, LadderIsNonTrivial) {
  const Eigen::VectorXd d = cal::bucketed_delta(prob, xstar, pf);
  EXPECT_GT(d.cwiseAbs().maxCoeff(), 0.0);
  // A swap book has meaningful sensitivity to the long swap quotes.
  std::cout << "  [risk ladder] |delta|_inf=" << d.cwiseAbs().maxCoeff() << "\n";
}
