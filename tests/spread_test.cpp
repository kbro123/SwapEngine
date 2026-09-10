// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference market; the
// E5 taxonomy: T2 calibration (optimum / stationarity / recovery)
// engine is compared to ITSELF / hand formulas, not to a QuantLib number). DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Phase 4 gate: forward-spread curve calibrated to a fixed base curve.
//
//   (a) Decomposition: SpreadCurve DF = base DF * exp(-spread integral); forward = base + spread.
//   (b) Recovery: with targets generated from a KNOWN spread, calibration over the spread forwards
//       recovers it (residual -> 0), driven by the SAME AAD-LM code as the base curve.
//   (c) The spread AAD Jacobian matches bump-and-reprice.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "spread_reference.hpp"
#include "swaps/curve/curve_module.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

Eigen::MatrixXd bump_jacobian(const swaps::testing::SpreadCalibrationProblem& p, const Eigen::VectorXd& s,
                              double h) {
  Eigen::MatrixXd J(p.n_residuals(), p.n_knots());
  for (int k = 0; k < p.n_knots(); ++k) {
    Eigen::VectorXd sp = s, sm = s;
    sp[k] += h;
    sm[k] -= h;
    J.col(k) = (p.residuals<double>(sp) - p.residuals<double>(sm)) / (2 * h);
  }
  return J;
}

}  // namespace

struct SpreadCal : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem base_prob = rb::build_square_problem(mk);
  swaps::curve::ModularCurve<double> base =
      swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(base_prob.meeting_times, base_prob.back_times));
  swaps::testing::SpreadCalibrationProblem sp;
  Eigen::VectorXd s_true;

  void SetUp() override {
    // Calibrate the base curve to the SOFR market.
    base.set_forwards(cal::calibrate(base_prob, Eigen::VectorXd::Constant(base_prob.n_knots(), 0.035), true).x);

    // Spread problem: same instrument/knot structure, spread knots, fixed base.
    sp.inst = rb::build_square_problem(mk);
    sp.base = &base;

    // A non-trivial known spread (~10-35 bp), and set targets so residual(s_true) == 0.
    s_true.resize(sp.n_knots());
    for (int i = 0; i < s_true.size(); ++i) s_true[i] = 0.0022 + 0.0012 * std::cos(0.3 * i);
    const Eigen::VectorXd r0 = sp.residuals<double>(s_true);
    // Instrument insertion order IS the residual order (avg | comp | swaps).
    for (int i = 0; i < static_cast<int>(sp.inst.instruments.size()); ++i)
      sp.inst.instruments[i].market += r0[i];
  }
};

TEST_F(SpreadCal, DecomposesIntoBaseTimesSpread) {
  swaps::testing::SpreadCurve<double> total(base, sp.inst.meeting_times, sp.inst.back_times);
  total.set_spreads(s_true);
  auto spread_only = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(sp.inst.meeting_times, sp.inst.back_times));
  spread_only.set_forwards(s_true);  // the spread's own two-region curve

  double worst_df = 0, worst_fwd = 0;
  for (int i = 1; i <= 300; ++i) {
    const double t = base.max_time() * i / 300.0;
    worst_df = std::max(worst_df, std::abs(total.discount(t) - base.discount(t) * spread_only.discount(t)));
    worst_fwd = std::max(worst_fwd, std::abs(total.forward(t) - (base.forward(t) + spread_only.forward(t))));
  }
  std::cout << "  [spread] max|DF - base*spread|=" << worst_df << " max|fwd - (base+spread)|=" << worst_fwd << "\n";
  EXPECT_LT(worst_df, 1e-14);
  EXPECT_LT(worst_fwd, 1e-15);
}

TEST_F(SpreadCal, RecoversTheKnownSpread) {
  EXPECT_LT(sp.residuals<double>(s_true).cwiseAbs().maxCoeff(), 1e-12) << "targets set to zero residual";

  const Eigen::VectorXd s0 = Eigen::VectorXd::Zero(sp.n_knots());  // cold start: zero spread
  const cal::CalibrationResult res = cal::calibrate(sp, s0, true);
  std::cout << "  [spread] iters=" << res.iterations << " rms=" << res.rms_residual
            << " ||s*-s_true||_inf=" << (res.x - s_true).cwiseAbs().maxCoeff()
            << " stat=" << res.stationarity << "\n";
  EXPECT_LT(res.rms_residual, 1e-9);
  EXPECT_LT((res.x - s_true).cwiseAbs().maxCoeff(), 1e-7);
}

TEST_F(SpreadCal, AadJacobianMatchesBump) {
  const Eigen::MatrixXd Jaad = cal::aad_jacobian(sp, s_true);
  const Eigen::MatrixXd Jbump = bump_jacobian(sp, s_true, 1e-6);
  const double scale = Jbump.cwiseAbs().maxCoeff();
  double worst_rel_sig = 0;
  for (int i = 0; i < Jaad.rows(); ++i)
    for (int k = 0; k < Jaad.cols(); ++k)
      if (std::abs(Jbump(i, k)) > 1e-3 * scale)
        worst_rel_sig = std::max(worst_rel_sig, std::abs(Jaad(i, k) - Jbump(i, k)) / std::abs(Jbump(i, k)));
  std::cout << "  [spread] AAD vs bump worst_rel(significant)=" << worst_rel_sig << "\n";
  EXPECT_LT(worst_rel_sig, swaps::tol::jacobian_rel);
}
