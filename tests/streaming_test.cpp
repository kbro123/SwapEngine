// Stage-2 streaming gate. The EXACT (frozen-Newton) streaming path is what a live pricer runs, so it
// must, on EVERY tick:
//   (a) ROUND-TRIP: reprice the calibration instruments back to the input quotes -- ||model_rates(x) - q||
//       at the Newton tolerance, so we never show an arbitrageable (stale) price;
//   (b) EXACTNESS: land on an independent from-scratch exact solve;
//   (c) reuse the cached Jacobian -- recomputing it only on genuine staleness, never per envelope crossing.
//
// The square problem is exactly solvable, and the market here is generated from a real curve, so the
// round-trip residual is genuinely ~0 (this is NOT the over-determined case where residuals are non-zero
// by construction -- see calibration_test for that).

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>

#include "reference_curve.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

struct Streaming : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  cal::CompiledResidual cr{prob};
  Eigen::VectorXd x0, q0;

  void SetUp() override {
    x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
    q0 = cr.model_rates(x0);
  }

  // Independent exact solve at market q (full Newton with a fresh AAD Jacobian each step).
  Eigen::VectorXd exact(const Eigen::VectorXd& q, Eigen::VectorXd x) const {
    for (int k = 0; k < 12; ++k) {
      const Eigen::VectorXd r = cr.model_rates(x) - q;
      const Eigen::MatrixXd J = cal::aad_jacobian(prob, x);
      const Eigen::VectorXd dx = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(r);
      x -= dx;
      if (dx.cwiseAbs().maxCoeff() < 1e-13) break;
    }
    return x;
  }
};

TEST_F(Streaming, ExactPathRoundTripsAndMatchesExactSolve) {
  cal::StreamingCalibrator<>::Options opt;  // exact=true by default
  cal::StreamingCalibrator sc(prob, x0, q0, opt);

  double worst_rt = 0, worst_ex = 0;
  // 400 chained correlated moves ranging up to ~25bp from base (level sweep + twist).
  for (int t = 1; t <= 400; ++t) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i)
      q[i] += 25e-4 * std::sin(0.09 * t) * std::sin(0.6 * i + 1.0) +
              10e-4 * std::sin(0.05 * t + 1.0) * (i / double(q.size()) - 0.5);
    sc.update(q);
    const Eigen::VectorXd& x = sc.current();
    worst_rt = std::max(worst_rt, (cr.model_rates(x) - q).cwiseAbs().maxCoeff());
    worst_ex = std::max(worst_ex, (x - exact(q, x)).cwiseAbs().maxCoeff());
  }
  std::cout << "  [streaming exact] worst round-trip=" << worst_rt << " worst vs exact=" << worst_ex
            << " recalcs=" << sc.refresh_count() - 1 << "\n";
  EXPECT_LT(worst_rt, 1e-8) << "every tick must reprice the instruments back to the input quotes";
  EXPECT_LT(worst_ex, 1e-8) << "streamed curve must equal an independent exact solve";
}

TEST_F(Streaming, SubBpMovesReuseTheCachedJacobian) {
  cal::StreamingCalibrator<>::Options opt;
  cal::StreamingCalibrator sc(prob, x0, q0, opt);
  for (int t = 1; t <= 200; ++t) {
    Eigen::VectorXd q = q0;
    for (int i = 0; i < q.size(); ++i) q[i] += 0.5e-4 * std::sin(0.3 * t + 0.4 * i);
    sc.update(q);
    EXPECT_LT((cr.model_rates(sc.current()) - q).cwiseAbs().maxCoeff(), 1e-8);
  }
  EXPECT_EQ(sc.refresh_count() - 1, 0) << "sub-bp moves must ride the cached Jacobian with zero recalcs";
}
