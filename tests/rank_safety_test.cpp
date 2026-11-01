// E5 taxonomy: T2 calibration (optimum / stationarity / recovery) | T3 cross-path parity (two engine paths, same inputs)
// RANK SAFETY at the one shared threshold (E3 register G5 / B2 / S3, fixed 2026-09-10). Every operator that
// inverts a Jacobian does it rank-safely at kRankThreshold: the streamer. (WarmCalibrator was retired in E6.1 and
// the background prefetch worker on 2026-11-01, review item 1.2 -- the streamer is the one warm path.)
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}
// A SQUARE but RANK-DEFICIENT bundle: 6 knots, 6 rows, the 5y swap quoted twice (rank 5).
cal::BundleProblem deficient_square() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  for (int T : {1, 2, 3, 5, 5, 10}) p.instruments.push_back(par_swap(T));
  Eigen::VectorXd xt(6); xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < 6; ++i) p.instruments[i].market = r[i];
  return p;
}
Eigen::MatrixXd rank_safe_pinv(const Eigen::MatrixXd& J) {
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(cal::kRankThreshold);
  cod.compute(J);
  return cod.pseudoInverse();
}
const Eigen::VectorXd x0 = (Eigen::VectorXd(6) << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047).finished();
}  // namespace

// The STREAMER on a rank-deficient bundle (E5.4 2026-09-10: the mutation harness showed that replacing its
// COD pseudo-inverse with a tolerance-free LDLT of the singular normal matrix -- the exact bug the factor()
// comment describes -- survived every non-oracle test). A consistent tick must converge, stay finite, reprice
// the shifted market on every row, and leave the null state direction where the anchor put it.
TEST(RankSafety, StreamerLeavesTheNullDirectionAtTheAnchor) {
  const cal::BundleProblem p = deficient_square();
  const Eigen::VectorXd q0 = p.market();
  const Eigen::VectorXd xa = cal::calibrate(p, x0).x;  // the rank-safe anchor
  ASSERT_TRUE(xa.allFinite());
  // The null state direction: the right-singular vector of the Jacobian's smallest singular value.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(cal::aad_jacobian(p, xa), Eigen::ComputeThinV);
  ASSERT_LT(svd.singularValues()(5), 1e-12 * svd.singularValues()(0)) << "the fixture must be rank-deficient";
  const Eigen::VectorXd null_dir = svd.matrixV().col(5);
  cal::StreamingCalibrator<cal::BundleProblem> sc(p, xa, q0, {});
  for (int t = 1; t <= 5; ++t) {
    const Eigen::VectorXd q = q0.array() + 1e-4 * t;  // a consistent shift (both 5y rows move together)
    const cal::StreamTick tick = sc.update(q);
    ASSERT_TRUE(tick.converged) << "tick " << t << ": " << tick.reason();
    ASSERT_TRUE(sc.current().allFinite()) << "tick " << t;
    cal::BundleProblem pq = p;
    for (int i = 0; i < 6; ++i) pq.instruments[i].market = q[i];
    EXPECT_LT(pq.residuals<double>(sc.current()).cwiseAbs().maxCoeff(), 1e-8) << "tick " << t << " must reprice the shifted market";
    EXPECT_LT(std::abs(null_dir.dot(sc.current() - xa)), 1e-9) << "tick " << t << " moved the null direction";
  }
}
