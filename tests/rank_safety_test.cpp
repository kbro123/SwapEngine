// E5 taxonomy: T2 calibration (optimum / stationarity / recovery) | T3 cross-path parity (two engine paths, same inputs)
// RANK SAFETY at the one shared threshold (E3 register G5 / B2 / S3, fixed 2026-09-10). Every operator that
// inverts a Jacobian does it rank-safely at kRankThreshold: the streamer (already), the background prefetch
// worker (was an un-thresholded QR: |M| up to 6e14 on a rank-deficient bundle). (WarmCalibrator, the third
// operator, was retired in E6.1 -- the streamer is the warm path.) The prefetch is armed only where the worker's M-only hand-over is exact.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "swaps/calibration/background_jacobian.hpp"
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

TEST(RankSafety, BackgroundWorkerReturnsTheMinimumNormOperator) {
  const cal::BundleProblem p = deficient_square();
  const cal::HybridBundleResidual eng(p);
  const Eigen::MatrixXd ref = rank_safe_pinv(eng.jacobian(x0));
  cal::BackgroundJacobian<cal::BundleProblem> bg(p);
  bg.request(x0);
  Eigen::VectorXd xb; Eigen::MatrixXd M;
  const auto t0 = std::chrono::steady_clock::now();
  while (!bg.try_take(xb, M)) {
    ASSERT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 10.0) << "worker never answered";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_LT(M.cwiseAbs().maxCoeff(), 1e6) << "an un-thresholded QR gave |M| ~ 6e14 on this bundle";
  EXPECT_LT((M - ref).cwiseAbs().maxCoeff(), 1e-8 * ref.cwiseAbs().maxCoeff()) << "the worker's M is the same rank-safe pseudo-inverse the streamer factors";
}

// The prefetch is exact only for a square, unbanded, unregularised problem; elsewhere the option is ignored
// (the worker would hand over an un-regularised M without B and never see later quotes).
TEST(RankSafety, PrefetchIsArmedOnlyWhereItIsExact) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(par_swap(T));  // 10 rows, 6 knots: over-determined
  const Eigen::VectorXd r = p.residuals<double>(x0);
  for (int i = 0; i < 10; ++i) p.instruments[i].market = r[i] + (i % 2 ? 1e-4 : -1e-4);
  const Eigen::VectorXd xs = cal::calibrate(p, x0).x;
  cal::StreamingCalibrator<cal::BundleProblem>::Options popt;
  popt.prefetch = true;
  popt.prefetch_drift = 1e-5;
  cal::StreamingCalibrator<cal::BundleProblem> pref(p, xs, p.market(), popt);
  cal::StreamingCalibrator<cal::BundleProblem> sync(p, xs, p.market(), {});
  for (int t = 1; t <= 60; ++t) {
    Eigen::VectorXd q = p.market();
    for (int i = 0; i < q.size(); ++i) q[i] += 30e-4 * std::sin(0.1 * t + 0.3 * i);
    pref.update(q);
    sync.update(q);
    EXPECT_LT((pref.current() - sync.current()).cwiseAbs().maxCoeff(), 1e-12);
  }
  EXPECT_EQ(pref.prefetch_hits(), 0) << "an over-determined problem must not use the M-only worker";
  EXPECT_EQ(pref.refresh_count(), sync.refresh_count());
}

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
