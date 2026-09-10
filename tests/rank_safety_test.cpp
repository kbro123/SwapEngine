// RANK SAFETY at the one shared threshold (E3 register G5 / B2 / S3, fixed 2026-09-10). Every operator that
// inverts a Jacobian does it rank-safely at kRankThreshold: the streamer (already), the background prefetch
// worker (was an un-thresholded QR: |M| up to 6e14 on a rank-deficient bundle), WarmCalibrator (was the QR's
// default threshold). The prefetch is armed only where the worker's M-only hand-over is exact.
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
#include "swaps/calibration/warm.hpp"
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

TEST(RankSafety, WarmCalibratorIsRankSafe) {
  const cal::BundleProblem p = deficient_square();
  const cal::WarmCalibrator<cal::BundleProblem> wc(p, x0);
  const cal::HybridBundleResidual eng(p);
  const Eigen::MatrixXd ref = rank_safe_pinv(eng.jacobian(x0));
  EXPECT_LT(wc.sensitivity().cwiseAbs().maxCoeff(), 1e6);
  EXPECT_LT((wc.sensitivity() - ref).cwiseAbs().maxCoeff(), 1e-8 * ref.cwiseAbs().maxCoeff());
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
