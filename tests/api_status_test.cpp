// BundleSession seams for C1/C3 (2026-09-10): non-finite quotes are refused at recalibrate/rebind/
// stream_update; a failed stream tick is visible (last_converged/last_status/last_reason) and never commits.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

#include "swaps/api/bundle_api.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace api = swaps::api;
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
cal::BundleProblem square() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  for (int T : {1, 2, 3, 5, 7, 10}) p.instruments.push_back(par_swap(T));
  Eigen::VectorXd xt(6); xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < 6; ++i) p.instruments[i].market = r[i];
  return p;
}
}  // namespace

TEST(ApiStatus, NonFiniteQuotesAreRefusedAtEverySeam) {
  const auto p = square();
  api::BundleSession s(p);
  const auto& r = s.calibrate(Eigen::VectorXd::Constant(6, 0.03));
  EXPECT_TRUE(r.converged) << r.status;
  const Eigen::VectorXd x = s.x();
  Eigen::VectorXd q = p.market();
  q[2] = std::nan("");
  EXPECT_THROW(s.recalibrate(q), std::runtime_error);
  EXPECT_EQ((s.x() - x).cwiseAbs().maxCoeff(), 0.0) << "a refused market must not touch the curve";
  auto pbad = p;
  pbad.instruments[2].market = INFINITY;
  EXPECT_THROW(s.rebind(pbad), std::runtime_error);
  s.start_streaming();
  EXPECT_THROW(s.stream_update(q), std::runtime_error);
  EXPECT_EQ((s.x() - x).cwiseAbs().maxCoeff(), 0.0);
  // a session BUILT on a NaN quote cannot calibrate at all
  api::BundleSession sbad(pbad);
  EXPECT_THROW(sbad.calibrate(Eigen::VectorXd::Constant(6, 0.03)), std::invalid_argument);
}

TEST(ApiStatus, AFailedStreamTickIsVisibleAndNotCommitted) {
  const auto p = square();
  api::BundleSession s(p);
  s.calibrate(Eigen::VectorXd::Constant(6, 0.03));
  s.start_streaming();
  const Eigen::VectorXd x = s.x();
  EXPECT_TRUE(s.last_converged());
  EXPECT_EQ(s.last_status(), 0);
  EXPECT_STREQ(s.last_reason(), "converged");
  const Eigen::VectorXd q_abs = p.market().array() + 5.0;  // +500 %: the tick cannot converge
  s.stream_update(q_abs);
  EXPECT_FALSE(s.last_converged());
  EXPECT_NE(s.last_status(), 0);
  EXPECT_STRNE(s.last_reason(), "converged");
  EXPECT_EQ((s.x() - x).cwiseAbs().maxCoeff(), 0.0) << "a failed tick must leave x() at the last converged curve";
  s.stream_update(p.market());
  EXPECT_TRUE(s.last_converged()) << s.last_reason();
  EXPECT_LT((s.x() - x).cwiseAbs().maxCoeff(), 1e-12);
}
