// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T2 calibration (optimum / stationarity / recovery)
// Validation for the Hyman-filtered MonotoneCubic region -- the first NON-LINEAR (is_linear_map=false)
// region policy, the concrete curve that exercises the AAD fallback tier (CLAUDE.md §2). Engine-only
// (no QuantLib): the region contract (C0 clamp, no-overshoot monotonicity), the is_linear_map guard, the
// AAD-engine ROUTING for a non-linear-curve problem, and a real calibrate-through-AAD + reprice fit. The
// bit-exact match to QuantLib's MonotonicCubicNaturalSpline is in tests/monotone_cubic_oracle_test.cpp.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"

using swaps::curve::Boundary;
using swaps::curve::MonotoneCubic;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {
const std::vector<double> kKnots{1, 2, 3, 5, 7, 10};  // 6 back knots
constexpr double kT0 = 0.5, kV0 = 0.030, kI0 = 0.015;

MonotoneCubic<double> make(const Eigen::VectorXd& vals, double v0 = kV0) {
  MonotoneCubic<double> mc(kKnots);
  Boundary<double> in{kT0, v0, 0.0, kI0, /*has_predecessor=*/true};
  mc.build(vals, 0, static_cast<int>(vals.size()), in);
  return mc;
}
Eigen::VectorXd vals_monotone() {
  Eigen::VectorXd v(6);
  v << 0.033, 0.036, 0.038, 0.041, 0.043, 0.045;  // strictly increasing
  return v;
}
Eigen::VectorXd vals_wiggly() {
  Eigen::VectorXd v(6);
  v << 0.045, 0.030, 0.048, 0.032, 0.050, 0.031;  // up/down -> the filter must fire
  return v;
}
}  // namespace

TEST(MonotoneCubic, IsNonLinearAndPinsBoundary) {
  // The region policy declares non-linearity at compile time; a curve CONTAINING it must report it at
  // runtime, since that runtime flag is what routes the curve to the AAD tier instead of the W-cache.
  static_assert(!MonotoneCubic<double>::is_linear_map);
  EXPECT_FALSE(cv::make_modular_curve<double>(cv::flat_monotone({kT0}, kKnots)).is_linear_map())
      << "MonotoneCubic must NOT be a linear map -- it belongs on the AAD tier";
  const auto mc = make(vals_monotone());
  EXPECT_NEAR(mc.forward(kT0), kV0, 1e-14) << "clamped start pins the boundary value (C0 join)";
  EXPECT_NEAR(mc.integral(kT0), kI0, 1e-15) << "integral at the join equals the incoming integral";
}

TEST(MonotoneCubic, FilterKillsOvershootVsNaturalCubic) {
  // The filter's purpose, shown against the UNFILTERED base (NaturalCubic) on the same monotone STEP data
  // (flat, then a jump, then flat). An unfiltered C2 spline overshoots the [min,max] envelope Gibbs-style;
  // the Hyman-filtered curve stays exactly within it. Both regions are built identically -- only the
  // tangent filter differs -- so this isolates the filter's effect. (The monotone curve matches QuantLib's
  // MonotonicCubicNaturalSpline; the natural cubic matches QuantLib's plain cubic -- both proven elsewhere.)
  Eigen::VectorXd v(6);
  v << 0.020, 0.020, 0.020, 0.060, 0.060, 0.060;
  swaps::curve::MonotoneCubic<double> mc(kKnots);
  swaps::curve::NaturalCubic<double> nc(kKnots);
  const Boundary<double> in{kT0, 0.020, 0.0, kI0, /*has_predecessor=*/true};  // v0 = first value -> globally non-decreasing data
  mc.build(v, 0, 6, in);
  nc.build(v, 0, 6, in);
  std::vector<double> xs{kT0}, ys{0.020};
  for (int i = 0; i < v.size(); ++i) { xs.push_back(kKnots[i]); ys.push_back(v[i]); }
  auto overshoot = [&](auto& c) {
    double w = 0;
    for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
      const double lo = std::min(ys[i], ys[i + 1]), hi = std::max(ys[i], ys[i + 1]);
      for (int k = 0; k <= 200; ++k) {
        const double f = c.forward(xs[i] + (xs[i + 1] - xs[i]) * k / 200.0);
        w = std::max(w, std::max(lo - f, f - hi));
      }
    }
    return w;
  };
  const double mono = overshoot(mc), nat = overshoot(nc);
  std::cout << "  [monotone] step-data overshoot: monotone=" << mono << " natural-cubic=" << nat << "\n";
  EXPECT_LT(mono, 1e-12) << "Hyman filter => zero overshoot on monotone data";
  EXPECT_GT(nat, 1e-3) << "sanity: the unfiltered natural cubic DOES overshoot this data (else no contrast)";
}

TEST(MonotoneCubic, MonotoneDataGivesMonotoneForward) {
  const auto mc = make(vals_monotone());
  double prev = mc.forward(kT0), worst_dip = 0.0;
  for (double t = kT0; t <= kKnots.back(); t += 0.02) {
    const double f = mc.forward(t);
    worst_dip = std::max(worst_dip, prev - f);
    prev = f;
  }
  EXPECT_LT(worst_dip, 1e-12) << "strictly increasing node forwards => non-decreasing interpolated forward";
}

TEST(MonotoneCubic, IntegralMatchesDenseQuadrature) {
  const auto mc = make(vals_wiggly());
  auto quad = [&](double t) {
    const int N = 200000;
    const double h = (t - kT0) / N;
    double s = 0.5 * (mc.forward(kT0) + mc.forward(t));
    for (int i = 1; i < N; ++i) s += mc.forward(kT0 + i * h);
    return kI0 + s * h;
  };
  double worst = 0;
  for (double t : {1.0, 2.5, 5.0, 8.0, 10.0}) worst = std::max(worst, std::abs(mc.integral(t) - quad(t)));
  std::cout << "  [monotone] max |integral - dense quadrature| = " << worst << "\n";
  EXPECT_LT(worst, 1e-7) << "analytic quartic integral of the per-segment cubic vs a dense reference";
}

// A calibration problem whose curve is the NON-LINEAR MonotoneCubic. Duck-types CalibrationProblem
// (residuals<Scalar>/n_knots/n_residuals), exactly like BSplineProblem, so the generic LM + AAD Jacobian
// drive it unchanged. This is where the ROUTING matters: because the curve is value-dependent, this MUST
// calibrate through AAD, never the W-cache.
namespace {
struct MonotoneCubicProblem {
  cal::CalibrationProblem inst;
  int n_knots() const { return inst.n_knots(); }
  int n_residuals() const { return inst.n_residuals(); }
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = cv::make_modular_curve<Scalar>(cv::flat_monotone(inst.meeting_times, inst.back_times));
    c.set_forwards(x);
    return inst.price_residuals<Scalar>(c);
  }
};
cal::Instrument make_ois(double T) {
  std::vector<double> ends;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) ends.push_back(u);
  ends.push_back(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u : ends) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev});
    prev = u;
  }
  return ins;
}
}  // namespace

TEST(MonotoneCubic, NonLinearProblemRoutesToAadEngine) {
  // THE GUARD, at the type level: a problem whose curve is non-linear resolves to the AAD engine, NOT the
  // compiled W-cache engine (which its integral_weight_matrix static_assert would reject at compile time).
  static_assert(std::is_same_v<cal::residual_engine_t<MonotoneCubicProblem>,
                               cal::AadResidualEngine<MonotoneCubicProblem>>,
                "a non-linear-curve problem MUST route to the AAD engine, not the compiled W-cache");
  SUCCEED();
}

TEST(MonotoneCubic, CalibratesThroughAadAndReprices) {
  MonotoneCubicProblem prob;
  prob.inst.meeting_times = {0.5};
  prob.inst.back_times = {1, 2, 3, 4, 5, 7, 10};  // 7 back + 1 front = 8 free vars
  const std::vector<double> mats{0.5, 1, 2, 3, 4, 5, 7, 10};
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  // Self-consistent market from a KNOWN monotone-cubic curve, so x_true zeroes the residual exactly.
  Eigen::VectorXd xt(8);
  xt << 0.030, 0.033, 0.036, 0.039, 0.041, 0.043, 0.044, 0.046;
  auto ct = cv::make_modular_curve<double>(cv::flat_monotone(prob.inst.meeting_times, prob.inst.back_times));
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-13) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(8, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [monotone-calib] reprice=" << reprice << " ||x*-xt||=" << (res.x - xt).cwiseAbs().maxCoeff()
            << " stat=" << res.stationarity << " iters=" << res.iterations << "\n";
  EXPECT_LT(reprice, 1e-9) << "calibrated MonotoneCubic curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-7) << "first-order optimality through the AAD Jacobian";
}
