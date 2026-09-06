// Curve-space GAMMA gate (reverse-mode AAD tape, ad/reverse.hpp + calibration/gamma.hpp).
//
// QuantLib-FREE: the book and the calibration curve are hand-built from plain FloatCoupon/FixedCoupon data
// (no ql/extract), priced through the REAL templated portfolio::Portfolio / ModularCurve kernels. Two gates:
//   1. GRADIENT PARITY -- the reverse tape's d(NPV)/dx must equal the forward-AAD (ad::Dual) gradient
//      entry-for-entry to 1e-10 (same kernel, two AAD modes).
//   2. GAMMA vs FINITE DIFFERENCE -- the analytic Hessian d^2(NPV)/dx^2 must match a central finite
//      difference of the forward-AAD gradient (bump each knot, difference the analytic gradient) to a
//      sensible FD-limited tolerance. The book is deliberately OFF-PAR so gamma is materially non-zero.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/gamma.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/risk.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace pr = swaps::pricing;
namespace pf_ns = swaps::portfolio;

namespace {

// A vanilla OIS-style swap on the period grid `t` (t[0]=start ... t[K]=maturity): one FloatCoupon and one
// FixedCoupon per period, self-discounted. Float coupon k telescopes to DF(t[k-1])-DF(t[k]); the fixed leg
// is the annuity. Plain single-sub-period coupons (they hit cashflows.hpp's bit-exact fast path).
struct Legs {
  std::vector<pr::FloatCoupon> flt;
  std::vector<pr::FixedCoupon> fix;
};
Legs make_swap(const std::vector<double>& t) {
  Legs L;
  for (std::size_t k = 1; k < t.size(); ++k) {
    const double a = t[k - 1], b = t[k], tau = b - a;
    pr::FloatCoupon fc;
    fc.obs.sub_start = {a};
    fc.obs.sub_end = {b};
    fc.obs.tau_index = tau;
    fc.pay = b;
    fc.tau_pay = tau;
    L.flt.push_back(fc);
    pr::FixedCoupon xc;
    xc.pay = b;
    xc.tau = tau;
    L.fix.push_back(xc);
  }
  return L;
}

// A small, identifiable single-curve problem: 2 flat-front knots + 4 Hermite-back knots (6 total), with 6
// ParRate swaps whose maturities pin every knot so the calibration Jacobian J is full column rank (needed
// only by the market-space transport). The market quotes are irrelevant to gamma and set arbitrarily.
cal::CalibrationProblem build_problem() {
  cal::CalibrationProblem p;
  p.meeting_times = {0.5, 1.0};
  p.back_times = {2.0, 3.0, 5.0, 7.0};
  const std::vector<std::vector<double>> grids = {
      {0.0, 0.5},
      {0.0, 0.5, 1.0},
      {0.0, 1.0, 2.0},
      {0.0, 1.0, 2.0, 3.0},
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}};
  for (const auto& g : grids) {
    Legs L = make_swap(g);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd.coupons = L.flt;
    ins.fixed.coupons = L.fix;
    ins.market = 0.03;  // unused by gamma / by J
    p.instruments.push_back(ins);
  }
  return p;
}

// An OFF-PAR book: 4 swaps at fixed rates well away from the ~3% curve, alternating pay/receive and varied
// notional -> genuinely non-linear NPV(x), so the Hessian is non-trivial (not trivially zero).
pf_ns::Portfolio build_book() {
  pf_ns::Portfolio pf;
  const std::vector<std::vector<double>> grids = {
      {0.0, 1.0, 2.0},
      {0.0, 1.0, 2.0, 3.0},
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}};
  const double rates[4] = {0.055, 0.010, 0.050, 0.015};   // far off the ~3% curve -> real curvature
  const double notl[4] = {1.0, -2.0, 3.0, -1.5};
  for (int i = 0; i < 4; ++i) {
    Legs L = make_swap(grids[i]);
    pf.positions.push_back({L.flt, L.fix, rates[i], notl[i]});
  }
  return pf;
}

// Forward-AAD (ad::Dual) gradient d(NPV)/dx at an arbitrary x -- the parity/FD reference.
Eigen::VectorXd forward_grad(const cal::CalibrationProblem& prob, const Eigen::VectorXd& x,
                             const pf_ns::Portfolio& pf) {
  return cal::book_curve_grad<swaps::ad::Dual>(prob, x, pf, swaps::ad::seed(x));
}

Eigen::VectorXd base_forwards(int n) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x[i] = 0.030 + 0.002 * i;  // a mild upward-sloping forward curve
  return x;
}

}  // namespace

TEST(Gamma, ReverseGradientMatchesForwardAAD) {
  const cal::CalibrationProblem prob = build_problem();
  const pf_ns::Portfolio pf = build_book();
  const Eigen::VectorXd x = base_forwards(prob.n_knots());

  const Eigen::VectorXd g_rev = cal::curve_gradient(prob, x, pf);
  const Eigen::VectorXd g_fwd = forward_grad(prob, x, pf);

  ASSERT_EQ(g_rev.size(), g_fwd.size());
  const double worst = (g_rev - g_fwd).cwiseAbs().maxCoeff();
  EXPECT_LT(worst, 1e-10) << "reverse gradient:\n" << g_rev.transpose() << "\nforward gradient:\n"
                          << g_fwd.transpose();
}

TEST(Gamma, CurveGammaMatchesFiniteDifferenceOfGradient) {
  const cal::CalibrationProblem prob = build_problem();
  const pf_ns::Portfolio pf = build_book();
  const int n = prob.n_knots();
  const Eigen::VectorXd x = base_forwards(n);

  const Eigen::MatrixXd H = cal::curve_gamma(prob, x, pf);
  ASSERT_EQ(H.rows(), n);
  ASSERT_EQ(H.cols(), n);

  // Central FD of the analytic (forward-AAD) gradient: column k = (g(x+eps e_k) - g(x-eps e_k)) / 2eps.
  const double eps = 1e-5;
  Eigen::MatrixXd Hfd(n, n);
  for (int k = 0; k < n; ++k) {
    Eigen::VectorXd xp = x, xm = x;
    xp[k] += eps;
    xm[k] -= eps;
    Hfd.col(k) = (forward_grad(prob, xp, pf) - forward_grad(prob, xm, pf)) / (2 * eps);
  }

  // The book has real curvature: gamma must not be trivially zero.
  const double scale = H.cwiseAbs().maxCoeff();
  EXPECT_GT(scale, 1e-2) << "off-par book should have material gamma; scale=" << scale;

  // FD-limited relative tolerance.
  const double worst_abs = (H - Hfd).cwiseAbs().maxCoeff();
  EXPECT_LT(worst_abs, 1e-4 * scale) << "H=\n" << H << "\nHfd=\n" << Hfd;

  // The Hessian is symmetric.
  EXPECT_LT((H - H.transpose()).cwiseAbs().maxCoeff(), 1e-10 * scale);
}

TEST(Gamma, MarketGammaGaussNewtonTransportIsConsistentAndSymmetric) {
  const cal::CalibrationProblem prob = build_problem();
  const pf_ns::Portfolio pf = build_book();
  const int n = prob.n_knots();
  const int m = prob.n_residuals();
  const Eigen::VectorXd x = base_forwards(n);

  const Eigen::MatrixXd Hq = cal::market_gamma_gn(prob, x, pf);
  ASSERT_EQ(Hq.rows(), m);
  ASSERT_EQ(Hq.cols(), m);

  // market_gamma_gn == S^T H_x S with S = dx/dq (the IFT sensitivity), computed independently here.
  const Eigen::MatrixXd Hx = cal::curve_gamma(prob, x, pf);
  const Eigen::MatrixXd S = cal::ift_quote_sensitivity(prob, x);
  const Eigen::MatrixXd expect = S.transpose() * Hx * S;
  EXPECT_LT((Hq - expect).cwiseAbs().maxCoeff(), 1e-9 * (1.0 + Hq.cwiseAbs().maxCoeff()));

  // Symmetric by construction.
  EXPECT_LT((Hq - Hq.transpose()).cwiseAbs().maxCoeff(), 1e-9 * (1.0 + Hq.cwiseAbs().maxCoeff()));

  // Transport semantics: under the GN model x(q) = x* + S (q - q*), NPV(q) has Hessian S^T H_x S. Validate a
  // couple of diagonal entries against a central FD of NPV along the corresponding curve-space direction
  // s_j = S e_j (a unit quote bump's effect on x). d^2 NPV / dq_j^2 = s_j^T H_x s_j.
  auto npv_at = [&](const Eigen::VectorXd& xv) {
    auto c = swaps::curve::make_modular_curve<double>(
        swaps::curve::flat_hermite(prob.meeting_times, prob.back_times));
    c.set_forwards(xv);
    return pf.npv<double>(c);
  };
  const double dq = 1e-4;
  for (int j = 0; j < m; ++j) {
    const Eigen::VectorXd sj = S.col(j);
    const double f0 = npv_at(x);
    const double fp = npv_at(x + dq * sj);
    const double fm = npv_at(x - dq * sj);
    const double fd = (fp - 2 * f0 + fm) / (dq * dq);
    const double an = sj.transpose() * Hx * sj;  // == Hq(j,j)
    const double sc = 1.0 + std::abs(an);
    EXPECT_LT(std::abs(fd - an), 1e-3 * sc) << "quote " << j << " fd=" << fd << " analytic=" << an;
  }
}
