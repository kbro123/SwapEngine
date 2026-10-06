// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T3 cross-path parity (AAD vs FD; hybrid vs templated)
// Hagan-West MONOTONE CONVEX (curve/regions.hpp MonotoneConvex, 2026-10-06): the first scheme that is value-dependent and
// NOT piecewise-linear in its knots. No QuantLib oracle exists for it in the vendored stack (QuantLib has no Hagan-West
// interpolator), so the pins are the method's DEFINING properties and a hand computation per shape:
//   (1) every interval integrates to its discrete forward EXACTLY -- the property that defines the method;
//   (2) the forward is continuous across knots and across the shapes' internal breakpoints, and all four shapes occur;
//   (3) the closed-form integral equals a dense quadrature of the forward;
//   (4) inputs of one sign give a forward of that sign (the paper's positivity theorem, mirrored for negative rates);
//   (5) a hand computation of f_i and of g(x) in each of the four shapes;
//   (6) the AAD derivative equals a central finite difference inside every shape;
//   (7) ROUTING: scheme_is_linear / scheme_is_piecewise_linear are false, the piecewise-linear tier is REFUSED (not
//       silently engaged), rows reading the region ride the AAD block, and the hybrid matches the templated kernel;
//   (8) a curve with a MonotoneConvex back end calibrates through AAD and reprices its market.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/curve_handle.hpp"

namespace cv = swaps::curve;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace ad = swaps::ad;
using cv::MonotoneConvex;

namespace {

// A FOLLOWING region over knots 1..m with join (time 0, forward f0, integral 0): the paper's setting (intervals
// (0,1], (1,2], ...) with the left endpoint pinned.
template <class S = double>
MonotoneConvex<S> following(const std::vector<double>& knots, const std::vector<S>& fd, S f0) {
  MonotoneConvex<S> r(knots);
  cv::Boundary<S> in;
  in.time = 0.0; in.value = f0; in.integral = S(0.0); in.has_predecessor = true;
  r.build(fd.data(), 0, static_cast<int>(fd.size()), in);
  return r;
}

std::vector<double> random_fd(std::mt19937_64& g, int m, double lo, double hi) {
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> v(static_cast<std::size_t>(m));
  for (auto& x : v) x = u(g);
  return v;
}

}  // namespace

// ---- (1) the defining property ----------------------------------------------------------------------------------
TEST(MonotoneConvex, EveryIntervalIntegratesToItsDiscreteForwardExactly) {
  std::mt19937_64 g(20261006);
  const std::vector<double> knots{1.0, 2.0, 3.5, 5.0, 7.0, 10.0};
  for (int trial = 0; trial < 200; ++trial) {
    // positive, negative and mixed inputs; a join forward of either sign
    const double lo = (trial % 3 == 0) ? -0.02 : (trial % 3 == 1 ? 0.005 : -0.03);
    const double hi = (trial % 3 == 0) ? 0.06 : (trial % 3 == 1 ? 0.07 : -0.001);
    const auto fd = random_fd(g, 6, lo, hi);
    const double f0 = std::uniform_real_distribution<double>(lo, hi)(g);
    const auto r = following(knots, fd, f0);
    double prev_t = 0.0;
    for (std::size_t i = 0; i < knots.size(); ++i) {
      const double avg = (r.integral(knots[i]) - r.integral(prev_t)) / (knots[i] - prev_t);
      EXPECT_NEAR(avg, fd[i], 1e-14) << "trial " << trial << " interval " << i;
      prev_t = knots[i];
    }
    EXPECT_NEAR(r.forward(0.0), f0, 0.0) << "the join forward is pinned";
    EXPECT_NEAR(r.out().integral, r.integral(knots.back()), 0.0);
  }
}

// ---- (2) continuity, and all four shapes occur ---------------------------------------------------------------------
TEST(MonotoneConvex, ForwardIsContinuousAcrossKnotsAndBreakpointsInEveryShape) {
  std::mt19937_64 g(7);
  const std::vector<double> knots{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::set<int> seen;
  double worst = 0.0;
  for (int trial = 0; trial < 300; ++trial) {
    const auto fd = random_fd(g, 8, -0.02, 0.08);
    const double f0 = std::uniform_real_distribution<double>(-0.02, 0.08)(g);
    const auto r = following(knots, fd, f0);
    for (int i = 1; i <= r.n_intervals(); ++i) seen.insert(r.region(i));
    for (double b : r.pieces()) {
      if (b <= 0.0 || b >= knots.back()) continue;
      const double eps = 1e-9;
      worst = std::max(worst, std::abs(r.forward(b + eps) - r.forward(b - eps)));
    }
  }
  EXPECT_LT(worst, 1e-6) << "the forward must be continuous at every knot and breakpoint (slope is finite)";
  for (int shape : {1, 2, 3, 4}) EXPECT_TRUE(seen.count(shape)) << "random inputs never exercised shape " << shape;
}

// ---- (3) closed-form integral vs dense quadrature ------------------------------------------------------------------
TEST(MonotoneConvex, IntegralMatchesDenseQuadrature) {
  std::mt19937_64 g(11);
  const std::vector<double> knots{0.5, 1.5, 2.0, 4.0, 6.5, 9.0};
  double worst = 0.0;
  for (int trial = 0; trial < 20; ++trial) {
    const auto fd = random_fd(g, 6, -0.01, 0.07);
    const auto r = following(knots, fd, 0.03);
    const int N = 200000;
    const double T = knots.back(), h = T / N;
    double acc = 0.0;  // Simpson on a uniform grid; breakpoints are quadratic kinks, so the error is O(h^2) at worst
    for (int k = 0; k < N; ++k) {
      const double a = k * h, b = a + h;
      acc += h / 6.0 * (r.forward(a) + 4.0 * r.forward(0.5 * (a + b)) + r.forward(b));
      if (k % 1000 == 999) worst = std::max(worst, std::abs(acc - r.integral(b)));
    }
  }
  EXPECT_LT(worst, 1e-9);
}

// ---- (4) sign preservation (the paper's positivity theorem, mirrored) ---------------------------------------------
TEST(MonotoneConvex, InputsOfOneSignGiveAForwardOfThatSign) {
  std::mt19937_64 g(3);
  const std::vector<double> knots{1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
  double min_pos = 1.0, max_neg = -1.0;
  for (int trial = 0; trial < 200; ++trial) {
    const auto fdp = random_fd(g, 7, 0.0005, 0.09);
    const auto rp = following(knots, fdp, fdp.front());  // the join forward at the first discrete forward
    for (double t = 0.0; t <= 15.0; t += 0.01) min_pos = std::min(min_pos, rp.forward(t));
    std::vector<double> fdn(fdp);
    for (auto& v : fdn) v = -v;
    const auto rn = following(knots, fdn, fdn.front());
    for (double t = 0.0; t <= 15.0; t += 0.01) max_neg = std::max(max_neg, rn.forward(t));
  }
  EXPECT_GE(min_pos, 0.0) << "positive discrete forwards give a non-negative forward everywhere";
  EXPECT_LE(max_neg, 0.0) << "the mirrored amelioration keeps negative inputs non-positive";
}

// ---- (5) hand computation per shape --------------------------------------------------------------------------------
TEST(MonotoneConvex, HandComputationOfTheFourShapes) {
  // Three unit intervals, f0 pinned. Interior forwards (equal spacing): f_1 = (fd_2 + fd_1)/2, f_2 = (fd_3 + fd_2)/2,
  // far end f_3 = fd_3 - (f_2 - fd_3)/2, then the amelioration clamp into [0, 2 min].
  const std::vector<double> knots{1.0, 2.0, 3.0};
  {
    // The knot forwards: fd = {0.02, 0.04, 0.03}, f0 = 0.021 -> f1 = (0.04+0.02)/2 = 0.03, f2 = (0.03+0.04)/2 = 0.035,
    // f3 = fd3 - (f2 - fd3)/2 = 0.0275 (each inside its clamp). Interval 1: g0 = 0.001, g1 = 0.01 -> shape (iv).
    // (f0 == fd1 exactly would be the paper's degenerate point: eta = 1, g == 0 on [0, 1) and a jump to g1 at the
    // knot; the forward AT the knot still reads f_1 by construction.)
    const auto r = following(knots, std::vector<double>{0.02, 0.04, 0.03}, 0.021);
    EXPECT_NEAR(r.forward(1.0), 0.03, 1e-15);
    EXPECT_NEAR(r.forward(2.0), 0.035, 1e-15);
    EXPECT_NEAR(r.forward(3.0), 0.0275, 1e-15);
    EXPECT_EQ(r.region(1), 4);
    const auto d = following(knots, std::vector<double>{0.02, 0.04, 0.03}, 0.02);  // the degenerate point itself
    EXPECT_NEAR(d.forward(1.0), 0.03, 1e-15) << "at the knot: f_1";
    EXPECT_NEAR(d.forward(1.0 - 1e-9), 0.02, 1e-12) << "just before it: the paper's shape, fd_1 (the jump is the method's)";
    EXPECT_NEAR(d.integral(1.0), 0.02, 1e-15) << "the interval still integrates to fd_1";
  }
  {
    // Shape (i): need g0 > 0 and -2 g0 <= g1 <= -g0/2. fd = {0.03, 0.004, 0.03}, f0 = 0.05: interval 1: g0 = 0.02,
    // f1 = (0.004+0.03)/2 = 0.017 -> AMELIORATION clamps into [0, 2*min(0.03, 0.004) = 0.008] -> 0.008;
    // g1 = 0.008 - 0.03 = -0.022, inside [-0.04, -0.01] -> (i).
    // g(x) = g0(1-4x+3x^2) + g1(-2x+3x^2); at x = 0.5: 0.02*(-0.25) + (-0.022)*(-0.25) = -0.005 + 0.0055 = 0.0005
    const auto r = following(knots, std::vector<double>{0.03, 0.004, 0.03}, 0.05);
    ASSERT_EQ(r.region(1), 1);
    EXPECT_NEAR(r.forward(1.0), 0.008, 1e-15) << "the amelioration clamp bound f_1 to 2 min(fd_1, fd_2)";
    EXPECT_NEAR(r.forward(0.5), 0.03 + 0.0005, 1e-15);
    EXPECT_NEAR(r.integral(1.0), 0.03, 1e-15) << "G(1) = 0: the interval integrates to fd_1";
  }
  {
    // Shape (ii): g0 < 0 and g1 > -2 g0. fd = {0.04, 0.06, 0.06}, f0 = 0.01: g0 = -0.03; f1 = 0.05 (<= 2 min = 0.08);
    // g1 = 0.05 - 0.04 = 0.01 > 0.06? no. Take f0 = 0.035: g0 = -0.005, g1 = 0.01 > 0.01? equal -> not strict.
    // f0 = 0.036: g0 = -0.004, -2 g0 = 0.008 < g1 = 0.01 -> (ii). eta = (g1 + 2 g0)/(g1 - g0) = 0.002/0.014 = 1/7.
    // g(x) = g0 for x <= eta; at x = 0.5: g0 + (g1 - g0) ((0.5 - 1/7)/(6/7))^2 = -0.004 + 0.014 * (2.5/6)^2
    const auto r = following(knots, std::vector<double>{0.04, 0.06, 0.06}, 0.036);
    ASSERT_EQ(r.region(1), 2);
    EXPECT_NEAR(r.forward(0.1), 0.04 - 0.004, 1e-15) << "flat at g0 before eta";
    EXPECT_NEAR(r.forward(0.5), 0.04 - 0.004 + 0.014 * (2.5 / 6.0) * (2.5 / 6.0), 1e-15);
    EXPECT_NEAR(r.integral(1.0), 0.04, 1e-15);
  }
  {
    // Shape (iii): g0 > 0 and -g0/2 < g1 < 0. fd = {0.05, 0.05, 0.05}, f0 = 0.07: g0 = 0.02; f1 = 0.05 -> g1 = 0: not < 0.
    // fd = {0.05, 0.046, 0.05}, f0 = 0.07: f1 = 0.048, g1 = -0.002, -g0/2 = -0.01 < -0.002 < 0 -> (iii).
    // eta = 3 g1/(g1 - g0) = -0.006/-0.022 = 3/11; g(x) = g1 + (g0 - g1)((eta - x)/eta)^2 for x < eta, g1 after.
    const auto r = following(knots, std::vector<double>{0.05, 0.046, 0.05}, 0.07);
    ASSERT_EQ(r.region(1), 3);
    const double eta = 3.0 / 11.0;
    EXPECT_NEAR(r.forward(0.1), 0.05 - 0.002 + 0.022 * ((eta - 0.1) / eta) * ((eta - 0.1) / eta), 1e-15);
    EXPECT_NEAR(r.forward(0.8), 0.05 - 0.002, 1e-15) << "flat at g1 after eta";
    EXPECT_NEAR(r.integral(1.0), 0.05, 1e-15);
  }
  {
    // Shape (iv) proper (both positive): fd = {0.03, 0.05, 0.05}, f0 = 0.04: g0 = 0.01, f1 = 0.04, g1 = 0.01.
    // eta = 1/2, A = -g0 g1/(g0+g1) = -0.005; g(0.25) = A + (g0 - A)((0.5-0.25)/0.5)^2 = -0.005 + 0.015*0.25 = -0.00125
    const auto r = following(knots, std::vector<double>{0.03, 0.05, 0.05}, 0.04);
    ASSERT_EQ(r.region(1), 4);
    EXPECT_NEAR(r.forward(0.25), 0.03 - 0.00125, 1e-15);
    EXPECT_NEAR(r.forward(0.5), 0.03 - 0.005, 1e-15) << "the shape's minimum A at eta";
    EXPECT_NEAR(r.integral(1.0), 0.03, 1e-15);
  }
}

// ---- (6) AAD vs finite difference inside every shape ---------------------------------------------------------------
TEST(MonotoneConvex, AadDerivativeMatchesCentralDifferenceInEveryShape) {
  std::mt19937_64 g(5);
  const std::vector<double> knots{1.0, 2.0, 3.0, 4.5, 6.0};
  std::set<int> seen;
  double worst = 0.0;
  int checked = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const auto fd = random_fd(g, 5, -0.01, 0.07);
    const double f0 = 0.03;
    // dual build
    const int m = 5;
    std::vector<ad::Dual> xd(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) xd[static_cast<std::size_t>(k)] = ad::Dual(fd[static_cast<std::size_t>(k)], m, k);
    const auto rd = following<ad::Dual>(knots, xd, ad::Dual(f0));
    for (int i = 1; i <= rd.n_intervals(); ++i) seen.insert(rd.region(i));
    const auto rp0 = following(knots, fd, f0);
    const std::vector<double> bps = rp0.pieces();
    for (double t : {0.3, 0.9, 1.4, 2.2, 2.9, 3.6, 4.2, 5.1, 5.8}) {
      // The integral is C1 but not C2 in x where a shape's breakpoint eta crosses t (and across a shape or clamp
      // boundary), so a central difference straddling one is O(h) off a one-sided derivative: keep the generic points.
      bool near = false;
      for (double b : bps) near = near || std::abs(t - b) < 1e-4;
      if (near) continue;
      const ad::Dual I = rd.integral(t);
      for (int k = 0; k < m; ++k) {
        // The shapes' rational coefficients (g0, g1 / A, eta) are smooth but strongly curved near a zero discrete
        // forward, so a plain central difference carries a visible O(h^2) truncation there: Richardson-extrapolate
        // (h, h/2) to cancel it. A genuine kink in x_k (a clamp or shape boundary) is excluded by requiring the two
        // central estimates to agree, which a kink cannot do.
        const auto central = [&](double hh) {
          auto fp = fd, fm = fd;
          fp[static_cast<std::size_t>(k)] += hh; fm[static_cast<std::size_t>(k)] -= hh;
          return (following(knots, fp, f0).integral(t) - following(knots, fm, f0).integral(t)) / (2 * hh);
        };
        const double c1 = central(1e-5), c2 = central(5e-6);
        if (std::abs(c1 - c2) > 1e-5) continue;  // a kink between the samples: the derivative is one-sided there
        const double fdiff = (4.0 * c2 - c1) / 3.0;
        const double a = I.derivatives().size() == m ? I.derivatives()[k] : 0.0;
        worst = std::max(worst, std::abs(a - fdiff));
        ++checked;
      }
    }
  }
  EXPECT_LT(worst, 1e-7) << "forward-AAD integral derivative vs central FD (generic points, inside the shapes)";
  EXPECT_GT(checked, 2000) << "the kink exclusion must not hollow the test out";
  for (int shape : {1, 2, 3, 4}) EXPECT_TRUE(seen.count(shape)) << "shape " << shape << " not exercised";
}

// ---- (7) routing: value-dependent, NOT piecewise-linear, tier refused, AAD rows ------------------------------------
namespace {
cal::Instrument par_swap(double T, int curve) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev;
    c.pay = u; c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev});
    prev = u;
  }
  ins.fwd.forecast = ins.fwd.discount = ins.fixed.discount = curve;
  return ins;
}
cal::BundleProblem convex_bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_monotone_convex({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0})});
  for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) p.instruments.push_back(par_swap(T, 0));
  return p;
}
}  // namespace

TEST(MonotoneConvex, SchemeIsValueDependentAndNotPiecewiseLinear) {
  static_assert(!MonotoneConvex<double>::is_linear_map);
  EXPECT_FALSE(cv::scheme_is_linear(cv::Scheme::MonotoneConvex));
  EXPECT_FALSE(cv::scheme_is_piecewise_linear(cv::Scheme::MonotoneConvex));
  EXPECT_TRUE(cv::scheme_is_piecewise_linear(cv::Scheme::MonotoneCubic)) << "the Hyman filter picks among linear formulas";
  const cal::BundleProblem p = convex_bundle();
  EXPECT_FALSE(cv::make_modular_curve<double>(p.curves[0].modules()).is_linear_map());
  EXPECT_FALSE(px::curves_piecewise_linear(p.curves));
  EXPECT_DOUBLE_EQ(px::curve_linear_horizon(p.curves, 0), 0.5) << "the flat front is the linear prefix";
  // The tier itself refuses the curve set (never a silently frozen W).
  px::CompiledCurveSet cs;
  cs.init(p.curves);
  EXPECT_THROW(cs.enable_pwl(), std::invalid_argument);
  EXPECT_THROW(px::integral_weight_matrix_at(p.curves[0].modules(), {3.0}, Eigen::VectorXd::Constant(8, 0.03)), std::invalid_argument);
}

TEST(MonotoneConvex, RouterSendsRowsReadingTheRegionToAadAndMatchesTheKernel) {
  cal::BundleProblem p = convex_bundle();
  Eigen::VectorXd x(8);
  x << 0.030, 0.031, 0.033, 0.036, 0.038, 0.040, 0.041, 0.042;  // front levels, then the back's DISCRETE forwards
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return x[i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of) + 2e-4;
  const cal::HybridBundleResidual eng(p);  // default routing
  EXPECT_FALSE(eng.pwl_active()) << "the piecewise-linear tier must be refused, not engaged";
  for (int r = 0; r < p.n_residuals(); ++r) EXPECT_EQ(eng.compiled_row(r), -1) << "row " << r << " reads past the 0.5y horizon";
  EXPECT_TRUE(eng.aad_pooled());
  EXPECT_LT((eng.residuals(x) - p.residuals<double>(x)).cwiseAbs().maxCoeff(), 1e-14);
  const Eigen::MatrixXd Jh = eng.jacobian(x), Ja = cal::aad_jacobian(p, x);
  EXPECT_LT((Jh - Ja).cwiseAbs().maxCoeff(), 1e-12 * Ja.cwiseAbs().maxCoeff());
  // The streaming forms exist on this engine too (the AAD block's _vs forms).
  Eigen::VectorXd q = p.market();
  for (int i = 0; i < q.size(); ++i) q[i] += 1e-4;
  Eigen::MatrixXd J; Eigen::VectorXd rr;
  eng.jacobian_vs_into(x, q, J, &rr);
  EXPECT_LT((rr - eng.residuals_vs(x, q)).cwiseAbs().maxCoeff(), 1e-14);
}

// ---- (8) calibrates through AAD and reprices -----------------------------------------------------------------------
TEST(MonotoneConvex, CalibratesThroughAadAndReprices) {
  cal::BundleProblem p = convex_bundle();
  Eigen::VectorXd xt(8);
  xt << 0.030, 0.0305, 0.032, 0.035, 0.038, 0.040, 0.0415, 0.043;
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return xt[i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  static_assert(std::is_same_v<cal::residual_engine_t<cal::BundleProblem>, cal::HybridBundleResidual>);
  const auto res = cal::calibrate(p, Eigen::VectorXd::Constant(8, 0.03));
  EXPECT_LT(p.residuals<double>(res.x).cwiseAbs().maxCoeff(), 1e-9) << "a square market reprices through the AAD tier";
  EXPECT_LT(res.stationarity, 1e-7);
  // The calibrated curve's discrete forwards on the back intervals are recoverable from its own integral.
  const auto Cc = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return res.x[i]; });
  const std::vector<double> back{1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
  double prev = 0.5;
  for (std::size_t i = 0; i < back.size(); ++i) {
    const double avg = (Cc[0]->integral(back[i]) - Cc[0]->integral(prev)) / (back[i] - prev);
    EXPECT_NEAR(avg, res.x[2 + static_cast<int>(i)], 1e-12) << "interval " << i;
    prev = back[i];
  }
}
