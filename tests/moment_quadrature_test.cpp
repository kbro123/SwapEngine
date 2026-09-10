// The moment path's quadrature is KNOT-ALIGNED (E3 register R4 / G2, fixed 2026-09-10). Until then a fixed
// 32-panel x 2-pt rule integrated f^2 across the curve's breakpoints; on the shipped Flat-front Fed-funds shape
// (25 bp policy steps inside a 1Y window) it lost 3.3e-3 of the integral -- 1.7e-7 of the averaged rate --
// while the docs claimed ~5e-11. Now both kernels split every window at the curve's pieces (region knots, de
// Boor breakpoints, turn edges) and use 4-pt / 5-pt Gauss per piece panel: exact for cubic pieces.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cv = swaps::curve;
namespace px = swaps::pricing;
namespace cal = swaps::calibration;

namespace {
// An INDEPENDENT reference: composite 3-pt Gauss with 1000 panels per piece (a different order and a different
// panelisation from the kernels' 4/5-pt x 2; interior nodes only, so a Flat step or a turn edge at a piece
// boundary is never sampled on the wrong side). Exact to degree 5 per panel; the f^2 (degree 6) and f^3
// (degree 9) residuals are ~h^6 ~ 1e-18 relative.
template <class F>
double simpson_pieces(F f, double a, double b, std::vector<double> brk) {
  static const double gx[3] = {-0.7745966692414834, 0.0, 0.7745966692414834}, gw[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  std::vector<double> pts{a};
  std::sort(brk.begin(), brk.end());
  for (double k : brk) if (k > a + 1e-12 && k < b - 1e-12) pts.push_back(k);
  pts.push_back(b);
  double s = 0.0;
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    const int N = 1000;
    const double h = (pts[i + 1] - pts[i]) / N;
    for (int k = 0; k < N; ++k) {
      const double mid = pts[i] + (k + 0.5) * h, r = 0.5 * h;
      for (int q = 0; q < 3; ++q) s += gw[q] * r * f(mid + gx[q] * r);
    }
  }
  return s;
}
struct Shape { const char* name; std::vector<cv::CurveModule> mods; Eigen::VectorXd x; };
std::vector<Shape> shapes() {
  std::vector<Shape> out;
  {  // the shipped Fed-funds shape: Flat meeting-date front + Hermite back, 25 bp steps INSIDE the windows
    Eigen::VectorXd x(10); x << 0.0430, 0.0405, 0.0380, 0.0355, 0.0330, 0.033, 0.034, 0.036, 0.037, 0.038;
    out.push_back({"Flat front + Hermite (steps)", cv::flat_hermite({0.12, 0.37, 0.62, 0.87, 1.12}, {2, 3, 5, 7, 10}), x});
  }
  {  // smooth Hermite back only
    Eigen::VectorXd x(8); for (int i = 0; i < 8; ++i) x[i] = 0.03 + 0.012 * std::sin(1.7 * i) + 0.0004 * i;
    out.push_back({"Hermite (smooth)", {{{0.25, 0.5, 1, 2, 3, 5, 7, 10}, cv::Scheme::Hermite}}, x});
  }
  {  // Flat front + B-spline back (de Boor breakpoints are NOT the knots)
    Eigen::VectorXd x(10); x << 0.0430, 0.0405, 0.0380, 0.0355, 0.034, 0.034, 0.036, 0.037, 0.038, 0.039;
    out.push_back({"Flat front + BSpline (steps)", cv::flat_bspline({0.12, 0.37, 0.62, 0.87}, {1, 2, 3, 5, 7, 10}), x});
  }
  return out;
}
}  // namespace

TEST(MomentQuadrature, IntegralsAreExactOnEveryPieceOfEveryScheme) {
  for (const auto& s : shapes()) {
    auto c = cv::make_modular_curve<double>(s.mods);
    c.set_forwards(s.x);
    const std::vector<double> brk = c.pieces();
    for (auto win : {std::pair{0.02, 1.02}, std::pair{0.02, 0.27}, std::pair{0.30, 0.55}, std::pair{1.5, 2.5}, std::pair{2.0, 4.0}}) {
      const double a = win.first, b = win.second;
      const double q2 = px::curve_forward_sq_integral<double>(c, a, b), q3 = px::curve_forward_cube_integral<double>(c, a, b);
      const double r2 = simpson_pieces([&](double t) { const double f = c.forward(t); return f * f; }, a, b, brk);
      const double r3 = simpson_pieces([&](double t) { const double f = c.forward(t); return f * f * f; }, a, b, brk);
      EXPECT_NEAR(q2, r2, 1e-12 * r2) << s.name << " [" << a << "," << b << "] f^2";
      EXPECT_NEAR(q3, r3, 1e-11 * r3) << s.name << " [" << a << "," << b << "] f^3";
    }
  }
}

// The moment-averaged rate vs the EXACT daily arithmetic sum on uniform days, on the stepped shape: the
// residual is now the 2-moment truncation alone (~1e-10), not the quadrature (was 1.7e-7 relative).
TEST(MomentQuadrature, AveragedRateMatchesTheExactDailySumAcrossPolicySteps) {
  for (const auto& s : shapes()) {
    auto c = cv::make_modular_curve<double>(s.mods);
    c.set_forwards(s.x);
    double worst = 0.0;
    for (auto win : {std::pair{0.02, 1.02}, std::pair{0.02, 0.27}, std::pair{0.30, 0.55}}) {
      const double a = win.first, b = win.second, T = b - a;
      const int nd = std::max(1, static_cast<int>(std::round(T * 360)));
      const double step = T / nd;
      double exact_num = 0.0;
      for (int d = 0; d < nd; ++d) exact_num += c.discount(a + d * step) / c.discount(a + (d + 1) * step) - 1.0;
      const double exact = exact_num / T;
      const double moment = px::moment_average_rate<double>(c, a, b, step, T, step * step);
      worst = std::max(worst, std::abs(moment - exact) / exact);
    }
    std::cout << "  [moment] " << s.name << ": max rel |moment - exact daily| = " << worst << "\n";
    EXPECT_LT(worst, 1e-9) << s.name;  // measured 4e-10: the 2-moment truncation on a 1Y stepped window (was 1.7e-7)
  }
}

// The compiled batch (quadratic form Q + cubic node rows) integrates the SAME nodes as the templated path:
// a moment-path future straddling policy steps and a turn edge prices identically on both kernels.
TEST(MomentQuadrature, CompiledAndTemplatedKernelsAgreeAcrossStepsAndTurns) {
  cal::BundleProblem p;
  p.curves = {px::CurveStructure{.base = -1, .regions = cv::flat_hermite({0.12, 0.37, 0.62, 0.87, 1.12}, {2, 3, 5, 7, 10})}};
  p.curves[0].turns = {px::Turn{0.95, 1.05}};  // a year-end turn inside the 1Y window
  for (double T : {2.0, 3.0, 5.0, 7.0, 10.0}) {
    cal::Instrument in; in.quote = cal::QuoteKind::ParRate;
    double prev = 0.0;
    for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
      px::FloatCoupon f; f.obs.sub_start = {prev}; f.obs.sub_end = {u}; f.obs.tau_index = u - prev; f.pay = u; f.tau_pay = u - prev;
      in.fwd.coupons.push_back(f); in.fixed.coupons.push_back({u, u - prev, 1.0}); prev = u;
    }
    p.instruments.push_back(in);
  }
  for (auto win : {std::pair{0.02, 1.02}, std::pair{0.30, 0.55}, std::pair{0.90, 1.10}}) {
    cal::Instrument fut; fut.quote = cal::QuoteKind::Rate; fut.forecast = 0;
    fut.obs.sub_start = {win.first}; fut.obs.sub_end = {win.second}; fut.obs.tau_index = win.second - win.first;
    fut.obs.fixing_step = 1.0 / 360.0; fut.obs.fixing_step3 = 1.0 / (360.0 * 360.0);
    p.instruments.push_back(fut);
  }
  Eigen::VectorXd x(p.n_knots());
  x << 0.0430, 0.0405, 0.0380, 0.0355, 0.0330, 0.033, 0.034, 0.036, 0.037, 0.038, 0.0025;  // last = the turn δ
  const Eigen::VectorXd r_templ = p.residuals<double>(x);
  const cal::CompiledBundleResidual eng(p);
  const Eigen::VectorXd r_comp = eng.residuals(x);
  for (int i = 5; i < p.n_residuals(); ++i)
    EXPECT_NEAR(r_comp[i], r_templ[i], 1e-14) << "moment row " << i;
  // and the compiled Jacobian rows match a central finite difference of the templated residual
  const Eigen::MatrixXd J = eng.jacobian(x);
  for (int i = 5; i < p.n_residuals(); ++i)
    for (int j = 0; j < p.n_knots(); ++j) {
      Eigen::VectorXd xp = x, xm = x; xp[j] += 1e-6; xm[j] -= 1e-6;
      const double fd = (p.residuals<double>(xp)[i] - p.residuals<double>(xm)[i]) / 2e-6;
      EXPECT_NEAR(J(i, j), fd, 1e-8) << "row " << i << " col " << j;
    }
}
