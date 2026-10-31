// E5 taxonomy: T3 cross-path parity (the closed-form curve tangent against the AAD gradient of the same curve) | T5 properties (a central difference inside every shape)
// THE CURVE TANGENT (2026-10-31, ASSUMPTIONS K52): ModularCurve::integral_tangent is d integral(t)/dx at the current state,
// chained region by region -- a linear region's constant row (one dual pass over the region alone, taken once), a
// MonotoneConvex region's closed form (the four shapes, eta and A differentiated through, the amelioration clamps, the
// join), composed front to back through the C0 handoff. Pinned here against the AAD gradient of the SAME curve (every
// layout, every shape and clamp, random states, both signs of rates) to 1e-13, and against a Richardson central
// difference inside the shapes; and integral_tangent_matrix against integral_weight_matrix on a linear curve.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/compiled.hpp"

namespace cv = swaps::curve;
namespace ad = swaps::ad;

namespace {
struct Layout { const char* name; std::vector<cv::CurveModule> modules; };
std::vector<Layout> layouts() {
  return {
      {"monoconvex leading", {cv::CurveModule{{1.0, 2.0, 3.0, 5.0, 7.0, 10.0}, cv::Scheme::MonotoneConvex}}},
      {"flat -> monoconvex (the desk RFR)", cv::flat_monotone_convex({0.25, 0.5, 0.8}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0})},
      {"monoconvex -> hermite", {cv::CurveModule{{1.0, 2.0, 3.0}, cv::Scheme::MonotoneConvex}, cv::CurveModule{{5.0, 7.0, 10.0}, cv::Scheme::Hermite}}},
      {"hermite -> monoconvex -> linear", {cv::CurveModule{{0.5, 1.0}, cv::Scheme::Hermite}, cv::CurveModule{{2.0, 3.0, 5.0}, cv::Scheme::MonotoneConvex}, cv::CurveModule{{7.0, 10.0}, cv::Scheme::Linear}}},
      {"flat -> monoconvex -> tension", {cv::CurveModule{{0.3}, cv::Scheme::Flat}, cv::CurveModule{{1.0, 2.0, 4.0}, cv::Scheme::MonotoneConvex}, cv::CurveModule{{6.0, 8.0, 10.0}, cv::Scheme::Tension, 2.0}}},
      {"flat -> hermite (linear only)", cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0, 10.0})},
      {"natural cubic -> monoconvex", {cv::CurveModule{{0.5, 1.0, 1.5}, cv::Scheme::NaturalCubic}, cv::CurveModule{{3.0, 5.0, 7.0}, cv::Scheme::MonotoneConvex}}},
  };
}
// Random states: mostly positive rates with a few near-zero and negative knots (the amelioration's branches: a clamp to
// the zero bound, to twice a neighbour, the mirrored clamp), and an occasional exactly-equal pair.
Eigen::VectorXd random_state(std::mt19937_64& g, int n, int trial) {
  std::uniform_real_distribution<double> u(-0.01, 0.07);
  Eigen::VectorXd x(n);
  for (int k = 0; k < n; ++k) x[k] = u(g);
  if (trial % 5 == 1) for (int k = 0; k < n; ++k) x[k] = -std::abs(x[k]);  // all negative: the mirrored clamp
  if (trial % 7 == 2 && n > 2) x[1] = 0.0005;                             // a near-zero knot: clamp to 2 min
  if (trial % 11 == 3 && n > 3) x[2] = x[3];                              // an equal pair
  return x;
}
}  // namespace

TEST(CurveTangent, MatchesTheAadGradientOnEveryLayoutShapeAndClamp) {
  std::mt19937_64 g(11);
  int checked = 0;
  for (const Layout& L : layouts()) {
    auto c = cv::make_modular_curve<double>(L.modules);
    auto cd = cv::make_modular_curve<ad::Dual>(L.modules);
    ASSERT_TRUE(c.has_tangent()) << L.name;
    const int n = c.n_knots();
    std::vector<double> row(static_cast<std::size_t>(n));
    double worst = 0.0;
    for (int trial = 0; trial < 40; ++trial) {
      const Eigen::VectorXd x = random_state(g, n, trial);
      c.set_forwards(x);
      cd.set_forwards(ad::seed(x));
      for (double t = 0.05; t <= c.max_time() + 3.0; t += 0.137) {
        c.integral_tangent(t, row.data());
        const ad::Dual I = cd.integral(t);
        for (int k = 0; k < n; ++k) {
          const double a = I.derivatives().size() == n ? I.derivatives()[k] : 0.0;
          const double d = std::abs(row[static_cast<std::size_t>(k)] - a);
          worst = std::max(worst, d / (1.0 + std::abs(a)));
          EXPECT_NEAR(row[static_cast<std::size_t>(k)], a, 1e-13 * (1.0 + std::abs(a))) << L.name << " trial " << trial << " t " << t << " knot " << k;
          ++checked;
        }
      }
      // integral(t <= 0) is 0 by convention: a zero row
      c.integral_tangent(0.0, row.data());
      for (int k = 0; k < n; ++k) EXPECT_EQ(row[static_cast<std::size_t>(k)], 0.0) << L.name;
    }
    std::printf("  [%-36s] worst |tangent - AAD| / (1 + |AAD|) = %.2e\n", L.name, worst);
  }
  EXPECT_GT(checked, 50000);
}

TEST(CurveTangent, EveryShapeAndClampIsExercised) {
  // The MonotoneConvex region reports the shape each interval took; the random states must reach all four, and both
  // clamp kinds must fire (seen through the tangent: a clamped node's gradient is 0 or 2 e_neighbour).
  std::mt19937_64 g(3);
  cv::MonotoneConvex<double> r({1.0, 2.0, 3.0, 4.5, 6.0, 8.0});
  std::set<int> shapes;
  int clamped_zero = 0, clamped_twice = 0;
  std::vector<double> own(6);
  for (int trial = 0; trial < 400; ++trial) {
    const Eigen::VectorXd x = random_state(g, 6, trial);
    std::vector<double> xs(x.data(), x.data() + 6);
    cv::Boundary<double> in;
    in.time = 0.0; in.value = 0.03; in.integral = 0.0; in.has_predecessor = true;
    r.build(xs.data(), 0, 6, in);
    for (int i = 1; i <= r.n_intervals(); ++i) shapes.insert(r.region(i));
    double d_inv = 0.0;
    r.out_value_tangent(own.data(), d_inv);  // f_m's gradient: 1.5 e_m - 0.5 df_{m-1} unclamped, 0 or 2 e_m clamped
    double sum = d_inv;
    for (double v : own) sum += v;
    if (sum == 0.0) ++clamped_zero;
    else if (own[5] == 2.0) ++clamped_twice;
  }
  for (int sh : {1, 2, 3, 4}) EXPECT_TRUE(shapes.count(sh)) << "shape " << sh;
  EXPECT_GT(clamped_zero, 0);
  EXPECT_GT(clamped_twice, 0);
}

// Generic states only: the engineered ties of random_state (two equal neighbours) put the amelioration's min at a kink
// ON the point, where a symmetric difference averages the two branches while the tangent (like AAD) takes one -- that
// one-sidedness is pinned against AAD above; a central difference can only check the smooth interior.
TEST(CurveTangent, MatchesACentralDifferenceInsideTheShapes) {
  std::mt19937_64 g(5);
  const std::vector<cv::CurveModule> mods = cv::flat_monotone_convex({0.25, 0.5}, {1.0, 2.0, 3.0, 4.5, 6.0});
  auto c = cv::make_modular_curve<double>(mods);
  const int n = c.n_knots();
  std::vector<double> row(static_cast<std::size_t>(n));
  double worst = 0.0;
  int checked = 0;
  std::uniform_real_distribution<double> u(-0.01, 0.07);
  for (int trial = 0; trial < 60; ++trial) {
    Eigen::VectorXd x(n);
    for (int k = 0; k < n; ++k) x[k] = u(g);
    c.set_forwards(x);
    const std::vector<double> bps = c.pieces();
    for (double t : {0.3, 0.9, 1.4, 2.2, 2.9, 3.6, 4.2, 5.1, 5.8, 7.0}) {
      bool near = false;
      for (double b : bps) near = near || std::abs(t - b) < 1e-4;
      if (near) continue;
      c.integral_tangent(t, row.data());
      for (int k = 0; k < n; ++k) {
        const auto central = [&](double hh) {
          Eigen::VectorXd xp = x, xm = x;
          xp[k] += hh; xm[k] -= hh;
          auto cp = cv::make_modular_curve<double>(mods); cp.set_forwards(xp);
          auto cm = cv::make_modular_curve<double>(mods); cm.set_forwards(xm);
          return (cp.integral(t) - cm.integral(t)) / (2 * hh);
        };
        const double c1 = central(1e-5), c2 = central(5e-6);
        if (std::abs(c1 - c2) > 1e-5) continue;  // a kink between the samples: one-sided there
        const double fd = (4.0 * c2 - c1) / 3.0;
        worst = std::max(worst, std::abs(row[static_cast<std::size_t>(k)] - fd));
        ++checked;
      }
    }
  }
  EXPECT_LT(worst, 1e-7);
  EXPECT_GT(checked, 2000);
}

TEST(CurveTangent, TheTangentMatrixOfALinearCurveIsItsW) {
  const std::vector<cv::CurveModule> mods = cv::flat_hermite({0.25, 0.5, 0.75}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0});
  std::vector<double> times;
  for (double t = 0.1; t <= 12.0; t += 0.3) times.push_back(t);
  const Eigen::MatrixXd W = swaps::pricing::integral_weight_matrix(mods, times);
  const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(9, 0.02, 0.05);
  const Eigen::MatrixXd T = swaps::pricing::integral_tangent_matrix(mods, times, x);
  EXPECT_LT((W - T).cwiseAbs().maxCoeff(), 1e-14);
  // and a curve with a region without a tangent is refused
  const std::vector<cv::CurveModule> mc = cv::flat_monotone({0.25}, {1.0, 2.0, 3.0});
  EXPECT_THROW(swaps::pricing::integral_tangent_matrix(mc, times, Eigen::VectorXd::Constant(4, 0.03)), std::invalid_argument);
}
