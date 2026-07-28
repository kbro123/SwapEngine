// Gates for the TENSION-ENERGY regularizer (swaps::calibration::tension_energy_operator, research note
// §5). These are the QuantLib-FREE gates -- pure operator math on hand-built curves:
//   (A1) EXACTNESS: an AFFINE forward has zero bending energy and INT(f')^2 = slope^2 * T -- the operator
//        must reproduce both to machine precision (validates the closed-form K1/K2 assembly).
//   (A2) APPROXIMATES THE CURVATURE PENALTY: on a smooth (Hermite) curve, x^T K2 x from the operator
//        matches INT(f'')^2 of the actually-built forward (independent fine finite-difference reference).
//   (A3) sigma DECOMPOSITION: x^T(K2+sigma^2 K1)x = x^T K2 x + sigma^2 x^T K1 x exactly, so sigma->0 is the
//        pure curvature penalty and sigma just scales the membrane term (the note's continuous knob).
// The conditioning-drop and first-order-optimality gates need a realistic ill-conditioned bundle and live
// in tension_regularizer_oracle_test.cpp (QuantLib-linked).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace curve = swaps::curve;

namespace {

// A single-curve BundleProblem carrying exactly one interpolation region (no instruments needed -- the
// operator only reads prob.curves). `scheme`'s knots are `knots`.
cal::BundleProblem one_region_problem(curve::Scheme scheme, const std::vector<double>& knots) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{knots, scheme});
  p.curves.push_back(spec);
  return p;
}

// x^T (R^T R) x = ||R x||^2 -- the energy the operator penalises.
double energy(const Eigen::MatrixXd& R, const Eigen::VectorXd& x) { return (R * x).squaredNorm(); }

}  // namespace

// (A1) A Linear region whose knot forwards lie on the straight line f(t) = beta*t (through the origin,
// which the C0 pin at t=0 makes exact) is a perfectly AFFINE forward: f'' == 0 and f' == beta. So the
// bending operator must annihilate it and the tension operator's membrane term must return beta^2 * T.
TEST(TensionRegularizer, AffineForwardHasZeroBendingEnergyAndExactMembrane) {
  const std::vector<double> knots{1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::Linear, knots);
  const double beta = 0.013;
  Eigen::VectorXd x(knots.size());
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = beta * knots[i];  // on the line f = beta*t

  // sigma = 0 -> pure bending energy INT(f'')^2, which is exactly zero for an affine forward.
  const Eigen::MatrixXd R_bend = cal::tension_energy_operator(p, /*weight=*/1.0, /*sigma=*/0.0, {0});
  EXPECT_LT(energy(R_bend, x), 1e-18) << "affine forward carries no bending energy";

  // With sigma = 1 the added membrane term is INT(f')^2 = beta^2 * t_last (bending still 0).
  const Eigen::MatrixXd R_full = cal::tension_energy_operator(p, /*weight=*/1.0, /*sigma=*/1.0, {0});
  const double expect = beta * beta * knots.back();
  EXPECT_NEAR(energy(R_full, x), expect, 1e-12 * expect) << "membrane energy = beta^2 * T (closed form)";
}

// (A2) On a genuinely curved (Hermite cubic) forward the operator's bending energy x^T K2 x must equal
// INT(f'')^2 of the built curve. Reference: a fine finite-difference quadrature of the realised forward,
// independent of the operator's closed form. (Hermite is only C1, so f'' jumps at knots; the composite
// rule below skips a small neighbourhood of each knot, where the jump is a measure-zero set.)
TEST(TensionRegularizer, BendingEnergyMatchesBuiltCurveCurvature) {
  const std::vector<double> knots{0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::Hermite, knots);
  Eigen::VectorXd x(knots.size());
  // A deliberately wiggly forward so the curvature is large and well-resolved.
  const double xs[] = {0.030, 0.020, 0.028, 0.018, 0.026, 0.022, 0.031};
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = xs[i];

  const Eigen::MatrixXd R = cal::tension_energy_operator(p, 1.0, 0.0, {0});
  const double e_op = energy(R, x);

  // Independent reference: build the curve at x, second-difference the forward on a fine grid.
  auto crv = curve::make_modular_curve<double>(p.curves[0].modules());
  crv.set_forwards(x);
  const double T = knots.back(), h = 1e-4;
  const int N = 400000;
  const double dt = T / N;
  double e_ref = 0.0;
  auto near_knot = [&](double t) {
    for (double k : knots)
      if (std::abs(t - k) < 3.0 * h) return true;
    return false;
  };
  for (int i = 1; i < N; ++i) {
    const double t = i * dt;
    if (t - h <= 0.0 || t + h >= T || near_knot(t)) continue;
    const double fpp = (crv.forward(t + h) - 2.0 * crv.forward(t) + crv.forward(t - h)) / (h * h);
    e_ref += fpp * fpp * dt;  // trapezoid weight dt (endpoints skipped)
  }
  EXPECT_GT(e_ref, 1e-6) << "the test forward must actually be curved";
  EXPECT_NEAR(e_op, e_ref, 5e-3 * e_ref) << "operator bending energy matches INT(f'')^2 of the built curve";
}

// (A3) The operator is affine in the tension parameter^2: energy(sigma) = energy_bend + sigma^2 *
// energy_membrane. This is what makes sigma = 0 reproduce the pure curvature penalty and sigma a clean
// continuous taut-ness knob (note §5), and it is exact (both K1 and K2 are structure-only quadratic forms).
TEST(TensionRegularizer, TensionParameterDecomposesExactly) {
  const std::vector<double> knots{1.0, 2.0, 4.0, 6.0, 9.0, 12.0};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::NaturalCubic, knots);
  Eigen::VectorXd x(knots.size());
  const double xs[] = {0.031, 0.017, 0.027, 0.020, 0.029, 0.024};
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = xs[i];

  const double e_bend = energy(cal::tension_energy_operator(p, 1.0, 0.0, {0}), x);
  const double e_full1 = energy(cal::tension_energy_operator(p, 1.0, 1.0, {0}), x);
  const double e_membrane = e_full1 - e_bend;
  ASSERT_GT(e_bend, 0.0);
  ASSERT_GT(e_membrane, 0.0);

  for (double sigma : {0.3, 0.7, 1.5, 3.0}) {
    const double e = energy(cal::tension_energy_operator(p, 1.0, sigma, {0}), x);
    const double expect = e_bend + sigma * sigma * e_membrane;
    EXPECT_NEAR(e, expect, 1e-10 * expect) << "energy(sigma) = bend + sigma^2 * membrane, sigma=" << sigma;
  }

  // weight scales the whole penalty by weight^2 (mu = weight^2).
  const double w = 2.5;
  const double e_w = energy(cal::tension_energy_operator(p, w, 0.0, {0}), x);
  EXPECT_NEAR(e_w, w * w * e_bend, 1e-10 * w * w * e_bend) << "row weight scales energy as weight^2";
}
