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

// Independent reference tension energy of a SMOOTH single-region curve. Reconstructs each piece's cubic
// from the ENDPOINT-inclusive nodes {0, 1/3, 2/3, 1} -- a DIFFERENT node set than the operator's interior
// {1/8, 3/8, 5/8, 7/8} -- and integrates the derivative squares in the same per-piece closed form. For a
// smooth scheme every piece IS a single cubic on its interval, so BOTH node sets recover it EXACTLY; the
// two energies must therefore agree. That agreement is exactly the property the interior-node fix has to
// preserve for smooth regions (the fix changed only the sampling nodes, and any 4 distinct nodes are exact
// for a cubic), so this is the regression guard that the fix left smooth-region smoothing untouched.
double reference_smooth_tension_energy(const cal::BundleProblem& p, const Eigen::VectorXd& x, double sigma) {
  const auto mods = p.curves[0].modules();
  auto crv = curve::make_modular_curve<double>(mods);
  crv.set_forwards(x);
  const std::vector<double> bp = cal::detail::forward_pieces(mods);
  const double s[4] = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
  Eigen::Matrix4d V;
  for (int m = 0; m < 4; ++m)
    for (int c = 0; c < 4; ++c) V(m, c) = std::pow(s[m], c);
  const Eigen::Matrix4d Vinv = V.inverse();
  double e = 0.0;
  for (int pc = 0; pc + 1 < static_cast<int>(bp.size()); ++pc) {
    const double a = bp[pc], h = bp[pc + 1] - a;
    if (h <= 1e-13) continue;
    Eigen::Vector4d f;
    for (int m = 0; m < 4; ++m) f[m] = crv.forward(a + s[m] * h);
    const Eigen::Vector4d cc = Vinv * f;  // cubic coeffs c0..c3 in the normalised s
    const double c1 = cc[1], c2 = cc[2], c3 = cc[3];
    const double bend = (1.0 / (h * h * h)) * (4.0 * c2 * c2 + 12.0 * c2 * c3 + 12.0 * c3 * c3);
    const double memb =
        (1.0 / h) * (c1 * c1 + 2.0 * c1 * c2 + (4.0 / 3.0) * c2 * c2 + 2.0 * c1 * c3 + 3.0 * c2 * c3 +
                     (9.0 / 5.0) * c3 * c3);
    e += bend + sigma * sigma * memb;
  }
  return e;
}

}  // namespace

// (A1) A Linear region whose knot forwards lie on the straight line f(t) = beta*t is AFFINE (f'==beta,
// f''==0) over the interpolated span [t1, t_last]. As the LEADING (only) region it FLAT-extrapolates its
// first free value backwards, so f == beta*t1 (constant) on the pre-segment [0, t1] -- the corrected short
// end (it no longer ramps from a phantom 0). So bending energy is still 0 (both flat and affine have
// f''==0; the slope kink at t1 is a measure-zero set the operator skips), and the membrane term is
// INT(f')^2 = beta^2 over [t1, t_last] only: beta^2 * (t_last - t1).
TEST(TensionRegularizer, AffineForwardHasZeroBendingEnergyAndExactMembrane) {
  const std::vector<double> knots{1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::Linear, knots);
  const double beta = 0.013;
  Eigen::VectorXd x(knots.size());
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = beta * knots[i];  // on the line f = beta*t

  // sigma = 0 -> pure bending energy INT(f'')^2, which is exactly zero for a flat-then-affine forward.
  const Eigen::MatrixXd R_bend = cal::tension_energy_operator(p, /*weight=*/1.0, /*sigma=*/0.0, {0});
  EXPECT_LT(energy(R_bend, x), 1e-18) << "flat pre-segment + affine span carry no bending energy";

  // With sigma = 1 the added membrane term is INT(f')^2 = beta^2 over the affine span [t1, t_last]; the
  // flat leading pre-segment [0, t1] has f'==0 and contributes nothing (bending still 0).
  const Eigen::MatrixXd R_full = cal::tension_energy_operator(p, /*weight=*/1.0, /*sigma=*/1.0, {0});
  const double expect = beta * beta * (knots.back() - knots.front());
  EXPECT_NEAR(energy(R_full, x), expect, 1e-12 * expect)
      << "membrane energy = beta^2 * (t_last - t1): affine span only, flat leading pre-segment excluded";
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

// (B1) THE DISCONTINUITY-PRESERVATION INVARIANT. A Flat (piecewise-constant, meeting-date front) region is
// DISCONTINUOUS at its knots: on each open interval the forward is a pure CONSTANT, and the curve jumps by
// the (deliberate) step size at every knot. The tension smoother MUST NEVER charge energy for those
// explicit structural steps, whatever their size -- otherwise strong smoothing would erase the meeting-date
// discontinuities. The interior sampling nodes {1/8,3/8,5/8,7/8} guarantee this: each node sits strictly
// inside one constant segment, so the per-piece cubic fit reads a constant (c1=c2=c3=0) and the piece
// contributes EXACTLY zero bending AND membrane energy. (The buggy left-breakpoint node s=0 read the
// PREVIOUS segment across a knot -- because Flat is left-continuous -- so the fit saw a spurious steep ramp
// and charged bending energy, collapsing the steps under strong smoothing.) Big, well-separated steps here
// so the OLD behaviour would have charged a large energy; the invariant is that it is now zero.
TEST(TensionRegularizer, FlatStepDiscontinuitiesHaveZeroTensionEnergy) {
  const std::vector<double> knots{0.25, 0.5, 0.75, 1.0, 1.25};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::Flat, knots);
  Eigen::VectorXd x(knots.size());
  const double steps[] = {0.05, 0.02, 0.06, 0.01, 0.04};  // distinct, well-separated => large jumps
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = steps[i];

  // A generous scale for what "spurious energy" WOULD have looked like: a step of magnitude ~ds over an
  // interval of width ~h contributes O((ds/h)^2 / h) if mistaken for a ramp -- here O(0.05^2 / 0.25^3) ~ 1.
  // Assert the true energy is ~0 to a tolerance FAR below that, for both bending (sigma=0) and membrane.
  const double scale = 1.0;
  const auto mods = p.curves[0].modules();
  const int nl = p.curves[0].n_interp_knots();
  for (double sigma : {0.0, 1.0, 5.0}) {
    // (i) directly on the stiffness K = K2 + sigma^2 K1: the flat forward's energy x^T K x must vanish.
    const Eigen::MatrixXd K = cal::detail::curve_tension_stiffness(mods, nl, 1.0, sigma);
    const double eK = x.transpose() * K * x;
    EXPECT_LT(std::abs(eK), 1e-12 * scale)
        << "x^T K x for a stepped Flat forward must be ZERO (sigma=" << sigma << ")";
    EXPECT_LT(K.cwiseAbs().maxCoeff(), 1e-12 * scale)
        << "a Flat region's whole stiffness matrix must be ZERO (sigma=" << sigma << ")";

    // (ii) through the assembled pseudo-residual operator (the form calibrate/streaming actually use).
    const Eigen::MatrixXd R = cal::tension_energy_operator(p, /*weight=*/1.0, sigma, {0});
    EXPECT_LT(energy(R, x), 1e-12 * scale)
        << "operator energy for the stepped Flat front must be ZERO (sigma=" << sigma << ")";
  }
}

// (B1b) SHORT-INTERVAL edge case. The bending energy carries a 1/h^3 factor, so an intuition says a tiny
// gap between two knots should blow up. It must NOT: the per-piece fit is in normalised s in [0,1]
// (h-INDEPENDENT), and a Flat piece returns the BIT-IDENTICAL segment constant at every interior node, so
// the cubic coefficients are EXACTLY zero and 0 * (1/h^3) = 0 for any h. Steps across a ~1-day and an
// extreme ~1e-6y interval therefore still cost ZERO energy. (Sub-1e-13 gaps can't arise:
// require_increasing_knots rejects duplicates and the operator skips pieces with h <= 1e-13.)
TEST(TensionRegularizer, FlatStepZeroEnergyEvenForTinyIntervals) {
  const std::vector<double> knots{0.25, 0.2527, 0.2527 + 1e-6, 0.75, 1.0};  // ~1-day then ~1e-6y gaps
  const cal::BundleProblem p = one_region_problem(curve::Scheme::Flat, knots);
  Eigen::VectorXd x(knots.size());
  const double steps[] = {0.05, 0.02, 0.06, 0.01, 0.04};  // big jumps ACROSS the tiny intervals
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = steps[i];
  const auto mods = p.curves[0].modules();
  const int nl = p.curves[0].n_interp_knots();
  for (double sigma : {0.0, 1.0, 5.0}) {
    const Eigen::MatrixXd K = cal::detail::curve_tension_stiffness(mods, nl, 1.0, sigma);
    EXPECT_LT(std::abs(double(x.transpose() * K * x)), 1e-12)
        << "a stepped Flat front with TINY intervals must still cost ZERO energy (sigma=" << sigma << ")";
    EXPECT_LT(K.cwiseAbs().maxCoeff(), 1e-12)
        << "Flat stiffness must be ZERO regardless of interval width (sigma=" << sigma << ")";
    const Eigen::MatrixXd R = cal::tension_energy_operator(p, /*weight=*/1.0, sigma, {0});
    EXPECT_LT(energy(R, x), 1e-12)
        << "operator energy must be ZERO for a stepped Flat front with tiny intervals (sigma=" << sigma << ")";
  }
}

// (B2) REGRESSION GUARD for the interior-node change on SMOOTH regions. Moving the sampling nodes from the
// endpoints to the interior must not change the energy of a smooth (here NaturalCubic) forward one bit,
// because both node sets recover a cubic piece exactly. Compare the operator (interior nodes) against an
// independent endpoint-node reconstruction of the same closed-form energy: they must agree to roundoff for
// bending (sigma=0) AND membrane (sigma>0). If a smooth-region energy ever shifted, the fix would NOT be
// exact for cubics -- a red flag, not a re-baselining.
TEST(TensionRegularizer, SmoothRegionTensionEnergyUnchangedByInteriorNodes) {
  const std::vector<double> knots{0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
  const cal::BundleProblem p = one_region_problem(curve::Scheme::NaturalCubic, knots);
  Eigen::VectorXd x(knots.size());
  const double xs[] = {0.030, 0.020, 0.028, 0.018, 0.026, 0.022, 0.031};  // wiggly => real curvature
  for (std::size_t i = 0; i < knots.size(); ++i) x[i] = xs[i];

  for (double sigma : {0.0, 1.0, 2.5}) {
    const double e_op = energy(cal::tension_energy_operator(p, 1.0, sigma, {0}), x);
    const double e_ref = reference_smooth_tension_energy(p, x, sigma);
    ASSERT_GT(e_ref, 1e-6) << "the reference energy must be non-trivial (the curve is genuinely curved)";
    EXPECT_NEAR(e_op, e_ref, 1e-9 * e_ref)
        << "interior-node operator energy must match the endpoint-node reference for smooth cubics, sigma="
        << sigma;
  }
}

// ---- Phase 1: PER-REGION smoothing (second_difference_operator reads each region's reg_lambda) --------

// A curve's two regions carry different curvature lambda: the operator weights each row by the lambda of
// its centre knot's region. Front region lambda=0 (unpenalised), back region lambda=5.
TEST(RegionSmoothing, SecondDifferenceIsPerRegion) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{{1.0, 2.0}, curve::Scheme::Flat, 0.0, 0.0});             // reg_lambda 0
  spec.regions.push_back(curve::CurveModule{{3.0, 4.0, 5.0, 6.0}, curve::Scheme::Hermite, 0.0, 5.0}); // reg_lambda 5
  p.curves.push_back(spec);

  const Eigen::MatrixXd R = cal::second_difference_operator(p, /*default lambda*/ 1.0, {0});
  ASSERT_EQ(R.rows(), 4);  // 6 interp knots -> interior rows i=1..4
  ASSERT_EQ(R.cols(), 6);
  EXPECT_NEAR(R.row(0).cwiseAbs().sum(), 0.0, 1e-15);  // knot i=1 in the lambda=0 front region: zero row
  for (int r = 1; r <= 3; ++r) {                        // knots i=2,3,4 in the lambda=5 back region
    const int i = r + 1;
    EXPECT_NEAR(R(r, i - 1), 5.0, 1e-12);
    EXPECT_NEAR(R(r, i), -10.0, 1e-12);
    EXPECT_NEAR(R(r, i + 1), 5.0, 1e-12);
  }
}

// Backward compatibility: with no region overriding (reg_lambda<0), the operator is the plain single-lambda
// stencil everywhere -- byte-identical to the pre-Phase-1 global-lambda behaviour.
TEST(RegionSmoothing, UniformLambdaMatchesGlobal) {
  cal::BundleProblem p;
  cal::BundleCurveSpec spec;
  spec.base = -1;
  spec.regions.push_back(curve::CurveModule{{1.0, 2.0, 3.0}, curve::Scheme::Flat});     // reg_lambda default (-1)
  spec.regions.push_back(curve::CurveModule{{4.0, 5.0, 6.0}, curve::Scheme::Hermite});  // reg_lambda default (-1)
  p.curves.push_back(spec);

  const Eigen::MatrixXd R = cal::second_difference_operator(p, /*global lambda*/ 0.7, {0});
  ASSERT_EQ(R.rows(), 4);
  for (int r = 0; r < 4; ++r) {
    const int i = r + 1;
    EXPECT_NEAR(R(r, i - 1), 0.7, 1e-12);
    EXPECT_NEAR(R(r, i), -1.4, 1e-12);
    EXPECT_NEAR(R(r, i + 1), 0.7, 1e-12);
  }
}

// Phase 2: per-region TENSION-ENERGY σ (and weight). Setting a region's reg_sigma changes only that
// region's membrane term; inheriting (reg_sigma<0) reproduces the global-σ operator byte-for-byte.
TEST(RegionSmoothing, TensionEnergyIsPerRegionSigma) {
  auto make = [](double back_reg_sigma, double back_reg_lambda) {
    cal::BundleProblem p;
    cal::BundleCurveSpec spec;
    spec.base = -1;
    spec.regions.push_back(curve::CurveModule{{0.25, 0.5}, curve::Scheme::Flat});  // front: zero energy
    curve::CurveModule back{{1.0, 2.0, 5.0, 10.0}, curve::Scheme::NaturalCubic};    // back: real curvature
    back.reg_sigma = back_reg_sigma;
    back.reg_lambda = back_reg_lambda;
    spec.regions.push_back(back);
    p.curves.push_back(spec);
    return p;
  };
  const double weight = 0.5, gsig = 1.0;
  Eigen::VectorXd x(6);
  x << 0.030, 0.028, 0.020, 0.026, 0.019, 0.031;  // wiggly -> genuine tension energy

  const Eigen::MatrixXd R_inh = cal::tension_energy_operator(make(-1.0, -1.0), weight, gsig, {0});  // inherit
  const Eigen::MatrixXd R_exp = cal::tension_energy_operator(make(1.0, -1.0), weight, gsig, {0});   // σ=1 explicit
  const Eigen::MatrixXd R_hi = cal::tension_energy_operator(make(3.0, -1.0), weight, gsig, {0});    // σ=3
  const Eigen::MatrixXd R_w2 = cal::tension_energy_operator(make(-1.0, 2.0 * weight), weight, gsig, {0});  // ρ=2

  const double e_inh = (R_inh * x).squaredNorm();
  ASSERT_GT(e_inh, 1e-9) << "the back region must carry non-trivial tension energy";
  // Explicit σ == the global default: byte-identical energy.
  EXPECT_NEAR((R_exp * x).squaredNorm(), e_inh, 1e-12 * e_inh);
  // Higher per-region σ adds membrane energy.
  EXPECT_GT((R_hi * x).squaredNorm(), e_inh * (1.0 + 1e-6));
  // Per-region weight ρ=2 scales this region's energy by ρ²=4 (front carries none).
  EXPECT_NEAR((R_w2 * x).squaredNorm(), 4.0 * e_inh, 1e-9 * e_inh);
}
