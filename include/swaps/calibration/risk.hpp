#pragma once
// Analytic bucketed risk via AAD + the implicit-function theorem (CLAUDE.md §1, north-star #4).
//
// The calibration fixes the knot forwards x by r(x, q) = 0 (square) / min ||r(x,q)||^2, where q are
// the market quotes (rate units). By the IFT the solution's sensitivity to the quotes is
//     dx/dq = J⁺ · D,   J⁺ = (J^T J)^{-1} J^T (Gauss-Newton; = J^{-1} when square),  D = diag(−∂r/∂q)
// with J = dr/dx the calibration Jacobian we ALREADY have from AAD. D is the identity for a hard pin
// (r = model − q) but NOT in general: a banded row is w(q_model)·(q_model − q) (−∂r/∂q = w) and an FX
// forward is (ln F − ln q)/T (−∂r/∂q = 1/(q·T)); dropping D overstated those columns by 1/decay and by
// q·T. J⁺ is a rank-thresholded pseudo-inverse (kRankThreshold, shared with LM/streaming), never a
// tolerance-free LDLT of a possibly-singular JᵀJ. For any portfolio output NPV,
//     d(NPV)/dq = d(NPV)/dx * dx/dq
// where d(NPV)/dx is one more AAD pass. So the FULL bucketed delta ladder over all quotes costs one
// calibration + one AAD gradient + one small linear solve -- no bump-and-reprice, and no bump noise.

#include <Eigen/Dense>

#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/residual_engine.hpp"  // kRankThreshold
#include "swaps/curve/curve_module.hpp"

namespace swaps::calibration {

// D = diag(−∂r_i/∂q_i) at x for a single-curve problem (see the header note).
inline Eigen::VectorXd residual_market_scale(const CalibrationProblem& prob, const Eigen::VectorXd& /*x*/) {
  return residual_market_scale(prob.instruments);  // problem.hpp: the one definition (D needs no curve)
}

// The IFT quote-sensitivity operator dx/dq = J⁺ D  (n_knots x n_residuals), rank-safe.
inline Eigen::MatrixXd ift_operator(const CalibrationProblem& prob, const Eigen::VectorXd& x) {
  const Eigen::MatrixXd J = aad_jacobian(prob, x);  // n_resid x n_knots
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(kRankThreshold);
  cod.compute(J);
  return cod.pseudoInverse() * residual_market_scale(prob, x).asDiagonal();
}

// One AAD pass for d(NPV)/dx off the calibration curve, in a caller-chosen forward-AAD scalar.
template <class Scalar, class Portfolio>
inline Eigen::VectorXd book_curve_grad(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                                       const Portfolio& pf,
                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& seeded) {
  auto c = prob.template make_curve<Scalar>();
  c.set_forwards(seeded);
  const Scalar pv = pf.template npv<Scalar>(c);
  return Eigen::VectorXd(pv.derivatives());  // length n_knots (empty for a curve-independent book)
}

// d(NPV)/dq_j for every market quote j, at the calibrated x. `pf` must expose
//   template <class Scalar, class Curve> Scalar npv(const Curve&) const.
template <class Portfolio>
Eigen::VectorXd bucketed_delta(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                               const Portfolio& pf) {
  // d(NPV)/dx via one AAD pass — R11 pooled (heap-free) for a narrow problem, heap Dual beyond.
  const Eigen::VectorXd dnpv_dx =
      prob.n_knots() <= ad::kPooledMaxW
          ? book_curve_grad<ad::DualPooled<ad::kPooledMaxW>>(prob, x, pf,
                                                             ad::seed_pooled<ad::kPooledMaxW>(x))
          : book_curve_grad<ad::Dual>(prob, x, pf, ad::seed(x));

  // Chain through the IFT: d(NPV)/dq = (dx/dq)ᵀ d(NPV)/dx = (J⁺ D)ᵀ dnpv_dx.
  return ift_operator(prob, x).transpose() * dnpv_dx;  // length n_resid = d(NPV)/dq per quote
}

}  // namespace swaps::calibration
