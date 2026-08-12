#pragma once
// Analytic Jacobian J[i][k] = d residual_i / d knot_k via forward-mode AAD, in ONE differentiated
// evaluation of the residual code (CLAUDE.md §1 -- no bump-and-reprice). Its own header so the
// residual-engine trait can use it without pulling in lm.hpp's calibrate() (which now depends on the
// trait -- see calibration/residual_engine.hpp). `Problem` is anything exposing residuals<Scalar>(x),
// n_knots(), n_residuals().

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"

namespace swaps::calibration {

// Scatter one differentiated residual pass (whatever the seed's Scalar) into J. `Seeded` is the seeded
// knot vector (Dual or DualPooled); the residual gradients are size m for curve-dependent rows.
template <class Scalar, class Problem, class Seeded>
inline Eigen::MatrixXd aad_jacobian_with(const Problem& prob, const Seeded& xd) {
  const auto rd = prob.template residuals<Scalar>(xd);
  const int n = prob.n_residuals(), m = prob.n_knots();
  Eigen::MatrixXd J(n, m);
  for (int i = 0; i < n; ++i) {
    // A residual with no curve dependence would have an empty gradient; guard defensively.
    if (rd[i].derivatives().size() == m)
      J.row(i) = rd[i].derivatives().transpose();
    else
      J.row(i).setZero();
  }
  return J;
}

template <class Problem>
Eigen::MatrixXd aad_jacobian(const Problem& prob, const Eigen::VectorXd& x) {
  // R11: for a narrow problem the pooled dual keeps the whole sweep off the heap (bit-identical to the
  // `Dual` path); a wider bundle falls back to the heap `Dual`.
  if (prob.n_knots() <= ad::kPooledMaxW)
    return aad_jacobian_with<ad::DualPooled<ad::kPooledMaxW>>(prob, ad::seed_pooled<ad::kPooledMaxW>(x));
  return aad_jacobian_with<ad::Dual>(prob, ad::seed(x));
}

}  // namespace swaps::calibration
