#pragma once
// Analytic Jacobian J[i][k] = d residual_i / d knot_k via forward-mode AAD, in ONE differentiated
// evaluation of the residual code (CLAUDE.md §1 -- no bump-and-reprice). Its own header so the
// residual-engine trait can use it without pulling in lm.hpp's calibrate() (which now depends on the
// trait -- see calibration/residual_engine.hpp). `Problem` is anything exposing residuals<Scalar>(x),
// n_knots(), n_residuals().

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"

namespace swaps::calibration {

template <class Problem>
Eigen::MatrixXd aad_jacobian(const Problem& prob, const Eigen::VectorXd& x) {
  const auto rd = prob.template residuals<ad::Dual>(ad::seed(x));
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

}  // namespace swaps::calibration
