#pragma once
// Analytic bucketed risk via AAD + the implicit-function theorem (CLAUDE.md §1, north-star #4).
//
// The calibration fixes the knot forwards x by r(x, q) = 0 (square) / min ||r(x,q)||^2, where q are
// the market quotes (rate units; residual_i = model_i(x) - q_i, so dr/dq = -I). By the IFT the
// solution's sensitivity to the quotes is
//     dx/dq = (J^T J)^{-1} J^T          (Gauss-Newton; = J^{-1} when square)
// with J = dr/dx the calibration Jacobian we ALREADY have from AAD. For any portfolio output NPV,
//     d(NPV)/dq = d(NPV)/dx * dx/dq
// where d(NPV)/dx is one more AAD pass. So the FULL bucketed delta ladder over all quotes costs one
// calibration + one AAD gradient + one small linear solve -- no bump-and-reprice, and no bump noise.

#include <Eigen/Dense>

#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"

namespace swaps::calibration {

// d(NPV)/dq_j for every market quote j, at the calibrated x. `pf` must expose
//   template <class Scalar, class Curve> Scalar npv(const Curve&) const.
template <class Portfolio>
Eigen::VectorXd bucketed_delta(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                               const Portfolio& pf) {
  // d(NPV)/dx via one AAD pass.
  curve::TwoRegionForwardCurve<ad::Dual> c(prob.meeting_times, prob.back_times);
  c.set_forwards(ad::seed(x));
  const ad::Dual pv = pf.template npv<ad::Dual>(c);
  const Eigen::VectorXd dnpv_dx = pv.derivatives();

  // J = dr/dx (analytic), then chain through the IFT.
  const Eigen::MatrixXd J = aad_jacobian(prob, x);      // n_resid x n_knots
  const Eigen::MatrixXd JtJ = J.transpose() * J;        // n_knots x n_knots
  const Eigen::VectorXd a = JtJ.ldlt().solve(dnpv_dx);  // (J^T J)^{-1} dnpv_dx
  return J * a;                                          // length n_resid = d(NPV)/dq per quote
}

}  // namespace swaps::calibration
