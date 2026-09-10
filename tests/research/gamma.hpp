#pragma once
// RESEARCH (moved out of include/ in E6.1, 2026-09-10): correct and tested (tests/gamma_test.cpp, bench/gamma_bench.cpp),
// but no production consumer -- second-order curve/market gamma via the reverse tape. Lives with its tests.
// Curve-space GAMMA of a portfolio via the reverse-mode AAD tape (CLAUDE.md §1, north-star #3/#4 -- the
// SECOND-ORDER companion to risk.hpp's first-order bucketed delta). risk.hpp gives d(NPV)/dq (delta) by a
// forward-AAD gradient + the implicit-function theorem. This header gives the curvature: the Hessian
// d^2(NPV)/dx^2 of a book's NPV in the fitted knot forwards x (curve-space gamma), and its Gauss-Newton
// transport to quote space (market-space gamma, to the documented first-order-transport approximation).
//
// METHOD -- forward-over-reverse (see ad/reverse.hpp):
//   * gradient  g = d(NPV)/dx : ONE reverse sweep of Scalar = ad::Rev<double> over the SAME templated npv
//     (portfolio prices with the tape scalar, one backward pass yields every knot's derivative). O(cost of
//     one npv). This is the reverse analogue of risk.hpp::book_curve_grad and matches it entry-for-entry.
//   * Hessian   H = d^2(NPV)/dx^2 : Scalar = ad::Rev<ad::Tangent>. Seeding the tape's forward direction with
//     e_j and sweeping once gives adj[i] = {g_i, (H e_j)_i} -- one Hessian-COLUMN (== HVP) per reverse pass.
//     n = n_knots seeds e_0..e_{n-1} give the full symmetric Hessian in n reverse sweeps.
//     COMPLEXITY: O(n_knots * cost_of_one_npv) time, O(tape) memory (2 doubles/node) -- i.e. the same order
//     as building an n-wide forward-AAD Jacobian, and ~n_knots x the cost of the first-order delta gradient.
//     (A true O(1)-pass full Hessian needs edge-pushing/Hessian-tape accumulation; the n-HVP form was chosen
//     because it REUSES the existing generic npv unchanged and is trivially correct -- CLAUDE.md's
//     "correctness and a clean drop-in Scalar matter more than peak speed".)
//
// DELIVERED vs DEFERRED (market space):
//   * DELIVERED: curve_gamma() -- the exact curve-space Hessian d^2(NPV)/dx^2.
//   * DELIVERED: market_gamma_gn() -- the Gauss-Newton transport H_q = S^T H_x S with S = dx/dq the SAME IFT
//     sensitivity risk.hpp uses (dx/dq = (J^T J)^{-1} J^T). This is the market-space gamma UNDER THE
//     APPROXIMATION that the calibration map x(q) is locally affine (dx/dq constant).
//   * DEFERRED: the full market-space cross-gamma also carries a calibration-curvature term
//     sum_k (d NPV/d x_k) * (d^2 x_k / dq^2). d^2x/dq^2 is the second derivative of the implicit function and
//     needs the calibration Hessian d^2 r/dx^2 (a rank-3 tensor contraction) -- out of scope here; for a
//     well-conditioned near-linear fit it is small, but market_gamma_gn does NOT include it and says so.

#include <Eigen/Dense>

#include <vector>

#include "research/reverse.hpp"
#include "swaps/calibration/jacobian.hpp"       // aad_jacobian (the IFT sensitivity's J)
#include "swaps/calibration/problem.hpp"        // CalibrationProblem
#include "swaps/calibration/risk.hpp"           // ift_operator (the shared, rank-safe dx/dq)
#include "swaps/curve/curve_module.hpp"

namespace swaps::calibration {

// One reverse evaluation of a book's NPV off the calibration curve, in a caller-chosen tape value type T.
// Returns the n_knots adjoints of the seeded knot forwards. `seed[i]` is the T-value to seed knot i as an
// independent variable (its primal x_i, plus -- for T = Tangent -- the forward direction component).
// `pf` must expose `template <class Scalar, class Curve> Scalar npv(const Curve&) const` (portfolio::Portfolio,
// and duck-typed the same way risk.hpp's book_curve_grad requires it).
template <class T, class Portfolio>
inline std::vector<T> npv_reverse(const CalibrationProblem& prob, const std::vector<T>& seed,
                                  const Portfolio& pf) {
  const int n = static_cast<int>(seed.size());
  ad::RevTape<T> tape;
  ad::RevTapeScope<T> scope(tape);  // make `tape` active for T and clear it; restored on return
  std::vector<ad::Rev<T>> xr(n);
  for (int i = 0; i < n; ++i) xr[i] = ad::rev_leaf<T>(seed[i]);  // leaves occupy nodes 0..n-1
  auto c = curve::make_modular_curve<ad::Rev<T>>(curve::flat_hermite(prob.meeting_times, prob.back_times));
  c.set_forwards(xr);
  const ad::Rev<T> pv = pf.template npv<ad::Rev<T>>(c);
  return ad::rev_sweep<T>(tape, pv.idx, n);  // adjoints of the n leaves
}

// Curve-space gradient d(NPV)/dx via ONE reverse sweep -- the reverse-tape twin of risk.hpp::book_curve_grad
// (which does the same in forward mode). Provided so the reverse tape's gradient can be validated against the
// forward-AAD gradient entry-for-entry (tests/gamma_test.cpp).
template <class Portfolio>
inline Eigen::VectorXd curve_gradient(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                                      const Portfolio& pf) {
  const int n = prob.n_knots();
  std::vector<double> seed(n);
  for (int i = 0; i < n; ++i) seed[i] = x[i];
  const std::vector<double> g = npv_reverse<double>(prob, seed, pf);
  return Eigen::Map<const Eigen::VectorXd>(g.data(), n);
}

// Curve-space GAMMA: the Hessian d^2(NPV)/dx^2 (n_knots x n_knots), forward-over-reverse, n_knots HVPs.
// The returned matrix is symmetric up to the reverse sweep's floating error; symmetrized on return.
template <class Portfolio>
inline Eigen::MatrixXd curve_gamma(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                                   const Portfolio& pf) {
  const int n = prob.n_knots();
  Eigen::MatrixXd H(n, n);
  std::vector<ad::Tangent> seed(n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) seed[i] = ad::Tangent(x[i], i == j ? 1.0 : 0.0);  // forward direction e_j
    const std::vector<ad::Tangent> adj = npv_reverse<ad::Tangent>(prob, seed, pf);
    for (int i = 0; i < n; ++i) H(i, j) = adj[i].d;  // adj[i].d = (H e_j)_i ; adj[i].v = gradient_i
  }
  return 0.5 * (H + H.transpose());  // symmetrize away reverse-sweep round-off
}

// IFT quote sensitivity S = dx/dq = (J^T J)^{-1} J^T, J = dr/dx the calibration Jacobian (analytic, AAD) --
// the SAME operator risk.hpp uses for the delta ladder (there applied as J * (J^T J)^{-1} * dNPV/dx).
// Shape: n_knots x n_residuals.
inline Eigen::MatrixXd ift_quote_sensitivity(const CalibrationProblem& prob, const Eigen::VectorXd& x) {
  return ift_operator(prob, x);  // risk.hpp: rank-safe J⁺ times D = diag(−∂r/∂q); one operator, one place
}

// Market-space GAMMA (Gauss-Newton transport): H_q = S^T H_x S, over all market quotes.
// This is d^2(NPV)/dq^2 under the first-order-transport approximation dx/dq = const (see the header note:
// it OMITS the calibration-curvature term sum_k dNPV/dx_k * d^2 x_k/dq^2, which needs the calibration
// Hessian -- deferred). Shape: n_residuals x n_residuals; symmetric by construction.
template <class Portfolio>
inline Eigen::MatrixXd market_gamma_gn(const CalibrationProblem& prob, const Eigen::VectorXd& x,
                                       const Portfolio& pf) {
  const Eigen::MatrixXd Hx = curve_gamma(prob, x, pf);          // n_knots x n_knots
  const Eigen::MatrixXd S = ift_quote_sensitivity(prob, x);     // n_knots x n_resid
  return S.transpose() * Hx * S;                                // n_resid x n_resid
}

}  // namespace swaps::calibration
