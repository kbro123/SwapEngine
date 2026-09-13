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

// ---- rank-completed risk (the generate_risk ladder; moved from api/generate_risk.cpp, E7 stage 3.7) ------------------
// Left-multiply a curve gradient g = dP/dx by the RANK-COMPLETED risk operator. J is the calibration Jacobian dr/dx
// (n_res x n_knots). Any null direction of J -- a knot the quotes cannot resolve -- is SELF-QUOTED: appended to J as a
// unit-pinned row, so J_full has full column rank and M_full = pinv(J_full) is well-posed with NO curvature penalty.
// A direction is "unseen" when its SINGULAR VALUE is null at the engine's ONE rank threshold (kRankThreshold,
// relative to sigma_max) -- the same test calibrate()'s rank_deficiency and the streamer's operator use (E3-G5).
//
// KNOWN BUG, carried unchanged by the move and fixed separately (TASKS-ENGINE E7 "RISK SCALE BUGS" (1)): the real
// rows are pinv(J_full)ᵀ g WITHOUT the residual market scale D that risk_operator applies, so a banded row is
// overstated by 1/decay and an FX row by q·T.
struct NullCompletedLadder {
  Eigen::VectorXd full;             // n_residuals real quotes first, then one entry per synthetic pillar
  int n_residuals = 0;
  std::vector<int> synthetic_knot;  // each synthetic pillar's dominant knot index (for labelling)
};

inline NullCompletedLadder null_completed_ladder(const Eigen::MatrixXd& J, const Eigen::VectorXd& g) {
  const int n_res = static_cast<int>(J.rows());
  const int nk = static_cast<int>(J.cols());
  NullCompletedLadder out;
  out.n_residuals = n_res;
  int rank = 0;
  Eigen::MatrixXd V = Eigen::MatrixXd::Identity(nk, nk);
  if (n_res > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullV);
    svd.setThreshold(kRankThreshold);
    rank = static_cast<int>(svd.rank());
    V = svd.matrixV();  // columns rank.. are the null directions (singular values descending)
  }
  const int n_null = nk - rank;
  Eigen::MatrixXd Jf(n_res + n_null, nk);
  if (n_res) Jf.topRows(n_res) = J;
  for (int j = 0; j < n_null; ++j) {
    const Eigen::VectorXd v = V.col(rank + j);  // a unit null direction in knot space
    Jf.row(n_res + j) = v.transpose();          // self-quote it (a direct pin on that direction)
    int idx = 0;
    v.cwiseAbs().maxCoeff(&idx);                // the knot it loads on most -> its label
    out.synthetic_knot.push_back(idx);
  }
  // Jf has full column rank by construction; the pseudo-inverse (rank-safe at the same threshold) is
  // (JfᵀJf)⁻¹Jfᵀ without forming the normal matrix.
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(kRankThreshold);
  cod.compute(Jf);
  const Eigen::MatrixXd Mf = cod.pseudoInverse();  // n_knots x (n_res + n_null)
  out.full = Mf.transpose() * g;                   // length n_res + n_null
  return out;
}

}  // namespace swaps::calibration
