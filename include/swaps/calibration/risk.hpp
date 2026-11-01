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

#include <stdexcept>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/normal_op.hpp"  // THE operator: dx/dq = M·D, rank-safe at the one threshold
#include "swaps/calibration/residual_engine.hpp"  // kRankThreshold
#include "swaps/curve/curve_module.hpp"

namespace swaps::calibration {

// (The single-curve ift_operator / bucketed_delta over a CalibrationProblem are test fixtures since 2026-11-01,
// tests/single_curve.hpp; the shipped ladder is BundleSession::risk_operator / generate_risk below, over the
// one rank-safe NormalOp.)

// ---- rank-completed risk (the generate_risk ladder; moved from api/generate_risk.cpp, E7 stage 3.7) ------------------
// Left-multiply a curve gradient g = dP/dx by the RANK-COMPLETED risk operator. J is the calibration Jacobian dr/dx
// (n_res x n_knots). Any null direction of J -- a knot the quotes cannot resolve -- is SELF-QUOTED: appended to J as a
// unit-pinned row, so J_full has full column rank and M_full = pinv(J_full) is well-posed with NO curvature penalty.
// A direction is "unseen" when its SINGULAR VALUE is null at the engine's ONE rank threshold (kRankThreshold,
// relative to sigma_max) -- the same test calibrate()'s rank_deficiency and the streamer's operator use (E3-G5).
//
// J is the RESIDUAL Jacobian, so pinv(J_full)ᵀ g is dP/dr; the quote ladder is dP/dq = D ⊙ dP/dr with D = diag(−dr/dq)
// = residual_market_scale -- exactly the scale risk_operator puts on M. `market_scale` is REQUIRED (no default: its
// omission is the bug). FIXED 2026-09-13: D was omitted, so a banded row was overstated by 1/decay and an FX forward
// by q·T (tests/risk_scale_repro_test.cpp).
struct NullCompletedLadder {
  Eigen::VectorXd full;             // n_residuals real quotes first, then one entry per synthetic pillar
  int n_residuals = 0;
  std::vector<int> synthetic_knot;  // each synthetic pillar's dominant knot index (for labelling)
};

inline NullCompletedLadder null_completed_ladder(const Eigen::MatrixXd& J, const Eigen::VectorXd& g,
                                                 const Eigen::VectorXd& market_scale) {
  const int n_res = static_cast<int>(J.rows());
  const int nk = static_cast<int>(J.cols());
  if (market_scale.size() != n_res)
    throw std::invalid_argument("null_completed_ladder: the market scale needs one entry per residual");
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
  // Jf has full column rank by construction; THE operator (normal_op.hpp) gives (JfᵀJf)⁻¹Jfᵀ rank-safe at the same
  // threshold without forming the normal matrix.
  NormalOp op(n_res + n_null, nk);
  op.factor(Jf);
  out.full = op.M().transpose() * g;  // length n_res + n_null: dP/dr, then the synthetic pillars
  out.full.head(n_res).array() *= market_scale.array();  // dP/dq on the real quotes
  return out;
}

}  // namespace swaps::calibration
