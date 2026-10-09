#pragma once
// THE ONE normal-equations operator (2026-09-22, architecture review item 3a).
//
// Every solver-side consumer of "the pseudo-inverse of the stacked [J; R]" -- the streaming corrector's frozen
// operator, the session's risk operator dx/dq, the single-curve IFT ladder, the rank-completed generate_risk
// ladder, the LM's rank-deficiency test and the background Jacobian worker -- used to carry its own
// CompleteOrthogonalDecomposition at kRankThreshold, four implementations of one object (PRINCIPLES P4). This is
// that object, written once:
//
//     M = J⁺ = (JᵀJ)⁺ Jᵀ        the frozen-Newton preconditioner and, times D = diag(−∂r/∂q), the IFT operator dx/dq
//     G = (JᵀJ)⁺                kept for the O(n·m) rank-one band re-scale (Sherman–Morrison) and the KKT tests
//
// over the ROWS it is given. CONSTRAINT ROWS (the review's item 3, last half, 2026-10-09): a smoothing regulariser,
// the LM's seed anchor, generate_risk's self-quoted null pillars are all ROWS stacked under the instrument rows --
// the streamer appends R·x to its residual and R to its frozen Jacobian (streaming.hpp), the LM composes
// RegularizedEngine, risk stacks [J; R] -- and this operator is the pseudo-inverse of the stack. Until 2026-10-09 it
// also carried the regulariser as an operator BLOCK (RᵀR, the stacked scratch S, the curvature pull B = G·RᵀR
// added to every step): the same algebra written a second way, with its own update in the rank-one re-scale.
// That block is gone: one row kind, one stack, one pseudo-inverse.
//
// RANK-SAFE by construction: a rank-thresholded COD of the stacked rows. A bundle can be legitimately
// rank-deficient (a knot no instrument pins, redundant xccy / basis rows), and a regulariser's null space (level +
// linear forward moves) can meet J's when a curve has knots but no level-pinning row: a tolerance-free LDLT of that
// singular normal matrix used to send every tick to the refresh cap and walk the unpinned curve to negative forwards.
// The COD zeroes the null directions instead, so an unpinned state never moves off its anchor.
//
// Allocation: the decomposition is sized ONCE (reset / the sizing ctor), so a refresh on the streaming tick
// recomputes into storage it already owns (C6); form() keeps the temporaries the hot-path census pins (an identity,
// the solve, the product). The rank-one update allocates nothing after first use. The two-threshold WALK policy
// (kWalkRankThreshold, truncation, re-anchoring) belongs to the streamer: this type only exposes
// decompose-at-a-threshold and rank-at-a-threshold so that policy can be written on top of it.
#include <Eigen/Dense>

#include <cmath>
#include <utility>

#include "swaps/calibration/residual_engine.hpp"  // kRankThreshold: the ONE rank rule

namespace swaps::calibration {

class NormalOp {
 public:
  NormalOp() = default;
  // Size everything once for n_rows stacked rows (instrument rows, then whatever constraint rows the caller stacks).
  NormalOp(int n_rows, int n_knots) { reset(n_rows, n_knots); }

  void reset(int n_rows, int n_knots) {
    n_res_ = n_rows;
    n_knots_ = n_knots;
    cod_ = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(n_rows, n_knots);  // sized once
  }

  // (1) Decompose the stacked rows at `threshold`. Nothing is formed yet (a caller that only wants the rank stops here).
  void decompose(const Eigen::MatrixXd& J, double threshold = kRankThreshold) {
    // ONE decomposition for the operator's life (C6, 2026-09-15): compute() on a same-shaped matrix writes into
    // storage it already owns -- the same Eigen operations on the same data as a fresh local, bit-identical.
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& cod = cod_;
    cod.setThreshold(threshold);
    cod.compute(J);
  }
  // The rank of the CURRENT decomposition at `threshold` (no recompute; the threshold stays set).
  Eigen::Index rank_at(double threshold) {
    cod_.setThreshold(threshold);
    return cod_.rank();
  }
  Eigen::Index rank() const { return cod_.rank(); }
  double max_pivot() const { return cod_.maxPivot(); }

  // (2) Form M and G from the current decomposition.
  void form() {
    M_ = cod_.solve(Eigen::MatrixXd::Identity(n_res_, n_res_));
    G_.noalias() = M_ * M_.transpose();  // (JᵀJ)⁺ = J⁺ J⁺ᵀ
  }
  // decompose + form at the ONE shared threshold: what every non-streaming consumer wants.
  void factor(const Eigen::MatrixXd& J) {
    decompose(J, kRankThreshold);
    form();
  }

  // (3) A rank-one row re-scale (E3-C7, 2026-09-10): row `row` of the frozen Jacobian changes slope by a factor c,
  // a RANK-ONE change of JᵀJ (β·u uᵀ, β = c² − 1, u = the OLD row). The operator is updated exactly by
  // Sherman–Morrison on G = (JᵀJ)⁺ (u lies in G's range, so the pseudo-inverse form holds):
  //     v = G u,  d = 1 + β uᵀv,   G' = G − (β/d) v vᵀ,
  //     M' = G' J'ᵀ = M − (β/d) v (J v)ᵀ + ((c−1)/d) v e_rowᵀ,
  // O(n·m) and allocation-free after first use, where a re-factorisation is O(n³). `J_old` is the Jacobian BEFORE
  // the row changed. Returns false when the update is degenerate (releasing a pinned row whose leverage cancels d):
  // the caller must re-factorise instead.
  bool rank_one_update(int row, double c, const Eigen::MatrixXd& J_old) {
    u_ = J_old.row(row).transpose();
    v_.noalias() = G_ * u_;
    const double s = u_.dot(v_), beta = c * c - 1.0, d = 1.0 + beta * s;
    // Releasing a PIN (c = decay / 1e3) on a row whose leverage s is close to 1 cancels d = 1 + βs
    // catastrophically; such a row is rare (one release per row per tick) -- re-factorise instead.
    if (std::isfinite(d) && std::abs(d) > 1e-3) {
      jv_.noalias() = J_old * v_;  // the OLD J
      const double f = beta / d;
      M_.noalias() -= f * v_ * jv_.transpose();
      M_.col(row) += ((c - 1.0) / d) * v_;
      G_.noalias() -= f * v_ * v_.transpose();
      return true;
    }
    return false;
  }

  // Adopt an M computed elsewhere (the background worker hands over M alone for a square, unbanded, unregularised
  // problem). G is NOT updated: a rank-one update after this would use a stale G, which is why the worker is
  // armed only where no band re-scale can happen.
  void adopt_M(Eigen::MatrixXd M) { M_ = std::move(M); }

  const Eigen::MatrixXd& M() const { return M_; }
  const Eigen::MatrixXd& G() const { return G_; }
  int n_res() const { return n_res_; }  // the stacked row count
  int n_knots() const { return n_knots_; }

  // The IFT quote-sensitivity operator dx/dq = M · D with D = diag(−∂r/∂q) (residual_market_scale), n_knots × n_res:
  // over every row, or over the first `n_quote_rows` (the instrument rows of a stack whose tail is constraint rows).
  Eigen::MatrixXd quote_sensitivity(const Eigen::VectorXd& market_scale) const {
    return M_ * market_scale.asDiagonal();
  }
  Eigen::MatrixXd quote_sensitivity(const Eigen::VectorXd& market_scale, int n_quote_rows) const {
    return M_.leftCols(n_quote_rows) * market_scale.asDiagonal();
  }

 private:
  int n_res_ = 0, n_knots_ = 0;
  Eigen::MatrixXd M_;
  Eigen::MatrixXd G_;        // (JᵀJ)⁺ from the last form(), updated rank-one per band re-scale
  Eigen::VectorXd u_, v_, jv_;  // rank_one_update scratch (no per-rescale allocation after first use)
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;  // sized once (C6)
};

}  // namespace swaps::calibration
