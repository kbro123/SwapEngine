#pragma once
// Tikhonov SMOOTHNESS regularizer (CLAUDE.md Phase 3's planned fix for weak identification).
//
// A globally-fitted curve can have a NULL SPACE the instruments cannot see: par-swap / basis quotes are
// INTEGRAL quantities (they pin the discount factor at each pillar, i.e. ∫f), so a high-frequency
// oscillation of the forward that preserves those pillar integrals is invisible. This is acute for
// MULTI-CURVE forecast curves pinned only by basis (difference) instruments — e.g. a EURIBOR curve
// pinned by ESTR/3M or 3s6s basis: there is no telescoping 1−DF(T) term, so a forward wiggle averages
// out of pv = Σ f·τ·DF and is unconstrained. A sequential bootstrap hides this (it pins forwards
// pillar-by-pillar, imposing locality); a global LM exposes it as a rank-deficient Jacobian, so the
// cold solve wanders in the null space.
//
// The fix: add rows that penalise the SECOND DIFFERENCE (curvature) of the knot forwards, λ·(f_{i-1} −
// 2f_i + f_{i+1}), for the chosen curves. These are LINEAR in x, so they lift the near-zero singular
// values without touching the data fit in the observable directions: the solve now selects the
// SMOOTHEST curve consistent with the market. Duck-types the problem interface (residuals / n_knots /
// n_residuals), so `calibrate` and `aad_jacobian` drive it unchanged.
//
// Apply it ONLY to the curves that need it (a non-smooth reference curve like SOFR — genuine policy
// steps + a shaped term structure — should not be penalised, or it would be biased toward a line).

#include <Eigen/Core>

#include <utility>
#include <vector>

namespace swaps::calibration {

template <class Problem>
struct SmoothedProblem {
  const Problem* prob;
  double lambda;
  std::vector<std::pair<int, int>> segs;  // (global knot offset, n_knots) of each REGULARISED curve

  SmoothedProblem(const Problem& p, double lam, std::vector<std::pair<int, int>> s)
      : prob(&p), lambda(lam), segs(std::move(s)) {}

  int n_knots() const { return prob->n_knots(); }
  int n_reg() const {
    int n = 0;
    for (const auto& s : segs)
      if (s.second > 2) n += s.second - 2;  // one 2nd-difference row per interior knot
    return n;
  }
  int n_residuals() const { return prob->n_residuals() + n_reg(); }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    const auto r0 = prob->template residuals<Scalar>(x);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    for (int i = 0; i < r0.size(); ++i) r[i] = r0[i];
    int k = static_cast<int>(r0.size());
    for (const auto& s : segs)
      for (int i = 1; i < s.second - 1; ++i)
        r[k++] = lambda * (x[s.first + i - 1] - 2.0 * x[s.first + i] + x[s.first + i + 1]);
    return r;
  }
};

// Regularise EVERY curve of a bundle problem (uses prob.curves for the segment layout).
template <class Problem>
SmoothedProblem<Problem> smoothed(const Problem& p, double lambda) {
  std::vector<std::pair<int, int>> segs;
  int off = 0;
  for (const auto& c : p.curves) {
    segs.push_back({off, c.n_knots()});
    off += c.n_knots();
  }
  return SmoothedProblem<Problem>(p, lambda, std::move(segs));
}

// Regularise ONLY the listed curves (leave shaped reference curves like SOFR untouched).
template <class Problem>
SmoothedProblem<Problem> smoothed(const Problem& p, double lambda, const std::vector<int>& curves) {
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();
  std::vector<std::pair<int, int>> segs;
  for (int c : curves) segs.push_back({off[c], p.curves[c].n_knots()});
  return SmoothedProblem<Problem>(p, lambda, std::move(segs));
}

// The explicit second-difference (curvature) operator R = λ·D as a DENSE matrix (n_reg × n_knots), one
// row λ·(e_{i-1} − 2e_i + e_{i+1}) per interior knot of each listed curve. This is the same penalty
// SmoothedProblem appends as residual rows, materialised so the STREAMING calibrator can fold RᵀR into
// its frozen-Newton operator M = (JᵀJ + RᵀR)⁻¹Jᵀ and stream a rank-deficient (basis-only) build directly.
template <class Problem>
Eigen::MatrixXd second_difference_operator(const Problem& p, double lambda, const std::vector<int>& curves) {
  const int nk = p.n_knots();
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();
  int rows = 0;
  for (int c : curves)
    if (p.curves[c].n_knots() > 2) rows += p.curves[c].n_knots() - 2;
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, nk);
  int r = 0;
  for (int c : curves) {
    const int o = off[c], n = p.curves[c].n_knots();
    for (int i = 1; i < n - 1; ++i) {
      R(r, o + i - 1) = lambda;
      R(r, o + i) = -2.0 * lambda;
      R(r, o + i + 1) = lambda;
      ++r;
    }
  }
  return R;
}

}  // namespace swaps::calibration
