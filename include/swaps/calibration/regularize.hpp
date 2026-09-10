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

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "swaps/curve/curve_module.hpp"  // ModularCurve: extract the structure-only forward shape functions

namespace swaps::calibration {

// Per-knot smoothing weight for one curve: each REGION's own reg_lambda if set (>=0), else the bundle
// default. Length == the curve's interpolation-knot count, in knot order. A curve whose regions never
// override (reg_lambda<0 everywhere — e.g. the legacy Flat/Hermite layout) yields a uniform vector equal
// to the old single global λ, so the penalty rows are byte-identical to before. (Phase 1.)
template <class CurveSpec>
inline std::vector<double> region_knot_lambdas(const CurveSpec& c, double default_lambda) {
  std::vector<double> lam;
  lam.reserve(static_cast<std::size_t>(c.n_interp_knots()));
  for (const auto& m : c.modules())
    lam.insert(lam.end(), m.knots.size(), m.reg_lambda >= 0.0 ? m.reg_lambda : default_lambda);
  return lam;
}

template <class Problem>
struct SmoothedProblem {
  const Problem* prob;
  std::vector<std::pair<int, int>> segs;  // (global knot offset, n_interp_knots) of each REGULARISED curve
  std::vector<std::vector<double>> lam;   // per-seg, per-knot smoothing weight (region λ or the default)

  SmoothedProblem(const Problem& p, std::vector<std::pair<int, int>> s, std::vector<std::vector<double>> l)
      : prob(&p), segs(std::move(s)), lam(std::move(l)) {}

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
    for (std::size_t si = 0; si < segs.size(); ++si) {
      const int off = segs[si].first, n = segs[si].second;
      for (int i = 1; i < n - 1; ++i)  // curvature row weighted by the λ of its centre knot's region
        r[k++] = lam[si][i] * (x[off + i - 1] - 2.0 * x[off + i] + x[off + i + 1]);
    }
    return r;
  }
};

// Regularise EVERY curve of a bundle problem (uses prob.curves for the segment layout).
template <class Problem>
SmoothedProblem<Problem> smoothed(const Problem& p, double lambda) {
  std::vector<std::pair<int, int>> segs;
  std::vector<std::vector<double>> lam;
  int off = 0;
  for (const auto& c : p.curves) {
    segs.push_back({off, c.n_interp_knots()});  // penalise the interpolation knots only, NOT turn δ's
    lam.push_back(region_knot_lambdas(c, lambda));
    off += c.n_knots();                          // but stride over the WHOLE state block (incl. δ's)
  }
  return SmoothedProblem<Problem>(p, std::move(segs), std::move(lam));
}

// Regularise ONLY the listed curves (leave shaped reference curves like SOFR untouched).
template <class Problem>
SmoothedProblem<Problem> smoothed(const Problem& p, double lambda, const std::vector<int>& curves) {
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();
  std::vector<std::pair<int, int>> segs;
  std::vector<std::vector<double>> lam;
  for (int c : curves) {
    segs.push_back({off[c], p.curves[c].n_interp_knots()});  // interp knots only (no δ)
    lam.push_back(region_knot_lambdas(p.curves[c], lambda));
  }
  return SmoothedProblem<Problem>(p, std::move(segs), std::move(lam));
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
    if (p.curves[c].n_interp_knots() > 2) rows += p.curves[c].n_interp_knots() - 2;
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, nk);
  int r = 0;
  for (int c : curves) {
    const int o = off[c], n = p.curves[c].n_interp_knots();  // curvature over interp knots only (no δ)
    const std::vector<double> lam = region_knot_lambdas(p.curves[c], lambda);  // per-region λ (Phase 1)
    for (int i = 1; i < n - 1; ++i) {
      const double li = lam[i];  // the λ of knot i's region
      R(r, o + i - 1) = li;
      R(r, o + i) = -2.0 * li;
      R(r, o + i + 1) = li;
      ++r;
    }
  }
  return R;
}

// =================================================================================================
// TENSION-ENERGY regularizer (tension-spline research note §5) -- a SMOOTHNESS penalty, not a new spline.
//
// Where second_difference_operator penalises a DISCRETE proxy (the 2nd difference of the knot forwards),
// this penalises the true CONTINUOUS tension energy of the interpolated forward f(t) = Phi(t)*x:
//
//     tension energy = mu * x^T (K2 + sigma^2 K1) x ,   K1 = INT Phi' Phi'^T,  K2 = INT Phi'' Phi''^T
//
//   * sigma = 0  -> pure bending energy INT (f'')^2  (Holladay / natural-cubic objective).
//   * sigma > 0  -> adds the membrane term INT (f')^2  (spline under tension): taut, overshoot-damped.
//
// K1, K2 are STRUCTURE-ONLY (depend on the knot TIMES and the region schemes, never on x) because every
// shipped region is a LINEAR MAP of the knot values (regions.hpp): f = Phi(t)*x with Phi structure-only.
// So the penalty is delivered exactly like the existing one -- as CONSTANT pseudo-residual rows
// R = weight * L, where L^T L = K2 + sigma^2 K1 (a symmetric SPSD factorisation). Then R^T R =
// weight^2 (K2 + sigma^2 K1), i.e. mu = weight^2, and R drops into calibrate / risk_operator / streaming
// as the SAME kind of constant R matrix second_difference_operator already produces -- no AAD, no
// per-iteration cost, nothing on the differentiated hot loop. All the sinh-free spline algebra here is
// ONE-TIME setup (build L once), exactly as the research note requires.
//
// Numerics (rebuilt 2026-09-10): every region exposes its EXACT forward derivatives (forward_d1/forward_d2)
// and its analytic-piece breakpoints (pieces()); the energy is Gauss quadrature of (f'')² + σ²(f')² per piece
// with a rule that is exact for the piece's form — 3-point Gauss for polynomial pieces (Flat/Linear/cubics/
// B-spline on its TRUE de Boor breakpoints), a σ·h-sized composite 6-point rule for Tension pieces (sinh/cosh,
// which the old per-piece cubic fit could not represent: 18-100 % energy error). The flat meeting-date front has
// f'' = f' = 0 inside each segment, so it contributes zero energy (the note's "minimise energy only in
// the residual freedom": meetings stay discontinuous), while its LAST knot still couples into the back
// region's energy through the C0 join -- captured because the whole ModularCurve is evaluated, not a
// region in isolation.

namespace detail {

// (forward_pieces retired 2026-09-10: the curve's regions own their breakpoints — ModularCurve::pieces(). The old
// helper assumed UNIFORM B-spline breakpoints, stale since the de Boor-averaged knots: 25 % bending-energy error.)

// K1 (membrane) and K2 (bending) stiffness of ONE curve's own knots, from its region schemes.
// Returns K = Σ_pieces ρ²·(K2_p + σ_p²·K1_p) (n_local x n_local, symmetric SPSD). Off the hot path (setup).
//
// PER-REGION (Phase 2): each interval (piece) uses the σ and a RELATIVE weight ρ = reg_λ/default of the
// region containing its MIDPOINT — so the C0-join interval [A.back, B.front] belongs to the back region B
// (which owns that extrapolation energy). A region inheriting (reg_lambda<0 -> ρ=1, reg_sigma<0 -> σ=default)
// contributes exactly as before, so with no overrides K == K2 + default_σ²·K1 UNCHANGED (the outer `weight`
// in tension_energy_operator then reproduces the old operator byte-for-byte, incl. the eigen-rank tolerance).
inline Eigen::MatrixXd curve_tension_stiffness(const std::vector<curve::CurveModule>& mods, int n_local,
                                               double default_weight, double default_sigma) {
  auto crv = curve::make_modular_curve<double>(mods);
  crv.set_forwards(Eigen::VectorXd::Zero(n_local));  // regions know their pieces (joins, de Boor knots) once BUILT
  // Pieces: 0 (the flat pre-segment start) plus every breakpoint the regions declare — the TRUE analytic
  // pieces (B-spline: the de Boor-averaged interior knots; Tension: its nodes; cubics: their nodes incl. joins).
  std::vector<double> bp = crv.pieces();
  bp.insert(bp.begin(), 0.0);
  std::sort(bp.begin(), bp.end());
  bp.erase(std::unique(bp.begin(), bp.end(),
                       [](double a, double b) { return std::abs(a - b) <= 1e-13 * (1.0 + std::abs(a)); }),
           bp.end());
  const int P = static_cast<int>(bp.size()) - 1;

  // Per-region relative weight ρ (=1 when inheriting; default_weight>0 is guaranteed by the operator guard)
  // and σ, plus each region's last-knot time for the midpoint->region lookup.
  std::vector<double> region_end, rho, rsig;
  for (const auto& m : mods) {
    if (m.knots.empty()) continue;
    region_end.push_back(m.knots.back());
    rho.push_back((m.reg_lambda >= 0.0 ? m.reg_lambda : default_weight) / default_weight);
    rsig.push_back(m.reg_sigma >= 0.0 ? m.reg_sigma : default_sigma);
  }
  auto region_of = [&](double t_mid) -> int {
    for (std::size_t r = 0; r < region_end.size(); ++r)
      if (t_mid <= region_end[r] + 1e-12) return static_cast<int>(r);
    return static_cast<int>(region_end.size()) - 1;
  };

  // Quadrature nodes per piece, EXACT for the piece's analytic form:
  //   polynomial pieces (Flat/Linear/cubics/B-spline): (f')² has degree <= 4 and (f'')² <= 2, so 3-point
  //     Gauss-Legendre (exact to degree 5) integrates both exactly;
  //   tension pieces (sinh/cosh with rate σ): composite Gauss with ceil(σ·h) sub-intervals of 6-point
  //     Gauss each (the exponent per sub-interval is <= 2, integrated to ~1e-14).
  // Nodes are strictly INTERIOR, so a Flat piece (d1 = d2 = 0 inside, jumps at its ends) contributes EXACTLY
  // zero — the explicit meeting-date steps are never smoothed — and a Tension/cubic piece never reads its
  // neighbour.
  static const double g3x[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
  static const double g3w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  static const double g6x[6] = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                                0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
  static const double g6w[6] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
  struct Node { double t, w; int piece; };
  std::vector<Node> nodes;
  for (int p = 0; p < P; ++p) {
    const double a = bp[p], b = bp[p + 1], h = b - a;
    if (h <= 1e-13) continue;
    const double sg_interp = crv.tension_sigma_at(0.5 * (a + b));
    if (sg_interp > 0.0) {
      const int m = std::max(1, static_cast<int>(std::ceil(sg_interp * h)));
      const double hs = h / m;
      for (int s = 0; s < m; ++s) {
        const double c = a + (s + 0.5) * hs, r = 0.5 * hs;
        for (int q = 0; q < 6; ++q) nodes.push_back({c + g6x[q] * r, g6w[q] * r, p});
      }
    } else {
      const double c = 0.5 * (a + b), r = 0.5 * h;
      for (int q = 0; q < 3; ++q) nodes.push_back({c + g3x[q] * r, g3w[q] * r, p});
    }
  }
  const int Q = static_cast<int>(nodes.size());

  // Derivative shape functions: D1(q, j) = ∂f'(t_q)/∂x_j, D2(q, j) = ∂f''(t_q)/∂x_j — evaluate the curve on each
  // unit knot vector (one build per local knot; setup only). Linear maps => these rows are structure-only.
  Eigen::MatrixXd D1 = Eigen::MatrixXd::Zero(Q, n_local), D2 = Eigen::MatrixXd::Zero(Q, n_local);
  Eigen::VectorXd e = Eigen::VectorXd::Zero(n_local);
  for (int j = 0; j < n_local; ++j) {
    e.setZero();
    e[j] = 1.0;
    crv.set_forwards(e);
    for (int q = 0; q < Q; ++q) {
      D1(q, j) = crv.forward_d1(nodes[q].t);
      D2(q, j) = crv.forward_d2(nodes[q].t);
    }
  }
  // K = Σ_q w_q ρ²_p [ D2_qᵀ D2_q + σ_p² D1_qᵀ D1_q ]  (symmetric SPSD by construction)
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_local, n_local);
  for (int q = 0; q < Q; ++q) {
    const int p = nodes[q].piece;
    const int rgn = region_of(0.5 * (bp[p] + bp[p + 1]));
    const double r2 = rho[rgn] * rho[rgn], sg = rsig[rgn], w = nodes[q].w;
    K.noalias() += (w * r2) * (D2.row(q).transpose() * D2.row(q));
    K.noalias() += (w * r2 * sg * sg) * (D1.row(q).transpose() * D1.row(q));
  }
  return K;
}

}  // namespace detail

// The tension-energy pseudo-residual block R = weight * L (rows x n_knots), with L^T L = K2 + sigma^2 K1
// assembled over the listed curves (a block-diagonal SPSD form; each curve's energy touches only its own
// knots). Factored via a symmetric eigendecomposition (K = V D V^T, D >= 0): L keeps one row
// sqrt(D_i) v_i^T per POSITIVE eigenvalue, so R^T R = weight^2 (K2 + sigma^2 K1) exactly on the range and
// the rank-deficient null (constant/linear forward directions that carry no tension energy) is simply
// dropped rather than forcing a plain Cholesky to fail. Slots in wherever second_difference_operator does
// (streaming's opt.regularizer, risk_operator's R^T R). weight <= 0 or empty `curves` => 0 rows (off).
template <class Problem>
Eigen::MatrixXd tension_energy_operator(const Problem& p, double weight, double sigma,
                                        const std::vector<int>& curves) {
  const int nk = p.n_knots();
  if (weight <= 0.0 || curves.empty()) return Eigen::MatrixXd(0, nk);
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();

  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nk, nk);
  for (int c : curves) {
    const int nl = p.curves[c].n_interp_knots();  // tension energy is over the interp forward, not δ's
    const int o = off[c];
    K.block(o, o, nl, nl) += detail::curve_tension_stiffness(p.curves[c].modules(), nl, weight, sigma);
  }
  K = 0.5 * (K + K.transpose());  // kill any roundoff asymmetry before the self-adjoint solve

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  const Eigen::VectorXd ev = es.eigenvalues();
  const double tol = 1e-12 * std::max(1.0, ev.cwiseAbs().maxCoeff());
  int rows = 0;
  for (int i = 0; i < ev.size(); ++i)
    if (ev[i] > tol) ++rows;
  Eigen::MatrixXd R(rows, nk);
  int r = 0;
  for (int i = 0; i < ev.size(); ++i)
    if (ev[i] > tol) R.row(r++) = (weight * std::sqrt(ev[i])) * es.eigenvectors().col(i).transpose();
  return R;
}

// A problem wrapped with an ARBITRARY constant pseudo-residual block R (rows x n_knots): appends R*x as
// extra residual rows, so `calibrate` / `aad_jacobian` drive the tension-regularised least squares
// UNCHANGED (duck-types n_knots / n_residuals / residuals<Scalar>, exactly like SmoothedProblem). This is
// the calibrate-side counterpart of folding R^T R into the streaming operator: the same R, two views.
template <class Problem>
struct LinearRegularizedProblem {
  const Problem* prob;
  Eigen::MatrixXd R;  // constant Jacobian block for the pseudo-residuals R*x (no AAD; structure-only)

  LinearRegularizedProblem(const Problem& p, Eigen::MatrixXd r) : prob(&p), R(std::move(r)) {}

  int n_knots() const { return prob->n_knots(); }
  int n_reg() const { return static_cast<int>(R.rows()); }
  int n_residuals() const { return prob->n_residuals() + n_reg(); }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    const auto r0 = prob->template residuals<Scalar>(x);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    for (int i = 0; i < r0.size(); ++i) r[i] = r0[i];
    int k = static_cast<int>(r0.size());
    for (int i = 0; i < R.rows(); ++i) {
      Scalar acc(0.0);
      for (int j = 0; j < R.cols(); ++j) acc += R(i, j) * x[j];  // constant row . x
      r[k++] = acc;
    }
    return r;
  }
};

template <class Problem>
LinearRegularizedProblem<Problem> linearly_regularized(const Problem& p, Eigen::MatrixXd R) {
  return LinearRegularizedProblem<Problem>(p, std::move(R));
}

}  // namespace swaps::calibration
