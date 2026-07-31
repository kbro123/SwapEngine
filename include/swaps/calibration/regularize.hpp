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
    segs.push_back({off, c.n_interp_knots()});  // penalise the interpolation knots only, NOT turn δ's
    off += c.n_knots();                          // but stride over the WHOLE state block (incl. δ's)
  }
  return SmoothedProblem<Problem>(p, lambda, std::move(segs));
}

// Regularise ONLY the listed curves (leave shaped reference curves like SOFR untouched).
template <class Problem>
SmoothedProblem<Problem> smoothed(const Problem& p, double lambda, const std::vector<int>& curves) {
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();
  std::vector<std::pair<int, int>> segs;
  for (int c : curves) segs.push_back({off[c], p.curves[c].n_interp_knots()});  // interp knots only (no δ)
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
    if (p.curves[c].n_interp_knots() > 2) rows += p.curves[c].n_interp_knots() - 2;
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, nk);
  int r = 0;
  for (int c : curves) {
    const int o = off[c], n = p.curves[c].n_interp_knots();  // curvature over interp knots only (no δ)
    for (int i = 1; i < n - 1; ++i) {
      R(r, o + i - 1) = lambda;
      R(r, o + i) = -2.0 * lambda;
      R(r, o + i + 1) = lambda;
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
// Numerics: within each knot interval the forward is a single polynomial of degree <= 3 (Flat const,
// Linear affine, Hermite/NaturalCubic/BSpline cubic), so f'' is affine and INT(f'')^2 / INT(f')^2 are
// computed in CLOSED FORM per interval from the interval's cubic coefficients -- EXACT, no quadrature
// error. The coefficients are recovered by interpolating the (structure-only) forward shape functions at
// 4 nodes of the interval (a fixed, well-conditioned Vandermonde). The flat meeting-date front has
// f'' = f' = 0 inside each segment, so it contributes zero energy (the note's "minimise energy only in
// the residual freedom": meetings stay discontinuous), while its LAST knot still couples into the back
// region's energy through the C0 join -- captured because the whole ModularCurve is evaluated, not a
// region in isolation.

namespace detail {

// Piece breakpoints over which the forward is a SINGLE polynomial (<= cubic), so the per-interval cubic
// fit below is exact. Nodes are the region knots for every scheme except BSpline, whose polynomial pieces
// break at the region's UNIFORM interior knots (regions.hpp BSpline), not at the input knots.
inline std::vector<double> forward_pieces(const std::vector<curve::CurveModule>& mods) {
  std::vector<double> bp{0.0};
  double t0 = 0.0;
  for (const auto& m : mods) {
    const auto& k = m.knots;
    if (k.empty()) continue;
    const double te = k.back();
    if (m.scheme == curve::Scheme::BSpline) {
      const int n = static_cast<int>(k.size());  // n free control points -> uniform interior breakpoints
      for (int j = 1; j <= n - 3; ++j) bp.push_back(t0 + (te - t0) * (static_cast<double>(j) / (n - 2)));
      bp.push_back(te);
    } else {
      for (double kv : k) bp.push_back(kv);
    }
    t0 = te;
  }
  std::sort(bp.begin(), bp.end());
  bp.erase(std::unique(bp.begin(), bp.end(),
                       [](double a, double b) { return std::abs(a - b) <= 1e-13 * (1.0 + std::abs(a)); }),
           bp.end());
  return bp;
}

// K1 (membrane) and K2 (bending) stiffness matrices of ONE curve's own knots, from its region schemes.
// Returns K = K2 + sigma^2 K1 (n_local x n_local, symmetric SPSD). Off the hot path (setup only).
inline Eigen::MatrixXd curve_tension_stiffness(const std::vector<curve::CurveModule>& mods, int n_local,
                                               double sigma) {
  auto crv = curve::make_modular_curve<double>(mods);
  const std::vector<double> bp = detail::forward_pieces(mods);
  const int P = static_cast<int>(bp.size()) - 1;

  // Fixed cubic Vandermonde on the INTERIOR nodes {1/8, 3/8, 5/8, 7/8} (working in s = (t-a)/h keeps the
  // fit well-conditioned regardless of the interval width h). The forward is a single polynomial (<= cubic)
  // on the OPEN interval, so any 4 distinct nodes recover it EXACTLY for every smooth scheme -- the energy
  // is identical to sampling the endpoints. But the nodes MUST stay interior: a Flat region is
  // DISCONTINUOUS at its knots (the piece endpoints), so sampling an endpoint reads the neighbouring
  // segment and makes an intended meeting-date STEP look like a steep ramp -- the smoother would then
  // charge bending energy for it and erase the discontinuity. Interior nodes see only the constant
  // segment, so a Flat piece contributes ZERO energy and the explicit jumps we put in are never smoothed.
  const double s[4] = {0.125, 0.375, 0.625, 0.875};
  Eigen::Matrix4d V;
  for (int m = 0; m < 4; ++m)
    for (int c = 0; c < 4; ++c) V(m, c) = std::pow(s[m], c);
  const Eigen::Matrix4d Vinv = V.inverse();

  // Forward shape functions sampled at every piece node: F[p](m, j) = Phi_j(a_p + s_m*h_p), obtained by
  // evaluating the curve on each unit knot vector e_j (one build per local knot -- cheap, setup only).
  std::vector<Eigen::MatrixXd> F(P, Eigen::MatrixXd::Zero(4, n_local));
  Eigen::VectorXd e = Eigen::VectorXd::Zero(n_local);
  for (int j = 0; j < n_local; ++j) {
    e.setZero();
    e[j] = 1.0;
    crv.set_forwards(e);
    for (int p = 0; p < P; ++p) {
      const double a = bp[p], h = bp[p + 1] - a;
      for (int m = 0; m < 4; ++m) F[p](m, j) = crv.forward(a + s[m] * h);
    }
  }

  Eigen::MatrixXd K1 = Eigen::MatrixXd::Zero(n_local, n_local);
  Eigen::MatrixXd K2 = Eigen::MatrixXd::Zero(n_local, n_local);
  auto sym = [](const Eigen::RowVectorXd& a, const Eigen::RowVectorXd& b) {
    return (a.transpose() * b + b.transpose() * a).eval();
  };
  for (int p = 0; p < P; ++p) {
    const double h = bp[p + 1] - bp[p];
    if (h <= 1e-13) continue;
    // Cubic coeffs (in the normalised s) as linear functions of x: rows of G = Vinv*F are c0..c3. Subtract
    // the piece's CONSTANT baseline (row 0) before the solve: the energy uses only c1..c3, which are
    // invariant to a constant shift, but this keeps the (large) constant out of the fit so it is never
    // amplified by the 1/h^3 / 1/h factors below. For a Flat piece F is bit-identical across nodes, so
    // F - row0 == 0 EXACTLY => c1=c2=c3=0 EXACTLY and the energy is zero for ANY interval width (a very
    // short interval would otherwise blow the Vandermonde-inverse roundoff up by 1/h^3). Smooth pieces are
    // unchanged (the subtraction only shifts c0).
    const Eigen::MatrixXd G = Vinv * (F[p].rowwise() - F[p].row(0));
    const Eigen::RowVectorXd g1 = G.row(1), g2 = G.row(2), g3 = G.row(3);
    // f(u) = c0 + c1 s + c2 s^2 + c3 s^3, s = u/h.  INT_0^h (f'')^2 du = (1/h^3)[4c2^2+12c2c3+12c3^2];
    // INT_0^h (f')^2 du = (1/h)[c1^2 + 2c1c2 + (4/3)c2^2 + 2c1c3 + 3c2c3 + (9/5)c3^2].  (see the note's
    // closed forms; the constant term c0 drops out of both derivatives.)
    K2 += (1.0 / (h * h * h)) * (4.0 * (g2.transpose() * g2) + 6.0 * sym(g2, g3) + 12.0 * (g3.transpose() * g3));
    K1 += (1.0 / h) * ((g1.transpose() * g1) + sym(g1, g2) + (4.0 / 3.0) * (g2.transpose() * g2) +
                       sym(g1, g3) + 1.5 * sym(g2, g3) + (9.0 / 5.0) * (g3.transpose() * g3));
  }
  return K2 + (sigma * sigma) * K1;
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
    K.block(o, o, nl, nl) += detail::curve_tension_stiffness(p.curves[c].modules(), nl, sigma);
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
