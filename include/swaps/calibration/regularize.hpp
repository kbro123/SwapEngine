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
// The fix: add rows that penalise the CURVATURE of the knot forwards -- the divided second difference at
// each interior knot with its trapezoid weight, λ_i·√(½(h₀+h₁))·f''_i (detail::bending_row; since 2026-09-21,
// zero on a straight line in time over ANY pillar spacing; the spacing-blind λ(f_{i-1} − 2f_i + f_{i+1}) it
// replaced was not, and biased every fit on non-uniform pillars) -- for the chosen curves. These are LINEAR
// in x, so they lift the near-zero singular values without touching the data fit in the observable
// directions: the solve now selects the SMOOTHEST curve consistent with the market. Duck-types the problem
// interface (residuals / n_knots / n_residuals), so `calibrate` and `aad_jacobian` drive it unchanged.
// THIS is the smoothing operator (smoothing_preset). The tension-energy operator that lived beside it was retired
// on 2026-09-22 (see RegSpec below).
//
// Apply it ONLY to the curves that need it (a non-smooth reference curve like SOFR — genuine policy
// steps + a shaped term structure — should not be penalised, or it would be biased toward a line).

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "swaps/curve/curve_module.hpp"  // CurveModule: knots, scheme, per-region reg_lambda

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

namespace detail {
// The DISCRETE BENDING ROW at an interior knot with neighbour spacings h0 (left) and h1 (right): the divided second
// difference f''_i ≈ 2[(f_{i+1}−f_i)/h1 − (f_i−f_{i-1})/h0]/(h0+h1) as coefficients (a, b, c) on (f_{i-1}, f_i, f_{i+1}),
// and its trapezoid quadrature weight w = ½(h0+h1), so that Σ_i w·(f''_i)² is a composite-trapezoid ∫(f'')² dt. This is
// Zero on an affine forward for ANY knot spacing -- the property the spacing-blind (e_{i-1} − 2e_i + e_{i+1}) row lacked.
inline void bending_row(double h0, double h1, double& a, double& b, double& c, double& w) {
  const double k = 2.0 / (h0 + h1);
  a = k / h0;
  c = k / h1;
  b = -a - c;
  w = 0.5 * (h0 + h1);
}
}  // namespace detail

// The explicit second-difference (curvature) operator R = λ·D as a DENSE matrix (n_reg × n_knots), one row per interior
// knot of each listed curve: λ_i · √w_i · (divided second difference at knot i) -- detail::bending_row -- so RᵀR is the
// discrete BENDING ENERGY Σ_i w_i (f''_i)² of the knot forwards -- a composite-trapezoid ∫(f'')² dt.
// Until 2026-09-21 the row was the spacing-blind λ(e_{i-1} − 2e_i + e_{i+1}), which is NOT zero on a straight line over
// non-uniform pillars (1y..5y, 7y, 10y, ...): it pulled a square bundle's exact linear solution off the line (the
// ConsistentRisk fixture: a book NPV of 0.0013 became 0.0037 under Light), which is why the tension operator had become
// the default. The divided form is zero on any affine forward, so Light no longer biases an identifiable fit -- and it is
// what makes the second difference fit to be THE default operator (smoothing_preset). This is the same penalty the
// calibrate path composes onto its engine (RegularizedEngine) and the STREAMING calibrator folds as RᵀR into its
// frozen-Newton operator M = (JᵀJ + RᵀR)⁺Jᵀ -- ONE definition of the penalty (E6.1c, 2026-09-10).
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
    std::vector<double> t;
    std::vector<bool> flat;  // a knot in a Flat region: a step level, not a point on a smooth forward
    t.reserve(static_cast<std::size_t>(n));
    for (const auto& m : p.curves[c].modules()) {
      t.insert(t.end(), m.knots.begin(), m.knots.end());
      flat.insert(flat.end(), m.knots.size(), m.scheme == curve::Scheme::Flat);
    }
    for (int i = 1; i < n - 1; ++i) {
      // No curvature row centred on a Flat knot, nor on a knot
      // whose RIGHT neighbour is Flat (a following Flat region ignores its incoming boundary, so the smooth region ends at
      // knot i and has no curvature there). The LEFT neighbour may be a Flat region's last knot: that is the C0 join value
      // the smooth region starts from, and it couples through this row -- which is what pins a basis-only curve's single Flat front knot (EurCurves.*). Meeting-date policy steps are never
      // smoothed; a trailing Flat knot no instrument reaches stays genuinely unreached (ConsistentRisk.UnreachedKnot).
      // A row is kept (as zeros) so the row count stays n - 2 per curve. ASSUMPTIONS.md D18.
      if (flat[static_cast<std::size_t>(i)] || flat[static_cast<std::size_t>(i) + 1]) { ++r; continue; }
      const double li = lam[i];  // the λ of knot i's region
      const double h0 = t[static_cast<std::size_t>(i)] - t[static_cast<std::size_t>(i) - 1];
      const double h1 = t[static_cast<std::size_t>(i) + 1] - t[static_cast<std::size_t>(i)];
      double a, b, cc, w;
      detail::bending_row(h0, h1, a, b, cc, w);
      const double s = li * std::sqrt(w);
      R(r, o + i - 1) = s * a;
      R(r, o + i) = s * b;
      R(r, o + i + 1) = s * cc;
      ++r;
    }
  }
  return R;
}

// The smoothing request every entry point takes (moved here from api/bundle_api.hpp on 2026-09-13): penalise the
// curvature of the listed curves' knot forwards with second_difference_operator at weight `lambda`. lambda <= 0 or
// empty `curves` => off. Needed for basis-only forecast curves whose forward shape is a rank-deficient null
// (CLAUDE.md §7b, EUR trio) and, since the streaming soak (2026-09-21), for any long end whose quotes are integrals
// of the forward. (The continuous tension-energy operator that lived beside this one, with its `tension` flag and
// membrane σ, was RETIRED on 2026-09-22: at equal weight the two were indistinguishable on the soak -- same failed
// ticks, same tick cost, zigzag within 1 bp -- and it cost ~200 lines of shape-function quadrature, a preset column
// of its own, a value-dependent fallback and a UI control. The Tension interpolation SCHEME is unrelated and stays.)
struct RegSpec {
  double lambda = 0.0;
  std::vector<int> curves;
  bool on() const { return lambda > 0.0 && !curves.empty(); }
};

// A request that still names the RETIRED tension-energy operator -- a reg_op other than "second_difference", or a
// non-zero tension_sigma -- is refused here (std::invalid_argument), never silently remapped onto the curvature
// penalty. The compiler and the codecs call this; the rule lives with the regulariser (P14).
inline void refuse_retired_regulariser(bool has_reg_op, std::string_view reg_op, double tension_sigma) {
  if (has_reg_op && reg_op != "second_difference")
    throw std::invalid_argument("reg_op '" + std::string(reg_op) +
                                "': the tension-energy regulariser was retired on 2026-09-22; the only operator is the curvature "
                                "(second-difference) penalty -- omit reg_op or pass \"second_difference\"");
  if (tension_sigma != 0.0)
    throw std::invalid_argument("tension_sigma: the tension-energy regulariser was retired on 2026-09-22; omit it (or pass 0)");
}

// The composer's named smoothing strengths -- ONE table for every entry point (E7 stage 3). compile_reg_spec maps
// a spec's "off"/"light"/"strong" onto it; a verb that needs a well-posed default calibration asks for Light
// instead of writing lambda out. tests/smoothing_preset_test.cpp pins the values.
enum class Smoothing { Off, Light, Strong };

inline double smoothing_lambda(Smoothing s) {
  switch (s) {
    case Smoothing::Off: return 0.0;
    case Smoothing::Light: return 0.5;
    case Smoothing::Strong: return 5.0;
  }
  return 0.0;
}

// A RegSpec at strength `s` over curves 0..n_curves-1. Off => RegSpec{} (no penalty).
inline RegSpec smoothing_preset(Smoothing s, int n_curves) {
  RegSpec reg;
  if (s == Smoothing::Off) return reg;
  reg.lambda = smoothing_lambda(s);
  for (int c = 0; c < n_curves; ++c) reg.curves.push_back(c);
  return reg;
}

}  // namespace swaps::calibration
