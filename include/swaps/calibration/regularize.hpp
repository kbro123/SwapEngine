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

#include "swaps/calibration/problem.hpp"  // Instrument, QuoteKind::StateLinear, StateRef: a penalty row IS an instrument
#include "swaps/curve/curve_module.hpp"    // CurveModule: knots, scheme, per-region reg_lambda

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

// THE SMOOTHING PENALTY AS ROWS (constraint rows, review item 3, 2026-10-12). One StateLinear instrument per interior
// knot of each listed curve: coefficients λ_i·√w_i·(a, b, c) on (f_{i-1}, f_i, f_{i+1}) -- detail::bending_row, the
// divided second difference with its trapezoid weight, so Σ rows² is the discrete BENDING ENERGY Σ_i w_i (f''_i)², a
// composite-trapezoid ∫(f'')² dt -- target 0, plain penalty map. Until 2026-09-21 the row was the spacing-blind
// λ(e_{i-1} − 2e_i + e_{i+1}), which is NOT zero on a straight line over non-uniform pillars (1y..5y, 7y, 10y, ...): it
// pulled a square bundle's exact linear solution off the line (the ConsistentRisk fixture: a book NPV of 0.0013
// became 0.0037 under Light), which is why the tension operator had become the default. The divided form is zero on
// any affine forward, so Light no longer biases an identifiable fit -- and it is what makes the second difference THE
// default operator (smoothing_preset).
//
// These rows go into BundleProblem::constraints (add_smoothing_rows / BundleSession's RegSpec arguments): the ONE
// compiled engine then carries them as three Src::State terms each -- residual R·x, Jacobian row R, in the same stack
// [instruments; constraints] the streamer's frozen system, the LM and the risk operator read. Until 2026-10-12 the
// same penalty was a dense matrix input (second_difference_operator) that every consumer stacked under its own J;
// second_difference_operator below is now the dense VIEW of these rows, kept for the tests that pin the coefficients.
//
// No row is emitted for a knot in a Flat region, nor for a knot whose RIGHT neighbour is Flat (a following Flat region
// ignores its incoming boundary, so the smooth region ends at knot i and has no curvature there). The LEFT neighbour
// may be a Flat region's last knot: that is the C0 join value the smooth region starts from, and it couples through
// this row -- which is what pins a basis-only curve's single Flat front knot (EurCurves.*). Meeting-date policy steps
// are never smoothed; a trailing Flat knot no instrument reaches stays genuinely unreached
// (ConsistentRisk.UnreachedKnot). ASSUMPTIONS.md D18. (Until 2026-10-12 a skipped knot kept an all-zero row so the
// dense operator's row count stayed n − 2 per curve; a zero row is no row.)
template <class Problem>
std::vector<Instrument> smoothing_rows(const Problem& p, double lambda, const std::vector<int>& curves) {
  std::vector<Instrument> rows;
  for (int c : curves) {
    const int n = p.curves[c].n_interp_knots();  // curvature over interp knots only (no δ)
    const std::vector<double> lam = region_knot_lambdas(p.curves[c], lambda);  // per-region λ (Phase 1)
    std::vector<double> t;
    std::vector<bool> flat;  // a knot in a Flat region: a step level, not a point on a smooth forward
    t.reserve(static_cast<std::size_t>(n));
    for (const auto& m : p.curves[c].modules()) {
      t.insert(t.end(), m.knots.begin(), m.knots.end());
      flat.insert(flat.end(), m.knots.size(), m.scheme == curve::Scheme::Flat);
    }
    for (int i = 1; i < n - 1; ++i) {
      if (flat[static_cast<std::size_t>(i)] || flat[static_cast<std::size_t>(i) + 1]) continue;
      const double li = lam[i];  // the λ of knot i's region
      const double h0 = t[static_cast<std::size_t>(i)] - t[static_cast<std::size_t>(i) - 1];
      const double h1 = t[static_cast<std::size_t>(i) + 1] - t[static_cast<std::size_t>(i)];
      double a, b, cc, w;
      detail::bending_row(h0, h1, a, b, cc, w);
      const double sc = li * std::sqrt(w);
      Instrument r;
      r.quote = QuoteKind::StateLinear;
      r.market = 0.0;
      r.state = {{c, i - 1, sc * a}, {c, i, sc * b}, {c, i + 1, sc * cc}};
      rows.push_back(std::move(r));
    }
  }
  return rows;
}

// Install the smoothing rows for `reg` as the problem's constraint rows (appended). Off => nothing appended.
template <class Problem, class Reg>
void add_smoothing_rows(Problem& p, const Reg& reg) {
  if (!reg.on()) return;
  std::vector<Instrument> rows = smoothing_rows(p, reg.lambda, reg.curves);
  p.constraints.insert(p.constraints.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
}

// A SESSION's installed smoothing (2026-10-12): the rows for `reg` kept as the TAIL `n_tail` rows of the problem's
// constraints. Returns false (nothing touched) when `reg` is the installed one; otherwise replaces the tail with the rows
// for `reg` (none when off), records it in `installed` / `n_tail` and returns true -- the row set changed, so the caller
// drops anything compiled against it. The client's own constraint rows (before the tail) are never touched.
template <class Problem, class Reg>
bool install_smoothing_rows(Problem& p, int& n_tail, Reg& installed, const Reg& reg) {
  if (reg.lambda == installed.lambda && reg.curves == installed.curves) return false;
  p.constraints.resize(static_cast<std::size_t>(p.n_constraints() - n_tail));  // drop the old tail
  n_tail = 0;
  if (reg.on()) {
    std::vector<Instrument> rows = smoothing_rows(p, reg.lambda, reg.curves);
    n_tail = static_cast<int>(rows.size());
    p.constraints.insert(p.constraints.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
  }
  installed = reg;
  return true;
}

// The coefficient matrix of a list of StateLinear rows over the problem's stacked state (n_rows × n_knots): each
// ref's coefficient at its global state index (curve offset + index). The dense view of smoothing_rows.
template <class Problem>
Eigen::MatrixXd state_rows_matrix(const Problem& p, const std::vector<Instrument>& rows) {
  const int nk = p.n_knots();
  std::vector<int> off(p.curves.size(), 0);
  for (std::size_t c = 1; c < p.curves.size(); ++c) off[c] = off[c - 1] + p.curves[c - 1].n_knots();
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(static_cast<int>(rows.size()), nk);
  for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
    if (rows[static_cast<std::size_t>(r)].quote != QuoteKind::StateLinear)
      throw std::invalid_argument("state_rows_matrix: row " + std::to_string(r) + " is not a StateLinear row");
    for (const StateRef& s : rows[static_cast<std::size_t>(r)].state) R(r, off[static_cast<std::size_t>(s.curve)] + s.index) += s.coeff;
  }
  return R;
}

// The explicit second-difference (curvature) operator R = λ·D as a DENSE matrix: the dense view of smoothing_rows
// (one row per emitted interior knot, in knot order). RᵀR is the discrete bending energy. Kept for the tests that pin
// the coefficients and for a caller that wants the operator as a matrix; the engine reads the rows.
template <class Problem>
Eigen::MatrixXd second_difference_operator(const Problem& p, double lambda, const std::vector<int>& curves) {
  return state_rows_matrix(p, smoothing_rows(p, lambda, curves));
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
