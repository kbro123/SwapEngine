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
// knot of a smoothed region: coefficients λᵢ·√wᵢ·(a, b, c) on (f_{i-1}, f_i, f_{i+1}) -- detail::bending_row, the
// divided second difference with its trapezoid weight, so Σ rows² is the discrete BENDING ENERGY Σ_i w_i (f''_i)², a
// composite-trapezoid ∫(f'')² dt -- target 0, plain penalty map. Until 2026-09-21 the row was the spacing-blind
// λ(e_{i-1} − 2e_i + e_{i+1}), which is NOT zero on a straight line over non-uniform pillars (1y..5y, 7y, 10y, ...): it
// pulled a square bundle's exact linear solution off the line (the ConsistentRisk fixture: a book NPV of 0.0013
// became 0.0037 under Light), which is why the tension operator had become the default. The divided form is zero on
// any affine forward, so Light no longer biases an identifiable fit.
//
// SMOOTHING IS REGION DATA (owner's rule, 2026-10-13). A region declares its own weight when the curve is defined --
// `CurveModule::reg_lambda`: > 0 smoothed at that weight, anything else unsmoothed -- and that is the ONLY source of
// the calibration's smoothing rows: there is no bundle-level default a region inherits (the `RegSpec` that every verb
// used to take for calibration is gone from those verbs; it survives as the RISK-side damping, a separate penalty on
// the risk operator -- second_difference_operator below). A session installs these rows at construction
// (BundleProblem::constraints, the session's tail rows) and the one compiled engine carries them as three Src::State
// terms each -- residual R·x, Jacobian row R, in the same stack the streamer's frozen system, the LM and the risk
// operator read.
//
// No row is emitted for a knot in a Flat region, nor for a knot whose RIGHT neighbour is Flat (a following Flat region
// ignores its incoming boundary, so the smooth region ends at knot i and has no curvature there). The LEFT neighbour
// may be a Flat region's last knot: that is the C0 join value the smooth region starts from, and it couples through
// this row -- which is what pins a basis-only curve's single Flat front knot (EurCurves.*). Meeting-date policy steps
// are never smoothed; a trailing Flat knot no instrument reaches stays genuinely unreached
// (ConsistentRisk.UnreachedKnot). ASSUMPTIONS.md D18.
namespace detail {
// The rows for ONE weighting: lambda_of(c, i) is knot i of curve c's weight (0 or less = no row).
template <class Problem, class LambdaOf>
std::vector<Instrument> curvature_rows(const Problem& p, LambdaOf&& lambda_of) {
  std::vector<Instrument> rows;
  for (int c = 0; c < static_cast<int>(p.curves.size()); ++c) {
    const int n = p.curves[c].n_interp_knots();  // curvature over interp knots only (no δ)
    std::vector<double> t;
    std::vector<bool> flat;  // a knot in a Flat region: a step level, not a point on a smooth forward
    t.reserve(static_cast<std::size_t>(n));
    for (const auto& m : p.curves[c].modules()) {
      t.insert(t.end(), m.knots.begin(), m.knots.end());
      flat.insert(flat.end(), m.knots.size(), m.scheme == curve::Scheme::Flat);
    }
    for (int i = 1; i < n - 1; ++i) {
      if (flat[static_cast<std::size_t>(i)] || flat[static_cast<std::size_t>(i) + 1]) continue;
      const double li = lambda_of(c, i);
      if (!(li > 0.0)) continue;  // unsmoothed: no row
      const double h0 = t[static_cast<std::size_t>(i)] - t[static_cast<std::size_t>(i) - 1];
      const double h1 = t[static_cast<std::size_t>(i) + 1] - t[static_cast<std::size_t>(i)];
      double a, b, cc, w;
      bending_row(h0, h1, a, b, cc, w);
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
}  // namespace detail

// SMOOTHING IS A TIE-BREAKER, NEVER A TRADE AGAINST THE QUOTES (owner's rule, 2026-10-13, ASSUMPTIONS K48). The
// smoothing rows choose the minimum-curvature curve AMONG the curves the quotes allow -- the unique point when the
// quotes are square and full-rank (nothing to choose), the smooth completion of the directions an under-determined
// quote set leaves free, the minimum-curvature point inside the bid/offer bands where a banded quote is a BOUND (a
// dead zone inside, PenaltyMap) -- and never move a hard quote off its market or a banded quote outside its band.
// Written as rows of the least squares, that is the lexicographic limit of a weight w -> 0: the leak of a curvature
// row into the quotes is O(w^2) (at the optimum J_q^T r_q = -w^2 R^T R x), while the directions the quotes do not
// pin are set by the rows alone whatever w is. kSmoothingTieBreak is that weight, fixed by numerics only:
// MEASURED (scratch sweep over every ladder rung and the G10 desk, 2026-10-13) the worst hard-quote misfit is
// 7e-6 bp at 1e-4 (2.7e-9 bp on annual OIS, 7.3e-6 on monthly averaged futures -- denser knots leak more), 7e-4 bp
// at 1e-3 and 0.1..0.7 bp at the retired 0.5, and the rows' singular values at 1e-4 sit ~1e-5 of the quotes': far
// above kRankThreshold (1e-10). A region's reg_lambda is therefore a SWITCH: > 0 smooths that region's knots at this
// weight, 0 / unset leaves them to the quotes alone; its magnitude is not read (a "strength" would be a trade
// against the quotes, which is exactly what this rule forbids). tests/smoothing_tie_breaker_test.cpp pins the
// round trip (< 1e-4 bp on every hard quote, every banded quote inside its band) on every shape.
inline constexpr double kSmoothingTieBreak = 1e-4;
// THE PULL INSIDE A BAND WHEN ITS REGION IS SMOOTHED (the hierarchy's third level, 2026-10-13): a banded quote is a
// BOUND -- outside the band the row pulls at slope 1 -- and inside it needs SOME pull toward mid, or the curvature rows
// (which annihilate a straight line) leave the curve's level and slope undetermined within the bands and the
// minimum-curvature point parks the quotes on the band EDGES (measured on annual OIS: six of twelve quotes at +-0.25
// bp, 279..546 LM iterations, the streamer's band walk tripping on a 10^4 slope ratio at the edge). The pull must lose
// to the curvature at the jitter scale and win above it: MEASURED on alternating +-0.03 bp jitter in +-0.25 bp bands,
// the exact fit zigzags the knots by 32.5 bp; inside-band decay 1e-3 (10x the tie-breaker) takes that to 2.6 bp with
// every quote within 0.06 bp of mid in 6 iterations, 1e-2 leaves 21 bp, 3e-2 and above leave it all. This is the
// recommended `band_decay` for a quote on a smoothed region (the composer / SDK default); the engine reads the row's
// own decay as data and applies nothing itself.
inline constexpr double kSmoothedBandDecay = 10.0 * kSmoothingTieBreak;

// The calibration's smoothing rows: one per interior knot of every region whose reg_lambda is set (> 0), at the
// tie-breaker weight. Nothing else (no bundle-level default, no strength).
template <class Problem>
std::vector<Instrument> smoothing_rows(const Problem& p) {
  std::vector<std::vector<double>> on(p.curves.size());
  for (std::size_t c = 0; c < p.curves.size(); ++c) on[c] = region_knot_lambdas(p.curves[c], 0.0);
  return detail::curvature_rows(p, [&](int c, int i) {
    return on[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] > 0.0 ? kSmoothingTieBreak : 0.0;
  });
}
// Stamp the switch onto every non-Flat region of the listed curves (curves empty = all): how a fixture, a test or a
// default layout SETS smoothing as region data before a session is built (`lambda` > 0 = on, 0 = off; the value is
// data for the switch only). A Flat region needs no stamp (no rows).
template <class Problem>
void stamp_region_smoothing(Problem& p, double lambda, const std::vector<int>& curves = {}) {
  for (int c = 0; c < static_cast<int>(p.curves.size()); ++c) {
    if (!curves.empty() && std::find(curves.begin(), curves.end(), c) == curves.end()) continue;
    for (auto& m : p.curves[c].regions)
      if (m.scheme != curve::Scheme::Flat) m.reg_lambda = lambda;
  }
}
// A LEGACY layout's word onto its regions: for every curve whose regions ALL say nothing, stamp `lambda` onto the
// non-Flat ones (the compile verb's migration of a spec-level `smoothness`; a curve with any region that said is left
// alone). Returns whether anything was stamped.
template <class Problem>
bool stamp_silent_regions(Problem& p, double lambda) {
  bool stamped = false;
  for (auto& cs : p.curves) {
    bool any_said = false;
    for (const auto& m : cs.regions) any_said = any_said || m.reg_lambda >= 0.0;
    if (any_said) continue;
    for (auto& m : cs.regions)
      if (m.scheme != curve::Scheme::Flat) { m.reg_lambda = lambda; stamped = true; }
  }
  return stamped;
}
// Append them to the problem's constraint rows (the session does this at construction). Empty when no region says.
template <class Problem>
int add_smoothing_rows(Problem& p) {
  std::vector<Instrument> rows = smoothing_rows(p);
  p.constraints.insert(p.constraints.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
  return static_cast<int>(rows.size());
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

// The RISK-side damping operator R = λ·D as a DENSE matrix: one row per interior knot of each LISTED curve at the
// uniform weight `lambda` (the Flat rule as above; a region's own calibration weight is NOT consulted -- this is the
// separate penalty `risk_operator(reg)` stacks under the engine's rows to damp a delta ladder's fan-out, DV01
// preserved since R annihilates level + linear moves). Also the dense view the coefficient tests read.
template <class Problem>
Eigen::MatrixXd second_difference_operator(const Problem& p, double lambda, const std::vector<int>& curves) {
  const auto listed = [&](int c) { return std::find(curves.begin(), curves.end(), c) != curves.end(); };
  return state_rows_matrix(p, detail::curvature_rows(p, [&](int c, int) { return listed(c) ? lambda : 0.0; }));
}

// The RISK-side regularisation request (2026-10-13: no calibration verb takes one any more -- the calibration's
// smoothing is region data, CurveModule::reg_lambda): damp the risk operator with second_difference_operator over the
// listed curves at weight `lambda`. lambda <= 0 or empty `curves` => off. The named strengths below are the composer's
// table for a REGION's weight (smoothing_lambda), shared with the legacy-spec migration in the compile verb. (The continuous tension-energy operator that lived beside this one, with its `tension` flag and
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

// The composer's smoothing words -- ONE table for every entry point (E7 stage 3). Since 2026-10-13 (K48) smoothing
// is a switch: "on" (and the legacy words "light" / "strong", which named strengths when smoothing traded against
// the quotes) stamp 1 onto a region's reg_lambda, "off" stamps 0. The enum keeps Light / Strong as spellings of On
// so a saved spec still reads; they are the same switch. tests/smoothing_preset_test.cpp pins the values.
enum class Smoothing { Off, Light, Strong, On = Light };

inline double smoothing_lambda(Smoothing s) {
  switch (s) {
    case Smoothing::Off: return 0.0;
    case Smoothing::Light: return 1.0;
    case Smoothing::Strong: return 1.0;
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
