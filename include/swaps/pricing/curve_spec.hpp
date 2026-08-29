#pragma once
// CurveStructure -- the topology of ONE curve inside a bundle: its knot times (or interpolation
// regions), whether it is outright or a spread off another curve, and an engine-blind currency tag.
//
// This is the SINGLE shared description used by both layers that need it: the pricing engine
// (CompiledCurveSet / CompiledBundleResidual, which build W from modules()) and the calibration problem
// (BundleProblem, whose CurveSpec is an alias of this type). It lives in the pricing layer -- the lower
// of the two -- so calibration can depend on it without a cycle. Previously calibration carried its own
// byte-identical `BundleCurveSpec` and a hand-written field-by-field copy kept the two in sync; folding
// them into one type removes that drift hazard.

#include <algorithm>
#include <vector>

#include "swaps/curve/curve_module.hpp"

namespace swaps::pricing {

// A TURN -- a localized ~1-day jump in the overnight forward (year/quarter/month-end, CB maintenance).
// It is an additive OVERLAY on top of the smooth interpolation, NOT an interpolation knot or a region
// (docs/turns-calibration.md §5): the jump δ is its own flat basis function over the accrual window
// [start, end] (year fractions). The engine names NO calendar -- the window is RESOLVED by the
// web/compile layer from the turn's single date against the overnight index calendar, and threaded in
// here as {start, end}. The turn contributes δ·overlap(t, [start,end]) to the log-DF integral, so
// DF = exp(−(W·x + Σ δⱼ·overlapⱼ)) stays on the linear W-cache fast path.
struct Turn {
  double start = 0.0;  // accrual-window start (year fraction on the curve day count)
  double end = 0.0;    // accrual-window end   (year fraction); end > start
};

// overlap(t, [a,b]) = max(0, min(t,b) − a): the integrated width of the turn window seen by a cashflow
// at time t. 0 before the window, the full width (b−a) after it, linearly ramping within. A closed-form
// weight depending ONLY on the turn times (never on x), so it needs no AAD (docs/turns-calibration.md §2).
inline double turn_overlap(double t, const Turn& w) {
  return std::max(0.0, std::min(t, w.end) - w.start);
}

struct CurveStructure {
  int base = -1;                // -1 = outright; else this curve = curves[base] + spread (spread knots)
  // Engine-BLIND currency tag (multi-currency). The kernel -- build_bundle_curves, residuals, the
  // W-cache -- NEVER reads it; it exists only so a BUILDER can resolve a per-index default discount
  // curve and FX conversion at construction time (CLAUDE.md §1: the engine names no currency). Default
  // 0 keeps every single-currency bundle byte-identical.
  int currency = 0;
  // Interpolation regions (each a scheme + its knot times) -- the ONE representation of this curve's
  // interpolation. Exactly one curve type is built from modules(); region knots are the free forwards,
  // region by region, in this order. (The legacy front/back Flat+Hermite pair is now just the two-region
  // layout `flat_hermite(meeting, back)` a builder puts here -- see curve/curve_module.hpp.)
  std::vector<curve::CurveModule> regions;

  // Calibration TURNS (docs/turns-calibration.md, Mode 2). Each turn appends ONE free overlay state
  // variable δ to this curve's state block -- AFTER its interpolation knots -- with a closed-form
  // `overlap` weight column in W. A turn is NOT an interpolation knot: it does not appear in modules()
  // and is EXCLUDED from the curvature regulariser (its columns sit outside the penalised interp block).
  std::vector<Turn> turns;

  // The ONE description of this curve's interpolation. Turns are an overlay, not a region, so they
  // never appear here.
  std::vector<curve::CurveModule> modules() const { return regions; }

  // The INTERPOLATION-knot count: the size of the piece of this curve's state that feeds the region
  // interpolation (the region knots). This is what integral_weight_matrix, the region math and the
  // curvature regulariser operate on -- turns are NOT interpolation knots.
  int n_interp_knots() const {
    int n = 0;
    for (const auto& r : regions) n += static_cast<int>(r.knots.size());
    return n;
  }

  // The STATE-block size: interpolation knots PLUS one δ per turn. The turn δ's live at the END of the
  // block, at local indices [n_interp_knots(), n_knots()). This is the size used for state offsets,
  // stacked-x layout, the Jacobian width and AAD block sizing -- everything EXCEPT the interpolation
  // itself (which uses n_interp_knots()).
  int n_knots() const { return n_interp_knots() + static_cast<int>(turns.size()); }
};

// Convenience: a CurveStructure whose interpolation is the shipped Flat(meeting)+Hermite(back) two-region
// layout. This is the successor to the old `{meeting, back, base}` aggregate init -- a builder/test that
// wants that default layout writes `flat_hermite_curve(meeting, back, base)` instead of hand-listing regions.
inline CurveStructure flat_hermite_curve(const std::vector<double>& meeting, const std::vector<double>& back,
                                         int base = -1, int currency = 0) {
  return {.base = base, .currency = currency, .regions = curve::flat_hermite(meeting, back)};
}

}  // namespace swaps::pricing
