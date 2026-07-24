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

#include <vector>

#include "swaps/curve/curve_module.hpp"

namespace swaps::pricing {

struct CurveStructure {
  std::vector<double> meeting;  // front (flat) knot times   -- used only when `regions` is empty
  std::vector<double> back;     // back (Hermite) knot times  -- used only when `regions` is empty
  int base = -1;                // -1 = outright; else this curve = curves[base] + spread (spread knots)
  // Engine-BLIND currency tag (multi-currency). The kernel -- build_bundle_curves, residuals, the
  // W-cache -- NEVER reads it; it exists only so a BUILDER can resolve a per-index default discount
  // curve and FX conversion at construction time (CLAUDE.md §1: the engine names no currency). Default
  // 0 keeps every single-currency bundle byte-identical. Kept before `regions` so the existing positional
  // aggregate inits {meeting, back, base} and {meeting, back, base, currency, regions} stay valid.
  int currency = 0;
  // Interpolation regions (each a scheme + its knot times). When non-empty these define the curve and
  // `meeting`/`back` are ignored; when empty they mean the shipped layout: Flat(meeting) + Hermite(back).
  // Either way exactly one curve type is built from modules(). Region knots are the free forwards,
  // region by region, in this order.
  std::vector<curve::CurveModule> regions;

  // The ONE description of this curve's interpolation -- custom regions or the shipped default.
  std::vector<curve::CurveModule> modules() const {
    return regions.empty() ? curve::flat_hermite(meeting, back) : regions;
  }
  int n_knots() const {
    if (!regions.empty()) {
      int n = 0;
      for (const auto& r : regions) n += static_cast<int>(r.knots.size());
      return n;
    }
    return static_cast<int>(meeting.size() + back.size());
  }
};

}  // namespace swaps::pricing
