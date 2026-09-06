#pragma once
// curve/hazard.hpp — the CREDIT SURVIVAL (hazard-rate) CURVE: a Scalar-templated projection of an
// obligor's survival probability Q(t) built on top of THE curve object (curve_module.hpp). Additive and
// QuantLib-free. It layers over the curve machinery EXACTLY as curve/inflation.hpp does — read that first.
//
// THE ONE IDEA (mirrors inflation): the free, CALIBRATED object is a plain `ModularCurve` whose
// "instantaneous forward" f(u) is the FORWARD HAZARD RATE h(u) (piecewise-flat on a flat-front layout,
// smooth Hermite on the back). A ModularCurve's discount is exp(−∫f), which is precisely the survival
// probability:
//
//     Q(t) = exp( −∫₀ᵗ h(u) du ) = DF_hazard(t)
//
// so a survival curve needs NO new curve TYPE and NO new calibration machinery — the engine already
// calibrates a ModularCurve (LM + AAD + W-cache). `SurvivalCurve` is the thin projection layer that names
// the credit-specific accessors on top of that hazard curve:
//
//     survival(t)         = Q(t)            = hz.discount(t)     (Q(0)=1, monotone decreasing since h≥0)
//     hazard(t)           = h(t)            = hz.forward(t)      (the instantaneous default intensity)
//     default_density(t)  = −dQ/dt = h·Q   = hazard(t)·survival(t)
//
// Templated on Scalar so AAD flows straight through `hz->discount(t)` (the only curve-dependent term) to a
// risk gradient / calibration Jacobian, exactly like every other curve. It ALSO satisfies the generic
// curve contract — discount/forward/integral delegate to the underlying hazard ModularCurve — so a
// SurvivalCurve can itself be handed to a generic kernel that only needs `Scalar discount(double)`.

#include <cmath>

#include "swaps/curve/curve_module.hpp"

namespace swaps::curve {

template <class Scalar>
struct SurvivalCurve {
  const ModularCurve<Scalar>* hz = nullptr;  // the (calibrated) forward-hazard curve; discount == Q

  // Q(t) = exp(−∫₀ᵗ h) = DF of the hazard curve. Carries the AAD derivatives w.r.t. the hazard knots.
  Scalar survival(double t) const { return hz->discount(t); }
  // h(t): the instantaneous forward hazard (default intensity) at t.
  Scalar hazard(double t) const { return hz->forward(t); }
  // ln Q(t) = −∫₀ᵗ h. Additive form (the negated cumulative hazard) — handy where a log is wanted.
  Scalar log_survival(double t) const { return Scalar(0.0) - hz->integral(t); }
  // Default density −dQ/dt = h(t)·Q(t): the probability mass of default per unit time at t.
  Scalar default_density(double t) const { return hazard(t) * survival(t); }

  // ---- generic curve contract (delegates to the hazard curve) ------------------------------------
  Scalar discount(double t) const { return hz->discount(t); }
  Scalar forward(double t) const { return hz->forward(t); }
  Scalar integral(double t) const { return hz->integral(t); }
};

}  // namespace swaps::curve
