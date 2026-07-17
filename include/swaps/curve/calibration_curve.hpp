#pragma once
// The interpolation used for CALIBRATION: piecewise-flat meeting-date front + LOCAL C1 Hermite back.
//
// Hermite (vs the global natural cubic) keeps the calibrated forwards local and non-oscillating -- a
// par-swap constrains an integral, so a global spline lets point forwards swing wildly while rates
// still match; the local Hermite does not. It remains a LINEAR MAP of the knot values, so the
// W-cache / analytic-Jacobian / microsecond warm-recal fast path is fully preserved (CLAUDE.md §2).
//
// This is the shipped curve everywhere; it is validated end-to-end against QuantLib via the
// YieldTermStructure oracle (tests/pricing_test.cpp, tests/modular_curve_test.cpp).

#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"

namespace swaps::curve {

template <class S>
using CalibrationCurve = MultiRegionCurve<S, Flat, Hermite>;

template <class S>
inline CalibrationCurve<S> make_calibration_curve(const std::vector<double>& meeting,
                                                  const std::vector<double>& back) {
  // Per-region strictly-increasing knots are enforced in the Flat/Hermite ctors; here also enforce the
  // CROSS-region join: the first back knot must sit strictly after the last front knot (else the first
  // Hermite segment has zero length -> NaN).
  if (!meeting.empty() && !back.empty() && !(back.front() > meeting.back()))
    throw std::invalid_argument("make_calibration_curve: first back knot (" + std::to_string(back.front()) +
                                ") must exceed last front knot (" + std::to_string(meeting.back()) + ")");
  return CalibrationCurve<S>{Flat<S>(meeting), Hermite<S>(back)};
}

// Alternative back end: a control-point cubic B-spline (C2, convex-hull, positivity-friendly). Same
// flat meeting-date front; the back free variables are B-SPLINE CONTROL POINTS, not forward-at-knot
// (docs/bezier-and-moments.md Part A). Drop-in wherever a curve type is templated.
template <class S>
using BSplineCurve = MultiRegionCurve<S, Flat, BSpline>;

template <class S>
inline BSplineCurve<S> make_bspline_curve(const std::vector<double>& meeting,
                                          const std::vector<double>& back) {
  if (!meeting.empty() && !back.empty() && !(back.front() > meeting.back()))
    throw std::invalid_argument("make_bspline_curve: first back knot (" + std::to_string(back.front()) +
                                ") must exceed last front knot (" + std::to_string(meeting.back()) + ")");
  return BSplineCurve<S>{Flat<S>(meeting), BSpline<S>(back)};
}

}  // namespace swaps::curve
