#pragma once
// The interpolation used for CALIBRATION: piecewise-flat meeting-date front + LOCAL C1 Hermite back.
//
// Hermite (vs the global natural cubic) keeps the calibrated forwards local and non-oscillating -- a
// par-swap constrains an integral, so a global spline lets point forwards swing wildly while rates
// still match; the local Hermite does not. It remains a LINEAR MAP of the knot values, so the
// W-cache / analytic-Jacobian / microsecond warm-recal fast path is fully preserved (CLAUDE.md §2).
//
// (The curve-vs-QuantLib golden tests still validate the natural-cubic TwoRegionForwardCurve; this
// type is specifically the calibration/pricing curve.)

#include <vector>

#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"

namespace swaps::curve {

template <class S>
using CalibrationCurve = MultiRegionCurve<S, Flat, Hermite>;

template <class S>
inline CalibrationCurve<S> make_calibration_curve(const std::vector<double>& meeting,
                                                  const std::vector<double>& back) {
  return CalibrationCurve<S>{Flat<S>(meeting), Hermite<S>(back)};
}

}  // namespace swaps::curve
