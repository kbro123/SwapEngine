#pragma once
// Exposes our TwoRegionForwardCurve to QuantLib as a YieldTermStructure.
//
// This is a CORE ENGINE COMPONENT, not just a test tool (CLAUDE.md §1): our calibrated curve is a
// QuantLib::YieldTermStructure, so it plugs straight into QuantLib pricing engines. QuantLib
// supplies the instrument machinery it is authoritative for — SOFR arithmetic-average vs
// compounded accrual, IMM schedules, business-day and day-count conventions — while every discount
// factor comes from our two-region interpolator.
//
// It is also how we validate: QuantLib pricing off this curve is the oracle for our own templated
// pricing kernel, and any disagreement is our pricing bug, not an interpolation mismatch.

#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include "swaps/curve/two_region_forward_curve.hpp"

namespace swaps::qlx {

// Wrap ANY curve exposing `discount(double)` as a QuantLib::YieldTermStructure -- so our calibrated
// curve, WHATEVER its region composition (natural-cubic TwoRegionForwardCurve, the Hermite
// make_calibration_curve, or a runtime ModularCurve), plugs straight into QuantLib pricing engines.
// This is a CORE ENGINE COMPONENT, not just a test tool (CLAUDE.md §1): QuantLib supplies the
// instrument machinery it is authoritative for (SOFR average vs compounded accrual, IMM schedules,
// conventions) while every discount factor comes from our interpolator -- and QuantLib pricing off
// this structure is the oracle for our own templated kernel, whatever curve we deploy.
template <class Curve>
class CurveTermStructure : public QuantLib::YieldTermStructure {
 public:
  CurveTermStructure(const QuantLib::Date& referenceDate, const QuantLib::DayCounter& dc,
                     const Curve* curve)
      : QuantLib::YieldTermStructure(referenceDate, QuantLib::Calendar(), dc), curve_(curve) {}

  QuantLib::Date maxDate() const override { return QuantLib::Date::maxDate(); }

 protected:
  // The single point of contact. QuantLib derives zero rates, forwards and every instrument price
  // from this. `t` is a year fraction in this structure's day counter -- the same one used to build
  // the curve's knot times.
  QuantLib::DiscountFactor discountImpl(QuantLib::Time t) const override { return curve_->discount(t); }

 private:
  const Curve* curve_;
};

// Back-compat: the original natural-cubic wrapper is one instantiation.
using TwoRegionTermStructure = CurveTermStructure<curve::TwoRegionForwardCurve<double>>;

}  // namespace swaps::qlx
