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

class TwoRegionTermStructure : public QuantLib::YieldTermStructure {
 public:
  using Curve = curve::TwoRegionForwardCurve<double>;

  TwoRegionTermStructure(const QuantLib::Date& referenceDate, const QuantLib::DayCounter& dc,
                         const Curve* curve)
      : QuantLib::YieldTermStructure(referenceDate, QuantLib::Calendar(), dc), curve_(curve) {}

  QuantLib::Date maxDate() const override { return QuantLib::Date::maxDate(); }

 protected:
  // The single point of contact. QuantLib derives zero rates, forwards and every instrument
  // price from this. `t` is a year fraction measured with this structure's day counter, which
  // must be the same one used to build the curve's knot times.
  QuantLib::DiscountFactor discountImpl(QuantLib::Time t) const override {
    return curve_->discount(t);
  }

 private:
  const Curve* curve_;
};

}  // namespace swaps::qlx
