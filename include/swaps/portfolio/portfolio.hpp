#pragma once
// A swap portfolio and its NPV kernel.
//
// QuantLib-free and templated on Scalar: cashflow schedules are extracted from QuantLib once, then
// the portfolio NPV is a plain function of curve discount factors. With Scalar = double this prices;
// with Scalar = AutoDiffScalar it yields d(NPV)/d(knot forwards) in one pass (see calibration/risk).

#include <cassert>
#include <vector>

#include "swaps/pricing/cashflows.hpp"

namespace swaps::portfolio {

struct Portfolio {
  // A position is a swap in the GENERIC coupon model: a floating leg + a fixed leg (unit-notional
  // coupon shapes), a contract fixed rate and a notional. Single self-discounting curve, so the
  // floating leg forecasts and discounts off the same curve.
  struct Position {
    std::vector<pricing::FloatCoupon> float_coupons;
    std::vector<pricing::FixedCoupon> fixed_coupons;
    double fixed_rate;
    double notional;
  };
  std::vector<Position> positions;

  // NPV of one position per unit notional: float_leg_pv - fixed_rate * annuity (payer-of-fixed).
  template <class Scalar, class Curve>
  static Scalar position_npv(const Position& p, const Curve& c) {
    return pricing::float_leg_pv<Scalar>(p.float_coupons, c, c) -
           p.fixed_rate * pricing::annuity<Scalar>(p.fixed_coupons, c);
  }

  // Total NPV. AAD-safe: seed the accumulator from the first position (carries derivatives).
  template <class Scalar, class Curve>
  Scalar npv(const Curve& c) const {
    assert(!positions.empty());
    Scalar total = positions[0].notional * position_npv<Scalar>(positions[0], c);
    for (std::size_t i = 1; i < positions.size(); ++i)
      total += positions[i].notional * position_npv<Scalar>(positions[i], c);
    return total;
  }
};

}  // namespace swaps::portfolio
