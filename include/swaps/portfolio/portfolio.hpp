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
  struct Position {
    pricing::OisSwap sched;
    double fixed_rate;
    double notional;
  };
  std::vector<Position> positions;

  // Total NPV. AAD-safe: seed the accumulator from the first position (carries derivatives).
  template <class Scalar, class Curve>
  Scalar npv(const Curve& c) const {
    assert(!positions.empty());
    Scalar total =
        positions[0].notional * pricing::ois_swap_npv<Scalar>(positions[0].sched, positions[0].fixed_rate, c);
    for (std::size_t i = 1; i < positions.size(); ++i)
      total += positions[i].notional * pricing::ois_swap_npv<Scalar>(positions[i].sched, positions[i].fixed_rate, c);
    return total;
  }
};

}  // namespace swaps::portfolio
