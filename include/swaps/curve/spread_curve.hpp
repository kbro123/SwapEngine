#pragma once
// Forward-spread curve (CLAUDE.md §2): a curve defined as a spread to a fixed base curve.
//
//   forward(t)  = forward_base(t) + spread(t)
//   integral(t) = integral_base(t) + integral_spread(t)
//   DF(t)       = exp(-integral(t)) = DF_base(t) * exp(-integral_spread(t))
//
// The free variables are the SPREAD forwards at the spread knots; the base curve is held fixed
// (jointly-calibrated base is a later extension). The spread is interpolated with the SAME two-region
// scheme (piecewise-flat front, C2 spline back), so integral_spread(t) = w(t)·s is linear in the
// spread knots and the whole thing stays AAD-differentiable. The base contributes a constant additive
// integral (zero derivative w.r.t. s), so this reuses TwoRegionForwardCurve unchanged.
//
// Templated on Scalar: base is always `double` (fixed); the spread is `Scalar` so AAD flows through s.

#include <cmath>

#include "swaps/curve/two_region_forward_curve.hpp"

namespace swaps::curve {

template <class Scalar>
class SpreadCurve {
 public:
  SpreadCurve(const TwoRegionForwardCurve<double>& base, std::vector<double> meeting_times,
              std::vector<double> back_times)
      : base_(&base), spread_(std::move(meeting_times), std::move(back_times)) {}

  int n_knots() const { return spread_.n_knots(); }
  double join_time() const { return spread_.join_time(); }
  double max_time() const { return spread_.max_time(); }

  // Spread forwards at the knots (front spreads, then back spreads). May be negative.
  template <class Vec>
  void set_spreads(const Vec& s) {
    spread_.set_forwards(s);
  }

  Scalar forward(double t) const { return spread_.forward(t) + base_->forward(t); }

  // Scalar term (carries derivatives) leads; base double added second (AAD-safe).
  Scalar integral(double t) const { return spread_.integral(t) + base_->integral(t); }

  Scalar discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }

  Scalar zero(double t) const {
    if (t <= 0.0) return forward(0.0);
    return integral(t) / t;
  }

  const TwoRegionForwardCurve<Scalar>& spread_component() const { return spread_; }

 private:
  const TwoRegionForwardCurve<double>* base_;
  TwoRegionForwardCurve<Scalar> spread_;
};

}  // namespace swaps::curve
