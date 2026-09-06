#pragma once
// curve/inflation.hpp — the INFLATION INDEX CURVE: a Scalar-templated projection of a price index level
// I(t) built on top of THE curve object (curve_module.hpp). It is additive and QuantLib-free.
//
// THE ONE IDEA (CLAUDE.md §1 "a curve is anything with discount(t) / an index projection"): the free,
// CALIBRATED object is a plain `ModularCurve` — the *breakeven-inflation* curve, whose "instantaneous
// forward" f_bei(u) is the continuously-compounded forward breakeven inflation rate. The engine already
// knows how to calibrate a ModularCurve (LM + AAD + W-cache), so an inflation curve needs NO new curve
// TYPE. `InflationIndexCurve` is the thin projection layer that turns that breakeven curve into the index
// level growth an inflation swap pays:
//
//     I(t) / I(0)  =  seasonality(t) · exp( ∫₀ᵗ f_bei(u) du )  =  seasonality(t) / DF_bei(t)
//
// because a ModularCurve's DF is exp(−∫f). So the log index growth is simply `bei.integral(t)` (which
// carries the AAD derivatives w.r.t. the knot forwards) plus a build-time-constant seasonal term.
//
// SEASONALITY (design: additive per-month factors) is modelled as a continuous multiplicative adjustment
// that ACCUMULATES over a year and returns to 1 at every integer-year point (Σ of the 12 monthly log
// increments == 0). That is the standard construction and it has the property the inflation desk relies
// on: a seasonal pattern moves the *monthly* index fixing but leaves an *annual* zero-coupon breakeven
// (whose maturity is a whole number of years) untouched — see tests/inflation_test.cpp.

#include <cmath>
#include <cstddef>
#include <vector>

#include "swaps/curve/curve_module.hpp"

namespace swaps::curve {

// Per-month seasonality: 12 multiplicative monthly factors expressed as log increments g_m. They are
// normalised so Σ g_m == 0, hence the cumulative adjustment S(t) below is continuous, periodic with a
// one-year period, and EXACTLY 1 at every integer-year point. `active == false` (or an empty schedule)
// is a pure no-op (S ≡ 1) that keeps a non-seasonal curve byte-identical.
struct Seasonality {
  std::vector<double> g;   // 12 monthly log increments (any real numbers); normalised on construction
  bool active = false;

  Seasonality() = default;
  // Build from 12 raw monthly log factors (e.g. estimated seasonal deviations). Recentred to sum zero so
  // the annual product is 1. A size other than 12 (or all-zero) leaves the curve non-seasonal.
  explicit Seasonality(std::vector<double> monthly) {
    if (monthly.size() != 12) return;
    double mean = 0.0;
    for (double v : monthly) mean += v;
    mean /= 12.0;
    g.resize(12);
    for (int m = 0; m < 12; ++m) g[m] = monthly[m] - mean;
    active = true;
  }

  // ln S(t): the cumulative seasonal log-adjustment from the year start to t. With y=⌊t⌋ and u=t−y in
  // [0,1), let p = 12u, m = ⌊p⌋, r = p−m: lnS = Σ_{k<m} g_k + r·g_m. lnS(integer) == 0 (empty prefix),
  // so it drops out of any whole-year zero-coupon breakeven. A plain double (build-time constant): added
  // to an AAD Scalar it preserves the derivatives.
  double log_factor(double t) const {
    if (!active || g.size() != 12) return 0.0;
    const double u = t - std::floor(t);
    double p = u * 12.0;
    int m = static_cast<int>(p);
    if (m < 0) m = 0;
    if (m > 11) m = 11;
    const double r = p - m;
    double s = 0.0;
    for (int k = 0; k < m; ++k) s += g[k];
    s += r * g[m];
    return s;
  }
  double factor(double t) const { return std::exp(log_factor(t)); }
};

// The inflation index curve. Templated on Scalar so AAD flows straight through `bei->integral(t)` (the
// only curve-dependent term) to a risk gradient / calibration Jacobian, exactly like every other curve.
//
// It ALSO satisfies the generic curve contract — `discount`/`forward`/`integral` delegate to the
// underlying breakeven ModularCurve — so an InflationIndexCurve can itself be handed to a generic kernel
// that only needs `Scalar discount(double)`. The inflation-specific accessors are `index` / `growth` /
// `zc_breakeven` / `yoy_forward`.
template <class Scalar>
struct InflationIndexCurve {
  double base = 100.0;                        // I(0), the base CPI level
  const ModularCurve<Scalar>* bei = nullptr;  // the (calibrated) breakeven forward-CPI curve
  const Seasonality* seas = nullptr;          // optional per-month seasonality (null => none)

  // ln( I(t)/I(0) ) = ∫₀ᵗ f_bei + ln S(t). The integral carries the AAD derivatives; the seasonal term
  // is a build-time constant double, added so the derivative vector is preserved (AAD-safe).
  Scalar log_growth(double t) const {
    Scalar g = bei->integral(t);
    if (seas) g = g + seas->log_factor(t);
    return g;
  }
  // I(t)/I(0) and the index level I(t).
  Scalar growth(double t) const {
    using std::exp;
    return exp(log_growth(t));
  }
  Scalar index(double t) const { return base * growth(t); }

  // Zero-coupon breakeven to maturity T, ANNUALLY compounded: the k with (1+k)^T = I(T)/I(0), i.e.
  // k = (I(T)/I(0))^(1/T) − 1 = exp( ln growth(T) / T ) − 1. Discount-curve independent (deterministic).
  Scalar zc_breakeven(double T) const {
    using std::exp;
    return exp(log_growth(T) / T) - 1.0;
  }
  // Year-on-year period return over [t0, t1]: I(t1)/I(t0) − 1 = exp( lnG(t1) − lnG(t0) ) − 1. This is the
  // full-period return (NOT annualised); a YoY coupon pays exactly this.
  Scalar yoy_forward(double t0, double t1) const {
    using std::exp;
    return exp(log_growth(t1) - log_growth(t0)) - 1.0;
  }

  // ---- generic curve contract (delegates to the breakeven curve) ---------------------------------
  Scalar discount(double t) const { return bei->discount(t); }
  Scalar forward(double t) const { return bei->forward(t); }
  Scalar integral(double t) const { return bei->integral(t); }
};

}  // namespace swaps::curve
