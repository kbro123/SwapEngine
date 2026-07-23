#pragma once
// ModularCurve<Scalar> -- THE forward curve. Build it from CurveModule pieces, each a set of knot times
// + an interpolation scheme over those knots (Flat / Linear / NaturalCubic / Hermite / MonotoneCubic /
// BSpline from regions.hpp), stitched left-to-right with a C0 (level) Boundary handoff.
//
// There is exactly ONE curve type and ONE factory (make_modular_curve). A "curve flavour" is DATA -- a
// list of modules -- not a type: the shipped flat-front/Hermite-back calibration curve, a B-spline back
// end and a monotone-cubic back end are just three different module lists (see the layout helpers
// below). That keeps every curve, whether shipped or user-composed in the web composer, on one code
// path: one set of region math, one linearity check, one W-cache, one risk path.
//
// Region types are erased, so composing a curve costs a virtual call per region per evaluation. That is
// setup-only cost: the microsecond calibration hot path runs on the cached W matrix (CompiledResidual),
// never on a curve object. Measured against the compile-time alternative it was not slower --
// risk_full_jacobian 662us vs 809us -- because AAD gradient allocation dominates the dispatch entirely.
// Templated on Scalar so AAD flows through (e.g. to build W or a risk gradient).

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/curve/regions.hpp"

namespace swaps::curve {

// Type-erased region: the runtime face of a compile-time region policy (Flat<S>, Hermite<S>, ...).
template <class S>
struct RegionIface {
  virtual ~RegionIface() = default;
  virtual int n_values() const = 0;
  virtual double t_end() const = 0;
  virtual bool linear() const = 0;
  virtual void build(const S* x, int off, const Boundary<S>& in) = 0;
  virtual S forward(double t) const = 0;
  virtual S integral(double t) const = 0;
  virtual Boundary<S> out() const = 0;
};

template <class S, template <class> class Policy>
struct RegionHolder final : RegionIface<S> {
  Policy<S> p;
  explicit RegionHolder(std::vector<double> knots) : p(std::move(knots)) {}
  int n_values() const override { return p.n_values(); }
  double t_end() const override { return p.t_end(); }
  bool linear() const override { return Policy<S>::is_linear_map; }
  // Policy::build indexes x[off+k]; a raw pointer satisfies that (pointer indexing), no copy of policy.
  void build(const S* x, int off, const Boundary<S>& in) override { p.build(x, off, p.n_values(), in); }
  S forward(double t) const override { return p.forward(t); }
  S integral(double t) const override { return p.integral(t); }
  Boundary<S> out() const override { return p.out(); }
};

// MonotoneCubic is the one NON-LINEAR scheme here: RegionHolder<S,MonotoneCubic>::linear() returns false,
// so a ModularCurve containing it reports is_linear_map()==false -- the runtime signal to route that curve
// through the AAD engine rather than the W-cache (which its static_assert would reject at compile time).
// BSpline is linear too, but note its free values are CONTROL POINTS, which do not lie on the curve
// (docs/bezier-and-moments.md Part A) -- calibrated x are control points, mapped to forward-at-knot for
// a risk ladder by pricing::bspline_collocation. That is a property of the REGION, not of a curve type.
enum class Scheme { Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline };

// One building block of a curve: the knot times of a region and the interpolation over them.
struct CurveModule {
  std::vector<double> knots;  // knot times (year fractions), ascending, within this region
  Scheme scheme;              // interpolation scheme over those knots
};

template <class S>
class ModularCurve {
 public:
  ModularCurve() = default;

  template <template <class> class Policy>
  ModularCurve& add(std::vector<double> knots) {
    regions_.push_back(std::make_unique<RegionHolder<S, Policy>>(std::move(knots)));
    n_ += regions_.back()->n_values();
    return *this;
  }
  ModularCurve& add(const CurveModule& m) {
    switch (m.scheme) {
      case Scheme::Flat: return add<Flat>(m.knots);
      case Scheme::Linear: return add<Linear>(m.knots);
      case Scheme::NaturalCubic: return add<NaturalCubic>(m.knots);
      case Scheme::Hermite: return add<Hermite>(m.knots);
      case Scheme::MonotoneCubic: return add<MonotoneCubic>(m.knots);
      case Scheme::BSpline: return add<BSpline>(m.knots);
    }
    return *this;
  }

  int n_knots() const { return n_; }
  int n_regions() const { return static_cast<int>(regions_.size()); }
  double max_time() const { return regions_.back()->t_end(); }
  bool is_linear_map() const {
    for (const auto& r : regions_)
      if (!r->linear()) return false;
    return true;
  }

  // x = all knot forwards, region by region; stitched left-to-right with a C0 boundary handoff.
  template <class Vec>
  void set_forwards(const Vec& x) {
    if (static_cast<int>(x.size()) != n_) throw std::invalid_argument("set_forwards: wrong size");
    std::vector<S> buf(n_);  // contiguous scalar buffer so each region can index x[off+k]
    for (int i = 0; i < n_; ++i) buf[i] = x[i];
    Boundary<S> b{};
    int off = 0;
    for (auto& r : regions_) {
      r->build(buf.data(), off, b);
      off += r->n_values();
      b = r->out();
    }
  }

  S forward(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward(u); });
  }
  S integral(double t) const {
    if (t <= 0.0) return S(0.0);
    return locate(t, [](const RegionIface<S>& r, double u) { return r.integral(u); });
  }
  S discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }
  S zero(double t) const { return t <= 0.0 ? forward(0.0) : integral(t) / t; }

 private:
  template <class Fn>
  S locate(double t, Fn fn) const {
    for (const auto& r : regions_)
      if (t <= r->t_end()) return fn(*r, t);
    return fn(*regions_.back(), t);  // beyond the last region -> extrapolate
  }
  std::vector<std::unique_ptr<RegionIface<S>>> regions_;
  int n_ = 0;
};

// Cross-region join check, for ANY layout: each region must start strictly after the previous one ends.
// (Per-region strictly-increasing knots are enforced in the region ctors.) A zero-length first segment
// in a spline region produces NaN rather than an error, so this is checked up front -- it applies to
// user-composed region lists from the web composer exactly as it does to the shipped layouts.
inline void check_region_joins(const std::vector<CurveModule>& modules) {
  for (std::size_t i = 1; i < modules.size(); ++i) {
    const auto& prev = modules[i - 1].knots;
    const auto& cur = modules[i].knots;
    if (prev.empty() || cur.empty()) continue;
    if (!(cur.front() > prev.back()))
      throw std::invalid_argument("make_modular_curve: region " + std::to_string(i) + " starts at " +
                                  std::to_string(cur.front()) + ", which must exceed the previous region's "
                                  "last knot (" + std::to_string(prev.back()) + ")");
  }
}

// THE curve factory. Build a curve from a list of (knots, scheme) modules -- shipped layout or bespoke.
template <class S>
ModularCurve<S> make_modular_curve(const std::vector<CurveModule>& modules) {
  check_region_joins(modules);
  ModularCurve<S> c;
  for (const auto& m : modules) c.add(m);
  return c;
}

// ---- Named layouts -------------------------------------------------------------------------------
// A curve "flavour" is a module list, not a type. These are the shipped ones; anything else is just a
// different list. Empty regions are dropped so a knots-only-in-the-back curve is a one-region layout.

inline std::vector<CurveModule> two_region_layout(const std::vector<double>& front,
                                                  const std::vector<double>& back, Scheme back_scheme) {
  std::vector<CurveModule> m;
  if (!front.empty()) m.push_back({front, Scheme::Flat});
  if (!back.empty()) m.push_back({back, back_scheme});
  return m;
}

// The SHIPPED calibration layout: piecewise-flat meeting-date front + LOCAL C1 Hermite back.
//
// Hermite (vs a global natural cubic) keeps calibrated forwards local and non-oscillating -- a par swap
// constrains an INTEGRAL, so a global spline lets point forwards swing wildly while the rates still
// match; the local Hermite does not. It stays a LINEAR MAP of the knot values, so the W-cache /
// analytic-Jacobian / microsecond warm-recal fast path is preserved (CLAUDE.md §2). Validated
// end-to-end against QuantLib via the YieldTermStructure oracle.
inline std::vector<CurveModule> flat_hermite(const std::vector<double>& meeting,
                                             const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::Hermite);
}

// Control-point cubic B-spline back end (C2, convex-hull, positivity-friendly). Linear in the control
// points, so still W-cacheable -- but the free variables are control points (see Scheme::BSpline).
inline std::vector<CurveModule> flat_bspline(const std::vector<double>& meeting,
                                             const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::BSpline);
}

// Hyman-filtered MONOTONE cubic back end (matches QuantLib MonotonicCubicNaturalSpline). The filter is
// VALUE-dependent, so this layout reports is_linear_map()==false and calibrates on the AAD tier rather
// than the W-cache -- it is the concrete layout that exercises that fallback (CLAUDE.md §2).
inline std::vector<CurveModule> flat_monotone(const std::vector<double>& meeting,
                                              const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::MonotoneCubic);
}

}  // namespace swaps::curve
