#pragma once
// ModularCurve<Scalar> -- a RUNTIME-composable forward curve: build it up from CurveModule pieces,
// each a set of knot times + an interpolation scheme over those knots, stitched left-to-right with a
// C0 (level) Boundary handoff. It is the runtime sibling of the compile-time MultiRegionCurve: same
// region-policy math (Flat / Linear / NaturalCubic / Hermite from regions.hpp), but the region set is
// chosen at RUNTIME (type-erased), so ONE type composes any curve without a new template instantiation.
//
// Use it for flexible curve construction, the YieldTermStructure adapter and validation. It is NOT on
// the microsecond calibration hot path: that runs on the cached W matrix (CompiledResidual), never a
// curve object, so the (setup-only) virtual dispatch here never touches the fast path -- and the
// perf-critical calibration curve stays the compile-time make_calibration_curve. Templated on Scalar so
// AAD still flows through it (e.g. to build W or a risk gradient) when a runtime composition is used.

#include <cmath>
#include <memory>
#include <stdexcept>
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

enum class Scheme { Flat, Linear, NaturalCubic, Hermite };

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

// Build a runtime curve from a list of (knots, scheme) modules.
template <class S>
ModularCurve<S> make_modular_curve(const std::vector<CurveModule>& modules) {
  ModularCurve<S> c;
  for (const auto& m : modules) c.add(m);
  return c;
}

}  // namespace swaps::curve
