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
#include <algorithm>
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
  virtual S forward_d1(double t) const = 0;           // exact df/dt (0 outside / at a Flat step)
  virtual S forward_d2(double t) const = 0;           // exact d²f/dt²
  virtual std::vector<double> pieces() const = 0;     // breakpoints between which f is ONE analytic piece
  virtual double tension_sigma() const = 0;           // Scheme::Tension's σ (0 for every other scheme)
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
  S forward_d1(double t) const override { return p.forward_d1(t); }
  S forward_d2(double t) const override { return p.forward_d2(t); }
  std::vector<double> pieces() const override { return p.pieces(); }
  double tension_sigma() const override { return 0.0; }
  Boundary<S> out() const override { return p.out(); }
};

// Tension needs a second construction argument (σ), so it gets its own holder rather than the generic
// RegionHolder (which passes only knots). Everything else about the contract is identical.
template <class S>
struct TensionHolder final : RegionIface<S> {
  Tension<S> p;
  TensionHolder(std::vector<double> knots, double sigma) : p(std::move(knots), sigma) {}
  int n_values() const override { return p.n_values(); }
  double t_end() const override { return p.t_end(); }
  bool linear() const override { return Tension<S>::is_linear_map; }
  void build(const S* x, int off, const Boundary<S>& in) override { p.build(x, off, p.n_values(), in); }
  S forward(double t) const override { return p.forward(t); }
  S integral(double t) const override { return p.integral(t); }
  S forward_d1(double t) const override { return p.forward_d1(t); }
  S forward_d2(double t) const override { return p.forward_d2(t); }
  std::vector<double> pieces() const override { return p.pieces(); }
  double tension_sigma() const override { return p.sigma(); }
  Boundary<S> out() const override { return p.out(); }
};

// MonotoneCubic is the one NON-LINEAR scheme here: RegionHolder<S,MonotoneCubic>::linear() returns false,
// so a ModularCurve containing it reports is_linear_map()==false -- the runtime signal to route that curve
// through the AAD engine rather than the W-cache (which its static_assert would reject at compile time).
// BSpline is linear too, but note its free values are CONTROL POINTS, which do not lie on the curve
// (docs/bezier-and-moments.md Part A) -- calibrated x are control points, mapped to forward-at-knot for
// a risk ladder by pricing::bspline_collocation. That is a property of the REGION, not of a curve type.
// Tension is the second LINEAR hyperbolic scheme (research note §3): with σ a FIXED hyperparameter its
// coefficients depend only on knot spacings, so is_linear_map = true -- it rides the W-cache exactly
// like NaturalCubic/BSpline, unlike the value-dependent MonotoneCubic. σ travels in CurveModule::sigma.
enum class Scheme { Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline, Tension };

// One building block of a curve: the knot times of a region and the interpolation over them.
struct CurveModule {
  std::vector<double> knots;  // knot times (year fractions), ascending, within this region
  Scheme scheme;              // interpolation scheme over those knots
  // Tension hyperparameter (Scheme::Tension only): pulls the spline taut, σ→0 recovers NaturalCubic,
  // σ→∞ approaches piecewise-linear. Ignored by every other scheme. <=0 means "use the default 1.0".
  double sigma = 0.0;
  // PER-REGION smoothing weight (Phase 1): the curvature/tension-energy penalty strength for THIS region's
  // knots. <0 (the default) means "inherit the bundle default λ" — so a curve whose regions never set this
  // is penalised exactly as before (one global λ). Set it to give the meeting-date front λ=0 while the long
  // end is smoothed, or two regions different tension. The regulariser reads it per region.
  double reg_lambda = -1.0;
  // PER-REGION tension-energy membrane σ (Phase 2): the σ in the (bending + σ²·membrane) tension-energy
  // penalty, applied to THIS region's intervals. <0 (the default) inherits the bundle default σ. Distinct
  // from `sigma` above (that is the INTERPOLATION tension of a Scheme::Tension region); this is the
  // REGULARISER's σ, and it may vary per region while the interpolation stays whatever the scheme is.
  double reg_sigma = -1.0;
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
      case Scheme::Tension: {
        regions_.push_back(
            std::make_unique<TensionHolder<S>>(m.knots, m.sigma > 0.0 ? m.sigma : 1.0));
        n_ += regions_.back()->n_values();
        return *this;
      }
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
  // buf_ is a REUSED member (sized once): on a streaming re-solve the same curve is set_forwards'd every
  // tick, so allocating a fresh contiguous buffer each call is pure churn. `resize` keeps capacity, so
  // after the first call this never reallocates.
  template <class Vec>
  void set_forwards(const Vec& x) {
    if (static_cast<int>(x.size()) != n_) throw std::invalid_argument("set_forwards: wrong size");
    buf_.resize(n_);  // contiguous scalar buffer so each region can index x[off+k]; reused across calls
    for (int i = 0; i < n_; ++i) buf_[i] = x[i];
    Boundary<S> b{};  // has_predecessor=false -> the FIRST region is LEADING (flat-extrapolates its 1st knot)
    int off = 0;
    for (auto& r : regions_) {
      r->build(buf_.data(), off, b);
      off += r->n_values();
      b = r->out();
      b.has_predecessor = true;  // every region after the first FOLLOWS its predecessor (C0 join, unchanged)
    }
  }

  S forward(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward(u); });
  }
  // Exact derivatives of the forward and the analytic-piece breakpoints of the whole composed curve (the
  // regulariser integrates (f'')² + σ²(f')² piece by piece; each piece is one polynomial or one tension piece).
  S forward_d1(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward_d1(u); });
  }
  S forward_d2(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward_d2(u); });
  }
  // Requires set_forwards() to have run at least once: a region's pieces include its C0 join with the predecessor
  // and (B-spline) its de Boor breakpoints, both fixed at build time. An unbuilt curve throws.
  std::vector<double> pieces() const {
    if (static_cast<int>(buf_.size()) != n_) throw std::logic_error("ModularCurve::pieces(): call set_forwards() first (pieces are fixed at build)");
    std::vector<double> bp;
    for (const auto& r : regions_) { const auto pr = r->pieces(); bp.insert(bp.end(), pr.begin(), pr.end()); }
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end(), [](double a, double b) { return std::abs(a - b) <= 1e-13 * (1.0 + std::abs(a)); }), bp.end());
    return bp;
  }
  // The interpolation tension σ of the region containing t (0 for a non-Tension region): sizes the quadrature.
  double tension_sigma_at(double t) const {
    return locate(t, [](const RegionIface<S>& r, double) { return r.tension_sigma(); });
  }
  // CONVENTION (E3 register D11): integral(t <= 0) == 0, i.e. discount(t) == 1 for ANY non-positive time.
  // Nothing in pricing may rely on a negative-time discount factor: a seasoned MtM reset in the past must
  // carry its fixed FX (cashflows.hpp xccy_mtm_leg_pv refuses one that does not), and the P&L roll floors
  // its shifted times at 0 by design. A negative time here is a stale-input smell, not a curve query.
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
  std::vector<S> buf_;  // reused set_forwards scratch (see set_forwards) — sized once, no per-call alloc
};

// Cross-region join check, for ANY layout: each region must start strictly after the previous one ends.
// (Per-region strictly-increasing knots are enforced in the region ctors.) A zero-length first segment
// in a spline region produces NaN rather than an error, so this is checked up front -- it applies to
// user-composed region lists from the web composer exactly as it does to the shipped layouts.
inline void check_region_joins(const std::vector<CurveModule>& modules) {
  // E3-E1/E2 (2026-09-10): a curve with NO knots used to build an empty ModularCurve (n = 0) and crash
  // downstream (run_json `"regions": []` segfaulted); a knot at t <= 0 built silently and corrupted the
  // integral / W (a negative weight, ~49 bp of log-DF) while calibration "succeeded". Both are refused here,
  // the one factory every path (templated, AAD block, compiled W) builds curves through.
  std::size_t n_knots = 0;
  for (const auto& m : modules) n_knots += m.knots.size();
  if (n_knots == 0) throw std::invalid_argument("make_modular_curve: a curve needs at least one region with knots (regions: [] is not a curve)");
  for (std::size_t i = 0; i < modules.size(); ++i) {
    if (modules[i].knots.empty()) continue;
    const double t0 = modules[i].knots.front();
    if (!(t0 > 0.0))
      throw std::invalid_argument("make_modular_curve: region " + std::to_string(i) + " has a knot at t = " + std::to_string(t0) +
                                  "; every knot time must be > 0 (times are year fractions from the evaluation date)");
  }
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

// Tension-spline back end (research note §3): a flat meeting-date front + a spline-under-tension back
// with fixed hyperparameter σ. Like flat_hermite/flat_bspline it stays a LINEAR MAP of the knot forwards
// (σ fixed), so it is W-cacheable and analytic-Jacobian ready; σ trades smoothness (σ→0 == the natural
// cubic) for tautness (large σ suppresses overshoot). Validated end-to-end against QuantLib.
inline std::vector<CurveModule> flat_tension(const std::vector<double>& meeting,
                                             const std::vector<double>& back, double sigma) {
  std::vector<CurveModule> m;
  if (!meeting.empty()) m.push_back({meeting, Scheme::Flat, 0.0});
  if (!back.empty()) m.push_back({back, Scheme::Tension, sigma});
  return m;
}

}  // namespace swaps::curve
