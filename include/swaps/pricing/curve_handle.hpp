#pragma once
// CurveHandle<Scalar> and its family — type-erased ACCESS to a bundle's curves (outright / spread / turned),
// plus the two builders that materialise a whole bundle's handles from a state vector.
//
// This is pricing, not calibration: nothing here mentions an Instrument, a residual or a solver. It reads a
// `CurveStructure` (curve_spec.hpp) and a value-supplier, and hands back objects the pricing kernel can call
// `discount()` / `forward()` / `integral()` on. It lived in calibration/bundle_problem.hpp until E6.4
// (2026-09-10), which forced the portfolio layer to include the calibration layer to price a book off a
// bundle — closing a cycle with calibration/pnl_explain.hpp's dependency on portfolio. The names are
// re-exported into `swaps::calibration`, so `cal::CurveHandle` still spells the same type.
#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "swaps/curve/curve_module.hpp"  // runtime ModularCurve for user-defined interpolation regions
#include "swaps/pricing/cashflows.hpp"   // Turn / turn_overlap
#include "swaps/pricing/curve_spec.hpp"  // CurveStructure — the curve definition these handles read

namespace swaps::pricing {

// Type-erased curve so a bundle can hold heterogeneous curves (outright / spread) and price through one
// kernel. Scalar-templated so AAD flows through the virtual calls.
template <class S>
struct CurveHandle {
  virtual ~CurveHandle() = default;
  virtual S forward(double t) const = 0;
  virtual S integral(double t) const = 0;
  virtual S discount(double t) const = 0;
  // Overwrite this curve's knot forwards IN PLACE (the structure is fixed; only the values change). Lets a
  // streaming caller reuse one handle across ticks instead of rebuilding the object -- see BundleCurveSet.
  virtual void set_forwards(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x) = 0;
  // The jump size δⱼ of this curve's j-th turn (docs/turns-calibration.md). Only a TurnedCurve overrides
  // it; every other handle has no turns. It is the model quote of a TurnJump calibration instrument.
  virtual S turn_jump(int) const { throw std::logic_error("turn_jump: this curve has no turns"); }
  // Append the breakpoints between which forward(t) is ONE analytic piece (region knots, de Boor
  // breakpoints, turn edges, the base chain's pieces): the moment path's knot-aligned quadrature splits
  // its windows there (cashflows.hpp moment_gauss_nodes). Unknown by default (an empty append).
  virtual void pieces_into(std::vector<double>&) const {}
};
template <class S>
struct OutrightHandle : CurveHandle<S> {
  curve::ModularCurve<S> c;
  explicit OutrightHandle(curve::ModularCurve<S> cc) : c(std::move(cc)) {}
  S forward(double t) const override { return c.forward(t); }
  S integral(double t) const override { return c.integral(t); }
  S discount(double t) const override { return c.discount(t); }
  void set_forwards(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x) override { c.set_forwards(x); }
  void pieces_into(std::vector<double>& out) const override { const auto p = c.pieces(); out.insert(out.end(), p.begin(), p.end()); }
};
// forward = base + spread ; integral = base + spread ; DF = base_DF * exp(-int spread).
template <class S>
struct SpreadHandle : CurveHandle<S> {
  curve::ModularCurve<S> spread;
  const CurveHandle<S>* base;
  SpreadHandle(curve::ModularCurve<S> sp, const CurveHandle<S>* b) : spread(std::move(sp)), base(b) {}
  S forward(double t) const override { return base->forward(t) + spread.forward(t); }
  S integral(double t) const override { return base->integral(t) + spread.integral(t); }
  S discount(double t) const override {
    using std::exp;
    return exp(-integral(t));
  }
  void set_forwards(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x) override { spread.set_forwards(x); }
  void pieces_into(std::vector<double>& out) const override {
    base->pieces_into(out);
    const auto p = spread.pieces();
    out.insert(out.end(), p.begin(), p.end());
  }
};
// TURN OVERLAY adapter (docs/turns-calibration.md §3, the templated/AAD/QuantLib-oracle path). Wraps a
// built curve (outright or spread) and adds each turn's jump δⱼ over its window [aⱼ,bⱼ]:
//     forward(t)  = base_forward(t)  + Σⱼ δⱼ·1_{[aⱼ,bⱼ]}(t)          (a flat bump inside the window)
//     integral(t) = base_integral(t) + Σⱼ δⱼ·overlap(t, [aⱼ,bⱼ])     (matches the compiled W overlap col)
//     discount(t) = base_discount(t)·exp(−Σⱼ δⱼ·overlap(t, [aⱼ,bⱼ]))
// This is the closed-form analogue of SpreadHandle's additive overlay, and it keeps the templated path
// bit-consistent with the compiled W-cache (which fills the SAME overlap columns). A dependent SPREAD
// curve whose base is a TurnedCurve observes the base's turns automatically -- SpreadHandle recurses into
// base->integral/forward, which already include the base overlay.
template <class S>
struct TurnedCurve : CurveHandle<S> {
  std::unique_ptr<CurveHandle<S>> base;  // the underlying outright/spread curve
  std::vector<Turn> windows;    // turn accrual windows (year fractions)
  Eigen::Matrix<S, Eigen::Dynamic, 1> deltas;  // δⱼ, the free overlay state variables
  int n_interp = 0;                      // size of `base`'s own state (the interp-knot count)

  TurnedCurve(std::unique_ptr<CurveHandle<S>> b, std::vector<Turn> w, int ni)
      : base(std::move(b)), windows(std::move(w)), n_interp(ni) {
    deltas.setZero(static_cast<int>(windows.size()));
  }
  S forward(double t) const override {
    S f = base->forward(t);
    for (std::size_t j = 0; j < windows.size(); ++j)
      if (t >= windows[j].start && t < windows[j].end) f += deltas[static_cast<int>(j)];
    return f;
  }
  S integral(double t) const override {
    S I = base->integral(t);
    for (std::size_t j = 0; j < windows.size(); ++j)
      I += deltas[static_cast<int>(j)] * S(turn_overlap(t, windows[j]));
    return I;
  }
  S discount(double t) const override {
    using std::exp;
    return exp(-integral(t));
  }
  // The stacked state is [ interp knots | δ's ]: the first n_interp go to the base curve, the rest are
  // the turn jumps. Matches the state layout CurveStructure::n_knots() / build_bundle_curves lay out.
  void set_forwards(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x) override {
    base->set_forwards(x.head(n_interp));
    for (int j = 0; j < deltas.size(); ++j) deltas[j] = x[n_interp + j];
  }
  S turn_jump(int j) const override { return deltas[j]; }
  void pieces_into(std::vector<double>& out) const override {
    base->pieces_into(out);
    for (const auto& w : windows) { out.push_back(w.start); out.push_back(w.end); }  // the forward steps there
  }
};

// Build every bundle curve as a handle. `value(c, i)` supplies curve c's local knot i (free from the
// parameter vector, or a frozen constant). Requires base < c so each spread's base is already built.
template <class Scalar, class ValueFn>
std::vector<std::unique_ptr<CurveHandle<Scalar>>> build_bundle_curves(
    const std::vector<CurveStructure>& specs, ValueFn value) {
  std::vector<std::unique_ptr<CurveHandle<Scalar>>> C(specs.size());
  for (int c = 0; c < static_cast<int>(specs.size()); ++c) {
    const auto& spec = specs[c];
    if (spec.base >= c) throw std::invalid_argument("bundle curve: base index must be < curve index");
    const int ni = spec.n_interp_knots();  // the first ni state entries feed the interpolation
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> xi(ni);
    for (int i = 0; i < ni; ++i) xi[i] = value(c, i);
    auto inner = curve::make_modular_curve<Scalar>(spec.modules());
    inner.set_forwards(xi);
    std::unique_ptr<CurveHandle<Scalar>> h;
    if (spec.base < 0)
      h = std::make_unique<OutrightHandle<Scalar>>(std::move(inner));
    else
      h = std::make_unique<SpreadHandle<Scalar>>(std::move(inner), C[spec.base].get());
    // Turns overlay this curve's own forwards (and, via the base recursion, any dependent curve's).
    // The δ's are the LAST turns.size() entries of the state block, after the ni interp knots.
    if (!spec.turns.empty()) {
      auto turned = std::make_unique<TurnedCurve<Scalar>>(std::move(h), spec.turns, ni);
      for (int j = 0; j < static_cast<int>(spec.turns.size()); ++j)
        turned->deltas[j] = value(c, ni + j);
      h = std::move(turned);
    }
    C[c] = std::move(h);
  }
  return C;
}

// A reusable set of bundle curves for a FIXED structure. Build the handles ONCE, then `update()` their
// knot forwards in place each tick -- no object reconstruction, no per-tick handle allocation. This is the
// streaming form of build_bundle_curves: the curve topology (knot times, schemes, spread graph) never
// changes tick to tick, only the values do, so the objects (and their per-curve knot buffers) are reused.
template <class Scalar>
class BundleCurveSet {
 public:
  BundleCurveSet() = default;
  // Build the handles once with zero forwards. `specs` must outlive this object (it is not copied).
  void build(const std::vector<CurveStructure>& specs) {
    handles_ = build_bundle_curves<Scalar>(specs, [](int, int) { return Scalar(0.0); });
    scratch_.resize(specs.size());
    for (std::size_t c = 0; c < specs.size(); ++c) scratch_[c].resize(specs[c].n_knots());
  }
  // Overwrite every curve's forwards from value(c, i), reusing the per-curve scratch buffers (no alloc).
  template <class ValueFn>
  void update(ValueFn value) {
    for (int c = 0; c < static_cast<int>(handles_.size()); ++c) {
      auto& xi = scratch_[c];
      for (int i = 0; i < static_cast<int>(xi.size()); ++i) xi[i] = value(c, i);
      handles_[c]->set_forwards(xi);
    }
  }
  const CurveHandle<Scalar>& operator[](int i) const { return *handles_[i]; }
  bool empty() const { return handles_.empty(); }

 private:
  std::vector<std::unique_ptr<CurveHandle<Scalar>>> handles_;
  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> scratch_;  // reused per-curve knot buffers
};


// THE LINEAR HORIZON of a curve: the largest time up to which its log-discount integral is a LINEAR map of
// the state, so a constant W row exists for it (2026-09-10).
//
// Regions are built front to back, each from its own knots plus the boundary handed in by its predecessor,
// and none feeds anything backwards. So a value-dependent region cannot invalidate any time before it: the
// horizon is the last knot of the maximal LINEAR PREFIX of regions, or −inf if the very first region is
// non-linear (a leading region flat-extrapolates its first value, so even earlier times use it).
//
// A spread curve's weight row is its own PLUS its base's, evaluated at the SAME time (see logdf_weight), so
// each term must be linear at that time independently. The base therefore only has to be linear up to t as
// well, which makes the horizon a simple recursive minimum. It replaces the old whole-curve-set veto, under
// which one MonotoneCubic region anywhere dropped every instrument in the bundle to the AAD path.
inline double curve_linear_horizon(const std::vector<CurveStructure>& curves, int c) {
  double h = std::numeric_limits<double>::infinity();
  const auto& regions = curves[static_cast<std::size_t>(c)].regions;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (curve::scheme_is_linear(regions[i].scheme)) continue;
    h = (i == 0 || regions[i - 1].knots.empty()) ? -std::numeric_limits<double>::infinity()
                                                 : regions[i - 1].knots.back();
    break;
  }
  const int base = curves[static_cast<std::size_t>(c)].base;
  return base >= 0 ? std::min(h, curve_linear_horizon(curves, base)) : h;
}

// Every curve's horizon, indexed by curve role.
inline std::vector<double> curve_linear_horizons(const std::vector<CurveStructure>& curves) {
  std::vector<double> h(curves.size());
  for (std::size_t c = 0; c < curves.size(); ++c) h[c] = curve_linear_horizon(curves, static_cast<int>(c));
  return h;
}

// True iff NO time on this curve set is W-cacheable (every curve's horizon is −inf). Retained for the
// callers that only need the all-or-nothing answer.
inline bool curves_are_noncacheable(const std::vector<CurveStructure>& curves) {
  for (const auto& c : curves)
    for (const auto& r : c.regions)
      if (!curve::scheme_is_linear(r.scheme)) return true;
  return false;
}

}  // namespace swaps::pricing
