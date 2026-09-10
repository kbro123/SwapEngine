#pragma once
// BundleProblem -- multiple curves calibrated SIMULTANEOUSLY over one stacked parameter vector
// x = [x_0 ; x_1 ; ...] (CLAUDE.md Stage 3). Each instrument references curves by index for its
// forecast/benchmark/discount roles; cross-curve instruments (basis swaps) couple their blocks.
//
// A curve's DEFINITION carries its parameterization, and the engine does the right thing automatically:
//   - Outright (base < 0): the free variables are this curve's own forwards.
//   - Spread  (base >= 0): the curve IS `curves[base] + spread`; the free variables are the SPREAD
//     knots, and it inherits the base's structure. If `base` is another FREE curve in the bundle, the
//     base and the spread are calibrated JOINTLY (the block Jacobian couples them via AAD through the
//     base's discount); if `base` is a curve you never pin, it is effectively a fixed base.
// Both are wrapped as a type-erased CurveHandle<Scalar> so the pricing kernel is oblivious. The
// virtual dispatch lands only in the calibration residual; the microsecond streaming path is on W.
//
// The key reuse: BundleProblem exposes the SAME interface as CalibrationProblem -- residuals<Scalar>(x),
// n_knots(), n_residuals() -- so calibrate / aad_jacobian / streaming / risk drive it UNCHANGED, and
// the block-structured Jacobian falls straight out of AAD over the stacked residual.

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>
#include <vector>

#include "swaps/ad/dual.hpp"               // ad::Dual, for the MtM funding-term numerical guard
#include "swaps/calibration/problem.hpp"  // the generic FloatLeg/FixedLeg/Instrument model (design §3)
#include "swaps/curve/curve_module.hpp"   // runtime ModularCurve for user-defined interpolation regions
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/curve_spec.hpp"   // CurveStructure -- BundleCurveSpec is an alias of it

namespace swaps::calibration {

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
};
template <class S>
struct OutrightHandle : CurveHandle<S> {
  curve::ModularCurve<S> c;
  explicit OutrightHandle(curve::ModularCurve<S> cc) : c(std::move(cc)) {}
  S forward(double t) const override { return c.forward(t); }
  S integral(double t) const override { return c.integral(t); }
  S discount(double t) const override { return c.discount(t); }
  void set_forwards(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x) override { c.set_forwards(x); }
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
  std::vector<pricing::Turn> windows;    // turn accrual windows (year fractions)
  Eigen::Matrix<S, Eigen::Dynamic, 1> deltas;  // δⱼ, the free overlay state variables
  int n_interp = 0;                      // size of `base`'s own state (the interp-knot count)

  TurnedCurve(std::unique_ptr<CurveHandle<S>> b, std::vector<pricing::Turn> w, int ni)
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
      I += deltas[static_cast<int>(j)] * S(pricing::turn_overlap(t, windows[j]));
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
};

// A curve's definition in a bundle: knots (or regions), outright/spread, currency. This is THE SAME
// TYPE the pricing engine uses -- pricing::CurveStructure (curve_spec.hpp) -- not a mirror of it, so the
// two can never drift. Named `BundleCurveSpec` here for the calibration layer's vocabulary.
using BundleCurveSpec = pricing::CurveStructure;

// Build every bundle curve as a handle. `value(c, i)` supplies curve c's local knot i (free from the
// parameter vector, or a frozen constant). Requires base < c so each spread's base is already built.
template <class Scalar, class ValueFn>
std::vector<std::unique_ptr<CurveHandle<Scalar>>> build_bundle_curves(
    const std::vector<BundleCurveSpec>& specs, ValueFn value) {
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
  void build(const std::vector<BundleCurveSpec>& specs) {
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

// (mtm_funding_term_negligible retired 2026-09-09: the compiled batch prices the MtM funding leg exactly.)


class BundleProblem {
 public:
  using CurveSpec = BundleCurveSpec;

  std::vector<CurveSpec> curves;
  // Every calibration instrument is a generic Instrument (design §3): legs carrying their OWN curve
  // roles + a quote transform. OIS swaps, tenor bases, averaging/compounding futures, IBOR legs,
  // spreads, mixed day counts are all THIS type with different DATA -- there is no index-flavoured
  // shorthand and no per-shape engine type.
  std::vector<Instrument> instruments;

  int n_curves() const { return static_cast<int>(curves.size()); }
  int n_knots() const {
    int n = 0;
    for (const auto& c : curves) n += c.n_knots();
    return n;
  }
  int n_residuals() const { return static_cast<int>(instruments.size()); }
  int offset(int k) const {
    int o = 0;
    for (int i = 0; i < k; ++i) o += curves[i].n_knots();
    return o;
  }

  // THE RESIDUAL ORDER is the instruments' INSERTION order (deterministic and DOCUMENTED — the
  // Jacobian rows, the W-cache batches in CompiledBundleResidual, market(), the risk ladder and the
  // warm/streaming feeds all index off it). The compiled engine batches by quote kind internally and
  // scatters each batch back to its instrument's insertion-order row.

  // Target quotes in residual order — lets the generic AadResidualEngine recover
  // model_rates = residuals + market.
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(n_residuals());
    for (int i = 0; i < static_cast<int>(instruments.size()); ++i) m[i] = instruments[i].market;
    return m;
  }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    // Hoist the per-curve state offsets: offset(c) is an O(n_curves) walk and the value lambda below
    // runs once per knot, so inlining it would cost O(n_curves x n_knots) per residual evaluation.
    std::vector<int> off(curves.size());
    for (int c = 0; c < static_cast<int>(curves.size()); ++c) off[c] = offset(c);
    const auto C = build_bundle_curves<Scalar>(
        curves, [&](int c, int i) { return x[off[c] + i]; });

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    // Each generic leg resolves its OWN role, so forecast != discount and cross-curve legs need no
    // special case here — the role indices are just lookups into the built curve handles.
    const auto curve_of = [&C](int i) -> const CurveHandle<Scalar>& { return *C[i]; };
    int row = 0;
    for (const auto& ins : instruments) r[row++] = instrument_residual<Scalar>(ins, curve_of);
    return r;
  }
};

// BUNDLE VALIDATION at the engine / session seam (E3-E1/E2/B12, 2026-09-10). Refuses, with a message naming
// the curve or row: a curve with no knots, a knot at t <= 0, a base index out of range or self-referential,
// an instrument with an empty leg (validate_instrument), and a curve role index outside the bundle -- each of
// which used to crash, price NaN or silently corrupt W. O(n), no allocation beyond the message on failure.
inline void validate_problem(const BundleProblem& p, const std::string& where = "bundle") {
  const int nc = static_cast<int>(p.curves.size());
  const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + ": " + what); };
  if (nc == 0) fail("the bundle has no curves");
  for (int c = 0; c < nc; ++c) {
    const auto& cs = p.curves[c];
    if (cs.n_interp_knots() == 0) fail("curve " + std::to_string(c) + " has no regions/knots (regions: [] is not a curve)");
    for (const auto& r : cs.regions)
      if (!r.knots.empty() && !(r.knots.front() > 0.0))
        fail("curve " + std::to_string(c) + " has a knot at t = " + std::to_string(r.knots.front()) + " (every knot time must be > 0)");
    if (cs.base >= nc || cs.base == c) fail("curve " + std::to_string(c) + " has an invalid base curve index " + std::to_string(cs.base));
  }
  const auto idx = [&](int i, const char* role, int row) {
    if (i < 0 || i >= nc)
      fail("instrument " + std::to_string(row) + " references " + role + " curve " + std::to_string(i) + " outside the bundle's " + std::to_string(nc) + " curves");
  };
  std::function<void(const Instrument&, int)> check = [&](const Instrument& ins, int row) {
    validate_instrument(ins, where + " instrument " + std::to_string(row));
    switch (ins.quote) {
      case QuoteKind::Rate: idx(ins.forecast, "forecast", row); break;
      case QuoteKind::FxForward: idx(ins.fx_num, "fx_num", row); idx(ins.fx_den, "fx_den", row); break;
      case QuoteKind::TurnJump:
        idx(ins.turn_curve, "turn", row);
        if (ins.turn_index >= static_cast<int>(p.curves[ins.turn_curve].turns.size()))
          fail("instrument " + std::to_string(row) + " pins turn " + std::to_string(ins.turn_index) + " but curve " + std::to_string(ins.turn_curve) + " has " + std::to_string(p.curves[ins.turn_curve].turns.size()) + " turns");
        break;
      case QuoteKind::Portfolio:
        for (const auto& c : ins.combination) check(c.instrument, row);
        break;
      case QuoteKind::ParSpread:
      case QuoteKind::XccyMtmBasis:
        idx(ins.bench.forecast, "benchmark forecast", row); idx(ins.bench.discount, "benchmark discount", row);
        [[fallthrough]];
      case QuoteKind::ParRate:
      case QuoteKind::ZeroCouponRate:
        idx(ins.fwd.forecast, "forecast", row); idx(ins.fwd.discount, "discount", row); idx(ins.fixed.discount, "fixed discount", row);
        break;
    }
  };
  for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i) check(p.instruments[i], i);
}

}  // namespace swaps::calibration
