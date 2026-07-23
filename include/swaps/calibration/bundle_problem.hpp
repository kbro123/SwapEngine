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
#include <vector>

#include "swaps/calibration/problem.hpp"  // the generic FloatLeg/FixedLeg/Instrument model (design §3)
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/curve_module.hpp"  // runtime ModularCurve for user-defined interpolation regions
#include "swaps/pricing/cashflows.hpp"

namespace swaps::calibration {

// Type-erased curve so a bundle can hold heterogeneous curves (outright / spread) and price through one
// kernel. Scalar-templated so AAD flows through the virtual calls.
template <class S>
struct CurveHandle {
  virtual ~CurveHandle() = default;
  virtual S forward(double t) const = 0;
  virtual S integral(double t) const = 0;
  virtual S discount(double t) const = 0;
};
template <class S>
struct OutrightHandle : CurveHandle<S> {
  curve::ModularCurve<S> c;
  explicit OutrightHandle(curve::ModularCurve<S> cc) : c(std::move(cc)) {}
  S forward(double t) const override { return c.forward(t); }
  S integral(double t) const override { return c.integral(t); }
  S discount(double t) const override { return c.discount(t); }
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
};
// A curve's definition: knots + interpolation (Flat front + Hermite back) + parameterization.
struct BundleCurveSpec {
  std::vector<double> meeting;  // front (flat) knot times
  std::vector<double> back;     // back (Hermite) knot times
  int base = -1;                // -1 = outright; else this curve = curves[base] + spread (spread knots)
  // Engine-BLIND currency tag (multi-currency support). The kernel — build_bundle_curves, residuals,
  // the W-cache — NEVER reads it; it exists only so a BUILDER can resolve a per-index default discount
  // curve and FX conversion at construction time (CLAUDE.md §1: the engine names no currency). Default
  // 0 keeps every existing single-currency bundle byte-identical.
  int currency = 0;
  // Interpolation regions (each a scheme + its knot times). When non-empty these define the curve and
  // `meeting`/`back` are ignored; when empty they mean "the shipped layout", i.e. Flat(meeting) +
  // Hermite(back). Either way exactly one curve type is built from modules() below. Region knots are
  // the free forwards, region by region, in this order.
  std::vector<curve::CurveModule> regions;

  // The ONE description of this curve's interpolation -- custom regions or the shipped default.
  std::vector<curve::CurveModule> modules() const {
    return regions.empty() ? curve::flat_hermite(meeting, back) : regions;
  }
  int n_knots() const {
    if (!regions.empty()) {
      int n = 0;
      for (const auto& r : regions) n += static_cast<int>(r.knots.size());
      return n;
    }
    return static_cast<int>(meeting.size() + back.size());
  }
};

// Build every bundle curve as a handle. `value(c, i)` supplies curve c's local knot i (free from the
// parameter vector, or a frozen constant). Requires base < c so each spread's base is already built.
template <class Scalar, class ValueFn>
std::vector<std::unique_ptr<CurveHandle<Scalar>>> build_bundle_curves(
    const std::vector<BundleCurveSpec>& specs, ValueFn value) {
  std::vector<std::unique_ptr<CurveHandle<Scalar>>> C(specs.size());
  for (int c = 0; c < static_cast<int>(specs.size()); ++c) {
    const auto& spec = specs[c];
    if (spec.base >= c) throw std::invalid_argument("bundle curve: base index must be < curve index");
    const int nk = spec.n_knots();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> xi(nk);
    for (int i = 0; i < nk; ++i) xi[i] = value(c, i);
    auto inner = curve::make_modular_curve<Scalar>(spec.modules());
    inner.set_forwards(xi);
    if (spec.base < 0)
      C[c] = std::make_unique<OutrightHandle<Scalar>>(std::move(inner));
    else
      C[c] = std::make_unique<SpreadHandle<Scalar>>(std::move(inner), C[spec.base].get());
  }
  return C;
}

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
    const auto C = build_bundle_curves<Scalar>(
        curves, [&](int c, int i) { return x[offset(c) + i]; });

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    // Each generic leg resolves its OWN role, so forecast != discount and cross-curve legs need no
    // special case here — the role indices are just lookups into the built curve handles.
    const auto curve_of = [&C](int i) -> const CurveHandle<Scalar>& { return *C[i]; };
    int row = 0;
    for (const auto& ins : instruments) r[row++] = instrument_residual<Scalar>(ins, curve_of);
    return r;
  }
};

}  // namespace swaps::calibration
