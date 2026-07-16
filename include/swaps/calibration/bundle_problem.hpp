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
#include "swaps/curve/calibration_curve.hpp"
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
  curve::CalibrationCurve<S> c;
  explicit OutrightHandle(curve::CalibrationCurve<S> cc) : c(std::move(cc)) {}
  S forward(double t) const override { return c.forward(t); }
  S integral(double t) const override { return c.integral(t); }
  S discount(double t) const override { return c.discount(t); }
};
// forward = base + spread ; integral = base + spread ; DF = base_DF * exp(-int spread).
template <class S>
struct SpreadHandle : CurveHandle<S> {
  curve::CalibrationCurve<S> spread;
  const CurveHandle<S>* base;
  SpreadHandle(curve::CalibrationCurve<S> sp, const CurveHandle<S>* b) : spread(std::move(sp)), base(b) {}
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
  int n_knots() const { return static_cast<int>(meeting.size() + back.size()); }
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
    auto inner = curve::make_calibration_curve<Scalar>(spec.meeting, spec.back);
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
  // OIS: float forecasts `forecast`, everything discounts `discount` (forecast == discount = single-curve).
  struct Swap {
    int forecast, discount;
    pricing::OisSwap sched;
    double market_rate;
  };
  // Basis swap over an arbitrary benchmark: spread leg forecasts `forecast`, benchmark leg forecasts
  // `benchmark`, both legs discount `discount`.
  struct Basis {
    int forecast, benchmark, discount;
    pricing::OisSwap sched;
    double market_rate;  // par basis spread
  };
  // Futures forecast ONE curve (a rate, no discounting). Front-end instruments.
  struct AvgFut {
    int forecast;
    pricing::AveragedFuture sched;
    double convexity, market_rate;  // market_rate = 1 - price/100
  };
  struct CompFut {
    int forecast;
    pricing::CompoundedFuture sched;
    double convexity, market_rate;
  };

  std::vector<CurveSpec> curves;
  std::vector<Swap> swaps;
  std::vector<Basis> bases;
  std::vector<AvgFut> avg_futs;
  std::vector<CompFut> comp_futs;
  // Generic instruments (design §3): legs carrying their OWN curve roles + a quote transform. The four
  // vectors above are index-flavoured SHORTHAND for the commonest shapes and force both legs of a swap
  // onto one schedule; an Instrument does not. Anything the shorthand cannot say — an IBOR leg, a
  // spread, 30/360-fixed vs ACT/360-float, a semi-annual basis leg against an annual fixed leg,
  // weighted averaging — is expressed here, and needs no new engine type.
  std::vector<Instrument> instruments;

  int n_curves() const { return static_cast<int>(curves.size()); }
  int n_knots() const {
    int n = 0;
    for (const auto& c : curves) n += c.n_knots();
    return n;
  }
  int n_residuals() const {
    return static_cast<int>(swaps.size() + bases.size() + avg_futs.size() + comp_futs.size() +
                            instruments.size());
  }
  int offset(int k) const {
    int o = 0;
    for (int i = 0; i < k; ++i) o += curves[i].n_knots();
    return o;
  }

  // THE RESIDUAL ORDER (deterministic and DOCUMENTED — the Jacobian rows, the W-cache batches in
  // CompiledBundleResidual, market(), the risk ladder and the warm/streaming feeds all index off it;
  // every one of them must fill these rows in exactly this order):
  //   1. avg_futs     in vector order
  //   2. comp_futs    in vector order
  //   3. swaps        in vector order
  //   4. bases        in vector order
  //   5. instruments  in vector order   (generic; appended AFTER the legacy groups, so adding the
  //                                      generic model cannot renumber an existing row)
  // The generic block keeps INSERTION order even though it mixes quote kinds — the compiled engine
  // batches by kind internally and scatters each batch back to its residual row.

  // Target quotes in residual order — lets the generic AadResidualEngine recover
  // model_rates = residuals + market.
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(n_residuals());
    int i = 0;
    for (const auto& a : avg_futs) m[i++] = a.market_rate;
    for (const auto& c : comp_futs) m[i++] = c.market_rate;
    for (const auto& s : swaps) m[i++] = s.market_rate;
    for (const auto& b : bases) m[i++] = b.market_rate;
    for (const auto& ins : instruments) m[i++] = ins.market;
    return m;
  }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    const auto C = build_bundle_curves<Scalar>(
        curves, [&](int c, int i) { return x[offset(c) + i]; });

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    int row = 0;
    for (const auto& a : avg_futs)
      r[row++] = pricing::averaged_future_rate<Scalar>(a.sched, *C[a.forecast]) + (a.convexity - a.market_rate);
    for (const auto& cf : comp_futs)
      r[row++] = pricing::compounded_future_rate<Scalar>(cf.sched, *C[cf.forecast]) + (cf.convexity - cf.market_rate);
    for (const auto& s : swaps)
      r[row++] = pricing::ois_par_rate<Scalar>(s.sched, *C[s.forecast], *C[s.discount]) - Scalar(s.market_rate);
    for (const auto& b : bases)
      r[row++] =
          pricing::basis_par_spread<Scalar>(b.sched, *C[b.forecast], *C[b.benchmark], *C[b.discount]) -
          Scalar(b.market_rate);
    // Each generic leg resolves its OWN role, so forecast != discount and cross-curve legs need no
    // special case here — the role indices are just lookups into the built curve handles.
    const auto curve_of = [&C](int i) -> const CurveHandle<Scalar>& { return *C[i]; };
    for (const auto& ins : instruments) r[row++] = instrument_residual<Scalar>(ins, curve_of);
    return r;
  }
};

}  // namespace swaps::calibration
