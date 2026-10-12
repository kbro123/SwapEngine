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

#include <algorithm>
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
#include "swaps/pricing/curve_handle.hpp"  // CurveHandle & co (E6.4: they are pricing, re-exported below)
#include "swaps/pricing/curve_spec.hpp"   // CurveStructure -- BundleCurveSpec is an alias of it

namespace swaps::calibration {

// The curve-ACCESS family lives in pricing/ (E6.4, 2026-09-10 — it mentions no Instrument and no
// solver, and keeping it here made the portfolio layer include this one). Re-exported so every
// `cal::CurveHandle` / `cal::build_bundle_curves` spelling in the engine, the api and the tests
// still names exactly these types.
using pricing::BundleCurveSet;
using pricing::build_bundle_curves;
using pricing::CurveHandle;
using pricing::curves_are_noncacheable;
using pricing::OutrightHandle;
using pricing::SpreadHandle;
using pricing::TurnedCurve;

// A curve's definition in a bundle: knots (or regions), outright/spread, currency. This is THE SAME
// TYPE the pricing engine uses -- pricing::CurveStructure (curve_spec.hpp) -- not a mirror of it, so the
// two can never drift. Named `BundleCurveSpec` here for the calibration layer's vocabulary.
using BundleCurveSpec = pricing::CurveStructure;

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
  // CONSTRAINT ROWS (review item 3, 2026-10-12): rows of the same Instrument type that carry NO market feed -- a
  // curvature (smoothness) penalty row (regularize.hpp smoothing_rows), a bound on a knot forward, any row whose
  // target is a property of the problem, not a quote. They are rows of the ONE engine like the quotes (residual,
  // Jacobian, penalty map, tier routing), stacked BELOW the instruments in residual order, so n_residuals() counts
  // both; the client-facing surfaces (set_market, stream_update, the risk ladder, quote diagnostics) index the
  // instruments alone. A BundleSession installs the regulariser it is asked for HERE (its tail rows).
  std::vector<Instrument> constraints;
  // The ISO code of each curve currency TAG (index = BundleCurveSpec::currency), e.g. {"USD", "EUR"}. Optional: only
  // an FX move on a book with cross-currency positions needs it, to know which pair each position's spot quotes (SC2).
  std::vector<std::string> currency_codes;

  int n_curves() const { return static_cast<int>(curves.size()); }
  int n_knots() const {
    int n = 0;
    for (const auto& c : curves) n += c.n_knots();
    return n;
  }
  int n_quotes() const { return static_cast<int>(instruments.size()); }
  int n_constraints() const { return static_cast<int>(constraints.size()); }
  int n_residuals() const { return n_quotes() + n_constraints(); }  // every row: the quotes, then the constraints
  // Row i of the residual: instruments first, then constraints (THE residual order below).
  const Instrument& row(int i) const { return i < n_quotes() ? instruments[static_cast<std::size_t>(i)] : constraints[static_cast<std::size_t>(i - n_quotes())]; }
  Instrument& row(int i) { return i < n_quotes() ? instruments[static_cast<std::size_t>(i)] : constraints[static_cast<std::size_t>(i - n_quotes())]; }
  int offset(int k) const {
    int o = 0;
    for (int i = 0; i < k; ++i) o += curves[i].n_knots();
    return o;
  }

  // THE RESIDUAL ORDER is the instruments' INSERTION order, then the constraints' (deterministic and
  // DOCUMENTED — the Jacobian rows, the W-cache batches in CompiledBundleResidual, market(), the risk ladder
  // and the warm/streaming feeds all index off it). The compiled engine batches by quote kind internally and
  // scatters each batch back to its row.

  // Target quotes in residual order (a constraint row's target below the quotes) — lets the generic
  // AadResidualEngine recover model_rates = residuals + market.
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(n_residuals());
    market_into(m);
    return m;
  }
  void market_into(Eigen::VectorXd& m) const {  // allocation-free form for a per-solve scratch
    m.resize(n_residuals());
    for (int i = 0; i < n_residuals(); ++i) m[i] = row(i).market;
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
    int k = 0;
    for (const auto& ins : instruments) r[k++] = instrument_residual<Scalar>(ins, curve_of);
    for (const auto& ins : constraints) r[k++] = instrument_residual<Scalar>(ins, curve_of);
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
  // Every curve reference the instrument reads (for_each_curve_ref: the ONE encoding of which members a kind
  // reads; Portfolio components included), then the one check that is not a curve reference: a turn's index.
  std::function<void(const Instrument&, int)> check = [&](const Instrument& ins, int row) {
    validate_instrument(ins, where + " instrument " + std::to_string(row));
    ins.for_each_curve_ref([&](int c, const char* role) { idx(c, role, row); });
    if (ins.quote == QuoteKind::TurnJump && ins.turn_index >= static_cast<int>(p.curves[ins.turn_curve].turns.size()))
      fail("instrument " + std::to_string(row) + " pins turn " + std::to_string(ins.turn_index) + " but curve " + std::to_string(ins.turn_curve) + " has " + std::to_string(p.curves[ins.turn_curve].turns.size()) + " turns");
    if (ins.quote == QuoteKind::StateLinear)
      for (const auto& s : ins.state)
        if (s.index >= p.curves[s.curve].n_knots())
          fail("instrument " + std::to_string(row) + " reads state " + std::to_string(s.index) + " of curve " + std::to_string(s.curve) + ", which has " + std::to_string(p.curves[s.curve].n_knots()) + " states");
    if (ins.quote == QuoteKind::Portfolio)
      for (const auto& c : ins.combination) check(c.instrument, row);  // each component's own validate_instrument + turn check
  };
  for (int i = 0; i < p.n_residuals(); ++i) check(p.row(i), i);  // the quotes, then the constraint rows
}

// A flat starting guess sized to the problem: outright curves at `level`, spread curves at 0.
// level <= 0 (the default) derives the flat level from the market itself: the mean outright
// (ParRate/Rate/ZeroCouponRate) quote, clamped to [0.1%, 20%], falling back to 2% for a bundle with no outright
// rows. Spread curves and turn deltas always seed at 0. (Moved from api/bundle_api.cpp, E7 3.6: it is a function
// of the problem alone, and calibration-layer code seeds from it.)
inline Eigen::VectorXd flat_x0(const BundleProblem& prob, double level = 0.0) {
  if (level <= 0.0) {
    // Derive the flat seed level FROM THE MARKET: the mean outright quote (ParRate / Rate rows). This is
    // what makes the seed a defensible ANCHOR for rank-deficient completion (calibrate_with): an
    // unconstrained state then reports "the average market level", not an arbitrary constant. Spread
    // curves still seed at zero; clamped to a sane band; falls back to 2% when no outright rows exist.
    double sum = 0.0;
    int n = 0;
    for (const auto& ins : prob.instruments)
      if (ins.quote == QuoteKind::ParRate || ins.quote == QuoteKind::Rate || ins.quote == QuoteKind::ZeroCouponRate) {
        sum += ins.market;
        ++n;
      }
    level = n ? std::min(0.20, std::max(1e-3, sum / n)) : 0.02;
  }
  Eigen::VectorXd x(prob.n_knots());
  int o = 0;
  for (const auto& c : prob.curves) {
    const double v = (c.base < 0) ? level : 0.0;  // outright at the level, spread at zero
    const int ni = c.n_interp_knots();            // interp knots first, then one δ per turn
    // Interp knots seed at the level/zero; turn δ's are an overlay -> seed at 0 (no jump), NOT the level.
    for (int i = 0; i < c.n_knots(); ++i) x[o++] = (i < ni) ? v : 0.0;
  }
  return x;
}

}  // namespace swaps::calibration
