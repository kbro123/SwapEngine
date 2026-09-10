#pragma once
// The calibration problem: free variables x = knot forwards, residual vector r(x) in RATE units.
//
// QuantLib-free and templated on Scalar (CLAUDE.md §1): the schedules are extracted from QuantLib
// once (swaps/ql/extract.hpp) and stored as plain data here; residuals(x) rebuilds the curve and
// reprices every instrument with the templated kernel. With Scalar = double this drives the LM
// solve; with Scalar = AutoDiffScalar (Phase 3) the SAME code yields the analytic Jacobian.
//
// Residual convention (CLAUDE.md §2): everything is in rate units so futures (quoted as prices) do
// not swamp swaps. A futures market quote of price P contributes target rate (1 - P/100); the model
// side is the reference rate plus its convexity adjustment.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::calibration {

// =================================================================================================
// GENERIC INSTRUMENT MODEL (docs/generic-instrument-pipeline.md §3)
// =================================================================================================
// An instrument = LEGS + a quote transform + a market quote. Nothing here names an index, a currency,
// a calendar or a convention (CLAUDE.md §1, design §6): a leg is a list of generic FloatCoupon /
// FixedCoupon (pricing/cashflows.hpp) and a curve role is an integer. "SOFR OIS", "EURIBOR 3M swap",
// "FF/SOFR basis", "1M averaging future" are all THIS type with different DATA, built by a test.

// Curve ROLES belong to the LEG, not the instrument. That is the whole point: a tenor-basis or
// cross-curve instrument is just two legs with different `forecast` curves, and forecast != discount
// (multi-curve) is a leg-local fact — neither needs a new instrument type.
struct FloatLeg {
  std::vector<pricing::FloatCoupon> coupons;
  int forecast = 0;  // curve that FORECASTS this leg's index fixings
  int discount = 0;  // curve that DISCOUNTS this leg's payments
  // MtM (mark-to-market cross-currency) notional-reset roles. -1/-1 => a plain constant-notional leg
  // (the default). When set, this is an FX-resettable leg whose coupon notional is
  // fx_spot · DF[reset_num](reset)/DF[reset_den](reset) -- priced by pricing::xccy_mtm_leg_pv on the templated
  // path and, since 2026-09-09, EXACTLY on the W-cache too (BundleFloatBatch::add_mtm: a product of registered DFs).
  int reset_num = -1;   // FX-forward NUMERATOR (foreign) discount curve role
  int reset_den = -1;   // FX-forward DENOMINATOR (domestic) discount curve role
  double fx_spot = 1.0;  // FX spot for the notional reset
};

struct FixedLeg {
  std::vector<pricing::FixedCoupon> coupons;
  int discount = 0;  // curve that DISCOUNTS this leg's payments (the annuity curve)
};

// The quote transform. ALL are in RATE units (CLAUDE.md §2) so a futures row cannot outweigh a swap row.
enum class QuoteKind {
  ParRate,    // float_leg_pv(fwd) / annuity(fixed)
  ParSpread,  // (float_leg_pv(bench) - float_leg_pv(fwd)) / annuity(fixed)
  Rate,       // rate(obs) + convexity
  ZeroCouponRate, // ANNUALLY-COMPOUNDED zero-coupon par rate: r = (1 + τ·q)^(1/τ) − 1 with q the ParRate
                  // quotient of the same legs and τ the ONE fixed accrual (BRL DI×Pre: fixed pays
                  // (1+r)^τ − 1 at maturity vs CDI compounded to maturity, BUS/252). A NONLINEAR transform
                  // of ParRate (zero_coupon_transform); rides the W-cache with a chain-rule row scale, and
                  // the band residual applies to the TRANSFORMED quote.
  // Cross-currency (multi-currency). BOTH are W-cacheable in their STANDARD form (compiled_bundle.hpp):
  // a standalone FxForward's log-residual is affine in x (constant Jacobian row), and a MtM basis with a
  // funding leg is priced EXACTLY on the W-cache since 2026-09-09 (BundleFloatBatch::add_mtm; the retired mtm_funding_term_negligible
  // guard proves the dropped term is 0). Only the non-standard cases -- an FX/MtM NESTED in a Portfolio,
  // or a MtM whose funding term is genuinely nonzero (payment lag / averaging convexity / CSA discount)
  // -- ride the width-reduced AAD block of the hybrid engine (hybrid_residual.hpp).
  FxForward,      // FX-forward point: fx_spot · DF[fx_num](fx_time)/DF[fx_den](fx_time) (pins fx_num vs fx_den)
  XccyMtmBasis,   // MtM (FX-resettable-notional) xccy basis: par basis incl. the resetting funding leg
  Portfolio,      // linear combination of component instruments: model quote = Σ weight·quote(component).
                  // `market` is the COMBINED quote (a butterfly/condor spread), so you calibrate to the
                  // combo directly without pinning each leg's outright rate. Components are full nested
                  // Instruments, so portfolios compose. ONE residual, no knots (knots are in the curve
                  // spec). W-CACHEABLE when every component is: the components register weighted onto the
                  // one row and accumulate in the compiled batches (an FX/MtM component forces AAD).
  TurnJump,       // a TURN's jump δ (docs/turns-calibration.md). The model quote is the raw overlay state
                  // variable δ of (turn_curve, turn_index) -- a STATE-PIN, LINEAR in x (Jacobian row is a
                  // unit vector at δ's state index). Almost always BANDED (target/lower/upper): the band's
                  // target regularises δ so it is always identifiable; bracketing futures then sharpen it.
                  // It IS W-cacheable (linear, no DF), but its Jacobian entry is direct (∂δ/∂x), not via DF.
};

// Forward declarations for the recursive Portfolio components (each component is itself an Instrument).
struct Instrument;
struct WeightedInstrument;

// One calibration instrument.
//
// Which members a given `quote` reads (the others are ignored and may stay default-constructed):
//   ParRate   : `fwd` (the float leg, enters POSITIVELY) and `fixed` (the annuity).
//   ParSpread : `fwd` (the SPREAD/quoted leg, enters NEGATIVELY), `bench` (the benchmark leg, enters
//               POSITIVELY) and `fixed`. The two float legs are INDEPENDENT — different frequency,
//               day count, spread and forecast curve are all fine (design §6.7).
//   Rate      : `obs`, `forecast` and `convexity`.
//
// Sign convention for ParSpread: the quoted spread s satisfies pv(fwd) + s·annuity = pv(bench), hence
// s = (pv_bench − pv_fwd)/annuity.
struct Instrument {
  QuoteKind quote = QuoteKind::ParRate;
  FloatLeg fwd;
  FloatLeg bench;
  FixedLeg fixed;
  pricing::RateObservation obs;  // Rate only
  int forecast = 0;              // Rate only: the curve that forecasts `obs`
  // Rate only. An INPUT NUMBER (design §3): the convexity MODEL (Hull-White etc.) lives in tests.
  double convexity = 0.0;
  double market = 0.0;  // the market quote, in the units of `quote` (always rate units)

  // Bid/offer BAND (soft calibration target). When `band_upper > band_lower` (bounds in the quote's rate
  // units) the residual is the HUBER band residual (band_residual()): a `band_decay`-slope pull to the mid
  // inside [lower, upper], a unit-slope pull to the nearer EDGE outside it, continuous at the edges. So
  // the solver treats any model value within [lower, upper] as ~satisfied and spends its freedom on the
  // hard targets, while an overlapping instrument that cannot be hit exactly settles inside its band. The
  // default (band_upper <= band_lower, band_decay = 1) leaves the residual as the plain (q − market).
  double band_lower = 0.0, band_upper = 0.0, band_decay = 1.0;

  // Portfolio (QuoteKind::Portfolio) components: model quote = Σ weight·model_quote(component). Ignored
  // for every other quote kind. Defined out-of-line below (recursive type).
  std::vector<WeightedInstrument> combination;

  // The currency the quote/residual is expressed in (multi-currency). Consulted ONLY by a cross-
  // currency quote that mixes legs of different currencies (to name the PV numeraire); every single-
  // currency quote ignores it. Default 0 keeps existing instruments byte-identical.
  int pv_currency = 0;

  // FxForward only: F = fx_spot · DF[fx_num](fx_time) / DF[fx_den](fx_time). fx_num is the FOREIGN
  // (collateral) curve this pins (e.g. EUR-in-USD), fx_den the DOMESTIC (e.g. SOFR). `market` is the
  // outright forward. The residual is in rate/implied-basis units: (ln F_model − ln F_market)/fx_time.
  int fx_num = -1, fx_den = -1;
  double fx_spot = 1.0;
  double fx_time = 0.0;
  // XccyMtmBasis only: the resetting-notional funding leg (its reset_num/reset_den/fx_spot on the leg
  // define the FX-forward notional). fwd = the pinned curve's self-forecast leg, bench = the other-
  // currency forecast leg, fixed = the annuity — all discounted on the pinned (collateral) curve.
  FloatLeg mtm;

  // TurnJump only: which turn this instrument pins. `turn_curve` is the curve carrying the turn and
  // `turn_index` its position in that curve's `turns` list. The model quote is that turn's jump δ; the
  // residual is (banded) δ − market, with `market` the target jump (rate units). See QuoteKind::TurnJump.
  int turn_curve = 0, turn_index = 0;

  // Set the FULL quote RHS -- target and soft-quote band -- from any target-shaped object exposing
  // {target, band_lower, band_upper, band_decay}. This is THE one hand-off from a market quote into a
  // calibration instrument: market::Quote::to_target() (market/quote.hpp CalibrationTarget) and the
  // API compiler's wire quote (api/compile.cpp) both feed THIS setter, so the band semantics above are
  // defined exactly once -- here. Duck-typed on purpose: the calibration layer sits BELOW the market
  // layer and must not include it; any POD with those four fields is a valid source.
  template <class Target>
  void set_target(const Target& t) {
    market = t.target;
    band_lower = t.band_lower;
    band_upper = t.band_upper;
    band_decay = t.band_decay;
  }

  // The curve this instrument primarily PINS. Used by the staged solver to assign it to a dependency
  // block; FxForward pins its FOREIGN (fx_num) curve, every leg-based quote its fwd leg's forecast, a
  // Portfolio its first component's. Defined out-of-line (Portfolio dereferences the nested type).
  int primary_curve() const;
};

// A weighted component of a Portfolio instrument. Holds a full Instrument by value, so portfolios nest.
struct WeightedInstrument {
  double weight = 1.0;
  Instrument instrument;
};

inline int Instrument::primary_curve() const {
  if (quote == QuoteKind::Rate) return forecast;
  if (quote == QuoteKind::FxForward) return fx_num;
  if (quote == QuoteKind::TurnJump) return turn_curve;  // a turn pins the curve that carries it
  if (quote == QuoteKind::Portfolio)
    return combination.empty() ? 0 : combination.front().instrument.primary_curve();
  return fwd.forecast;
}

// Compile-time detection: does the curve object a CurveOf accessor returns expose turn_jump(int)? True
// for the bundle's CurveHandle, false for a bare ModularCurve (single-curve CalibrationProblem, which
// never carries a TurnJump instrument). Lets instrument_model_quote's TurnJump branch stay well-formed
// for BOTH curve types via `if constexpr`.
template <class C, class = void>
struct has_turn_jump : std::false_type {};
template <class C>
struct has_turn_jump<C, std::void_t<decltype(std::declval<const C&>().turn_jump(0))>> : std::true_type {};

// Bid/offer band residual for a model quote q against market mid m (see the Instrument band fields).
// The band exists so that OVERLAPPING instruments that cannot all be reconciled exactly (1M vs 3M futures,
// a future vs a swap at the same pillar) can each sit off their mid within a bid/offer tolerance. The
// residual is HUBER-shaped: a reduced-rate pull to the mid INSIDE the band, full-slope pull to the nearer
// EDGE outside it, continuous at the edges:
//     inside  [lower, upper] : r = decay·(q − m)
//     above   upper          : r = decay·(upper − m) + (q − upper)
//     below   lower          : r = decay·(lower − m) + (q − lower)
// so dr/dq is exactly `decay` inside and exactly 1 outside, with no ramp in between. That is deliberate:
// the previous smooth Gaussian ramp (w = decay + (1−decay)(1 − e^{−z²})) made r an S-curve whose slope
// overshoots above 1 just outside the edge, which (a) gives the summed-squares objective MULTIPLE minima
// along the direction the quotes barely see (two stationary fits 110 bp apart in a knot at the same
// market, chosen by the seed), and (b) turns the Jacobian into a moving target at every edge, so the
// frozen-Newton streamer converged, silently, to points up to 147 bp from the least-squares optimum. The
// piecewise-linear residual is monotone with a single zero, so r² is convex in q; and its Jacobian is
// piecewise CONSTANT (row = slope·∂q/∂x with slope ∈ {decay, 1}), which is what lets the streamer keep
// the quote Jacobian frozen and treat a band crossing as a cheap row re-scale (streaming.hpp).
// Far outside the band the pull is to the EDGE (plus the decay pull to mid), not to the mid at full
// weight: continuity at the edge forces that, and it is the right reading of a bid/offer tolerance.
// No band (upper <= lower): the plain residual q − m. AAD-safe: branch selection on q (one-sided
// derivative exactly at an edge), constants folded so `q` is always the plain-scalar operand.
template <class Scalar>
Scalar band_residual(const Scalar& q, double market, double lower, double upper, double decay) {
  if (!(upper > lower)) return q - Scalar(market);
  if (q > Scalar(upper)) return q + Scalar(decay * (upper - market) - upper);
  if (q < Scalar(lower)) return q + Scalar(decay * (lower - market) - lower);
  return (q - Scalar(market)) * decay;
}

// Plain-double (r, dr/dq) of band_residual, for the compiled path's residual + ANALYTIC Jacobian. MUST
// match band_residual() above term-for-term (the compiled-vs-AAD band parity test pins the equality).
inline std::pair<double, double> band_residual_d(double q, double market, double lower, double upper,
                                                 double decay) {
  if (!(upper > lower)) return {q - market, 1.0};
  if (q > upper) return {decay * (upper - market) + (q - upper), 1.0};
  if (q < lower) return {decay * (lower - market) + (q - lower), 1.0};
  return {decay * (q - market), decay};
}

// The residual's slope dr/dq at model quote q: `decay` inside the band, 1 outside, 1 with no band. This is
// the "effective weight" quote diagnostics report, and the per-row scale the streamer tracks.
inline double band_slope(double q, double lower, double upper, double decay) {
  if (!(upper > lower)) return 1.0;
  return (q > upper || q < lower) ? 1.0 : decay;
}

// ZeroCouponRate: the annually-compounded rate r with (1+r)^τ − 1 == τ·q, i.e. r = (1+τq)^(1/τ) − 1, and
// dr/dq = (1+τq)^(1/τ − 1). Written with exp/log so the AAD scalar types carry the derivative.
template <class Scalar>
Scalar zero_coupon_transform(const Scalar& q, double tau) {
  using std::exp; using std::log;
  return exp(log(Scalar(1.0) + Scalar(tau) * q) / tau) - Scalar(1.0);
}
inline std::pair<double, double> zero_coupon_transform_d(double q, double tau) {  // {r, dr/dq}
  const double base = 1.0 + tau * q;
  const double r = std::exp(std::log(base) / tau) - 1.0;
  return {r, std::exp((1.0 / tau - 1.0) * std::log(base))};
}
// The single fixed accrual τ of a ZeroCouponRate instrument (its fixed leg IS one coupon; anything else is a
// build error, never a silent Σ).
inline double zero_coupon_tau(const Instrument& ins) {
  if (ins.fixed.coupons.size() != 1)
    throw std::invalid_argument("ZeroCouponRate instrument must have exactly ONE fixed coupon (the zero-coupon accrual)");
  return ins.fixed.coupons.front().tau;
}

// Model quote of an instrument, per the design §3 table. `C(role)` maps a curve role index to the
// curve object (anything with `Scalar discount(double)`); a single-curve problem passes a lambda that
// returns its one curve for every role.
// INPUT VALIDATION of one instrument's shape (E3-B12, 2026-09-10): an instrument with an empty leg priced to
// NaN on both the templated and the compiled route with no exception (annuity() returns 0 for an empty leg
// and the par rate divides by it). Called once at engine construction (HybridBundleResidual) and at the
// session seam; `where` names the caller in the message. Only SHAPE is checked here -- curve-index ranges
// need the bundle and are checked by validate_problem (bundle_problem.hpp).
inline void validate_instrument(const Instrument& ins, const std::string& where) {
  const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + ": " + what); };
  const auto leg = [&](const FloatLeg& l, const char* name) {
    if (l.coupons.empty()) fail(std::string("the ") + name + " leg has no coupons");
    for (const auto& c : l.coupons)
      if (!(c.obs.tau_index > 0.0)) fail(std::string("a ") + name + " coupon has tau_index <= 0");
  };
  switch (ins.quote) {
    case QuoteKind::Rate:
      if (!(ins.obs.tau_index > 0.0)) fail("a Rate observation has tau_index <= 0");
      break;
    case QuoteKind::ParRate:
    case QuoteKind::ZeroCouponRate:
      leg(ins.fwd, "float");
      if (ins.fixed.coupons.empty()) fail("the fixed leg has no coupons");
      break;
    case QuoteKind::ParSpread:
      leg(ins.fwd, "float");
      leg(ins.bench, "benchmark");
      if (ins.fixed.coupons.empty()) fail("the fixed (annuity) leg has no coupons");
      break;
    case QuoteKind::XccyMtmBasis:
      leg(ins.fwd, "float");
      leg(ins.bench, "benchmark");
      if (ins.fixed.coupons.empty()) fail("the fixed (annuity) leg has no coupons");
      break;
    case QuoteKind::FxForward:
      if (!(ins.fx_time > 0.0)) fail("an FX forward needs fx_time > 0");
      if (!(ins.fx_spot > 0.0)) fail("an FX forward needs fx_spot > 0");
      break;
    case QuoteKind::TurnJump:
      if (ins.turn_curve < 0 || ins.turn_index < 0) fail("a TurnJump needs turn_curve and turn_index >= 0");
      break;
    case QuoteKind::Portfolio:
      if (ins.combination.empty()) fail("a Portfolio quote has no components");
      for (const auto& c : ins.combination) validate_instrument(c.instrument, where + " (portfolio component)");
      break;
  }
}

template <class Scalar, class CurveOf>
Scalar instrument_model_quote(const Instrument& ins, const CurveOf& C) {
  switch (ins.quote) {
    case QuoteKind::Portfolio: {
      // Σ weight·model_quote(component). Recursive, so a component may itself be a Portfolio.
      Scalar acc(0.0);
      for (const auto& c : ins.combination)
        acc += Scalar(c.weight) * instrument_model_quote<Scalar>(c.instrument, C);
      return acc;
    }
    case QuoteKind::Rate:
      return pricing::rate<Scalar>(ins.obs, C(ins.forecast)) + ins.convexity;
    case QuoteKind::TurnJump: {
      // The model quote is the raw turn jump δ, read straight off the (turned) curve handle. Linear in
      // x: δ IS a state variable. Only the bundle's CurveHandle exposes turn_jump; a bare ModularCurve
      // never carries a TurnJump instrument, so guard the call so BOTH curve types compile.
      using CurveT = std::decay_t<decltype(C(ins.turn_curve))>;
      if constexpr (has_turn_jump<CurveT>::value)
        return C(ins.turn_curve).turn_jump(ins.turn_index);
      else
        throw std::logic_error("TurnJump instrument requires a turned bundle curve handle");
    }
    case QuoteKind::ParSpread:
      return (pricing::float_leg_pv<Scalar>(ins.bench.coupons, C(ins.bench.forecast),
                                            C(ins.bench.discount)) -
              pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount))) /
             pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
    case QuoteKind::FxForward:
      // FX-forward outright = fx_spot · DF_foreign(fx_time) / DF_domestic(fx_time). This is exactly the
      // machinery that converts a forward foreign cashflow back to the domestic currency at any date.
      return ins.fx_spot * (C(ins.fx_num).discount(ins.fx_time) / C(ins.fx_den).discount(ins.fx_time));
    case QuoteKind::XccyMtmBasis: {
      // Par basis of a MtM (FX-resettable-notional) xccy swap. fwd = the collateral curve's self-forecast
      // leg (pv telescopes to the par-float value), bench = the foreign-index forecast leg, fixed = the
      // annuity, all discounted on the collateral (pinned) curve; mtm = the resetting funding leg.
      //   b = (pv_self − pv_foreign)/annuity + mtm_leg_pv / (fx_spot · annuity)
      // The funding leg is par (SOFR-flat) so the mtm term ~0, but computing it exercises the resettable-
      // notional coupon and couples the residual to the funding (SOFR) curve.
      const Scalar ann = pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
      const Scalar pv_self =
          pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount));
      const Scalar pv_fx =
          pricing::float_leg_pv<Scalar>(ins.bench.coupons, C(ins.bench.forecast), C(ins.bench.discount));
      const Scalar mtm = pricing::xccy_mtm_leg_pv<Scalar>(
          ins.mtm.coupons, ins.mtm.fx_spot, C(ins.mtm.forecast), C(ins.mtm.discount),
          C(ins.mtm.reset_num), C(ins.mtm.reset_den));
      return (pv_self - pv_fx) / ann + mtm / (ins.mtm.fx_spot * ann);
    }
    case QuoteKind::ZeroCouponRate:
      return zero_coupon_transform<Scalar>(
          pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount)) /
              pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount)),
          zero_coupon_tau(ins));
    case QuoteKind::ParRate:
    default:
      return pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount)) /
             pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
  }
}

// residual = model_quote - market, in RATE units.
//
// `Rate` is spelled `rate + (convexity - market)` rather than `(rate + convexity) - market`: the two
// are algebraically identical but not bit-identical, and this association is the one the legacy
// futures residual uses, so a legacy future re-expressed as a generic instrument reprices to the last
// bit (design §2's backward-compatibility invariant).
//
// AAD note: a fully-fixed observation (empty sub-periods) makes `Rate` a genuine CONSTANT residual
// row with an empty derivative vector — aad_jacobian() zeroes such rows defensively. The quotient
// transforms never hit this: DF(pay) always carries the derivatives (see float_coupon_pv).
// A banded instrument (band_upper > band_lower) uses the Huber band residual band_residual() (decay-slope
// to mid inside, unit slope outside). The band is NOT applied to FxForward (its residual is already a
// log-basis transform).
//
// `market` is the target quote the residual is measured against. It defaults (overload below) to the
// instrument's stored mid `ins.market` — the calibration case — but the frozen-Newton STREAMER passes the
// LIVE feed instead, so the SAME residual definition drives both cold calibrate and each tick. FX needs
// the market INSIDE the log (ln F_model − ln market), so a live feed cannot be applied by subtracting
// afterward; threading it through here keeps FX (and the band `q − market` term) exact per tick. The band
// bounds stay absolute bid/offer levels, independent of the live market.
template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C, double market) {
  if (ins.quote == QuoteKind::FxForward) {
    // Residual in RATE units (CLAUDE.md §2): the implied-basis discrepancy (ln F_model − ln F_market)/T.
    // A 1bp basis error maps to ~1bp REGARDLESS of tenor, so short-dated forwards are not swamped by 1y.
    using std::log;
    return (log(instrument_model_quote<Scalar>(ins, C)) - std::log(market)) / ins.fx_time;
  }
  const bool banded = ins.band_upper > ins.band_lower;
  // Rate keeps its bit-exact `rate + (convexity - market)` association when there is NO band (design §2's
  // backward-compatibility invariant); a banded Rate uses the general q·weight form.
  if (ins.quote == QuoteKind::Rate && !banded)
    return pricing::rate<Scalar>(ins.obs, C(ins.forecast)) + (ins.convexity - market);
  const Scalar q = instrument_model_quote<Scalar>(ins, C);
  if (banded) return band_residual<Scalar>(q, market, ins.band_lower, ins.band_upper, ins.band_decay);
  return q - market;
}
// Calibration default: residual against the instrument's stored mid. Bit-identical to the pre-override
// code (same value flows into the same expressions), so every existing caller is unchanged.
template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C) {
  return instrument_residual<Scalar>(ins, C, ins.market);
}

struct CalibrationProblem {
  // Curve topology (year fractions on the curve day count).
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  // Every calibration instrument is a generic Instrument (design §3). This problem has exactly ONE
  // curve, so every leg role resolves to it and the legs' curve indices are ignored (single_curve_bundle()
  // likewise forces them all to curve 0, so the templated and compiled paths cannot diverge).
  std::vector<Instrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r in rate units for an ARBITRARY curve (anything with `Scalar discount(double)`). RESIDUAL ORDER
  // is the instruments' INSERTION order (the Jacobian rows, the W-cache batches and the warm/streaming
  // feeds all index off it; CompiledResidual fills the same rows).
  template <class Scalar, class Curve>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> price_residuals(const Curve& c) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    const auto one_curve = [&c](int) -> const Curve& { return c; };  // every role IS this curve
    int i = 0;
    for (const auto& ins : instruments) r[i++] = instrument_residual<Scalar>(ins, one_curve);
    return r;
  }

  // r(x) with x = the knot forwards of a standalone two-region curve.
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times));
    c.set_forwards(x);
    return price_residuals<Scalar>(c);
  }
};

// NOTE: fixed-base spread calibration (the former SpreadCalibrationProblem, with its SpreadCurve) was
// TEST-ONLY -- production spreads calibrate jointly through the bundle path (SpreadHandle). Both now live
// in tests/spread_reference.hpp (namespace swaps::testing) so they don't sit in the shipped object model.

}  // namespace swaps::calibration
