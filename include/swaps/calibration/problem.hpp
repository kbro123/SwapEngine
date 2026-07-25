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
  // fx_spot · DF[reset_num](reset)/DF[reset_den](reset) -- priced by pricing::xccy_mtm_leg_pv on the
  // AAD/templated path (a curve-dependent notional is not a single exp(-Wx), so it is never W-cached).
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
  // Cross-currency (multi-currency). NEITHER is W-cacheable (a DF ratio / a curve-dependent notional is
  // not a single exp(-Wx)); CompiledBundleResidual rejects them, so they ride the AAD/templated path.
  FxForward,      // FX-forward point: fx_spot · DF[fx_num](fx_time)/DF[fx_den](fx_time) (pins fx_num vs fx_den)
  XccyMtmBasis,   // MtM (FX-resettable-notional) xccy basis: par basis incl. the resetting funding leg
  Portfolio,      // linear combination of component instruments: model quote = Σ weight·quote(component).
                  // `market` is the COMBINED quote (a butterfly/condor spread), so you calibrate to the
                  // combo directly without pinning each leg's outright rate. Components are full nested
                  // Instruments, so portfolios compose. ONE residual, no knots (knots are in the curve
                  // spec). Not W-cacheable -> rides the AAD/templated path.
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
  // units) the residual is a WEIGHTED pull to mid: r = w(q)·(q − market), with the weight w decaying from
  // 1 outside the band down to a floor `band_decay` inside it (see band_weight()). So the solver treats
  // any model value within [lower, upper] as ~satisfied and spends its freedom on the hard targets. The
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
  if (quote == QuoteKind::Portfolio)
    return combination.empty() ? 0 : combination.front().instrument.primary_curve();
  return fwd.forecast;
}

// Bid/offer band weight for a model quote q (see the Instrument band fields). Returns 1 when there is no
// band (upper <= lower). Otherwise w = decay + (1-decay)·(1 - exp(-(outside/s)^2)), where `outside` is
// the distance of q OUTSIDE [lower, upper] (0 inside the band) and s = upper - lower. w is 1 far outside
// the band and decays smoothly to the floor `decay` inside it; it is C1 across the edges (dw/dq -> 0 as
// q approaches an edge from outside), so LM/AAD see no kink. Max is done by branch selection so it is
// AAD-safe (the derivative flows through the selected term; inside the band `outside` is the constant 0).
template <class Scalar>
Scalar band_weight(const Scalar& q, double lower, double upper, double decay) {
  if (!(upper > lower)) return Scalar(1.0);
  using std::exp;
  const Scalar below = Scalar(lower) - q;   // > 0 below the band
  const Scalar above = q - Scalar(upper);   // > 0 above the band
  Scalar outside = below > above ? below : above;      // max(below, above)
  if (outside < Scalar(0.0)) outside = Scalar(0.0);     // inside the band -> 0 (constant, zero derivative)
  const Scalar z = outside / (upper - lower);
  return Scalar(decay) + Scalar(1.0 - decay) * (Scalar(1.0) - exp(-(z * z)));
}

// Plain-double (w, dw/dq) of band_weight, for the compiled path's ANALYTIC Jacobian. MUST match
// band_weight() above term-for-term -- a divergence would make the compiled and AAD residuals disagree
// (the compiled_residual oracle test pins exactly this equality).
inline std::pair<double, double> band_weight_d(double q, double lower, double upper, double decay) {
  if (!(upper > lower)) return {1.0, 0.0};
  const double s = upper - lower;
  const double below = lower - q, above = q - upper;
  const double outside = std::max(std::max(below, above), 0.0);
  const double doutside = outside <= 0.0 ? 0.0 : (below > above ? -1.0 : 1.0);  // d outside / dq
  const double z = outside / s, e = std::exp(-(z * z));
  const double w = decay + (1.0 - decay) * (1.0 - e);
  const double dw = (1.0 - decay) * e * 2.0 * z * (doutside / s);  // dw/dq = (1-decay)·e·2z·(doutside/s)
  return {w, dw};
}

// Model quote of an instrument, per the design §3 table. `C(role)` maps a curve role index to the
// curve object (anything with `Scalar discount(double)`); a single-curve problem passes a lambda that
// returns its one curve for every role.
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
// A banded instrument (band_upper > band_lower) weights its residual: r = w(q)·(q − market) with w from
// band_weight(). The band is NOT applied to FxForward (its residual is already a log-basis transform).
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
  const Scalar raw = q - market;
  return banded ? band_weight<Scalar>(q, ins.band_lower, ins.band_upper, ins.band_decay) * raw : raw;
}
// Calibration default: residual against the instrument's stored mid. Bit-identical to the pre-override
// code (same value flows into the same expressions), so every existing caller is unchanged.
template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C) {
  return instrument_residual<Scalar>(ins, C, ins.market);
}

// True iff a MtM-xccy basis has a PAR funding (mtm) leg: discount == forecast and every coupon is a plain
// single-period OIS coupon paying at its period end (no weights/spread/realized, tau_pay == tau_index,
// scale == 1). Then each funding bracket float_coupon_pv(c) + (DF_dc(e) − DF_dc(s)) is IDENTICALLY zero --
// DF(e)·(DF(s)/DF(e) − 1) + DF(e) − DF(s) = 0 for all x -- so the whole FX-reset-notional term (value AND
// derivative) vanishes and the quote collapses to the ParSpread quotient (pv_self − pv_fx)/ann, which is
// W-cacheable. A NON-par funding leg keeps a genuine curve-dependent notional and must use the AAD engine.
// (The funding leg is par by construction in a standard xccy basis, so this is the common case.)
inline bool mtm_funding_leg_is_par(const Instrument& ins) {
  if (ins.quote != QuoteKind::XccyMtmBasis) return false;
  if (ins.mtm.forecast != ins.mtm.discount) return false;
  for (const auto& c : ins.mtm.coupons) {
    const pricing::RateObservation& o = c.obs;
    if (o.compounded || o.sub_start.size() != 1 || !o.weight.empty()) return false;
    if (o.realized != 0.0 || c.spread != 0.0 || c.tau_pay != o.tau_index || c.scale != 1.0) return false;
    if (c.pay != o.sub_end.back()) return false;  // pays at period end -> the bracket is identically 0
  }
  return true;
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
