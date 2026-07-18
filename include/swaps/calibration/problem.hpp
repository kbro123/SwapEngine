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

#include <vector>

#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/spread_curve.hpp"
#include "swaps/curve/calibration_curve.hpp"
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
};

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

  // The curve this instrument primarily PINS (its quoted leg's forecast curve). Used by the staged
  // solver to assign the instrument to a dependency block.
  int primary_curve() const { return quote == QuoteKind::Rate ? forecast : fwd.forecast; }
};

// Model quote of an instrument, per the design §3 table. `C(role)` maps a curve role index to the
// curve object (anything with `Scalar discount(double)`); a single-curve problem passes a lambda that
// returns its one curve for every role.
template <class Scalar, class CurveOf>
Scalar instrument_model_quote(const Instrument& ins, const CurveOf& C) {
  switch (ins.quote) {
    case QuoteKind::Rate:
      return pricing::rate<Scalar>(ins.obs, C(ins.forecast)) + ins.convexity;
    case QuoteKind::ParSpread:
      return (pricing::float_leg_pv<Scalar>(ins.bench.coupons, C(ins.bench.forecast),
                                            C(ins.bench.discount)) -
              pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount))) /
             pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
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
template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C) {
  if (ins.quote == QuoteKind::Rate)
    return pricing::rate<Scalar>(ins.obs, C(ins.forecast)) + (ins.convexity - ins.market);
  return instrument_model_quote<Scalar>(ins, C) - ins.market;
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
    auto c = curve::make_calibration_curve<Scalar>(meeting_times, back_times);
    c.set_forwards(x);
    return price_residuals<Scalar>(c);
  }
};

// Calibrating a spread to a FIXED base curve. Reuses the instrument set + pricing of a
// CalibrationProblem (its meeting_times/back_times are the SPREAD knot times); the free variables
// are the spread forwards s. Duck-types with CalibrationProblem (residuals / n_knots / n_residuals),
// so the same LM + AAD + risk code drives it with no change.
struct SpreadCalibrationProblem {
  CalibrationProblem inst;                             // instruments + spread knot times
  const curve::CalibrationCurve<double>* base;    // fixed base curve

  int n_knots() const { return inst.n_knots(); }
  int n_residuals() const { return inst.n_residuals(); }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& s) const {
    curve::SpreadCurve<Scalar> c(*base, inst.meeting_times, inst.back_times);
    c.set_spreads(s);
    return inst.price_residuals<Scalar>(c);
  }
};

}  // namespace swaps::calibration
