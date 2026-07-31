#pragma once
// A swap portfolio and its NPV kernel.
//
// QuantLib-free and templated on Scalar: cashflow schedules are extracted from QuantLib once, then
// the portfolio NPV is a plain function of curve discount factors. With Scalar = double this prices;
// with Scalar = AutoDiffScalar it yields d(NPV)/d(knot forwards) in one pass (see calibration/risk).

#include <cassert>
#include <vector>

#include "swaps/pricing/cashflows.hpp"

namespace swaps::portfolio {

struct Portfolio {
  // A position is a swap in the GENERIC coupon model: a floating leg + a fixed leg (unit-notional
  // coupon shapes), a contract fixed rate and a notional. Single self-discounting curve, so the
  // floating leg forecasts and discounts off the same curve.
  struct Position {
    std::vector<pricing::FloatCoupon> float_coupons;
    std::vector<pricing::FixedCoupon> fixed_coupons;
    double fixed_rate;
    double notional;
  };
  std::vector<Position> positions;

  // NPV of one position per unit notional: float_leg_pv - fixed_rate * annuity (payer-of-fixed).
  template <class Scalar, class Curve>
  static Scalar position_npv(const Position& p, const Curve& c) {
    return pricing::float_leg_pv<Scalar>(p.float_coupons, c, c) -
           p.fixed_rate * pricing::annuity<Scalar>(p.fixed_coupons, c);
  }

  // Total NPV. AAD-safe: seed the accumulator from the first position (carries derivatives).
  template <class Scalar, class Curve>
  Scalar npv(const Curve& c) const {
    assert(!positions.empty());
    Scalar total = positions[0].notional * position_npv<Scalar>(positions[0], c);
    for (std::size_t i = 1; i < positions.size(); ++i)
      total += positions[i].notional * position_npv<Scalar>(positions[i], c);
    return total;
  }
};

// =================================================================================================
// MULTI-CURVE + XCCY BOOK  (the batched reprice kernel behind api::BundleSession::price_portfolio)
// =================================================================================================
// The single-curve `Portfolio` above forecasts and discounts one curve. A real book off a calibrated
// BUNDLE is multi-curve: a vanilla swap FORECASTS one curve and DISCOUNTS another (EURIBOR-3M forecast,
// ESTR discount), and an xccy position touches the bundle's basis/xccy curve via an FX-forward notional
// reset. This kernel prices such a book against a bundle's per-curve handles, curves referenced by
// their BUNDLE INDEX. It is the same reuse the calibration ParRate/XccyMtmBasis residuals already make:
// it calls the identical pricing primitives (float_leg_pv / annuity / xccy_mtm_leg_pv, cashflows.hpp),
// so it can never drift from what calibration prices.
//
// `CurveOf` is any callable mapping a bundle curve index -> a curve object exposing `Scalar discount(t)`
// (the bundle's CurveHandle<Scalar>, exactly what build_bundle_curves() returns). Templated on Scalar so
// Scalar=double prices and Scalar=ad::Dual yields d(NPV)/d(knot forwards) in ONE differentiated pass
// (the PV01 pass) — the same dual-use the single-curve Portfolio has.
struct MultiCurveBook {
  enum class Kind { Swap, Xccy };
  struct Position {
    Kind kind = Kind::Swap;
    double notional = 1.0;

    // --- vanilla multi-curve swap (payer-of-fixed) AND the DOMESTIC leg of an xccy position ---
    std::vector<pricing::FloatCoupon> float_coupons;  // the floating leg
    int fwd_curve = 0;                                 // curve that FORECASTS the float index
    int disc_curve = 0;                                // curve that DISCOUNTS the float payments
    std::vector<pricing::FixedCoupon> fixed_coupons;   // the fixed annuity (empty => float-only leg)
    int fixed_curve = 0;                               // curve that discounts the fixed leg
    double fixed_rate = 0.0;                           // contract fixed rate (payer pays this)

    // --- xccy resetting FOREIGN funding leg (Kind::Xccy only) ---
    // Its coupon notional resets to the FX forward N_i = fx_spot·DF[reset_num]/DF[reset_den], so the leg
    // genuinely depends on the bundle's xccy/basis curve (one of reset_num/reset_den) AND on fx_spot.
    std::vector<pricing::FloatCoupon> mtm_coupons;
    int mtm_fwd_curve = 0, mtm_disc_curve = 0;         // foreign leg forecast / discount roles
    int mtm_reset_num = 0, mtm_reset_den = 0;          // FX-forward numerator (foreign) / denominator (dom)
    double fx_spot = 1.0;
  };
  std::vector<Position> positions;

  // Value of ONE position, in the discount curve's currency, per the model. `C(i)` maps a bundle curve
  // index to its handle. AAD-safe: every returned expression is seeded from a curve-dependent leg PV, and
  // `notional`/`fixed_rate` fold in as plain `double` factors (double·Scalar preserves the derivatives).
  template <class Scalar, class CurveOf>
  static Scalar position_value(const Position& p, const CurveOf& C) {
    if (p.kind == Kind::Xccy) {
      // Net MtM (in the discount currency) of RECEIVING the foreign resetting funding leg and PAYING the
      // domestic float leg. The mtm leg carries the FX conversion (N_i = fx_spot·DF_num/DF_den) so it is
      // already in domestic units; both legs depend on their own forecast+discount curves, and the mtm
      // notional couples in the xccy/basis curve via reset_num/reset_den. See cashflows.hpp xccy_mtm_leg_pv.
      const Scalar mtm = pricing::xccy_mtm_leg_pv<Scalar>(
          p.mtm_coupons, p.fx_spot, C(p.mtm_fwd_curve), C(p.mtm_disc_curve),
          C(p.mtm_reset_num), C(p.mtm_reset_den));
      const Scalar dom = pricing::float_leg_pv<Scalar>(p.float_coupons, C(p.fwd_curve), C(p.disc_curve));
      return p.notional * (mtm - dom);
    }
    // Vanilla multi-curve swap, payer-of-fixed: NPV/notional = float_leg_pv(fwd, disc) − fixed_rate·annuity.
    const Scalar fpv = pricing::float_leg_pv<Scalar>(p.float_coupons, C(p.fwd_curve), C(p.disc_curve));
    if (p.fixed_coupons.empty()) return p.notional * fpv;  // float-only leg
    const Scalar ann = pricing::annuity<Scalar>(p.fixed_coupons, C(p.fixed_curve));
    return p.notional * (fpv - p.fixed_rate * ann);
  }

  // Total book NPV. AAD-safe: the accumulator seeds from the first position (which carries derivatives).
  // Precondition: at least one position (the api layer short-circuits an empty book to NPV 0).
  template <class Scalar, class CurveOf>
  Scalar value(const CurveOf& C) const {
    assert(!positions.empty());
    Scalar total = position_value<Scalar>(positions[0], C);
    for (std::size_t i = 1; i < positions.size(); ++i) total += position_value<Scalar>(positions[i], C);
    return total;
  }
};

}  // namespace swaps::portfolio
