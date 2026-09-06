#pragma once
// A swap portfolio and its NPV kernel.
//
// QuantLib-free and templated on Scalar: cashflow schedules are extracted from QuantLib once, then
// the portfolio NPV is a plain function of curve discount factors. With Scalar = double this prices;
// with Scalar = AutoDiffScalar it yields d(NPV)/d(knot forwards) in one pass (see calibration/risk).

#include <cassert>
#include <utility>
#include <vector>

#include "swaps/pricing/cashflows.hpp"

namespace swaps::portfolio {

// Value of a STEPPED (per-coupon) fixed leg per unit notional: Σ_i DF(pay_i)·tau_i·scale_i·rates[i]. This
// is the generic form of `fixed_rate · annuity` with the rate folded INTO each coupon, so a booked step-up /
// amortizer-with-step / structured fixed leg needs no new coupon struct -- only a per-coupon rate vector.
// A CONSTANT schedule (all rates == r) equals r·annuity up to floating-point summation ORDER (Σ DFτr vs
// r·ΣDFτ) -- i.e. ~1e-15 relative, not bitwise; callers wanting the exact scalar leave the vector empty and
// take the annuity path. Precondition: rates.size() == leg.size() (>=1). AAD-safe: seeded from DF(pay_0).
template <class Scalar, class DCurve>
Scalar stepped_annuity_pv(const std::vector<pricing::FixedCoupon>& leg, const std::vector<double>& rates,
                          const DCurve& dc) {
  assert(!leg.empty() && rates.size() == leg.size());
  Scalar a = dc.discount(leg[0].pay) * (leg[0].tau * leg[0].scale * rates[0]);
  for (std::size_t i = 1; i < leg.size(); ++i)
    a += dc.discount(leg[i].pay) * (leg[i].tau * leg[i].scale * rates[i]);
  return a;
}

// PV of principal-exchange cashflows per unit notional: Σ_i amount_i · DF(time_i), discounted on the leg's
// discount curve. `flows` are (discount-curve time, signed amount) pairs (amount in notional units, signed
// from OUR perspective). Precondition: non-empty. AAD-safe: seeded from the first (curve-dependent) term.
template <class Scalar, class DCurve>
Scalar principal_pv(const std::vector<std::pair<double, double>>& flows, const DCurve& dc) {
  assert(!flows.empty());
  Scalar pv = dc.discount(flows[0].first) * flows[0].second;
  for (std::size_t i = 1; i < flows.size(); ++i) pv += dc.discount(flows[i].first) * flows[i].second;
  return pv;
}

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

    // --- optional booked structure (stepped fixed rate + principal exchange) ---
    // Per-coupon fixed rates for a STEP-UP / amortizer-with-step / structured swap. EMPTY (the default) =>
    // the scalar `fixed_rate` applies to every coupon, byte-identical to before; when set, its size MUST
    // equal fixed_coupons.size() and the fixed leg pays Σ DF·tau·scale·fixed_rates[i]. A stepped position is
    // NOT W-cacheable (the compiled book's nf_ row-scale is a single scalar per position), so it rides the
    // templated fallback -- CompiledMultiCurveBook::swap_is_compilable returns false for it.
    std::vector<double> fixed_rates;
    // Principal-exchange cashflows (initial / final notional exchange for xccy / resolved trades), as
    // (discount-curve time, signed amount per unit notional) pairs discounted on `disc_curve`. EMPTY => none
    // (byte-identical). Present => the position rides the fallback (extra dated flows the batch has no row for).
    std::vector<std::pair<double, double>> principal_flows;

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
    // Optional booked structure (stepped fixed rate / principal exchange) folds into the same per-unit core;
    // both default empty, so the expressions below stay byte-identical to the plain swap.
    const Scalar fpv = pricing::float_leg_pv<Scalar>(p.float_coupons, C(p.fwd_curve), C(p.disc_curve));
    Scalar core = fpv;  // seeds the derivatives (the float leg is always present for a swap)
    if (!p.fixed_coupons.empty()) {
      if (p.fixed_rates.empty())
        core = core - p.fixed_rate * pricing::annuity<Scalar>(p.fixed_coupons, C(p.fixed_curve));
      else
        core = core - stepped_annuity_pv<Scalar>(p.fixed_coupons, p.fixed_rates, C(p.fixed_curve));
    }
    if (!p.principal_flows.empty()) core = core + principal_pv<Scalar>(p.principal_flows, C(p.disc_curve));
    return p.notional * core;
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
