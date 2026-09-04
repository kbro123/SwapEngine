#pragma once
// Par-par asset-swap spread — the engine's ParSpread quote for a bond. NO new pricing primitive: it reuses
// the curve-space bond PV (pricing/bond.hpp) and a plain float-leg annuity. QuantLib-free; validated against
// QuantLib::AssetSwap in tests/bond_asset_swap_oracle.cpp.
//
// Structure (notional 1): pay the bond's fixed coupons into the swap, receive a floating leg + the spread s,
// against buying the bond at a DIRTY price. The float side is a par floater on the curve, so the package NPV
// is  NPV(s) = dirty_curve − dirty_purchase + s · annuity_float,  and the fair spread zeroes it:
//     s = (dirty_purchase − dirty_curve) / annuity_float
// but with the ParSpread sign the engine uses (bond value vs purchase price):
//     s = (dirty_curve − dirty_purchase) / annuity_float
// dirty_curve = the bond's dirty price on the swap curve, annuity_float = Σ δ_i·DF(t_i)/DF(settle) (the float
// leg PV01, forward to settlement — the same basis the prices use).
//
// TWO conventions by what `dirty_purchase` is:
//   * PAR-PAR: purchase at par (dirty = 1 + accrued). Price-independent running spread — the market price is
//     monetised in the upfront. This reproduces QuantLib::AssetSwap::fairSpread EXACTLY (the oracle).
//   * PROCEEDS / market: purchase at the market dirty. The running spread then absorbs the price, so it
//     widens as the bond cheapens — the desk RV number. Reduces to the par-par value at a par purchase.
#include <vector>

#include "swaps/pricing/bond.hpp"

namespace swaps::build {

// `float_pay`/`float_tau` are the floating leg's payment times (discount-curve time) and accrual fractions;
// only their annuity Σ δ·DF enters (par-par is projection-independent). `dirty_market` is per unit notional.
template <class Curve>
double float_annuity(const Curve& curve, double df_settle, const std::vector<double>& float_pay,
                     const std::vector<double>& float_tau) {
  double a = 0.0;
  for (std::size_t i = 0; i < float_pay.size(); ++i) a += float_tau[i] * curve.discount(float_pay[i]);
  return a / df_settle;
}

template <class Curve>
double par_asset_swap_spread(const pricing::Bond& bond, const Curve& curve, double dirty_market,
                             const std::vector<double>& float_pay, const std::vector<double>& float_tau) {
  const double dirty_curve = pricing::bond_dirty_price<double>(bond, curve);
  const double annuity = float_annuity(curve, curve.discount(bond.settle), float_pay, float_tau);
  return (dirty_curve - dirty_market) / annuity;
}

}  // namespace swaps::build
