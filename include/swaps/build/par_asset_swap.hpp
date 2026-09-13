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
// What `dirty_purchase` is decides WHICH par asset swap:
//   * at a PAR purchase (dirty = 1 + accrued): the par-par spread;
//   * at the MARKET dirty: the par asset-swap spread at that price, QuantLib::AssetSwap(parSwap=true) at the clean
//     price. It widens as the bond cheapens. It is NOT the proceeds (market-value) spread -- that is this number
//     divided by the dirty purchase price (QuantLib parSwap=false). CORRECTED 2026-09-13: this comment used to call
//     the market-price number "proceeds" and the par-par spread "price-independent", both wrong -- the par spread
//     depends on the price through the upfront (tests/asset_swap_oracle_test.cpp pins the identity).
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/build/bond.hpp"         // BondTerms, bond_from_terms
#include "swaps/build/conventions.hpp"  // swap_conv
#include "swaps/build/day_count.hpp"    // year_frac
#include "swaps/build/schedule.hpp"     // swap_periods_between, curve_time
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

// =================================================================================================
// asset_swap_analytics — the `asset_swap` verb's per-bond computation (E7 stage 3.5).
// =================================================================================================
// A bond in an asset-swap request. The float leg is the bond currency's default swap product (or `index`'s par
// product when given); the purchase price is AT MOST ONE of clean / dirty, and neither means a purchase at par
// (clean 1). NB the spread is the PAR asset-swap spread at that purchase price -- QuantLib AssetSwap(parSwap=true)
// -- NOT the proceeds (market-value) spread, which is this divided by the dirty purchase price.
struct AssetSwapBond {
  std::string convention;  // bonds[] id -- REQUIRED
  Date issue, settle, maturity;
  double coupon = 0.0;
  std::optional<int> freq;             // overrides the convention's coupon frequency
  std::string index;                   // float leg index; empty => the currency's default swap product
  std::optional<double> clean, dirty;  // at most one, per unit notional
};

struct AssetSwapAnalytics {
  double asw_spread = 0.0, clean_curve = 0.0, dirty_curve = 0.0, annuity = 0.0, accrued = 0.0;
};

// The float leg an asset swap receives, as payment times and accruals: the swap product's periods from settlement to
// the business-day-adjusted maturity, every boundary rolled BACKWARD from maturity on the maturity's own day of month
// (a short front stub). This is build::swap_periods_between; until 2026-09-13 a hand loop CHAINED the roll from the
// previous date, so a clamped day stayed clamped (a 2036-02-29 maturity got a 2028-02-28 boundary).
struct FloatLegTimes {
  std::vector<double> pay, tau;
};
inline FloatLegTimes asset_swap_float_leg(const SwapConv& swc, const Date& value_date, const Date& settle,
                                          const Date& maturity) {
  ScheduleRule rule;
  rule.side = StubSide::Front;
  rule.roll_dom = int(maturity.day());  // the UNADJUSTED maturity's day: the anchor below is the adjusted date
  const std::vector<Period> periods =
      swap_periods_between(settle, swc.calendar, adjust(swc.calendar, maturity, swc.bdc), swc.float_freq_tok, swc.bdc, rule);
  FloatLegTimes f;
  f.pay.reserve(periods.size());
  f.tau.reserve(periods.size());
  for (const Period& p : periods) {
    f.tau.push_back(year_frac(swc.float_dc, p.first, p.second));
    f.pay.push_back(curve_time(value_date, p.second));
  }
  return f;
}

template <class Curve>
AssetSwapAnalytics asset_swap_analytics(const AssetSwapBond& b, const Date& value_date, const Curve& curve) {
  if (b.clean && b.dirty) throw std::invalid_argument("asset_swap: give at most one of 'clean' or 'dirty'");
  if (b.settle < value_date)
    throw std::invalid_argument("asset_swap: settlement is before the value date (the curve starts there)");
  BondTerms t;
  t.convention = b.convention;
  t.settle = b.settle;
  t.issue = b.issue;
  t.maturity = b.maturity;
  t.coupon = b.coupon;
  t.freq = b.freq;
  const BuiltBond bond = bond_from_terms(t, value_date);
  const SwapConv swc = swap_conv(std::string(conventions::require_bond(b.convention).currency), b.index);
  const FloatLegTimes leg = asset_swap_float_leg(swc, value_date, b.settle, b.maturity);
  const double dirty_purchase = b.dirty ? *b.dirty : (b.clean ? *b.clean : 1.0) + bond.accrued;

  AssetSwapAnalytics a;
  a.accrued = bond.accrued;
  a.dirty_curve = pricing::bond_dirty_price<double>(bond.curve, curve);
  a.clean_curve = a.dirty_curve - bond.accrued;
  a.annuity = float_annuity(curve, curve.discount(bond.curve.settle), leg.pay, leg.tau);
  a.asw_spread = par_asset_swap_spread(bond.curve, curve, dirty_purchase, leg.pay, leg.tau);
  return a;
}

template <class Curve>
std::vector<AssetSwapAnalytics> asset_swaps(const std::vector<AssetSwapBond>& bonds, const Date& value_date,
                                            const Curve& curve) {
  std::vector<AssetSwapAnalytics> out;
  out.reserve(bonds.size());
  for (const AssetSwapBond& b : bonds) out.push_back(asset_swap_analytics(b, value_date, curve));
  return out;
}

}  // namespace swaps::build
