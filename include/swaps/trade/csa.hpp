#pragma once
// A first-class CSA (Credit Support Annex) reference OBJECT (trade-domain object model).
//
// Discounting in this engine has so far been chosen by a bare int `role` on a leg (see FloatLeg.discount /
// FixedLeg.discount in calibration/problem.hpp, with comments like "curve that DISCOUNTS this leg's payments"
// and "all discounted on the pinned (collateral) curve"). The object-model review found that "CSA is an int
// role, not an object": the thing that ECONOMICALLY drives the discount-curve choice — the collateral
// agreement — had no typed representation. THIS header adds it.
//
// A CSA is the collateral agreement under which a trade is funded. Its single most consequential output for
// curve construction is the discount curve: a trade collateralised in cash in currency X discounts on X's
// OIS. This object therefore DERIVES its discount index by delegating to market::Currency::discount_index()
// (SOFR for USD, ESTR for EUR), so it can never drift from the DB-derived currency registry that already
// drives calibration. It mirrors the reference-data value objects (build::Index / market::Currency): value
// semantics, header-only, QuantLib-free, every cross-object accessor returning the same typed objects the
// rest of the model uses. The economic terms (threshold, MTA, independent amount, rounding) are plain data.

#include <optional>
#include <string>
#include <utility>

#include "swaps/build/ref_data.hpp"
#include "swaps/market/currency.hpp"

namespace swaps::trade {

namespace mkt = swaps::market;
namespace bld = swaps::build;

// A COLLATERAL AGREEMENT — a typed reference object describing how a trade is collateralised, and therefore
// which curve it discounts on. Value semantics; holds the collateral terms and delegates the discount-curve
// choice to the collateral currency's OIS via market::Currency.
struct CSA {
  // How the exposure is collateralised. Cash collateral in `collateral_currency` pins discounting to that
  // currency's OIS; an uncollateralised trade has no collateral curve to pin to (it funds at the currency's
  // OIS here as a modelling default — a genuine funding/discount spread is out of scope for this object).
  enum class Type { Cash, Uncollateralized };

  std::string collateral_currency;      // the CSA currency, ISO code (e.g. "USD"); drives the discount curve
  Type type = Type::Cash;               // cash-collateralised vs uncollateralised
  double threshold = 0.0;               // unsecured exposure tolerated before collateral is called
  double mta = 0.0;                     // minimum transfer amount
  double independent_amount = 0.0;      // independent amount / initial margin (IM)
  std::optional<double> rounding;       // collateral-call rounding increment, if any

  CSA() = default;
  explicit CSA(std::string ccy, Type t = Type::Cash)
      : collateral_currency(std::move(ccy)), type(t) {}

  // --- Static factories ---------------------------------------------------------------------------------
  // A clean cash CSA in `ccy` (zero threshold / MTA / IM) — the standard collateralised case.
  static CSA cash(std::string ccy) { return CSA(std::move(ccy), Type::Cash); }
  // An uncollateralised trade in `ccy`.
  static CSA uncollateralized(std::string ccy) { return CSA(std::move(ccy), Type::Uncollateralized); }

  // --- Queries ------------------------------------------------------------------------------------------
  bool is_cash_collateralized() const { return type == Type::Cash; }

  // The collateral currency as a typed reference object (SOFR/ESTR discount index, settlement calendar, ...).
  mkt::Currency currency() const { return mkt::Currency::of(collateral_currency); }

  // The index whose curve this trade DISCOUNTS on under this CSA = the collateral currency's OIS. Delegates
  // to market::Currency::discount_index(), so it stays wired to the DB-derived currency registry. An unknown
  // collateral currency yields an unknown (empty) Index, exactly as market::Currency does.
  bld::Index discount_index() const { return currency().discount_index(); }

  // The discount index id ("USD-SOFR", "EUR-ESTR", ...); empty for an unknown collateral currency.
  std::string discount_index_id() const { return discount_index().id; }
};

}  // namespace swaps::trade
