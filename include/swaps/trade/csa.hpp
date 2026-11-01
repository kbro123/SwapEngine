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
// OIS. This object therefore DERIVES its discount index from the conventions DB's currencies[] row
// (currencies[].discount_index: SOFR for USD, ESTR for EUR), so it can never drift from the registry that
// already drives calibration. It mirrors the reference-data value object build::Index: value semantics,
// header-only, QuantLib-free. The economic terms (threshold, MTA, independent amount, rounding) are plain
// data. (The typed market::Currency it used to delegate to was retired on 2026-11-01: no production reader.)

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "swaps/build/ref_data.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::trade {

namespace bld = swaps::build;

// A COLLATERAL AGREEMENT — a typed reference object describing how a trade is collateralised, and therefore
// which curve it discounts on. Value semantics; holds the collateral terms and reads the discount-curve choice
// (the collateral currency's OIS) off the conventions DB.
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

  // The discount index id ("USD-SOFR", "EUR-ESTR", ...) = the collateral currency's OIS, currencies[].discount_index
  // in the conventions DB (baked or overlay); EMPTY for an unknown collateral currency (the callers --
  // discount_index_for, the exposure verb -- refuse an empty id with their own message).
  std::string discount_index_id() const {
    const auto c = swaps::conventions::currency(collateral_currency);
    return c ? std::string(c->discount_index) : std::string();
  }
  // The index whose curve this trade DISCOUNTS on under this CSA; an unknown collateral currency yields an unknown
  // (empty) Index, exactly as build::Index treats an unknown id.
  bld::Index discount_index() const { return bld::Index{discount_index_id()}; }
};

// The index a booked trade DISCOUNTS on: its CSA's collateral-currency OIS when it has a CSA, else an explicitly
// named discount index. Neither -- or a CSA whose currency resolves no OIS -- THROWS: discounting on the trade's
// own forecast index is never assumed. One rule for every entry point (api book_from_json decodes into it, E7).
inline std::string discount_index_for(const std::optional<CSA>& csa, const std::string& explicit_index) {
  std::string id = csa ? csa->discount_index_id() : explicit_index;
  if (id.empty())
    throw std::invalid_argument("trade: needs a 'csa' (collateral_currency) or an explicit 'discount_index' — "
                                "discounting on the trade's own index is not assumed");
  return id;
}

}  // namespace swaps::trade
