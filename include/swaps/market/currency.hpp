#pragma once
// A first-class Currency reference OBJECT (market-domain object model, cf. build/ref_data.hpp).
//
// The engine has so far represented a currency only as an engine-blind int tag; this adds the missing typed
// value object, addressed by ISO code ("USD", "EUR"). It mirrors the reference-data objects in
// build/ref_data.hpp exactly: a value type holding an id (here the ISO `code`), QuantLib-free, header-only,
// every query delegating to the constexpr conventions DB (conventions_data.hpp) or to a small explicit table
// for the one piece the DB does not carry (ISO minor units). A new currency is data, not code.
//
// The currency SET and each currency's default OIS/discount index + settlement calendar are DERIVED from the
// DB's overnight indices, so they cannot drift from the conventions that drive calibration. Cross-object
// accessors return the same typed reference objects the rest of the model uses: settlement_calendar() yields a
// swaps::build::Calendar, discount_index() a swaps::build::Index. Unknown codes stay valid-but-empty
// (known()==false), matching how build::Index treats an unknown id.

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "swaps/build/ref_data.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::market {

namespace cvd = swaps::conventions;
namespace bld = swaps::build;

// The currency registry IS the conventions DB (currencies[] in conventions.json, codegen'd as kCurrencies,
// plus any runtime overlay). Until 2026-09-08 minor units and the discount preference were C++ tables here
// (6 and 2 rows) and every other currency silently got "2 decimals" and "first overnight index in JSON order".
struct CurrencyInfo {
  std::string code;
  std::string discount_index;       // default RFR / discount index id (currencies[].discount_index)
  std::string settlement_calendar;  // currency-level calendar id (currencies[].settlement_calendar)
  int minor_units = -1;
};
inline std::optional<CurrencyInfo> currency_info(std::string_view code) {
  const auto c = cvd::currency(code);
  if (!c) return std::nullopt;
  return CurrencyInfo{std::string(c->code), std::string(c->discount_index), std::string(c->settlement_calendar),
                      c->minor_units};
}
inline std::vector<CurrencyInfo> currency_registry() {
  std::vector<CurrencyInfo> r;
  const auto L = cvd::Registry::instance().list_currencies();
  for (const auto& code : L.baked) if (auto i = currency_info(code)) r.push_back(*i);
  for (const auto& code : L.overlay) if (auto i = currency_info(code)) r.push_back(*i);
  return r;
}

// A CURRENCY — a typed reference object addressed by its ISO code. PURE reference data: minor units, the
// default settlement calendar, and the default OIS/discount index, each exposed as its own typed object. Value
// semantics; holds only the code and delegates, mirroring build::Index. Unknown codes stay valid-but-empty.
struct Currency {
  std::string code;  // ISO code, public like build::Index::id / build::Calendar::id

  Currency() = default;
  explicit Currency(std::string code_) : code(std::move(code_)) {}

  // An unrecognised code is a valid HANDLE (known()==false); every convention accessor throws for it.
  static Currency of(std::string code) { return Currency(std::move(code)); }

  // The known currencies (currencies[] in the DB + runtime overlay).
  static std::vector<std::string> known_codes() {
    std::vector<std::string> v;
    for (const auto& e : currency_registry()) v.push_back(e.code);
    return v;
  }

  bool valid() const { return !code.empty(); }        // a usable value object (non-empty code)
  bool known() const { return cvd::currency(code).has_value(); }  // in the DB (baked or overlay)

  // ISO minor units — a DB field (currencies[].minor_units); throws for an unknown code (no "2" default).
  int minor_units() const { return cvd::require_currency(code).minor_units; }

  // The currency's default settlement (holiday) calendar (currencies[].settlement_calendar); throws if unknown.
  bld::Calendar settlement_calendar() const {
    return bld::Calendar{std::string(cvd::require_currency(code).settlement_calendar)};
  }

  // The currency's default RFR / discount index (currencies[].discount_index); throws if unknown. The returned
  // Index carries its own conventions and currency straight off the DB.
  bld::Index discount_index() const { return bld::Index{std::string(cvd::require_currency(code).discount_index)}; }

 private:
  // (no cached pointer: the registry can grow at runtime; every accessor asks the DB)
};

}  // namespace swaps::market
