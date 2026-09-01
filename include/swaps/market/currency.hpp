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

// ISO minor units (decimal places) — the one piece of currency metadata the conventions DB index rows do not
// carry. Explicit table, exactly as the calendar holiday rules are explicit; a new currency is one data row.
// Most world currencies use 2; JPY is the common 0-decimal case.
struct MinorUnits {
  std::string_view code;
  int units;
};
inline constexpr std::array<MinorUnits, 4> kMinorUnits = {{
    {"USD", 2},
    {"EUR", 2},
    {"GBP", 2},
    {"JPY", 0},
}};
inline int minor_units_for(std::string_view code) {
  for (const auto& m : kMinorUnits)
    if (m.code == code) return m.units;
  return 2;  // ISO default: the vast majority of currencies quote to 2 decimal places
}

// The default OIS / discount benchmark per currency. A currency may carry several overnight indices in the DB
// (USD lists both SOFR and FedFunds); this pins the modern discount benchmark. Currencies NOT listed here fall
// back to the first overnight index the DB lists for them, so the registry still populates for any new one.
struct DiscountPref {
  std::string_view code, index;
};
inline constexpr std::array<DiscountPref, 2> kDiscountPref = {{
    {"USD", "USD-SOFR"},
    {"EUR", "EUR-ESTR"},
}};

// One derived registry row. `settlement_calendar` is the currency-level holiday calendar id, which the DB names
// with the ISO code itself ("USD", "EUR") — distinct from an index's own fixing calendar (e.g. "USD-SOFR").
struct CurrencyInfo {
  std::string code;
  std::string discount_index;       // default OIS index id (from the DB overnight indices)
  std::string settlement_calendar;  // currency-level calendar id (== the ISO code)
};

// The known-currency set, DERIVED once from the DB's overnight indices. Kept DB-driven in spirit: a currency
// appears here iff the conventions DB carries an overnight index for it, and its discount index/calendar are
// read straight off that DB row.
inline const std::vector<CurrencyInfo>& currency_registry() {
  static const std::vector<CurrencyInfo> reg = [] {
    std::vector<CurrencyInfo> r;
    const auto find = [&r](std::string_view c) -> CurrencyInfo* {
      for (auto& e : r)
        if (e.code == c) return &e;
      return nullptr;
    };
    for (const auto& ix : cvd::kIndices) {
      if (ix.type != std::string_view("overnight")) continue;
      CurrencyInfo* e = find(ix.currency);
      if (!e) {
        // First overnight index seen for this currency seeds it (calendar = the currency-level id == code).
        r.push_back({std::string(ix.currency), std::string(ix.id), std::string(ix.currency)});
        e = &r.back();
      }
      // Pin the preferred discount benchmark when the DB offers more than one overnight index for the currency.
      for (const auto& p : kDiscountPref)
        if (p.code == ix.currency && p.index == ix.id) e->discount_index = std::string(ix.id);
    }
    return r;
  }();
  return reg;
}

// A CURRENCY — a typed reference object addressed by its ISO code. PURE reference data: minor units, the
// default settlement calendar, and the default OIS/discount index, each exposed as its own typed object. Value
// semantics; holds only the code and delegates, mirroring build::Index. Unknown codes stay valid-but-empty.
struct Currency {
  std::string code;  // ISO code, public like build::Index::id / build::Calendar::id

  Currency() = default;
  explicit Currency(std::string code_) : code(std::move(code_)) {}

  // Registry lookup. An unrecognised code returns a valid Currency that carries the code but no DB data
  // (known()==false), exactly as build::Index{"NOT-AN-INDEX"} stays valid with empty conventions.
  static Currency of(std::string code) { return Currency(std::move(code)); }

  // The known currencies, in DB order — a currency is known iff the conventions DB carries an overnight index.
  static std::vector<std::string> known_codes() {
    std::vector<std::string> v;
    for (const auto& e : currency_registry()) v.push_back(e.code);
    return v;
  }

  bool valid() const { return !code.empty(); }        // a usable value object (non-empty code)
  bool known() const { return info() != nullptr; }    // recognised in the DB-derived registry

  // ISO minor units. Available even for a currency without a DB index (table lookup), like build::Index's
  // day-count fallback; defaults to 2 for anything unlisted.
  int minor_units() const { return minor_units_for(code); }

  // The currency's default settlement (holiday) calendar. Empty calendar (weekends-only) for an unknown code.
  bld::Calendar settlement_calendar() const {
    const auto* e = info();
    return bld::Calendar{e ? e->settlement_calendar : std::string()};
  }

  // The currency's default OIS / discount index (SOFR for USD, ESTR for EUR). Empty (unknown) Index for an
  // unknown code; the returned Index carries its own conventions and currency straight off the DB.
  bld::Index discount_index() const {
    const auto* e = info();
    return bld::Index{e ? e->discount_index : std::string()};
  }

 private:
  const CurrencyInfo* info() const {
    for (const auto& e : currency_registry())
      if (e.code == code) return &e;
    return nullptr;
  }
};

}  // namespace swaps::market
