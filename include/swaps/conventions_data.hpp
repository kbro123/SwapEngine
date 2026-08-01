#pragma once
// GENERATED from conventions/conventions.json by tools/gen_conventions_hpp.py -- DO NOT EDIT.
// The market-conventions DB is the single source of truth; re-run the generator after editing the
// JSON (conventions_test.cpp fails if this header is stale). See conventions/conventions.json.
#include <array>
#include <cctype>
#include <cstddef>
#include <optional>
#include <string_view>

namespace swaps::conventions {

struct LegConv {
  std::string_view index, day_count, frequency, compounding;
  int fixing_lag; bool carries_spread, notional_resets, flat;
};
struct ProductConv {
  std::string_view id, currency, calendar, bdc;
  int spot_lag, payment_lag;
  LegConv fixed, floating;
};
struct IndexConv {
  std::string_view id, currency, type, day_count, calendar, par_product, tenor;
  int fixing_lag, publication_lag;
};

inline constexpr std::array<ProductConv, 12> kProducts = {{
  {"EUR-3S6S-BASIS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "", "", "", -1, false, false, false}, {"EUR-EURIBOR-3M", "ACT/360", "3M", "", -1, true, false, false}},
  {"EUR-ESTR-OIS", "EUR", "EUR", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"EUR-ESTR", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"EUR-EURIBOR-3M-IRS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "30E/360", "1Y", "", -1, false, false, false}, {"EUR-EURIBOR-3M", "ACT/360", "3M", "", 2, false, false, false}},
  {"EUR-EURIBOR-6M-IRS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "30E/360", "1Y", "", -1, false, false, false}, {"EUR-EURIBOR-6M", "ACT/360", "6M", "", 2, false, false, false}},
  {"FX-FWD-EURUSD", "", "EURUSD", "", 2, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-FEDFUNDS-1M-FUTURE", "USD", "USD-FED", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-FEDFUNDS-OIS", "USD", "USD-FED", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"USD-FEDFUNDS", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"USD-PRIME", "USD", "USD-FED", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-1M-FUTURE", "USD", "USD-SOFR", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-3M-FUTURE", "USD", "USD-SOFR", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-OIS", "USD", "USD-SOFR", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"USD-SOFR", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"XCCY-MTM-EURUSD", "", "EURUSD", "", 2, 2, {"", "", "", "", -1, false, false, false}, {"USD-SOFR", "ACT/360", "", "compounded", -1, false, true, true}},
}};

inline constexpr std::array<IndexConv, 5> kIndices = {{
  {"EUR-ESTR", "EUR", "overnight", "ACT/360", "EUR", "EUR-ESTR-OIS", "", -1, 1},
  {"EUR-EURIBOR-3M", "EUR", "ibor", "ACT/360", "EUR", "EUR-EURIBOR-3M-IRS", "3M", 2, -1},
  {"EUR-EURIBOR-6M", "EUR", "ibor", "ACT/360", "EUR", "EUR-EURIBOR-6M-IRS", "6M", 2, -1},
  {"USD-FEDFUNDS", "USD", "overnight", "ACT/360", "USD-FED", "USD-FEDFUNDS-OIS", "", -1, 1},
  {"USD-SOFR", "USD", "overnight", "ACT/360", "USD-SOFR", "USD-SOFR-OIS", "", -1, 1},
}};

inline std::optional<ProductConv> product(std::string_view id) {
  for (const auto& p : kProducts) if (p.id == id) return p;
  return std::nullopt;
}
inline std::optional<IndexConv> index(std::string_view id) {
  for (const auto& i : kIndices) if (i.id == id) return i;
  return std::nullopt;
}

// Approximate year-fraction of a frequency/tenor token ('3M'->0.25, '6M'->0.5, '1Y'->1.0), for the
// coupon-period length a curve build needs when it only has year fractions (conventions_db.period_years).
inline double period_years(std::string_view tok) {
  if (tok.empty()) return 0.0;
  const char u = char(std::toupper((unsigned char)tok.back()));
  double unit = 1.0;
  switch (u) { case 'D': unit = 1.0 / 365.0; break; case 'W': unit = 7.0 / 365.0; break;
               case 'M': unit = 1.0 / 12.0; break; case 'Y': unit = 1.0; break; default: return 0.0; }
  int n = 0;
  for (std::size_t k = 0; k + 1 < tok.size(); ++k) { if (tok[k] < '0' || tok[k] > '9') return 0.0; n = n * 10 + (tok[k] - '0'); }
  return n * unit;
}

}  // namespace swaps::conventions
