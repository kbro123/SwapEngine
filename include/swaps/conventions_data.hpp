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
// A BOND convention. `stub_discount` is the one thing the per-flow exponent cannot express (see
// pricing/bond.hpp YieldConvention): "compound" -> dirty = Q(v)*v^w, "simple" -> Q(v)/(1 + w*y/f).
// `final_period_simple` forces the simple form once a single cashflow remains (US street, Bund).
struct BondConv {
  std::string_view id, currency, calendar, day_count, frequency, stub_discount;
  int settle_lag; bool final_period_simple;
};
// A HOLIDAY rule (calendars[].holidays in the JSON) — interpreted by swaps/build/calendar.hpp.
// kind: "fixed" (month/day, from_year 0 = always), "nth_weekday" (month/weekday/n),
// "last_weekday" (month/weekday), "easter_offset" (days vs Easter Sunday). weekday is Mon=0..Sun=6.
// observance: "" = inherit the calendar default; else "none" | "sat_to_fri_sun_to_mon" | "sun_to_mon".
struct HolidayRule {
  std::string_view kind;
  int month, day, weekday, n, days, from_year;
  std::string_view observance;
};
// A CALENDAR: either rule-based (rule_count > 0) or a JOIN of other calendars (closed if any leg is
// closed). `weekend_mask` bit w (Mon=0..Sun=6) marks a weekend day. Rules/joins are slices of the flat
// kHolidayRules / kCalendarJoins arrays below.
struct CalendarConv {
  std::string_view id, name, observance;
  int weekend_mask;
  std::size_t rule_begin, rule_count, join_begin, join_count;
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

inline constexpr std::array<BondConv, 2> kBonds = {{
  {"US-TREASURY", "USD", "USD", "ACT/ACT-ICMA", "6M", "compound", 1, true},
  {"US-TREASURY-TSY", "USD", "USD", "ACT/ACT-ICMA", "6M", "simple", 1, false},
}};

inline constexpr std::array<HolidayRule, 41> kHolidayRules = {{
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 1, 0, 0, 3, 0, 0, "none"},
  {"nth_weekday", 2, 0, 0, 3, 0, 0, "none"},
  {"easter_offset", 0, 0, -1, 0, -2, 0, "none"},
  {"last_weekday", 5, 0, 0, 0, 0, 0, "none"},
  {"fixed", 6, 19, -1, 0, 0, 2021, ""},
  {"fixed", 7, 4, -1, 0, 0, 0, ""},
  {"nth_weekday", 9, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 10, 0, 0, 2, 0, 0, "none"},
  {"fixed", 11, 11, -1, 0, 0, 0, ""},
  {"nth_weekday", 11, 0, 3, 4, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 1, 0, 0, 3, 0, 0, "none"},
  {"nth_weekday", 2, 0, 0, 3, 0, 0, "none"},
  {"last_weekday", 5, 0, 0, 0, 0, 0, "none"},
  {"fixed", 6, 19, -1, 0, 0, 2021, ""},
  {"fixed", 7, 4, -1, 0, 0, 0, ""},
  {"nth_weekday", 9, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 10, 0, 0, 2, 0, 0, "none"},
  {"fixed", 11, 11, -1, 0, 0, 0, ""},
  {"nth_weekday", 11, 0, 3, 4, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 1, 0, 0, 3, 0, 0, "none"},
  {"nth_weekday", 2, 0, 0, 3, 0, 0, "none"},
  {"easter_offset", 0, 0, -1, 0, -2, 0, "none"},
  {"last_weekday", 5, 0, 0, 0, 0, 0, "none"},
  {"fixed", 6, 19, -1, 0, 0, 2021, ""},
  {"fixed", 7, 4, -1, 0, 0, 0, ""},
  {"nth_weekday", 9, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 10, 0, 0, 2, 0, 0, "none"},
  {"fixed", 11, 11, -1, 0, 0, 0, ""},
  {"nth_weekday", 11, 0, 3, 4, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
}};

inline constexpr std::array<std::string_view, 2> kCalendarJoins = {{
  "USD",
  "EUR",
}};

inline constexpr std::array<CalendarConv, 5> kCalendars = {{
  {"EUR", "TARGET (EUR settlement)", "none", 96, 0, 6, 0, 0},
  {"EURUSD", "Joint US-SIFMA + TARGET (FX/xccy USD side = SIFMA government-bond)", "", 96, 6, 0, 0, 2},
  {"USD", "US SIFMA / US government securities (bond market)", "sat_to_fri_sun_to_mon", 96, 6, 12, 2, 0},
  {"USD-FED", "US Federal Reserve (Fedwire)", "sun_to_mon", 96, 18, 11, 2, 0},
  {"USD-SOFR", "SOFR fixing calendar (SIFMA, incl. Good Friday close)", "sat_to_fri_sun_to_mon", 96, 29, 12, 2, 0},
}};

inline std::optional<CalendarConv> calendar(std::string_view id) {
  for (const auto& c : kCalendars) if (c.id == id) return c;
  return std::nullopt;
}
inline std::optional<BondConv> bond(std::string_view id) {
  for (const auto& b : kBonds) if (b.id == id) return b;
  return std::nullopt;
}
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
