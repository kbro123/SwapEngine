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

inline constexpr std::array<ProductConv, 33> kProducts = {{
  {"ARS-BADLAR-IRS", "ARS", "ARS", "ModifiedFollowing", 2, 0, {"", "ACT/360", "1M", "", -1, false, false, false}, {"ARS-BADLAR", "ACT/360", "1M", "", 0, false, false, false}},
  {"AUD-AONIA-OIS", "AUD", "AUD", "ModifiedFollowing", 1, 2, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"AUD-AONIA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"AUD-BBSW-3M-IRS", "AUD", "AUD", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "3M", "", -1, false, false, false}, {"AUD-BBSW-3M", "ACT/365F", "3M", "", 0, false, false, false}},
  {"AUD-BBSW-6M-IRS", "AUD", "AUD", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "6M", "", -1, false, false, false}, {"AUD-BBSW-6M", "ACT/365F", "6M", "", 0, false, false, false}},
  {"BRL-CDI-SWAP", "BRL", "BRL", "Following", 1, 0, {"", "BUS/252", "1Y", "", -1, false, false, false}, {"BRL-CDI", "BUS/252", "1Y", "compounded", -1, false, false, false}},
  {"CAD-CORRA-OIS", "CAD", "CAD", "ModifiedFollowing", 1, 2, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"CAD-CORRA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"CHF-SARON-OIS", "CHF", "CHF", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"CHF-SARON", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"CNY-FR007-IRS", "CNY", "CNY", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "3M", "", -1, false, false, false}, {"CNY-FR007", "ACT/365F", "3M", "compounded", -1, false, false, false}},
  {"CNY-SHIBOR-3M-IRS", "CNY", "CNY", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "3M", "", -1, false, false, false}, {"CNY-SHIBOR-3M", "ACT/360", "3M", "", 1, false, false, false}},
  {"EUR-3S6S-BASIS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "", "", "", -1, false, false, false}, {"EUR-EURIBOR-3M", "ACT/360", "3M", "", -1, true, false, false}},
  {"EUR-ESTR-OIS", "EUR", "EUR", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"EUR-ESTR", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"EUR-EURIBOR-3M-IRS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "30E/360", "1Y", "", -1, false, false, false}, {"EUR-EURIBOR-3M", "ACT/360", "3M", "", 2, false, false, false}},
  {"EUR-EURIBOR-6M-IRS", "EUR", "EUR", "ModifiedFollowing", 2, 0, {"", "30E/360", "1Y", "", -1, false, false, false}, {"EUR-EURIBOR-6M", "ACT/360", "6M", "", 2, false, false, false}},
  {"FX-FWD-EURUSD", "", "EURUSD", "", 2, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"GBP-SONIA-OIS", "GBP", "GBP", "ModifiedFollowing", 0, 0, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"GBP-SONIA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"IDR-INDONIA-OIS", "IDR", "IDR", "ModifiedFollowing", 2, 0, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"IDR-INDONIA", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"INR-MIBOR-OIS", "INR", "INR", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"INR-MIBOR-ON", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"JPY-TONA-OIS", "JPY", "JPY", "ModifiedFollowing", 2, 2, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"JPY-TONA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"KRW-KOFR-OIS", "KRW", "KRW", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"KRW-KOFR", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"MXN-FTIIE-OIS", "MXN", "MXN", "ModifiedFollowing", 1, 0, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"MXN-FTIIE", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"MXN-TIIE-28-IRS", "MXN", "MXN", "ModifiedFollowing", 1, 0, {"", "ACT/360", "28D", "", -1, false, false, false}, {"MXN-TIIE-28", "ACT/360", "28D", "", 1, false, false, false}},
  {"RUB-RUONIA-OIS", "RUB", "RUB", "ModifiedFollowing", 1, 0, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"RUB-RUONIA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
  {"SAR-SAIBOR-3M-IRS", "SAR", "SAR", "ModifiedFollowing", 2, 0, {"", "ACT/360", "6M", "", -1, false, false, false}, {"SAR-SAIBOR-3M", "ACT/360", "3M", "", 2, false, false, false}},
  {"TRY-TLREF-OIS", "TRY", "TRY", "ModifiedFollowing", 1, 0, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"TRY-TLREF", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"USD-FEDFUNDS-1M-FUTURE", "USD", "USD-FED", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-FEDFUNDS-OIS", "USD", "USD-FED", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"USD-FEDFUNDS", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"USD-PRIME", "USD", "USD-FED", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-1M-FUTURE", "USD", "USD-SOFR", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-3M-FUTURE", "USD", "USD-SOFR", "", -1, -1, {"", "", "", "", -1, false, false, false}, {"", "", "", "", -1, false, false, false}},
  {"USD-SOFR-OIS", "USD", "USD-SOFR", "ModifiedFollowing", 2, 2, {"", "ACT/360", "1Y", "", -1, false, false, false}, {"USD-SOFR", "ACT/360", "1Y", "compounded", -1, false, false, false}},
  {"XCCY-MTM-EURUSD", "", "EURUSD", "", 2, 2, {"", "", "", "", -1, false, false, false}, {"USD-SOFR", "ACT/360", "", "compounded", -1, false, true, true}},
  {"ZAR-JIBAR-3M-IRS", "ZAR", "ZAR", "ModifiedFollowing", 0, 0, {"", "ACT/365F", "3M", "", -1, false, false, false}, {"ZAR-JIBAR-3M", "ACT/365F", "3M", "", 0, false, false, false}},
  {"ZAR-ZARONIA-OIS", "ZAR", "ZAR", "ModifiedFollowing", 0, 0, {"", "ACT/365F", "1Y", "", -1, false, false, false}, {"ZAR-ZARONIA", "ACT/365F", "1Y", "compounded", -1, false, false, false}},
}};

inline constexpr std::array<IndexConv, 26> kIndices = {{
  {"ARS-BADLAR", "ARS", "ibor", "ACT/360", "ARS", "ARS-BADLAR-IRS", "1M", 0, -1},
  {"AUD-AONIA", "AUD", "overnight", "ACT/365F", "AUD", "AUD-AONIA-OIS", "", -1, 1},
  {"AUD-BBSW-3M", "AUD", "ibor", "ACT/365F", "AUD", "AUD-BBSW-3M-IRS", "3M", 0, -1},
  {"AUD-BBSW-6M", "AUD", "ibor", "ACT/365F", "AUD", "AUD-BBSW-6M-IRS", "6M", 0, -1},
  {"BRL-CDI", "BRL", "overnight", "BUS/252", "BRL", "BRL-CDI-SWAP", "", -1, 1},
  {"CAD-CORRA", "CAD", "overnight", "ACT/365F", "CAD", "CAD-CORRA-OIS", "", -1, 1},
  {"CHF-SARON", "CHF", "overnight", "ACT/360", "CHF", "CHF-SARON-OIS", "", -1, 0},
  {"CNY-FR007", "CNY", "ibor", "ACT/365F", "CNY", "CNY-FR007-IRS", "1W", 1, -1},
  {"CNY-SHIBOR-3M", "CNY", "ibor", "ACT/360", "CNY", "CNY-SHIBOR-3M-IRS", "3M", 1, -1},
  {"EUR-ESTR", "EUR", "overnight", "ACT/360", "EUR", "EUR-ESTR-OIS", "", -1, 1},
  {"EUR-EURIBOR-3M", "EUR", "ibor", "ACT/360", "EUR", "EUR-EURIBOR-3M-IRS", "3M", 2, -1},
  {"EUR-EURIBOR-6M", "EUR", "ibor", "ACT/360", "EUR", "EUR-EURIBOR-6M-IRS", "6M", 2, -1},
  {"GBP-SONIA", "GBP", "overnight", "ACT/365F", "GBP", "GBP-SONIA-OIS", "", -1, 1},
  {"IDR-INDONIA", "IDR", "overnight", "ACT/360", "IDR", "IDR-INDONIA-OIS", "", -1, 1},
  {"INR-MIBOR-ON", "INR", "overnight", "ACT/365F", "INR", "INR-MIBOR-OIS", "", -1, 0},
  {"JPY-TONA", "JPY", "overnight", "ACT/365F", "JPY", "JPY-TONA-OIS", "", -1, 1},
  {"KRW-KOFR", "KRW", "overnight", "ACT/365F", "KRW", "KRW-KOFR-OIS", "", -1, 1},
  {"MXN-FTIIE", "MXN", "overnight", "ACT/360", "MXN", "MXN-FTIIE-OIS", "", -1, 1},
  {"MXN-TIIE-28", "MXN", "ibor", "ACT/360", "MXN", "MXN-TIIE-28-IRS", "28D", 1, -1},
  {"RUB-RUONIA", "RUB", "overnight", "ACT/365F", "RUB", "RUB-RUONIA-OIS", "", -1, 1},
  {"SAR-SAIBOR-3M", "SAR", "ibor", "ACT/360", "SAR", "SAR-SAIBOR-3M-IRS", "3M", 2, -1},
  {"TRY-TLREF", "TRY", "overnight", "ACT/360", "TRY", "TRY-TLREF-OIS", "", -1, 1},
  {"USD-FEDFUNDS", "USD", "overnight", "ACT/360", "USD-FED", "USD-FEDFUNDS-OIS", "", -1, 1},
  {"USD-SOFR", "USD", "overnight", "ACT/360", "USD-SOFR", "USD-SOFR-OIS", "", -1, 1},
  {"ZAR-JIBAR-3M", "ZAR", "ibor", "ACT/365F", "ZAR", "ZAR-JIBAR-3M-IRS", "3M", 0, -1},
  {"ZAR-ZARONIA", "ZAR", "overnight", "ACT/365F", "ZAR", "ZAR-ZARONIA-OIS", "", -1, 1},
}};

inline constexpr std::array<BondConv, 2> kBonds = {{
  {"US-TREASURY", "USD", "USD", "ACT/ACT-ICMA", "6M", "compound", 1, true},
  {"US-TREASURY-TSY", "USD", "USD", "ACT/ACT-ICMA", "6M", "simple", 1, false},
}};

inline constexpr std::array<HolidayRule, 209> kHolidayRules = {{
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -48, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -47, 0, ""},
  {"fixed", 3, 24, -1, 0, 0, 0, ""},
  {"fixed", 4, 2, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 5, 25, -1, 0, 0, 0, ""},
  {"fixed", 6, 17, -1, 0, 0, 0, ""},
  {"fixed", 6, 20, -1, 0, 0, 0, ""},
  {"fixed", 7, 9, -1, 0, 0, 0, ""},
  {"nth_weekday", 8, 0, 0, 3, 0, 0, "none"},
  {"fixed", 10, 12, -1, 0, 0, 0, ""},
  {"fixed", 11, 20, -1, 0, 0, 0, ""},
  {"fixed", 12, 8, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 1, 26, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, ""},
  {"fixed", 4, 25, -1, 0, 0, 0, ""},
  {"nth_weekday", 6, 0, 0, 2, 0, 0, "none"},
  {"nth_weekday", 8, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 10, 0, 0, 1, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -48, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -47, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"fixed", 4, 21, -1, 0, 0, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 60, 0, ""},
  {"fixed", 9, 7, -1, 0, 0, 0, ""},
  {"fixed", 10, 12, -1, 0, 0, 0, ""},
  {"fixed", 11, 2, -1, 0, 0, 0, ""},
  {"fixed", 11, 15, -1, 0, 0, 0, ""},
  {"fixed", 11, 20, -1, 0, 0, 2024, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 2, 0, 0, 3, 0, 0, "none"},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"nth_weekday", 5, 0, 0, 3, 0, 0, "none"},
  {"fixed", 7, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 8, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 9, 0, 0, 1, 0, 0, "none"},
  {"fixed", 9, 30, -1, 0, 0, 2021, ""},
  {"nth_weekday", 10, 0, 0, 2, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 1, 2, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 39, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 50, 0, ""},
  {"fixed", 8, 1, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 2, 1, -1, 0, 0, 0, ""},
  {"fixed", 2, 2, -1, 0, 0, 0, ""},
  {"fixed", 2, 3, -1, 0, 0, 0, ""},
  {"fixed", 4, 5, -1, 0, 0, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 6, 10, -1, 0, 0, 0, ""},
  {"fixed", 9, 20, -1, 0, 0, 0, ""},
  {"fixed", 10, 1, -1, 0, 0, 0, ""},
  {"fixed", 10, 2, -1, 0, 0, 0, ""},
  {"fixed", 10, 3, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, ""},
  {"nth_weekday", 5, 0, 0, 1, 0, 0, "none"},
  {"last_weekday", 5, 0, 0, 0, 0, 0, "none"},
  {"last_weekday", 8, 0, 0, 0, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 6, 1, -1, 0, 0, 0, ""},
  {"fixed", 8, 17, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 26, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"fixed", 4, 14, -1, 0, 0, 0, ""},
  {"fixed", 8, 15, -1, 0, 0, 0, ""},
  {"fixed", 10, 2, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 1, 2, -1, 0, 0, 0, "none"},
  {"fixed", 1, 3, -1, 0, 0, 0, "none"},
  {"nth_weekday", 1, 0, 0, 2, 0, 0, "none"},
  {"fixed", 2, 11, -1, 0, 0, 0, ""},
  {"fixed", 2, 23, -1, 0, 0, 2020, ""},
  {"fixed", 3, 20, -1, 0, 0, 0, ""},
  {"fixed", 4, 29, -1, 0, 0, 0, ""},
  {"fixed", 5, 3, -1, 0, 0, 0, ""},
  {"fixed", 5, 4, -1, 0, 0, 0, ""},
  {"fixed", 5, 5, -1, 0, 0, 0, ""},
  {"nth_weekday", 7, 0, 0, 3, 0, 0, "none"},
  {"fixed", 8, 11, -1, 0, 0, 0, ""},
  {"nth_weekday", 9, 0, 0, 3, 0, 0, "none"},
  {"fixed", 9, 23, -1, 0, 0, 0, ""},
  {"nth_weekday", 10, 0, 0, 2, 0, 0, "none"},
  {"fixed", 11, 3, -1, 0, 0, 0, ""},
  {"fixed", 11, 23, -1, 0, 0, 0, ""},
  {"fixed", 12, 31, -1, 0, 0, 0, "none"},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 2, 1, -1, 0, 0, 0, ""},
  {"fixed", 3, 1, -1, 0, 0, 0, ""},
  {"fixed", 5, 5, -1, 0, 0, 0, ""},
  {"fixed", 5, 15, -1, 0, 0, 0, ""},
  {"fixed", 6, 6, -1, 0, 0, 0, ""},
  {"fixed", 8, 15, -1, 0, 0, 0, ""},
  {"fixed", 9, 20, -1, 0, 0, 0, ""},
  {"fixed", 10, 3, -1, 0, 0, 0, ""},
  {"fixed", 10, 9, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"nth_weekday", 2, 0, 0, 1, 0, 0, "none"},
  {"nth_weekday", 3, 0, 0, 3, 0, 0, "none"},
  {"easter_offset", 0, 0, -1, 0, -3, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 9, 16, -1, 0, 0, 0, ""},
  {"nth_weekday", 11, 0, 0, 3, 0, 0, "none"},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 1, 2, -1, 0, 0, 0, ""},
  {"fixed", 1, 3, -1, 0, 0, 0, ""},
  {"fixed", 1, 4, -1, 0, 0, 0, ""},
  {"fixed", 1, 5, -1, 0, 0, 0, ""},
  {"fixed", 1, 6, -1, 0, 0, 0, ""},
  {"fixed", 1, 7, -1, 0, 0, 0, ""},
  {"fixed", 1, 8, -1, 0, 0, 0, ""},
  {"fixed", 2, 23, -1, 0, 0, 0, ""},
  {"fixed", 3, 8, -1, 0, 0, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 5, 9, -1, 0, 0, 0, ""},
  {"fixed", 6, 12, -1, 0, 0, 0, ""},
  {"fixed", 11, 4, -1, 0, 0, 0, ""},
  {"fixed", 4, 10, -1, 0, 0, 0, ""},
  {"fixed", 6, 17, -1, 0, 0, 0, ""},
  {"fixed", 9, 23, -1, 0, 0, 0, ""},
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 4, 23, -1, 0, 0, 0, ""},
  {"fixed", 4, 10, -1, 0, 0, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 5, 19, -1, 0, 0, 0, ""},
  {"fixed", 6, 17, -1, 0, 0, 0, ""},
  {"fixed", 7, 15, -1, 0, 0, 2017, ""},
  {"fixed", 8, 30, -1, 0, 0, 0, ""},
  {"fixed", 10, 29, -1, 0, 0, 0, ""},
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
  {"fixed", 1, 1, -1, 0, 0, 0, ""},
  {"fixed", 3, 21, -1, 0, 0, 0, ""},
  {"easter_offset", 0, 0, -1, 0, -2, 0, ""},
  {"easter_offset", 0, 0, -1, 0, 1, 0, "none"},
  {"fixed", 4, 27, -1, 0, 0, 0, ""},
  {"fixed", 5, 1, -1, 0, 0, 0, ""},
  {"fixed", 6, 16, -1, 0, 0, 0, ""},
  {"fixed", 8, 9, -1, 0, 0, 0, ""},
  {"fixed", 9, 24, -1, 0, 0, 0, ""},
  {"fixed", 12, 16, -1, 0, 0, 0, ""},
  {"fixed", 12, 25, -1, 0, 0, 0, ""},
  {"fixed", 12, 26, -1, 0, 0, 0, ""},
}};

inline constexpr std::array<std::string_view, 2> kCalendarJoins = {{
  "USD",
  "EUR",
}};

inline constexpr std::array<CalendarConv, 21> kCalendars = {{
  {"ARS", "Buenos Aires / Argentina settlement", "none", 96, 0, 16, 0, 0},
  {"AUD", "Sydney / Australia (ASX, NSW) settlement", "sun_to_mon", 96, 16, 10, 0, 0},
  {"BRL", "Brazil (B3 / ANBIMA) settlement", "none", 96, 26, 13, 0, 0},
  {"CAD", "Toronto / Canada settlement", "sun_to_mon", 96, 39, 11, 0, 0},
  {"CHF", "Zurich / Switzerland settlement", "none", 96, 50, 10, 0, 0},
  {"CNY", "Beijing / China interbank (CFETS) settlement", "none", 96, 60, 11, 0, 0},
  {"EUR", "TARGET (EUR settlement)", "none", 96, 71, 6, 0, 0},
  {"EURUSD", "Joint US-SIFMA + TARGET (FX/xccy USD side = SIFMA government-bond)", "", 96, 77, 0, 0, 2},
  {"GBP", "London / United Kingdom settlement", "sun_to_mon", 96, 77, 8, 2, 0},
  {"IDR", "Jakarta / Indonesia settlement", "none", 96, 85, 6, 2, 0},
  {"INR", "Mumbai / India settlement", "none", 96, 91, 6, 2, 0},
  {"JPY", "Tokyo / Japan settlement", "sun_to_mon", 96, 97, 19, 2, 0},
  {"KRW", "Seoul / South Korea settlement", "none", 96, 116, 11, 2, 0},
  {"MXN", "Mexico City / Mexico settlement", "none", 96, 127, 9, 2, 0},
  {"RUB", "Moscow / Russia (MOEX) settlement", "none", 96, 136, 14, 2, 0},
  {"SAR", "Riyadh / Saudi Arabia settlement (Fri/Sat weekend)", "none", 48, 150, 3, 2, 0},
  {"TRY", "Istanbul / Turkey settlement", "none", 96, 153, 9, 2, 0},
  {"USD", "US SIFMA / US government securities (bond market)", "sat_to_fri_sun_to_mon", 96, 162, 12, 2, 0},
  {"USD-FED", "US Federal Reserve (Fedwire)", "sun_to_mon", 96, 174, 11, 2, 0},
  {"USD-SOFR", "SOFR fixing calendar (SIFMA, incl. Good Friday close)", "sat_to_fri_sun_to_mon", 96, 185, 12, 2, 0},
  {"ZAR", "Johannesburg / South Africa settlement", "sun_to_mon", 96, 197, 12, 2, 0},
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
