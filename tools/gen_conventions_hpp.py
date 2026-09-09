#!/usr/bin/env python3
"""Generate include/swaps/conventions_data.hpp from conventions/conventions.json.

The JSON is the single source of truth for market conventions (shared with the Python web layer). C++ can't
afford a runtime JSON parse in the oracle-test binary (no boost::json linked there), so we codegen a plain
committed header the reference builders read from. Re-run after any edit to conventions.json; a test
(conventions_test.cpp) fails if the header is out of sync, so drift can't slip through.

Usage:
  python3 tools/gen_conventions_hpp.py            # writes the header in place
  python3 tools/gen_conventions_hpp.py --stdout   # print to stdout (verify.sh sync guard diffs this)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, "conventions", "conventions.json")
OUT = os.path.join(ROOT, "include", "swaps", "conventions_data.hpp")


def sv(x):
    return f'"{x}"' if x is not None else '""'


def leg(d):
    d = d or {}
    return ("{" + ", ".join([
        sv(d.get("index")), sv(d.get("day_count")), sv(d.get("frequency")), sv(d.get("compounding")),
        str(d.get("fixing_lag", -1)),
        "true" if d.get("carries_spread") else "false",
        "true" if d.get("notional_resets") else "false",
        "true" if d.get("flat") else "false",
    ]) + "}")


def main():
    db = json.load(open(JSON))
    products, indices, bonds = db["products"], db["indices"], db.get("bonds", {})
    calendars = db.get("calendars", {})
    currencies = db.get("currencies", {})

    lines = [
        "#pragma once",
        "// GENERATED from conventions/conventions.json by tools/gen_conventions_hpp.py -- DO NOT EDIT.",
        "// The market-conventions DB is the single source of truth; re-run the generator after editing the",
        "// JSON (conventions_test.cpp fails if this header is stale). See conventions/conventions.json.",
        "#include <array>",
        "#include <cctype>",
        "#include <cstddef>",
        "#include <optional>",
        "#include <string_view>",
        "",
        "namespace swaps::conventions {",
        "",
        "struct LegConv {",
        "  std::string_view index, day_count, frequency, compounding;",
        "  int fixing_lag; bool carries_spread, notional_resets, flat;",
        "};",
        "// type: ois | irs | basis | xccy_mtm | fx_forward | future | administered-basis. `floating` is the quoted",
        "// float / spread / usd leg; `other` is the flat / eur leg of a basis or xccy product (empty otherwise).",
        "// `frequency` is the product-level payment frequency (xccy), `discount_index` the DB discount index (basis).",
        "struct ProductConv {",
        "  std::string_view id, type, currency, calendar, bdc, frequency, discount_index, pair, base_currency;",
        "  int spot_lag, payment_lag;",
        "  LegConv fixed, floating, other;",
        "};",
        "// A CURRENCY row (currencies[] in the JSON): ISO minor units, currency-level settlement calendar, the",
        "// default discount (RFR) index and the default swap product used when a curve names no index.",
        "struct CurrencyConv {",
        "  std::string_view code, name, settlement_calendar, discount_index, default_swap_product;",
        "  int minor_units;",
        "};",
        "struct IndexConv {",
        "  std::string_view id, currency, type, day_count, calendar, par_product, tenor;",
        "  int fixing_lag, publication_lag;",
        "};",
        "// A BOND convention. `stub_discount` is the one thing the per-flow exponent cannot express (see",
        "// pricing/bond.hpp YieldConvention): \"compound\" -> dirty = Q(v)*v^w, \"simple\" -> Q(v)/(1 + w*y/f).",
        "// `final_period_simple` forces the simple form once a single cashflow remains (US street, Bund).",
        "struct BondConv {",
        "  std::string_view id, currency, calendar, day_count, frequency, stub_discount;",
        "  int settle_lag; bool final_period_simple;",
        "};",
        "// A HOLIDAY rule (calendars[].holidays in the JSON) — interpreted by swaps/build/calendar.hpp.",
        "// kind: \"fixed\" (month/day, from_year 0 = always), \"nth_weekday\" (month/weekday/n),",
        "// \"last_weekday\" (month/weekday), \"easter_offset\" (days vs Easter Sunday). weekday is Mon=0..Sun=6.",
        "// from_year/to_year bound the years a rule applies (0 = open-ended); a tabulated per-year holiday",
        "// (lunar / Islamic / announced) is a \"fixed\" rule with from_year==to_year==that year.",
        "// observance: \"\" = inherit the calendar default; else \"none\" | \"sat_to_fri_sun_to_mon\" | \"sun_to_mon\".",
        "struct HolidayRule {",
        "  std::string_view kind;",
        "  int month, day, weekday, n, days, from_year, to_year;",
        "  std::string_view observance;",
        "};",
        "// A CALENDAR: either rule-based (rule_count > 0) or a JOIN of other calendars (closed if any leg is",
        "// closed). `weekend_mask` bit w (Mon=0..Sun=6) marks a weekend day. Rules/joins are slices of the flat",
        "// kHolidayRules / kCalendarJoins arrays below.",
        "struct CalendarConv {",
        "  std::string_view id, name, observance;",
        "  int weekend_mask;",
        "  std::size_t rule_begin, rule_count, join_begin, join_count;",
        "};",
        "",
        f"inline constexpr std::array<ProductConv, {len(products)}> kProducts = {{{{",
    ]
    for pid in sorted(products):
        p = products[pid]
        lines.append("  {" + ", ".join([
            sv(pid), sv(p.get("type")), sv(p.get("currency")), sv(p.get("calendar")), sv(p.get("bdc")),
            sv(p.get("frequency")), sv(p.get("discount_index")), sv(p.get("pair")), sv(p.get("base_currency")),
            str(p.get("spot_lag", -1)), str(p.get("payment_lag", -1)),
            leg(p.get("fixed_leg")),
            leg(p.get("float_leg") or p.get("spread_leg") or p.get("usd_leg")),
            leg(p.get("flat_leg") or p.get("eur_leg")),
        ]) + "},")
    lines += ["}};", ""]

    lines.append(f"inline constexpr std::array<IndexConv, {len(indices)}> kIndices = {{{{")
    for iid in sorted(indices):
        i = indices[iid]
        lines.append("  {" + ", ".join([
            sv(iid), sv(i.get("currency")), sv(i.get("type")), sv(i.get("day_count")), sv(i.get("calendar")),
            sv(i.get("par_product")), sv(i.get("tenor")),
            str(i.get("fixing_lag", -1)), str(i.get("publication_lag", -1)),
        ]) + "},")
    lines += ["}};", ""]

    lines.append(f"inline constexpr std::array<CurrencyConv, {len(currencies)}> kCurrencies = {{{{")
    for cc in sorted(currencies):
        c = currencies[cc]
        lines.append("  {" + ", ".join([
            sv(cc), sv(c.get("name")), sv(c.get("settlement_calendar")), sv(c.get("discount_index")),
            sv(c.get("default_swap_product")), str(c.get("minor_units", 2)),
        ]) + "},")
    lines += ["}};", ""]

    lines.append(f"inline constexpr std::array<BondConv, {len(bonds)}> kBonds = {{{{")
    for bid in sorted(bonds):
        b = bonds[bid]
        lines.append("  {" + ", ".join([
            sv(bid), sv(b.get("currency")), sv(b.get("calendar")), sv(b.get("day_count")),
            sv(b.get("frequency")), sv(b.get("stub_discount")),
            str(b.get("settle_lag", -1)),
            "true" if b.get("final_period_simple") else "false",
        ]) + "},")
    lines += ["}};", ""]

    # Calendars: flatten every calendar's holiday rules (and join legs) into single arrays, sliced per
    # calendar by (begin, count) — constexpr-friendly, no nested variable-length initializers.
    rules, joins, cal_rows = [], [], []
    for cid in sorted(calendars):
        c = calendars[cid]
        rb, jb = len(rules), len(joins)
        for r in c.get("holidays", []):
            rules.append("  {" + ", ".join([
                sv(r["rule"]),
                str(r.get("month", 0)), str(r.get("day", 0)), str(r.get("weekday", -1)),
                str(r.get("n", 0)), str(r.get("days", 0)), str(r.get("from_year", 0)),
                str(r.get("to_year", 0)),
                sv(r.get("observance")),
            ]) + "},")
        for jleg in c.get("join", []):
            joins.append(f"  {sv(jleg)},")
        mask = sum(1 << w for w in c.get("weekend", [5, 6]))
        cal_rows.append("  {" + ", ".join([
            sv(cid), sv(c.get("name")), sv(c.get("observance")), str(mask),
            str(rb), str(len(rules) - rb), str(jb), str(len(joins) - jb),
        ]) + "},")
    lines.append(f"inline constexpr std::array<HolidayRule, {len(rules)}> kHolidayRules = {{{{")
    lines += rules + ["}};", ""]
    lines.append(f"inline constexpr std::array<std::string_view, {len(joins)}> kCalendarJoins = {{{{")
    lines += joins + ["}};", ""]
    lines.append(f"inline constexpr std::array<CalendarConv, {len(calendars)}> kCalendars = {{{{")
    lines += cal_rows + ["}};", ""]

    lines += [
        "// Approximate year-fraction of a frequency/tenor token ('3M'->0.25, '6M'->0.5, '1Y'->1.0), for the",
        "// coupon-period length a curve build needs when it only has year fractions (conventions_db.period_years).",
        "inline double period_years(std::string_view tok) {",
        "  if (tok.empty()) return 0.0;",
        "  const char u = char(std::toupper((unsigned char)tok.back()));",
        "  double unit = 1.0;",
        "  switch (u) { case 'D': unit = 1.0 / 365.0; break; case 'W': unit = 7.0 / 365.0; break;",
        "               case 'M': unit = 1.0 / 12.0; break; case 'Y': unit = 1.0; break; default: return 0.0; }",
        "  int n = 0;",
        "  for (std::size_t k = 0; k + 1 < tok.size(); ++k) { if (tok[k] < '0' || tok[k] > '9') return 0.0; n = n * 10 + (tok[k] - '0'); }",
        "  return n * unit;",
        "}",
        "",
        "}  // namespace swaps::conventions",
        "",
        "// The LOOKUPS live in the hand-written registry (baked defaults + runtime overlay; unknown id throws on",
        "// the require_* forms). It is included here, after the arrays, so every consumer of this header sees them.",
        "#define SWAPS_CONVENTIONS_DATA_INCLUDED 1",
        "#include \"swaps/conventions_db.hpp\"",
        "",
    ]
    text = "\n".join(lines)
    if "--stdout" in sys.argv:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(products)} products, {len(indices)} indices, {len(bonds)} bonds, "
          f"{len(calendars)} calendars / {len(rules)} holiday rules)")


if __name__ == "__main__":
    main()
