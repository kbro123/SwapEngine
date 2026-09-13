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


def need(row, key, what, rid):
    """A field the DB cannot default: fail generation loudly, naming the row (the runtime verb throws the same)."""
    if key not in row:
        raise SystemExit(f"conventions.json: {what} '{rid}' needs '{key}'")
    return row[key]


def one_leg(p, keys, pid):
    """At most one of the alias keys for a leg; the runtime verb enforces the same (E7 stage 4.1: the old
    `a or b or c` silently took the first and treated an empty {} leg as absent)."""
    present = [k for k in keys if k in p]
    if len(present) > 1:
        raise SystemExit(f"conventions.json: product '{pid}' names more than one leg for the same role: {present}")
    return p[present[0]] if present else None


def leg(d):
    d = d or {}
    return ("{" + ", ".join([
        sv(d.get("index")), sv(d.get("day_count")), sv(d.get("frequency")), sv(d.get("compounding")),
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
        "  bool carries_spread = false, notional_resets = false, flat = false;  // the fixing lag lives on the INDEX row (one owner)",
        "};",
        "// type: ois | irs | basis | xccy_mtm | fx_forward | future | administered-basis. `floating` is the quoted",
        "// float / spread / usd leg; `other` is the flat / eur leg of a basis or xccy product (empty otherwise).",
        "// `frequency` is the product-level payment frequency (xccy), `discount_index` the DB discount index (basis).",
        "struct ProductConv {",
        "  std::string_view id, type, currency, calendar, bdc, frequency, discount_index, pair, base_currency;",
        "  int spot_lag = -1, payment_lag = -1;  // -1 = unset (the builders require it where it matters)",
        "  bool zero_coupon = false;  // one period spot->maturity, quoted annually-compounded (QuoteKind::ZeroCouponRate)",
        "  LegConv fixed, floating, other;",
        "};",
        "// A CURRENCY row (currencies[] in the JSON): ISO minor units, currency-level settlement calendar, the",
        "// default discount (RFR) index and the default swap product used when a curve names no index.",
        "struct CurrencyConv {",
        "  std::string_view code, name, settlement_calendar, discount_index, default_swap_product, repo_day_count;",
        "  int minor_units = -1;",
        "};",
        "// A CDS product (credit.cds_products[]): premium schedule + default recovery + protection integration.",
        "struct CreditConv {",
        "  std::string_view id, currency, calendar, day_count, frequency, roll;",
        "  double recovery_default = -1.0; int settlement_lag = -1, protection_steps = -1;",
        "};",
        "// A bond-futures CONTRACT (bond_futures[]): deliverable convention, CF notional coupon, rounding, repo basis.",
        "struct BondFutureConv {",
        "  std::string_view id, currency, exchange_calendar, deliverable_convention, repo_day_count, delivery;",
        "  double notional_coupon = -1.0, basket_min_years = 0.0, basket_max_years = 0.0; int maturity_rounding_months = -1;",
        "  int conversion_factor_decimals = -1;  // the exchange rounds the conversion factor to this many decimals",
        "};",
        "// An FX pair (fx_pairs[]): quoting/settlement/option conventions. id == base+quote.",
        "struct FxPairConv {",
        "  std::string_view id, base, quote, calendar, premium_currency, delta_convention, atm_convention, xccy_product, forward_product;",
        "  int spot_lag = -1; double smile_pillar_lo = 0.0, smile_pillar_hi = 0.0;",
        "};",
        "// A central-bank meeting schedule (cb_schedules[]): a slice of kCbMeetings (Unix-day serials, ascending).",
        "struct CbScheduleConv {",
        "  std::string_view currency, bank, source, as_of;",
        "  std::size_t begin = 0, count = 0;",
        "};",
        "// A fixing SOURCE (fixing_sources[]): where an index's realized fixings are fetched from. Fetching is",
        "// API-side; the engine carries the metadata so no client keeps its own provider table. id == index id.",
        "struct FixingSourceConv {",
        "  std::string_view id, provider, series, start, granularity;",
        "};",
        "// An INFLATION index (inflation[]): the swap reference index's observation lag and interpolation rule.",
        "struct InflationIndexConv {",
        "  std::string_view id, label, currency, calendar, interpolation, frequency;",
        "  int observation_lag_months = -1;",
        "};",
        "struct IndexConv {",
        "  std::string_view id, currency, type, day_count, calendar, par_product, tenor;",
        "  int fixing_lag = -1, publication_lag = -1;",
        "};",
        "// A BOND convention. `stub_discount` is the one thing the per-flow exponent cannot express (see",
        "// pricing/bond.hpp YieldConvention): \"compound\" -> dirty = Q(v)*v^w, \"simple\" -> Q(v)/(1 + w*y/f).",
        "// `final_period_simple` forces the simple form once a single cashflow remains (US street, Bund).",
        "struct BondConv {",
        "  std::string_view id, currency, calendar, day_count, frequency, stub_discount;",
        "  int settle_lag = -1; bool final_period_simple = false;",
        "};",
        "// A HOLIDAY rule (calendars[].holidays in the JSON) — interpreted by swaps/build/calendar.hpp.",
        "// kind: \"fixed\" (month/day, from_year 0 = always), \"nth_weekday\" (month/weekday/n),",
        "// \"last_weekday\" (month/weekday), \"easter_offset\" (days vs Easter Sunday). weekday is Mon=0..Sun=6.",
        "// from_year/to_year bound the years a rule applies (0 = open-ended); a tabulated per-year holiday",
        "// (lunar / Islamic / announced) is a \"fixed\" rule with from_year==to_year==that year.",
        "// observance: \"\" = inherit the calendar default; else \"none\" | \"sat_to_fri_sun_to_mon\" | \"sun_to_mon\".",
        "struct HolidayRule {",
        "  std::string_view kind;",
        "  int month = 0, day = 0, weekday = -1, n = 0, days = 0, from_year = 0, to_year = 0;",
        "  std::string_view observance;",
        "  bool except_first_friday = false;  // SIFMA Good Friday: no closure when it is the first Friday of the month",
        "};",
        "// A CALENDAR: either rule-based (rule_count > 0) or a JOIN of other calendars (closed if any leg is",
        "// closed). `weekend_mask` bit w (Mon=0..Sun=6) marks a weekend day. Rules/joins are slices of the flat",
        "// kHolidayRules / kCalendarJoins arrays below.",
        "struct CalendarConv {",
        "  std::string_view id, name, observance;",
        "  int weekend_mask = 0;",
        "  std::size_t rule_begin = 0, rule_count = 0, join_begin = 0, join_count = 0;",
        "  bool sandwich = false;  // Japan: a weekday between two holidays is a holiday",
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
            "true" if p.get("zero_coupon") else "false",
            leg(p.get("fixed_leg")),
            leg(one_leg(p, ("float_leg", "spread_leg", "usd_leg"), pid)),
            leg(one_leg(p, ("flat_leg", "eur_leg"), pid)),
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
            sv(c.get("default_swap_product")), sv(c.get("repo_day_count")), str(need(c, "minor_units", "currency", cc)),
        ]) + "},")
    lines += ["}};", ""]

    credit = db.get("credit", {}).get("cds_products", {})
    lines.append(f"inline constexpr std::array<CreditConv, {len(credit)}> kCredit = {{{{")
    for cid in sorted(credit):
        c = credit[cid]
        lines.append("  {" + ", ".join([
            sv(cid), sv(c.get("currency")), sv(c.get("calendar")), sv(c.get("day_count")), sv(c.get("frequency")),
            sv(c.get("roll")), repr(float(c.get("recovery_default", -1))), str(c.get("settlement_lag", -1)),
            str(c.get("protection_steps", -1)),
        ]) + "},")
    lines += ["}};", ""]
    futs = db.get("bond_futures", {})
    lines.append(f"inline constexpr std::array<BondFutureConv, {len(futs)}> kBondFutures = {{{{")
    for fid in sorted(futs):
        f = futs[fid]
        lines.append("  {" + ", ".join([
            sv(fid), sv(f.get("currency")), sv(f.get("exchange_calendar")), sv(f.get("deliverable_convention")),
            sv(f.get("repo_day_count")), sv(f.get("delivery")), repr(float(f.get("notional_coupon", -1))),
            repr(float(f.get("basket_min_years", 0))), repr(float(f.get("basket_max_years", 0))),
            str(f.get("maturity_rounding_months", -1)), str(f["conversion_factor_decimals"]),
        ]) + "},")
    lines += ["}};", ""]
    pairs = db.get("fx_pairs", {})
    lines.append(f"inline constexpr std::array<FxPairConv, {len(pairs)}> kFxPairs = {{{{")
    for pid2 in sorted(pairs):
        f = pairs[pid2]
        sp = f.get("smile_pillars", [])
        lines.append("  {" + ", ".join([
            sv(pid2), sv(f.get("base")), sv(f.get("quote")), sv(f.get("calendar")), sv(f.get("premium_currency")),
            sv(f.get("delta_convention")), sv(f.get("atm_convention")), sv(f.get("xccy_product")), sv(f.get("forward_product")),
            str(f.get("spot_lag", -1)), repr(float(sp[0]) if sp else 0.0), repr(float(sp[-1]) if sp else 0.0),
        ]) + "},")
    lines += ["}};", ""]
    srcs = db.get("fixing_sources", {})
    lines.append(f"inline constexpr std::array<FixingSourceConv, {len(srcs)}> kFixingSources = {{{{")
    for sid in sorted(srcs):
        s = srcs[sid]
        lines.append("  {" + ", ".join([sv(sid), sv(s.get("provider")), sv(s.get("series")), sv(s.get("start")),
                                        sv(s.get("granularity"))]) + "},")
    lines += ["}};", ""]
    infl = db.get("inflation", {})
    lines.append(f"inline constexpr std::array<InflationIndexConv, {len(infl)}> kInflationIndices = {{{{")
    for iid2 in sorted(infl):
        x = infl[iid2]
        lines.append("  {" + ", ".join([sv(iid2), sv(x.get("label")), sv(x.get("currency")), sv(x.get("calendar")),
                                        sv(x.get("interpolation")), sv(x.get("frequency")),
                                        str(x.get("observation_lag_months", -1))]) + "},")
    lines += ["}};", ""]
    cbs = db.get("cb_schedules", {})
    import datetime as _dt
    meetings, cb_rows = [], []
    for cc in sorted(cbs):
        c = cbs[cc]
        b = len(meetings)
        for iso in c.get("meetings", []):
            d = _dt.date.fromisoformat(iso)
            meetings.append("  " + str((d - _dt.date(1970, 1, 1)).days) + ",")
        cb_rows.append("  {" + ", ".join([sv(cc), sv(c.get("bank")), sv(c.get("source")), sv(c.get("as_of")),
                                         str(b), str(len(meetings) - b)]) + "},")
    lines.append(f"inline constexpr std::array<long, {len(meetings)}> kCbMeetings = {{{{")
    lines += meetings + ["}};", ""]
    lines.append(f"inline constexpr std::array<CbScheduleConv, {len(cbs)}> kCbSchedules = {{{{")
    lines += cb_rows + ["}};", ""]

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
                "true" if r.get("except_first_friday") else "false",
            ]) + "},")
        for jleg in c.get("join", []):
            joins.append(f"  {sv(jleg)},")
        mask = sum(1 << w for w in need(c, "weekend", "calendar", cid))
        cal_rows.append("  {" + ", ".join([
            sv(cid), sv(c.get("name")), sv(c.get("observance")), str(mask),
            str(rb), str(len(rules) - rb), str(jb), str(len(joins) - jb),
            "true" if c.get("sandwich") else "false",
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
          f"{len(calendars)} calendars / {len(rules)} holiday rules, {len(currencies)} currencies, "
          f"{len(credit)} cds products, {len(futs)} bond futures, {len(pairs)} fx pairs, {len(cbs)} cb schedules)")


if __name__ == "__main__":
    main()
