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
    products, indices = db["products"], db["indices"]

    lines = [
        "#pragma once",
        "// GENERATED from conventions/conventions.json by tools/gen_conventions_hpp.py -- DO NOT EDIT.",
        "// The market-conventions DB is the single source of truth; re-run the generator after editing the",
        "// JSON (conventions_test.cpp fails if this header is stale). See conventions/conventions.json.",
        "#include <array>",
        "#include <optional>",
        "#include <string_view>",
        "",
        "namespace swaps::conventions {",
        "",
        "struct LegConv {",
        "  std::string_view index, day_count, frequency, compounding;",
        "  int fixing_lag; bool carries_spread, notional_resets, flat;",
        "};",
        "struct ProductConv {",
        "  std::string_view id, currency, calendar, bdc;",
        "  int spot_lag, payment_lag;",
        "  LegConv fixed, floating;",
        "};",
        "struct IndexConv {",
        "  std::string_view id, currency, type, day_count, calendar, par_product;",
        "  int fixing_lag, publication_lag;",
        "};",
        "",
        f"inline constexpr std::array<ProductConv, {len(products)}> kProducts = {{{{",
    ]
    for pid in sorted(products):
        p = products[pid]
        lines.append("  {" + ", ".join([
            sv(pid), sv(p.get("currency")), sv(p.get("calendar")), sv(p.get("bdc")),
            str(p.get("spot_lag", -1)), str(p.get("payment_lag", -1)),
            leg(p.get("fixed_leg")),
            leg(p.get("float_leg") or p.get("spread_leg") or p.get("usd_leg")),
        ]) + "},")
    lines += ["}};", ""]

    lines.append(f"inline constexpr std::array<IndexConv, {len(indices)}> kIndices = {{{{")
    for iid in sorted(indices):
        i = indices[iid]
        lines.append("  {" + ", ".join([
            sv(iid), sv(i.get("currency")), sv(i.get("type")), sv(i.get("day_count")), sv(i.get("calendar")),
            sv(i.get("par_product")),
            str(i.get("fixing_lag", -1)), str(i.get("publication_lag", -1)),
        ]) + "},")
    lines += ["}};", ""]

    lines += [
        "inline std::optional<ProductConv> product(std::string_view id) {",
        "  for (const auto& p : kProducts) if (p.id == id) return p;",
        "  return std::nullopt;",
        "}",
        "inline std::optional<IndexConv> index(std::string_view id) {",
        "  for (const auto& i : kIndices) if (i.id == id) return i;",
        "  return std::nullopt;",
        "}",
        "",
        "}  // namespace swaps::conventions",
        "",
    ]
    text = "\n".join(lines)
    if "--stdout" in sys.argv:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(products)} products, {len(indices)} indices)")


if __name__ == "__main__":
    main()
