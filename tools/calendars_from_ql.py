#!/usr/bin/env python3
"""Regenerate the TABULATED calendars in conventions/conventions.json from QuantLib's dump (build/ql_calendars_dump).

QuantLib is the independent source (PRINCIPLES.md P8/P11): within each market's QuantLib coverage year the
per-year rows are QuantLib's closed days (+ working weekend days for China), tagged `source`. Open-ended
rules (fixed national days, weekday rules) are kept; per-year rows we generated ourselves are kept only
BEYOND QuantLib's coverage and tagged "generated". Run: build/ql_calendars_dump 2024 > dump.json &&
python3 tools/calendars_from_ql.py dump.json && python3 tools/gen_conventions_hpp.py
"""
import collections, datetime as dt, json, sys
O = collections.OrderedDict
ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
dump = json.load(open(sys.argv[1]))
p = ROOT + "/conventions/conventions.json"
db = json.load(open(p), object_pairs_hook=O)
for cid, q in dump.items():
    cal = db["calendars"][cid]
    cov = int(q["coverage"])
    src = f"QuantLib 1.35 {q['quantlib']}"
    old = cal.get("holidays", [])
    keep_open = [r for r in old if not (r.get("from_year") and r.get("from_year") == r.get("to_year"))]
    keep_beyond = [r for r in old if r.get("from_year") and r.get("from_year") == r.get("to_year") and r["from_year"] > cov]
    for r in keep_beyond:
        r.setdefault("source", "generated (tools/gen_em_holidays.py) — beyond QuantLib coverage")
    ql_rows = []
    for iso in q["closed"]:
        d = dt.date.fromisoformat(iso)
        ql_rows.append(O([("label", f"{d.strftime('%Y-%m-%d')} (QuantLib)"), ("rule", "fixed"), ("month", d.month), ("day", d.day),
                          ("from_year", d.year), ("to_year", d.year), ("observance", "none"), ("source", src)]))
    for iso in q["working_weekend"]:
        d = dt.date.fromisoformat(iso)
        ql_rows.append(O([("label", f"{iso} working weekend day (QuantLib)"), ("rule", "working_day"), ("month", d.month), ("day", d.day),
                          ("from_year", d.year), ("to_year", d.year), ("source", src)]))
    cal["holidays"] = keep_open + ql_rows + keep_beyond
    cal["oracle_coverage_year"] = cov
    print(f"{cid}: kept {len(keep_open)} open-ended rules, {len(ql_rows)} QuantLib rows through {cov}, {len(keep_beyond)} generated rows beyond")
json.dump(db, open(p, "w"), indent=2, ensure_ascii=False); open(p, "a").write("\n")
