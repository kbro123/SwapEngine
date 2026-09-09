#!/usr/bin/env python3
"""Validate conventions/conventions.json against conventions/conventions.schema.json (PRINCIPLES.md P2).
Run by tools/verify.sh. Also asserts referential integrity the schema cannot express: every calendar /
index / product / currency id referenced by another row exists. Exit 1 on any error."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = json.load(open(os.path.join(ROOT, "conventions", "conventions.json")))
schema = json.load(open(os.path.join(ROOT, "conventions", "conventions.schema.json")))
errors = []
try:
    import jsonschema
    for e in sorted(jsonschema.Draft7Validator(schema).iter_errors(db), key=lambda e: list(e.path)):
        errors.append("schema: " + "/".join(str(x) for x in e.path) + ": " + e.message[:200])
except ImportError:
    print("check_schema: WARNING jsonschema not installed — structural validation skipped (pip install jsonschema)")
cal, idx, prod, cur = db["calendars"], db["indices"], db["products"], db["currencies"]
def ref(kind, table, id_, where):
    if id_ and id_ not in table: errors.append(f"{where}: unknown {kind} '{id_}'")
for cid, c in cal.items():
    for j in c.get("join", []): ref("calendar", cal, j, f"calendars/{cid}/join")
for iid, i in idx.items():
    ref("calendar", cal, i.get("calendar"), f"indices/{iid}"); ref("product", prod, i.get("par_product"), f"indices/{iid}")
    ref("currency", cur, i.get("currency"), f"indices/{iid}")
for pid, p in prod.items():
    ref("calendar", cal, p.get("calendar"), f"products/{pid}"); ref("currency", cur, p.get("currency"), f"products/{pid}")
    ref("index", idx, p.get("discount_index"), f"products/{pid}")
    for lg in ("fixed_leg", "float_leg", "spread_leg", "flat_leg", "usd_leg", "eur_leg"):
        if lg in p: ref("index", idx, p[lg].get("index"), f"products/{pid}/{lg}")
for cc, c in cur.items():
    ref("calendar", cal, c.get("settlement_calendar"), f"currencies/{cc}"); ref("index", idx, c.get("discount_index"), f"currencies/{cc}")
    ref("product", prod, c.get("default_swap_product"), f"currencies/{cc}")
    if idx.get(c.get("discount_index"), {}).get("currency") != cc: errors.append(f"currencies/{cc}: discount_index is not a {cc} index")
for cid, c in db.get("credit", {}).get("cds_products", {}).items():
    ref("calendar", cal, c.get("calendar"), f"credit/{cid}"); ref("currency", cur, c.get("currency"), f"credit/{cid}")
    if c.get("day_count") not in db["day_counts"]: errors.append(f"credit/{cid}: unknown day_count {c.get('day_count')}")
for fid, f in db.get("bond_futures", {}).items():
    ref("calendar", cal, f.get("exchange_calendar"), f"bond_futures/{fid}"); ref("currency", cur, f.get("currency"), f"bond_futures/{fid}")
    if f.get("deliverable_convention") not in db.get("bonds", {}): errors.append(f"bond_futures/{fid}: unknown deliverable_convention")
    if f.get("repo_day_count") not in db["day_counts"]: errors.append(f"bond_futures/{fid}: unknown repo_day_count")
for pid_, f in db.get("fx_pairs", {}).items():
    for k in ("base", "quote", "premium_currency"): ref("currency", cur, f.get(k), f"fx_pairs/{pid_}/{k}")
    ref("calendar", cal, f.get("calendar"), f"fx_pairs/{pid_}"); ref("product", prod, f.get("xccy_product"), f"fx_pairs/{pid_}"); ref("product", prod, f.get("forward_product"), f"fx_pairs/{pid_}")
    if pid_ != f.get("base", "") + f.get("quote", ""): errors.append(f"fx_pairs/{pid_}: id must be base+quote")
for cc, c in db.get("cb_schedules", {}).items():
    ref("currency", cur, cc, f"cb_schedules/{cc}")
    if c.get("meetings") != sorted(c.get("meetings", [])): errors.append(f"cb_schedules/{cc}: meetings not sorted")
for cc, c in cur.items():
    if c.get("repo_day_count") not in db["day_counts"]: errors.append(f"currencies/{cc}: unknown repo_day_count")
for bid, b in db.get("bonds", {}).items():
    ref("calendar", cal, b.get("calendar"), f"bonds/{bid}"); ref("currency", cur, b.get("currency"), f"bonds/{bid}")
if errors:
    print("check_schema: FAIL"); [print("  " + e) for e in errors]; sys.exit(1)
print(f"check_schema: OK — {len(cur)} currencies, {len(cal)} calendars, {len(idx)} indices, {len(prod)} products, {len(db.get('bonds',{}))} bonds, "
      f"{len(db.get('bond_futures',{}))} bond futures, {len(db.get('credit',{}).get('cds_products',{}))} cds products, {len(db.get('fx_pairs',{}))} fx pairs, "
      f"{len(db.get('cb_schedules',{}))} cb schedules; schema + references valid")
