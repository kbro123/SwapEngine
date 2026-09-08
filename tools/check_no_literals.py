#!/usr/bin/env python3
"""check_no_literals.py -- gate (PRINCIPLES.md P2): no market-convention literal may appear in engine code.

Invoked from tools/verify.sh. Adopted 2026-09-08 from the hardcode sweep (review-2026-09-08/hardcode-sweep/).
The ratchet file tools/check_no_literals.baseline lists the 130 hits that existed at adoption; E2 burns it
down file by file (build/conventions.hpp first). The gate fails on any NEW hit. Scans include/swaps/** (minus the codegen'd conventions_data.hpp), api/** and src/** for class-A/B
literals (see the 2026-09 hardcode sweep) and FAILS on any hit not covered by the allow-list.

Rules (each hit carries its rule id):
  CCY      quoted ISO currency code                       "USD"
  DBID     quoted conventions-DB id / id-shaped token      "USD-SOFR", "US-TREASURY", "EURUSD", "TARGET"
  CONV     quoted day-count / bdc / frequency / stub token "ACT/360", "ModifiedFollowing", "1Y"
  FALLBACK a CCY/DBID/CONV literal used as a silent default  `? "USD"`, `.empty() ? "ACT/360"`, `= "1Y";`
  NUM      numeric convention constant in accrual/lag/recovery/contract context
           (/360.0, 365.0/360.0, recovery = 0.4, notional_coupon = 0.06, *_lag = <int>, freq = 2)
  JSONDEF  a JSON accessor giving a CONVENTION key a non-empty default
           jd(o,"recovery",0.40), js(o,"currency","USD"), get_s(c,"convention","US-TREASURY") ...

Allow-list (tools/check_no_literals.allow), one entry per line:
    <path-glob>[:<line>]  <RULE|*>  <regex on the literal | *>   # reason
The vocabulary-dispatch files (day_count.hpp's `if (dc == "ACT/360")`, calendar.hpp's bdc switch, the
policy-name switch) are allowed by RULE+path, never by blanket path. A line-pinned entry goes stale the
moment the file is edited above it -- that is deliberate: it forces the entry to be re-justified.

  --baseline FILE   write every current hit as an allow entry (ratchet: the gate then fails only on NEW hits)
  --allow FILE      allow-list to apply (default: tools/check_no_literals.allow next to this script, if present)
  --include-tests   also scan tests/ (reported separately as class E, never fatal)
  --list            print every hit (default prints only the summary + unallowed hits)
Exit 1 iff any unallowed hit in the engine scope.
"""
import argparse, fnmatch, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SWAPS_ROOT", os.path.dirname(HERE))
SCOPES = ["include/swaps", "api", "src"]
TEST_SCOPES = ["tests", "bench", "tools"]
EXT = {".hpp", ".cpp", ".h", ".inc", ".py"}
SKIP = {"include/swaps/conventions_data.hpp", "api/run_json_dispatch.gen.inc"}

db = json.load(open(os.path.join(ROOT, "conventions/conventions.json")))
DB_IDS = set(db["calendars"]) | set(db["indices"]) | set(db["products"]) | set(db.get("bonds", {}))
CCY = {"USD","EUR","GBP","JPY","AUD","CAD","CHF","CNY","INR","BRL","MXN","KRW","ZAR","TRY","IDR","RUB","SAR","ARS",
       "SEK","NOK","DKK","NZD","SGD","HKD","PLN","CZK","HUF","ILS","CLP","COP","PEN","TWD","THB","MYR","PHP"}
CAL_WORDS = {"TARGET","SIFMA","NYSE","FED","LONDON","TOKYO","ANBIMA"}
DC = set(db["day_counts"]) | {"ACT/ACT","ACT/ACT.ISDA","ACT/365","30/360"}
BDC = {"Following","ModifiedFollowing","Preceding","ModifiedPreceding","Unadjusted"}
STUB = {"ShortFront","ShortBack","LongFront","LongBack"}
ID_SHAPE = re.compile(r'^[A-Z]{2,4}-[A-Z0-9]+(-[A-Z0-9]+)*$')
PAIR = re.compile(r'^[A-Z]{6}$')
FREQ = re.compile(r'^\d+[DWMY]$')
STR = re.compile(r'"((?:[^"\\]|\\.)*)"')
JSON_ACC = re.compile(r'\b(jd|js|ji|jb|get_d|get_i|get_s|get_b)\s*\(\s*[\w.>()-]+\s*,\s*"([^"]+)"\s*,\s*([^)]+)\)')
CONV_KEYS = {"currency","index","convention","settle_calendar","settle_lag","recovery","premium_freq","notional_coupon",
             "round_months","freq","delta_convention","atm_convention","tenor","fx_time","base","spot_lag","payment_lag",
             "fixing_lag","day_count","calendar","bdc","pair","discount_index","float_freq"}
NUM_RULES = [
    (re.compile(r'/\s*360\.0\b|\b360\.0\s*/|\(\s*360\.0\s*\)|\*\s*360\.0\b'), "accrual-360"),
    (re.compile(r'\b365\.0\s*/\s*360\.0\b'), "365/360-ratio"),
    (re.compile(r'\brecovery\w*\s*=\s*0\.4\b'), "recovery-0.40"),
    (re.compile(r'\bnotional_coupon\s*=\s*0\.06\b'), "cf-yield-6pct"),
    (re.compile(r'\b(spot_lag|pay_lag|payment_lag|fixing_lag|settle_lag|publication_lag)\s*(=|:\s*p\w*\s*:)\s*-?\d+\s*[;,)]'), "lag-default"),
    (re.compile(r'\b(int\s+)?freq\s*=\s*2\s*;'), "freq-semiannual"),
    (re.compile(r'\bweekday\(\)\s*(<|>=)\s*5\b'), "weekend-hardwired"),
    (re.compile(r'\bplus_months\(\s*(-?\s*3|12\s*\*\s*\w+)\s*\)'), "schedule-step-literal"),
]

def strip_comments(text, ext):
    if ext == ".py":
        out = []
        for ln in text.split("\n"):
            res, inq, i = [], None, 0
            while i < len(ln):
                ch = ln[i]
                if inq:
                    res.append(ch)
                    if ch == "\\": res.append(ln[i+1] if i+1 < len(ln) else ""); i += 2; continue
                    if ch == inq: inq = None
                elif ch in "\"'": inq = ch; res.append(ch)
                elif ch == "#": break
                else: res.append(ch)
                i += 1
            out.append("".join(res))
        return out
    out, i, n, buf, in_block, in_str = [], 0, len(text), [], False, None
    while i < n:
        ch = text[i]
        if in_block:
            if text.startswith("*/", i): in_block = False; i += 2; continue
            if ch == "\n": buf.append("\n")
            i += 1; continue
        if in_str:
            buf.append(ch)
            if ch == "\\": buf.append(text[i+1] if i+1 < n else ""); i += 2; continue
            if ch == in_str or ch == "\n": in_str = None
            i += 1; continue
        if text.startswith("//", i):
            j = text.find("\n", i); i = n if j < 0 else j; continue
        if text.startswith("/*", i): in_block = True; i += 2; continue
        if ch in "\"'": in_str = ch; buf.append(ch); i += 1; continue
        buf.append(ch); i += 1
    return "".join(buf).split("\n")

def classify_str(s):
    if s in CCY: return "CCY"
    if s in DB_IDS or s in CAL_WORDS or ID_SHAPE.match(s) or (PAIR.match(s) and s[:3] in CCY and s[3:] in CCY): return "DBID"
    if s in DC or s in BDC or s in STUB or FREQ.match(s): return "CONV"
    return None

def scan(rel):
    ext = os.path.splitext(rel)[1]
    text = open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read()
    hits = []
    for ln_no, ln in enumerate(strip_comments(text, ext), 1):
        if not ln.strip(): continue
        for m in STR.finditer(ln):
            s = m.group(1); rule = classify_str(s)
            if not rule: continue
            pre = ln[:m.start()]
            p = pre.rstrip()
            is_cmp = bool(re.search(r'[=!<>]=$', p))                       # `dc == "ACT/360"` is dispatch, not a default
            is_fallback = (not is_cmp) and bool(re.search(r'(\?$|\.empty\(\)\s*\?$|value_or\($|=$|\?\s*std::string\($)', p)
                                                 or re.search(r'\?\s*(std::string\()?$', p))
            hits.append((rel, ln_no, "FALLBACK" if is_fallback else rule, s, ln.strip()[:140]))
        for m in JSON_ACC.finditer(ln):
            key, dflt = m.group(2), m.group(3).strip()
            if key in CONV_KEYS and dflt not in ('""', "0", "0.0", "false", "-1", "-1.0", "\"\"", "d"):
                hits.append((rel, ln_no, "JSONDEF", f'{key}={dflt}', ln.strip()[:140]))
        for rx, tag in NUM_RULES:
            if rx.search(ln): hits.append((rel, ln_no, "NUM", tag, ln.strip()[:140]))
    return hits

def walk(scopes):
    for sc in scopes:
        base = os.path.join(ROOT, sc)
        if not os.path.isdir(base): continue
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in ("__pycache__", "golden")]
            for f in sorted(fn):
                rel = os.path.relpath(os.path.join(dp, f), ROOT)
                if rel in SKIP or os.path.splitext(f)[1] not in EXT or ".gen." in f: continue
                yield rel

def load_allow(path):
    entries = []
    if not path or not os.path.exists(path): return entries
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 3: continue
        pathspec, rule, rx = parts[0], parts[1], " ".join(parts[2:])
        pat, _, lno = pathspec.partition(":")
        entries.append((pat, int(lno) if lno else None, rule, re.compile(rx) if rx != "*" else None))
    return entries

def allowed(hit, entries):
    rel, ln, rule, lit, _ = hit
    for pat, lno, r, rx in entries:
        if not fnmatch.fnmatch(rel, pat): continue
        if lno is not None and lno != ln: continue
        if r != "*" and r != rule: continue
        if rx is not None and not rx.search(lit): continue
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", default=os.path.join(HERE, "check_no_literals.allow"))
    ap.add_argument("--baseline"); ap.add_argument("--include-tests", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    entries = load_allow(a.allow)
    hits = [h for rel in walk(SCOPES) for h in scan(rel)]
    bad = [h for h in hits if not allowed(h, entries)]
    if a.baseline:
        with open(a.baseline, "w") as f:
            f.write("# generated by check_no_literals.py --baseline: RATCHET, shrink it, never grow it\n")
            for rel, ln, rule, lit, _ in sorted(bad): f.write(f"{rel}:{ln}  {rule}  {re.escape(lit)}\n")
        print(f"wrote {len(bad)} baseline entries to {a.baseline}")
    by_rule = collections.Counter(h[2] for h in hits); by_rule_bad = collections.Counter(h[2] for h in bad)
    by_dir = collections.Counter(h[0].split("/")[0] if not h[0].startswith("include/swaps/") else "include/swaps/" + h[0].split("/")[2] for h in bad)
    print(f"engine scope: {len(hits)} literal hits, {len(hits)-len(bad)} allow-listed, {len(bad)} UNALLOWED")
    print("  by rule (unallowed/total): " + ", ".join(f"{r}={by_rule_bad[r]}/{by_rule[r]}" for r in sorted(by_rule)))
    print("  unallowed by dir: " + ", ".join(f"{d}={n}" for d, n in sorted(by_dir.items())))
    if a.list or bad:
        for rel, ln, rule, lit, snip in sorted(bad if not a.list else hits):
            print(f"  {rel}:{ln}  [{rule}] {lit}  | {snip}")
    if a.include_tests:
        th = [h for rel in walk(TEST_SCOPES) for h in scan(rel)]
        tc = collections.Counter(h[2] for h in th)
        print(f"tests/bench/tools (class E, informational): {len(th)} hits: " + ", ".join(f"{r}={n}" for r, n in sorted(tc.items())))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
