#!/usr/bin/env python3
"""oracle_coverage.py — WHICH ENGINE CODE DOES A QUANTLIB NUMBER ACTUALLY REACH?

tests/ORACLE_TESTS.md answers "what does each oracle FILE compare?". This answers the dual question, the
one the 2026-09-10 averaged-weight bug turned on: for a given header, is there any registered ORACLE test
(engine number vs QuantLib number) whose include closure reaches it at all? That bug lived in
build/observations.hpp; every oracle that could have caught it built its observations TEST-SIDE, so no
oracle's closure entered the shipped builder, and eleven green oracles said nothing about it.

Reachability is a NECESSARY, NOT SUFFICIENT condition -- an oracle can include a header and still compare
only numbers that never touch it. So this is a floor, not a certificate: a header with no oracle in its
closure provably has no QuantLib check; one with an oracle merely might.

The verb table is computed separately because api/*.hpp is reached by LINKING, not by including: a test
that calls run_json("bonds", ...) exercises api/bond.hpp without ever naming the header. Verbs are matched
by their request key appearing as a string literal in a test.

  --check  compare against tests/oracle_coverage.lock and FAIL if any header LOST its oracle reach
           (gaining reach is always fine and updates nothing; closing a gap is not a gate failure)
  --update rewrite the lock (say why in the commit message)
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "tests", "oracle_coverage.lock")


def registered(name):
    """The oracle / consistency registries ARE the CMake lists (ORACLE_TESTS.md rule 2)."""
    cm = open(os.path.join(ROOT, "tests", "CMakeLists.txt"), encoding="utf-8").read()
    m = re.search(r"add_executable\(\s*" + name + r"\b(.*?)^\s*\)", cm, re.S | re.M)
    if not m:
        raise SystemExit("oracle_coverage: cannot find add_executable(" + name + ") in tests/CMakeLists.txt")
    return set(re.findall(r"(\w+\.cpp)", m.group(1)))


def includes(path):
    return re.findall(r'#include "([^"]+)"', open(path, encoding="utf-8", errors="ignore").read())


def resolve(inc):
    for base in ("include/", "tests/", "api/", ""):
        p = os.path.join(ROOT, base + inc)
        if os.path.exists(p):
            return p
    return None


def closure(test_cpp):
    """Every swaps/ header reachable from one test through #include, transitively."""
    seen, hit, stack = set(), set(), [os.path.join(ROOT, "tests", test_cpp)]
    while stack:
        p = stack.pop()
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        for inc in includes(p):
            if inc.startswith("swaps/"):
                hit.add(inc)
            r = resolve(inc)
            if r:
                stack.append(r)
    return hit


def header_reach():
    oracles = registered("swaps_oracle_tests") - {"ql_test_isolation.cpp"}
    reach = {}
    for t in sorted(oracles):
        for h in closure(t):
            reach.setdefault(h, set()).add(t)
    return reach


def all_headers():
    out = []
    for p in glob.glob(os.path.join(ROOT, "include", "swaps", "*", "*.hpp")):
        out.append(os.path.relpath(p, os.path.join(ROOT, "include")).replace(os.sep, "/"))
    return sorted(out)


def verbs():
    """request key -> tests naming it, split by whether any of those is a registered oracle."""
    d = open(os.path.join(ROOT, "api", "run_json_dispatch.gen.inc"), encoding="utf-8").read()
    s = open(os.path.join(ROOT, "api", "run_json_stateless.gen.inc"), encoding="utf-8").read()
    keys = set(re.findall(r'o\.contains\("([a-z0-9_]+)"\)', d)) | set(re.findall(r'\{"([a-z0-9_]+)",\s*&', s))
    oracles = registered("swaps_oracle_tests")
    named, by_oracle = {}, {}
    for p in sorted(glob.glob(os.path.join(ROOT, "tests", "*.cpp"))):
        t = open(p, encoding="utf-8", errors="ignore").read()
        base = os.path.basename(p)
        for k in keys:
            if '"' + k + '"' in t:
                named.setdefault(k, []).append(base)
                if base in oracles:
                    by_oracle.setdefault(k, []).append(base)
    return keys, named, by_oracle


def report():
    reach, heads = header_reach(), all_headers()
    layers = {}
    for h in heads:
        layers.setdefault(h.split("/")[1], []).append(h)
    print("ORACLE REACH BY LAYER (headers whose closure contains a registered oracle test)\n")
    print(f"  {'layer':<12} {'covered':>9}  headers with NO oracle in any closure")
    for l in sorted(layers):
        hs = layers[l]
        cov = [h for h in hs if h in reach]
        gap = [h.split("/")[-1] for h in hs if h not in reach]
        print(f"  {l:<12} {len(cov):>4}/{len(hs):<4}  {' '.join(gap) if gap else '-'}")
    keys, named, by_oracle = verbs()
    print(f"\nrun_json VERBS: {len(keys)} total, {len(named)} named by some test, "
          f"{len(by_oracle)} named by an ORACLE test")
    if keys - set(named):
        print("  named by NO test: " + " ".join(sorted(keys - set(named))))
    if keys - set(by_oracle):
        print("  no oracle names:  " + " ".join(sorted(keys - set(by_oracle))))


def lock_lines():
    reach = header_reach()
    return sorted(reach)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    cur = lock_lines()
    if a.update:
        with open(LOCK, "w", encoding="utf-8") as f:
            f.write("# Headers a registered QuantLib ORACLE test's include closure reaches (tools/oracle_coverage.py).\n"
                    "# Losing an entry means a QuantLib check stopped reaching shipped code -- usually a silent\n"
                    "# side effect of a refactor, which is exactly how coverage erodes. Gaining one is free.\n")
            f.write("\n".join(cur) + "\n")
        print(f"oracle_coverage: locked {len(cur)} headers")
        return 0
    if a.check:
        if not os.path.exists(LOCK):
            print("oracle_coverage: FAIL — no lock; run --update", file=sys.stderr)
            return 1
        want = [l.strip() for l in open(LOCK, encoding="utf-8") if l.strip() and not l.startswith("#")]
        lost = [h for h in want if h not in cur]
        if lost:
            print("oracle_coverage: FAIL — these headers LOST their oracle reach:", file=sys.stderr)
            for h in lost:
                print("    " + h, file=sys.stderr)
            print("  Restore the oracle's path to them, or re-lock with --update in a commit that says why.",
                  file=sys.stderr)
            return 1
        gained = [h for h in cur if h not in want]
        extra = f" (+{len(gained)} newly reached: {' '.join(gained)})" if gained else ""
        print(f"oracle_coverage: OK — {len(want)} locked headers still reached by an oracle{extra}")
        return 0
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
