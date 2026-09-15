#!/usr/bin/env python3
"""check_hotpath_census.py — the streamer's REFRESH / FACTORISATION call sites are locked (2026-09-15).

The hot path kept acquiring stages nothing pinned: a refresh added inside a tick, a factorisation fallback, a
re-anchor on commit. Each was found afterwards by an audit. This guard counts, in
include/swaps/calibration/streaming.hpp (comments stripped), the call sites that cost a Jacobian, a
factorisation or a residual evaluation, and compares them to the [callsites] section of
tests/hotpath_census.lock. A new site FAILs verify.sh until the lock is edited, and the edit is the cue to give
it a StreamStage, a census scenario (tests/hotpath_census_test.cpp) and pins. It also requires every StreamStage
enumerator to be stamped somewhere.

  python3 tools/check_hotpath_census.py              check (exit 1 on a mismatch)
  python3 tools/check_hotpath_census.py --print      print the current counts in the lock's format
  python3 tools/check_hotpath_census.py --selftest   prove it can fail: an injected factor() site and a dropped stamp
"""
import argparse, os, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "include", "swaps", "calibration", "streaming.hpp")
LOCK = os.path.join(ROOT, "tests", "hotpath_census.lock")
CALLS = ["set_anchor", "factor", "rescale_row", "residuals_vs", "jacobian_vs_into", "model_rates", "request", "update_exact"]


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def counts(text):
    code = strip_comments(text)
    return {c: len(re.findall(r"(?<![A-Za-z0-9_])" + c + r"\(", code)) for c in CALLS}


def unstamped(text):
    code = strip_comments(text)
    m = re.search(r"enum class StreamStage[^{]*\{(.*?)\};", code, flags=re.S)
    if not m:
        return ["<no enum class StreamStage>"]
    names = [n.strip() for n in m.group(1).split(",") if n.strip() and n.strip() != "kCount"]
    return [n for n in names if not re.search(r"stamp\((?:[^;]*?)StreamStage::" + n + r"\b", code)]


def read_lock(path):
    want, section = {}, ""
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line
            continue
        if section == "[callsites]":
            name, n = line.split()
            want[name] = int(n)
    return want


def check(header, lock):
    text = open(header).read()
    got, want = counts(text), read_lock(lock)
    bad = []
    for c in CALLS:
        if got[c] != want.get(c):
            bad.append(f"  {c}(): {got[c]} call sites in streaming.hpp, the census lock says {want.get(c)} -- a new (or removed) site: give it a "
                       f"StreamStage, a scenario in tests/hotpath_census_test.cpp and pins, then update [callsites] in tests/hotpath_census.lock")
    for n in unstamped(text):
        bad.append(f"  StreamStage::{n} is never stamped: a stage no tick can report")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.print:
        print("[callsites]")
        for c, n in counts(open(HEADER).read()).items():
            print(f"{c:18s} {n}")
        return 0
    if a.selftest:
        src = open(HEADER).read()
        with tempfile.TemporaryDirectory() as d:
            inj = os.path.join(d, "streaming.hpp")
            anchor = "}  // namespace swaps::calibration"
            assert anchor in src, "selftest anchor moved"
            open(inj, "w").write(src.replace(anchor, "inline void census_selftest_(auto& s) { s.factor(s.J_cur_); }\n" + anchor, 1))
            b1 = check(inj, LOCK)
            open(inj, "w").write(re.sub(r"stamp\(StreamStage::Pin\);", "", src, count=1))
            b2 = check(inj, LOCK)
        ok = any("factor()" in b for b in b1) and any("StreamStage::Pin" in b for b in b2) and not check(HEADER, LOCK)
        print("check_hotpath_census selftest: " + ("OK -- an injected factor() site and a dropped stamp both FAIL" if ok else "FAILED"))
        return 0 if ok else 1
    bad = check(HEADER, LOCK)
    if bad:
        print("check_hotpath_census: FAIL\n" + "\n".join(bad))
        return 1
    print("check_hotpath_census: OK -- " + ", ".join(f"{c} {n}" for c, n in counts(open(HEADER).read()).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
