#!/usr/bin/env python3
"""Performance gate: run the benchmarks, compare to the fingerprint-keyed baseline, enforce policy.

Rules (CLAUDE.md sections 3 & 4):
  * Baselines are keyed by a machine+toolchain fingerprint (tools/fingerprint.sh). We REFUSE to
    compare across fingerprints -- a Kaby Lake AVX2 number and an M-series / AVX-512 number are not
    comparable, and silently comparing them would manufacture a fake speedup.
  * A metric passes iff  speedup = quantlib_ns / ours_ns  >=  thresholds.min_speedup_vs_quantlib
    (this ratio is load-robust: both sides are measured back to back) AND we have not self-regressed
    beyond thresholds.max_self_regression vs the committed ours_ns for this fingerprint.

Usage:
  check_perf.py --build BUILD --baselines FILE        # the gate (exit 0 iff all metrics pass)
  check_perf.py --build BUILD --baselines FILE --update  # re-capture this fingerprint's baseline
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# metric key -> (benchmark executable, QuantLib BM name, our BM name)
METRICS = {
    "curve_build":         ("curve_build_bench", "BM_CurveBuild_QuantLib",   "BM_CurveBuild_Ours"),
    "risk_full_jacobian":  ("risk_bench",        "BM_Risk_QuantLib_Bump",    "BM_Risk_Ours_Analytic"),
    "portfolio_analytics": ("portfolio_bench",   "BM_Portfolio_QuantLib",    "BM_Portfolio_Ours"),
}
UNIT_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def fingerprint():
    out = subprocess.check_output([os.path.join(HERE, "fingerprint.sh")], text=True)
    return json.loads(out)


def run_bench(build, exe, min_time, reps):
    path = os.path.join(build, "bench", exe)
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark not built: {path} (run cmake --build first)")
    cmd = [path, "--benchmark_format=json", f"--benchmark_min_time={min_time}s"]
    if reps > 1:
        cmd += [f"--benchmark_repetitions={reps}", "--benchmark_report_aggregates_only=true"]
    data = json.loads(subprocess.check_output(cmd, text=True))
    times = {}  # base BM name -> real_time in ns (median if aggregated)
    for b in data["benchmarks"]:
        name = b["name"]
        if reps > 1:
            if not name.endswith("_median"):
                continue
            name = name[: -len("_median")]
        times[name] = b["real_time"] * UNIT_NS[b["time_unit"]]
    return times


def measure(build, min_time, reps):
    """Return {metric: {'quantlib_ns':.., 'ours_ns':.., 'speedup':..}}."""
    result = {}
    for key, (exe, ql_name, ours_name) in METRICS.items():
        t = run_bench(build, exe, min_time, reps)
        if ql_name not in t or ours_name not in t:
            raise KeyError(f"{exe}: expected {ql_name} and {ours_name}, got {list(t)}")
        ql, ours = t[ql_name], t[ours_name]
        result[key] = {"quantlib_ns": round(ql), "ours_ns": round(ours), "speedup": round(ql / ours, 2)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--update", action="store_true", help="rewrite this fingerprint's baseline")
    ap.add_argument("--min-time", default="0.5")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    with open(args.baselines) as f:
        base = json.load(f)
    fp = fingerprint()
    key = fp["key"]
    print(f">> fingerprint {key}  ({fp['cpu']}, {fp['isa']}, {fp['compiler']})")

    print(">> running benchmarks ...")
    meas = measure(args.build, args.min_time, args.reps)

    if args.update:
        m = base["machines"].setdefault(key, {})
        m["fingerprint"] = {k: fp[k] for k in
                            ("arch", "os", "cpu", "physical_cores", "isa", "doubles_per_register",
                             "compiler", "arch_flag")}
        # timestamp is passed by the caller's environment to stay reproducible-friendly
        m["captured_utc"] = os.environ.get("SWAPS_CAPTURE_UTC", m.get("captured_utc", "unknown"))
        m.setdefault("notes", "Captured by tools/check_perf.py --update.")
        m["metrics"] = {mk: dict(mv) for mk, mv in meas.items()}
        # keep any metric-specific extras (e.g. book_size) that were already there
        for mk in m["metrics"]:
            prev = base["machines"].get(key, {}).get("metrics", {}).get(mk, {})
            for extra in ("book_size",):
                if extra in prev:
                    m["metrics"][mk][extra] = prev[extra]
        with open(args.baselines, "w") as f:
            json.dump(base, f, indent=2)
            f.write("\n")
        print(f">> baseline updated for fingerprint {key}")
        return 0

    # ---- gate ----
    if key not in base["machines"]:
        print(f"!! no committed baseline for fingerprint {key}.")
        print("   Refusing to compare across fingerprints. Re-baseline with:")
        print("     SWAPS_CAPTURE_UTC=$(date -u +%FT%TZ) ./tools/check_perf.py "
              f"--build {args.build} --baselines {args.baselines} --update")
        return 1

    committed = base["machines"][key]["metrics"]
    thr = base["thresholds"]
    print()
    print(f"  {'metric':<22}{'speedup':>9}{'need>=':>8}{'ours_ns':>13}{'baseline':>13}{'regr':>7}  result")
    ok = True
    for key_m, mv in meas.items():
        t = thr[key_m]
        need = t["min_speedup_vs_quantlib"]
        max_regr = t["max_self_regression"]
        base_ours = committed.get(key_m, {}).get("ours_ns")
        speed_ok = mv["speedup"] >= need
        regr = (mv["ours_ns"] / base_ours) if base_ours else float("nan")
        regr_ok = (base_ours is None) or (regr <= max_regr)
        row_ok = speed_ok and regr_ok
        ok = ok and row_ok
        flag = "PASS" if row_ok else ("SLOW" if not speed_ok else "REGRESSED")
        print(f"  {key_m:<22}{mv['speedup']:>8.2f}x{need:>7.1f}x{mv['ours_ns']:>13,}"
              f"{(base_ours or 0):>13,}{regr:>6.2f}x  {flag}")
    print()
    print("PERF GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
