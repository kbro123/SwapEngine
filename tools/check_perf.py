#!/usr/bin/env python3
"""Performance gate — PRINCIPLES.md P9/P10 (ratified 2026-09-08).

The engine is gated against ITSELF and against ABSOLUTE desk-scale TARGETS, never against QuantLib:

  HARD  self-baseline   ours_ns <= max_self_regression (1.25) x committed ours_ns for THIS fingerprint
  HARD  absolute target ours_ns <= baselines/targets.json[metric].target_ns   (targets only ratchet DOWN)
  INFO  reference       QuantLib (and any other reference BM) timings + speedups are printed and recorded,
                        never gated. They inform the commercial story; publish losses as well as wins.

Integrity (P10):
  * Baselines are keyed by a machine+toolchain fingerprint (tools/fingerprint.sh) that includes the
    QuantLib reference's toolchain. Comparing across fingerprints is refused.
  * Timings are the MIN over --reps repetitions (load-robust estimator), each of --min-time seconds.
  * The gate REFUSES to run on a busy machine (CPU busier than --max-busy % over a 2 s sample; the 1-min
    load average is printed as a diagnostic only — on macOS it counts I/O-wait threads and is not a
    contention measure) instead of warning: a warning nobody reads is not a gate. --force-load runs anyway
    but disables --update.

Usage:
  check_perf.py --build BUILD --baselines FILE --targets FILE            # the gate (exit 0 iff all pass)
  check_perf.py ... --update                                             # re-capture THIS fingerprint's baseline
  check_perf.py ... --propose-targets [--headroom 1.3]                   # print target suggestions
  check_perf.py ... --record OUT.json                                    # also write the full measurement
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# metric key -> (benchmark executable, OURS BM name, REFERENCE BM name or None)
# The reference BM (QuantLib etc.) is informational only. Every metric is gated on self-baseline + target.
METRICS = {
    # --- THE SHAPE LADDER (bench/fixtures/shape_ladder.hpp): every instrument shape, simplest -> most complex; real DB
    #     conventions and dates; stream tick (0.1 bp, frozen J) / refresh tick (25 bp: J + factorise) / hybrid Jacobian ---
    "shape_ois_nolag_stream_tick": ("shape_ladder_bench", "BM_Shape_ois_nolag_StreamTick", None),
    "shape_ois_nolag_refresh_25bp": ("shape_ladder_bench", "BM_Shape_ois_nolag_RefreshTick25bp", None),
    "shape_ois_nolag_jacobian": ("shape_ladder_bench", "BM_Shape_ois_nolag_Jacobian", None),
    "shape_ois_lag_stream_tick": ("shape_ladder_bench", "BM_Shape_ois_lag_StreamTick", None),
    "shape_ois_lag_refresh_25bp": ("shape_ladder_bench", "BM_Shape_ois_lag_RefreshTick25bp", None),
    "shape_ois_lag_jacobian": ("shape_ladder_bench", "BM_Shape_ois_lag_Jacobian", None),
    "shape_ibor_multicurve_stream_tick": ("shape_ladder_bench", "BM_Shape_ibor_multicurve_StreamTick", None),
    "shape_ibor_multicurve_refresh_25bp": ("shape_ladder_bench", "BM_Shape_ibor_multicurve_RefreshTick25bp", None),
    "shape_ibor_multicurve_jacobian": ("shape_ladder_bench", "BM_Shape_ibor_multicurve_Jacobian", None),
    "shape_basis_spread_stream_tick": ("shape_ladder_bench", "BM_Shape_basis_spread_StreamTick", None),
    "shape_basis_spread_refresh_25bp": ("shape_ladder_bench", "BM_Shape_basis_spread_RefreshTick25bp", None),
    "shape_basis_spread_jacobian": ("shape_ladder_bench", "BM_Shape_basis_spread_Jacobian", None),
    "shape_averaged_stream_tick": ("shape_ladder_bench", "BM_Shape_averaged_StreamTick", None),
    "shape_averaged_refresh_25bp": ("shape_ladder_bench", "BM_Shape_averaged_RefreshTick25bp", None),
    "shape_averaged_jacobian": ("shape_ladder_bench", "BM_Shape_averaged_Jacobian", None),
    "shape_averaged_leg_stream_tick": ("shape_ladder_bench", "BM_Shape_averaged_leg_StreamTick", None),
    "shape_averaged_leg_refresh_25bp": ("shape_ladder_bench", "BM_Shape_averaged_leg_RefreshTick25bp", None),
    "shape_averaged_leg_jacobian": ("shape_ladder_bench", "BM_Shape_averaged_leg_Jacobian", None),
    "shape_averaged_leg_moment_stream_tick": ("shape_ladder_bench", "BM_Shape_averaged_leg_moment_StreamTick", None),
    "shape_averaged_leg_moment_refresh_25bp": ("shape_ladder_bench", "BM_Shape_averaged_leg_moment_RefreshTick25bp", None),
    "shape_averaged_leg_moment_jacobian": ("shape_ladder_bench", "BM_Shape_averaged_leg_moment_Jacobian", None),
    "shape_banded_stream_tick": ("shape_ladder_bench", "BM_Shape_banded_StreamTick", None),
    "shape_banded_refresh_25bp": ("shape_ladder_bench", "BM_Shape_banded_RefreshTick25bp", None),
    "shape_banded_jacobian": ("shape_ladder_bench", "BM_Shape_banded_Jacobian", None),
    "shape_banded_edge_osc_tick": ("shape_ladder_bench", "BM_Shape_banded_EdgeOscTick", None),
    "shape_turns_stream_tick": ("shape_ladder_bench", "BM_Shape_turns_StreamTick", None),
    "shape_turns_refresh_25bp": ("shape_ladder_bench", "BM_Shape_turns_RefreshTick25bp", None),
    "shape_turns_jacobian": ("shape_ladder_bench", "BM_Shape_turns_Jacobian", None),
    "shape_turns_edge_osc_tick": ("shape_ladder_bench", "BM_Shape_turns_EdgeOscTick", None),
    "shape_portfolio_stream_tick": ("shape_ladder_bench", "BM_Shape_portfolio_StreamTick", None),
    "shape_portfolio_refresh_25bp": ("shape_ladder_bench", "BM_Shape_portfolio_RefreshTick25bp", None),
    "shape_portfolio_jacobian": ("shape_ladder_bench", "BM_Shape_portfolio_Jacobian", None),
    "shape_zero_coupon_stream_tick": ("shape_ladder_bench", "BM_Shape_zero_coupon_StreamTick", None),
    "shape_zero_coupon_refresh_25bp": ("shape_ladder_bench", "BM_Shape_zero_coupon_RefreshTick25bp", None),
    "shape_zero_coupon_jacobian": ("shape_ladder_bench", "BM_Shape_zero_coupon_Jacobian", None),
    "shape_fx_xccy_stream_tick": ("shape_ladder_bench", "BM_Shape_fx_xccy_StreamTick", None),
    "shape_fx_xccy_refresh_25bp": ("shape_ladder_bench", "BM_Shape_fx_xccy_RefreshTick25bp", None),
    "shape_fx_xccy_jacobian": ("shape_ladder_bench", "BM_Shape_fx_xccy_Jacobian", None),
    "shape_desk_stream_tick": ("shape_ladder_bench", "BM_Shape_desk_StreamTick", None),
    "shape_desk_refresh_25bp": ("shape_ladder_bench", "BM_Shape_desk_RefreshTick25bp", None),
    "shape_desk_jacobian": ("shape_ladder_bench", "BM_Shape_desk_Jacobian", None),
    "shape_desk_edge_osc_tick": ("shape_ladder_bench", "BM_Shape_desk_EdgeOscTick", None),
    # --- calibration kernels (single curve, 23x23; QuantLib GlobalBootstrap is the reference) ---
    "sofr_23k_square_cold_calibrate": ("curve_build_bench",   "BM_CurveBuild_Ours",              "BM_CurveBuild_QuantLib"),
    "sofr_23k_risk_ladder_23q_book9": ("risk_bench",          "BM_Risk_Ours_Analytic",           "BM_Risk_QuantLib_Bump"),
    "sofr_23k_book1000_ois_reprice": ("portfolio_bench",     "BM_Portfolio_Ours",               "BM_Portfolio_QuantLib"),
    "sofr_23k_warm_recal_0p3bp": ("warm_bench",          "BM_WarmRecal_Ours",               "BM_WarmRecal_QuantLib"),
    "sofr_23k_warm_recal_10bp": ("warm_bench",          "BM_WarmRecal_Ours_10bp",          "BM_WarmRecal_QuantLib_10bp"),
    # --- bonds ---
    "ust_5000_clean_to_yield_sweep": ("bond_sweep_bench",    "BM_BondSweep_Ours",               "BM_BondSweep_QuantLib"),
    "ust_5000_curve_book_dirty_price": ("bond_sweep_bench",    "BM_BondBook_Ours",                "BM_BondBook_QuantLib"),
    # --- options ---
    "sofr_swaption_cube17_sabr_reprice_warm": ("vol_cube_bench",      "BM_VolCube_Warm",                 None),
    "sofr_swaption_cube17_cold_calibrate_price": ("vol_cube_bench",      "BM_VolCube_Cold",                 None),
    # --- the SHIPPED session paths at desk scale (8-curve spread chain, 26 knots/curve) ---
    "chain8x26_session_cold_build_calibrate": ("session_warm_bench",  "BM_Session_ColdBuildCalibrate",   None),
    "chain8x26_session_rebind": ("session_warm_bench",  "BM_Session_RebindWarm",           None),
    "chain8x26_session_rebind_tension_reg": ("session_warm_bench", "BM_Session_RebindTensionWarm",   None),
    "chain8x26_session_stream_tick": ("session_warm_bench",  "BM_Session_StreamTick",           None),
    "chain8x26_book200_price_portfolio_oneshot": ("session_warm_bench",  "BM_Session_PricePortfolio",       None),
    # --- bundle kernels at desk scale ---
    "chain8x26_kernel_cold_joint_lm": ("bundle_scale_bench",  "BM_BundleScale_ColdJoint",        None),
    "chain8x26_kernel_one_residual": ("bundle_scale_bench",  "BM_BundleScale_OneResidual",      None),
    "chain8x26_kernel_one_jacobian": ("bundle_scale_bench",  "BM_BundleScale_OneJacobian",      None),
    # --- compiled book (the reprice_bound kernel) ---
    "chain8x26_book200_compiled_reprice": ("compiled_multi_bench", "BM_MultiCurveBook_Compiled",     "BM_MultiCurveBook_Templated"),
    # --- FX/MtM hybrid tier ---
    "usd_eur_xccy_3c8k_hybrid_jacobian": ("fx_stream_bench",     "BM_FxStream_HybridJacobian",      None),
    "usd_eur_xccy_3c8k_aad_block_jacobian": ("fx_stream_bench",    "BM_FxStream_AadBlockJacobian",    "BM_FxStream_AadBlockJacobianHeap"),
    "usd_eur_xccy_3c8k_stream_tick": ("fx_stream_bench",     "BM_FxStream_StreamTick",          None),
}
UNIT_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def fingerprint():
    out = subprocess.check_output([os.path.join(HERE, "fingerprint.sh")], text=True)
    return json.loads(out)


def load_avg():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return float("nan")


def cpu_busy_pct(sample_s=2.0):
    """Instantaneous CPU busy % (100 - idle). On macOS the 1-min load average counts threads in I/O wait
    (Spotlight, iCloud, dasd) and sat at 4-8 with the CPU 97% idle on 2026-09-08 — useless as a quiesce test.
    A short direct sample of CPU idle is what actually predicts benchmark contention."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["/usr/bin/top", "-l", "2", "-n", "0", "-s", str(int(sample_s))],
                                          text=True, stderr=subprocess.DEVNULL)
            idle = [float(m) for m in re.findall(r"CPU usage:.*?([\d.]+)% idle", out)]
            return 100.0 - idle[-1] if idle else float("nan")
        def snap():
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            return sum(v), v[3] + v[4]
        t0, i0 = snap(); time.sleep(sample_s); t1, i1 = snap()
        return 100.0 * (1.0 - (i1 - i0) / max(1, t1 - t0))
    except Exception:
        return float("nan")


def run_bench(build, exe, min_time, reps):
    """Run one benchmark executable; return {BM name: min real_time in ns over repetitions}."""
    path = os.path.join(build, "bench", exe)
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark not built: {path} (run cmake --build; is QuantLib present?)")
    cmd = [path, "--benchmark_format=json", f"--benchmark_min_time={min_time}s",
           f"--benchmark_repetitions={reps}", "--benchmark_report_aggregates_only=false"]
    data = json.loads(subprocess.check_output(cmd, text=True))
    times = {}
    for b in data["benchmarks"]:
        if b.get("run_type", "iteration") != "iteration":
            continue  # skip mean/median/stddev aggregates; we take the MIN over repetitions ourselves
        name = b["name"]
        ns = b["real_time"] * UNIT_NS[b["time_unit"]]
        times[name] = min(times.get(name, float("inf")), ns)
    return times


def measure(build, min_time, reps, only=None):
    """Return {metric: {'ours_ns', 'reference_ns', 'speedup'}} running each executable ONCE."""
    by_exe = {}
    for key, (exe, ours, ref) in METRICS.items():
        if only and key not in only:
            continue
        by_exe.setdefault(exe, []).append((key, ours, ref))
    result = {}
    for exe, items in by_exe.items():
        t = run_bench(build, exe, min_time, reps)
        for key, ours, ref in items:
            if ours not in t:
                raise KeyError(f"{exe}: expected {ours}, got {sorted(t)}")
            o = t[ours]
            r = t.get(ref) if ref else None
            result[key] = {"ours_ns": round(o),
                           "reference_ns": (round(r) if r is not None else None),
                           "reference_bm": ref,
                           "speedup": (round(r / o, 2) if r is not None else None)}
    return result


def fmt_ns(ns):
    if ns is None:
        return "      —"
    if ns >= 1e6:
        return f"{ns/1e6:8.2f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:8.1f} us"
    return f"{ns:8.0f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--targets", default=os.path.join(ROOT, "baselines", "targets.json"))
    ap.add_argument("--update", action="store_true", help="rewrite this fingerprint's baseline (quiesced only)")
    ap.add_argument("--propose-targets", action="store_true")
    ap.add_argument("--headroom", type=float, default=1.3)
    ap.add_argument("--min-time", default="0.5")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-busy", type=float, default=15.0, help="refuse to run if the CPU is more than this %% busy (2 s sample)")
    ap.add_argument("--max-load", type=float, default=None, help="(legacy) also refuse above this 1-min load average; off by default — macOS load counts I/O-wait threads")
    ap.add_argument("--force-load", action="store_true", help="run under load anyway (disables --update)")
    ap.add_argument("--only", nargs="*", help="metric keys to run (default: all)")
    ap.add_argument("--record", help="write the full measurement + verdicts to this JSON file")
    ap.add_argument("--describe", action="store_true", help="print what each metric builds/measures (from baselines.json) and exit")
    args = ap.parse_args()

    with open(args.baselines) as f:
        base = json.load(f)
    targets = {}
    if os.path.exists(args.targets):
        with open(args.targets) as f:
            targets = json.load(f).get("targets", {})

    if args.describe:
        for mk in METRICS:
            print(f"{mk}\n    {base.get('metric_descriptions', {}).get(mk, '(no description)')}\n")
        return 0

    fp = fingerprint()
    key = fp["key"]
    la = load_avg()
    busy = cpu_busy_pct()
    print(f">> fingerprint {key}  ({fp['cpu']}, {fp['isa']} {fp['arch_flag']}, {fp['compiler']})")
    print(f">> quantlib toolchain: {fp.get('quantlib_toolchain', 'unknown')}")
    print(f">> cpu busy (2 s sample): {busy:.1f}%  (max {args.max_busy}%)   load average (1 min, diagnostic): {la:.2f}")
    quiesced = not (busy > args.max_busy) and not (args.max_load is not None and la > args.max_load)
    if not quiesced:
        if not args.force_load:
            print("!! machine is busy — refusing to run the perf gate (use --force-load to run anyway; "
                  "results will not be eligible for --update).")
            return 2
        print("!! running under load (--force-load): numbers are NOT authoritative; --update disabled.")

    print(f">> running benchmarks (min over {args.reps} reps x {args.min_time}s) ...")
    t0 = time.time()
    meas = measure(args.build, args.min_time, args.reps, set(args.only) if args.only else None)
    print(f">> done in {time.time()-t0:.0f}s")

    # ---- --update ----
    if args.update:
        if not quiesced:
            print("!! --update refused: machine not quiesced.")
            return 2
        if args.only:
            print("!! --update refused with --only (baselines must be captured together).")
            return 2
        m = base["machines"].setdefault(key, {})
        m["fingerprint"] = {k: fp[k] for k in fp if k != "key"}
        m["captured_utc"] = os.environ.get("SWAPS_CAPTURE_UTC", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        m["load_avg_at_capture"] = round(la, 2)
        m["cpu_busy_pct_at_capture"] = round(busy, 1)
        m["reps"] = args.reps
        m["min_time_s"] = float(args.min_time)
        m.setdefault("notes", "Captured by tools/check_perf.py --update (min-of-reps, quiesced).")
        m["metrics"] = {mk: dict(mv) for mk, mv in meas.items()}
        with open(args.baselines, "w") as f:
            json.dump(base, f, indent=2)
            f.write("\n")
        print(f">> baseline updated for fingerprint {key}")

    # ---- --propose-targets ----
    if args.propose_targets:
        print(f"\n  proposed targets (ours_ns x {args.headroom}, rounded up; edit baselines/targets.json by hand):")
        for mk, mv in meas.items():
            cur = targets.get(mk, {}).get("target_ns")
            prop = int(mv["ours_ns"] * args.headroom)
            note = "" if cur is None else (f"  current {cur:,}" + ("  (RATCHET DOWN)" if prop < cur else ""))
            print(f"    {mk:<44}{prop:>14,}{note}")

    # ---- the gate ----
    if key not in base["machines"]:
        print(f"!! no committed baseline for fingerprint {key}. Refusing to compare across fingerprints.")
        print("   Re-baseline (quiesced) with:  ./tools/check_perf.py --build BUILD --baselines FILE --update")
        return 1
    committed = base["machines"][key]["metrics"]
    max_regr = float(base.get("policy", {}).get("max_self_regression", 1.25))

    print()
    print(f"  {'metric':<44}{'ours':>12}{'baseline':>12}{'regr':>7}{'target':>12}  result")
    ok = True
    verdicts = {}
    for mk, mv in meas.items():
        ours = mv["ours_ns"]
        b = committed.get(mk, {}).get("ours_ns")
        tgt = targets.get(mk, {}).get("target_ns")
        regr = (ours / b) if b else None
        flags = []
        if b is None:
            flags.append("NO-BASELINE")
        elif regr > max_regr:
            flags.append(f"REGRESSED>{max_regr}x")
        if tgt is None:
            flags.append("NO-TARGET")
        elif ours > tgt:
            flags.append("ABOVE-TARGET")
        row_ok = not any(f.startswith(("REGRESSED", "ABOVE")) for f in flags)
        ok = ok and row_ok
        verdicts[mk] = {"ok": row_ok, "flags": flags, "regr": regr, "target_ns": tgt}
        print(f"  {mk:<44}{fmt_ns(ours):>12}{fmt_ns(b):>12}"
              f"{(f'{regr:5.2f}x' if regr else '    — '):>7}{fmt_ns(tgt):>12}  "
              f"{'PASS' if row_ok else 'FAIL'}{(' ' + ','.join(flags)) if flags else ''}")

    # ---- informational reference table ----
    refs = [(mk, mv) for mk, mv in meas.items() if mv["reference_ns"] is not None]
    if refs:
        print("\n  reference (informational, NOT gated):")
        print(f"  {'metric':<44}{'reference':>12}{'ours':>12}{'speedup':>9}  reference BM")
        for mk, mv in refs:
            print(f"  {mk:<44}{fmt_ns(mv['reference_ns']):>12}{fmt_ns(mv['ours_ns']):>12}"
                  f"{mv['speedup']:>8.2f}x  {mv['reference_bm']}")

    if args.record:
        with open(args.record, "w") as f:
            json.dump({"fingerprint": fp, "load_avg": la, "cpu_busy_pct": busy, "quiesced": quiesced, "reps": args.reps,
                       "min_time_s": float(args.min_time), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "metrics": meas, "verdicts": verdicts}, f, indent=2)
    print()
    print("PERF GATE: " + ("PASS" if ok else "FAIL") + ("" if quiesced else "  (under load — not authoritative)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
