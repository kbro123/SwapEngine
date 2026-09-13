#!/usr/bin/env python3
"""tools/mutate.py — T6 MUTATION GATE for the engine's header-only kernels (E5.4, 2026-09-10).

Each curated mutation is a single exact-string edit of one header (a real bug class: a swapped weight, a
dropped chain-rule term, a stale memo, a wrong quantile, a structural field ignored ...). For every mutation
the harness copies the header into a scratch include root, applies the edit, compiles the test TUs that
CLAIM to pin that behaviour against the mutated root (searched before include/), runs them, and records
"caught" when the binary fails or crashes. The kill rate (caught / total) is the gate: below --min-kill the
tool exits 1 and names the survivors -- each survivor is a test that does not pin what it claims.

  python3 tools/mutate.py [--jobs 4] [--only NAME[,NAME]] [--min-kill 0.9] [--keep]

Only swaps_tests TUs (header-only, gtest) are used: compiling them standalone takes ~30-90 s each, so the
whole set is a few minutes with --jobs 4. Add a mutation when you fix a bug that a test should have caught:
the harness is the executable form of "each test shown to fail on the reverted bug".
"""
import argparse, concurrent.futures, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTEST_INC = "third_party/gtest/src/googletest/include"
GTEST_LIBS = ["third_party/gtest/build/lib/libgtest.a", "third_party/gtest/install/lib/libgtest_main.a"]
CXX = ["xcrun", "-sdk", "macosx", "clang++"] if sys.platform == "darwin" else ["c++"]
FLAGS = ["-std=c++20", "-O2", "-DNDEBUG", "-fno-math-errno", "-w"]

# (name, header (repo-relative: include/... or tests/research/...), old, new, [test TU, ...], gtest filter, what a survivor would mean)
MUTATIONS = [
    # 2026-09-13 risk-scale fixes: the quote ladder and identifiability carry the residual market scale D.
    ("ladder_market_scale_dropped", "include/swaps/calibration/risk.hpp",
     "  out.full.head(n_res).array() *= market_scale.array();  // dP/dq on the real quotes",
     "",
     ["consistent_risk_library_test.cpp"], "*", "the generate_risk ladder's market scale is unpinned"),
    ("identifiability_market_scale_not_divided", "include/swaps/calibration/diagnostics.hpp",
     "      P.col(i) /= market_scale[i];",
     "      P.col(i) /= 1.0;",
     ["calibration_diagnostics_test.cpp"], "*", "identifiability's market-scale division is unpinned"),
    # E7 stage 3.7: calibration/consistent_risk.hpp and null_completed_ladder, the library behind generate_risk.
    ("consistent_risk_relevel_skipped", "include/swaps/calibration/consistent_risk.hpp",
     "      for (auto& ins : bk.instruments) ins.market = anchor.model_quote(ins);",
     "",
     ["consistent_risk_library_test.cpp"], "*", "re-leveling a bundle onto the anchor is unpinned (E7 3.7)"),
    ("consistent_risk_bp_scale_drifts", "include/swaps/calibration/consistent_risk.hpp",
     "inline constexpr double kBasisPoint = 1e-4;",
     "inline constexpr double kBasisPoint = 1e-2;",
     ["consistent_risk_library_test.cpp"], "*", "the ladder DV01's basis-point scale is unpinned (E7 3.7)"),
    ("null_completion_threshold_dropped", "include/swaps/calibration/risk.hpp",
     "    svd.setThreshold(kRankThreshold);",
     "",
     ["consistent_risk_library_test.cpp"], "*", "the null completion's shared rank threshold is unpinned (E7 3.7)"),
    # E7 stage 3.6: calibration/diagnostics.hpp and flat_x0, the library behind calib_report.
    ("condition_number_inverted", "include/swaps/calibration/diagnostics.hpp",
     "double cond = smin > 0.0 ? smax / smin : std::numeric_limits<double>::max();",
     "double cond = smin > 0.0 ? smin / smax : std::numeric_limits<double>::max();",
     ["calibration_diagnostics_test.cpp"], "*", "the condition number's orientation is unpinned (E7 3.6)"),
    ("hat_diagonal_unclamped", "include/swaps/calibration/diagnostics.hpp",
     "h[i] = std::clamp(J.row(i).dot(M.col(i)), 0.0, 1.0);",
     "h[i] = J.row(i).dot(M.col(i));",
     ["calibration_diagnostics_test.cpp"], "*", "the identifiability clamp is unpinned (E7 3.6)"),
    ("flat_x0_turns_seed_at_the_level", "include/swaps/calibration/bundle_problem.hpp",
     "x[o++] = (i < ni) ? v : 0.0;",
     "x[o++] = v;",
     ["calibration_diagnostics_test.cpp"], "*", "flat_x0's zero turn seed is unpinned (E7 3.6)"),
    # E7 stage 3.5: the asset-swap float leg rolls backward from the maturity (short front stub).
    ("asset_swap_leg_rolls_from_settlement", "include/swaps/build/par_asset_swap.hpp",
     "  rule.side = StubSide::Front;",
     "  rule.side = StubSide::Back;",
     ["asset_swap_leg_test.cpp"], "*", "the asset-swap float leg's roll anchor is unpinned (E7 3.5)"),
    # E7 stage 3.4: the RV library behind bond_universe / govvie_fit / swap_spread.
    ("rv_settlement_lag_override_ignored", "include/swaps/derive/asset_swap.hpp",
     "  conv.settle_lag = o.lag ? *o.lag : bc.settle_lag;",
     "  conv.settle_lag = bc.settle_lag;",
     ["derive_rv_test.cpp"], "*", "the settlement-lag override is unpinned (E7 3.4)"),
    ("rv_zspread_target_is_clean_not_dirty", "include/swaps/derive/bond_rv.hpp",
     "target_dirty[b] = fit.market_clean[b] + fit.bonds[static_cast<std::size_t>(b)].accrued;",
     "target_dirty[b] = fit.market_clean[b];",
     ["derive_rv_test.cpp"], "*", "the z-spread dirty target is unpinned (E7 3.4)"),
    ("rv_anchor_default_is_bond_maturity", "include/swaps/derive/asset_swap.hpp",
     "  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, mat));",
     "  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, r.bond.maturity));",
     ["derive_rv_test.cpp"], "*", "the swap-spread anchor default is unpinned (E7 3.4)"),
    ("rv_parametric_tau2_request_dropped", "include/swaps/derive/bond_rv.hpp",
     "r.tau2.value_or(defaults.tau2)",
     "defaults.tau2",
     ["derive_rv_test.cpp"], "*", "a requested parametric decay tau2 is unpinned (E7 3.4)"),
    # E7 stage 3.3: the exchange invoices on the ROUNDED conversion factor (CME IR232).
    ("bond_future_cf_left_unrounded", "include/swaps/build/bond_future.hpp",
     "  return std::round(cf * scale) / scale;",
     "  return cf;",
     ["delivery_basket_test.cpp"], "*", "the exchange's CF rounding is unpinned (E7 3.3)"),
    # ... and counts the remaining term from the 1st of the delivery month, not from the date passed.
    ("bond_future_term_counts_from_the_delivery_day", "include/swaps/build/bond_future.hpp",
     "  return {total / 12, (total % 12 / round_months) * round_months};",
     "  const int t2 = total - (int(maturity.day()) < int(first_delivery.day()) ? 1 : 0);\n"
     "  return {t2 / 12, (t2 % 12 / round_months) * round_months};",
     ["delivery_basket_test.cpp"], "*", "the CF term's first-of-month rule is unpinned (E7 3.3)"),
    # E7 stage 3.2: the one terms->bond builder. Ignoring a request's frequency override would silently price a
    # non-catalogued bond on the convention's schedule.
    ("bond_terms_freq_override_ignored", "include/swaps/build/bond.hpp",
     "return t.freq ? *t.freq : int(yield_convention(t.convention).freq + 0.5);",
     "return int(yield_convention(t.convention).freq + 0.5);",
     ["bond_terms_test.cpp"], "*", "the bond terms builder's frequency override is unpinned (E7 3.2)"),
    # ... and the street analytics' clean price must net out accrued.
    ("street_analytics_clean_keeps_accrued", "include/swaps/pricing/bond.hpp",
     "a.clean = a.dirty - b.accrued;",
     "a.clean = a.dirty;",
     ["bond_terms_test.cpp"], "*", "street_analytics' clean/dirty relation is unpinned (E7 3.2)"),
    # E7 stage 3: the one smoothing table. A drifted light value would re-smooth every verb's default calibration.
    ("smoothing_light_value_drifts", "include/swaps/calibration/regularize.hpp",
     "case Smoothing::Light: return tension ? 0.02 : 0.5;",
     "case Smoothing::Light: return tension ? 0.2 : 0.5;",
     ["smoothing_preset_test.cpp"], "*", "the composer's smoothing table is unpinned (E7 stage 3)"),
    ("hermite_bessel_weights_swapped", "include/swaps/curve/regions.hpp",
     "m[j] = (h[j] * sec[j - 1] + h[j - 1] * sec[j]) / (h[j - 1] + h[j]);",
     "m[j] = (h[j - 1] * sec[j - 1] + h[j] * sec[j]) / (h[j - 1] + h[j]);",
     ["kernel_pins_test.cpp"], "SchemeValues.*", "the Hermite interpolant has no value pin (audit M5)"),
    ("band_chain_rule_dropped", "include/swaps/calibration/compiled_bundle.hpp",
     "const double sc = band_residual_d(qb_[i], q[b.row], b.lower, b.upper, b.decay).second;",
     "const double sc = 1.0;",
     ["portfolio_instrument_test.cpp"], "BandResidual.*", "the band chain rule is tested only outside the band (audit M4)"),
    ("quotient_rule_annuity_term_dropped", "include/swaps/calibration/compiled_bundle.hpp",
     "q_rows_[j].weight * (dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]));",
     "q_rows_[j].weight * (dnum.row(j) / ann[j]);",
     ["generic_instrument_test.cpp"], "*", "the analytic Jacobian is not compared to AAD (audit M4b)"),
    ("df_memo_never_invalidates", "include/swaps/calibration/compiled_bundle.hpp",
     "if (x.size() != df_x_.size() || (x.array() != df_x_.array()).any()) {",
     "if (x.size() != df_x_.size()) {",
     ["generic_instrument_test.cpp", "portfolio_instrument_test.cpp"], "*", "a stale DF memo is invisible (audit M9)"),
    ("pfe_is_the_median", "include/swaps/xva/exposure.hpp",
     "std::floor(pfe_q * (n_paths - 1))", "std::floor(0.5 * (n_paths - 1))",
     ["xva_exposure_test.cpp"], "*", "PFE is not pinned to its quantile (audit finding 9)"),
    ("sinhm1_x7_coefficient", "include/swaps/curve/regions.hpp",
     "s = s * x2 + 1.0 / 5040.0;", "s = s * x2 + 1.0 / 5000.0;",
     ["kernel_pins_test.cpp", "tension_test.cpp"], "*", "the tension series helpers have no accuracy test (audit M5c)"),
    ("coshm2_x10_term_dropped", "include/swaps/curve/regions.hpp",
     "s = s * x2 + 1.0 / 3628800.0;    // x¹⁰/10!\n", "",
     ["bug_hunt_2026_09_test.cpp"], "BugHunt.TensionCoshm2*", "bug-hunt #8 is not regression-pinned (audit B2)"),
    ("tangent_product_rule_dropped", "tests/research/reverse.hpp",
     "return {a.v * b.v, a.v * b.d + a.d * b.v}; }", "return {a.v * b.v, a.v * b.d}; }",
     ["gamma_test.cpp"], "*", "second-order AAD is not FD-checked (audit M7b)"),
    ("rev_division_derivative_wrong", "tests/research/reverse.hpp",
     "rev_record<T>(a.idx, inv, b.idx, -(val)*inv));", "rev_record<T>(a.idx, inv, b.idx, -(val)));",
     ["gamma_test.cpp"], "*", "the reverse tape's division rule is not checked against forward AAD (audit M7a)"),
    ("structure_equal_ignores_leg_fx_spot", "include/swaps/calibration/structure_fingerprint.hpp",
     "if (a.fx_spot != b.fx_spot || a.coupons.size() != b.coupons.size()) return false;",
     "if (a.coupons.size() != b.coupons.size()) return false;",
     ["kernel_pins_test.cpp"], "StructureEqual.*", "a structural field can be dropped from the warm-vs-recompile gate unnoticed (audit M8)"),
    ("rescale_anchor_normalisation_dropped", "include/swaps/calibration/streaming.hpp",
     "J_cur_.row(row) = J_ref_.row(row) * (new_slope / slope_ref_[k]);",
     "J_cur_.row(row) = J_ref_.row(row) * new_slope;",
     ["streaming_band_test.cpp"], "StreamingBand.*", "a wrong band re-scale is rescued by a refresh (audit M2)"),
    ("streaming_ldlt_instead_of_cod", "include/swaps/calibration/streaming.hpp",
     "M_ = cod.solve(Eigen::MatrixXd::Identity(n_res_, n_res_));",
     "M_ = (J.transpose() * J).ldlt().solve(J.transpose());",
     ["rank_safety_test.cpp", "streaming_contract_test.cpp"], "*", "the rank-safe streaming operator has no non-oracle test (audit M3b)"),
    ("lm_rank_deficiency_never_reported", "include/swaps/calibration/lm.hpp",
     "res.rank_deficiency = n_knots - static_cast<int>(cod.rank());",
     "res.rank_deficiency = 0;",
     ["calibration_status_test.cpp", "rank_safety_test.cpp"], "*", "the LM min-norm completion has no non-oracle test (audit M3)"),
    # Re-anchored 2026-09-12: the weight now lives in the ONE decomposition (build::fixing_rows), so this
    # single mutation reaches every builder that consumes it -- which is the point of the collapse.
    ("averaged_daily_weight_by_curve_time", "include/swaps/build/observations.hpp",
     "                                 obs_weight(dc, cal, p.acc_start, p.acc_end, p.fix_start, p.fix_end)});",
     "                                 (curve_time(vd, p.fix_end) > curve_time(vd, p.fix_start))\n"
     "                                     ? year_frac(dc, p.acc_start, p.acc_end, cal) /\n"
     "                                           (curve_time(vd, p.fix_end) - curve_time(vd, p.fix_start))\n"
     "                                     : 1.0});",
     ["build_instruments_test.cpp"], "*", "the DAILY averaged observation's day-count weight is unpinned (item 17)"),
    # E3: drop the fixing that applies on a non-business START (roll forward instead of back), which is what
    # the builder did until 2026-09-10 -- 29/31 of the correct rate for a Saturday-start contract month.
    ("averaged_window_drops_its_leading_fixing", "include/swaps/build/observations.hpp",
     "  while (!is_business_day(cal, first)) first = first.plus_days(-1);  // the fixing that applies on `start`",
     "  while (!is_business_day(cal, first)) first = first.plus_days(1);",
     ["build_instruments_test.cpp"], "*", "a non-business-day window start is unpinned (E3)"),
    # E3, mirrored: truncate the TRAILING fixing at the window end, pricing a 2-day rate where the contract
    # pays Friday's 3-day rate for 2 days.
    ("averaged_window_truncates_its_trailing_fixing", "include/swaps/build/observations.hpp",
     "    Date fe = fs.plus_days(1);\n    while (!is_business_day(cal, fe)) fe = fe.plus_days(1);",
     "    Date fe = fs.plus_days(1);\n    while (!is_business_day(cal, fe)) fe = fe.plus_days(1);\n    if (end < fe) fe = end;",
     ["build_instruments_test.cpp"], "*", "a non-business-day window end is unpinned (E3)"),
    ("averaged_moment_weight_by_curve_time", "include/swaps/build/observations.hpp",
     "  o.sub_start = {a}; o.sub_end = {b};\n  o.tau_index = year_frac(dc, start, end, cal);",
     "  o.sub_start = {a}; o.sub_end = {b}; if (std::abs(rmax - 1.0) > 1e-15) o.weight = {rmax};\n  o.tau_index = year_frac(dc, start, end, cal);",
     ["build_instruments_test.cpp"], "*", "the MOMENT averaged observation's day-count weight is unpinned (item 17)"),
    ("hyman_clamp_disabled", "include/swaps/curve/regions.hpp",
     "correction = m[i] / abs(m[i]) * smin(abs(m[i]), abs(3.0 * S[0]));",
     "correction = m[i];",
     ["kernel_pins_test.cpp", "monotone_cubic_test.cpp"], "SchemeValues.*:MonotoneCubic.*", "the Hyman end-clamp branch is pinned only by the oracle binary"),
]


def compile_and_run(name, header, old, new, tus, flt, keep_dir, jobs_note=""):
    """Returns (name, caught: bool|None, detail)."""
    work = os.path.join(keep_dir, name)
    # the mutant lives in a scratch tree that mirrors the repo (include/... or tests/...); both roots are
    # searched BEFORE the real ones, so the TU sees the mutated header and the pristine rest
    mut = os.path.join(work, header)
    os.makedirs(os.path.dirname(mut), exist_ok=True)
    src = open(os.path.join(ROOT, header)).read()
    n = src.count(old)
    if n == 0:
        return name, None, f"anchor not found in {header} (the code moved; update the mutation)"
    open(mut, "w").write(src.replace(old, new))
    inc = os.path.join(work, "include")
    tst = os.path.join(work, "tests")
    os.makedirs(inc, exist_ok=True); os.makedirs(tst, exist_ok=True)
    for tu in tus:
        exe = os.path.join(work, tu.replace(".cpp", ""))
        cmd = CXX + FLAGS + ["-I", inc, "-I", tst, "-I", os.path.join(ROOT, "include"), "-I", os.path.join(ROOT, "third_party/eigen"),
                             "-I", os.path.join(ROOT, "third_party/boost"), "-I", os.path.join(ROOT, "build/generated"),
                             "-I", os.path.join(ROOT, "tests"), "-I", os.path.join(ROOT, GTEST_INC),
                             os.path.join(ROOT, "tests", tu)] + [os.path.join(ROOT, l) for l in GTEST_LIBS] + ["-o", exe]
        cc = subprocess.run(cmd, capture_output=True, text=True)
        if cc.returncode != 0:
            # a mutation that no longer compiles is a "caught at compile time" only if it is deliberate; report it
            return name, None, f"{tu} did not compile against the mutant:\n" + cc.stderr[-800:]
        run = subprocess.run([exe, f"--gtest_filter={flt}"], capture_output=True, text=True)
        if run.returncode != 0:
            failed = [l.strip() for l in run.stdout.splitlines() if l.startswith("[  FAILED  ]")]
            return name, True, f"{tu}: " + (failed[0] if failed else f"exit {run.returncode}")
    return name, False, "every claimed test PASSED against the mutant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--min-kill", type=float, default=0.9)
    ap.add_argument("--keep", action="store_true", help="keep the scratch dir (printed)")
    a = ap.parse_args()
    sel = [m for m in MUTATIONS if not a.only or m[0] in a.only.split(",")]
    for l in GTEST_LIBS:
        if not os.path.exists(os.path.join(ROOT, l)):
            print(f"mutate: missing {l} (bootstrap gtest first)", file=sys.stderr); return 2
    keep_dir = tempfile.mkdtemp(prefix="swaps-mutate-")
    t0 = time.time()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(compile_and_run, m[0], m[1], m[2], m[3], m[4], m[5], keep_dir): m for m in sel}
        for f in concurrent.futures.as_completed(futs):
            name, caught, detail = f.result()
            results[name] = (caught, detail)
    caught = sum(1 for c, _ in results.values() if c is True)
    broken = [n for n, (c, _) in results.items() if c is None]
    total = len(results) - len(broken)
    print(f"mutate: {len(sel)} mutations in {time.time() - t0:.0f}s (scratch {keep_dir if a.keep else 'removed'})")
    print(f"  {'mutation':44s} {'result':10s} detail")
    for m in sel:
        c, d = results[m[0]]
        tag = "CAUGHT" if c is True else ("SURVIVED" if c is False else "BROKEN")
        print(f"  {m[0]:44s} {tag:10s} {d.splitlines()[0] if d else ''}")
        if c is False:
            print(f"  {'':44s} {'':10s} => {m[6]}")
    rate = caught / total if total else 0.0
    print(f"mutate: kill rate {caught}/{total} = {rate:.0%} (gate {a.min_kill:.0%})" + (f"; {len(broken)} broken anchor(s)" if broken else ""))
    if not a.keep:
        shutil.rmtree(keep_dir, ignore_errors=True)
    return 0 if rate >= a.min_kill and not broken else 1


if __name__ == "__main__":
    sys.exit(main())
