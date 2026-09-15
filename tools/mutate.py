#!/usr/bin/env python3
"""tools/mutate.py — T6 MUTATION GATE for the engine's header-only kernels (E5.4, 2026-09-10).

Each curated mutation is a single exact-string edit of one header (a real bug class: a swapped weight, a
dropped chain-rule term, a stale memo, a wrong quantile, a structural field ignored ...). For every mutation
the harness copies the header into a scratch include root, applies the edit, compiles the test TUs that
CLAIM to pin that behaviour against the mutated root (searched before include/), runs them, and records
"caught" when the binary fails or crashes. The kill rate (caught / total) is the gate: below --min-kill the
tool exits 1 and names the survivors -- each survivor is a test that does not pin what it claims.

  python3 tools/mutate.py [--jobs 4] [--only NAME[,NAME]] [--min-kill 0.9] [--keep]

Test TUs are compiled standalone (header-only, gtest): ~30-90 s each, so the whole set is a few minutes with
--jobs 4. THE SESSION LAYER (2026-09-15, P3): a mutation whose TU list also names api/*.cpp sources rebuilds those
sources against the mutant (an api/*.cpp mutated directly, or a header they instantiate) and links them AHEAD of the
built build/api/libswaps_api.a -- the linker takes their symbols from the mutant and only untouched members from the
archive -- so a BundleSession behaviour is mutation-gated like a kernel. Add a mutation when you fix a bug that a test should have caught:
the harness is the executable form of "each test shown to fail on the reverted bug".
"""
import argparse, concurrent.futures, json, os, shlex, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTEST_INC = "third_party/gtest/src/googletest/include"
GTEST_LIBS = ["third_party/gtest/build/lib/libgtest.a", "third_party/gtest/install/lib/libgtest_main.a"]
CXX = ["xcrun", "-sdk", "macosx", "clang++"] if sys.platform == "darwin" else ["c++"]
FLAGS = ["-std=c++20", "-O2", "-DNDEBUG", "-fno-math-errno", "-w"]

# (name, header (repo-relative: include/..., tests/research/... or api/*.cpp), old, new, [api/*.cpp source, ..., test TU, ...],
#  gtest filter, what a survivor would mean)
MUTATIONS = [
    # 2026-09-15 G1-G3: the streamer's hot-path stages each carry an allocation / factorisation / refresh-count pin.
    ("rescale_always_refactorises", "include/swaps/calibration/streaming.hpp",
     "      if (std::isfinite(d) && std::abs(d) > 1e-3) {\n",
     "      if (std::isfinite(d) && std::abs(d) > 1e300) {\n",
     ["streaming_walk_guard_test.cpp"], "*", "a band crossing refactorising instead of its rank-one update is unpinned (G1)"),
    ("verify_pins_never_releases", "include/swaps/calibration/streaming.hpp",
     "&& releases_[k] == 0) {\n",
     "&& false) {\n",
     ["streaming_walk_guard_test.cpp"], "*", "the KKT release of a pin is unreached by the walk guard (G1)"),
    ("apply_pins_skipped", "include/swaps/calibration/streaming.hpp",
     "    if (n_pinned_ == 0) return;\n",
     "    return;\n",
     ["streaming_walk_guard_test.cpp", "streaming_band_test.cpp"], "*", "a pinned row's stiff residual is unpinned (G1)"),
    ("failed_tick_keeps_divergent_anchor", "include/swaps/calibration/streaming.hpp",
     "    if (t.refreshes > 0 || t.rescales > 0) (void)set_anchor(x_cur_, q_cur_);\n",
     "",
     ["streaming_walk_guard_test.cpp", "streaming_contract_test.cpp"], "*", "a failed tick restoring its anchor is unpinned (G1)"),
    ("final_refresh_repeats", "include/swaps/calibration/streaming.hpp",
     "          final_refresh_done = true;\n",
     "",
     ["streaming_walk_guard_test.cpp"], "*", "the one post-convergence refresh per drift refresh is unpinned (G2)"),
    ("drift_refresh_on_square", "include/swaps/calibration/streaming.hpp",
     "    drift_refresh_ = (n_res_ != static_cast<int>(x0.size())) || !bands_.empty() || RtR_.size() > 0;\n",
     "    drift_refresh_ = true;\n",
     ["streaming_walk_guard_test.cpp"], "*", "a square rung's 25 bp tick refreshing is unpinned (G2)"),
    ("tick_state_copied_per_tick", "include/swaps/calibration/streaming.hpp",
     "    Eigen::VectorXd& x = x_;  // reused scratch: the frozen-Newton loop below allocates nothing\n",
     "    Eigen::VectorXd x = x_;\n",
     ["streaming_walk_guard_test.cpp"], "*", "a per-tick allocation on the streamer is unpinned outside swaps_api_tests (G3)"),
    # 2026-09-15 P3: a streamed tick commits its targets to the session (the instruments and the shared engine).
    ("session_tick_quotes_not_committed", "api/bundle_api.cpp",
     "  cal::commit_targets(prob_.instruments, engine_.get(), new_market);\n",
     "",
     ["api/bundle_api.cpp", "session_stream_commit_repro_test.cpp"], "*", "a streamed tick leaving the session's quotes at the old market is unpinned (P3)"),
    ("commit_targets_skips_engine", "include/swaps/calibration/streaming.hpp",
     "  if (engine) engine->set_market(target);\n",
     "",
     ["api/bundle_api.cpp", "session_stream_commit_repro_test.cpp"], "*", "the shared engine keeping the old targets after a tick is unpinned (P3)"),
    # 2026-09-15 C6a: a refresh writes J in place and reuses one decomposition.
    ("refresh_jacobian_by_value", "include/swaps/calibration/streaming.hpp",
     "    engine_->jacobian_vs_into(x, q, J_ref_);  // in place (C6); band term consistent with residuals_vs(·,q)\n",
     "    J_ref_ = engine_->jacobian_vs(x, q);\n",
     ["streaming_refresh_alloc_repro_test.cpp"], "*", "a refresh writing J in place is unpinned (C6)"),
    ("decomposition_rebuilt_per_refresh", "include/swaps/calibration/streaming.hpp",
     "    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& cod = cod_;\n",
     "    cod_ = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>();\n    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& cod = cod_;\n",
     ["streaming_refresh_alloc_repro_test.cpp"], "*", "reusing the refresh's decomposition storage is unpinned (C6)"),
    ("stacked_regulariser_rows_not_written", "include/swaps/calibration/streaming.hpp",
     "      S_.bottomRows(opt_.regularizer.rows()) = opt_.regularizer;\n",
     "",
     ["jacobian_into_parity_test.cpp"], "*", "the regularised refresh's R rows are unpinned (C6)"),
    # 2026-09-15 C8: a tick reports the drift it refreshed on.
    ("tick_drift_zeroed_after_refresh", "include/swaps/calibration/streaming.hpp",
     "      if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);\n    }\n    int frozen = 0;\n",
     "      if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);\n      t.drift = 0.0;\n    }\n    int frozen = 0;\n",
     ["streaming_drift_report_repro_test.cpp"], "*", "reporting the drift of a tick that refreshed on it is unpinned (C8)"),
    # 2026-09-14 K5': a quote is {target, lower, upper, decay}; a target outside its band, an inverted band or a decay outside [0, 1]
    # is refused; a four-number requote moves the streamer's bands in place.
    ("quote_target_outside_band_accepted", "include/swaps/calibration/problem.hpp",
     "(target >= lower && target <= upper && decay >= 0.0 && decay <= 1.0)",
     "(decay >= 0.0 && decay <= 1.0)",
     ["quote_band_contract_test.cpp"], "*", "a target outside its band is accepted (K5')"),
    ("quote_inverted_band_misreported", "include/swaps/calibration/problem.hpp",
     "  else if (upper < lower)\n",
     "  else if (false)\n",
     ["quote_band_contract_test.cpp"], "*", "an inverted band is refused under another name (K5')"),
    ("quote_decay_outside_unit_accepted", "include/swaps/calibration/problem.hpp",
     "target <= upper && decay >= 0.0 && decay <= 1.0)))",
     "target <= upper)))",
     ["quote_band_contract_test.cpp"], "*", "a decay outside [0, 1] is accepted (K5')"),
    ("streamer_update_skips_band_check", "include/swaps/calibration/streaming.hpp",
     "      if (quote_up_[i] > quote_lo_[i] && (q_new[i] < quote_lo_[i] || q_new[i] > quote_up_[i]))\n",
     "      if (false)\n",
     ["quote_band_contract_test.cpp"], "*", "the streamer's own tick accepts a target outside its band (K5')"),
    ("set_bands_keeps_stale_band_values", "include/swaps/calibration/streaming.hpp",
     "          b.lower = lower[i];\n          b.upper = upper[i];\n",
     "",
     ["quote_band_contract_test.cpp"], "*", "the streamer's band table is not moved by a requote (K5')"),
    ("set_bands_ignores_membership_change", "include/swaps/calibration/streaming.hpp",
     "      if (tracked != was_tracked || banded != (quote_up_[i] > quote_lo_[i])) return false;\n",
     "",
     ["quote_band_contract_test.cpp"], "*", "a change in which rows are banded is applied in place (K5')"),
    ("band_target_pin_mismatch_accepted", "include/swaps/market/quote.hpp",
     "    if (target != upper) throw",
     "    if (false) throw",
     ["quote_band_contract_test.cpp"], "*", "a wire pin that differs from its target is accepted (K5')"),
    ("band_target_target_outside_accepted", "include/swaps/market/quote.hpp",
     "  if (target < lower || target > upper)\n    throw std::invalid_argument(\"quote: target outside",
     "  if (false)\n    throw std::invalid_argument(\"quote: target outside",
     ["quote_band_contract_test.cpp"], "*", "a wire quote whose target is outside its band is accepted (K5')"),
    # 2026-09-15 two-threshold streamer operator (anchor-relative): walk drops NEW weak directions, commit re-converges at full rank.
    ('walk_threshold_is_the_shared_one', "include/swaps/calibration/streaming.hpp",
     'return n_pinned_ > 0 ? kWalkRankThreshold / kPinWeight : kWalkRankThreshold; }\n',
     'return kRankThreshold; }\n',
     ["streaming_near_singular_repro_test.cpp"], "*", 'the walk threshold that stops a near-singular runaway is unpinned'),
    ('walk_threshold_ignores_pin_scale', "include/swaps/calibration/streaming.hpp",
     'return n_pinned_ > 0 ? kWalkRankThreshold / kPinWeight : kWalkRankThreshold; }\n',
     'return kWalkRankThreshold; }\n',
     ["streaming_near_singular_repro_test.cpp"], "*", "measuring the walk cut against a pinned row's 1e3 weight (a spurious full-rank re-anchor) is unpinned"),
    ('commit_rule_skips_the_full_rank_reconvergence', "include/swaps/calibration/streaming.hpp",
     '        if (truncated_ && !full_rank_) {\n',
     '        if (false) {\n',
     ["streaming_near_singular_repro_test.cpp"], "*", "committing a truncated operator's fixed point (silently wrong on a weak direction that appears mid-walk) is unpinned"),
    ('truncated_operator_not_flagged', "include/swaps/calibration/streaming.hpp",
     '      truncated_ = true;\n',
     '      truncated_ = false;\n',
     ["streaming_near_singular_repro_test.cpp"], "*", 'flagging a truncated operator for the commit rule is unpinned'),
    ('structural_weak_directions_truncated', "include/swaps/calibration/streaming.hpp",
     '    } else if (weak > anchor_weak_) {\n',
     '    } else if (weak > 0) {\n',
     ["streaming_near_singular_repro_test.cpp"], "*", "keeping the anchor's structural weak directions (the xccy stream tick's extra re-anchor) is unpinned"),
    ('walk_never_marked', "include/swaps/calibration/streaming.hpp",
     '    in_walk_ = true;\n',
     '    in_walk_ = false;\n',
     ["streaming_near_singular_repro_test.cpp"], "*", 'telling a walk factorisation from an anchor one is unpinned'),
    # 2026-09-14 FLK2: a full step reversing the previous one (a kink 2-cycle) is halved; the break-even can be pinned.
    ("kink_cycle_step_not_halved", "include/swaps/calibration/streaming.hpp",
     "            damp = 0.5;\n",
     "            damp = 1.0;\n",
     ["streaming_kink_cycle_repro_test.cpp"], "*", "breaking a MonotoneCubic kink 2-cycle is unpinned (FLK2)"),
    ("breakeven_override_ignored", "include/swaps/calibration/streaming.hpp",
     "      if (opt_.breakeven_steps > 0.0) breakeven_steps_ = opt_.breakeven_steps;  // pinned (Options::breakeven_steps)\n",
     "",
     ["streaming_kink_cycle_repro_test.cpp"], "*", "Options::breakeven_steps is unpinned (FLK2)"),
    # 2026-09-14 ASW weekend month-end: the asset-swap float leg ends on the maturity adjusted Following (ql::AssetSwap), the
    # interior boundaries keep the product convention.
    ("schedule_termination_ignores_rule", "include/swaps/build/schedule.hpp",
     "  const Date end = adjust(cal_id, maturity_date, rule.termination_bdc.empty() ? bdc : rule.termination_bdc);\n",
     "  const Date end = adjust(cal_id, maturity_date, bdc);\n",
     ["asset_swap_leg_test.cpp"], "*", "the schedule's termination convention is unpinned"),
    ("asw_float_leg_terminates_modified_following", "include/swaps/build/par_asset_swap.hpp",
     "  rule.termination_bdc = \"Following\";  // the float end sits on the adjusted redemption date\n",
     "",
     ["asset_swap_leg_test.cpp"], "*", "the asset-swap float end on the Following-adjusted maturity is unpinned"),
    # 2026-09-14 O-X3 fx_spot_time (piece 2): a passed FX fixing seasons the MtM coupon ...
    ("seasoning_ignores_fx_fixing", "include/swaps/pricing/cashflows.hpp",
     "  if (c.fx_fixing_set && c.fx_fixing_time < 0.0) return true;  // the FX already fixed: the notional is a known number",
     "",
     ["fx_spot_time_piece2_test.cpp"], "*", "a passed FX fixing making the coupon seasoned is unpinned"),
    ("kernel_prices_a_passed_fixing_off_the_forward", "include/swaps/pricing/cashflows.hpp",
     "    if (c.fx_fixing_set && c.fx_fixing_time < 0.0)\n      throw std::runtime_error(",
     "    if (false)\n      throw std::runtime_error(",
     ["fx_spot_time_piece2_test.cpp"], "*", "refusing a curve-implied forward for a known fixing is unpinned"),
    # ... the builder's CARR fixing lag and spot time ...
    ("builder_fixing_ignores_the_lag", "include/swaps/build/instruments.hpp",
     "      const Date fixing = advance_bd(x.fx_reset_calendar, periods[i].first, -x.fx_reset_lag);",
     "      const Date fixing = periods[i].first;",
     ["fx_spot_time_piece2_test.cpp"], "*", "the CARR fixing date (2 BD before the start) is unpinned"),
    ("builder_spot_time_is_today", "include/swaps/build/instruments.hpp",
     "  ins.mtm.fx_spot_time = curve_time(vd, spot_date(vd, x.calendar, x.spot_lag));",
     "  ins.mtm.fx_spot_time = 0.0;",
     ["fx_spot_time_piece2_test.cpp"], "*", "the MtM leg's spot-date quote is unpinned"),
    # ... and roll_book ages the fixing.
    ("roll_book_keeps_the_fixing_time", "include/swaps/calibration/pnl_explain.hpp",
     "        if (c.fx_fixing_set) c.fx_fixing_time -= dt;  // O-X3 piece 2: a fixing ages like any cashflow time (unfloored: passed = known)",
     "",
     ["fx_spot_time_piece2_test.cpp"], "*", "ageing the FX fixing time is unpinned"),
    # 2026-09-14 O-X3 fx_spot_time (piece 1): the FX outright rolls back from the spot date ...
    ("fx_forward_ignores_spot_time", "include/swaps/calibration/problem.hpp",
     "      if (ins.fx_spot_time == 0.0)  // the t = 0 reading, byte-identical",
     "      if (true)  // the t = 0 reading, byte-identical",
     ["fx_spot_time_repro_test.cpp"], "*", "the templated spot-date roll-back is unpinned (O-X3)"),
    # ... on the compiled row, value and Jacobian ...
    ("compiled_fx_row_ignores_spot_time", "include/swaps/calibration/compiled_bundle.hpp",
     "      if (f.idx_snum >= 0) out_[f.row] *= DF[f.idx_sden] / DF[f.idx_snum];  // O-X3: roll back from the spot date",
     "",
     ["fx_spot_time_compiled_test.cpp"], "*", "the compiled FX outright ignoring the spot time is unpinned"),
    ("compiled_fx_jacobian_drops_spot_entries", "include/swaps/calibration/compiled_bundle.hpp",
     "        G(f.row, f.idx_sden) += 1.0 / (DF[f.idx_sden] * f.fx_time);\n        G(f.row, f.idx_snum) += -1.0 / (DF[f.idx_snum] * f.fx_time);\n",
     "",
     ["fx_spot_time_compiled_test.cpp"], "*", "the compiled FX row's two spot-time Jacobian entries are unpinned"),
    # ... the MtM notional and the basis divisor ...
    ("mtm_notional_ignores_spot_time", "include/swaps/pricing/cashflows.hpp",
     "    if (fx_spot_time != 0.0) N = N * (denc.discount(fx_spot_time) / numc.discount(fx_spot_time));",
     "",
     ["fx_spot_time_repro_test.cpp"], "*", "N_0 = the spot quote (R4a) is unpinned"),
    ("basis_divides_by_raw_spot", "include/swaps/calibration/problem.hpp",
     "      return (pv_self - pv_fx) / ann + mtm / (fx0 * ann);",
     "      return (pv_self - pv_fx) / ann + mtm / (ins.mtm.fx_spot * ann);",
     ["fx_spot_time_compiled_test.cpp"], "*", "the basis quote's invariance to the spot time is unpinned"),
    # ... the compiled book's fallback, validation and structure.
    ("compiled_book_compiles_spot_time_position", "include/swaps/portfolio/compiled_multi.hpp",
     "    if (p.fx_spot_time != 0.0) return false;  // O-X3: the spot-date roll-back is curve-dependent; the templated fallback prices it",
     "",
     ["fx_spot_time_compiled_test.cpp"], "*", "compiling a spot-time xccy position (ignoring the roll-back) is unpinned"),
    ("validate_accepts_bad_spot_time", "include/swaps/calibration/problem.hpp",
     "      if (!(ins.fx_spot_time >= 0.0) || !std::isfinite(ins.fx_spot_time)) fail(\"an FX forward needs a finite fx_spot_time >= 0\");",
     "",
     ["fx_spot_time_compiled_test.cpp"], "*", "refusing a negative / non-finite FX spot time is unpinned"),
    ("structure_equal_ignores_fx_spot_time", "include/swaps/calibration/structure_fingerprint.hpp",
     "  if (a.fx_spot_time != b.fx_spot_time) return false;",
     "",
     ["fx_spot_time_compiled_test.cpp"], "*", "rebinding across a spot-time change is unpinned"),
    # 2026-09-14 FLK1: an allocation scope counts only the thread that armed it.
    ("alloc_scope_counts_every_thread", "bench/fixtures/malloc_count.hpp",
     "  if (!pthread_equal(pthread_self(), alloc_counting_thread())) return;  // FLK1: count only the arming thread\n",
     "",
     ["alloc_scope_thread_repro_test.cpp"], "*", "counting other threads' allocations in a hot-path pin is unpinned (FLK1)"),
    # 2026-09-14 xccy exchange-lag fields: a lagged notional exchange is refused, not ignored ...
    ("xccy_exchange_lag_accepted", "include/swaps/build/instruments.hpp",
     "  if (x.exchange_lag_initial != 0 || x.exchange_lag_intermediate != 0 || x.exchange_lag_final != 0)",
     "  if (false)",
     ["xccy_carr_example_repro_test.cpp"], "*", "refusing a lagged notional exchange is unpinned"),
    # ... and the DB rule requires the reset calendar.
    ("xccy_rule_fx_reset_calendar_optional", "include/swaps/conventions_db.hpp",
     '    r.need(p.fx_reset_calendar, "fx_reset_calendar");\n',
     "",
     ["conventions_rules_test.cpp"], "*", "the xccy fx_reset_calendar rule is unpinned"),
    # 2026-09-14 AS2: the asset-swap float accrual counts BUS/252 business days on the product calendar.
    ("asw_float_tau_without_calendar", "include/swaps/build/par_asset_swap.hpp",
     "    f.tau.push_back(year_frac(swc.float_dc, p.first, p.second, swc.calendar));",
     "    f.tau.push_back(year_frac(swc.float_dc, p.first, p.second));",
     ["asset_swap_bus252_repro_test.cpp"], "*", "the BUS/252 asset-swap float accrual is unpinned (AS2)"),
    # 2026-09-14 PN2b: a coupon rolled into its payment-lag window (accrual end < 0 < pay) owes its interest, not its settled
    # final notional exchange -- both the MtM leg and the xccy position's constant-notional domestic leg skip it.
    ("mtm_leg_books_settled_final_exchange", "include/swaps/pricing/cashflows.hpp",
     "    if (e >= 0.0) v += dc.discount(e);\n",
     "    v += dc.discount(e);\n",
     ["roll_book_settled_end_repro_test.cpp"], "*", "skipping a settled final MtM exchange is unpinned (PN2b)"),
    ("xccy_position_books_settled_final_exchange", "include/swaps/portfolio/portfolio.hpp",
     "        if (eN >= 0.0) dom = dom + C(p.disc_curve).discount(eN);\n",
     "        dom = dom + C(p.disc_curve).discount(eN);\n",
     ["roll_book_settled_end_repro_test.cpp"], "*", "skipping a settled final domestic exchange is unpinned (PN2b)"),
    ("compiled_xccy_books_settled_final_exchange", "include/swaps/portfolio/compiled_multi.hpp",
     "        if (eN >= 0.0) exch.push_back({eN, 1.0, 1.0});  // the final exchange, unless already settled (PN2b)\n",
     "        exch.push_back({eN, 1.0, 1.0});\n",
     ["roll_book_settled_end_repro_test.cpp"], "*", "the compiled domestic final-exchange guard is unpinned (PN2b)"),
    ("mtm_leg_skips_todays_final_exchange", "include/swaps/pricing/cashflows.hpp",
     "    if (e >= 0.0) v += dc.discount(e);\n",
     "    if (e > 0.0) v += dc.discount(e);\n",
     ["roll_book_settled_end_repro_test.cpp"], "*", "an exchange dated today still paying is unpinned (PN2b control)"),
    # 2026-09-14 PN2: roll_book ages the MtM accrual period and the principal exchanges with every other cashflow.
    ("roll_book_keeps_accrual_times", "include/swaps/calibration/pnl_explain.hpp",
     "          c.accrual_start -= dt;\n          c.accrual_end -= dt;\n",
     "",
     ["roll_book_accrual_repro_test.cpp"], "*", "the rolled MtM exchange dates are unpinned (PN2)"),
    ("roll_book_keeps_principal_times", "include/swaps/calibration/pnl_explain.hpp",
     "      q.principal_flows.emplace_back(shift ? t - dt : t, amount);",
     "      q.principal_flows.emplace_back(t, amount);",
     ["roll_book_accrual_repro_test.cpp"], "*", "the rolled principal exchange times are unpinned (PN2)"),
    ("roll_book_keeps_settled_principal", "include/swaps/calibration/pnl_explain.hpp",
     "      if (t <= dt) continue;\n",
     "",
     ["roll_book_accrual_repro_test.cpp"], "*", "dropping a settled principal exchange is unpinned (PN2)"),
    # 2026-09-14 ASW pay lag: the asset-swap float coupons pay the product's payment_lag after the accrual end ...
    ("asw_float_leg_pays_on_accrual_end", "include/swaps/build/par_asset_swap.hpp",
     "    f.pay.push_back(curve_time(value_date, advance_bd(swc.calendar, p.second, swc.pay_lag)));",
     "    f.pay.push_back(curve_time(value_date, p.second));",
     ["asset_swap_pay_lag_repro_test.cpp"], "*", "the asset-swap float leg's payment lag is unpinned"),
    # ... and the par spread carries the lag's deferred value E (a lagged floater is not DF(settle) - DF(T)).
    ("asw_spread_drops_lag_value", "include/swaps/build/par_asset_swap.hpp",
     "  return (dirty_curve - dirty_market + lag_value / df_settle) / annuity;",
     "  return (dirty_curve - dirty_market) / annuity;",
     ["asset_swap_pay_lag_repro_test.cpp"], "*", "the lagged par floater's deferred value E is unpinned"),
    # 2026-09-14 XB1: the xccy basis self leg is the exchange pair, paid on the accrual ends (not with the lagged coupons).
    ("xccy_self_leg_pays_lagged", "include/swaps/build/instruments.hpp",
     "  for (auto& c : ins.fwd.coupons) c.pay = c.accrual_end;\n",
     "\n",
     ["xccy_basis_exchange_pair_repro_test.cpp"], "*", "the exchange pair at the accrual dates is unpinned (XB1)"),
    # 2026-09-14 SC-CAL1: a joint calendar is open only when every leg is open (SOFR swaps: SIFMA AND New York).
    ("calendar_join_any_leg_open", "include/swaps/build/calendar.hpp",
     "      if (!is_business_day(std::string(view.joins[j]), d)) return false;",
     "      if (is_business_day(std::string(view.joins[j]), d)) return true;",
     ["sofr_calendar_repro_test.cpp"], "*", "a join requiring every leg open is unpinned (SC-CAL1)"),
    # 2026-09-14 O5 (owner decision): spot counts from the RAW value date, not from the next business day.
    ("spot_adjusts_value_date_first", "include/swaps/build/schedule.hpp",
     "  return advance_bd(cal_id, value_date, spot_lag);",
     "  return spot_lag == 0 ? advance_bd(cal_id, value_date, 0) : advance_bd(cal_id, roll(cal_id, value_date, 1), spot_lag);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the RAW spot rule on a non-business value date is unpinned (O5)"),
    # 2026-09-14 swap_spread matched_maturity: from the product's spot to the bond's maturity, rolled back from it.
    ("matched_swap_rolls_forward_from_spot", "include/swaps/derive/asset_swap.hpp",
     "    from_maturity.side = build::StubSide::Front;\n",
     "",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the matched swap's front stub (rolled back from the maturity) is unpinned"),
    ("matched_swap_rolls_from_adjusted_maturity", "include/swaps/derive/asset_swap.hpp",
     "    return {build::par_swap(r.value_date, sconv, r.bond.maturity, r.swap_curve, r.swap_curve, 0.0,",
     "    return {build::par_swap(r.value_date, sconv, maturity, r.swap_curve, r.swap_curve, 0.0,",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "a holiday / weekend maturity's roll day (the unadjusted day) is unpinned"),
    ("matched_swap_maturity_unadjusted", "include/swaps/derive/asset_swap.hpp",
     "    const build::Date maturity = build::adjust(sconv.calendar, r.bond.maturity, sconv.bdc);",
     "    const build::Date maturity = r.bond.maturity;",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the matched swap's reported termination being the rolled maturity is unpinned"),
    ("par_swap_rule_dropped_on_fixed_leg", "include/swaps/build/instruments.hpp",
     "  const auto periods = swap_periods_to(vd, conv.calendar, mat, freq, conv.bdc, conv.spot_lag, rule);",
     "  const auto periods = swap_periods_to(vd, conv.calendar, mat, freq, conv.bdc, conv.spot_lag);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "fixed_coupons forwarding its ScheduleRule is unpinned"),
    ("par_swap_rule_dropped_on_float_leg", "include/swaps/build/instruments.hpp",
     "  const auto periods = swap_periods_to(vd, conv.calendar, mat, freq_tok, conv.bdc, conv.spot_lag, rule);",
     "  const auto periods = swap_periods_to(vd, conv.calendar, mat, freq_tok, conv.bdc, conv.spot_lag);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "float_leg forwarding its ScheduleRule is unpinned"),
    ("matched_tenor_not_refused", "include/swaps/derive/asset_swap.hpp",
     "    if (!r.tenor.empty())\n",
     "    if (false)\n",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "a tenor on a matched_maturity request being refused is unpinned"),
    ("matched_bond_before_spot_not_refused", "include/swaps/derive/asset_swap.hpp",
     "    if (!(maturity > spot))\n",
     "    if (false)\n",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the named refusal of a bond maturing before the product's spot is unpinned"),
    ("matched_day_frequency_not_refused_by_name", "include/swaps/derive/asset_swap.hpp",
     "    if (!sconv.zero_coupon &&\n        (build::tok_step(sconv.fixed_freq_tok).days || build::tok_step(sconv.float_freq_tok).days))\n",
     "    if (false)\n",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the named refusal of a day-frequency product is unpinned"),
    ("matched_precheck_spot_lag_hardcoded_2", "include/swaps/derive/asset_swap.hpp",
     "    const build::Date spot = build::spot_date(r.value_date, sconv.calendar, sconv.spot_lag);",
     "    const build::Date spot = build::spot_date(r.value_date, sconv.calendar, 2);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the matched pre-check comparing against the product's own spot is unpinned"),
    ("swap_spot_lag_hardcoded_2", "include/swaps/build/schedule.hpp",
     "  return advance_bd(cal_id, value_date, spot_lag);",
     "  return advance_bd(cal_id, value_date, 2);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "a per-currency spot lag (GBP T+0, AUD T+1) is unpinned"),
    ("swap_spot_ignores_product_calendar", "include/swaps/build/schedule.hpp",
     "  return advance_bd(cal_id, value_date, spot_lag);",
     "  return advance_bd(\"USD-SOFR\", value_date, spot_lag);",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "the product calendar for spot is unpinned"),
    ("spot_zero_lag_does_not_roll", "include/swaps/build/calendar.hpp",
     "  if (n == 0) return roll(cal_id, d, 1);",
     "  if (n == 0) return d;",
     ["swap_spread_matched_maturity_repro_test.cpp"], "*", "a T+0 product valued on its own holiday starting on the holiday is unpinned"),
    # 2026-09-14 SC3 (owner decision): the base calibration's status is reported by scenario / grid / var.
    ("scenario_calibration_not_reported", "include/swaps/derive/scenario.hpp",
     "  out.calibration = sess.calibrate(",
     "  sess.calibrate(",
     ["derive_scenario_test.cpp"], "*", "scenario's calibration outcome is unpinned (SC3)"),
    ("grid_calibration_not_reported", "include/swaps/derive/scenario_grid.hpp",
     "  out.calibration = sess.calibrate(",
     "  sess.calibrate(",
     ["derive_scenario_grid_test.cpp"], "*", "scenario_grid's calibration outcome is unpinned (SC3)"),
    # 2026-09-14 SC2 (owner decision): FX moves are exact per currency pair.
    ("fx_pairs_orientation_flipped", "include/swaps/portfolio/fx_pairs.hpp",
     "    const int b = curves[static_cast<std::size_t>(p.mtm_reset_num)].currency;\n    const int q = curves[static_cast<std::size_t>(p.mtm_reset_den)].currency;",
     "    const int b = curves[static_cast<std::size_t>(p.mtm_reset_den)].currency;\n    const int q = curves[static_cast<std::size_t>(p.mtm_reset_num)].currency;",
     ["fx_pairs_test.cpp"], "*", "a position's pair orientation (num = base, den = quote) is unpinned (SC2)"),
    ("fx_pairs_swap_gets_a_slot", "include/swaps/portfolio/fx_pairs.hpp",
     "    if (p.kind != MultiCurveBook::Kind::Xccy) continue;\n",
     "",
     ["fx_pairs_test.cpp"], "*", "a position with no FX spot having no pair is unpinned (SC2)"),
    ("fx_pairs_inverse_pairs_merged", "include/swaps/portfolio/fx_pairs.hpp",
     "           !(s.base[static_cast<std::size_t>(slot)] == b && s.quote[static_cast<std::size_t>(slot)] == q))",
     "           !((s.base[static_cast<std::size_t>(slot)] == b && s.quote[static_cast<std::size_t>(slot)] == q) ||\n"
     "             (s.base[static_cast<std::size_t>(slot)] == q && s.quote[static_cast<std::size_t>(slot)] == b)))",
     ["fx_pairs_test.cpp"], "*", "EURUSD and USDEUR positions taking separate slots is unpinned (SC2)"),
    ("fx_spot_under_one_factor_for_all", "include/swaps/portfolio/fx_pairs.hpp",
     "fx_spot * slot_factor[static_cast<std::size_t>(slot)];",
     "fx_spot * slot_factor[0];",
     ["fx_pairs_test.cpp"], "*", "each position taking ITS pair's factor is unpinned -- the SC2 bug class"),
    ("fx_move_repeat_not_compounded", "include/swaps/derive/fx_move.hpp",
     "        eg_[e] *= (1.0 + b.rel);",
     "        eg_[e] = (1.0 + b.rel);",
     ["fx_pairs_test.cpp"], "*", "a repeated pair compounding in request order is unpinned (SC2)"),
    ("fx_move_inverse_repeat_multiplies", "include/swaps/derive/fx_move.hpp",
     "        eg_[e] /= (1.0 + b.rel);",
     "        eg_[e] *= (1.0 + b.rel);",
     ["fx_pairs_test.cpp"], "*", "a repeat named in the inverse orientation dividing is unpinned (SC2)"),
    ("fx_move_backward_edge_multiplies", "include/swaps/derive/fx_move.hpp",
     "val_[static_cast<std::size_t>(u)] / eg_[e];",
     "val_[static_cast<std::size_t>(u)] * eg_[e];",
     ["fx_pairs_test.cpp"], "*", "an inverse or cross path dividing by its edge is unpinned (SC2)"),
    ("fx_move_cycle_accepted", "include/swaps/derive/fx_move.hpp",
     "      if (ru == rv)\n",
     "      if (false)\n",
     ["fx_pairs_test.cpp"], "*", "refusing an over-determined bump triangle is unpinned (SC2)"),
    ("fx_move_undetermined_defaults", "include/swaps/derive/fx_move.hpp",
     "      if (!seen_[static_cast<std::size_t>(b)])\n",
     "      if (false)\n",
     ["fx_pairs_test.cpp"], "*", "refusing a pair the move does not determine is unpinned (SC2)"),
    ("fx_move_untouched_do_not_hold", "include/swaps/derive/fx_move.hpp",
     "      if (!touched_[static_cast<std::size_t>(c)]) add_edge(n_edges, c, rest);\n",
     "",
     ["fx_pairs_test.cpp"], "*", "unbumped currencies holding against each other is unpinned (SC2)"),
    ("fx_move_pivot_ignored", "include/swaps/derive/fx_move.hpp",
     "    if (pivot_ >= 0 && touched_[static_cast<std::size_t>(pivot_)]) add_edge(n_edges, pivot_, rest);\n",
     "",
     ["fx_pairs_test.cpp"], "*", "fx_pivot anchoring the unbumped currencies is unpinned (SC2)"),
    ("fx_move_codes_unchecked", "include/swaps/derive/fx_move.hpp",
     "        if (tag < 0 || tag >= n)\n",
     "        if (false)\n",
     ["fx_pairs_test.cpp"], "*", "refusing an xccy pair with no currency code is unpinned (SC2)"),
    ("book_fx_moves_codes_always_required", "include/swaps/derive/fx_move.hpp",
     "  if (out.slots.n_slots() == 0 || !any_bump) return out;\n",
     "  if (out.slots.n_slots() == 0) return out;\n",
     ["fx_pairs_test.cpp"], "*", "needing no codes without an FX bump is unpinned (SC2)"),
    ("compiled_fx_row_factor_ignored", "include/swaps/portfolio/compiled_multi.hpp",
     "      if (e.row >= 0) notional_[static_cast<Eigen::Index>(e.row)] = e.notional * fx;",
     "      if (e.row >= 0) notional_[static_cast<Eigen::Index>(e.row)] = e.notional * e.fx_spot;",
     ["fx_pairs_test.cpp"], "*", "a compiled xccy row taking its pair's factor is unpinned (SC2)"),
    ("compiled_fx_fallback_factor_ignored", "include/swaps/portfolio/compiled_multi.hpp",
     "      else fallback_.positions[static_cast<std::size_t>(e.fallback)].fx_spot = fx;",
     "      else {}",
     ["fx_pairs_test.cpp"], "*", "a fallback xccy position taking its pair's factor is unpinned (SC2)"),
    ("scenario_fx_move_unused", "include/swaps/derive/scenario.hpp",
     "        portfolio::set_xccy_fx(*moved, *r.book, fx.slots, f);\n",
     "",
     ["derive_scenario_test.cpp"], "*", "a scenario's FX row valuing the per-pair moved book is unpinned (SC2)"),
    ("grid_fx_factors_unused", "include/swaps/derive/scenario_grid.hpp",
     "      if (!f.empty()) book.set_fx_factors(fx.slots, f);\n",
     "",
     ["derive_scenario_grid_test.cpp"], "*", "an fx cell repricing at its pair's factors is unpinned (SC2)"),
    ("var_fx_factors_unused", "include/swaps/derive/var.hpp",
     "      book.set_fx_factors(fx.slots, f);\n      fx_moved = true;\n",
     "      fx_moved = true;\n",
     ["derive_var_test.cpp"], "*", "a var FX move repricing at its pair's factors is unpinned (SC2)"),
    ("var_fx_not_restored", "include/swaps/derive/var.hpp",
     "      book.set_fx_factors(fx.slots, ones);\n",
     "",
     ["derive_var_test.cpp"], "*", "a move after an FX move repricing at the base spots is unpinned (SC2)"),
    ("var_moves_not_resolved", "include/swaps/derive/var.hpp",
     "  for (const ScenarioMove& m : r.scenarios) moves.push_back(resolve_scenario_move(m, r.bundle.curves, \"var\"));\n",
     "  for (const ScenarioMove& m : r.scenarios) moves.push_back(ResolvedMove{std::vector<double>(r.bundle.curves.size(), 0.0)});\n",
     ["derive_var_test.cpp"], "*", "resolving every move's rates (and its roles, before calibrating) is unpinned (E7 var)"),
    ("var_pnl_not_against_base", "include/swaps/derive/var.hpp",
     "moves[k].curve_delta)) - out.base_npv);",
     "moves[k].curve_delta)));",
     ["derive_var_test.cpp"], "*", "the P&L against the base is unpinned (E7 var)"),
    # E7 var lift: derive/var.hpp, the library behind the var verb.
    ("var_base_at_the_seed", "include/swaps/derive/var.hpp",
     "  const Eigen::VectorXd x_base = sess.x();  // the anchor every move forks from; never mutated",
     "  const Eigen::VectorXd x_base = calibration::seed_or_flat(P, r.x0, \"var\");",
     ["derive_var_test.cpp"], "*", "revaluing at the calibrated state (not the seed) is unpinned (E7 var)"),
    ("var_calibration_not_reported", "include/swaps/derive/var.hpp",
     "  out.calibration = sess.calibrate(",
     "  sess.calibrate(",
     ["derive_var_test.cpp"], "*", "the base's calibration outcome is unpinned (E7 var / SC3)"),
    ("var_both_modes_accepted", "include/swaps/derive/var.hpp",
     "  if (r.pnl && r.reval)\n    throw",
     "  if (false)\n    throw",
     ["derive_var_test.cpp"], "*", "refusing pnl and a reval request together is unpinned (E7 var)"),
    ("var_quantiles_checked_after_calibrating", "include/swaps/derive/var.hpp",
     "  check_var_quantiles(r.quantiles);  // before any calibration, as the verb did\n",
     "",
     ["derive_var_test.cpp"], "*", "checking the quantiles before calibrating is unpinned (E7 var)"),
    ("var_quantile_not_interpolated", "include/swaps/derive/var.hpp",
     "  return s[static_cast<std::size_t>(i)] * (1.0 - frac) + s[static_cast<std::size_t>(i + 1)] * frac;",
     "  return s[static_cast<std::size_t>(i)];",
     ["derive_var_test.cpp"], "*", "the type-7 interpolation is unpinned (E7 var)"),
    # NOT a mutation: removing `if (i >= n - 1) return s[n - 1];` reads s[n] * 0.0 (frac is exactly 0 at p = 1) -- UB
    # that returns the right number unless the byte past the vector is inf/NaN. Only a sanitizer run could catch it.
    ("var_es_tail_unclamped", "include/swaps/derive/var.hpp",
     "  if (m < 1) m = 1;",
     "",
     ["derive_var_test.cpp"], "*", "ES over at least one point is unpinned (E7 var)"),
    ("var_stdev_population", "include/swaps/derive/var.hpp",
     "std::sqrt(var_acc / (out.n - 1))",
     "std::sqrt(var_acc / out.n)",
     ["derive_var_test.cpp"], "*", "the sample (n - 1) standard deviation is unpinned (E7 var)"),
    ("var_loss_not_negated", "include/swaps/derive/var.hpp",
     "  out.var = -out.var_pnl;",
     "  out.var = out.var_pnl;",
     ["derive_var_test.cpp"], "*", "var as a LOSS (-P&L) is unpinned (E7 var)"),
    # E7 stage 6.7: calibration::pnl_report, the library behind the pnl verb.
    ("pnl_report_x0_override_ignored", "include/swaps/calibration/pnl_explain.hpp",
     "  const Eigen::VectorXd x0 = r.x0 ? *r.x0 : s0.x();",
     "  const Eigen::VectorXd x0 = s0.x();",
     ["pnl_report_test.cpp"], "*", "the x0 decomposition override is unpinned (E7 6.7)"),
    ("pnl_report_x1_override_ignored", "include/swaps/calibration/pnl_explain.hpp",
     "  if (r.x1) x1 = *r.x1;",
     "",
     ["pnl_report_test.cpp"], "*", "the x1 decomposition override is unpinned (E7 6.7)"),
    ("pnl_report_dq_not_differenced", "include/swaps/calibration/pnl_explain.hpp",
     "    out.dq = s1.problem().market() - P0.market();",
     "    out.dq = s1.problem().market();",
     ["pnl_report_test.cpp"], "*", "dq = q1 - q0 is unpinned (E7 6.7)"),
    ("pnl_report_x1_length_unchecked", "include/swaps/calibration/pnl_explain.hpp",
     "  if (r.x1 && r.x1->size() != P0.n_knots())",
     "  if (false)",
     ["pnl_report_test.cpp"], "*", "the x1 length check is unpinned (E7 6.7)"),
    # 2026-09-14 PN1: bundle1 must be bundle0 re-quoted (structure_equal), not merely the same counts.
    ("pnl_report_bundle1_counts_only", "include/swaps/calibration/pnl_explain.hpp",
     "  if (r.bundle1 && !structure_equal(r.bundle0, *r.bundle1))",
     "  if (r.bundle1 && (r.bundle1->n_knots() != r.bundle0.n_knots() || r.bundle1->n_residuals() != r.bundle0.n_residuals()))",
     ["pnl_report_test.cpp"], "*", "bundle1's structure check is back to counts only (PN1)"),
    # 2026-09-14 SW1: a fixing day before curve time 0 is past whatever the evaluation date says.
    ("fixings_past_by_curve_time", "include/swaps/pricing/fixings.hpp",
     "        d.t_start < 0.0 || d.fixing_date < ctx.evaluation_date ||",
     "        d.fixing_date < ctx.evaluation_date ||",
     ["fixings_test.cpp"], "*", "a seasoned day with no evaluation date being refused is unpinned"),
    # 2026-09-14 FS1: a front-stub roll date that adjusts onto spot is not a boundary.
    ("schedule_front_roll_onto_spot_kept", "include/swaps/build/schedule.hpp",
     "    if (a < end && a > bounds.back()) bounds.push_back(a);",
     "    if (a < end) bounds.push_back(a);",
     ["schedule_front_stub_onto_spot_repro_test.cpp"], "*", "dropping a roll date that adjusts onto spot is unpinned (FS1)"),
    ("schedule_isda_interior_on_end", "include/swaps/build/schedule.hpp",
     "    if (a < end && a > bounds.back()) bounds.push_back(a);",
     "    if (a > bounds.back()) bounds.push_back(a);",
     ["trade_weekend_maturity_repro_test.cpp"], "*", "an ISDA boundary rolling onto the end is unpinned"),
    # 2026-09-14 weekend-maturity fix: a schedule ends on the termination date rolled by the convention.
    ("schedule_raw_termination", "include/swaps/build/schedule.hpp",
     "  const Date end = adjust(cal_id, maturity_date, bdc);",
     "  const Date end = maturity_date;",
     ["trade_weekend_maturity_repro_test.cpp"], "*", "rolling a booked non-business-day maturity is unpinned"),
    # 2026-09-14 spread-curve fix: a parallel move / PV01 moves every curve's forward once.
    ("parallel_direction_spread_knots_move", "include/swaps/pricing/curve_spec.hpp",
     "    if (c.base < 0) d.segment(offset, c.n_interp_knots()).setOnes();",
     "    d.segment(offset, c.n_interp_knots()).setOnes();",
     ["bundle_state_test.cpp"], "*", "the parallel direction skipping spread knots is unpinned"),
    ("parallel_direction_turns_move", "include/swaps/pricing/curve_spec.hpp",
     "d.segment(offset, c.n_interp_knots()).setOnes();",
     "d.segment(offset, c.n_knots()).setOnes();",
     ["bundle_state_test.cpp"], "*", "the parallel direction skipping turn deltas is unpinned"),
    ("compiled_pv01_all_ones_direction", "include/swaps/portfolio/compiled_multi.hpp",
     "rowsum_ = cs_.W() * dir_;",
     "rowsum_ = cs_.W().rowwise().sum();",
     ["bundle_state_test.cpp"], "*", "the compiled PV01 direction is unpinned"),
    ("compiled_pv01_fallback_all_ones", "include/swaps/portfolio/compiled_multi.hpp",
     "swaps::ad::seed_directional(x, dir_)",
     "swaps::ad::seed_directional(x)",
     ["bundle_state_test.cpp"], "*", "the fallback PV01 direction is unpinned"),
    ("parallel_shock_reaches_spread_curves", "include/swaps/derive/scenario.hpp",
     "    if (curves[c].base < 0) curve_delta[c] += bp / 1e4;",
     "    curve_delta[c] += bp / 1e4;",
     ["scenario_spread_repro_test.cpp"], "*", "a parallel skipping spread curves is unpinned (scenario, grid, var)"),
    # 2026-09-14 SC1 (owner decision): shocks ADD, one definition for scenario and scenario_grid.
    ("curve_shock_overrides", "include/swaps/derive/scenario.hpp",
     "  curve_delta[static_cast<std::size_t>(role)] += bp / 1e4;",
     "  curve_delta[static_cast<std::size_t>(role)] = bp / 1e4;",
     ["derive_scenario_test.cpp", "derive_scenario_grid_test.cpp"], "*", "a per-curve shock adding onto the parallel is unpinned (SC1)"),
    ("scenario_curve_key_ignored", "include/swaps/derive/scenario.hpp",
     "    add_curve_shock(role, bp, r.curve_delta);",
     "",
     ["derive_scenario_test.cpp"], "*", "a move's shift_curve keys are unpinned (SC1)"),
    ("grid_shift_curve_axis_ignored", "include/swaps/derive/scenario_grid.hpp",
     "      add_curve_shock(*ax.role, value, curve_delta);",
     "",
     ["derive_scenario_grid_test.cpp"], "*", "a shift_curve axis is unpinned (SC1)"),
    ("market_scenario_key_overrides", "include/swaps/market/scenario.hpp",
     "    if (it != curve_bp_.end()) d += it->second / 1e4;",
     "    if (it != curve_bp_.end()) d = it->second / 1e4;",
     ["market_scenario_test.cpp"], "*", "market::Scenario's key adding onto the global parallel is unpinned (SC1)"),
    # E7 stage 5.3: derive/scenario_grid.hpp, the library behind the scenario_grid verb.
    ("grid_second_axis_ignored", "include/swaps/derive/scenario_grid.hpp",
     "      if (out.axes.size() == 2)\n        add_axis_shock(",
     "      if (false)\n        add_axis_shock(",
     ["derive_scenario_grid_test.cpp"], "*", "the second axis is unpinned (E7 5.3)"),
    ("grid_role_optional", "include/swaps/derive/scenario_grid.hpp",
     "    if (!ax.role) throw std::invalid_argument(\"scenario_grid: a shift_curve axis needs a 'role'\");",
     "",
     ["derive_scenario_grid_test.cpp"], "*", "a shift_curve axis's required role is unpinned (E7 5.3)"),
    ("grid_axes_checked_after_calibrating", "include/swaps/derive/scenario_grid.hpp",
     "  for (const ShockAxis& ax : r.axes) check_shock_axis(ax, r.bundle.n_curves());\n",
     "",
     ["derive_scenario_grid_test.cpp"], "*", "checking the axes before calibrating is unpinned (E7 5.3)"),
    ("grid_pnl_not_against_base", "include/swaps/derive/scenario_grid.hpp",
     "      pnl_row.push_back(npv - out.base_npv);",
     "      pnl_row.push_back(npv);",
     ["derive_scenario_grid_test.cpp"], "*", "the P&L against the base is unpinned (E7 5.3)"),
    # E7 stage 5.2: derive/scenario.hpp, the library behind the scenario verb.
    ("scenario_parallel_dropped", "include/swaps/derive/scenario.hpp",
     "  if (m.parallel_bp) add_parallel_shock(curves, *m.parallel_bp, r.curve_delta);",
     "",
     ["derive_scenario_test.cpp"], "*", "a move's parallel shift is unpinned (E7 5.2)"),
    ("scenario_role_range_unchecked", "include/swaps/derive/scenario.hpp",
     "    if (role < 0 || role >= n_curves)",
     "    if (false)",
     ["derive_scenario_test.cpp"], "*", "the shift_curve role range is unpinned (E7 5.2)"),
    ("scenario_npv_delta_not_against_base", "include/swaps/derive/scenario.hpp",
     "      row.npv_delta = row.npv - out.base_npv;",
     "      row.npv_delta = row.npv;",
     ["derive_scenario_test.cpp"], "*", "npv_delta against the base is unpinned (E7 5.2)"),
    # E7 stage 5.1: calibration/bundle_state.hpp.
    ("bundle_state_shift_reaches_turns", "include/swaps/calibration/bundle_state.hpp",
     "x.segment(p.offset(c), p.curves[c].n_interp_knots()).array() += d;",
     "x.segment(p.offset(c), p.curves[c].n_knots()).array() += d;",
     ["bundle_state_test.cpp"], "*", "a shift leaving turn jumps alone is unpinned (E7 5.1)"),
    ("bundle_state_zero_at_t0_divides", "include/swaps/calibration/bundle_state.hpp",
     "t > 1e-12 ? C[c]->integral(t) / t : C[c]->forward(0.0)",
     "C[c]->integral(t) / t",
     ["bundle_state_test.cpp"], "*", "the t = 0 zero rate is unpinned (E7 5.1)"),
    # E7 stage 4.2: the conventions Registry's row rules and all-or-nothing batches.
    ("conventions_stub_discount_enum_dropped", "include/swaps/conventions_db.hpp",
     '  r.one_of(b.stub_discount, "stub_discount", kSchemaStubDiscounts);',
     "",
     ["conventions_rules_test.cpp"], "*", "a bond row's stub_discount enum is unpinned (E7 4.2)"),
    ("conventions_weekend_mask_unbounded", "include/swaps/conventions_db.hpp",
     "  if (c.weekend_mask & ~0x7F) r.fail(",
     "  if (c.weekend_mask & ~0xFF) r.fail(",
     ["conventions_rules_test.cpp"], "*", "the weekend-day range is unpinned (E7 4.2)"),
    ("conventions_recovery_bound_loosened", "include/swaps/conventions_db.hpp",
     "c.recovery_default <= 1.0))",
     "c.recovery_default <= 2.0))",
     ["conventions_rules_test.cpp"], "*", "the recovery bound is unpinned (E7 4.2)"),
    ("conventions_zero_coupon_frequency_allowed", "include/swaps/conventions_db.hpp",
     "  if (!p.fixed.frequency.empty() || !p.floating.frequency.empty())",
     "  if (false)",
     ["conventions_rules_test.cpp"], "*", "a zero-coupon leg frequency is unpinned (E7 4.2)"),
    ("conventions_full_leg_index_optional", "include/swaps/conventions_db.hpp",
     '    r.need(l.index, std::string(name) + ".index");',
     "",
     ["conventions_rules_test.cpp"], "*", "a coupon leg's index is unpinned (E7 4.2)"),
    ("conventions_tenor_digits_unchecked", "include/swaps/conventions_db.hpp",
     "    for (std::size_t k = 0; ok && k + 1 < v.size(); ++k) ok = v[k] >= '0' && v[k] <= '9';",
     "",
     ["conventions_rules_test.cpp"], "*", "a tenor's digit count is unpinned (E7 4.2)"),
    ("conventions_iso_date_unchecked", "include/swaps/conventions_db.hpp",
     """    if (!ok) fail("'" + std::string(field) + "' must be YYYY-MM-DD, got '" + std::string(v) + "'");""",
     "",
     ["conventions_rules_test.cpp"], "*", "the YYYY-MM-DD shape is unpinned (E7 4.2)"),
    ("conventions_apply_skips_validation", "include/swaps/conventions_db.hpp",
     "    validate(b);\n    std::unique_lock lk(mu_);",
     "    std::unique_lock lk(mu_);",
     ["conventions_rules_test.cpp"], "*", "apply committing unchecked rows is unpinned (E7 4.2)"),
    ("conventions_fixing_source_index_unchecked", "include/swaps/conventions_db.hpp",
     "      if (!index_known_after(b, s.id))",
     "      if (false)",
     ["conventions_rules_test.cpp"], "*", "a fixing source's index check is unpinned (E7 4.2)"),
    ("conventions_clear_first_keeps_old_indices", "include/swaps/conventions_db.hpp",
     "    if (!b.clear_first)\n      for (const auto& i : indices_)",
     "    if (true)\n      for (const auto& i : indices_)",
     ["conventions_rules_test.cpp"], "*", "a clear in the same batch removing the index is unpinned (E7 4.2)"),
    ("conventions_clear_first_ignored", "include/swaps/conventions_db.hpp",
     "    if (b.clear_first) clear_rows();",
     "",
     ["conventions_rules_test.cpp"], "*", "clear_first is unpinned (E7 4.2)"),
    ("conventions_batch_bumps_generation_twice", "include/swaps/conventions_db.hpp",
     "    gen_.fetch_add(1, std::memory_order_release);\n    return after;",
     "    gen_.fetch_add(2, std::memory_order_release);\n    return after;",
     ["conventions_rules_test.cpp"], "*", "one batch = one generation is unpinned (E7 4.2)"),
    ("conventions_listing_overlay_size_dropped", "include/swaps/conventions_db.hpp",
     "    out.overlay_size = overlay_n_.load(std::memory_order_acquire);",
     "",
     ["conventions_rules_test.cpp"], "*", "the listing snapshot's overlay size is unpinned (E7 4.2)"),
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
     "  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, m.maturity));",
     "  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, r.bond.maturity));",
     ["derive_rv_test.cpp", "swap_spread_matched_maturity_repro_test.cpp"], "*",
     "the swap-spread anchor default (the swap's rolled termination) is unpinned (E7 3.4)"),
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


def build_codegen_flags():
    """The optimisation / architecture flags the build compiled libswaps_api with (build/compile_commands.json). An object
    linked into that archive must agree on them: Eigen's heap alignment follows -march, and a mismatch corrupts the heap."""
    try:
        for e in json.load(open(os.path.join(ROOT, "build/compile_commands.json"))):
            if e["file"].endswith("api/bundle_api.cpp"):
                toks = e["arguments"] if "arguments" in e else shlex.split(e["command"])
                return [t for t in toks if t.startswith(("-march=", "-mcpu=", "-O"))]
    except (OSError, ValueError, KeyError):
        pass
    return None


def compile_and_run(name, header, old, new, tus, flt, keep_dir, jobs_note=""):
    """Returns (name, caught: bool|None, detail)."""
    work = os.path.join(keep_dir, name)
    # the mutant lives in a scratch tree that mirrors the repo (include/..., tests/... or bench/fixtures/...); every
    # mirrored root is searched BEFORE the real one, so the TU sees the mutated header and the pristine rest. A test
    # must reach a header THROUGH these -I roots (never a relative "../" include, which bypasses the mirror: FLK1).
    mut = os.path.join(work, header)
    os.makedirs(os.path.dirname(mut), exist_ok=True)
    src = open(os.path.join(ROOT, header)).read()
    n = src.count(old)
    if n == 0:
        return name, None, f"anchor not found in {header} (the code moved; update the mutation)"
    open(mut, "w").write(src.replace(old, new))
    inc = os.path.join(work, "include")
    tst = os.path.join(work, "tests")
    fix = os.path.join(work, "bench", "fixtures")
    os.makedirs(inc, exist_ok=True); os.makedirs(tst, exist_ok=True); os.makedirs(fix, exist_ok=True)
    incs = ["-I", inc, "-I", tst, "-I", fix, "-I", os.path.join(ROOT, "include"), "-I", os.path.join(ROOT, "third_party/eigen"),
            "-I", os.path.join(ROOT, "third_party/boost"), "-I", os.path.join(ROOT, "build/generated"),
            "-I", os.path.join(ROOT, "tests"), "-I", os.path.join(ROOT, "bench/fixtures"), "-I", os.path.join(ROOT, GTEST_INC)]
    flags, link = FLAGS, []
    lib_srcs = [t for t in tus if t.startswith("api/")]
    if lib_srcs:  # the session layer: rebuild the named api sources against the mutant, link them ahead of the archive
        archive = os.path.join(ROOT, "build/api/libswaps_api.a")
        codegen = build_codegen_flags()
        if not os.path.exists(archive) or codegen is None:
            return name, None, "build/api/libswaps_api.a or build/compile_commands.json missing (build the engine first)"
        flags = FLAGS + codegen
        incs = incs + ["-I", os.path.join(ROOT, "api")]  # an api source's quoted includes, from its mirrored copy
        for s in lib_srcs:
            obj = os.path.join(work, os.path.basename(s).replace(".cpp", ".o"))
            cc = subprocess.run(CXX + flags + incs + ["-c", mut if s == header else os.path.join(ROOT, s), "-o", obj],
                                capture_output=True, text=True)
            if cc.returncode != 0:
                return name, None, f"{s} did not compile against the mutant:\n" + cc.stderr[-800:]
            link.append(obj)
        link.append(archive)
    for tu in [t for t in tus if not t.startswith("api/")]:
        exe = os.path.join(work, tu.replace(".cpp", ""))
        cmd = CXX + flags + incs + [os.path.join(ROOT, "tests", tu)] + link + [os.path.join(ROOT, l) for l in GTEST_LIBS] + ["-o", exe]
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
