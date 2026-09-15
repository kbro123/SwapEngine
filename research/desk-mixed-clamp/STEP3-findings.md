# Step 3 findings: desk_mixed (probes on ac7217c / 009af08; break-even pinned 64)

Probe files: `probe_dm_stall.txt` and `probe_dm_xccy.txt`.

## 1. The 47 bp knot swing is a fixture artefact, not a desk_mixed defect

- Knot 54 (curve 3 = EUR-in-USD xccy, its 30Y knot) moves 4.69e-3 per 0.1 bp tick on desk as well as desk_mixed. The top-6 knot moves are identical on both rungs.
- Bumping only group 3 (FX forwards + xccy MtM basis) reproduces it: 4.72e-3. Bumping only SOFR moves knot 54 by 1.25e-4; bumping only ESTR moves it by 1.55e-4.
- The rows driving it are r42–r47, the FX forwards and short xccy rows. J⁺ entries there are about ±80, with alternating signs.
- Cause: `q_small` bumps every row by `1e-5 × sin(pattern)`, with FX forwards bumped relatively. Neighbouring FX forward tenors (1W…1Y) move in contradictory directions. The only way the xccy curve can fit that is to swing its long end.
- On desk this is harmless: frozen steps contract with spectral radius 0.002, 3 steps, no refresh.
- This is the same class of problem C2 fixed for `q_big`: the "0.1 bp tick" is not a realistic market move on FX rows. A fixture change would re-baseline the stream-tick metrics, so it needs an owner decision.

## 2. Why desk_mixed refreshes every tick

- It is not the stall detector.
  - With `adaptive_stall` OFF, every tick still takes 1 refresh (10 steps), via the `max_frozen` cap.
  - With `max_steps` = 400 it also refreshes once per tick, so it is not the step cap.
- The frozen iteration does not contract.
  - ‖J(x_small) − J(x0)‖/‖J(x0)‖ is 4.4e-3 on desk_mixed against 1.6e-4 on desk (28×).
  - The spectral radius of I − J0⁺·J1 is 1.365 on desk_mixed against 0.002 on desk.
  - A frozen-J Newton iteration with radius > 1 cannot converge without a refresh. Its early steps look like they contract (rho 0.577), then it hits the cap.
- The source is the MonotoneCubic SOFR long end. Bumping only SOFR still gives 7.5 steps and 1 refresh per tick; every other group alone gives 2–3 steps and no refresh.
  - The Jacobian's value-dependence (the Hyman filter slopes) moves with the state.
  - The swing that item 1 forces amplifies it.
- The kink half-step (FLK2) plus the adaptive stall is only how the q0 → q_small direction ends. The refresh is needed either way. The false-stall fix alone would move the refresh from step 3 to step 8, not remove it, and would cost more steps.

## 3. Walk failures (hard-row ±1 bp and ±3 bp sequences)

- Ticks 7–9 at ±1 bp are DIVERGED after 8–10 steps, with 12–15 rescales, 2–3 pins, 0 refreshes and 0 releases.
- Tick 25 is NON-FINITE after 4 refreshes and 22 rescales.
- At ±3 bp, tick 21 is DIVERGED with 1 refresh and 16 rescales.
- Pattern: pins accumulate on a frozen operator that no longer contracts (item 2). The band walk takes breakpoint steps on a J whose MonotoneCubic rows are stale.

## Options for the owner

- **(a)** Fix the fixture: make `q_small` realistic on FX/xccy rows (as C2 did for `q_big`). This removes the artificial swing; desk_mixed's refresh would likely remain.
- **(b)** A secant / Broyden rank-one update of M per frozen step on hybrid (AAD-block) rows. It restores contraction without a full Jacobian, as a new mechanism, and interacts with the band rescale's G.
- **(c)** Refresh only the AAD-block rows: the MonotoneCubic rows are a handful. A partial Jacobian followed by a rank-k update of the factorisation, instead of a full refresh. Cheaper than today's 1.2 ms refresh.
- **(d)** Research tie-in: the recorded graph's clamp mask flips would say exactly when the MonotoneCubic rows' J changed pattern. The streamer would refresh or rescale those rows only then, making it an active set, as in the research Part 3.
- **(e)** Walk divergence: guard the breakpoint walk with a contraction check on hybrid rows, refreshing before pinning when the frozen map is not contracting. Needs the (b)/(c) machinery or a cheap radius estimate from consecutive |dx|.

## 4. q_small variants (`probe_qsmall.txt`, owner chose "fixture first")

| Variant | fx_xccy | desk | desk_mixed |
|---|---|---|---|
| V0 today (FX forwards moved relatively) | 3.5 steps, \|dx\| 1.05e-2 | 3 steps, \|dx\| 4.69e-3 @54 | 7.5 steps, 1 refresh, radius 1.365, 4530 / 3380 allocs |
| V1 FX forwards unchanged | 2 steps, 3.9e-4 | 2 steps, 2.75e-4 @11 | 7.5 steps, 1 (be64) / 1.5 (measured) refresh, radius 1.365, 4530 / 3920 allocs |
| V2 V1 + xccy basis ×0.1 | 2 steps | 2 steps | same as V1 |
| V3 FX moved consistently with a 0.1 bp rate move | 2 steps | 2 steps, 4.6e-4 @54 | same as V1 |
| V4 parallel 0.1 bp (no noise) | 2 steps | 2 steps | **2 steps, 0 refresh, radius 0.000, 920 allocs** |

**Conclusions**
- The FX-forward relative bump caused the swing (items 1–2 above). V1 fixes desk and fx_xccy: 3 or 3.5 steps drop to 2.
- desk_mixed still refreshes on V1–V3, with the frozen-map radius still 1.365 after a knot move of only 2e-4.
- Its Jacobian is DISCONTINUOUS between x(q0) and x(q_small) whenever the tick carries per-row noise. A MonotoneCubic (Hyman) clamp flips between the two states.
- V4, a smooth parallel move, flips nothing: no refresh, and 920 allocs (the hybrid-eval floor).
- **Chosen:** V1, which keeps the realistic per-row noise. V4 would hide the kink behaviour and flatter the benchmarks.
- The desk_mixed fix candidates now target the clamp flip specifically:
  - refresh only the MonotoneCubic rows when a clamp flips;
  - treat a clamp flip as an active-set event, like a band edge (research Part 3, mask flips).

## 5. Clamp-flip trace (research `guardtrace2`, `aadjit/coarse/run_part3b.txt`, counts only)

- **Where the value-dependence lives.** desk_mixed's only value-dependent region is SOFR (curve 0) knots 6..11: MonotoneCubic over t = 10.02..30.03. The tape carries 94 guards at every state.
- **Degenerate-but-harmless margins.**
  - At the committed x(q0), many Hyman run predicates sit at margin 1e-20..1e-23. The fixture's `x_true` long end is exactly linear (0.030 + 0.0004 i), so consecutive secants are equal and their product (S_i − S_i−1)(S_i+1 − S_i) ≈ 0.
  - Both V1 AND V4 flip these, but V4 has no refresh and frozen radius 0.000. At equal secants both arms give the same value and slope, so J is continuous through these flips.
- **The harmful flips.**
  - Only V1 flips node 5 (t = 25Y), `clamp M < |m|` (margin −1.58e-4 → +6.0e-5), and node 6 (t = 30Y), the end condition `m·S > 0` / `|3S| < |m|`.
  - Within the V1 tick they toggle on every Newton step: step 1 clamp false→true, step 2 true→false, step 3 false→true. That is the FLK2 2-cycle.
  - The half step lands on the kink, the stall detector reads rho = 1, and the tick refreshes.
- **Read.** The refresh on every noisy tick is a genuine Newton 2-cycle across the 25Y clamp and 30Y end-slope kinks of the MonotoneCubic long end. Per-row noise pushes the solution across them; a parallel move does not.
- **Cost arithmetic for the options.** On desk_mixed, a refresh is J 777 µs (mostly the AAD rows) + factor ~214 µs + model_rates ~198 µs (P8).
  - A partial refresh of only the AAD rows saves at most the compiled rows' J (~200 µs) plus a rank-k factor update instead of COD (~150 µs): roughly 1.2 → 0.8 ms per refresh, still one per tick.
  - Avoiding the refresh needs the 2-cycle not to happen: a clamp active set (hold the pattern, verify at convergence, like band pins), or a secant update across the kink.
