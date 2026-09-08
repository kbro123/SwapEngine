# baselines/ — perf policy and the 2026-09-08 like-for-like re-baseline

Policy: PRINCIPLES.md P9/P10. `baselines.json` = per-fingerprint self-baselines (min of 5 reps, quiesced);
`targets.json` = absolute desk-scale targets (ratchet down only). QuantLib is an informational reference.
`tools/check_perf.py` is the gate; `tools/nightly.sh` writes `NIGHTLY.md`.

## Before / after: the toolchain-mismatch correction (E1.1 + E1.7)

Until 2026-09-08 the QuantLib reference was built 2026-07-18 with Apple clang 16 and `-march=native`
(AVX-512 codegen) while the engine was built with clang 21 and `-march=x86-64-v3` (AVX2); the fingerprint did
not include the QuantLib build, so the 2026-08-29 / 2026-09-04 baselines silently paired them. QuantLib was
rebuilt with the engine's exact toolchain (probed from `cmake/DetectISA.cmake`) and re-measured:

| metric | QuantLib before (clang16/AVX-512, mean-of-3) | QuantLib after (clang21/AVX2, min-of-5) | ours before | ours after | speedup before | speedup after |
|---|---|---|---|---|---|---|
| curve_build | 15.49 ms | 15.41 ms | 524.3 us | 504.3 us | 29.6x | 30.6x |
| risk_full_jacobian | 16.76 ms | 16.18 ms | 92.9 us | 109.0 us | 180.4x | 148.3x |
| portfolio_analytics | 21.64 ms | 21.27 ms | 84.0 us | 80.5 us | 257.7x | 264.2x |
| warm_recalibration | 0.98 ms | 0.88 ms | 18.6 us | 17.6 us | 52.8x | 50.1x |
| bond_sweep | 2579.78 ms | 2573.31 ms | 3832.9 us | 3461.2 us | 673.1x | 743.5x |
| bond_book | 30.42 ms | 27.58 ms | 426.9 us | 405.6 us | 71.3x | 68.0x |

Reading: QuantLib's time is algorithm-dominated (N re-bootstraps, per-object pricing), so the compiler/ISA
mismatch was worth low single-digit percent on the reference side — the old ratios were inflated/deflated by
a few percent, not by an order of magnitude. `risk_full_jacobian` ours 92.9 → 109 us is the deliberate
rank-safe COD change from the 2026-09-07 bug fixes (a correctness cost), not the toolchain. `warm_recalibration`
is now LIKE-FOR-LIKE (both sides ±0.3 bp on every quote; the old bench moved QuantLib ±10 bp on swaps and
±0.01 bp on futures against a fixed 0.3 bp on ours); the honest large-move pair `warm_recal_10bp` is 22x.
Method change (mean-of-3 → min-of-5) explains ~3-4 % of the "ours" improvement on unchanged code.

Fingerprint history: `52e94be82bc4` (Kaby Lake laptop, Jul 10) → `a8c9a844826e` (this Xeon, clang 16, AVX2
switch Jul 29) → `86d5211c2c03` (clang 21 engine vs clang 16 QuantLib, Aug 29 – Sep 8; mismatched) →
`966685f93279` (like-for-like, Sep 8).
