# baselines/ — perf policy and the 2026-09-08 like-for-like re-baseline

Policy: PRINCIPLES.md P9/P10. `baselines.json` = per-fingerprint self-baselines (min of 5 reps, quiesced);
`targets.json` = absolute desk-scale targets (ratchet down only). QuantLib is an informational reference.
`tools/check_perf.py` is the gate; `tools/nightly.sh` writes `NIGHTLY.md`.

`soak_pins.json` (2026-09-22) = the streaming soak's platform-sensitive pins, keyed by the NUMERIC fingerprint
`OS|compiler|arch flags|ISA` (no CPU model: one instruction stream, one set of doubles). A value-dependent rung's
failed-tick count on a seeded walk is deterministic per key and differs across keys (0 vs 2 on mixed_scheme, Apple
clang vs GCC/glibc), so it is a per-key pin like a timing, not a universal invariant. The universal rows (linear
rungs; every rung on the smoothed factor walk) stay hard-coded in `tests/streaming_soak_test.cpp`. No entry for the
running key = report-only; the test (and the CI "streaming soak report" step) prints the stanza to paste. Ratchet
down only, per key.

## Before / after: the toolchain-mismatch correction (E1.1 + E1.7)

Until 2026-09-08 the QuantLib reference was built 2026-07-18 with Apple clang 16 and `-march=native`
(AVX-512 codegen) while the engine was built with clang 21 and `-march=x86-64-v3` (AVX2); the fingerprint did
not include the QuantLib build, so the 2026-08-29 / 2026-09-04 baselines silently paired them. QuantLib was
rebuilt with the engine's exact toolchain (probed from `cmake/DetectISA.cmake`) and re-measured:

| metric | QuantLib before (clang16/AVX-512, mean-of-3) | QuantLib after (clang21/AVX2, min-of-5) | ours before | ours after | speedup before | speedup after |
|---|---|---|---|---|---|---|
| sofr_23k_square_cold_calibrate | 15.49 ms | 15.41 ms | 524.3 us | 504.3 us | 29.6x | 30.6x |
| sofr_23k_risk_ladder_23q_book9 | 16.76 ms | 16.18 ms | 92.9 us | 109.0 us | 180.4x | 148.3x |
| sofr_23k_book1000_ois_reprice | 21.64 ms | 21.27 ms | 84.0 us | 80.5 us | 257.7x | 264.2x |
| sofr_23k_warm_recal_0p3bp | 0.98 ms | 0.88 ms | 18.6 us | 17.6 us | 52.8x | 50.1x |
| ust_5000_clean_to_yield_sweep | 2579.78 ms | 2573.31 ms | 3832.9 us | 3461.2 us | 673.1x | 743.5x |
| ust_5000_curve_book_dirty_price | 30.42 ms | 27.58 ms | 426.9 us | 405.6 us | 71.3x | 68.0x |

Reading: QuantLib's time is algorithm-dominated (N re-bootstraps, per-object pricing), so the compiler/ISA
mismatch was worth low single-digit percent on the reference side — the old ratios were inflated/deflated by
a few percent, not by an order of magnitude. `sofr_23k_risk_ladder_23q_book9` ours 92.9 → 109 us is the deliberate
rank-safe COD change from the 2026-09-07 bug fixes (a correctness cost), not the toolchain. `sofr_23k_warm_recal_0p3bp`
is now LIKE-FOR-LIKE (both sides ±0.3 bp on every quote; the old bench moved QuantLib ±10 bp on swaps and
±0.01 bp on futures against a fixed 0.3 bp on ours); the honest large-move pair `sofr_23k_warm_recal_10bp` is 22x.
Method change (mean-of-3 → min-of-5) explains ~3-4 % of the "ours" improvement on unchanged code.

Fingerprint history: `52e94be82bc4` (Kaby Lake laptop, Jul 10) → `a8c9a844826e` (this Xeon, clang 16, AVX2
switch Jul 29) → `86d5211c2c03` (clang 21 engine vs clang 16 QuantLib, Aug 29 – Sep 8; mismatched) →
`966685f93279` (like-for-like, Sep 8).
