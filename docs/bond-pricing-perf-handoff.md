# Bond pricing — perf handoff: CLOSED

Opened 2026-08-29 from a **cloud** session (fingerprint `b6c7ec30a7fe`), stating that the bond kernel's
*correctness* was well evidenced but its *performance* was not evidenced at all. Closed the same day on the
dev Mac Pro. Companion to `docs/bond-pricing.md` (the design) and `docs/bond-pricing-handoff.md` (the Aug-6
integration handoff).

**The one-line state:** the four gap-closing steps are done. The performance claim is now a gated
measurement, the library characterizations were read out of the source rather than recalled, the external
golden exists — and generating it exposed a real convention gap that no existing test could have caught.

---

## 1. What closed

| Step (from the original plan) | Status |
|---|---|
| 1 — read QuantLib / Rateslib source, correct the prose | **done**, §2 |
| 2 — `bench/bond_sweep_bench.cpp` + a `baselines.json` metric, Mac Pro | **done**, §3 — commit `e4f03b4` |
| 3 — time a Python library on the same universe (optional) | **not done, and deliberately dropped** — §5 |
| 4 — generate the Rateslib golden | **done**, and it FAILED, for a real reason — §4 |

`CLAUDE.md` §8 Stage 6 is corrected (it still said "ASSET SWAPS PENDING" for work that landed in
`d4e3184`/`88e9196`/`e4d0b6b`), and its §4 machine block is corrected for the OS/toolchain upgrade (§6).

---

## 2. Step 1 — the library characterizations, now read rather than recalled

Read in `third_party/quantlib/src/ql/` (QuantLib 1.35) and in the installed `rateslib` 2.7.1.

**QuantLib `BondFunctions::yield`.** The recalled description ("a root find re-walking a
`vector<shared_ptr<CashFlow>>` with day counters recomputed per iteration") is **correct, and understated**.
`BondFunctions::yield` → `CashFlows::yield<NewtonSafe>` → `solver.solve(IrrFinder, accuracy, guess=0.05,
step=0.005)`. Per solver iteration it walks the `Leg` **twice**: `IrrFinder::operator()` → `CashFlows::npv`,
`IrrFinder::derivative` → `modifiedDuration`. Per cashflow per walk: virtual `hasOccurred()`, `amount()`,
`tradingExCoupon()`; a `dynamic_pointer_cast<Coupon>` inside `getStepwiseDiscountTime`; one-or-two
`DayCounter::yearFraction` calls; and one `std::pow` in `InterestRate::compoundFactor`.

What was **not** known, and matters more than any of that: **the solver is handed a mis-scaled derivative.**
`IrrFinder::operator()(y) = npv_ − P(y)`, whose derivative is `−P′(y) = P·modDur`. But
`IrrFinder::derivative` returns `modifiedDuration(...)`, which is `−P′/P`. `BondFunctions` normalizes the
leg to a 100-face basis, so `P ≈ 99` and the derivative is ~99× too small: every Newton step undershoots
and `NewtonSafe` falls back to bisection. Measured by substituting a counting solver into the same
`CashFlows::yield<Solver>` template: **34 `npv` walks + 29 `duration` walks = 63 leg walks for ONE bond**,
~11 µs each, ~712 µs total. Starting the solve *at* the answer saves only ~16%, so it is the step scaling,
not the bracket hunt.

Cost decomposition of one 45-cashflow leg walk: ~11 µs, i.e. ~245 ns/cashflow, of which **~215 ns is
`ActualActual(ISMA)::yearFraction`**. Swapping the day counter for `Actual365Fixed` drops the full solve
783 µs → 230 µs. That is the design thesis of `docs/bond-pricing.md` §3 quantified: the expensive part is
recomputing accrual structure that never changes.

**Rateslib.** The recalled claim was that it "vectorizes within a bond rather than across bonds". That was
not the useful finding, and is superseded: what matters is that its `calc_mode`s encode two *different
conventions*, and only one of them is ours (§4).

**Also corrected: Rateslib is source-available, NOT open-source.** `gen_golden.py` and
`tools/bond_reference/README.md` both said "MIT". The installed package prints a licence notice on import:
dual-licensed, and without a registered commercial licence, use is permitted for **non-commercial purposes
only**. Running the generator is a use of it and the CSV is derived from it. Both files now say so, and
point at the QL-free fallbacks (Excel/OpenFormula + 31 CFR App B reimplementations) that need no
third-party code. **Decide before committing `tests/golden/bond_reference.csv`.**

---

## 3. Step 2 — the perf gate (commit `e4f03b4`)

`bench/bond_sweep_bench.cpp`: 5,000 seasoned semiannual treasuries, built from the same QuantLib with the
same compiler and flags (CLAUDE.md §3). Two paired metrics in `check_perf.py` + `baselines.json`, on
fingerprint `86d5211c2c03`:

| metric | ours | QuantLib | speedup |
|---|---|---|---|
| `bond_sweep` — price→YTM | **3.51 ms** | 2,500 ms (`BondFunctions::yield` per bond) | **712×** |
| — vs the harder baseline | 3.67 ms | 459 ms (`BM_BondSweep_QuantLibTuned`) | **125×** |
| `bond_book` — curve-space dirty prices | **414 µs** | 27.9 ms (per-bond `DiscountingBondEngine`) | **67×** |

**Never quote the 712× alone.** It is the number a user of QuantLib's public API actually experiences, so
it is the gate metric — but ~5.5× of it is the derivative-scaling artifact in §2, not our vectorization.
`BM_BondSweep_QuantLibTuned` re-runs the **same** QuantLib pricing (`CashFlows::npv` / `CashFlows::duration`
over the same `Leg`) from a correctly-scaled Newton, ~4 iterations: **125×** is the honest kernel-vs-kernel
number. The `bond_sweep` threshold is 50×, below both, so the gate cannot be said to rest on the artifact.

Gate integrity: the fixture **aborts** rather than reporting a speedup unless the batched yields match a
converged `BondFunctions::yield` to 1e-12 (measured 1.0e-14), the tuned baseline to 1e-12 (7.8e-16), and the
curve-space dirty prices match `DiscountingBondEngine` to 1e-9 relative (5.9e-15). The correctness check
drives QuantLib at accuracy 1e-14; at its **default** 1e-10 the two differ by ~1e-10, which is QuantLib's
own stopping rule, not a disagreement. Both *timed* sides run at library defaults, and ours is the stricter
(1e-13 on the dirty-price residual).

The cloud session's `tools/bond_lever_bench.cpp` had reported a 3.4–6.3× / 8.0–13.3× range for the
*algorithmic lever alone* on a shared container. That was an honest reading of a noisy box and of a
narrower question (its side A had no virtual dispatch, no `shared_ptr`, no date arithmetic — it was a
hand-written stand-in, not a library). The gap between it and 125× is exactly the object-model cost it
deliberately excluded. Keep the file; it still isolates the lever.

**No small-universe crossover.** The batched Newton cannot arrest a converged bond, so it was expected to
lose ground on small universes. It does not: 680 / 600 / 736 / 738 ns per bond at 100 / 1k / 5k / 10k,
against a flat ~500 µs per bond for QuantLib. The probe is opt-in (`SWAPS_BOND_SCALE=1`, filter `Scale`)
because QuantLib alone costs ~8 s per repetition and the gate runs the whole executable.

---

## 4. Step 4 — the golden was generated, and it FAILED. That is the most valuable result here.

`ust_31bii` prices disagreed with ours by ~7e-6 of price (~0.7 bp), growing with yield and maturity, with
accrued agreeing exactly. Chasing it:

**Our kernel matches Rateslib `us_gb` to 13 significant figures, and does not match `ust_31bii`.** Reading
`rateslib/instruments/bonds/conventions/`, `US_GB` uses `v1="compounding_final_simple"` and `US_GB_TSY`
(aliased `ust_31bii`) uses `v1="simple_long_stub_compounding"`. Confirmed against the regulation itself
(31 CFR Part 356 Appendix B, Section II): **every** sub-case — regular first period, short first, long
first, and the three reopened cases — is written `P[1 + (r/s)(i/2)] = …`. Simple interest over the
fractional period, never a compound `(1+i/2)^(r/s)`.

So there are two conventions, agreeing on cashflows, accrued and the whole coupon polynomial `Q(v)`, and
differing only in the fractional first period:

```
STREET   (ours, QuantLib BondFunctions/Compounded, Rateslib us_gb):   dirty = Q(v)·v^w
TREASURY (31 CFR App B, Rateslib ust_31bii = us_gb_tsy):              dirty = Q(v)/(1 + w·y/f)

dirty_AppB = dirty_street · (1 + y/f)^w / (1 + w·y/f)      (exact, not an approximation)
```

**We implement STREET only** — including in `when_issued_bond`, whose *proration* is App B but whose
*discounting* is not.

**Why no existing test caught it.** `BondReference.CfrAppendixBShortFirstCoupon` claimed to reimplement the
regulation; it reimplemented `v^{w0}` and compared it to our `v^{w0}`. A tautology that could not fail.
This is precisely the "a single oracle can hide a shared convention assumption" failure the
cross-validation section exists to prevent — and it survived because the external oracle had never
actually been run.

**Fixed.** `gen_golden.py` now emits **both** modes with a `mode` column.
`BondReference.ExternalGoldenIfPresent` asserts equality for `us_gb` (1e-9) and, for `ust_31bii`, asserts
our price times the exact convention factor equals the golden — so drift in **either** convention now
fails. `CfrAppendixBShortFirstCoupon` now checks the App B coupon polynomial (hence the proration) exactly,
checks the street↔Treasury identity, and asserts the two genuinely differ by more than 1e-6. All three
`BondReference.*` tests pass with the golden present.

**Open decision for the user, deliberately not taken here:** whether to implement a Treasury-convention
mode on `YieldBond`. It is cheap and stays on the Horner fast path — only the single `pow(v,w)` factor
becomes `1/(1+w·y/f)`, derivatives follow — but it changes what `bond_dirty_from_yield` *means*, so it
wants an explicit mode flag, not a silent switch. Bloomberg's Treasury method is the App B one, so anyone
reconciling to a Bloomberg YAS Treasury quote will need it.

---

## 5. Step 3 — dropped, with a reason

Timing `rateslib`/`financepy` on the same universe was listed as optional. It is now actively unattractive:
it is confounded by Python-vs-C++ (so it can never be quoted as the engine's speedup), and Rateslib's
licence makes benchmarking it in a potentially-commercial repo a question not worth the near-zero
information. The QuantLib pairing is same-language, same-compiler, same-flags, and now has both an
as-shipped and an artifact-free baseline. That is enough.

---

## 6. Machine / fingerprint change (affects everything, not just bonds)

The dev Mac Pro was upgraded to **macOS 26.5 / Apple clang 21** since the last capture, so the fingerprint
moved `a8c9a844826e` → **`86d5211c2c03`** and `check_perf.py` correctly refused to compare. Every metric was
recaptured under the new key; the old entries are retained and never compared against. Two notes:

- `risk_full_jacobian` reads **183×** vs the old entry's 34.9×. That is the R5/R11 pooled-AAD commits
  already on `main`, not the toolchain.
- A stale `build/` caches the absolute `-isysroot`; the upgrade deleted `MacOSX15.2.sdk` and every compile
  failed `'stdexcept' file not found`. `rm -rf build` and reconfigure. The vendored clang-16 `.a`s link
  fine against clang 21, and the old macOS-14.5-SDK `CPLUS_INCLUDE_PATH` workaround for QuantLib's
  `std::format` ADL collision is no longer needed (nor possible — that SDK is gone).

Capture caveat: taken with the desktop app running, ~0.7 of 8 cores busy. The benchmarks are
single-threaded and reproduce run-to-run within 1% (three separate full gate runs agreed to ≤1%), but an
authoritative absolute-ns capture still wants a quiesced box.

---

## 7. The published artifact still carries the pre-correction claims

<https://claude.ai/code/artifact/aef831bc-e35e-4a07-afac-a1b81b35672b>

Accurate on the mechanism (`E_i = w+i` ⇒ `P = v^w·Q(v)`, Horner with synthetic differentiation, the batched
Newton, the oracle table). Now wrong or incomplete on three points: it quotes 5.9–6.3× / 11–13× from the
noisy container instead of the gated 712× / 125× / 67×; its QuantLib/Rateslib characterizations predate §2
(in particular it does not know about the derivative-scaling artifact); and it predates the street-vs-App B
finding in §4.

---

## 8. Still outstanding (unchanged by this session)

From `docs/bond-pricing.md` §7 and CLAUDE.md §8: curve-space key-rate DV01 beyond PV; long-first-coupon
when-issued; non-treasury builders (Gilt / Bund / corporate / FRN) — builders only, no kernel change. Plus
the two decisions raised above: the Treasury (App B) discounting mode, and whether to commit the
Rateslib-derived golden CSV.
