# Bond pricing — perf handoff: the speed claim is not yet evidence

Session handoff, 2026-08-29, written from a **cloud** session (fingerprint `b6c7ec30a7fe`) for resumption
on the dev Mac Pro (`a8c9a844826e`). Companion to `docs/bond-pricing.md` (the design) and
`docs/bond-pricing-handoff.md` (the Aug-6 integration handoff).

**The one-line state:** the bond kernel's *correctness* is well evidenced; its *performance* is not
evidenced at all. Nothing in this repo has ever timed our bond code against any external library.

---

## 1. Where the code stands

All nine bond commits are merged to `main`; `claude/bond-pricing-commits-wzt2ap` is level with it and the
tree is clean. Nothing is in flight.

| Commit | What landed |
|---|---|
| `aa3eaaf` | Foundation — `pricing/bond.hpp` (curve + street kernels), `portfolio/bond_universe.hpp`, `build/bond.hpp`, ACT/ACT ISDA+ICMA |
| `0324ac5` | Horner coupon-polynomial cache for the price↔YTM sweep |
| `ff88adc` | Reverse (yield→price) pass + O(1) cached/rolling accrued |
| `7d078b3` | When-issued path + independent (non-QuantLib) reference oracles |
| `24fbb82` | Integration handoff doc |
| `ddeb0da` | Portable relative tolerances; restored `@oracle-test` banners |
| `e4d0b6b` | Stateless `bonds` run_json verb |
| `d4e3184` | Par-par asset-swap spread + QuantLib `AssetSwap` oracle |
| `88e9196` | Curve-space `asset_swap` run_json verb |

**`CLAUDE.md` §8 is stale.** Line 842 still says "ASSET SWAPS PENDING" and line 886 lists asset swaps and
the bond API verb as next; both landed in `d4e3184` / `88e9196` / `e4d0b6b`. Fix in the same pass as the
keep-the-map rule requires.

---

## 2. What is actually verified, and what is not

| Claim | Status | Evidence |
|---|---|---|
| Street price / yield / accrued / duration / convexity | **Verified** | `tests/bond_oracle_test.cpp` vs `QuantLib::BondFunctions`, `Bond::accruedAmount` |
| Curve-space dirty/clean, z-spread | **Verified** | vs `DiscountingBondEngine`, `ZeroSpreadedTermStructure` |
| Par asset-swap spread | **Verified** | `tests/bond_asset_swap_oracle.cpp` vs `AssetSwap::fairSpread`, 5e-5 |
| Batched sweep == per-bond QuantLib | **Verified** | `BondFunctions::yield` bond-for-bond |
| Convention independence | **Verified** | `tests/bond_reference_test.cpp` — Excel/OpenFormula `PRICE`/`YIELD` and 31 CFR Part 356 App B, both reimplemented, 1e-12 |
| Rateslib golden | **Not generated** | `BondReference.ExternalGoldenIfPresent` skips when the CSV is absent; it has never been produced |
| **Bond performance vs any external library** | **NOT VERIFIED** | no `bench/bond_*.cpp`, no `baselines.json` metric, no library ever executed or timed |

The last row is the whole point of this document. There is a real risk of the design doc's framing
("without a per-bond QuantLib pricing loop", "not a per-bond solve loop") being read as a measured result.
It is a design intent. It has not been measured.

---

## 3. What this session did measure — and its limits

`tools/bond_lever_bench.cpp` (added here; standalone, no QuantLib, no Eigen). It times two hand-written
evaluation *orders* of identical street math over a 5,000-bond universe (≤60 semiannual cashflows):

- **(A) bond-major** — `std::pow` per cashflow, a scalar Newton per bond. This is *our* construction of a
  per-bond loop's shape. It has **no** virtual dispatch, no `shared_ptr` indirection, no date arithmetic
  and no observer machinery, so it is a deliberately conservative stand-in — a real library loop is slower
  than this by an unmeasured margin.
- **(B) cashflow-major** — the `BondUniverse` shape: Horner across bond lanes with synthetic
  differentiation, one `pow(v,w)` per bond, one batched Newton for the universe.

Observed on this container (Xeon @2.10 GHz, 4 cores, shared, `-O3 -march=native`), across repeated runs:

| | ratio (B vs A) | agreement |
|---|---|---|
| Solve yield from clean price | **3.4× – 6.3×** | max \|Δy\| = 4.622e-13 |
| Re-mark (yield → dirty price) | **8.0× – 13.3×** | max \|ΔP\| = 3.109e-15 |

**The ~2× spread in those ratios is the most important number here.** The accuracy figures are bit-stable
run to run; the timings are not, because the container is shared and cannot be quiesced. Any single figure
picked from that range is noise dressed as a result. Re-take on the quiesced Mac Pro before quoting.

Two things the benchmark does show robustly:

- **The factoring is exact.** Polynomial arithmetic and `Σ CF·v^E` agree to ~1e-13 on yields and ~1e-15 on
  prices. The speed does not come from an approximation.
- **The batch wins while doing *more* arithmetic.** Path A stops each bond as it converges (23,320
  bond-iterations total, ~4.7/bond); path B cannot, so it does 6 × 5,000 = 30,000 bond-equivalents, ~29%
  more work — and is still several times faster on wall clock. The win is memory order and transcendental
  count, not fewer flops.

---

## 4. The gap to close, in order

**Step 1 — read the library source, then correct the prose. No build needed; do it anywhere.**
The claims about what QuantLib and Rateslib *do* (`CashFlows::yield` as a root find re-walking a
`vector<shared_ptr<CashFlow>>`; day counters recomputed per iteration; Rateslib/FinancePy vectorizing
within a bond rather than across bonds) currently come from recall, not from reading the code. Clone
QuantLib and read `CashFlows::yield` / `IrrFinder` / `BondFunctions`; read Rateslib's bond module
(`pip download rateslib` works through the proxy — verified this session). Correct anything that is wrong
in `docs/bond-pricing.md`, in this file, and in the published artifact (§6).

**Step 2 — `bench/bond_sweep_bench.cpp`, the benchmark that would settle it. Mac Pro only.**
Pair `BM_bond_sweep_QuantLib` / `BM_bond_sweep_Ours` (the naming `check_perf.py` requires):

- *Ours*: `BondUniverse::yields_from_clean` over B bonds, plus `CompiledBondBook::dirty_prices(x)` for the
  curve-space leg.
- *QuantLib*: `BondFunctions::yield` bond-for-bond, and a per-bond `Bond` + `DiscountingBondEngine` loop —
  built from the **same** QuantLib, same compiler, same flags (CLAUDE.md §3 perf-gate integrity).
- Add `bond_sweep` to `baselines/baselines.json` `thresholds` + `metric_descriptions`, capture under
  `a8c9a844826e`, and let `verify.sh` gate it.

Sweep B (100 / 1,000 / 10,000) — the crossover matters more than a single point, since the batch's
inability to arrest converged bonds should make it relatively weaker on small universes.

**Step 3 — optional, and label it.** Time `rateslib` / `financepy` on the same universe. Useful as a
sanity check on order of magnitude, but it is confounded by Python-vs-C++ and should never be quoted as
the engine's speedup.

**Step 4 — generate the Rateslib golden** (`tools/bond_reference/gen_golden.py`) and commit the CSV so
`BondReference.ExternalGoldenIfPresent` stops skipping.

---

## 5. Why steps 2–4 must run on the Mac Pro

| | This cloud container | Dev Mac Pro |
|---|---|---|
| Fingerprint | `b6c7ec30a7fe` | `a8c9a844826e` |
| CPU | Xeon @ 2.10 GHz, 4 physical cores | Xeon W-3223 @ 3.5 GHz, 8 physical cores |
| ISA | AVX-512 (8 doubles/reg) | AVX-512 (8 doubles/reg) |
| Toolchain | GCC 13.3 / clang 18 (Ubuntu) | Apple clang 16 |
| `third_party/` | **empty** — no Eigen, no QuantLib | built with our flags |
| Quiescable | no (shared) | yes |

`tools/check_perf.py` refuses to compare across fingerprints, so **anything measured here is uncommittable
by construction** — it cannot enter `baselines.json` and cannot be compared to the existing
`portfolio_analytics` 260× / `risk_full_jacobian` 34.9× / `warm_recalibration` 51.7× entries. The ISA
matches, but clock, core count and noise do not.

What *can* usefully be done in the cloud: reading library source (step 1), writing the benchmark so it
compiles, correctness work, and doc/API changes.

---

## 6. Published artifact — carries the unverified claims

An explainer was published this session:
<https://claude.ai/code/artifact/aef831bc-e35e-4a07-afac-a1b81b35672b>

It is accurate on the mechanism (the two pricing spaces, `E_i = w+i` ⇒ `P = v^w·Q(v)`, Horner with
synthetic differentiation, the batched Newton, the oracle table) and it does label the micro-benchmark as
a replica and not a gate benchmark. **Two things in it need correcting after step 1:** the
characterizations of QuantLib/Rateslib internals are unverified recall, and the measured figures are
quoted as 5.9–6.3× / 11–13× when the fuller run-to-run range is 3.4–6.3× / 8.0–13.3×.

---

## 7. Resume checklist (Mac Pro)

```bash
git fetch origin && git checkout claude/bond-pricing-commits-wzt2ap && git pull

./tools/bootstrap_deps.sh                 # Eigen, GTest, Benchmark, Boost, QuantLib 1.35 with our flags
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
./tools/verify.sh                         # confirm the bond oracle tests are green here first

c++ -std=c++20 -O3 -march=native tools/bond_lever_bench.cpp -o /tmp/blb && /tmp/blb   # quiesced

# then step 2: write bench/bond_sweep_bench.cpp, add the baselines.json metric, and
SWAPS_CAPTURE_UTC=$(date -u +%FT%TZ) ./tools/check_perf.py --build build \
  --baselines baselines/baselines.json --update
```

Also outstanding from `docs/bond-pricing.md` §7 and CLAUDE.md §8, unchanged by this session: curve-space
key-rate DV01 beyond PV, long-first-coupon when-issued, and non-treasury builders (Gilt / Bund /
corporate / FRN) — builders only, no kernel change.
