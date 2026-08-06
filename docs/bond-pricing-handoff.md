# Handoff — bond pricing & universe sweep (Stage 6, foundation)

Integration handoff for the bond work on branch **`claude/bond-pricing-asset-swaps-j3hkb8`** (base: `main`).
Read `docs/bond-pricing.md` for the design; this doc is the "how to pick it up and integrate it" summary.

## 1. What this branch delivers

Bond pricing as an extension of the engine — US Treasuries first, but a bond is **DATA** (dated cashflows +
a small yield convention), never a subclass, so new bond types are new builders filling the same structs
(CLAUDE.md §0/§1). Four commits, each gate-scoped:

| Commit | Delivers |
|--------|----------|
| `fc9c072` | Foundation: curve + street pricing kernels, universe sweep, US-Treasury builder, tests, docs |
| `8b9d9a9` | The price↔YTM cache: coupon-polynomial (Horner) evaluation — O(cashflows) FMAs + O(bonds) pows |
| `0a364dc` | Fast reverse (yield→clean, allocation-light) + O(1) cached/rolling accrued |
| `63f074b` | When-issued (WI) yield path + independent (non-QuantLib) reference oracles |

**Two pricing modes** (`include/swaps/pricing/bond.hpp`, templated on `Scalar`):
- **Curve space** — `dirty = Σ amount·DF(pay)/DF(settle)`, linear in `DF = exp(-Wx)` → rides the W-cache;
  `bond_z_spread` per-bond Newton. `portfolio::CompiledBondBook` is the batched curve-space book.
- **Yield/street space** — `dirty(y) = Σ CF·(1+y/f)^{−E}`, penny-perfect vs the street convention.
  `portfolio::BondUniverse` is the batched **price↔yield-to-maturity** sweep (Horner fast path).

**When-issued** (`build::when_issued_bond`/`us_treasury_wi`): issue-date settlement (zero accrued for a new
issue, original-dated accrual for a reopening) + prorated short first coupon, per 31 CFR Part 356 App B.

## 2. File inventory

New engine headers (all header-only, templated, QuantLib-free):
- `include/swaps/pricing/bond.hpp` — `Bond`/`BondCashflow`, `YieldBond`/`YieldFlow`, the pricing kernels.
- `include/swaps/portfolio/bond_universe.hpp` — `BondUniverse` (yield sweep), `CompiledBondBook` (curve).
- `include/swaps/build/bond.hpp` — `fixed_rate_bond`/`us_treasury`/`when_issued_bond`/`us_treasury_wi`,
  `accrued_interest`.
- `include/swaps/build/day_count.hpp` — added ACT/ACT ISDA (`year_frac`) and ICMA (`act_act_icma`).

Tests:
- `tests/bond_yield_test.cpp` — QL-FREE self-consistency (in `swaps_tests`).
- `tests/bond_reference_test.cpp` — QL-FREE independent references: Excel/OpenFormula + 31 CFR App B +
  optional Rateslib golden (in `swaps_tests`).
- `tests/bond_oracle_test.cpp` — QuantLib oracle (in `swaps_oracle_tests`; `@oracle-test` banner; listed in
  `tests/ORACLE_TESTS.md`).

Tooling / docs:
- `tools/bond_reference/{gen_golden.py,README.md}` — external-library (Rateslib) golden generator.
- `docs/bond-pricing.md` — design; `docs/bond-pricing-handoff.md` — this doc.
- `ARCHITECTURE.md` (pricing/portfolio/build rows) and `CLAUDE.md` (§8 Stage 6) updated per the
  "keep the map current" rule.

## 3. Build & verify

Standard two-gate flow (CLAUDE.md §3–4). QuantLib must be built from source with the project flags:

```bash
./tools/bootstrap_deps.sh                                   # vendors Eigen/GTest/Benchmark/Boost/QuantLib
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure                 # correctness gate
./tools/verify.sh                                           # both gates
```

Bond tests to watch: `BondYield.*`, `BondCurve.*`, `BondWhenIssued.*`, `BondAccrued.*` (in `swaps_tests`);
`BondReference.*` (in `swaps_tests`; the external-golden case skips unless the CSV exists); `BondOracle.*`
(in `swaps_oracle_tests`). To enable the Rateslib pin:
`pip install rateslib && python tools/bond_reference/gen_golden.py` then re-run `ctest -R BondReference`.

## 4. Verification status (IMPORTANT — read before merging)

This branch was developed in a container that **cannot build the vendored QuantLib oracle** (the
`bootstrap_deps.sh` toolchain step is macOS-only, and the Boost fetch is 403'd by the sandbox proxy). So:

- ✅ **Ran and passed here:** a QL-free numeric self-check compiled with `clang++ -std=c++20` against the
  vendored Eigen — price↔yield round-trip, duration/convexity vs finite-difference, batched sweep == scalar
  kernel (to 1e-13), curve PV/z-spread, reverse yield→clean == scalar (1e-13), `clean == dirty − accrued`
  (1e-15), accrued rolls linearly, WI new-issue accrued == 0, WI short-first == 31 CFR App B (1e-12),
  Excel/OpenFormula PRICE == kernel (1e-12).
- ⏳ **NOT run here — must run on a QuantLib host before merge:** `swaps_oracle_tests` (the `BondOracle.*`
  penny-perfect gate vs `BondFunctions`/`DiscountingBondEngine`/`ZeroSpreadedTermStructure`) and the perf
  gate. Every commit trailer says `Verified: correctness=SKIP perf=SKIP` accordingly — do not read those as
  "passed".

**Action for the integrator:** run `./tools/verify.sh` on a QL-built host and confirm `BondOracle.*` green
before merging.

## 5. Known issues / caveats

1. **Pre-existing oracle-guard failure (NOT introduced by this branch).** `tools/check_oracle_tests.sh`
   already fails on `main` because `ql_test_isolation.cpp`, `vol_bachelier_ql_oracle.cpp`, and
   `vol_cms_replication_oracle.cpp` lack the `@oracle-test` banner. My new `bond_oracle_test.cpp` **has** the
   banner and passes the guard. This will block `verify.sh`'s correctness gate until those three files are
   addressed separately — flagging so it isn't mistaken for a bond regression. (`ql_test_isolation.cpp`
   looks like isolation scaffolding, not an oracle, so it likely wants exclusion rather than a banner.)
2. **`gen_golden.py` is rateslib-version-sensitive.** Three calls (`bond.price(...)`, `bond.accrued(...)`)
   have shifted across rateslib releases; adjust them to your installed version. The CSV schema is the
   stable contract, and the test skips cleanly if the CSV is absent.
3. **Long first coupon rejected.** WI/odd bonds whose first payment spans >1 quasi-coupon period throw
   `std::invalid_argument` — the App B quasi-period sum is a documented follow-up. Regular + short-first +
   reopening are handled.
4. **No PR opened** (per repo instructions — open one only on explicit request).

## 6. Integration invariants to preserve

- **Penny-perfect** to QuantLib within `tests/tolerances.hpp`; add a QL-oracle case for any new bond shape.
- **The Horner fast path** requires integer-spaced exponents (`E_i = w + integer`). Keep new builders on it
  (WI stays on it because the short coupon only changes the first coefficient); `BondUniverse::is_regular()`
  guards the fallback. Anything irregular routes to the general `exp` path — correct, just slower.
- **Accrued stays decoupled** from pricing: one definition (`build::accrued_interest`), computed once and
  cached; never recompute it on the price hot path.
- **Keep the map current** (CLAUDE.md rule): any object-model change updates `ARCHITECTURE.md` + `CLAUDE.md`
  in the same commit, and any QuantLib oracle test stays registered in `tests/ORACLE_TESTS.md`.

## 7. Next steps (not in this branch)

Ordered, each building on what's here:
1. **Bond asset swaps** — par-par ASW spread is the engine's `ParSpread` quote with the bond fixed leg vs a
   market dirty price: `A·annuity = PV_bond,curve − dirty_market`. Reuses `float_leg_pv`/`annuity` +
   `CompiledBondBook`. (`build/asset_swap.hpp` + a residual/quote.) This is the originally-requested next
   product; the design sketch is in `docs/bond-pricing.md §6`.
2. **API/JSON seam** — a `bond` / `bond_universe` verb on `BundleSession` (`api/`) so the sweep reaches the
   Python/Excel/web bindings, mirroring `price_portfolio`.
3. **QL-linked sweep benchmark** — `BondUniverse`/`CompiledBondBook` vs a per-bond `QuantLib::Bond` loop, to
   put a number on the speedup under the perf gate (`bench/`, `baselines/baselines.json`).
4. **Curve-space key-rate risk** — bond DV01/key-rate via the existing `Scalar = ad::Dual` dual-use (the
   curve PV is already templated); reuse the calibration risk ladder.
5. **Long first coupon** and **non-Treasury builders** (Gilt/Bund/corporate/FRN) — new builders, same
   structs.
