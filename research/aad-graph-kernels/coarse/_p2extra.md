
### 2.4 Coverage — measured (`coarse/ir.cpp` census over all 16 ladder rungs, `coarse/run_counts.txt`)

**Every row of every rung was lowered through `ir::Table::lower`.** A row "fits" when it lowers with no exception.

| rung | rows | fit the IR | on today's compiled tables | features used (quote rows) | atoms / sub-periods / coupons / MtM rows | IR vs templated (exact atoms) | IR vs engine (W atoms) |
|---|---|---|---|---|---|---|---|
| ois_nolag | 12 | 12 | 12 | par 12 | 31 / 134 / 134 / 0 | **0.0** | 1.4e-17 |
| ois_lag | 12 | 12 | 12 | par 12 | 61 / 134 / 134 / 0 | **0.0** | 3.5e-17 |
| ibor_multicurve | 24 | 24 | 24 | par 24 | 152 / 402 / 402 / 0 | **0.0** | 1.5e-16 |
| basis_spread | 24 | 24 | 24 | par 12, spread 12 | 92 / 402 / 402 / 0 | **0.0** | 2.2e-16 |
| averaged | 29 | 29 | 29 | par 12, spread 11, rate 6 | 218 / 526 / 406 / 0 | **0.0** | 1.0e-14 |
| averaged_leg | 24 | 24 | 24 | par 24 | 7,581 / 33,722 / 268 / 0 | **0.0** | 4.1e-15 |
| averaged_leg_moment | 24 | **12** | 24 | — | — | — | — |
| all_schemes | 18 | 18 | 18 | par 18 | 81 / 243 / 243 / 0 | **0.0** | 6.9e-18 |
| mixed_scheme | 29 | 29 | **17** | par 12, spread 11, rate 6 | 218 / 526 / 406 / 0 | **0.0** | n/a (value-dependent curve: W atoms invalid past the horizon) |
| banded | 12 | 12 | 12 | par 12 | 61 / 134 / 134 / 0 | **0.0** | 3.5e-17 |
| turns | 14 | 14 | 14 | par 12, rate 1, pin 1 | 63 / 135 / 135 / 0 | **0.0** | 2.6e-15 |
| portfolio | 14 | 14 | 14 | par 18 (incl. components), portfolio 2 | 61 / 196 / 196 / 0 | **0.0** | 3.5e-17 |
| zero_coupon | 7 | 7 | 7 | zc 7 | 8 / 7 / 7 / 0 | **0.0** | 0.0 |
| fx_xccy | 37 | 37 | 37 | par 24, fx 5, mtm 8 | 637 / 1,372 / 1,372 / 368 | **0.0** | 4.5e-16 |
| desk | 70 | 70 | 70 | par 42, spread 11, rate 7, fx 5, mtm 8, portfolio 2, pin 1 | 857 / 2,095 / 1,975 / 368 | **0.0** | 7.5e-15 |
| desk_mixed | 70 | 70 | **52** | same as desk | 857 / 2,095 / 1,975 / 368 | **0.0** | n/a |
| **total** | **420** | **408 (97.1 %)** | **390 (92.9 %)** | | | | |

**Findings.**
- **The only exception is the moment path.** The 12 `averaged_leg_moment` FF legs (`fixing_step > 0`) are a quadratic form in x, not a
  function of DFs. That is a genuine new kind: a "moment atom" `½·step·xᵀQx` is expressible, but it is not a cashflow row.
- **The IR covers 18 rows that today's compiled tables do not** (mixed_scheme's 12 and desk_mixed's 18 AAD-block rows, minus overlaps).
  They are routed to AAD today **because of the curve, not the instrument**. Their instruments are ordinary par, spread and rate rows.
  The IR puts the curve below the atoms, so the instrument side never needs routing.
- **Lowering is exact.** With atoms evaluated on the templated curves, IR quotes equal `instrument_model_quote<double>` **bit for bit on
  every covered rung** (`-ffp-contract=off`). With W-cache atoms they agree with the engine's model rates to ≤ 1e-14.
- **Gap in the fixture, not the IR:** no ladder row exercises the **compounded product** or **weighted sub-period** coupon modes (0 rows
  each; `turns`' compounded future builds as a telescoped single bracket). The IR implements both (`Mode::Cmp`, `weighted`), but nothing
  here gates them. A rung is needed before relying on them.

**Conventions DB** (`conventions/conventions.json`).
- **All 34 bundle products map to an IR quote kind:**
  - 15 `ois` + 11 `irs` → ParRate, or ZeroCoupon for `BRL-CDI-SWAP`;
  - 2 `basis` + 1 `administered-basis` (`USD-PRIME`) → ParSpread;
  - 3 `future` → Rate;
  - `FX-FWD-EURUSD` → Fx;
  - `XCCY-MTM-EURUSD` → Xccy.
- **Adjacent families need extensions, not exceptions to the table idea:**
  - `bonds` (2): fixed rows × DF; a price quote fits; a *yield* quote is a class-(D) implicit transform.
  - `credit` (1): CDS needs a survival atom Q(t) and DF·Q products.
  - `inflation` (4): needs index-ratio atoms with lag and seasonality constants.
  - `bond_futures` (4): CTD is a class-(E) argmin.

### 2.5 What it would retire or simplify

| today (hand-written) | with the IR |
|---|---|
| `problem.hpp:346` `instrument_model_quote` switch + `cashflows.hpp` `float_coupon_pv` / `float_leg_pv` / `annuity` / `par_rate` / `par_spread` / `xccy_mtm_leg_pv` as the *hot path* | kept only as the **parity oracle**; the hot path is the IR's ~8 kernels |
| `compiled_bundle.hpp:430-498` `register_at` per-kind dispatch; `compiled_book.hpp:372-415` `add` / `add_mtm` / `add_future`, `:834` `push_obs`, `:885` `push_coupon`, `BundleFixedLegs::add :934` | `ir::Table::lower` (one walk, one row type per concept). The batches' SoA arrays *are* the IR tables, renamed and made total |
| hand partials: `d_pv_from_num :721-726`, `d_rate :810`, `d_annuity :987`, quotient / FX / turn / band / moment rules in `jacobian_vs_into :253-383` | one fixed adjoint per kernel (growth, coupon mode, MtM reset, leg sum, annuity, transform, portfolio); J = G·(−diag(DF)·W) via the existing support-blocked product, or Part 1's owner sweep |
| `hybrid_residual.hpp:27-153` row router, `instrument_is_noncacheable`, `instrument_within_horizons`, `MaxTimeProbe`; `aad_block.hpp` (all 386 lines: `CachedDisc`, `RecordingCurve`, pooled-dual sweep) | **gone.** A value-dependent region is a curve-layer atom producer; instruments never route |
| the cacheable / noncacheable split and `kPooledMaxW` cliff (`dual.hpp:49`) | gone from the calibration hot path |

### 2.6 What it costs

- **Migration size (estimate):**
  - lowering ~400 lines;
  - kernels + adjoints ~600 lines;
  - curve-layer atom producers (a W block for linear regions, as today; a select kernel per value-dependent scheme) ~300 lines;
  - parity tests reuse the T3/T4 ladder machinery.
  
  The probe's `ir.cpp` lowering + value kernels are ~330 lines and already bit-exact on 15 rungs.
- **Codec and compile verb: no wire change needed.**
  - `api/codec.cpp` (1,638 lines) and `api/compile.cpp` (816 lines) produce and consume `cal::Instrument` / `BundleProblem`.
  - Keep `Instrument` as the **Blueprint** representation (editable, forkable, serialisable) and lower to IR tables in the compile step.
  - The IR is internal to the CompiledModel, so the Excel/Python/C ABI see nothing.
- **Object model (E4.A, decided 2026-09-09): a clean fit.**
  - Blueprint = today's `Instrument` list.
  - CompiledModel = IR tables + curve-layer atom producers + row layout.
  - CalibratedState = x + quote arrays.
  - `set_quotes` / `set_bands` stay scalar row arrays.
  - **E4.A.3 fixings become natural:** a fixing updates a coupon row's `konst` / `rf` and its first-unfixed sub-period index. The daily
    boundaries are already atoms, so W never changes.
- **Risks:**
  - a dual-maintenance period, with templated (oracle) and IR (hot path) both live;
  - untested compounded and weighted modes (a fixture gap);
  - the moment path needs its own atom kind or stays on a fallback.

### 2.7 Does the IR remove recording?

For instruments, yes: lowering goes straight from the table to the kernels (2.3). Recording stays valuable for three things:
- the **parity oracle** (bit-level against the templated maths);
- the **curve layer** of value-dependent schemes, unless a hand-written select kernel is preferred;
- a **generic fallback** for maths that has no row kind yet: moment path, bond yield (D), CTD (E), options.

