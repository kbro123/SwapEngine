# Coarse-grained compiled kernels, a cashflow-table instrument IR, and branching

Research note, 2026-09-15. Engine `SwapEngine` @ `55ab196` (branch e2-conventions), read-only. Scope set by the owner: **no JIT**. The
problem is data structures; the kernels are pre-compiled engine code.

**Where things are.**
- Code, logs and generated outputs: `scratchpad/aadjit/coarse/`.
- Engine paths are relative to `SwapEngine/include/swaps/`.
- Builds used `-O3 -DNDEBUG -fno-math-errno -march=x86-64-v3` (the engine's flags) plus, where stated, `-ffp-contract=off`.

**How the machine was shared.**
- Every timing loop held while any of `shape_ladder_bench`, `check_perf.py`, `verify.sh`, `ctest`, `mutate.py`, `probe_dm_stall`,
  `ninja`, `clang++` or this study's own probes ran, and resumed only after 120 s of none.
- Each timing block runs under a watcher that discards and re-runs the block if any of those appears mid-block (`coarse/util.hpp`).
- Counts-only probes ran without that hold, niced, and never during a benchmark or test step.
- The load at each timing is printed in the logs.

**Status of this file.** Every count, parity result, coverage number and streamer trace below is measured. The **timing tables are
pending**: the timed runs (`coarse/run_chain.txt`, `coarse/run_part3.txt`) have been holding since 13:45 behind the main line's G4
pipeline. They will be appended, not re-run elsewhere.

---

## PART 1 — Coarse-grained lowering of the recorded graph

### 1.1 Method (`coarse/coarse.cpp`, on top of the unchanged recorder and collapse from `tape.cpp`: `coarse/rec.hpp`, `coarse/kernel.hpp`)

The recorded scalar tape (CSE + DCE, guards) is lowered into a handful of **pre-compiled loop types** over SoA index/weight arrays.
Dispatch happens once per group, never once per scalar op.

| requested kernel | what the lowering produces |
|---|---|
| (a) sparse linear block → exp → DF | the affine collapse's rows, grouped into **dense column-span blocks** (rows reading the same knot span `[lo, hi)` share a dense matrix: one Eigen GEMV per block, the generalised `CompiledCurveSet::df_into`), then an `EXP` group |
| (b) gather-and-weighted-sum | **fold-sum rows**: every left-deep `ADD`/`SUB` chain whose inner nodes have a single use becomes one `SUM` element (terms + ±1 signs), evaluated as the same left fold, so it is **bit-identical** to the scalar chain. Legs, annuities and sub-period sums all become this |
| (c) elementwise ratio / log / product across instruments | **level scheduling**: every remaining op gets level = 1 + max(operand levels); ops are grouped by (level, op type); slots are renumbered so each group writes a **contiguous** output range. One loop per group: `o[i] = v[a[i]] ⊕ v[b[i]]` |
| (d) quote transforms | fall out as the top levels (`DIV` for par/spread quotients, `LOG`/`EXP` for ZC and FX), batched across instruments by op type |
| (e) residual scatter | the output slot list (the band/FX-log tail stays per row, as today) |
| irregular nonlinear parts (Hyman) | the same level/type groups (they are just a few more elements), under guards |

**Three modes.**
- **exact:** no rows, no reciprocals.
- **fast:** affine rows as span blocks.
- **fast+inv:** fast, plus shared reciprocals `a/b → a·inv(b)`, the engine's `INV` trick.

**Three Jacobians.**
- **(A) owner-partitioned reverse sweep.**
  - Every node is owned by the set of residual rows whose cone contains it.
  - A single-owner node keeps one adjoint; a node shared by k rows (a DF, a curve-layer op) keeps k adjoints ("owner pairs").
  - One reverse pass over the **same level/type groups** (descending) then produces **all rows' Jacobians at once**.
  - It finishes by scattering row-slot adjoints through the rows' W. That is exactly the "J of exp(−W x) is −diag(DF)·W" identity,
    derived rather than hand-written.
- **(B) per-row cone reverse** on the scalar collapsed tape (from `tape.cpp`).
- **(C) forward-vector mode** through the same groups, tangent width n; W rows seed the tangents.

### 1.2 Counts (no timing) — `coarse/run_counts.txt`, build `-ffp-contract=off`, 2026-09-15 15:17

**Kernel modes.**
- **exact:** levelised op-type groups plus fold-sum rows, no W and no reciprocals. It must match the templated double kernel bit for bit.
- **fast:** affine rows as dense column-span blocks, plus groups.
- **fast+inv:** fast, with a shared reciprocal replacing each divide.

| rung | dispatches/eval: scalar interp | scalar collapsed | coarse exact (elements) | coarse fast | fast+inv | rows / span blocks | adjoint elements / owner-pairs / groups | record-free build | allocs/replay |
|---|---|---|---|---|---|---|---|---|---|
| ois_nolag | 815 | 220 | 108 (778) | 43 | 58 | 30 / 11 | 960 / 1,038 / 47 | 1.2 ms | 0 |
| mixed_scheme | 4,154 | 967 | 238 (3,848) | 82 | 97 | 248 / 34 | 3,482 / 3,739 / 61 | 4.3 ms | 0 |
| fx_xccy | 10,629 | 3,459 | 136 (9,989) | 93 | 101 | 637 / 34 | 10,956 / 12,354 / 67 | 10.9 ms | 0 |
| desk | 15,269 | 4,537 | 184 (13,928) | 122 | 128 | 857 / 53 | 15,599 / 17,572 / 75 | 20.9 ms | 0 |
| desk_mixed | 15,351 | 4,566 | 270 (14,010) | 137 | 142 | 888 / 65 | 15,599 / 17,572 / 75 | 21.1 ms | 0 |

"Record-free build" covers the scalar kernel, the collapse, the three plans and the adjoint program; recording itself is extra.

**Parity at an x the plans were not recorded at** (x_true + 1e-5·sin):

| rung | exact vs templated | fast vs engine | fast+inv vs engine | owner-reverse J vs engine | forward-vector J vs engine |
|---|---|---|---|---|---|
| ois_nolag | **0.0** | 6.9e-17 | 6.9e-17 | 1.7e-16 | 2.2e-16 |
| mixed_scheme | **0.0** | 8.3e-15 | 2.5e-15 | 1.8e-15 | 2.9e-15 |
| fx_xccy | **0.0** | 4.5e-16 | 2.2e-16 | 6.2e-16 | 4.4e-16 |
| desk | **0.0** | 7.5e-15 | 2.2e-16 | 1.9e-15 | 2.0e-15 |
| desk_mixed | **0.0** | 7.5e-15 | 2.2e-16 | 1.9e-15 | 2.0e-15 |

Observations:
- **Dispatch drops two orders of magnitude.** On desk it goes 15,269 → 122-184 per evaluation (a group dispatch replaces a per-op switch).
- **The coarse exact kernel is bit-identical** to the engine's templated double kernel on every rung, provided both are built without
  FMA contraction. (With the engine's default flags, see the `fma` run.)
- **Rounding noise in the representation flips degenerate-tie guards.** fast+inv reports **5 guard flips** on the MonotoneCubic rungs at the
  same x where exact and fast report 0: `a·inv(b)` differs from `a/b` by up to 1 ulp. Those flips are the ~1e-20 degenerate ties found in 3.5
  (equally spaced knots, a linear x_true), not real kinks. They are harmless for the value (the arms are equal), but a guard-and-re-record
  policy must not re-record on them. **Classify flips by the gap between the two arms, not by the bit alone.**

**The same parity check under the engine's default flags** (`-O3 -march=x86-64-v3`, clang's default `-ffp-contract=on`):

| rung | exact vs templated | fast vs engine | fast+inv vs engine (tie flips) | owner-reverse J | forward-vector J |
|---|---|---|---|---|---|
| ois_nolag | 7.3e-17 | 3.1e-17 | 2.8e-17 (0) | 2.2e-16 | 1.7e-16 |
| mixed_scheme | 2.5e-15 | 5.6e-15 | 4.2e-15 (2) | 1.8e-15 | 2.1e-15 |
| fx_xccy | 2.8e-17 | 2.2e-16 | 3.4e-16 (0) | 5.0e-16 | 4.4e-16 |
| desk | 2.5e-15 | 6.6e-15 | 5.6e-15 (0) | 2.6e-15 | 3.9e-15 |
| desk_mixed | 2.5e-15 | 6.6e-15 | 5.6e-15 (2) | 2.6e-15 | 3.9e-15 |

**Max diff with the engine's default flags: 6.6e-15 on values, 3.9e-15 on the Jacobian.** The compiler contracts the templated kernel and
the coarse loops differently, so "bit-identical" holds only in a no-FMA build of both sides (as in 1.x above). The residual gap is far
inside the T3 tolerance (1e-12).

### 1.3 Timings — PENDING (held for a quiet machine; see status at top)

For orientation only, from the loaded-machine scalar probe of the previous note (`AADJIT-research.md`, load 3.5-11):

| rung | value eval µs, engine | value eval µs, scalar collapsed | value+J µs, engine J alone | value+J µs, scalar collapsed fwd+J |
|---|---|---|---|---|
| ois_nolag | 0.49-0.82 | 0.79-1.34 | 3.0-5.2 | 5.0-8.6 |
| desk | 9.3-11.6 | 19.5-22.7 | 195-264 | 111-116 |
| mixed_scheme | 39-50 | 4.4-5.0 | 93-187 | 23-35 |
| desk_mixed | 141-175 | 20.6-22.7 | 786-1,156 | 111-168 |

The coarse lowering cuts dispatch from ~15k to 122-184 per desk eval (1.2), which removes the loss mechanism that note identified. Whether
it **matches** the hand-written W-cache on linear rungs is exactly what the held timing run measures; it is not claimed here.


---

## PART 2 — Would redefining instruments as a cashflow-table IR help?

### 2.1 How instruments are defined today

**The definition is already mostly data.** Paths below are relative to the engine's `include/swaps/`.

- `calibration/problem.hpp:108` `Instrument` is legs plus a quote transform plus a target:
  - `FloatLeg` (`:40`: coupons, forecast/discount roles, MtM reset roles + `fx_spot` + `fx_spot_time`);
  - `FixedLeg` (`:56`);
  - `RateObservation obs` + `convexity` (Rate);
  - FX fields; `TurnJump` (curve, index);
  - `combination` (Portfolio, `WeightedInstrument` `:174`, recursive).
- `QuoteKind` (`:62`) has 8 kinds: ParRate, ParSpread, Rate, ZeroCouponRate, FxForward, XccyMtmBasis, Portfolio, TurnJump.
- `pricing/cashflows.hpp` defines the rows:
  - `FloatCoupon` (`:104`): pay, `tau_pay`, spread, scale, reset/accrual/`reset_fx`;
  - `RateObservation` (`:56`): sub-period start/end, weights, realized, `tau_index`, `fixing_step`, compounded,
    `realized_factor`, fixing schedule;
  - `FixedCoupon` (`:151`).

**Builders.** `build/instruments.hpp` fills those structs from the conventions DB (`par_swap :229`, `basis_swap :252`,
`xccy_mtm_basis :271`, `rate_instrument :330`, `fx_forward :349`, `turn_jump :372`, `zero_coupon_swap :204`) together with
`build/observations.hpp` (`observation :133`, `moment_observation :180`, `rfr_observation :249`, `scheduled_observation :296`).

**The maths lives in templates that branch on data** (`cashflows.hpp`):
- `float_coupon_pv :279` has three modes: compounded / plain fast path / general k-form;
- `rate :248`, `obs_compound_growth :206`, `obs_numerator :225` (daily Σ or the moment path);
- `xccy_mtm_leg_pv :342`, `annuity :391`;
- `instrument_model_quote` (`problem.hpp:346`) is a switch on `QuoteKind`.

**`compiled_bundle.hpp` / `compiled_book.hpp` already lower the SAME structs to SoA tables.** `register_at`
(`compiled_bundle.hpp:430-498`) dispatches per kind into:
- `BundleFloatBatch::add / add_mtm / add_future` (`compiled_book.hpp:372, 387, 415`) → `push_obs :834` / `push_coupon :885`
  (sub-period index arrays `subS`/`subE`/`sub_w`, per-coupon `pay`/`konst`/`k`/`realized`/`inv_tau`/`convexity`, MtM
  `rN`/`rD`/`dS`/`dE`);
- `BundleFixedLegs::add :934`.

The hand-written part is not the table; it is **(i)** which kinds the batch accepts and **(ii)** the per-kind analytic
partials (`d_pv_from_num :721-726`, `d_rate :810`, `d_annuity :987`, the quotient/FX/band/turn/moment rules in
`jacobian_vs_into :253-383`).

**So a cashflow-table IR is not a new idea for this engine: it is `compiled_book`'s tables, promoted from an internal cache to
the definition, and generalised to every kind.**

### 2.2 The IR, as prototyped (`coarse/ir.cpp`)

```cpp
namespace ir {
// ---- atoms: the ONLY curve-dependent inputs -------------------------------------------------------------------
struct AtomRef { int curve; double t; };            // DF(curve, t); registered once, deduplicated
// (value-dependent curves plug in HERE: the atom vector is produced by the curve layer -- a W block + exp for linear
//  regions, a small recorded op-list for a MonotoneCubic region -- the tables above it never know which)

// ---- table rows ---------------------------------------------------------------------------------------------------
struct Sub    { int s, e; double w; };                    // growth g = DF[s]/DF[e] − 1 (w: day weight, 1 if unweighted)
enum class Mode : uint8_t { Plain, Gen, GenEmpty, Cmp, CmpEmpty };
struct Coupon {                                           // one floating cashflow
  Mode mode; int pay;                                     // DF atom at pay (−1: observation-only row, e.g. a future)
  int sub0, sub1; bool weighted;                          // Σ w·g (Gen) or Π (1 + w·g) (Cmp)
  double konst, k, rf;                                    // pv = DF[pay]·(num + konst)·k ; Cmp: DF[pay]·(rf·Π + konst)·k
};
struct MtmReset { int cpn, ex_s, ex_e, rn, rd, sn, sd; double fx_spot, reset_fx; };  // v = (pv + DF[e] − DF[s])·N
struct Fixed  { int pay; double w; };                     // annuity row DF[pay]·(τ·scale)
struct Range  { int b, e; };

// ---- quote transforms (the ONLY per-kind code, and it is a handful of scalar formulas) -----------------------------
enum class Tf : uint8_t { ParRate, ParSpread, ZeroCoupon, XccyBasis, Rate, Fx, StatePin, Portfolio };
struct Quote {
  Tf tf; Range pos, neg, fix, mtm, comps;                 // legs = coupon ranges, annuity = fixed range, comps = Σ w·Q
  double tau, conv, fx_spot, fx_spot_time, realized, tau_index, rfac; bool compounded;
  int obs;                                                // Rate: its observation row
  int fn, fd, fsn, fsd;                                   // FX atoms (T and spot date)
  int state;                                              // StatePin: x index (TurnJump δ)
};
struct Comp { double w; int q; };                         // Portfolio / butterfly: linear combination of quotes
struct Table {                                            // the CompiledModel's instrument half
  std::vector<AtomRef> atoms; std::vector<Sub> subs; std::vector<Coupon> cpns; std::vector<MtmReset> mtms;
  std::vector<Fixed> fixed; std::vector<Quote> quotes; std::vector<Comp> comps; std::vector<int> row_quote;
};
}
```

**Lowering to fixed kernels.** One loop per table, no recording, no per-instrument code:

| step | kernel | what it computes |
|---|---|---|
| K1 | atoms | `DF = exp(−W·x)` (block GEMV + exp, today's `df_into`), or per-region plans for value-dependent regions |
| K2 | growth | `G[k] = DF[s]/DF[e] − 1` over **all** sub-periods of all instruments |
| K3 | coupons | the three `float_coupon_pv` modes as row-mode buckets |
| K4 | MtM resets | `(pv + DF[e] − DF[s])·fx·DF[rN]/DF[rD]` |
| K5 | leg / annuity sums | segment sums over contiguous row ranges |
| K6 | quote transforms | per `Tf` bucket |
| K7 | portfolio | Σ w·Q |
| K8 | scatter to residual rows | then the band / FX-log tail, as today |

**Jacobian per kernel, not per instrument.** Each kernel has one fixed analytic adjoint (K2: `∂g/∂DF[s] = 1/DF[e]`,
`∂g/∂DF[e] = −DF[s]/DF[e]²`; K5: identity on the range; K6: quotient rule on (Σpos − Σneg, ann); …). The Jacobian is then:
1. one reverse sweep through K8→K2 produces G = ∂Q/∂DF, sparse per row;
2. J = G·(−diag(DF)·W), which is the existing support-blocked product.

This is exactly what Part 1's owner-partitioned sweep does generically. The IR makes the partials ~8 fixed formulas instead of
per-quote-kind code.

**Value-dependent pieces.**
- **Hyman clamps** live entirely *below* the atoms (curve layer). The IR tables do not change. The atom producer for a
  MonotoneCubic region is a small plan (Part 1: ~30 nonlinear ops + 78 guards per 7-node region), or, after the `select`
  rewrite, a select network, feeding per-pattern W rows. The router and linear horizons disappear, because instruments no
  longer care what their atoms' curve is.
- **Band edges** stay in the streamer (`side_of`, `breakpoint`, `rescale_row`): the IR ends at the model quote.
- **FX-log and ZC tails** are per-row scalar transforms on the live q, as today.

### 2.3 Does the IR remove the need to record?

**For the instrument layer: yes.** The table *is* the instrument graph. Lowering is a direct walk over `Instrument`: `ir::Table::lower`
is ~150 lines replacing `register_at`, the router and the `AadBlock`. It produces the same kernels a coarse lowering of a recording
would, with no graph analysis.

Recording keeps three jobs:
1. **Parity oracle.** Record (or just run) the templated maths and compare against the IR kernels. This is bit-level under
   `-ffp-contract=off` when atoms are evaluated on the templated curves, and rounding-level against W-cache atoms.
2. **Curve layer for value-dependent regions.** The atom producer for a MonotoneCubic region: a plan/select network, or a
   hand-written clamp-pattern W. The IR is silent about curves by design.
3. **Generic fallback for new maths** before it earns an IR row kind: the moment path, bond yield (a root-solve transform),
   bond-future CTD (a min over deliverables, i.e. a value branch), CDS survival and inflation index atoms, options.


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
- **The IR covers 30 rows that today's compiled tables do not** (mixed_scheme's 12 and desk_mixed's 18 AAD-block rows), and misses 12
  that they do (the moment path): 390 + 30 − 12 = 408.
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


---

## PART 3 — Branching under record → coarse kernels

### 3.1 Every branch on the pricing/interpolation path, classified

Paths are relative to the engine's `include/swaps/`.

#### (A) Structural — fixed per compiled structure; folds to constants at record/lowering time

Changing any of these is a structure change, so it means a recompile (the CompiledModel rule).

| where | branch | depends on |
|---|---|---|
| `curve/regions.hpp:83,90-101` Flat | `t <= t0_`, `t >= m_.back()`, `lower_bound` segment | query time vs knot times |
| `:125-130,196-200,314-318,434-438,982-986` | `lead = !in.has_predecessor` (a leading vs following region) | region order |
| `:146-174,258-288,366-396,495-525,1039-1079` | pre/post-segment `t <= xs_.front()`, `t >= xs_.back()`, `upper_bound` segment | time |
| `:216,336,1003` | `nseg >= 2` end-tangent formula | knot count |
| `:553,559,568,575` Hyman | `i == 0`, `i == N−1`, `i > 1`, `i < N−2` | node index |
| `:635,657,684-707,720-727,736,746,752-756,767,774,785,794-805` BSpline | domain clamps, de Boor span search, `den > 0` knot-multiplicity guards, `first` accumulator seeds | knot / time |
| `:836-940` Tension shape functions | `abs(x) < 0.5` series vs closed form; `x > 20` asymptotic | σ·h, σ·v (constants) |
| `curve/curve_module.hpp:141,154,207,214,220` | σ default, linearity query, `integral(t <= 0) = 0`, `zero(t <= 0)`, region `locate` | spec / time |
| `pricing/curve_handle.hpp:96,137,143,202-223` | turn window membership, outright vs spread, turns present, linear-horizon routing | time / spec |
| `pricing/cashflows.hpp:141-145` | `mtm_coupon_is_seasoned` (`reset_fx` set, FX fixed, start or reset in the past) | fixings / eval date |
| `:190-213` | weighted vs unweighted sub-periods | observation shape |
| `:226-234` | moment path (`fixing_step > 0`, `step3`, weight) | observation shape |
| `:250-255,291-311` | compounded, empty sub-periods, plain fast path | observation shape |
| `:320,344,392` | empty legs | leg shape |
| `:350-381` | accrual set, reset time, settled exchanges `e >= 0` / `s >= 0`, fixed `reset_fx`, `fx_spot_time != 0` | dates / fixings |
| `:472-482,500,509,517,526,537` | quadrature order, panel split at curve pieces, empty windows | structure |
| `calibration/problem.hpp:374,395,436,445,448` | `fx_spot_time == 0` spot roll, FX log residual, unbanded Rate association, banded vs unbanded | instrument spec |
| `pricing/compiled_book.hpp:467-627,683,729-790` | reduce layout, `sub_is_identity`, `cpn_is_plain`, `has_mtm_`, moment coupons, `rN >= 0` | batch shape (decided at `finalize`) |
| `curve/inflation.hpp:50,64-77,104` | seasonality active, phase, month index clamp | time / spec |
| `curve/parametric.hpp:42,49,70,76` | `t <= 0` | time |
| `pricing/fixings.hpp:137-187` | past vs future fixing days, compounded accumulation, all-ones weights | eval date + fixing table |

#### (B) Value-dependent min/max/abs/sign/limiter

- **The only one on the curve calibration path:** the Hyman monotonicity filter, `curve/regions.hpp:544-586`. It consists of:
  - `smin`/`smax` (`:544-545`);
  - `abs` and the sign `m/|m|` (`:555,561,583`);
  - the sign-agreement predicates `m·S > 0` (`:554,560,582`);
  - the monotone-run predicates `(S_{i−1}−S_{i−2})(S_i−S_{i−1}) > 0` and `pm·pd > 0 ∧ pm·(…) > 0` (`:569-579`);
  - the no-op test `correction != m` (`:558,564,586`).
- **Recorded guard count:** 78 for a 7-node region.
- **No floors** on the curve path. There is no zero floor on rates, no `max(0,·)` in the curve or instrument kernels, and hazard
  positivity is by reparameterisation, not a clamp. `max(V, 0)` exists only in XVA exposure, off the calibration path.

#### (C) Value-dependent formula selection

- **The only one:** the Huber band residual `band_residual` (`calibration/problem.hpp:219-224`), its double/slope twins (`:228-241`), and
  the streamer's active-set machinery over it (`calibration/streaming.hpp:530-700`: `side_of`, `breakpoint`, `apply_pins`,
  `track_bands`, `verify_pins`). There is one formula per side (below, inside, above) with slope `decay` or 1.
- The MonotoneCubic clamp is (B), not (C): both arms are the same formula family. The FX log residual has no branch.

#### (D) Iterative / implicit

**None inside a curve-calibration quote transform.** Every bundle quote is a closed form in DFs. Iterative solves elsewhere:

| where | what | calibration path? |
|---|---|---|
| `pricing/bond.hpp:90-104` | a spread-to-price Newton | a bond-fit quote, if bonds are ever fitted in yield/spread terms |
| `pricing/bond.hpp:228-234` | yield from clean price, Newton on dirty price | same |
| `vol/fx_vol_surface.hpp:110-135` | strike from delta: bisection on monotone branches around an interior maximum | FX vol-surface calibration |
| `vol/fx_black.hpp:264-276` | GK implied vol, safeguarded Newton-bisection | FX vol-surface calibration |

`calibration/bond_fit.hpp` fits on price (closed form), so no root-find is on the bond-fit path today.

#### (E) Discrete argmin

`pricing/bond_future.hpp:151-154` `select_ctd` takes the argmax of implied repo over deliverables. It is **not on the curve
calibration path**; it is on the bond-future pricing/RV verbs. Nothing on the bundle path picks a discrete argmin.

**Summary:** outside (A), the curve calibration path has exactly **one (B) site** (Hyman) and **one (C) site** (Huber bands,
already owned by the streamer). (D) and (E) exist only in adjacent products.

### 3.2 Representations compared per branch class (qualitative; measured numbers in 3.3)

| representation | value / derivative correctness | replay cost | SIMD inside a coarse kernel | ties / boundaries (hysteresis) |
|---|---|---|---|---|
| **(1) predicate / select** (both arms, mask) | exact value; derivative = the selected arm's (a one-sided subgradient at the kink, the same side AAD picks today: `smin` ties → first arg, `abs'(0) = +1`) | both arms every eval: cost = Σ arms + mask | yes: mask blend (`blendv`) across lanes/scenarios; no data-dependent control flow | none: the predicate is re-evaluated every eval, so the kink is re-decided at every iterate (the source of 2-cycles if the streamer does nothing) |
| **(2) guard-and-replay** | exact on the recorded pattern; **silently wrong after a flip until the guard is checked** (the check is part of every replay) | taken arm only, and under a fixed pattern the limiter **collapses into the linear rows** (no filter work at all) + O(#guards) compares; a flip → re-record + re-lower | the replay kernel is branch-free; the guard check is a vectorisable compare + OR | natural hysteresis point: the pattern can be HELD across iterates (a policy choice, not a numerical accident) |
| **(3) per-pattern cache** | as (2) | as (2); a flip to a seen pattern = a hash lookup (µs), a new pattern = (2)'s rebuild | as (2) | as (2); the cache makes ping-pong between two patterns cheap, which is also what hides a 2-cycle if nothing else detects it |
| **(4) implicit-function op** (D only) | exact value at convergence; derivative −F_x⁻¹F_p analytic (no iteration unrolling, no tape growth with iteration count, no dependence on the solver's path) | one converged solve (done anyway) + one small linear solve per derivative | the solve is scalar/serial; the derivative is a dense small solve | tolerance-dependent: the node is only as good as the convergence; kinks in F (e.g. bond price kinks) need (1)/(2) inside F |

Per class:
- **(A)** Fold. No representation needed; a change is a recompile.
- **(B) Hyman**:
  - (1) is correct and simple, but must avoid NaN in the discarded arm (study (a));
  - (2)/(3) are faster per eval because the whole filter folds into W under a pattern;
  - the break-even is set by the flip rate (study (b)).
- **(C) Huber bands**: keep in the streamer as today. Per row, the formula selection is (1) (a 3-way select on q) and costs nothing. The
  *active-set logic* is the streamer's, and should not be hidden inside a kernel.
- **(D)** (4), when a yield/implied quote is ever calibrated. Never unroll a Newton loop into a tape: that gives a long tape, a
  solver-path-dependent derivative, and iteration-count guards.
- **(E) CTD**: (2)/(3) with the argmin index as the guard (a discrete pattern), or (1) as a select over the candidates' values when
  the candidate count is small. The derivative is the chosen deliverable's (one-sided at a switch), the standard market convention.

### 3.4 The streamer interaction: desk_mixed's false stall IS a Hyman guard flip (measured)

**Probe.** `coarse/guardtrace.cpp`, counts only, output in `coarse/guardtrace_out.txt`.
- A `BundleProblem` subclass selects a logging engine through `residual_engine<>`, so the streamer runs unmodified. Breakeven is
  pinned at 64, as in `audit/probes/probe_dm_trace.cpp`.
- Every x the streamer evaluates is replayed through desk_mixed's tape, recorded at the committed state. The tape has 94 guards, all
  in SOFR's MonotoneCubic region; the count depends on the pattern, because nested predicates only record when their outer one holds.
- Guard outcomes are printed between the streamer's own trace lines. Ticks run q0 ↔ q_small from the committed state.

**q_small tick** (repeats identically on every q_small tick: 5 steps, 1 refresh):

```
[r eval 1: != anchor 0]
step 1 |dx|=4.69e-03
[r eval 2: != anchor 14, != previous 14]                          <- the big first step crosses 14 Hyman predicates
step 2 |dx|=1.31e-04
[r eval 3: != anchor 10, != previous 4: g86 (-6.0e-05->+2.2e-05) g88 (-1.7e-10->+4.1e-10) g89/g90 (-3.0e-05->+1.1e-05)]
kink 2-cycle: half step                                           <- the FLK2 reversal detector fires
step 3 |dx|=1.31e-04  adaptive: rho 1.000 -> REFRESH              <- compares the UNDAMPED reversing |dx| with the previous
[J eval 4: != previous 4: g86 g88 g89 g90 flip back]              <- the half step lands on the other side again
step 4 |dx|=1.01e-05 ; step 5 |dx|=4.89e-11 converged
```

**q0 tick** (repeats on every q0 tick: 10 steps, 1 refresh):

```
step 1: 6 guards flip back (g28 g61 g86 g88 g89 g90); pattern now "8 != anchor"
steps 2..8: NO guard changes, |dx| contracts linearly at rho = 0.577 (the frozen operator is the q_small side's)
refresh (the max_frozen = 8 hard cap, not the adaptive stall), step 9: the last 8 guards flip to the anchor pattern with margins 1e-12 -> 6e-18
```

**What this shows.**
1. **The 2-cycle is exactly one group of Hyman predicates** (four guards at the region's far end) flipping between consecutive
   Newton iterates. The Jacobian differs on the two sides, so each side's frozen step points back across the kink.
2. **The committed solution sits ON a kink.** On the q0 tick the final flips happen at margins of 1e-12 to 1e-18, i.e. at rounding.
   That is why the q0 walk contracts only linearly (rho 0.577): it approaches a kink point with the other side's operator.
3. **The refresh is forced by construction, not by staleness.** After the half step the adaptive stall computes
   `rho = |dx_undamped| / |dx_prev|`, and the 2-cycle test itself required those two to be the same size
   (`last_step_.squaredNorm() < 1.21·n2`). So rho ≈ 1 → REFRESH, every tick (`streaming.hpp:411-459`).
   - Two cheap fixes: exclude a damped step from the contraction history (set `dx_prev = -1` when `damp < 1`), or feed it
     `damp·|dx|`. Either removes the false refresh. The iterate path changes at rounding level, and the fixed point does not.
4. **Only 4 distinct guard patterns occur over 51 evaluations.** The oscillation is a small active set, not chaos.

**Treating interpolation kinks as an active set, like band edges.** The mask/guard bits give, per evaluation and for free:
- which predicates changed since the previous iterate (the flip set), and
- their signed margins, i.e. how far the iterate is from each kink.

The streamer already runs exactly this machinery for Huber band edges: `breakpoint` (step length to the nearest edge), `switch_row`,
the pin-on-two-flips rule (`flips_[k] >= 2`), `verify_pins` (a KKT release test). The kink version would be:

| band machinery (today) | kink machinery (proposed) |
|---|---|
| a row's side of its band | a predicate's truth value (guard bit) |
| `breakpoint`: α to the nearest band edge along dx | α to the nearest predicate zero along dx: each margin is a smooth function of x, so α ≈ −margin / (∂margin/∂x · dx). The partial is one row of the same forward/tangent pass, or a finite difference between the two iterates' margins (free: both margins are already computed) |
| `switch_row`: rescale the row's slope | switch the pattern: under (2)/(3) swap in the pattern's W rows, under (1) nothing to swap. The Jacobian rows of the instruments reading the region change, which is a low-rank update of M (those rows only), like `rescale_row` |
| pin on two flips in one tick | **HOLD the pattern within a walk:** when a predicate flips twice in one tick, pin it at its edge (a kink optimum), exactly the band rule |
| `verify_pins`: KKT release | release if the pinned side's one-sided directional derivative says the objective decreases into the other side (the same λ-sign test) |

**Would it replace the half-step heuristic?**
- **Yes, for Hyman kinks**, and more exactly. The half step is a blind bisection that detects the cycle only after it has happened
  (one wasted step + a forced refresh). A breakpoint walk stops *on* the kink in one step, knows which predicate it is, and pins it.
- The half step would stay as a generic safety net for kinks the kernel does not expose (for example a future scheme written
  without masks).

**Cost.**
- **Per evaluation:** the guard/mask compare, 78-94 compares ≈ 0.1 µs, plus keeping the previous bits.
- **Per step:** α computation over the flipped/near predicates only.
- **Per switch:** a low-rank update of M restricted to the rows reading the region: desk_mixed has 18 AAD rows on SOFR's long end,
  cheaper than the full refresh the heuristic triggers today.
- **Engineering:** the streamer's band code generalises (a "constraint" = band edge or kink predicate). Risk: rows can straddle
  several kinks; the pin/release logic must handle multiple simultaneous predicates, as it already does for multiple bands.
- **Prerequisite:** the pricing kernel must export the bits and margins. That falls out of (1) predicate/select (the mask *is* the
  bit, the predicate value *is* the margin) and of (2) (the guard check).

### 3.5 Main-line q_small variants: V1 flips real Hyman kinks, V4 only moves degenerate ties (measured)

**Probe.** `coarse/guardtrace2.cpp`, counts only, output in `coarse/run_part3b.txt`.
- `variant()` is copied verbatim from `scratchpad/dm/probe_dm_stall_qsmall.cpp`. Protocol as there: calibrate, break-even pinned 64,
  6 warm ticks alternating q0 ↔ V1.
- Every iterate goes through a **named** predicate evaluator for SOFR's MonotoneCubic region: the engine's `hyman_filter` predicate by
  predicate, giving node, knot interval, truth, margin, and whether the engine evaluates it at all. It is cross-checked against the
  recorded tape's 94 guards.
- desk_mixed's only value-dependent region is **SOFR (curve 0), knots 6..11, 12Y..30Y**, with the join at 10Y. Every other curve is linear.

**Committed-state differences.**

| from x(q0) to | knot move | Hyman predicates that change |
|---|---|---|
| **x(V1)** (5 steps, **1 refresh**/tick) | \|dx\|∞ 2.0e-4 at knot 54, **not** on SOFR | **Real kinks:** node 5 (t = 25Y, tangent over [20Y, 30Y]) **clamp `M < \|m\|` false → true**, margin −1.58e-4 → +2.54e-5; node 6 (30Y end, [25Y, 30Y]) **end sign `m·S > 0` true → false**, margin +6.33e-9 → −2.40e-10 (the end tangent is zeroed); node 2 (12Y, [10Y, 15Y]) min-switch `\|pu\| < \|pm\|` true → false, +1.99e-5 → −3.49e-5. **Degenerate ties:** nodes 3, 4, 5 monotone-run tests with margins **2e-20 / 3e-23** at x(q0) |
| **x(V4)** (2 steps, **0 refreshes**) | \|dx\|∞ 3.0e-5 at knot 10 (SOFR) | **only the same degenerate ties** at nodes 3, 4, 5 (margins 1e-19..1e-22 → −7e-10), plus predicates that stop being evaluated below them. No clamp or sign kink changes |

**Per iterate of one V1 tick** (identical on the repeat tick):

```
eval 1  x(q0)                                   (no change)
step 1  |dx| 4.69e-3
eval 2  node 2 min-switch, node 5 CLAMP on, node 6 END-SIGN off, + the node 3-5 ties     (14 tape guards change)
step 2  |dx| 1.31e-4
eval 3  node 5 clamp OFF (+6.0e-5 -> -2.2e-5), node 6 end-sign ON (-1.7e-10 -> +4.1e-10)  (4 tape guards)
        kink 2-cycle: half step ; adaptive rho = 1.000 -> REFRESH
J eval  node 5 clamp ON (-2.2e-5 -> +1.9e-5), node 6 end-sign OFF (+4.1e-10 -> -2.0e-10)   <- THE REFRESH is taken exactly here
step 4, step 5: no further change -> converged
```

**Answers.**
1. **Hypothesis confirmed.** V1's alternating per-row noise moves SOFR's 25Y/30Y tangents across two genuine Hyman kinks: the 25Y clamp
   activates, and the 30Y end tangent loses sign agreement with the last secant and is zeroed. That is a **Jacobian jump**: the rows
   reading 20Y..30Y see a different linear map on each side, which explains the frozen map's spectral radius 1.365 after a 2e-4 knot move.
2. **The flip coincides with the refresh.** The same two predicates alternate on iterates 2 → 3 → the half-step Jacobian point. The refresh
   is taken at a point on the V1 side; after it, no predicate changes and the tick converges in two steps.
3. **V4 crosses only degenerate ties.** At x(q0) the monotone-run products `(S_i − S_{i−1})(S_{i+1} − S_i)` are ~1e-20, because the
   15Y/20Y/25Y/30Y knots are equally spaced and the fixture's x_true is linear in knot index, so consecutive secants are equal.
   Crossing such a tie switches which formula computes M, but **both arms are equal at the tie**, so value and Jacobian are continuous
   and the frozen map stays contractive (radius 0.000, 2 steps). A predicate flip is not automatically a Jacobian jump; the margin *and*
   the arm gap matter.
4. **The end-sign predicate at node 6 is hypersensitive.** Its margin is ~6e-9 in m·S units; a 0.1 bp noisy tick moves it by ~6e-9. Any
   noisy q_small (V0, V1) crosses it, and the fixture change to V1 will **not** remove desk_mixed's per-tick refresh.

**Implications for the active-set idea (3.4).**
- The exported mask should carry the **arm gap** at each flipped predicate, not only the flip. For example, at the node 5 clamp the gap is
  |m| − M, and at node 6 it is the zeroed tangent |m|.
- The streamer could then ignore tie flips (gap ≈ 0: V4's case) and treat only real kinks (node 5 clamp, node 6 end sign) as active
  constraints. Both quantities are by-products of the select kernel: the predicate value and the two arm values are computed anyway.

### 3.3 Studies (a)-(d) on the Hyman limiter — PENDING (`coarse/branch.cpp`, held behind the chain's timing runs)

Built and parity-wired, not yet run. Measures:
- **(a)** NaN/inf in the discarded arm on a flat region (m = 0, so `m/|m|` = 0/0), for {engine `m/|m|`, safe-arm, copysign} ×
  {bitwise select, arithmetic blend}, on value and tangent, plus the reverse-mode `where` trap (naive vs masked vs safe-arm adjoint).
- **(b)** select vs branchy vs guard replay vs re-record vs pattern lookup, and the break-even flip rate.
- **(c)** 4-lane SIMD select vs scalar vs 8 threads on a 4,096-scenario batch, and the per-tick thread hand-off latency.
- **(d)** mask/margin export cost, and agreement between mask changes and tape guard flips over 2,000 random 0.1 bp / 1 bp / 25 bp stresses.

