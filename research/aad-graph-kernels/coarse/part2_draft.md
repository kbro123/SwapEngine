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

