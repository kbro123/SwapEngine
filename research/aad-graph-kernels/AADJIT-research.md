# AAD-JIT: compile the pricing graph into the streaming hot path

Research note, 2026-09-15. Engine `SwapEngine` @ `55ab196` (branch e2-conventions), read-only. Paths are relative to
`SwapEngine/include/swaps/` unless marked otherwise. Probe code, outputs and generated kernels are in this directory:
`census.cpp`, `tape.cpp`, `run1.txt`, `run2.txt`, `run2_nofma.txt`, `run3_codegen.txt`, `gen_*.cpp`.

**Headline.** The idea works, and it works on this engine's code **as it is today**. The probe records the engine's own
templated `instrument_model_quote<Rec>`; no pricing code was rewritten. The recording is compiled to a flat, allocation-free
op list, and an "affine collapse" pass then **rediscovers the hand-built W-cache automatically**: on desk, 857 rows, the same
857 DF times the compiled engine registers.

| path | hybrid rungs (mixed_scheme, desk_mixed) | linear rungs |
|---|---|---|
| residual eval | **7-11x faster** than today's hybrid path | **1.5-2.5x slower** than the hand-written W-cache |
| value + Jacobian | **5-7x faster** than today's hybrid Jacobian | roughly break-even |

Record + collapse takes 1-25 ms per structure. Straight-line codegen through clang takes 1.4-7 s: fine ahead of time or as a
background tier, not on a structure change or a guard flip.

**Recommendation.** Build a pure-C++ staged "Plan" compiler (option d): an interpreter tier, plus an optional codegen/JIT
tier later.
- First target: retire the `AadBlock` and the hybrid split.
- Value-dependent branches become recorded **guards**, or better, explicit `select` primitives.
- Keep the Huber band and FX-log quote transforms outside the tape.
- Do not replace the linear W-cache until an optimised plan measurably matches it.

---

## 1. Current state, grounded in code

### 1.1 How the AD "graph" is represented today: there is no graph

- **There is no tape, node list or reverse mode anywhere in the engine.** A grep for `Tape|Adjoint|Node|record(|reverse` over
  `include/swaps` finds only comments. The one `struct Node` is regularizer quadrature (`calibration/regularize.hpp:161`).
- **AD is forward-mode vector duals only.** `ad/dual.hpp:13` defines `Dual = Eigen::AutoDiffScalar<VectorXd>`, so every
  intermediate heap-allocates its gradient.
  - `DualPooled<MaxW>` (`dual.hpp:33-34`) keeps the gradient in-object with capacity `kPooledMaxW = 64` (`dual.hpp:49`).
  - That limit is a documented *cliff*, not a gradient: above 64 touched knots every operation allocates again
    (`dual.hpp:39-48`).
  - `DualDir` (`dual.hpp:55`) is the width-1 directional variant.
- **The "graph" is implicit in C++ control flow.** It lives in the templated kernels, which are instantiated with `double`,
  `Dual` or `DualPooled`:
  - `calibration/problem.hpp:346-420` `instrument_model_quote`, `:435-452` `instrument_residual`;
  - `pricing/cashflows.hpp` leg kernels;
  - `pricing/curve_handle.hpp` virtual `CurveHandle<S>` (Outright / Spread / Turned);
  - `curve/curve_module.hpp` `ModularCurve<S>`;
  - `curve/regions.hpp` per-scheme `build()` / `integral()`.
- **Each derivative pass re-executes all of it:** virtual calls, `std::upper_bound` region searches, per-region `build()`
  with `std::vector` temporaries.

**The engine already does "record once, replay" in three ad-hoc places.** A generic plan compiler would subsume all three:
- `AadBlock::build_df_cache` (`calibration/aad_block.hpp:290-350`) replays pricing once through `RecordingCurve`
  (`aad_block.hpp:71-80`). It logs every `discount(t)` in order and replays them by cursor with an equality **guard**,
  `(*seq_t)[cursor] == t` (`aad_block.hpp:52-58`).
- The router probes each instrument through `MaxTimeProbe` (`calibration/hybrid_residual.hpp:49-59, 65-86`) to learn which
  curve times it reads.
- The compiled W is itself sampled from a Dual pass at setup (`pricing/compiled.hpp:35-70`, `integral_weight_matrix`) and
  keyed by `(curve, time)` registration (`pricing/compiled_book.hpp` `CompiledCurveSet::reg` / `finalize`, lines 31-86 of
  the de-commented listing).

### 1.2 The compiled kernel (linear-map curves)

`calibration/compiled_bundle.hpp` + `pricing/compiled_book.hpp`:
- **DF layer:** `DF = exp(-W x)`, done as block-sparse GEMVs per curve ancestry (`compiled_book.hpp` `df_into`) plus a shared
  reciprocal `INV = 1/DF` (`compiled_bundle.hpp:389-396`).
- **Batches by quote kind:** `BundleFloatBatch` (`pos`/`neg`/`rate`/`mtm`) and `BundleFixedLegs` (`compiled_bundle.hpp:504-505`).
  - Each is a hand-fused pointer loop over index arrays, e.g. `df[p]*(df[ss]*iv[se]-1)` (`compiled_book.hpp` `pv`/`num`).
  - Registration dispatches per quote kind in `register_at` (`compiled_bundle.hpp:430-498`).
- **Hand-derived Jacobian** (`jacobian_vs_into`, `compiled_bundle.hpp:253-383`):
  - quotient rule for ParRate/ParSpread/MtM (`:269-301`), with `d_pv_from_num` / `d_annuity` in `compiled_book.hpp:721-726, 987`;
  - futures `d_rate` (`compiled_book.hpp:808-810`);
  - FX two-entry rows (`:314-321`);
  - band slope scaling (`:327-333`);
  - moment direct terms (`:360-369`);
  - turn state pins (`:373-382`);
  - then a support-blocked, time-major `-(G·diag(DF))·W` product (`:334-356`).

**What is hand-written per instrument or scheme** (all of it must stay in sync with the templated reference via the T3
1e-12 parity tests):
- per-kind registration (`register_at`);
- per-kind value transforms (`model_rates`, `compiled_bundle.hpp:174-216`);
- per-kind analytic partials (above);
- the MtM product-of-DF partials (`add_mtm`, `compiled_book.hpp:387`);
- `band_residual_d` / `zero_coupon_transform_d` (`calibration/problem.hpp:228-234, 250-254`), double-maintained against
  their templated twins (`:219-224, 246-249`);
- the per-scheme linear maps (`integral_weight_matrix`, which refuses value-dependent regions at `compiled.hpp:53`);
- the router's "can the batch express this?" rules (`hybrid_residual.hpp:96-115`: compounded obs, seasoned/incomplete MtM,
  FX or ZC inside a Portfolio).

### 1.3 The hybrid kernel (value-dependent curves)

**The router.** `HybridBundleResidual` (`hybrid_residual.hpp:117-285`) partitions rows. A row is sent to the `AadBlock` when:
- it is non-cacheable (`:96-115`); or
- it reads past its curve's *linear horizon* (`pricing/curve_handle.hpp:198`, `curve_linear_horizon`; `hybrid_residual.hpp:128-148`).

**What the `AadBlock` does per eval** (`aad_block.hpp:183-212, 354-362`):
- `refresh_curves(x)` rebuilds **every** bundle curve through `BundleCurveSet::update` → virtual `set_forwards` → every
  region's `build()`;
- `MonotoneCubic::build` allocates 7 vectors and `Hermite` 3 (`regions.hpp:326-333, 446-469`);
- `TurnedCurve` adds an `x.head()` temporary (`curve_handle.hpp:112`).

**Per-eval pricing.** It then prices each block row with `instrument_residual<double>` through the virtual chain. Only
linear curves get the `CachedDisc` shortcut; non-linear curves and their dependants are excluded (`aad_block.hpp:314-323`).

**The refresh Jacobian** re-runs the whole thing on `DualPooled<64>` (`aad_block.hpp:248-263`).

**Costs** (audit A1 §2.2, A2 §1):
- mixed_scheme: 13 allocations per eval;
- desk_mixed: 23 allocations per eval, 54 per refresh;
- desk_mixed refresh ≈ 1.2 ms, of which the Jacobian is 777 µs.

**Census (this probe, `census.cpp`: `HybridBundleResidual` in isolation, load ~3.5):**

| rung | m×n | fixed cpns | float cpns | sub-periods | compiled DF times | `discount()` calls/eval | compiled rows | AAD width | `residuals_vs` µs | `jacobian_vs_into` µs |
|---|---|---|---|---|---|---|---|---|---|---|
| ois_nolag | 12×12 | 134 | 134 | 134 | 31 | 536 | 12 | 0 | 0.57 | 3.15 |
| averaged | 29×29 | 267 | 400 | 526 | 218 | 1,719 | 29 | 0 | 2.38 | 21.5 |
| mixed_scheme | 29×29 | 267 | 400 | 526 | 149 | 1,719 | 17 | 29 | 38.6 | 100.5 |
| fx_xccy | 37×37 | 636 | 1,372 | 1,372 | 637 | 6,986 | 37 | 0 | 6.61 | 71.1 |
| desk | 70×67 | 965 | 1,968 | 2,095 | 857 | 9,357 | 70 | 0 | 9.35 | 192 |
| desk_mixed | 70×67 | 965 | 1,968 | 2,095 | 420 | 9,357 | 52 | 55 | 137.6 | 744 |
| averaged_leg | 24×24 | 268 | 268 | 33,722 | 7,581 | 67,980 | 24 | 0 | 124.9 | 969 |

### 1.4 Branches, and which ones break "record once, replay forever"

| branch | where | depends on | verdict |
|---|---|---|---|
| region lookup `seg(t)` (upper_bound) | `regions.hpp:99,173,287,395,524,1078`; `ModularCurve::locate` `curve_module.hpp:218` | query time `t` (a schedule constant) | **fixed per structure**: fold at record |
| pre/post-segment `t <= xs_.front()`, `t >= xs_.back()` | every region's `forward/integral` (e.g. `regions.hpp:384-389, 513-518`) | knot / join times (constants) | fixed |
| `integral(t<=0) == 0` | `curve_module.hpp:206-209` | time | fixed |
| turn windows `t >= start && t < end` | `curve_handle.hpp:96` | time | fixed |
| Tension series vs closed form `ax < 0.5` | `regions.hpp:837,853,871` | σ·h (constants) | fixed (the probe records **0 guards** on `all_schemes`) |
| coupon shape (compounded, plain fast path, fixings) | `cashflows.hpp:250,291,304` | observation structure | fixed |
| MtM exchange settled `e >= 0`, `reset < 0` | `cashflows.hpp:367,374` | times | fixed |
| router partition / linear horizon | `hybrid_residual.hpp:128-148`, `curve_handle.hpp:198` | structure | fixed (and **unnecessary** under a plan compiler) |
| **Hyman filter**: `m·S > 0`, `smin`/`smax`, `abs`, `correction != m[i]` | `regions.hpp:544-589` (`:554,558,564,586`) | **knot values x** | **genuinely value-dependent per tick** |
| **Huber band** `q > upper`, `q < lower` | `problem.hpp:219-224` (templated), `:228-241` (double / slope) | model quote vs live band | **value-dependent per tick**, already tracked by the streamer (`streaming.hpp:530-654`: `side_of`, `breakpoint`, `apply_pins`, `track_bands` → `rescale_row`) |
| FX residual `log F − log q` | `problem.hpp:436-441` | no branch, but depends on live q | keep outside the tape (q is a per-tick input) |
| `CachedDisc` cursor guard | `aad_block.hpp:53` | query order | fixed (an existing replay guard) |
| DualPooled width `touched ≤ 64` | `aad_block.hpp:160`, `dual.hpp:49` | structure | a cliff that a tape removes (reverse mode does not depend on width) |

**Measured guard load** (`tape.cpp`):
- A MonotoneCubic region of 7 nodes (the ladder's 12Y..30Y back end plus the join) records **78 guards**.
- Every linear scheme records **0**.
- Guard flips, starting from `x_true`:
  - a 0.1 bp knot move flips **0/78**;
  - a 25 bp maturity-tilted parallel move flips **0/78**;
  - a 25 bp `sin`-shaped knot perturbation (a stress, not a market move) flips **21/78**.
- Audit A1 F1 infers that desk_mixed's streamer ping-pongs between **two** Hyman patterns mid-walk. So flips *will* occur
  near kinks, but they revisit a small set of patterns.

---

## 2. Survey of alternatives

Licences are from memory unless a source is cited; **verify before any dependency decision.**

### a. Tape replay: a flat op array interpreted by a switch loop

**The tools.**
- XAD, Adept 2 and dco/c++ are *Jacobian-taping* reverse tools: they store partials. CoDiPack offers both Jacobian and
  *primal-value* tapes ([CoDiPack taping strategies](https://www.scicomp.uni-kl.de/codi/d3/d30/TapingStrategy.html),
  [TOMS paper](https://dl.acm.org/doi/fullHtml/10.1145/3356900)).
- A Jacobian tape stores numbers valid only at the recorded x. **It cannot be replayed at a new x**; you re-record every eval,
  which costs roughly what today's Dual pass costs.
- Only a **primal-value / op-code tape** (CoDiPack primal, ADOL-C traces) can be re-evaluated at new inputs. That is the
  variant this idea needs.
- ADOL-C is the precedent for guards: its forward re-evaluation reports when a recorded comparison changes outcome, which
  signals "retape". `condassign` gives branch-free selects.

**How they fare here.**
- *Value-dependent branches:* none of the reverse tools handles them natively. You need guard + re-tape, or explicit
  `condassign`-style selects.
- *Allocation:* zero at replay once the arrays are sized (measured in the probe).
- *SIMD:* none, beyond what the compiler does to a switch loop. Dispatch costs about **2.4-3 ns/op** on this Xeon W-3223
  (desk: 15.3k ops in 41-45 µs).
- *Compile latency:* record + CSE + DCE is 1-20 ms.
- *Jacobian:* reverse sweeps over per-row cones, or forward-vector.
- *Maintenance:* the lowest of all options.

**Licences.** Adept 2 is Apache-2.0. CoDiPack is GPL-3. XAD is AGPL-3 with a commercial option. dco/c++ is commercial (NAG).
Stan Math is BSD-3, but reverse-only, with no replay. None gives more than we can build in ~1k lines, and GPL/AGPL is awkward
for a commercial SDK.

### b. Code generation from the recorded graph

**Emit C++ per structure and compile it.** The probe emitted straight-line C++ with hex-float constants and guard checks, and
loaded it with `dlopen`.

| rung | lines | clang `-O1` | clang `-O2` | eval (interpreted tape → generated) | parity |
|---|---|---|---|---|---|
| mixed_scheme | 4.2k | 1.36 s | 1.67 s | 13.9 µs → **3.7 µs** | bit-identical to the interpreter |
| desk_mixed | 15.4k | 6.1 s | 7.0 s | 45 µs → **14.8 µs** | bit-identical to the interpreter |

- **AOT at build time** suits fixed bench/regression structures, but user bundles are dynamic.
- **Runtime clang** fits a background tier (as JVMs do), never the tick path.
- **libtcc** compiles about 100x faster, but its code quality is poor and it is LGPL.

**CasADi** (SX graphs → C with sparse Jacobians; LGPL-3):
- It is the mature form of exactly this: a symbolic graph with forward/reverse AD and sparsity patterns, exported as C.
- Known pain: large SX graphs give huge C files and slow compiles
  ([casadi-users](https://groups.google.com/g/casadi-users/c/JT73r_cYMkk/m/I2Z1sHbDDQAJ), [docs](https://web.casadi.org/docs/)).
- Branches must be expressed as `if_else` selects.
- It would mean re-expressing pricing in CasADi instead of our templates. Wrong fit, good design reference.

**Enzyme** (LLVM-level AD, Apache-2 with LLVM exception; [repo](https://github.com/EnzymeAD/enzyme)):
- It differentiates *compiled* C++, control flow included, so value branches are handled natively, and it has vector forward
  mode ([EuroLLVM slides](https://llvm.org/devmtg/2022-05/slides/2022EuroLLVM-AutomatedBatchingandDifferentiationofScalarCodeinEnzyme.pdf)).
- But it differentiates the **generic** kernel: virtual calls, `upper_bound`, region rebuilds. It does not specialise on a
  bundle's structure unless you also partially evaluate on it, which needs a JIT anyway.
- It needs a custom clang plugin toolchain on macOS and Linux. It would help the analytic-Jacobian maintenance problem more
  than the speed problem.

**JAX/XLA tracing.** Conceptually identical to our recorder: concrete structure is traced, and value branches must be
`lax.cond` / `select`. Not usable in a C++ hot path, but its "trace, then specialise, then cache by shape" model is the right
mental model.

### c. JIT to machine code at runtime

**MatLogica AADC** (commercial) — the finance-specific "record once, JIT kernel" product:
- Active types `idouble` / `ibool`; comparisons return `ibool`.
- **Static branches** (independent of kernel inputs) are baked in at record time. **Stochastic branches** must be written
  with `iIf()` / `condAssign()`, and missed branches are flagged by tooling.
- Claims millisecond JIT compiles (not LLVM), AVX2/AVX512, "adjoint factor < 1", and thread-safe, serialisable kernels
  ([FAQ](https://matlogica.com/resources/faq/), [technology](https://matlogica.com/How-MatLogica-AAD-Works.php)).
- Platforms listed: Intel/AMD CPUs. No ARM is mentioned, which matters for Apple Silicon dev boxes.
- It is the proof that this architecture is production-grade in finance. Cost: licence, lock-in (they say "no lock-in"),
  and an x86-only story.

**XAD JIT** ([docs](https://auto-differentiation.github.io/)):
- Record into a `JITGraph`; `ABool::If(cond, a, b)` records an `If` node "allowing the branch to vary at runtime"
  ([ABool](https://auto-differentiation.github.io/ref/jit-abool/)).
- The native backend `xad-codegen` (x86-64, AVX2) is under a **separate commercial licence** and claims 2-5x over the tape.

**Hand-rolled emitters.**
- asmjit (Zlib) and xbyak (BSD-3, x86; a separate aarch64 port) give sub-millisecond compiles for about 20k ops. You write
  register allocation and op emission yourself, and must keep separate x86 and ARM backends.
- MIR (MIT) is a light JIT with a C frontend and ~ms compiles, and is portable.
- LLVM ORC produces the best code but is a heavy dependency, with tens to hundreds of ms per module.

### d. Pure C++ staged "Plan": the probe, and the recommended base

An operator-overloading recorder (`aj::Rec`) instantiates the engine's **unchanged** templates. Passes:
- constant folding (both operands constant → no node);
- CSE by `(op, a, b)` — 85-89 % of recorded ops are duplicates, because the same `discount(t)` is asked by many coupons and
  rows;
- DCE;
- renumbering into `[consts | inputs | ops]`;
- per-output cones;
- the affine collapse (§3.2).

Branches become guards; allocation is 0 by construction and measured 0. There is no SIMD yet. Latency is 1-25 ms. Jacobian:
reverse over each row's nonlinear cone, then a scatter through W. Maintenance: new pricing maths is automatically covered; the
optimiser passes are generic.

Upgrades available without a JIT:
- reciprocal sharing (`a/b` → `a·inv(b)` with `inv` CSE'd);
- superinstructions (`exp(-row)`, `df·(df·inv − 1)`);
- op-type levelisation (group same-op nodes at equal depth into gathered SIMD loops, which is how the hand-written batches win).

### e. Sparse Jacobian structure

- **Per-curve blocks are dense-ish.** A par swap reads its curve up to maturity, and Hermite, NaturalCubic and MonotoneCubic
  couple neighbouring or all region knots; spread chains add the base's knots. Column colouring (Curtis-Powell-Reid) buys
  little *within* a curve, and the across-curve block structure is already exploited.
- **Forward-vector mode costs L·n.** Measured: desk 549-648 µs, worse than the engine's 192-264 µs.
- **Reverse per row over the full op tape costs Σ cone sizes.** Measured: desk Σ cone = 71.8k ops ≈ 4.7 L, 252-388 µs.
- **The structural win is two-level:** J = (∂q/∂DF-ish nonlinear layer, reverse over small cones) × W (sparse). That is the
  engine's `-(G·diag(DF))·W` identity, and the collapsed kernel derives it generically: desk fwd+J **111-116 µs** vs engine
  J 192-264 µs.
- The nonlinear cones are small (a coupon's exp/mul/div chain), so row-wise reverse is the right mode here. Vector modes only
  pay at width ≈ colour count, and that is not small for these blocks.

### f. Handling value-dependent branches

1. **Guard-and-re-record** (ADOL-C style, and what the probe does):
   - O(#guards) comparisons per eval (78 for a MonotoneCubic curve: negligible).
   - On a flip, fall back to the templated path for that eval, then re-record (1-25 ms) and switch.
   - Risk: flip storms at a Hyman kink. A1 F1 already sees a 2-pattern ping-pong.
2. **Pattern cache:** key compiled plans by the guard-outcome bitset (78 bits per MonotoneCubic curve → a hash) and keep an
   LRU of ~4-8. Ping-pong between two patterns then costs one hash compare. This is the generic form of the "per-Hyman-pattern
   W" spike.
3. **Branch-free `select` ops** (AADC `iIf`, XAD `ABool::If`, CasADi `if_else`):
   - Rewrite the ~12 value branches in `hyman_filter` (`regions.hpp:544-589`) as `select(cond, a, b)` on the Scalar type.
     With `double`, `select` is a ternary, so bits are unchanged.
   - The tape then stays valid for every x: **no guards, no flips**. `select` is nonlinear, so the collapse stops at it.
   - The MonotoneCubic region becomes a small select network over its 7 tangents, feeding affine rows over
     `[x, tangents]`. That is a two-level plan with one extra layer, not a per-pattern W zoo.
   - The Jacobian through a `select` is exactly AAD's one-sided derivative today.
   - **This is the preferred end state.** Guards remain as the safety net for any branch not yet converted: a guard fires
     → fallback + re-record + log.
4. **Band edges (Huber):** keep them **out of the tape**. The tape maps x → model quotes; the per-row transform r(q_model, q)
   and its slope stay in the streamer's existing active-set machinery (`streaming.hpp:530-654`), exactly as the compiled path
   does today (`compiled_bundle.hpp:224-236, 327-333`).
   - Recording bands would create flips at every edge crossing, and the streamer already turns those into rank-one rescales.
   - The FX log and ZC transforms, which need the live q, also stay as a per-row tail.

---

## 3. Fit to this engine

### 3.1 What the graph captures, and what folds

Graph: inputs x (all knots and turn δ's) → region `build()` (secants, Bessel/spline tangents, Thomas solve, Hyman guards,
Horner coefficients, cumulative integrals) → `integral(t)` at every queried t → `exp` → leg sums (DF ratios, accrual
weights, MtM notional products, compounding products) → quote quotients → model quote per row.

Folded at record time without any special code:
- all dates, accruals, day-count fractions, `u = t − xs_[i]` offsets, segment indices (`seg`) and region selection;
- the `h` arrays and tridiagonal bands (these are `double` in the templates);
- turn overlaps, FX spot scales and portfolio weights.

CSE shares every `integral(t)` / `exp` across the coupons, legs and rows that read the same `(curve, t)`. That is the DF cache
and the `reg()` registration, derived rather than written.

### 3.2 Op-list size and the affine collapse (measured, `run3_codegen.txt`)

| rung | recorded ops | live after CSE+DCE | CSE hits | guards | affine rows (nnz) | nonlinear ops | record + collapse |
|---|---|---|---|---|---|---|---|
| ois_nolag | 7,468 | 815 | 89 % | 0 | 30 (280) | 220 | 0.7 + 0.1 ms |
| averaged | 27,568 | 4,072 | 85 % | 0 | 217 (1,936) | 938 | 3.7 + 0.8 ms |
| mixed_scheme | 27,370 | 4,154 | 85 % | 78 | 248 (2,127) | 967 | 3.5 + 1.2 ms |
| fx_xccy | 90,451 | 10,629 | 88 % | 0 | 637 (6,422) | 3,459 | 11.3 + 3.2 ms |
| desk | 141,813 | 15,269 | 89 % | 0 | **857** (9,195) | 4,537 | 17.3 + 7.5 ms |
| desk_mixed | 141,516 | 15,351 | 89 % | 78 | 888 (9,548) | 4,566 | 14.4 + 5.8 ms |

**The affine collapse.** Under fixed guards, every node built from inputs by `+`, `−`, negation, `·const` and `/const` is
affine. Nodes read by a nonlinear op, a guard or an output become CSR rows. On desk this yields exactly 857 rows, the compiled
engine's `n_times()`: **the W-cache emerges from the tape.** On mixed_scheme and desk_mixed the MonotoneCubic region
collapses too, under its guard pattern. That is the per-Hyman-pattern W, obtained with zero scheme-specific code.

**Memory.** Desk's live tape is ~15k slots (122 KB) + 15k × 12-byte ops (183 KB), inside L2 (1 MB). The collapsed kernel is
smaller: 4.5k ops + 9.5k nnz.

**Not probed:**
- `averaged_leg` (33.7k sub-periods → roughly 100k+ live ops);
- the `averaged_leg_moment` quadrature path;
- `portfolio`, `banded`, `zero_coupon`, `basis_spread`, `ois_lag`.

The recorder instantiated the whole ladder's templates without error, so these are expected to work but are unmeasured.

### 3.3 Speed: can it match the hand-written compiled path, and beat the hybrid?

**Timing method.** Median of 7 batches, same process. **Machine load averages were 3.5-11** (another gate was compiling), so
treat every number as ±30 %. Ranges below are across three runs (`run2`, `run2_nofma`, `run3`).

**Residual eval.**
- `eng` = `HybridBundleResidual::residuals_vs`, which includes the band/FX tail the tape omits (a few ns per row).
- `interp` = the interpreted full tape; `collapsed` = sparse GEMV + nonlinear op interpreter; `codegen` = generated
  straight-line C++ from the uncollapsed tape.

| rung | eng µs | interp µs | collapsed µs | codegen `-O2` µs | verdict |
|---|---|---|---|---|---|
| ois_nolag | 0.49-0.82 | 1.8-3.2 | 0.79-1.34 | — | loses 1.0-1.6x |
| ibor_multicurve | 1.6 | 6.9-8.1 | 3.4-4.1 | — | loses ~2.3x |
| averaged | 2.3-3.3 | 10.5-13.7 | 4.1-6.7 | — | loses ~1.8x |
| all_schemes | 0.95-1.5 | 4.3-6.9 | 1.7-1.9 | — | loses ~1.6x |
| turns | 0.59-0.66 | 3.0-3.9 | 1.3-1.9 | — | loses ~2.5x |
| fx_xccy | 6.5-9.5 | 26-33 | 14.6-15.6 | — | loses ~1.8x |
| desk | 9.3-11.6 | 37-44 | 19.5-22.7 | — | loses ~2x |
| **mixed_scheme** | **39-50** | 10-16 | **4.4-5.0** | 3.7 (uncollapsed) | **wins 9-11x** |
| **desk_mixed** | **141-175** | 39-45 | **20.6-22.7** | 14.8 (uncollapsed) | **wins 7-8x** |

**Value + Jacobian.** The engine column times `jacobian_vs_into` **alone**; the collapsed column times *forward + reverse*.

| rung | engine J µs | collapsed fwd+J µs |
|---|---|---|
| ois_nolag | 3.0-5.2 | 5.0-8.6 (loses) |
| ibor_multicurve | 16.6-30.6 | 20.7-32.4 (≈) |
| averaged | 27.6-36.9 | 21.6-25.2 (wins) |
| fx_xccy | 72-80 | 70-72 (≈) |
| desk | 195-264 | **111-116** (wins ~2x) |
| mixed_scheme | 93-187 | **23-35** (wins 4-5x) |
| desk_mixed | 786-1,156 | **111-168** (wins 5-7x) |

**Estimated streaming ticks** (INFERRED: eval counts from A1, not measured end to end):
- **mixed_scheme** StreamTick today is 77 µs ≈ 2 evals × ~38 µs + ~1 µs of bookkeeping. With the collapsed kernel,
  2 × ~4.7 + ~1 ≈ **10-11 µs**. That matches A1 F2's "~10 µs class" estimate for a per-pattern W, with no hand-built W.
- **desk_mixed** small tick today is 1.88 ms. Per A1 F1 that is ≈ 5 evals + 1 refresh per tick. With the collapsed kernel:
  5 × 21 µs + J ~130 µs + COD (unchanged, a few hundred µs at 70×67) ≈ **0.4-0.6 ms**. Fixing the F1 refresh-every-tick
  schedule is separate and still worth doing.
- **Linear rungs** would *regress* if switched today: ois_nolag ~1.3 → ~1.8 µs tick; desk ~33 → ~60 µs.

**Where the plan loses on linear rungs, and what closes it.**
1. **Interpreter dispatch, ~2.5 ns/op × nonlinear ops** (desk: 4.5k ops ≈ 11 µs). Close with superinstructions, levelised
   SIMD batches, or codegen/JIT: codegen already cut the *uncollapsed* desk_mixed tape 3x, from 45 to 14.8 µs.
2. **Divides:** the tape keeps `DF(s)/DF(e)` as one divide per coupon; the engine uses the shared `INV` (see
   `compiled_book.hpp` NOTE on reciprocals). A reciprocal-sharing pass closes this, at 1-ulp cost.
3. **CSR row GEMV vs contiguous per-curve block GEMV** (`df_into`). Close with block-aware row storage: rows of one curve
   share a column span.
4. **The hand-written batches exploit coupon homogeneity** (`cpn_is_plain`, `sub_is_identity`). The plan sees only ops, and
   levelisation would recover this.

**Honest estimate.** A collapsed plan with (2) + (3) + levelised loops should land within ~1.0-1.5x of the W-cache on linear
rungs. Matching or beating it plausibly needs a JIT tier. **None of that is demonstrated here.**

**Where it clearly wins:**
- anything value-dependent (Hyman), compounded (today AAD-only), a seasoned MtM, FX or ZC inside a Portfolio — every
  `instrument_is_noncacheable` case;
- any future product (the "specifics as data" North Star);
- Jacobians at desk scale;
- the DualPooled width cliff, which disappears.

---

## 4. Probe details (`tape.cpp`, ~750 lines, research only)

**Recorder.**
- `aj::Rec {double v; int id}` with non-template free operators, `exp`/`log`/`sqrt` via ADL, and an Eigen `NumTraits`
  specialisation.
- Comparisons on active values push a guard; `abs` records a sign guard and returns `x` or `−x`, which is bit-identical and
  keeps the result affine.
- The engine's `build_bundle_curves<Rec>` + `instrument_model_quote<Rec>` compiled **unmodified**. The only probe bug was a
  `dlsym` cast.

**Parity.**
- **Bit-identity needs `-ffp-contract=off`.** Built that way, interpreted-tape quotes equal `instrument_model_quote<double>`
  at a new x **exactly (0.0) on all 9 probed rungs** (`run2_nofma.txt`).
- Under the engine's flags (`-O3 -march=x86-64-v3`, clang default `-ffp-contract=on`), the *templated double path* gets
  compiler-chosen FMAs, and the tape differs by up to 2.5e-15 on some rungs.
- So a "bit-identical to the templated kernel" gate needs FMA contraction off (or `#pragma STDC FP_CONTRACT OFF`) in the
  reference TU. This also affects today's T3 compiled-vs-templated parity at the ulp level.
- Codegen kernels are bit-identical to the interpreter (`max|d| = 0`). The collapsed kernel sits at rounding level: quotes
  ≤ 2.5e-15, J ≤ 8e-15 (rounding-level: sums are reassociated).
- The Jacobian matches the engine's `jacobian_vs_into` to ≤ 1e-14 on every rung, after applying the band slope and FX
  1/(F·T) row factors outside the tape.

**Allocation.** 0 over 50 forward+J replays, interpreted and collapsed, on every rung (`AllocScope` from
`bench/fixtures/malloc_count.hpp`).

**Guards.** 78 on a MonotoneCubic curve and 0 elsewhere. Flips 0 / 0 / 21 for the 0.1 bp, 25 bp tilt and 25 bp sin moves.

**Caveats.**
- Load 3.5-11 during runs; single machine (Xeon W-3223, AVX2 + AVX-512 capable, built for x86-64-v3); no pinning.
- The tape omits the band/FX tail (cheap).
- "Tick" numbers are inferred from eval counts, not measured in the streamer.
- The Jacobian-reverse implementation is a first cut: no cone merging across rows sharing a curve layer.

---

## 5. Recommendation: a staged path

**Stage 0 — hygiene that pays regardless** (bit-identical, days):
- A1 F5 / A2 A3: kill the region `build()` temporaries (member scratch).
- A1 F4: `AadBlock` rebuilds only the curves it reads.
- These still matter for cold LM, risk/Dual paths and recording itself, and they reduce hybrid allocations before the plan
  exists.
- Separately, decide the FMA policy for parity gates (`-ffp-contract=off` on reference TUs).

**Stage 1 — spike: `calibration/plan/` (header-only, test-only, ~1.5k lines)** (1-2 weeks):
- Recorder scalar, `Plan` (CSE / DCE / const-fold / renumber), guard table, cones, affine collapse, interpreter,
  `jacobian_reverse`.
- Must instantiate the full ladder. Gates:
  - (G1) interpreted plan == templated double kernel **bit-for-bit** with FMA off, over all 16 rungs × {x_true, 0.1 bp,
    25 bp};
  - (G2) collapsed plan vs `HybridBundleResidual` quotes ≤ 1e-12 and J ≤ 1e-10 (T3's existing tolerances);
  - (G3) 0 allocations per replay (T4 malloc guard);
  - (G4) record + collapse ≤ 50 ms on desk.

**Stage 2 — prototype on the hybrid rungs** (1-2 weeks):
- `PlanResidual` implementing the `ResidualEngine` interface (`residuals_vs`, `jacobian_vs_into`, `model_rates`,
  `set_quote(s)`): x → quotes via the plan, then the existing per-row band/FX/ZC tail.
- Route **only bundles that today have a non-empty `AadBlock`** to it.
- Guard flip → evaluate this call on the templated double path (always correct), re-record in place, keep a pattern-keyed
  LRU (~8). Report `plan_rerecords` in `StreamTick` telemetry.
- Gates:
  - ladder StreamTick / RefreshTick on mixed_scheme and desk_mixed improve ≥ 3x;
  - hotpath_shapes parity unchanged;
  - near-singular / kink-cycle repro tests (`streaming_near_singular_repro_test.cpp`, `streaming_kink_cycle_repro_test.cpp`)
    converge to the same committed x within 1e-10;
  - a flip-storm test (a 0.1 bp oscillation straddling a Hyman kink) shows ≤ 2 re-records then 0.

**Stage 3 — remove the guards where it matters** (days):
- Add a `select(cond,a,b)` customisation point in `curve/`: ternary for `double`/`Dual`, a recorded op for the plan scalar.
- Rewrite `hyman_filter` (`regions.hpp:544-589`) with it. For `double` and `Dual` it is bit-identical to today, and the
  MonotoneCubic AAD-vs-FD test (axis-3 M3) should land first.
- Extend the collapse to a two-level plan (a select network over intermediates → affine rows over `[x, intermediates]`).
- After this, flip storms are impossible by construction; guards remain only as an assertion net.

**Stage 4 — migration decision for the linear rungs** (measure first):
- Add reciprocal sharing, block-aware rows and op levelisation. Optionally a codegen tier: AOT for the ladder, background
  clang or asmjit/MIR for user bundles, with the interpreter always available as the tier-0 fallback.
- Replace `CompiledBundleResidual` **only if** the ladder's linear rungs are within 5 % of today. Otherwise keep the W-cache
  as a fast tier the router selects when the plan has no guards and no nonlinear layer above the DF exp.
- Either way the *hand-written Jacobian* can go first: plan fwd+J already beats `jacobian_vs_into` at desk scale.

**What this retires, eventually:**
- `AadBlock` + `CachedDisc` / `RecordingCurve` / `MaxTimeProbe`;
- the per-row router and linear horizons (`hybrid_residual.hpp:27-153`);
- `instrument_is_noncacheable`;
- `DualPooled` on the hot path, and the `kPooledMaxW` cliff;
- the hand-derived partials in `compiled_bundle.hpp:253-383`, `compiled_book.hpp` `d_*`, and `band_residual_d` /
  `zero_coupon_transform_d` twins, if the tail is also taken from the plan;
- if Stage 4 wins, `register_at` and the batch layer.

The templated kernels become the **single source of truth** for pricing maths; the hot path is derived from them.

**Risks.**
- **Compile latency on structure change:** recording is 1-25 ms, so it is fine for rebinds. Clang codegen is 1.4-7 s, so
  background only. AADC-class ms JITs need asmjit/MIR work.
- **Debuggability:** a failing plan row needs a "which template call recorded this op" map. Keep an optional
  `(row, coupon, curve, t)` provenance side-table at record time.
- **Branch-flip storms at kinks:** mitigated by the pattern LRU (Stage 2), eliminated by selects (Stage 3).
- **Band edges:** kept out of the tape by design, so no new flip source.
- **Silent missed branches:** a value branch written as `if (double(x) < …)` via `.v` bypasses guard recording. The recorder
  must not expose an implicit `double` conversion (the probe's `Rec` has none, and that made it compile-time safe). The
  templates must not call `.value()` on Scalar in pricing code.
- **Parity gates:** bit-identical only against a no-FMA templated reference; the collapsed and codegen tiers are
  rounding-level, and gates must say so (as T3 already does for `INV`).
- **Portability:** the interpreter is portable (macOS/Linux/ARM). Codegen via clang is portable but slow. asmjit and xbyak
  are x86-first. AADC is x86 and commercial. xad-codegen is x86, AVX2 and commercial.
- **Licences:** avoid GPL/AGPL (CoDiPack, XAD core) for the SDK; building in-house avoids all of this.

**Interaction with existing plan items.**
- **Per-Hyman-pattern W spike (A1 F2 fix B):** subsumed. The affine collapse under a guard pattern *is* that W, derived
  generically. Do the spike as Stage 1-2 rather than hand-building a frozen-clamp `integral_weight_matrix`.
- **Hybrid eval allocations (A1 F4/F5, A2 A3/A4: region-build temporaries, `AadBlock` rebuilding all curves):** the plan's
  hot path has neither (0 allocations, no curve rebuild; builds are CSE'd ops). Still do the cheap bit-identical F4/F5 fixes
  now (Stage 0), because recording, cold LM and the Dual risk paths keep using `build()`, and the plan's templated fallback
  on a guard flip pays them.
- **C6b explicit pseudo-inverse (C6-design §2c):** orthogonal. It lives in `StreamingCalibrator::factor`
  (`streaming.hpp:782-811`), which consumes J however it is produced. Once the plan cuts desk_mixed's J from ~0.8-1.2 ms to
  ~0.12-0.17 ms, the COD / solve becomes the dominant refresh cost, so C6b's *relative* value **rises**. No conflict; keep it
  on its own track.
- **The A1 F1 / A2 S1-S3 refresh-schedule fixes and A1 F3 free first step:** orthogonal and multiplicative. They cut evals and
  refreshes per tick; the plan cuts cost per eval or refresh.
