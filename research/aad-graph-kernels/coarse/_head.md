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

