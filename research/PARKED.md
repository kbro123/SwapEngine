# PARKED RESEARCH -- do not merge

**Branch:** `research/aad-graph-kernels`, cut from `e2-conventions` at 413900d on 2026-09-15.

The owner parked this for later investigation. It is research only: no file in this branch is built by the engine's CMake or gated by
`tools/verify.sh`. It must NOT be merged. Anything that graduates goes back through the main line's repro-first process and gates.

## aad-graph-kernels/ -- compiling the pricing graph into fixed kernels over data

**Question.** Record the engine's templated pricing once, then run residuals and Jacobians through pre-compiled kernels driven by data
(no JIT), so the hot path is generic rather than hand-written per instrument or scheme.

**AADJIT-research.md** (first pass: a scalar op list, one entry per operation)
- A recording scalar runs through the unmodified `instrument_model_quote` and `build_bundle_curves` templates.
- Constant folding and duplicate removal shrink desk from 141.8k recorded ops to 15.3k live ones.
- Replay allocates nothing and is bit-identical under `-ffp-contract=off`.
- Collapsing the linear parts rediscovers the W-cache (desk's 857 rows) and the per-Hyman-pattern W.
- It is faster on the hybrid rungs (mixed_scheme and desk_mixed) and slower on the linear rungs, because it dispatches one routine per operation.

**COARSE-research.md** (coarse kernels, a cashflow-table instrument IR, and branch handling)
- Part 1, coarse kernels:
  - Dispatch drops from 15,269 routine calls to 122-184 per evaluation on desk, with 0 allocations.
  - Bit-identical to the templated kernel on 5 rungs under `-ffp-contract=off`; within 6.6e-15 (values) and 3.9e-15 (Jacobian) under the engine's default flags.
- Part 2, the cashflow-table instrument IR:
  - 408 of 420 ladder rows (97.1 %) lower with no exception; the only exception is the 12 moment-path rows.
  - Exact against the templated kernel; within 1e-14 of the engine's model quotes.
  - No ladder rung exercises the compounded or weighted coupon modes, so those paths are untested.
- Part 3, branches:
  - Classifies every branch in the pricing code.
  - Compares compute-both-sides selects (preferred) with guards.
  - Includes the MonotoneCubic clamp-flip traces (guardtrace / guardtrace2).

**PENDING when parked (never measured):**
- Part 1 section 1.3: coarse kernel vs engine timing table, including whether it matches the hand-written W-cache on the linear rungs.
- Part 2: IR timings on 3 rungs.
- Part 3 section 3.3 (`coarse/branch.cpp`, the `branchprobe` binary):
  - a NaN in the discarded arm;
  - the cost of computing both arms and the break-even flip rate;
  - SIMD lanes vs threads;
  - mask bits vs guard flips.

The owner's later questions and my own assessment, which live in the session transcript and not in these files:
- No JIT is needed: data structures plus pre-compiled kernels are the plan.
- Compute-both-sides selects are the default. Watch the NaN in the unused arm, and use guards only for expensive arms.
- Options and CVA fit: XVA/CVA gains the most (the path axis becomes the batch axis). Discontinuous payoffs and adaptive numerics need extra design.

## desk-mixed-clamp/ -- step 3 diagnosis

The main line parked the fix as 3b. **STEP3-findings.md** covers:
- the 47 bp xccy 30Y knot swing, a fixture artefact fixed on the main line in 6b26213;
- the desk_mixed refresh on every tick: a Newton 2-cycle across SOFR's MonotoneCubic 25Y clamp and 30Y end-condition kinks (frozen-map spectral radius 1.365);
- the q_small variants V0-V4;
- the fix options, the leading one being a clamp active set fed by the kernel's mask bits and margins.

## Rebuilding the probes

Each probe is standalone and header-only. Compile it against the engine checkout, from the engine root:

```
xcrun -sdk macosx clang++ -std=c++20 -O3 -DNDEBUG -march=x86-64-v3 -fno-math-errno -w \
  -I include -I third_party/eigen -I build/generated -I bench/fixtures <probe>.cpp -o <probe>
```

- Add `-ffp-contract=off` for the bit-identity runs.
- The `coarse/*.sh` drivers hard-code the session scratchpad paths they ran from; the `run_*.txt` files are their recorded outputs.
- Generated sources (`gen_*.cpp`) and binaries are not included: the probes regenerate them.
