### 1.x Counts (no timing) — `coarse/run_counts.txt`, build `-ffp-contract=off`, 2026-09-15 15:17

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
