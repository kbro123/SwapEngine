
### 3.3 Studies (a)-(d) on the Hyman limiter — PENDING (`coarse/branch.cpp`, held behind the chain's timing runs)

Built and parity-wired, not yet run. Measures:
- **(a)** NaN/inf in the discarded arm on a flat region (m = 0, so `m/|m|` = 0/0), for {engine `m/|m|`, safe-arm, copysign} ×
  {bitwise select, arithmetic blend}, on value and tangent, plus the reverse-mode `where` trap (naive vs masked vs safe-arm adjoint).
- **(b)** select vs branchy vs guard replay vs re-record vs pattern lookup, and the break-even flip rate.
- **(c)** 4-lane SIMD select vs scalar vs 8 threads on a 4,096-scenario batch, and the per-tick thread hand-off latency.
- **(d)** mask/margin export cost, and agreement between mask changes and tape guard flips over 2,000 random 0.1 bp / 1 bp / 25 bp stresses.

