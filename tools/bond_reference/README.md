# External bond reference (`tests/golden/bond_reference.csv`)

An **independent, non-QuantLib** reference for the US Treasury **street** convention, checked by
`tests/bond_reference_test.cpp` section 3 at 1e-9 per unit notional. The CSV is **committed and required**: a missing
file fails the gate (PRINCIPLES.md P9). The engine never needs Java; only regenerating the CSV does.

| Source | Licence | Rows |
|---|---|---|
| [OpenGamma Strata](https://strata.opengamma.io) 2.12.56, `FixedCouponBondYieldConvention.US_STREET`, driven by `gen_golden.py` through JPype | Apache-2.0 | seasoned regular-period bonds, a when-issued new issue with a short (prorated) first coupon, bonds settling in their final coupon period |

Street = a compound fractional first period (`Q(v)·v^w`), simple once only the final coupon remains -- what
`build::us_treasury` / `us_treasury_wi` implement.

**Not covered here, deliberately:**
- 31 CFR Part 356 App B (the Treasury method): Strata has no such mode. It is pinned by `bond_reference_test.cpp`
  section 2 (the regulation reimplemented) and the QuantLib `SimpleThenCompounded` oracle (`tests/bond_oracle_test.cpp`).
- A LONG first coupon: the engine's builders reject it; support is a separate queued item.

**History.** Until 2026-09-14 this CSV came from Rateslib, whose licence permits non-commercial use only (LIC1). It was
replaced by Strata; the ust_31bii rows went with it.

## Regenerating

1. Fetch the jars listed in `strata_jars.sha256` from Maven Central into one directory (the file carries each jar's
   coordinates and SHA-256).
2. `python3 -m venv /tmp/jpype-venv && /tmp/jpype-venv/bin/pip install JPype1==1.5.2`
3. `/tmp/jpype-venv/bin/python tools/bond_reference/gen_golden.py --jars DIR [--jvm /path/to/libjvm.dylib]`

The script verifies every jar's SHA-256 and the Strata version before starting the JVM, then rewrites the CSV. Run
`ctest -R BondReference` afterwards and commit the CSV with the change that needed it.
