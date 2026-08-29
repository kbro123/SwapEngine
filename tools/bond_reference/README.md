# Bond math — independent reference oracles (beyond QuantLib)

We validate the bond kernels penny-perfect against QuantLib (`tests/bond_oracle_test.cpp`), but a single
oracle can hide a *shared* convention assumption. These are the independent cross-checks, ordered by how
much they add:

| Reference | What it is | How we use it | Independence |
|-----------|-----------|---------------|--------------|
| **Excel / OpenFormula `PRICE`/`YIELD`/`ACCRINT`** (basis 1 = Act/Act) | The spreadsheet street-convention formula (LibreOffice/Excel, an independent C++ implementation) | Reimplemented from its published formula in `tests/bond_reference_test.cpp` and checked against the kernel — a *different algebra* than our Horner evaluation | High (different code lineage, same convention) |
| **31 CFR Part 356 Appendix B** | The **official** US Treasury price/yield formula, incl. short/long first coupon & when-issued. It discounts the fractional first period by **simple** interest — a *different convention* from the street `v^w` our kernel implements | Reimplemented (regular + short-first) in `tests/bond_reference_test.cpp`, which pins the exact street↔Treasury relationship rather than equality | Authoritative — it *defines* the Treasury convention |
| **[Rateslib](https://rateslib.com)** (`calc_mode="us_gb"` **and** `"ust_31bii"`) | Independent Python fixed-income lib, Bloomberg-aligned. `us_gb` = the street convention we implement; `ust_31bii` (= `us_gb_tsy`) = the App B Treasury convention | `gen_golden.py` emits BOTH into `tests/golden/bond_reference.csv` with a `mode` column; `BondReference.ExternalGoldenIfPresent` asserts equality for `us_gb` and equality-after-the-convention-factor for `ust_31bii` (skips if absent) | High (independent implementation + Bloomberg-aligned) |
| **[FinancePy](https://github.com/domokane/FinancePy)** | Independent Python lib (`Bond.yield_to_maturity`, US street) | Alternative golden source — swap into `gen_golden.py` | High |
| **Bloomberg YAS / Tradeweb** | The market's own price/yield for a specific CUSIP | Manual spot-checks; paste a few (settle, ytm, clean) into the golden CSV | Gold standard (market truth) |

## Generating the external golden (Rateslib)

> **Licence.** Rateslib is **source-available, not open-source**. Without a registered commercial licence
> its dual licence permits **non-commercial use only** (at-home / academic). Running `gen_golden.py` is a
> use of the software, and the CSV it writes is derived from it. If this repository is ever used
> commercially, obtain a licence (<https://rateslib.com/licence>) or drop the Rateslib golden — the
> QuantLib oracle plus the QL-free Excel/OpenFormula and 31 CFR App B checks in
> `tests/bond_reference_test.cpp` need no third-party code. (An earlier version of this file and of
> `gen_golden.py` said "MIT". That was wrong.)

```bash
pip install rateslib            # tested with 2.7.1
python tools/bond_reference/gen_golden.py
ctest --test-dir build -R BondReference   # ExternalGoldenIfPresent now runs instead of skipping
```

The C++ test reads `tests/golden/bond_reference.csv` with columns
`mode,value_date,settle,issue,maturity,coupon,yield,clean,dirty,accrued` (dates ISO, coupon/yield decimals,
prices per unit notional; `mode` is the Rateslib `calc_mode` the row was priced under). The CSV is
intentionally NOT committed by default — it is machine-generated from
whichever external tool you trust; commit it if you want the pin to be part of the gate.

> Rateslib's method names have drifted across versions; if `gen_golden.py` errors on `bond.price(...)` /
> `bond.accrued(...)`, adjust those three calls to your installed version — the CSV schema is the contract.
> Rateslib >= 2 also requires `datetime`, not `date`, for schedule endpoints (`_dt()` in the script).
