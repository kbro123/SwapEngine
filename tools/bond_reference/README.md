# Bond math — independent reference oracles (beyond QuantLib)

We validate the bond kernels penny-perfect against QuantLib (`tests/bond_oracle_test.cpp`), but a single
oracle can hide a *shared* convention assumption. These are the independent cross-checks, ordered by how
much they add:

| Reference | What it is | How we use it | Independence |
|-----------|-----------|---------------|--------------|
| **Excel / OpenFormula `PRICE`/`YIELD`/`ACCRINT`** (basis 1 = Act/Act) | The spreadsheet street-convention formula (LibreOffice/Excel, an independent C++ implementation) | Reimplemented from its published formula in `tests/bond_reference_test.cpp` and checked against the kernel — a *different algebra* than our Horner evaluation | High (different code lineage, same convention) |
| **31 CFR Part 356 Appendix B** | The **official** US Treasury price/yield formula, incl. short/long first coupon & when-issued | Reimplemented (regular + short-first) in `tests/bond_reference_test.cpp` | Authoritative — it *defines* the convention |
| **[Rateslib](https://rateslib.com)** (`calc_mode="ust_31bii"` / `"us_gb"`) | Independent Python fixed-income lib; `ust_31bii` reprices the CFR App B examples, designed to match Bloomberg | `gen_golden.py` emits `tests/golden/bond_reference.csv`; `BondReference.ExternalGoldenIfPresent` pins it (skips if absent) | High (independent implementation + Bloomberg-aligned) |
| **[FinancePy](https://github.com/domokane/FinancePy)** | Independent Python lib (`Bond.yield_to_maturity`, US street) | Alternative golden source — swap into `gen_golden.py` | High |
| **Bloomberg YAS / Tradeweb** | The market's own price/yield for a specific CUSIP | Manual spot-checks; paste a few (settle, ytm, clean) into the golden CSV | Gold standard (market truth) |

## Generating the external golden (Rateslib)

```bash
pip install rateslib            # >= 1.2
python tools/bond_reference/gen_golden.py
ctest --test-dir build -R BondReference   # ExternalGoldenIfPresent now runs instead of skipping
```

The C++ test reads `tests/golden/bond_reference.csv` with columns
`value_date,settle,issue,maturity,coupon,yield,clean,dirty,accrued` (dates ISO, coupon/yield decimals,
prices per unit notional). The CSV is intentionally NOT committed by default — it is machine-generated from
whichever external tool you trust; commit it if you want the pin to be part of the gate.

> Rateslib's method names have drifted across versions; if `gen_golden.py` errors on `bond.price(...)` /
> `bond.accrued(...)`, adjust those three calls to your installed version — the CSV schema is the contract.
