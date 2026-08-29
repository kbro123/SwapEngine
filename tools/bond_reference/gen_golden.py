#!/usr/bin/env python3
"""Generate an EXTERNAL (non-QuantLib) golden reference for the bond kernels using Rateslib.

Rateslib (https://rateslib.com) is an independent fixed-income library whose US-Treasury calc modes
reprice the official Treasury math:
  - calc_mode="us_gb"      -> the US Treasury STREET convention: the fractional first period is discounted
                              by a COMPOUND factor, v^w. This is what our kernel implements (and what
                              QuantLib's BondFunctions with Compounded gives).
  - calc_mode="ust_31bii"  -> the 31 CFR Part 356 Appendix B TREASURY formula, which discounts the
                              fractional first period by SIMPLE interest, 1/(1 + w*y/f). The regulation is
                              explicit and uniform on this across all its sub-cases (regular / short-first
                              / long-first / reopened): every one is written "P[1 + (r/s)(i/2)] = ...".
It is deliberately designed to agree with Bloomberg. This makes it an ideal SECOND oracle beyond QuantLib.

BOTH modes are emitted, with a `mode` column. The C++ test asserts EQUALITY against us_gb and asserts the
ust_31bii rows equal our price times the exact convention factor (1+y/f)^w / (1 + w*y/f) -- so the
divergence is pinned as a KNOWN, understood convention difference rather than silently tolerated. See
docs/bond-pricing.md "Street vs Treasury (31 CFR App B) discounting".

This script writes tests/golden/bond_reference.csv, which tests/bond_reference_test.cpp loads (and skips if
absent). Regenerate whenever the bond set changes:

    pip install rateslib          # tested with rateslib 2.7.1
    python tools/bond_reference/gen_golden.py

LICENCE — READ BEFORE RUNNING. Rateslib is **source-available, not open-source**. It is distributed under
a dual licence: without a registered commercial licence, use is permitted for NON-COMMERCIAL purposes only
(at-home or academic). Running this script is a use of the software. The generated CSV is a table of bond
prices, but if this repository is ever used commercially, obtain a licence (https://rateslib.com/licence)
or drop the Rateslib golden and rely on the QuantLib oracle plus the QL-free Excel/OpenFormula and
31 CFR Part 356 App B reimplementations in tests/bond_reference_test.cpp, which need no third-party code.
(An earlier version of this header said "MIT". That was wrong.)

NOTE: Rateslib's method names have shifted slightly across versions (e.g. `price`/`ytm`/`accrued`). If your
installed version differs, adjust the three calls flagged below — the CSV schema is what the C++ test reads:
    mode,value_date,settle,issue,maturity,coupon,yield,clean,dirty,accrued
(all dates ISO; coupon/yield decimals; prices per unit notional, par = 1.0).
"""
import csv
import os
from datetime import date, datetime

try:
    from rateslib import FixedRateBond
except Exception as e:  # pragma: no cover
    raise SystemExit(f"rateslib not installed ({e}); run: pip install rateslib")

# (issue/dated, maturity, coupon%) — seasoned US Treasuries; extend freely.
BONDS = [
    (date(2019, 8, 15), date(2029, 8, 15), 2.5),
    (date(2019, 8, 15), date(2034, 2, 15), 5.0),
    (date(2014, 5, 15), date(2044, 5, 15), 3.375),
    (date(2021, 11, 15), date(2031, 11, 15), 1.375),
]
SETTLE = date(2024, 1, 16)   # a common settlement date
YIELDS = [2.0, 4.0, 6.0]     # percent
MODES = ["us_gb", "ust_31bii"]


def _dt(d: date) -> datetime:
    """rateslib >= 2 requires datetime, not date, for schedule endpoints."""
    return datetime(d.year, d.month, d.day)


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "golden", "bond_reference.csv")
    out = os.path.abspath(out)
    rows = []
    for mode in MODES:
        for issue, maturity, coupon in BONDS:
            bond = FixedRateBond(
                effective=_dt(issue),
                termination=_dt(maturity),
                frequency="S",
                convention="ActActICMA",
                calc_mode=mode,
                fixed_rate=coupon,
            )
            for y in YIELDS:
                # --- version-sensitive calls (adjust for your rateslib) ---
                clean = bond.price(ytm=y, settlement=_dt(SETTLE), dirty=False) / 100.0
                dirty = bond.price(ytm=y, settlement=_dt(SETTLE), dirty=True) / 100.0
                accrued = bond.accrued(_dt(SETTLE)) / 100.0
                # ----------------------------------------------------------
                rows.append([
                    mode,
                    SETTLE.isoformat(), SETTLE.isoformat(), issue.isoformat(), maturity.isoformat(),
                    f"{coupon / 100.0:.10f}", f"{y / 100.0:.10f}",
                    f"{clean:.12f}", f"{dirty:.12f}", f"{accrued:.12f}",
                ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "value_date", "settle", "issue", "maturity", "coupon", "yield",
                    "clean", "dirty", "accrued"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
