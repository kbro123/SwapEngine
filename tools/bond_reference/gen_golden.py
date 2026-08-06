#!/usr/bin/env python3
"""Generate an EXTERNAL (non-QuantLib) golden reference for the bond kernels using Rateslib.

Rateslib (https://rateslib.com, MIT) is an independent fixed-income library whose US-Treasury calc modes
reprice the official Treasury math:
  - calc_mode="ust_31bii"  -> the 31 CFR Part 356 Appendix B ("Treasury") formula, incl. the examples in
                              the regulation and when-issued / short-first-coupon handling;
  - calc_mode="us_gb"      -> the US Treasury street convention.
It is deliberately designed to agree with Bloomberg. This makes it an ideal SECOND oracle beyond QuantLib.

This script writes tests/golden/bond_reference.csv, which tests/bond_reference_test.cpp loads (and skips if
absent). Regenerate whenever the bond set changes:

    pip install rateslib          # tested with rateslib >= 1.2
    python tools/bond_reference/gen_golden.py

NOTE: Rateslib's method names have shifted slightly across versions (e.g. `price`/`ytm`/`accrued`). If your
installed version differs, adjust the three calls flagged below — the CSV schema is what the C++ test reads:
    value_date,settle,issue,maturity,coupon,yield,clean,dirty,accrued
(all dates ISO; coupon/yield decimals; prices per unit notional, par = 1.0).
"""
import csv
import os
from datetime import date

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


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "golden", "bond_reference.csv")
    out = os.path.abspath(out)
    rows = []
    for issue, maturity, coupon in BONDS:
        bond = FixedRateBond(
            effective=issue,
            termination=maturity,
            frequency="S",
            convention="ActActICMA",
            calc_mode="ust_31bii",   # the official 31 CFR App B Treasury method
            fixed_rate=coupon,
        )
        for y in YIELDS:
            # --- version-sensitive calls (adjust for your rateslib) ---
            clean = bond.price(ytm=y, settlement=SETTLE, dirty=False) / 100.0
            dirty = bond.price(ytm=y, settlement=SETTLE, dirty=True) / 100.0
            accrued = bond.accrued(SETTLE) / 100.0
            # ----------------------------------------------------------
            rows.append([
                SETTLE.isoformat(), SETTLE.isoformat(), issue.isoformat(), maturity.isoformat(),
                f"{coupon / 100.0:.10f}", f"{y / 100.0:.10f}",
                f"{clean:.12f}", f"{dirty:.12f}", f"{accrued:.12f}",
            ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value_date", "settle", "issue", "maturity", "coupon", "yield", "clean", "dirty", "accrued"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
