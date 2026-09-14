#!/usr/bin/env python3
"""Generate tests/golden/bond_reference.csv: an EXTERNAL, non-QuantLib reference for the US Treasury STREET convention,
priced with OpenGamma Strata 2.12.56 (Apache-2.0) through JPype.

WHY. tests/bond_reference_test.cpp section 3 pins build::us_treasury / us_treasury_wi to an independent fixed-income
library, so the street kernel is checked against something other than QuantLib and our own reimplementations. The CSV is
committed; this script only runs when the bond set changes. The engine's build and gate never need Java.

WHAT. Strata's FixedCouponBondYieldConvention.US_STREET (a compound fractional stub, SIMPLE once only the final coupon
remains) on ACT/ACT ICMA semiannual bonds:
  - seasoned bonds (regular periods) settling 2024-01-16;
  - a when-issued new issue with a SHORT, prorated first coupon (settle = the dated date);
  - bonds settling inside their FINAL coupon period.
Strata has no 31 CFR Part 356 App B mode, so no Treasury-method rows are emitted: App B stays pinned by
bond_reference_test.cpp section 2 and the QuantLib SimpleThenCompounded oracle (tests/bond_oracle_test.cpp). A LONG first
coupon is out of scope: the engine's builders reject it (queued as its own item).

LICENCE. Strata and its dependencies are Apache-2.0 (Strata, Guava, Joda-Beans/Convert, commons-math3), MIT (SLF4J) or
the CERN/LGPL-style Colt licence. They are used as a generator tool; no jar is committed or linked into the engine.

HOW. Python 3.8+ with JPype 1.5.2 and any Java 8+ runtime (libjvm):
    python3 -m venv /tmp/jpype-venv && /tmp/jpype-venv/bin/pip install JPype1==1.5.2
    /tmp/jpype-venv/bin/python tools/bond_reference/gen_golden.py --jars DIR [--jvm .../lib/server/libjvm.dylib]
DIR must hold exactly the jars listed in tools/bond_reference/strata_jars.sha256 (Maven Central coordinates in that file);
every jar is SHA-256-verified before the JVM starts, and the script refuses to run on any mismatch.
CSV columns: mode,value_date,settle,issue,maturity,coupon,yield,clean,dirty,accrued,first_coupon
(ISO dates; coupon / yield decimals; prices per unit notional; first_coupon empty unless the row is a when-issued bond).
"""
import argparse
import csv
import glob
import hashlib
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "strata_jars.sha256")
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "..", "tests", "golden", "bond_reference.csv"))

# (issue, maturity, coupon %) -- seasoned US Treasuries, regular periods at SETTLE.
SEASONED = [
    (date(2019, 8, 15), date(2029, 8, 15), 2.5),
    (date(2019, 8, 15), date(2034, 2, 15), 5.0),
    (date(2014, 5, 15), date(2044, 5, 15), 3.375),
    (date(2021, 11, 15), date(2031, 11, 15), 1.375),
]
SETTLE = date(2024, 1, 16)
SEASONED_YIELDS = [2.0, 4.0, 6.0]
# (dated, first coupon, maturity, coupon %) -- a new issue settling on its dated date with a SHORT first coupon.
WHEN_ISSUED = [(date(2024, 6, 15), date(2024, 11, 15), date(2034, 11, 15), 4.5)]
WHEN_ISSUED_YIELDS = [3.0, 4.7, 6.0]
# (issue, maturity, coupon %, settle) -- settling inside the FINAL coupon period (street discounts it simple).
FINAL_PERIOD = [
    (date(2019, 8, 15), date(2024, 8, 15), 2.5, date(2024, 3, 1)),
    (date(2014, 11, 15), date(2024, 11, 15), 2.25, date(2024, 6, 3)),
]
FINAL_PERIOD_YIELDS = [3.0, 5.0, 7.0]
HEADER = ["mode", "value_date", "settle", "issue", "maturity", "coupon", "yield", "clean", "dirty", "accrued", "first_coupon"]


def verify_jars(jar_dir):
    pins = {}
    with open(PINS) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split()[:2]
            pins[name] = digest
    present = {os.path.basename(p): p for p in glob.glob(os.path.join(jar_dir, "*.jar"))}
    missing = sorted(set(pins) - set(present))
    extra = sorted(set(present) - set(pins))
    if missing or extra:
        sys.exit(f"jar set differs from {PINS}: missing {missing}, unexpected {extra}")
    for name, path in sorted(present.items()):
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != pins[name]:
            sys.exit(f"SHA-256 mismatch for {name}: {got} != pinned {pins[name]}")
    return [present[n] for n in sorted(present)]


def default_jvm():
    home = os.environ.get("JAVA_HOME", "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home")
    for rel in ("lib/server/libjvm.dylib", "jre/lib/server/libjvm.dylib", "lib/server/libjvm.so"):
        p = os.path.join(home, rel)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jars", required=True, help="directory holding the pinned Strata jars")
    ap.add_argument("--jvm", default=default_jvm(), help="path to libjvm (default: $JAVA_HOME or the macOS Java 8 plugin)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    classpath = verify_jars(args.jars)
    if not args.jvm or not os.path.exists(args.jvm):
        sys.exit("no libjvm found: pass --jvm")

    import jpype  # noqa: E402  (imported after the pins are verified)
    jpype.startJVM(args.jvm, classpath=classpath)
    J = jpype.JClass
    LD = J("java.time.LocalDate")
    RD = J("com.opengamma.strata.basics.ReferenceData").standard()
    Pricer = J("com.opengamma.strata.pricer.bond.DiscountingFixedCouponBondProductPricer").DEFAULT
    version = J("com.opengamma.strata.pricer.bond.DiscountingFixedCouponBondProductPricer").class_.getPackage().getImplementationVersion()
    if str(version) != "2.12.56":
        sys.exit(f"unexpected Strata version {version}")
    YC = J("com.opengamma.strata.product.bond.FixedCouponBondYieldConvention")
    FCB = J("com.opengamma.strata.product.bond.FixedCouponBond")
    PS = J("com.opengamma.strata.basics.schedule.PeriodicSchedule")
    Freq = J("com.opengamma.strata.basics.schedule.Frequency")
    BDA = J("com.opengamma.strata.basics.date.BusinessDayAdjustment")
    Stub = J("com.opengamma.strata.basics.schedule.StubConvention")
    DC = J("com.opengamma.strata.basics.date.DayCounts")
    DA = J("com.opengamma.strata.basics.date.DaysAdjustment")
    HC = J("com.opengamma.strata.basics.date.HolidayCalendarIds")

    def ld(d):
        return LD.of(d.year, d.month, d.day)

    def street_bond(start, maturity, coupon_pct):
        sched = PS.builder().startDate(ld(start)).endDate(ld(maturity)).frequency(Freq.P6M) \
            .businessDayAdjustment(BDA.NONE).stubConvention(Stub.SHORT_INITIAL).build()
        return FCB.builder() \
            .securityId(J("com.opengamma.strata.product.SecurityId").of("SWAPS", "UST")) \
            .currency(J("com.opengamma.strata.basics.currency.Currency").USD).notional(1.0) \
            .accrualSchedule(sched).fixedRate(coupon_pct / 100.0).dayCount(DC.ACT_ACT_ICMA) \
            .yieldConvention(YC.US_STREET) \
            .legalEntityId(J("com.opengamma.strata.product.LegalEntityId").of("SWAPS", "USGOV")) \
            .settlementDateOffset(DA.ofBusinessDays(1, HC.NO_HOLIDAYS)) \
            .build().resolve(RD)

    def row(start, maturity, coupon_pct, settle, y_pct, first_coupon=None):
        rb = street_bond(start, maturity, coupon_pct)
        dirty = float(Pricer.dirtyPriceFromYield(rb, ld(settle), y_pct / 100.0))
        clean = float(Pricer.cleanPriceFromDirtyPrice(rb, ld(settle), dirty))
        accrued = float(Pricer.accruedInterest(rb, ld(settle)))
        return ["us_gb", settle.isoformat(), settle.isoformat(), start.isoformat(), maturity.isoformat(),
                f"{coupon_pct / 100.0:.10f}", f"{y_pct / 100.0:.10f}", f"{clean:.12f}", f"{dirty:.12f}", f"{accrued:.12f}",
                first_coupon.isoformat() if first_coupon else ""]

    rows = []
    for issue, maturity, cpn in SEASONED:
        for y in SEASONED_YIELDS:
            rows.append(row(issue, maturity, cpn, SETTLE, y))
    for dated, first, maturity, cpn in WHEN_ISSUED:
        for y in WHEN_ISSUED_YIELDS:
            rows.append(row(dated, maturity, cpn, dated, y, first_coupon=first))
    for issue, maturity, cpn, settle in FINAL_PERIOD:
        for y in FINAL_PERIOD_YIELDS:
            rows.append(row(issue, maturity, cpn, settle, y))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"Strata {version}: wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
