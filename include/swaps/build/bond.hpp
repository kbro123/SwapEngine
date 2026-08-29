// swaps::build — bond construction. Assembles the engine's plain bond structs (pricing::Bond in curve
// space, pricing::YieldBond in street/yield space) from bond terms + conventions. The bond-type specifics
// live HERE as DATA (US Treasury = semiannual, ACT/ACT ICMA, unadjusted regular periods, street yield
// f=2), never in the pricing kernel (CLAUDE.md §0/§1). A new bond type (Gilt, Bund, corporate,
// ACT/365 money-market bond, FRN) is a new builder filling the SAME structs — no engine change.
//
// TIMING conventions are absorbed into the per-flow exponent E_i here, which is why the sweep stays on
// the Horner fast path. The one thing an exponent cannot express is the DISCOUNT FORM of the fractional
// first period, so that travels as pricing::YieldConvention (stub + final_period_simple) — see the
// convention table in pricing/bond.hpp. Named builders below pick it; callers should not set it by hand.
//
// Faithful to QuantLib's FixedRateBond built on a semiannual Schedule with Unadjusted dates and
// ActualActual(ISMA): coupon amounts rate/f per unit notional, redemption 1.0 at maturity, ACT/ACT ISMA
// accrual, and the street exponent E_i = w + (i-1) (w = fraction of the current coupon period remaining
// at settlement) that reproduces QuantLib's chained (1+y/f)^{-f·τ} discounting exactly.
#ifndef SWAPS_BUILD_BOND_HPP
#define SWAPS_BUILD_BOND_HPP

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::build {

namespace px = swaps::pricing;

// Terms of a plain fixed-rate bullet bond (per unit notional; price basis 1.0 = par). A US Treasury fills
// freq = 2 and dc = "ACT/ACT". `value_date` is the curve reference (curve t = 0); `settle` is the
// settlement date the price/yield is quoted for (T+1 for Treasuries); `issue` is the dated date (first
// accrual start) so the coupon schedule and the current period's reference are well defined.
struct FixedBondTerms {
  Date value_date;
  Date settle;
  Date issue;
  Date maturity;
  double coupon = 0.0;  // annual coupon rate (0.04 = 4%)
  int freq = 2;         // coupons per year (also the yield compounding frequency f)
  // Yield CONVENTION — how the fractional first period is discounted (pricing/bond.hpp YieldConvention).
  // The default is the plain compound stub (UK gilt / French OAT). A US Treasury quoted STREET wants
  // final_period_simple = true; the 31 CFR App B / Bloomberg "Treasury method" wants stub = Simple. Use
  // the named builders below rather than setting these by hand.
  px::StubDiscount stub = px::StubDiscount::Compound;
  bool final_period_simple = false;
};

// Both representations of one built bond, plus the pieces a caller needs to relate them.
struct BuiltBond {
  px::Bond curve;        // curve-space: cashflows in curve time + settle time + accrued
  px::YieldBond yield;   // street-space: cumulative exponents + amounts + accrued
  double accrued = 0.0;  // accrued interest at settlement, per unit notional
  // Current coupon period + per-period coupon, so a caller can recompute accrued (and the street offset
  // w = (next-settle)/(next-prev)) for a DIFFERENT settlement date via accrued_interest() below — O(1),
  // no rebuild — as long as settlement stays within [prev_coupon, next_coupon] (a coupon-date crossing
  // changes the cashflow set and needs a rebuild).
  Date prev_coupon, next_coupon;
  double coupon_per_period = 0.0;
};

// FAST standalone accrued interest (per unit notional), ACT/ACT ICMA: coupon-per-period times the elapsed
// fraction of the current coupon period. A pure schedule quantity — no yield, no price, no curve — so it
// is O(1) and vectorizes trivially over a universe (and over a rolling settlement date). This is the ONE
// definition of accrued; fixed_rate_bond() calls it so the built value and any re-evaluation agree.
inline double accrued_interest(double coupon, int freq, const Date& prev_coupon, const Date& next_coupon,
                               const Date& settle) {
  const double c = coupon / double(freq);
  return c * double(settle - prev_coupon) / double(next_coupon - prev_coupon);
}

// Semiannual (freq) coupon dates strictly in (issue, maturity], ascending, stepping BACKWARD from
// maturity so the regular cycle is anchored at maturity (the market convention). `ref_start` returns the
// coupon date at/just before `issue` — the reference-period start of the first coupon (== issue for a
// bond dated on a coupon date).
inline std::vector<Date> coupon_dates_backward(const Date& issue, const Date& maturity, int freq,
                                               Date& ref_start) {
  if (maturity <= issue) throw std::invalid_argument("bond maturity must be after issue");
  const int step_m = 12 / freq;
  std::vector<Date> cpn;
  Date d = maturity;
  while (d > issue) {
    cpn.push_back(d);
    d = d.plus_months(-step_m);
  }
  ref_start = d;  // the (possibly virtual) coupon date at/before issue = first period's reference start
  std::reverse(cpn.begin(), cpn.end());
  return cpn;
}

// NOTE (scope): this builds a bond with REGULAR coupon periods (the standard on-the-run treasury issued
// on a coupon date). Settlement may fall anywhere inside the current period — that fractional exposure is
// carried exactly by `w` — but an ODD first/last coupon (issue NOT on the maturity-anchored grid) is a
// documented follow-up: the pricing kernel already accepts arbitrary per-flow exponents/amounts, so an
// odd-stub builder only needs per-period ACT/ACT ICMA year fractions here, no engine change.
inline BuiltBond fixed_rate_bond(const FixedBondTerms& t) {
  if (t.freq <= 0 || 12 % t.freq != 0) throw std::invalid_argument("bond freq must divide 12");
  Date ref_start;
  const std::vector<Date> cpn = coupon_dates_backward(t.issue, t.maturity, t.freq, ref_start);
  if (cpn.empty()) throw std::invalid_argument("bond has no coupons");

  // Locate the CURRENT coupon period: prev < settle <= next. `prev` is that period's reference start.
  std::size_t cur = 0;
  Date prev = ref_start;
  while (cur < cpn.size() && !(t.settle < cpn[cur])) {
    prev = cpn[cur];
    ++cur;
  }
  if (cur >= cpn.size()) throw std::invalid_argument("settlement is on/after the last coupon");
  const Date next = cpn[cur];

  const double period_days = double(next - prev);
  const double w = double(next - t.settle) / period_days;        // fraction of current period remaining
  const double cpn_amt = t.coupon / double(t.freq);              // coupon per unit notional per period

  BuiltBond out;
  out.accrued = accrued_interest(t.coupon, t.freq, prev, next, t.settle);
  out.prev_coupon = prev;
  out.next_coupon = next;
  out.coupon_per_period = cpn_amt;

  // Curve-space cashflows: every coupon from `cur` onward at its curve time; redemption 1.0 at maturity.
  out.curve.settle = curve_time(t.value_date, t.settle);
  out.curve.accrued = out.accrued;
  for (std::size_t j = cur; j < cpn.size(); ++j) {
    px::BondCashflow cf;
    cf.pay = curve_time(t.value_date, cpn[j]);
    cf.amount = cpn_amt + (j + 1 == cpn.size() ? 1.0 : 0.0);  // + redemption at maturity
    out.curve.flows.push_back(cf);
  }

  // Street-space flows: exponent E_i = w + i for the i-th future coupon (i = 0 at `cur`), amount rate/f
  // (+1.0 redemption at maturity). This is the ACT/ACT ISMA street convention (each full period adds
  // exactly 1 compounding period; the current partial period contributes w).
  out.yield.conv.freq = double(t.freq);
  out.yield.conv.stub = t.stub;
  out.yield.conv.final_period_simple = t.final_period_simple;
  out.yield.accrued = out.accrued;
  for (std::size_t j = cur; j < cpn.size(); ++j) {
    px::YieldFlow yf;
    yf.exponent = w + double(j - cur);
    yf.amount = cpn_amt + (j + 1 == cpn.size() ? 1.0 : 0.0);
    out.yield.flows.push_back(yf);
  }
  return out;
}

// Convenience: a US Treasury note/bond (semiannual, ACT/ACT ISMA), quoted on the US STREET convention —
// compound stub, EXCEPT once settlement reaches the final coupon period, where the market (and QuantLib
// via SimpleThenCompounded, and Rateslib `us_gb`) discounts the remaining stub simple. Before that final
// period the two are identical; inside it they differ by ~0.7 bp, so the flag is not cosmetic.
inline BuiltBond us_treasury(const Date& value_date, const Date& settle, const Date& issue,
                             const Date& maturity, double coupon) {
  return fixed_rate_bond(FixedBondTerms{value_date, settle, issue, maturity, coupon, /*freq=*/2,
                                        px::StubDiscount::Compound, /*final_period_simple=*/true});
}

// The TREASURY METHOD: 31 CFR Part 356 Appendix B, which is also what Bloomberg reports as the Treasury
// (as opposed to street) yield, and Rateslib's `ust_31bii`/`us_gb_tsy`. The regulation writes EVERY
// sub-case as "P[1 + (r/s)(i/2)] = ...", i.e. the fractional period is discounted SIMPLE always — not
// only in the final period. Oracle: QuantLib with Compounding::SimpleThenCompounded.
inline BuiltBond us_treasury_tsy(const Date& value_date, const Date& settle, const Date& issue,
                                 const Date& maturity, double coupon) {
  return fixed_rate_bond(FixedBondTerms{value_date, settle, issue, maturity, coupon, /*freq=*/2,
                                        px::StubDiscount::Simple, /*final_period_simple=*/false});
}

// =================================================================================================
// WHEN-ISSUED (WI) / odd-first-period bonds — the subtly-different yield path.
// =================================================================================================
// A WI treasury trades before issue for settlement ON the issue (dated) date, and its FIRST coupon period
// is frequently irregular. Two subtleties vs a seasoned bond, both centred on the first period:
//   1. Settlement == the dated date, so a NEW issue has ZERO accrued (a REOPENING settles later within the
//      first period and carries accrued from the ORIGINAL dated date — same formula, settle > dated).
//   2. A SHORT first coupon: when first_coupon − dated is less than a full period, the first coupon is
//      PRORATED to the actual days — interest = (coupon/f)·(first_coupon − dated)/E, E the full quasi-coupon
//      period [first_coupon − period, first_coupon] (31 CFR 356 App B / Treasury "daily interest decimal").
//
// The PRORATION is 31 CFR Part 356 App B; the DISCOUNTING here is NOT. App B writes every one of its
// sub-cases as "P[1 + (r/s)(i/2)] = ..." -- SIMPLE interest over the fractional period -- whereas this
// builder (like the rest of the kernel, like QuantLib's BondFunctions/Compounded, and like Rateslib's
// "us_gb") discounts it by the COMPOUND factor v^{w0}: the US STREET convention. The two are related
// exactly by dirty_AppB = dirty_street*(1+y/f)^{w0}/(1+w0*y/f) -- ~7e-6 of price (~0.7 bp) on a 6y note at
// 2%, so they are not interchangeable. A Treasury-convention mode is a documented follow-up; see the
// header note in tests/bond_reference_test.cpp, which pins the relationship against the regulation and
// against Rateslib "ust_31bii". Crucially this stays on the FAST Horner path: the discount exponents are still w0 + integer
// (w0 = (first_coupon − settle)/E), so only the FIRST coefficient (coupon·s) and the accrued differ — the
// geometric structure is intact, so BondUniverse::is_regular() is still true.
//
// SCOPE: regular + SHORT first coupon (dated within the current quasi-coupon period). A LONG first coupon
// (dated BEFORE the prior quasi-coupon date, so the first payment spans >1 quasi-period) is rejected — it
// needs the App B quasi-period sum and is the documented follow-up.
inline BuiltBond when_issued_bond(const Date& value_date, const Date& dated, const Date& first_coupon,
                                  const Date& maturity, double coupon, int freq, const Date& settle,
                                  px::StubDiscount stub = px::StubDiscount::Compound,
                                  bool final_period_simple = false) {
  if (freq <= 0 || 12 % freq != 0) throw std::invalid_argument("bond freq must divide 12");
  const int step_m = 12 / freq;
  if (first_coupon <= dated) throw std::invalid_argument("first_coupon must be after the dated date");
  if (settle < dated) throw std::invalid_argument("settlement before the dated date");
  if (!(settle < first_coupon))
    throw std::invalid_argument("settlement on/after first coupon: use fixed_rate_bond (regular regime)");
  const Date quasi_prev = first_coupon.plus_months(-step_m);  // start of the first coupon's quasi-period
  if (dated < quasi_prev)
    throw std::invalid_argument("long first coupon (dated before prior quasi-coupon) not yet supported");

  const double E = double(first_coupon - quasi_prev);     // full quasi-coupon period length (days)
  const double s = double(first_coupon - dated) / E;      // short coupon factor (<=1; ==1 for regular)
  const double a = double(settle - dated) / E;            // accrued fraction (0 at issue for a new issue)
  const double w0 = double(first_coupon - settle) / E;    // discount exponent to the first coupon (= s − a)
  const double cpn = coupon / double(freq);

  std::vector<Date> cd;  // coupon dates: first_coupon, +period, ..., maturity
  for (Date d = first_coupon; d <= maturity; d = d.plus_months(step_m)) cd.push_back(d);
  if (cd.empty() || cd.back() != maturity)
    throw std::invalid_argument("first_coupon and maturity are not on a common frequency grid");
  const int N = static_cast<int>(cd.size());

  BuiltBond out;
  out.accrued = cpn * a;
  out.prev_coupon = dated;  // the first period's accrual start (for a rolled-settlement accrued recompute)
  out.next_coupon = first_coupon;
  out.coupon_per_period = cpn;

  out.curve.settle = curve_time(value_date, settle);
  out.curve.accrued = out.accrued;
  out.yield.conv.freq = double(freq);
  out.yield.conv.stub = stub;
  out.yield.conv.final_period_simple = final_period_simple;
  out.yield.accrued = out.accrued;
  for (int j = 0; j < N; ++j) {
    const double amt = (j == 0 ? cpn * s : cpn) + (j + 1 == N ? 1.0 : 0.0);  // first coupon prorated by s
    px::BondCashflow cf;
    cf.pay = curve_time(value_date, cd[j]);
    cf.amount = amt;
    out.curve.flows.push_back(cf);
    px::YieldFlow yf;
    yf.exponent = w0 + double(j);  // integer-spaced from w0 => stays on the Horner fast path
    yf.amount = amt;
    out.yield.flows.push_back(yf);
  }
  return out;
}

// Convenience: a when-issued US Treasury settling on its issue (dated) date (semiannual), STREET quoted.
// New issue => zero accrued; pass an explicit `settle` (via when_issued_bond) for a reopening within the
// first period. A WI bond is by construction in its FIRST period, never its last, so `final_period_simple`
// cannot fire here — street and plain-compound coincide for a WI quote, and the meaningful choice is
// street vs the Treasury method below.
inline BuiltBond us_treasury_wi(const Date& value_date, const Date& dated, const Date& first_coupon,
                                const Date& maturity, double coupon) {
  return when_issued_bond(value_date, dated, first_coupon, maturity, coupon, /*freq=*/2, /*settle=*/dated,
                          px::StubDiscount::Compound, /*final_period_simple=*/true);
}

// A when-issued US Treasury on the TREASURY METHOD (31 CFR Part 356 App B). This is the combination the
// regulation is actually written for: App B's worked examples are new issues and reopenings, and its
// short-first-coupon proration (already applied by when_issued_bond) travels WITH its simple-stub
// discounting. Quoting a WI bond off `us_treasury_wi` gives the STREET number instead; they differ.
inline BuiltBond us_treasury_wi_tsy(const Date& value_date, const Date& dated, const Date& first_coupon,
                                    const Date& maturity, double coupon, const Date& settle) {
  return when_issued_bond(value_date, dated, first_coupon, maturity, coupon, /*freq=*/2, settle,
                          px::StubDiscount::Simple, /*final_period_simple=*/false);
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_BOND_HPP
