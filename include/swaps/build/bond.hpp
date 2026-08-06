// swaps::build — bond construction. Assembles the engine's plain bond structs (pricing::Bond in curve
// space, pricing::YieldBond in street/yield space) from bond terms + conventions. The bond-type specifics
// live HERE as DATA (US Treasury = semiannual, ACT/ACT ICMA, unadjusted regular periods, street street
// yield f=2), never in the pricing kernel (CLAUDE.md §0/§1). A new bond type (Gilt, Bund, corporate,
// ACT/365 money-market bond, FRN) is a new builder filling the SAME structs — no engine change.
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
  int freq = 2;         // coupons per year
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
  out.yield.freq = double(t.freq);
  out.yield.accrued = out.accrued;
  for (std::size_t j = cur; j < cpn.size(); ++j) {
    px::YieldFlow yf;
    yf.exponent = w + double(j - cur);
    yf.amount = cpn_amt + (j + 1 == cpn.size() ? 1.0 : 0.0);
    out.yield.flows.push_back(yf);
  }
  return out;
}

// Convenience: a US Treasury note/bond (semiannual, ACT/ACT ISMA street convention).
inline BuiltBond us_treasury(const Date& value_date, const Date& settle, const Date& issue,
                             const Date& maturity, double coupon) {
  return fixed_rate_bond(FixedBondTerms{value_date, settle, issue, maturity, coupon, /*freq=*/2});
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_BOND_HPP
