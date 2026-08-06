#pragma once
// Plain-data bond cashflows + the templated bond pricing kernel (curve space AND yield space).
//
// This header is deliberately QuantLib-free and templated on Scalar — the same dual-use the swap kernel
// (cashflows.hpp) has: Scalar = double prices, Scalar = ad::Dual yields the analytic gradient from the
// SAME code. A bond is modelled as the engine already models everything: as DATA (a list of dated
// cashflows + a small YieldConvention), not a subclass (CLAUDE.md §0). "US Treasury" is a set of field
// values a builder fills in (semiannual, ACT/ACT ICMA, street compounding), never a type in engine code.
//
// TWO pricing modes, both native shapes:
//
//   (A) CURVE space — value off a discount curve. A bond is Σ amount_i · DF(pay_i): LINEAR in
//       DF = exp(-Wx), so a universe of bonds rides the existing W-cache (portfolio/bond_universe.hpp)
//       exactly like the swap book. z-spread and the par-par asset-swap spread are per-bond root finds
//       against these DFs.
//
//   (B) YIELD space — the classic street price<->yield/accrued/duration/convexity calcs, curve-free.
//       We replicate QuantLib's discrete street convention EXACTLY (chained per-period discounting at
//       (1+y/f)^{-f·τ_j}), so the numbers are penny-perfect vs QuantLib::BondFunctions. A whole universe
//       is a BATCHED vectorized Newton (portfolio/bond_universe.hpp), not a per-bond QuantLib solve loop.
//
// All curve times are year fractions on the curve's day counter from the curve reference date, as
// everywhere else in the engine (the `Curve` template needs only `Scalar discount(double t) const`).

#include <cassert>
#include <cmath>
#include <vector>

namespace swaps::pricing {

// =================================================================================================
// (A) CURVE-SPACE BOND  — Σ amount · DF(pay); rides the W-cache.
// =================================================================================================

// One dated bond cashflow, in CURVE time on the discount curve. `amount` is the actual cash paid per
// unit notional (a coupon amount c·τ, or a coupon PLUS the redemption at maturity) — NOT a rate. Keeping
// the resolved amount (rather than rate×accrual) is what makes the curve PV a pure linear form in DF, so
// the sweep stays on the W-cache. A projected/floating amount (FRN) is resolved by a builder before it
// reaches here — the kernel never re-derives it.
struct BondCashflow {
  double pay = 0.0;     // payment time (discount-curve time)
  double amount = 0.0;  // cash paid per unit notional
};

// A bond as the pricing kernel sees it: future cashflows (settlement date onward) + the accrued interest
// and settlement time needed to relate model value to a quoted clean/dirty PRICE. `flows` are strictly
// after settlement (a builder drops any already-paid coupon). All per unit notional (price basis 1.0;
// scale by face/100 outside).
struct Bond {
  std::vector<BondCashflow> flows;
  double settle = 0.0;    // settlement time (discount-curve time)
  double accrued = 0.0;   // accrued interest at settlement, per unit notional
};

// Present value of the future cashflows AS OF THE CURVE REFERENCE DATE: Σ amount_i · DF(pay_i). Linear
// in DF. (The forward value to settlement — the model dirty price — is this / DF(settle); see
// bond_dirty_price below.) AAD-safe: the accumulator seeds from the first curve-dependent term.
template <class Scalar, class Curve>
Scalar bond_pv_today(const Bond& b, const Curve& c) {
  assert(!b.flows.empty());
  Scalar pv = b.flows[0].amount * c.discount(b.flows[0].pay);
  for (std::size_t i = 1; i < b.flows.size(); ++i) pv += b.flows[i].amount * c.discount(b.flows[i].pay);
  return pv;
}

// Model DIRTY price = forward value of the cashflows to the settlement date: (Σ amount·DF(pay))/DF(settle).
// This is what a curve reprices a bond's traded (dirty) price to; the clean price is this − accrued.
template <class Scalar, class Curve>
Scalar bond_dirty_price(const Bond& b, const Curve& c) {
  return bond_pv_today<Scalar>(b, c) / c.discount(b.settle);
}

template <class Scalar, class Curve>
Scalar bond_clean_price(const Bond& b, const Curve& c) {
  return bond_dirty_price<Scalar>(b, c) - b.accrued;
}

// z-spread: the constant continuously-compounded spread s (on curve time) that reprices the bond to a
// target DIRTY price when added to the curve: Σ amount·DF(pay)·e^{−s·pay} / (DF(settle)·e^{−s·settle}) =
// target. A scalar Newton in s (monotone: price strictly decreases in s), curve-time exponents. Returns
// the spread in the curve's continuous-compounding convention. QL-free; validated against QuantLib's
// BondFunctions::zSpread in the oracle test.
template <class Curve>
double bond_z_spread(const Bond& b, const Curve& c, double target_dirty, double tol = 1e-14,
                     int max_iter = 100) {
  // Precompute per-flow (amount, DF(pay), pay) and the settle DF once — the curve is fixed in the solve.
  const double df_settle = c.discount(b.settle);
  double s = 0.0;
  for (int it = 0; it < max_iter; ++it) {
    double price = 0.0, dprice = 0.0;  // price(s) and d price / d s
    for (const auto& f : b.flows) {
      const double dt = f.pay - b.settle;                         // discount horizon from settlement
      const double v = f.amount * (c.discount(f.pay) / df_settle) * std::exp(-s * dt);
      price += v;
      dprice += -dt * v;
    }
    const double resid = price - target_dirty;
    if (std::abs(resid) <= tol) break;
    s -= resid / dprice;  // Newton (dprice < 0, well-conditioned)
  }
  return s;
}

// =================================================================================================
// (B) YIELD-SPACE STREET KERNEL  — curve-free price<->yield/accrued/duration/convexity.
// =================================================================================================
// The street convention discounts cashflow i by the chained factor Π_{j≤i} (1+y/f)^{−f·τ_j}, τ_j the
// day-count year fraction of period j (settlement→first flow, then coupon period to coupon period). With
// the cumulative exponent E_i = f · Σ_{j≤i} τ_j and base = 1 + y/f:
//     dirty(y) = Σ_i CF_i · base^{−E_i}
// This is EXACTLY QuantLib's compounded bond yield discounting (its per-period chained discount factors
// multiply to base^{−E_i}), so the numbers match QuantLib::BondFunctions to machine precision. The clean
// price is dirty − accrued; duration/convexity are closed-form y-derivatives of dirty(y).
//
// `YieldFlow` is the resolved per-bond street data a builder produces: one entry per future cashflow, its
// cumulative exponent E_i and its amount. Curve-free — no dates, no curve. This is the row a universe
// sweep stacks into a matrix.
struct YieldFlow {
  double exponent = 0.0;  // E_i = f · Σ_{j≤i} τ_j  (number of compounding periods to this flow)
  double amount = 0.0;    // CF_i per unit notional
};

struct YieldBond {
  std::vector<YieldFlow> flows;
  double freq = 2.0;     // compounding frequency f (US Treasury: 2)
  double accrued = 0.0;  // accrued interest at settlement, per unit notional (for clean<->dirty)
};

// dirty(y) and its first two y-derivatives in one pass. base = 1 + y/f; d base/dy = 1/f, so
//   dirty      =  Σ CF · base^{−E}
//   d/dy dirty = −(1/f) Σ CF · E · base^{−E−1}
//   d²/dy²     =  (1/f²) Σ CF · E(E+1) · base^{−E−2}
// Written scalar and branch-light so the batched sweep can inline the same arithmetic per lane.
struct BondYieldValue {
  double dirty = 0.0;
  double d1 = 0.0;  // d dirty / dy
  double d2 = 0.0;  // d² dirty / dy²
};
inline BondYieldValue bond_yield_value(const YieldBond& b, double y) {
  const double f = b.freq, base = 1.0 + y / f;
  BondYieldValue v;
  for (const auto& fl : b.flows) {
    const double p = fl.amount * std::pow(base, -fl.exponent);   // CF · base^{−E}
    v.dirty += p;
    v.d1 += -(fl.exponent / f) * p / base;                        // CF·(−E/f)·base^{−E−1}
    v.d2 += (fl.exponent * (fl.exponent + 1.0) / (f * f)) * p / (base * base);
  }
  return v;
}

inline double bond_dirty_from_yield(const YieldBond& b, double y) { return bond_yield_value(b, y).dirty; }
inline double bond_clean_from_yield(const YieldBond& b, double y) {
  return bond_yield_value(b, y).dirty - b.accrued;
}

// Street yield from a CLEAN price via Newton on dirty(y) = clean + accrued. Robust: dirty is strictly
// decreasing and convex in y, so Newton from a sensible seed converges quadratically in a few steps.
inline double bond_yield_from_clean(const YieldBond& b, double clean_price, double y0 = 0.05,
                                    double tol = 1e-14, int max_iter = 100) {
  const double target = clean_price + b.accrued;  // solve on the dirty price
  double y = y0;
  for (int it = 0; it < max_iter; ++it) {
    const BondYieldValue v = bond_yield_value(b, y);
    const double resid = v.dirty - target;
    if (std::abs(resid) <= tol) break;
    y -= resid / v.d1;
  }
  return y;
}

// Modified duration = −(1/P_dirty)·d dirty/dy and convexity = (1/P_dirty)·d² dirty/dy², both on the
// DIRTY price — the QuantLib::BondFunctions::duration(Modified)/convexity convention.
struct BondRisk {
  double modified_duration = 0.0;
  double macaulay_duration = 0.0;
  double convexity = 0.0;
};
inline BondRisk bond_risk(const YieldBond& b, double y) {
  const BondYieldValue v = bond_yield_value(b, y);
  BondRisk r;
  r.modified_duration = -v.d1 / v.dirty;
  r.macaulay_duration = r.modified_duration * (1.0 + y / b.freq);  // Macaulay = Modified·(1+y/f)
  r.convexity = v.d2 / v.dirty;
  return r;
}

}  // namespace swaps::pricing
