#pragma once
// Plain-data cashflow schedules + the templated pricing kernel.
//
// This header is deliberately QuantLib-free and templated on Scalar. It is the differentiable,
// vectorizable HOT PATH (CLAUDE.md §1). Schedules are extracted from QuantLib instruments ONCE at
// setup (see swaps/ql/extract.hpp) and priced here many times — during calibration and analytics —
// with QuantLib objects and their virtual dispatch kept out of the loop.
//
// All times are year fractions measured on the curve's day counter, from the curve reference date.
// The `Curve` template parameter only needs a member `Scalar discount(double t) const`.
//
// AAD SAFETY (matters when Scalar = AutoDiffScalar<VectorXd>): never seed an accumulator from a bare
// constant. `Scalar(0.0)` has an EMPTY derivative vector, and `+=` a length-M term then mismatches
// sizes (silent UB under -DNDEBUG). So every accumulator here is initialised from the FIRST
// curve-dependent term (which carries the derivatives), and scalar constants are added as raw
// `double` (AutoDiffScalar has a `+ double` overload that preserves the derivatives).

#include <cassert>
#include <vector>

namespace swaps::pricing {

// ---- Overnight-indexed swap ------------------------------------------------------------------
// Float leg: each compounded overnight coupon telescopes WITHIN the coupon to
//   amount = DF(accStart)/DF(accEnd) - 1
// (the daily fixings are curve forwards, so the daily growth factors multiply to the DF ratio).
// We do NOT assume it telescopes ACROSS coupons, because the payment date may be lagged/adjusted
// away from the accrual end; each coupon is discounted at its own pay date. This matches QuantLib
// coupon-for-coupon.
struct OisSwap {
  // Float coupons.
  std::vector<double> float_acc_start;  // accrual start time
  std::vector<double> float_acc_end;    // accrual end time
  std::vector<double> float_pay;        // payment (discount) time
  // Fixed coupons.
  std::vector<double> fixed_pay;        // payment (discount) time
  std::vector<double> fixed_accrual;    // year fraction tau_i (fixed day count)
};

// PV of one floating coupon: DF(pay) * (DF(accStart)/DF(accEnd) - 1).
template <class Scalar, class Curve>
Scalar ois_float_coupon_pv(const OisSwap& s, std::size_t i, const Curve& c) {
  return c.discount(s.float_pay[i]) *
         (c.discount(s.float_acc_start[i]) / c.discount(s.float_acc_end[i]) - 1.0);
}

// PV of the floating leg for unit notional.
template <class Scalar, class Curve>
Scalar ois_float_pv(const OisSwap& s, const Curve& c) {
  assert(!s.float_acc_start.empty());
  Scalar pv = ois_float_coupon_pv<Scalar>(s, 0, c);  // seed from first term (carries derivatives)
  for (std::size_t i = 1; i < s.float_acc_start.size(); ++i) pv += ois_float_coupon_pv<Scalar>(s, i, c);
  return pv;
}

// Fixed-leg annuity per unit fixed rate and unit notional: sum(tau_i * DF(pay_i)).
template <class Scalar, class Curve>
Scalar ois_annuity(const OisSwap& s, const Curve& c) {
  assert(!s.fixed_pay.empty());
  Scalar a = c.discount(s.fixed_pay[0]) * s.fixed_accrual[0];
  for (std::size_t i = 1; i < s.fixed_pay.size(); ++i) a += c.discount(s.fixed_pay[i]) * s.fixed_accrual[i];
  return a;
}

// Fair (par) fixed rate.
template <class Scalar, class Curve>
Scalar ois_par_rate(const OisSwap& s, const Curve& c) {
  return ois_float_pv<Scalar>(s, c) / ois_annuity<Scalar>(s, c);
}

// Net swap NPV per unit notional, receiver-of-float / payer-of-fixed:
//   NPV = float_pv - fixed_rate * annuity.
// fixed_rate is a contract constant (not a curve variable), so it is a raw double.
template <class Scalar, class Curve>
Scalar ois_swap_npv(const OisSwap& s, double fixed_rate, const Curve& c) {
  return ois_float_pv<Scalar>(s, c) - fixed_rate * ois_annuity<Scalar>(s, c);
}

// --- Multi-curve variants (Stage 3): the float leg FORECASTS off `fc` (the DF-ratio compounding) but
// DISCOUNTS off `dc` (the pay-date DF and the annuity). fc == dc reduces to the single-curve forms.
// This is exactly QuantLib's multi-curve setup (index forecast curve + a separate discount curve).
template <class Scalar, class FCurve, class DCurve>
Scalar ois_float_coupon_pv(const OisSwap& s, std::size_t i, const FCurve& fc, const DCurve& dc) {
  return dc.discount(s.float_pay[i]) *
         (fc.discount(s.float_acc_start[i]) / fc.discount(s.float_acc_end[i]) - 1.0);
}
template <class Scalar, class FCurve, class DCurve>
Scalar ois_float_pv(const OisSwap& s, const FCurve& fc, const DCurve& dc) {
  assert(!s.float_acc_start.empty());
  Scalar pv = ois_float_coupon_pv<Scalar>(s, 0, fc, dc);
  for (std::size_t i = 1; i < s.float_acc_start.size(); ++i) pv += ois_float_coupon_pv<Scalar>(s, i, fc, dc);
  return pv;
}
// Par rate of an OIS forecasting `fc`, discounting `dc` (= QuantLib OIS fairRate with a discount curve).
template <class Scalar, class FCurve, class DCurve>
Scalar ois_par_rate(const OisSwap& s, const FCurve& fc, const DCurve& dc) {
  return ois_float_pv<Scalar>(s, fc, dc) / ois_annuity<Scalar>(s, dc);
}
// Par spread of a basis swap: spread leg forecasts `fwd`, benchmark leg forecasts `bench`, both
// discount `disc`. s = (float_pv_bench - float_pv_fwd) / annuity_disc.
template <class Scalar, class FwdCurve, class BenchCurve, class DiscCurve>
Scalar basis_par_spread(const OisSwap& s, const FwdCurve& fwd, const BenchCurve& bench, const DiscCurve& disc) {
  return (ois_float_pv<Scalar>(s, bench, disc) - ois_float_pv<Scalar>(s, fwd, disc)) /
         ois_annuity<Scalar>(s, disc);
}

// ---- 3M compounded (IMM) SOFR future ---------------------------------------------------------
// The reference rate is the daily-compounded SOFR over the accrual period, which telescopes to
//   R = (DF(start)/DF(end) - 1) / accrual.
struct CompoundedFuture {
  double start;    // accrual start time
  double end;      // accrual end time
  double accrual;  // year fraction (ACT/360)
};

template <class Scalar, class Curve>
Scalar compounded_future_rate(const CompoundedFuture& f, const Curve& c) {
  return (c.discount(f.start) / c.discount(f.end) - 1.0) / f.accrual;
}

// ---- 1M arithmetic-average SOFR future -------------------------------------------------------
// Settles on the arithmetic average of daily SOFR over the calendar month. This does NOT telescope
// (arithmetic, not geometric), so we keep the per-day sub-periods. Matching QuantLib's
// OvernightIndexFuture::averagedRate() exactly:
//   R = ( realized_sum + sum_{future days d} (DF(t_d)/DF(t_{d+1}) - 1) ) / period_yf
// where each forward day contributes SOFR_d * accrual_d = DF(t_d)/DF(t_{d+1}) - 1, the DENOMINATOR
// is the single span year fraction yf(valueDate, maturityDate) (index day count), and realized_sum
// is the fixed contribution of days already fixed before the evaluation date (zero derivative).
struct AveragedFuture {
  double realized_sum = 0.0;      // sum over past days of fixing_d * accrual_d (constant)
  std::vector<double> sub_start;  // forward business days only
  std::vector<double> sub_end;
  double period_yf = 0.0;         // yf(valueDate, maturityDate), the averaging denominator
};

template <class Scalar, class Curve>
Scalar averaged_future_rate(const AveragedFuture& f, const Curve& c) {
  if (f.sub_start.empty())  // whole period already fixed (no curve dependence)
    return Scalar(f.realized_sum / f.period_yf);
  // Seed from the first forward day so the accumulator carries derivatives, then fold in the
  // realized (constant) contribution as a plain double.
  Scalar num = c.discount(f.sub_start[0]) / c.discount(f.sub_end[0]) - 1.0;
  for (std::size_t d = 1; d < f.sub_start.size(); ++d)
    num += c.discount(f.sub_start[d]) / c.discount(f.sub_end[d]) - 1.0;
  return (num + f.realized_sum) / f.period_yf;
}

// Futures PRICE from the reference rate: price = 100 * (1 - (rate + convexity)).
template <class Scalar>
Scalar future_price(Scalar rate, double convexity) {
  return (1.0 - (rate + convexity)) * 100.0;
}

// =================================================================================================
// GENERIC CASHFLOW MODEL (docs/generic-instrument-pipeline.md §2)
// =================================================================================================
// Everything above this line is a SPECIAL CASE of what follows. `OisSwap`, `CompoundedFuture` and
// `AveragedFuture` are index-flavoured data shapes; the types below are the ONE rate/coupon model
// the engine will keep. The legacy forms are retained while call sites migrate; each reduces to the
// generic form BIT-EXACTLY (see the k-form note on float_coupon_pv, and tests/generic_cashflow_test).
//
// The unifying formula for every floating rate in scope:
//
//   rate = ( Σ_k w_k · [ DF_fc(s_k) / DF_fc(e_k) − 1 ] + realized ) / tau_index
//
// A compounded OIS coupon is ONE sub-period (daily compounding telescopes to the DF ratio); an
// averaged OIS/FF coupon is one sub-period per business day; an IBOR fixing is ONE sub-period over
// the fixing period. "SOFR 3M future", "FF future", "Euribor future" are not engine concepts — they
// are different `RateObservation` DATA built by a test (CLAUDE.md §1, design §1/§6).

// One observation of a floating index over a coupon's (or future's) period.
struct RateObservation {
  // Sub-period brackets, in FORECAST-curve time. Empty => the whole observation is already fixed
  // and the rate is a pure constant (`realized / tau_index`).
  std::vector<double> sub_start;
  std::vector<double> sub_end;
  // Per-sub-period weights. EMPTY MEANS ALL-ONES — and we then neither allocate nor multiply, which
  // is what keeps the standard OIS/IBOR reduction bit-exact (x * 1.0 is exact in IEEE-754, but the
  // empty path avoids the multiply and the storage entirely).
  std::vector<double> weight;
  double realized = 0.0;   // constant contribution of already-fixed days (ZERO derivative)
  double tau_index = 0.0;  // the denominator, on the INDEX's own day count
};

// One floating coupon: an observation, discounted at its own pay date on its own accrual basis.
// `tau_pay` is the PAYMENT accrual (the coupon's day count) and is deliberately distinct from
// `obs.tau_index` (the index's) — that separation is what makes 30/360-vs-ACT/360 correct.
struct FloatCoupon {
  RateObservation obs;
  double pay = 0.0;      // payment (discount-curve) time
  double tau_pay = 0.0;  // payment accrual, coupon day count
  double spread = 0.0;   // additive contractual spread (outside the index), may be 0
};

// One fixed coupon. The RATE is supplied by the instrument/quote, not stored here, so this type
// serves both a par-rate annuity and a fixed leg at a contract rate.
struct FixedCoupon {
  double pay = 0.0;
  double tau = 0.0;
};

// Σ_k w_k · (DF(s_k)/DF(e_k) − 1) — the curve-dependent numerator ONLY.
// Precondition: at least one sub-period (so the accumulator can be seeded from a curve-dependent
// term and carry derivatives — see the AAD SAFETY note at the top of this header).
template <class Scalar, class FCurve>
Scalar obs_forward_sum(const RateObservation& o, const FCurve& fc) {
  assert(!o.sub_start.empty());
  assert(o.sub_start.size() == o.sub_end.size());
  assert(o.weight.empty() || o.weight.size() == o.sub_start.size());
  const bool weighted = !o.weight.empty();
  Scalar num = fc.discount(o.sub_start[0]) / fc.discount(o.sub_end[0]) - 1.0;
  if (weighted) num = num * o.weight[0];
  for (std::size_t k = 1; k < o.sub_start.size(); ++k) {
    Scalar t = fc.discount(o.sub_start[k]) / fc.discount(o.sub_end[k]) - 1.0;
    if (weighted)
      num += t * o.weight[k];
    else
      num += t;
  }
  return num;
}

// rate = ( Σ_k w_k (DF(s_k)/DF(e_k) − 1) + realized ) / tau_index.
//
// AAD CAVEAT: when the observation is fully fixed (no sub-periods) the result is a genuine constant
// and `Scalar(...)` therefore carries an EMPTY derivative vector. That is fine standalone (a fully
// fixed future IS a constant residual row — this matches `averaged_future_rate` today), but such a
// value must NOT be summed with a curve-dependent sibling. `float_coupon_pv` below never does: it
// keeps the constant as a raw `double` and lets DF(pay) carry the derivatives.
template <class Scalar, class FCurve>
Scalar rate(const RateObservation& o, const FCurve& fc) {
  assert(o.tau_index > 0.0);
  if (o.sub_start.empty()) return Scalar(o.realized / o.tau_index);
  return (obs_forward_sum<Scalar>(o, fc) + o.realized) / o.tau_index;
}

// PV of one floating coupon, forecasting `fc` and discounting `dc`:
//
//   pv = DF_dc(pay) · ( rate(obs, fc) + spread ) · tau_pay
//
// evaluated in the algebraically identical "k-form"
//
//   A  = Σ_k w_k (DF(s_k)/DF(e_k) − 1) + realized + spread·tau_index      (rate × time)
//   k  = tau_pay / tau_index                                              (build-time constant)
//   pv = DF_dc(pay) · A · k
//
// WHY: the k-form makes the legacy reduction BIT-EXACT rather than merely 1e-15-close. At one
// sub-period with no weights, realized = 0, spread = 0 and tau_pay == tau_index we get the literal
// double `k == 1.0` and `A == DF(s)/DF(e) − 1`, so `pv == DF(pay)·(DF(s)/DF(e) − 1)` to the last
// bit — identical to `ois_float_coupon_pv`. The doc's form `(num/tau)·tau` would drift ~1-2 ulp on
// every existing OIS number instead. Same model, chosen evaluation order.
//
// AAD-safe by construction: DF_dc(pay) always exists and always carries the derivatives, so a fully
// fixed coupon (empty sub-periods) still returns a correctly-SIZED Scalar and can be summed with
// live siblings — the empty-derivative trap the legacy `averaged_future_rate` still has.
template <class Scalar, class FCurve, class DCurve>
Scalar float_coupon_pv(const FloatCoupon& c, const FCurve& fc, const DCurve& dc) {
  assert(c.obs.tau_index > 0.0);
  const double k = c.tau_pay / c.obs.tau_index;
  const double konst = c.obs.realized + c.spread * c.obs.tau_index;
  if (c.obs.sub_start.empty()) return dc.discount(c.pay) * (konst * k);
  Scalar a = obs_forward_sum<Scalar>(c.obs, fc) + konst;
  return dc.discount(c.pay) * a * k;
}

// PV of a floating leg for unit notional.
template <class Scalar, class FCurve, class DCurve>
Scalar float_leg_pv(const std::vector<FloatCoupon>& leg, const FCurve& fc, const DCurve& dc) {
  assert(!leg.empty());
  Scalar pv = float_coupon_pv<Scalar>(leg[0], fc, dc);
  for (std::size_t i = 1; i < leg.size(); ++i) pv += float_coupon_pv<Scalar>(leg[i], fc, dc);
  return pv;
}

// Annuity per unit rate and unit notional: Σ DF_dc(pay_i) · tau_i.
template <class Scalar, class DCurve>
Scalar annuity(const std::vector<FixedCoupon>& leg, const DCurve& dc) {
  assert(!leg.empty());
  Scalar a = dc.discount(leg[0].pay) * leg[0].tau;
  for (std::size_t i = 1; i < leg.size(); ++i) a += dc.discount(leg[i].pay) * leg[i].tau;
  return a;
}

// Quote transform `ParRate`: float_leg_pv / annuity.
template <class Scalar, class FCurve, class DCurve>
Scalar par_rate(const std::vector<FloatCoupon>& float_leg, const std::vector<FixedCoupon>& fixed_leg,
                const FCurve& fc, const DCurve& dc) {
  return float_leg_pv<Scalar>(float_leg, fc, dc) / annuity<Scalar>(fixed_leg, dc);
}

// Quote transform `ParSpread` (basis): (pv_bench − pv_fwd) / annuity.
// The two float legs are SEPARATE — they may differ in frequency, day count and spread (design §6.7);
// they need not share a schedule the way the legacy `basis_par_spread` forces them to.
template <class Scalar, class FwdCurve, class BenchCurve, class DCurve>
Scalar par_spread(const std::vector<FloatCoupon>& fwd_leg, const std::vector<FloatCoupon>& bench_leg,
                  const std::vector<FixedCoupon>& annuity_leg, const FwdCurve& fwd,
                  const BenchCurve& bench, const DCurve& dc) {
  return (float_leg_pv<Scalar>(bench_leg, bench, dc) - float_leg_pv<Scalar>(fwd_leg, fwd, dc)) /
         annuity<Scalar>(annuity_leg, dc);
}

// Quote transform `Rate` (any future): rate + convexity. `convexity` is an INPUT NUMBER — the
// convexity MODEL (Hull-White etc.) lives in tests, never in engine code (design §3/§6).
template <class Scalar, class FCurve>
Scalar future_rate(const RateObservation& o, double convexity, const FCurve& fc) {
  return rate<Scalar>(o, fc) + convexity;
}

// ---- Moment-integrated averaging (docs/bezier-and-moments.md Part B) -----------------------------
// The arithmetic-average numerator over a window [a,b] with a daily fixing schedule is, exactly,
// Sum_d (DF(t_d)/DF(t_{d+1}) - 1). Its MOMENT expansion replaces the day-by-day sum with closed-form
// curve integrals:
//     num = int_a^b f  +  1/2 * <fixing_step> * int_a^b f^2  +  O(f^3 tau^2)
// - int_a^b f = integral(b) - integral(a) telescopes to the exact log-DF difference, LINEAR in x.
// - int_a^b f^2 feeds only the small correction, so a modest composite Gauss is plenty (measured:
//   2 moments -> ~5e-11 rate error, under the 1e-10 gate, for windows up to 2y).
// `fixing_step` is the (uniform) daily accrual fraction; a real calendar folds Sum_d tau_d^2 per curve
// segment in here (a precomputed constant) -- that refinement is additive and does not change this API.

// Composite int_a^b f(u)^2 du over the curve (piecewise-polynomial), 2-point Gauss per sub-interval.
template <class Scalar, class Curve>
Scalar curve_forward_sq_integral(const Curve& c, double a, double b, int subdiv = 32) {
  if (b <= a) return Scalar(0.0);
  const double gx = 0.5773502691896257;  // 1/sqrt(3)
  const double H = (b - a) / subdiv;
  Scalar acc{0.0};
  bool first = true;
  for (int s = 0; s < subdiv; ++s) {
    const double mid = a + (s + 0.5) * H, h = 0.5 * H;
    for (int sg = -1; sg <= 1; sg += 2) {
      const Scalar f = c.forward(mid + sg * gx * h);
      const Scalar term = f * f * h;  // 2-pt Gauss weight is h on each node
      if (first) { acc = term; first = false; } else acc += term;
    }
  }
  return acc;
}

// Arithmetic average rate over [a,b], moment form: rate = num / tau_index.
template <class Scalar, class Curve>
Scalar moment_average_rate(const Curve& c, double a, double b, double fixing_step, double tau_index) {
  const Scalar num =
      (c.integral(b) - c.integral(a)) + 0.5 * fixing_step * curve_forward_sq_integral<Scalar>(c, a, b);
  return num / tau_index;
}

}  // namespace swaps::pricing
