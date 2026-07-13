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

}  // namespace swaps::pricing
