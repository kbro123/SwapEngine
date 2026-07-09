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

// PV of the floating leg for unit notional.
template <class Scalar, class Curve>
Scalar ois_float_pv(const OisSwap& s, const Curve& c) {
  Scalar pv(0.0);
  for (std::size_t i = 0; i < s.float_acc_start.size(); ++i) {
    const Scalar growth = c.discount(s.float_acc_start[i]) / c.discount(s.float_acc_end[i]);
    pv += c.discount(s.float_pay[i]) * (growth - Scalar(1.0));
  }
  return pv;
}

// Fixed-leg annuity per unit fixed rate and unit notional: sum(tau_i * DF(pay_i)).
template <class Scalar, class Curve>
Scalar ois_annuity(const OisSwap& s, const Curve& c) {
  Scalar a(0.0);
  for (std::size_t i = 0; i < s.fixed_pay.size(); ++i)
    a += Scalar(s.fixed_accrual[i]) * c.discount(s.fixed_pay[i]);
  return a;
}

// Fair (par) fixed rate.
template <class Scalar, class Curve>
Scalar ois_par_rate(const OisSwap& s, const Curve& c) {
  return ois_float_pv<Scalar>(s, c) / ois_annuity<Scalar>(s, c);
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
  return (c.discount(f.start) / c.discount(f.end) - Scalar(1.0)) / Scalar(f.accrual);
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
  Scalar num(f.realized_sum);
  for (std::size_t d = 0; d < f.sub_start.size(); ++d)
    num += c.discount(f.sub_start[d]) / c.discount(f.sub_end[d]) - Scalar(1.0);
  return num / Scalar(f.period_yf);
}

// Futures PRICE from the reference rate: price = 100 * (1 - (rate + convexity)).
template <class Scalar>
Scalar future_price(Scalar rate, double convexity) {
  return Scalar(100.0) * (Scalar(1.0) - (rate + Scalar(convexity)));
}

}  // namespace swaps::pricing
