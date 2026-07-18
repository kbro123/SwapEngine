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

// =================================================================================================
// GENERIC CASHFLOW MODEL (docs/generic-instrument-pipeline.md §2)
// =================================================================================================
// The ONE rate/coupon model the engine keeps. A compounded OIS coupon, an averaged OIS/FF coupon, an
// IBOR fixing and every futures flavour are all THIS type with different DATA -- there are no
// index-flavoured structs. (The retired OisSwap/CompoundedFuture/AveragedFuture were exactly this,
// each a special case: a compounded OIS coupon is ONE telescoped sub-period with tau_pay == tau_index;
// an averaged future is one sub-period per business day; a compounded future is ONE sub-period.)
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
  // > 0 selects the MOMENT path (docs/bezier-and-moments.md Part B): the arithmetic average over the
  // SINGLE window [sub_start[0], sub_end[0]] is computed from closed-form curve moments instead of a
  // day-by-day sum, with `fixing_step` the (uniform) daily accrual fraction. Default 0 => the exact
  // sub-period path, bit-for-bit unchanged.
  double fixing_step = 0.0;
  // Third day-count moment Sum_d (day step)^3 / (b-a). Real calendars (weekend 3-day accruals) make
  // the 3rd moment significant; default 0 keeps the 2-moment form (enough for uniform daily).
  double fixing_step3 = 0.0;
  // COMPOUNDED (product) mode. Default false => the arithmetic Σ form above (compounded OIS still
  // telescopes into ONE sub-period, so the standard shape never sets this). Set true ONLY when daily
  // compounding does NOT telescope -- RFR lookback WITHOUT observation shift, or lockout -- so the
  // numerator must be the actual product realized_factor · ∏_k (1 + w_k (DF(s_k)/DF(e_k) − 1)) − 1
  // (each term is 1 + r_k · dt_k). Obs-shift telescopes, so it stays a single arithmetic sub-period.
  bool compounded = false;
  // Multiplicative past product ∏_past (1 + r · dt) for a partially/fully realized COMPOUNDED coupon
  // (the compounded analogue of the additive `realized`): total growth = realized_factor · ∏_future.
  // 1.0 => no realized prefix. Only consulted when `compounded`.
  double realized_factor = 1.0;
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

// Composite ∫f² over the curve for the moment path (defined at the end of this header).
template <class Scalar, class Curve>
Scalar curve_forward_sq_integral(const Curve& c, double a, double b, int subdiv = 32);
template <class Scalar, class Curve>
Scalar curve_forward_cube_integral(const Curve& c, double a, double b, int subdiv = 32);

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

// ∏_k (1 + w_k · (DF(s_k)/DF(e_k) − 1)) — the curve-dependent daily-compounding growth (each factor is
// 1 + r_k·dt_k). Precondition: at least one sub-period, so the accumulator seeds from a curve-dependent
// factor and carries derivatives (AAD SAFETY note at the top of this header). Used only in `compounded`
// mode (RFR lookback-without-shift / lockout, where the product does NOT telescope to a single DF ratio).
template <class Scalar, class FCurve>
Scalar obs_compound_growth(const RateObservation& o, const FCurve& fc) {
  assert(!o.sub_start.empty());
  assert(o.sub_start.size() == o.sub_end.size());
  assert(o.weight.empty() || o.weight.size() == o.sub_start.size());
  const bool weighted = !o.weight.empty();
  auto factor = [&](std::size_t k) {
    Scalar t = fc.discount(o.sub_start[k]) / fc.discount(o.sub_end[k]) - 1.0;
    return weighted ? Scalar(1.0 + o.weight[k] * t) : Scalar(1.0 + t);
  };
  Scalar prod = factor(0);
  for (std::size_t k = 1; k < o.sub_start.size(); ++k) prod = prod * factor(k);
  return prod;
}

// The curve-dependent numerator: the sub-period sum, OR (fixing_step > 0) the moment-integrated
// arithmetic-average numerator over the single window [sub_start[0], sub_end[0]]:
//   int_a^b f + 1/2 * fixing_step * int_a^b f^2   (Part B; matches the exact daily sum to the gate).
// One entry point so rate() and float_coupon_pv() share the moment/sub-period choice.
template <class Scalar, class FCurve>
Scalar obs_numerator(const RateObservation& o, const FCurve& fc) {
  if (o.fixing_step > 0.0) {
    const double a = o.sub_start[0], b = o.sub_end[0];
    Scalar n = (fc.integral(b) - fc.integral(a)) +
               0.5 * o.fixing_step * curve_forward_sq_integral<Scalar>(fc, a, b);
    if (o.fixing_step3 > 0.0)
      n += (1.0 / 6.0) * o.fixing_step3 * curve_forward_cube_integral<Scalar>(fc, a, b);
    return n;
  }
  return obs_forward_sum<Scalar>(o, fc);
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
  if (o.compounded) {
    // rate = (realized_factor · ∏(1 + r_k dt_k) − 1) / tau_index; empty subs => a fully-realized product.
    if (o.sub_start.empty()) return Scalar((o.realized_factor - 1.0) / o.tau_index);
    return (o.realized_factor * obs_compound_growth<Scalar>(o, fc) - 1.0) / o.tau_index;
  }
  if (o.sub_start.empty()) return Scalar(o.realized / o.tau_index);
  return (obs_numerator<Scalar>(o, fc) + o.realized) / o.tau_index;
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
  const RateObservation& o = c.obs;
  // COMPOUNDED (product) mode -- separate from the arithmetic k-form below so the hot OIS path is
  // untouched. pv = DF(pay) · ((realized_factor·∏(1+r_k dt_k) − 1) + spread·tau_index) · (tau_pay/tau_index).
  if (o.compounded) {
    const double kf = c.tau_pay / o.tau_index;
    const double konst = c.spread * o.tau_index - 1.0;  // the "−1" of the product folds in with the spread
    if (o.sub_start.empty()) return dc.discount(c.pay) * ((o.realized_factor + konst) * kf);
    Scalar a = o.realized_factor * obs_compound_growth<Scalar>(o, fc) + konst;
    return dc.discount(c.pay) * a * kf;
  }
  // Fast path -- a PLAIN single-sub-period coupon (the compounded-OIS shape): one sub-period, no
  // weights, no moment path, no realized, no spread, tau_pay == tau_index (=> k == 1). Then
  //   pv == DF(pay) * (DF(s)/DF(e) - 1)
  // with none of the general k-form's extra scalar work. It is BIT-IDENTICAL to the general branch
  // (a * 1.0, konst == 0) but materially cheaper under AAD, so the risk ladder's per-coupon AAD pass
  // stays fast. This is the templated analogue of the compiled BundleFloatBatch cpn_is_plain fast path.
  if (o.sub_start.size() == 1 && o.weight.empty() && o.fixing_step == 0.0 && o.realized == 0.0 &&
      c.spread == 0.0 && c.tau_pay == o.tau_index) {
    return dc.discount(c.pay) * (fc.discount(o.sub_start[0]) / fc.discount(o.sub_end[0]) - 1.0);
  }
  const double k = c.tau_pay / c.obs.tau_index;
  const double konst = c.obs.realized + c.spread * c.obs.tau_index;
  if (c.obs.sub_start.empty()) return dc.discount(c.pay) * (konst * k);
  Scalar a = obs_numerator<Scalar>(c.obs, fc) + konst;
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
Scalar curve_forward_sq_integral(const Curve& c, double a, double b, int subdiv) {
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

// Composite int_a^b f(u)^3 du (the 3rd-moment term; 2-pt Gauss per sub-interval, enough for the
// small correction it feeds).
template <class Scalar, class Curve>
Scalar curve_forward_cube_integral(const Curve& c, double a, double b, int subdiv) {
  if (b <= a) return Scalar(0.0);
  const double gx = 0.5773502691896257;
  const double H = (b - a) / subdiv;
  Scalar acc{0.0};
  bool first = true;
  for (int s = 0; s < subdiv; ++s) {
    const double mid = a + (s + 0.5) * H, h = 0.5 * H;
    for (int sg = -1; sg <= 1; sg += 2) {
      const Scalar f = c.forward(mid + sg * gx * h);
      const Scalar term = f * f * f * h;
      if (first) { acc = term; first = false; } else acc += term;
    }
  }
  return acc;
}

// Arithmetic average rate over [a,b], moment form: rate = num / tau_index.
template <class Scalar, class Curve>
Scalar moment_average_rate(const Curve& c, double a, double b, double fixing_step, double tau_index,
                           double fixing_step3 = 0.0) {
  Scalar num =
      (c.integral(b) - c.integral(a)) + 0.5 * fixing_step * curve_forward_sq_integral<Scalar>(c, a, b);
  if (fixing_step3 > 0.0) num += (1.0 / 6.0) * fixing_step3 * curve_forward_cube_integral<Scalar>(c, a, b);
  return num / tau_index;
}

}  // namespace swaps::pricing
