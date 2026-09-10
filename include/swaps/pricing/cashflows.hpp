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
#include <string>
#include <vector>

namespace swaps::pricing {

// One fixing day of an RFR observation's schedule (E2 — fixings as a pricing context). Its fixing date
// (an integer serial the caller defines), the index-day-count accrual of its span, and — for a day that
// turns out to be in the FUTURE — the forecast sub-period [t_start, t_end] (curve time) + weight. When an
// observation carries a schedule, the engine RESOLVES it against a FixingTable (swaps/pricing/fixings.hpp)
// at build + on each table update, instead of the caller baking `realized`. Plain data (no deps).
struct FixingDay {
  int fixing_date = 0;
  double accrual = 0.0;
  double t_start = 0.0;
  double t_end = 0.0;
  double weight = 1.0;
};

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

  // E2 (fixings as a pricing context): when `fixing_schedule` is non-empty, the fields above (`realized`
  // / `realized_factor` and the forecast sub-periods) are RESOLVED from the pricing context's FixingTable
  // (swaps/pricing/fixings.hpp resolve_into) at build + on each table update — past days come from the
  // table (throwing MissingFixing if absent), future days become forecast sub-periods. Empty schedule =>
  // legacy already-baked path, untouched. `fixing_index` names the series in the table. These are read
  // only during resolution, never on the price/W-cache hot path.
  std::string fixing_index;
  std::vector<FixingDay> fixing_schedule;
  // Set by resolve_into() once the schedule above has been split into realized/forecast against a fixing
  // table. A schedule-carrying observation that is NOT resolved has empty sub-periods and realized = 0 --
  // it would price its whole period at a zero rate -- so the pricing kernels refuse it (see float_coupon_pv
  // / CompiledBook::push_obs) instead of returning a silent zero.
  bool resolved = false;
};

// One floating coupon: an observation, discounted at its own pay date on its own accrual basis.
// `tau_pay` is the PAYMENT accrual (the coupon's day count) and is deliberately distinct from
// `obs.tau_index` (the index's) — that separation is what makes 30/360-vs-ACT/360 correct.
struct FloatCoupon {
  RateObservation obs;
  double pay = 0.0;      // payment (discount-curve) time
  double tau_pay = 0.0;  // payment accrual, coupon day count
  double spread = 0.0;   // additive contractual spread (outside the index), may be 0
  // CONSTANT PV multiplier (multi-currency). For a foreign leg converted into the PV currency this is
  // the FX spot; being a build-time constant it scales the coupon PV WITHOUT touching the exp(-Wx)
  // linear map (it rides inside the W-cache's per-coupon `k`). Default 1.0 => byte-identical to before,
  // and the plain fast path below stays on x*1.0 (exact) only when scale == 1.0.
  double scale = 1.0;
  // MtM (mark-to-market cross-currency) FX-RESET observation time for this coupon's notional. < 0 => use
  // the period start (obs.sub_start.front()). Only consulted by xccy_mtm_leg_pv, whose notional
  // N = fx_spot · DF_num(reset_time)/DF_den(reset_time) is CURVE-DEPENDENT (a DF ratio of two curves) — a
  // product of registered DFs like every other coupon, so it rides the W-cache (BundleFloatBatch::add_mtm).
  double reset_time = -1.0;
  // ACCRUAL PERIOD (curve times) -- carried by the builders since 2026-09-10 (E3-S2/G4). When set, an MtM
  // coupon's notional exchanges (−DF(start), +DF(end)) and its default reset sit on the accrual period, not
  // on the observation window: a SEASONED coupon's window shrinks to its first future fixing on resolution,
  // which used to book an exchange that had already settled at a future date and evaluate a past reset at
  // DF ≡ 1. accrual_start < 0 = the period started in the past (the initial exchange has settled).
  bool accrual_set = false;
  double accrual_start = 0.0, accrual_end = 0.0;
  // FIXED FX reset (same quoting as the leg's fx_spot) for an MtM coupon whose reset date is in the past:
  // the notional is this number, not a curve-implied forward. < 0 = not fixed (the reset must then be >= 0).
  double reset_fx = -1.0;
};

// True iff an MtM coupon is SEASONED -- a fixed FX reset, a settled initial exchange or a past reset date --
// so it must price on the templated kernel (xccy_mtm_leg_pv), never on the W-cache batch (whose value is a
// product of registered DFs at non-negative times). Shared by the hybrid router and the compiled guard.
inline bool mtm_coupon_is_seasoned(const FloatCoupon& c) {
  if (c.reset_fx >= 0.0) return true;
  const double s = c.accrual_set ? c.accrual_start : (c.obs.sub_start.empty() ? -1.0 : c.obs.sub_start.front());
  if (s < 0.0) return true;  // strictly in the past: an exchange dated today (t = 0, DF = 1) is still to be paid
  const double reset = (c.reset_time >= 0.0) ? c.reset_time : s;
  return reset < 0.0;
}

// One fixed coupon. The RATE is supplied by the instrument/quote, not stored here, so this type
// serves both a par-rate annuity and a fixed leg at a contract rate.
struct FixedCoupon {
  double pay = 0.0;
  double tau = 0.0;
  double scale = 1.0;  // constant PV multiplier (FX spot for a foreign annuity); default 1.0 unchanged
};

// ∫f² / ∫f³ over the curve for the moment path (defined at the end of this header). KNOT-ALIGNED since
// 2026-09-10 (E3-R4 / G2): the window is split at the curve's pieces (region knots, de Boor breakpoints,
// turn edges, the base chain's pieces) and each piece gets `subdiv` (default 1) Gauss-Legendre panels of 4 (f²)
// / 5 (f³) points -- exact for the cubic pieces every polynomial scheme produces (one panel suffices; the
// cubic term's node rows are a per-tick GEMV in the compiled batch, so fewer nodes is cheaper there). The old fixed 32×2 / 8×2 rule was
// not aligned and lost 3.3e-3 of ∫f² (1.7e-7 of the rate) on the shipped Flat-front Fed-funds shape with
// policy steps inside the window. The compiled batch (compiled_book.hpp) builds its quadratic forms from
// the SAME node set (moment_gauss_nodes), so both kernels agree to rounding.
template <class Scalar, class Curve>
Scalar curve_forward_sq_integral(const Curve& c, double a, double b, int subdiv = 1);
template <class Scalar, class Curve>
Scalar curve_forward_cube_integral(const Curve& c, double a, double b, int subdiv = 1);

// Σ_k w_k · (DF(s_k)/DF(e_k) − 1) — the curve-dependent numerator ONLY.
// Precondition: at least one sub-period (so the accumulator can be seeded from a curve-dependent
// term and carry derivatives — see the AAD SAFETY note at the top of this header).
// THE curve growth over sub-period k: DF(s_k)/DF(e_k) − 1. Every rate, coupon and compounding form below
// is built out of this one quantity, so it is named once rather than re-typed at each use (2026-09-10: it
// was written out four times in this file; the same shape, written four times in the observation BUILDERS,
// was wrong in all four — ASSUMPTIONS.md E2). Returns a curve-dependent Scalar, so an accumulator seeded
// from it carries derivatives under AAD (see the AAD SAFETY note at the top of this header).
template <class Scalar, class FCurve>
Scalar sub_growth(const RateObservation& o, std::size_t k, const FCurve& fc) {
  return fc.discount(o.sub_start[k]) / fc.discount(o.sub_end[k]) - 1.0;
}

template <class Scalar, class FCurve>
Scalar obs_forward_sum(const RateObservation& o, const FCurve& fc) {
  assert(!o.sub_start.empty());
  assert(o.sub_start.size() == o.sub_end.size());
  assert(o.weight.empty() || o.weight.size() == o.sub_start.size());
  const bool weighted = !o.weight.empty();
  Scalar num = sub_growth<Scalar>(o, 0, fc);
  if (weighted) num = num * o.weight[0];
  for (std::size_t k = 1; k < o.sub_start.size(); ++k) {
    Scalar t = sub_growth<Scalar>(o, k, fc);
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
    Scalar t = sub_growth<Scalar>(o, k, fc);
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
    // weight[0] (if any) is the constant index-accrual / curve-time ratio of the window's days (365/360 for an
    // ACT/360 index on the ACT/365F curve clock) — build::moment_observation; the daily path carries it per day.
    if (!o.weight.empty()) n *= o.weight[0];
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
  // A fixings-resolvable coupon that has NOT been resolved has empty sub-periods and realized 0 /
  // realized_factor 1: every branch below would price its whole period at a zero rate. Refuse it FIRST
  // (before the compounded early-return), never return that silent zero.
  if (!o.fixing_schedule.empty() && !o.resolved)
    throw std::runtime_error(
        "float_coupon_pv: a fixings-resolvable coupon was priced before resolution against a fixing table "
        "(its realized part would silently be zero) -- attach fixings / set the evaluation date first");
  // COMPOUNDED (product) mode -- separate from the arithmetic k-form below so the hot OIS path is
  // untouched. pv = DF(pay) · ((realized_factor·∏(1+r_k dt_k) − 1) + spread·tau_index) · (tau_pay/tau_index).
  if (o.compounded) {
    const double kf = c.tau_pay / o.tau_index * c.scale;  // FX scale folds into the constant k (× 1.0 exact)
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
      c.spread == 0.0 && c.tau_pay == o.tau_index && c.scale == 1.0) {
    return dc.discount(c.pay) * sub_growth<Scalar>(o, 0, fc);
  }
  const double k = c.tau_pay / c.obs.tau_index * c.scale;  // FX scale folds into the constant k (× 1.0 exact)
  const double konst = c.obs.realized + c.spread * c.obs.tau_index;
  if (c.obs.sub_start.empty()) return dc.discount(c.pay) * (konst * k);
  Scalar a = obs_numerator<Scalar>(c.obs, fc) + konst;
  return dc.discount(c.pay) * a * k;
}

// PV of a floating leg for unit notional.
template <class Scalar, class FCurve, class DCurve>
Scalar float_leg_pv(const std::vector<FloatCoupon>& leg, const FCurve& fc, const DCurve& dc) {
  // An empty leg is worth nothing -- it must NOT dereference leg[0] (the assert is compiled out in
  // release, where this was a null read). Reachable from a JSON book with "float_coupons": [].
  if (leg.empty()) return Scalar(0.0);
  Scalar pv = float_coupon_pv<Scalar>(leg[0], fc, dc);
  for (std::size_t i = 1; i < leg.size(); ++i) pv += float_coupon_pv<Scalar>(leg[i], fc, dc);
  return pv;
}

// PV of a MARK-TO-MARKET (FX-resettable-notional) cross-currency leg (design: Phase 4). The notional of
// coupon i resets to the FX forward
//     N_i = fx_spot · DF_num(reset_i) / DF_den(reset_i)          (num = foreign, den = domestic discount)
// so, modelling each period as a self-financing one-period loan of N_i at the funding index (borrow N_i at
// the period start s_i, repay + interest at the end e_i), the leg value is
//     PV = Σ_i N_i · [ float_coupon_pv(c_i, fc, dc) + DF_dc(e_i) − DF_dc(s_i) ].
// The bracket is the per-period (interest + notional-exchange) value; for a funding-index-FLAT leg
// (dc == fc, spread == 0) each bracket is exactly 0, so the MtM funding leg is PAR — which is why an MtM
// xccy basis equals the constant-notional basis in DETERMINISTIC curves (their difference is an FX-vol
// convexity term, out of scope for a curve engine). N_i is CURVE-DEPENDENT (a DF ratio), so the coupon PV
// is a product of registered discount factors -- which the W-cache batch prices EXACTLY too
// (BundleFloatBatch::add_mtm, 2026-09-09; the compiled book since 2026-09-10). This templated form is the
// reference the compiled path is parity-tested against, and the ONLY path for a SEASONED coupon (fixed FX
// reset / settled exchange, mtm_coupon_is_seasoned). AAD flows through: N_i, float_coupon_pv and the DF
// differences all carry derivatives; the accumulator seeds from the first (curve-dependent) contribution.
template <class Scalar, class FCurve, class DCurve, class NumCurve, class DenCurve>
Scalar xccy_mtm_leg_pv(const std::vector<FloatCoupon>& leg, double fx_spot, const FCurve& fc,
                       const DCurve& dc, const NumCurve& numc, const DenCurve& denc) {
  if (leg.empty()) return Scalar(0.0);
  auto contrib = [&](const FloatCoupon& c) -> Scalar {
    // The notional exchanges sit on the ACCRUAL period when the coupon carries it (the builders do); a coupon
    // without it falls back to its observation window -- and a fully-fixed one without either has no dates
    // to place them on: refuse loudly rather than read .front() of an empty vector.
    double s, e;
    if (c.accrual_set) {
      s = c.accrual_start;
      e = c.accrual_end;
    } else {
      if (c.obs.sub_start.empty() || c.obs.sub_end.empty())
        throw std::runtime_error(
            "xccy_mtm_leg_pv: a fully-fixed MtM coupon has no accrual period or observation window to place "
            "its notional exchanges on (set accrual_start/accrual_end)");
      s = c.obs.sub_start.front();
      e = c.obs.sub_end.back();
    }
    const double reset = (c.reset_time >= 0.0) ? c.reset_time : s;  // notional fixes at the period start
    // The coupon's value: the float PV, the final exchange +DF(e), and the initial exchange −DF(s) ONLY if it
    // has not settled yet (s > 0). A seasoned coupon's initial exchange is cash already paid.
    Scalar v = float_coupon_pv<Scalar>(c, fc, dc) + dc.discount(e);
    if (s >= 0.0) v -= dc.discount(s);  // a start dated today is still a flow to pay (DF = 1)
    if (c.reset_fx >= 0.0) return v * c.reset_fx;  // the notional was FIXED at the reset
    if (reset < 0.0)
      throw std::runtime_error(
          "xccy_mtm_leg_pv: the MtM notional reset at t=" + std::to_string(reset) +
          " is in the past; supply reset_fx (the fixed FX rate) -- a curve-implied forward at a negative time is "
          "not defined (the curve would silently discount at 1)");
    const Scalar N = fx_spot * (numc.discount(reset) / denc.discount(reset));  // FX-forward notional
    return N * v;
  };
  Scalar pv = contrib(leg[0]);
  for (std::size_t i = 1; i < leg.size(); ++i) pv += contrib(leg[i]);
  return pv;
}

// Annuity per unit rate and unit notional: Σ DF_dc(pay_i) · tau_i · scale_i (scale = FX spot, default 1).
template <class Scalar, class DCurve>
Scalar annuity(const std::vector<FixedCoupon>& leg, const DCurve& dc) {
  if (leg.empty()) return Scalar(0.0);  // no coupons, no annuity (never dereference leg[0] in release)
  Scalar a = dc.discount(leg[0].pay) * (leg[0].tau * leg[0].scale);
  for (std::size_t i = 1; i < leg.size(); ++i) a += dc.discount(leg[i].pay) * (leg[i].tau * leg[i].scale);
  return a;
}

// Quote transform `ParRate`: float_leg_pv / annuity.
template <class Scalar, class FCurve, class DCurve>
Scalar par_rate(const std::vector<FloatCoupon>& float_leg, const std::vector<FixedCoupon>& fixed_leg,
                const FCurve& fc, const DCurve& dc) {
  return float_leg_pv<Scalar>(float_leg, fc, dc) / annuity<Scalar>(fixed_leg, dc);
}
// The general form (E6.1c, 2026-09-10 -- the ONE ParRate formula, used by instrument_model_quote): the float
// leg and the fixed annuity may discount on DIFFERENT curves (a cross-currency or CSA-split instrument).
template <class Scalar, class FCurve, class DFloat, class DFixed>
Scalar par_rate(const std::vector<FloatCoupon>& float_leg, const std::vector<FixedCoupon>& fixed_leg,
                const FCurve& fc, const DFloat& dc_float, const DFixed& dc_fixed) {
  return float_leg_pv<Scalar>(float_leg, fc, dc_float) / annuity<Scalar>(fixed_leg, dc_fixed);
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
// The general form (E6.1c, 2026-09-10 -- the ONE ParSpread formula, used by instrument_model_quote): each leg
// carries its own discount curve. The sign convention lives HERE and nowhere else: quote = (pv_bench − pv_fwd)/A.
template <class Scalar, class FwdCurve, class FwdDisc, class BenchCurve, class BenchDisc, class AnnDisc>
Scalar par_spread(const std::vector<FloatCoupon>& fwd_leg, const std::vector<FloatCoupon>& bench_leg,
                  const std::vector<FixedCoupon>& annuity_leg, const FwdCurve& fwd, const FwdDisc& fwd_dc,
                  const BenchCurve& bench, const BenchDisc& bench_dc, const AnnDisc& ann_dc) {
  return (float_leg_pv<Scalar>(bench_leg, bench, bench_dc) - float_leg_pv<Scalar>(fwd_leg, fwd, fwd_dc)) /
         annuity<Scalar>(annuity_leg, ann_dc);
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

// ---- knot-aligned Gauss quadrature for the moment path ------------------------------------------------
// The curve's PIECES (breakpoints between which f is one analytic piece): a handle exposes pieces_into()
// (regions + turn edges + the base chain), a bare ModularCurve exposes pieces(); anything else is treated
// as a single piece with a fine composite rule.
template <class Curve>
void collect_curve_pieces(const Curve& c, std::vector<double>& out) {
  if constexpr (requires(const Curve& cc, std::vector<double>& o) { cc.pieces_into(o); }) {
    c.pieces_into(out);
  } else if constexpr (requires(const Curve& cc) { cc.pieces(); }) {
    const std::vector<double> p = c.pieces();
    out.insert(out.end(), p.begin(), p.end());
  }
}
// Gauss-Legendre nodes/weights over [a,b] split at the pieces strictly inside it: `panels` panels of `order`
// points (4 or 5) per sub-interval. With no known pieces, 16 panels over the whole window. This is THE rule
// both kernels use (compiled_book.hpp builds its quadratic forms from exactly these nodes).
inline void moment_gauss_nodes(std::vector<double> pieces, double a, double b, int order, int panels,
                               std::vector<double>& nodes, std::vector<double>& wts) {
  static const double gx4[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  static const double gw4[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  static const double gx5[5] = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640};
  static const double gw5[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};
  const double* gx = order == 5 ? gx5 : gx4;
  const double* gw = order == 5 ? gw5 : gw4;
  const int n = order == 5 ? 5 : 4;
  std::vector<double> pts;
  pts.push_back(a);
  std::sort(pieces.begin(), pieces.end());
  for (double p : pieces)
    if (p > a + 1e-12 && p < b - 1e-12 && p > pts.back() + 1e-12) pts.push_back(p);
  pts.push_back(b);
  const bool known = pts.size() > 2 || !pieces.empty();
  const int per = known ? panels : 16;
  nodes.clear();
  wts.clear();
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    const double H = (pts[i + 1] - pts[i]) / per;
    for (int s = 0; s < per; ++s) {
      const double mid = pts[i] + (s + 0.5) * H, h = 0.5 * H;
      for (int k = 0; k < n; ++k) {
        nodes.push_back(mid + gx[k] * h);
        wts.push_back(gw[k] * h);
      }
    }
  }
}

// int_a^b f(u)^2 du, knot-aligned (4-pt Gauss per panel: exact for the square of a cubic piece).
template <class Scalar, class Curve>
Scalar curve_forward_sq_integral(const Curve& c, double a, double b, int subdiv) {
  if (b <= a) return Scalar(0.0);
  std::vector<double> pieces, nodes, wts;
  collect_curve_pieces(c, pieces);
  moment_gauss_nodes(std::move(pieces), a, b, 4, subdiv, nodes, wts);
  Scalar acc{0.0};
  bool first = true;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const Scalar f = c.forward(nodes[k]);
    const Scalar term = f * f * wts[k];
    if (first) { acc = term; first = false; } else acc += term;
  }
  return acc;
}

// int_a^b f(u)^3 du, knot-aligned (5-pt Gauss per panel: exact for the cube of a cubic piece).
template <class Scalar, class Curve>
Scalar curve_forward_cube_integral(const Curve& c, double a, double b, int subdiv) {
  if (b <= a) return Scalar(0.0);
  std::vector<double> pieces, nodes, wts;
  collect_curve_pieces(c, pieces);
  moment_gauss_nodes(std::move(pieces), a, b, 5, subdiv, nodes, wts);
  Scalar acc{0.0};
  bool first = true;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const Scalar f = c.forward(nodes[k]);
    const Scalar term = f * f * f * wts[k];
    if (first) { acc = term; first = false; } else acc += term;
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
