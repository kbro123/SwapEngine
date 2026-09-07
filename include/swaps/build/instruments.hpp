// swaps::build — rate-helper-style instrument builders. Each assembles a cal::Instrument (resolved coupon
// schedules + observation windows) from a value date + conventions. Faithful transcription of
// server/compile.py's _ois_coupon / _fixed_coupons / _float_leg / _par_swap / _basis_swap / _xccy_mtm_basis /
// _rate_obs and the FxForward / TurnJump / Portfolio quote assembly. Emits the frozen calc-engine structs.
#ifndef SWAPS_BUILD_INSTRUMENTS_HPP
#define SWAPS_BUILD_INSTRUMENTS_HPP

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "swaps/build/conventions.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/ref_data.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::build {

namespace cal = swaps::calibration;

// Per-period notional for an amortizing / step-up / custom-notional leg. `ns` is EMPTY => unit notional
// (1.0, byte-identical to the pre-amortization builders); size 1 => a constant notional broadcast to every
// period; otherwise its size MUST equal the leg's accrual-period count (one notional per period). The value
// is written to the coupon's constant PV multiplier `scale` (pricing::FloatCoupon::scale / FixedCoupon::
// scale), which the templated kernel (float_coupon_pv / annuity) AND the W-cache batch (BundleFloatBatch /
// BundleFixedLegs fold it into k / tau) already carry through the PV and the analytic Jacobian -- so an
// amortizing leg needs no new struct field, only this schedule fed into the existing `scale`. (A foreign
// converted leg that already uses `scale` for the FX spot would pre-multiply notional x fx into one value.)
inline double notional_at(const std::vector<double>& ns, std::size_t i, std::size_t n) {
  if (ns.empty()) return 1.0;
  if (ns.size() == 1) return ns.front();
  if (ns.size() != n)
    throw std::invalid_argument("notionals: expected empty, size 1, or one entry per accrual period");
  return ns[i];
}

// One compounded-overnight / float coupon over [s, e] (compile._ois_coupon). DF lookups in curve time
// (ACT/365F); accrual + pay on the instrument day count / calendar. `lag` (default inactive) applies an
// optional RFR observation-shift / lookback / lockout to the observation window ONLY; it leaves the payment
// accrual `tau_pay` on the actual [s, e] span and defaults BYTE-IDENTICAL to the plain telescoped bracket.
inline px::FloatCoupon ois_coupon(const Date& vd, const SwapConv& conv, const Date& s, const Date& e,
                                  const std::string& dc, const RfrLag& lag = {}) {
  const double tau = year_frac(dc, s, e, conv.calendar);  // conv.calendar carries BUS/252's business-day count
  const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
  px::FloatCoupon c;
  c.obs = rfr_observation(vd, s, e, dc, lag, conv.calendar);
  c.pay = curve_time(vd, pay);
  c.tau_pay = tau;
  return c;
}

// Fixed annuity leg to `mat`. `fixed_freq` overrides the coupon frequency (default "1Y" reproduces the old
// hard-coded annual leg); `notionals` gives an amortizing/step-up schedule (default empty => unit notional).
inline cal::FixedLeg fixed_coupons(const Date& vd, const SwapConv& conv, const Date& mat, int disc,
                                   const std::string& fixed_freq = "1Y",
                                   const std::vector<double>& notionals = {}) {
  cal::FixedLeg leg;
  leg.discount = disc;
  const auto periods = swap_periods_to(vd, conv.calendar, mat, fixed_freq, conv.bdc, conv.spot_lag);
  for (std::size_t i = 0; i < periods.size(); ++i) {
    const auto& [s, e] = periods[i];
    const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
    px::FixedCoupon fc;
    fc.pay = curve_time(vd, pay);
    fc.tau = year_frac(conv.fixed_dc, s, e, conv.calendar);  // conv.calendar needed only for BUS/252
    fc.scale = notional_at(notionals, i, periods.size());
    leg.coupons.push_back(fc);
  }
  return leg;
}

// Floating leg to `mat` on `freq_tok`/`dc`. `spread` is an additive contractual float-leg spread (rate
// units, e.g. +0.001 = +10bp) applied to every coupon; `notionals` gives an amortizing/step-up schedule.
// Both default to today's behaviour (spread 0, unit notional) so an existing call is byte-identical.
inline cal::FloatLeg float_leg(const Date& vd, const SwapConv& conv, const Date& mat, int forecast, int disc,
                               const std::string& freq_tok, const std::string& dc,
                               int reset_num = -1, int reset_den = -1, double fx_spot = 1.0,
                               double spread = 0.0, const std::vector<double>& notionals = {},
                               const RfrLag& lag = {}) {
  cal::FloatLeg leg;
  leg.forecast = forecast;
  leg.discount = disc;
  leg.reset_num = reset_num;
  leg.reset_den = reset_den;
  leg.fx_spot = fx_spot;
  const auto periods = swap_periods_to(vd, conv.calendar, mat, freq_tok, conv.bdc, conv.spot_lag);
  for (std::size_t i = 0; i < periods.size(); ++i) {
    px::FloatCoupon c = ois_coupon(vd, conv, periods[i].first, periods[i].second, dc, lag);
    c.spread = spread;
    c.scale = notional_at(notionals, i, periods.size());
    leg.coupons.push_back(c);
  }
  return leg;
}

// ---- BOOKED-trade legs: rolled from the trade's EFFECTIVE date, not from spot ----------------------------
// A seasoned (or forward-starting) deal's coupon dates are anchored at its own effective date. Rolling a
// booked trade from today's spot (what the calibration builders above do, correctly, for a NEW par swap)
// invents a different contract: wrong coupon dates, the elapsed part of the current period lost, a wrong
// final stub (-17.6% NPV on a 2020-effective 10y measured against the true schedule). These builders:
//   * roll `freq_tok` from `effective` to `maturity` (swap_periods_between), then
//   * DROP every period whose PAYMENT date is on/before the value date (already settled), and
//   * for a period that is ALREADY ACCRUING (start < value date) attach a fixings-resolvable observation:
//     overnight index -> one FixingDay per business day (scheduled_observation, compounded); term/IBOR
//     index -> ONE fixing at (start − fixing_lag) covering the whole period. The session resolves these
//     against its fixing table before pricing (realized part from real fixings, the rest forecast); an
//     unresolved one is refused by the kernels, never priced as zero.
inline cal::FloatLeg float_leg_from(const Date& vd, const SwapConv& conv, const Date& effective, const Date& mat,
                                    int forecast, int disc, const std::string& freq_tok, const std::string& dc,
                                    const std::string& index, double spread = 0.0,
                                    const std::vector<double>& notionals = {}) {
  cal::FloatLeg leg;
  leg.forecast = forecast;
  leg.discount = disc;
  const auto periods = swap_periods_between(effective, conv.calendar, mat, freq_tok, conv.bdc);
  const Index ix(index);
  const bool overnight = index.empty() ? true : ix.is_overnight();  // no DB entry: assume an RFR (compounded)
  for (std::size_t i = 0; i < periods.size(); ++i) {
    const auto& [s, e] = periods[i];
    const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
    if (!(pay > vd)) continue;  // settled: nothing left to value
    px::FloatCoupon c;
    if (s < vd) {  // accruing: realized part from fixings, forecast part from the curve
      if (overnight) {
        c.obs = scheduled_observation(vd, s, e, "compounded", dc, conv.calendar, index);
      } else {
        const Date fixing = advance_bd(conv.calendar, s, -ix.fixing_lag());
        const double tau = year_frac(dc, s, e, conv.calendar);
        c.obs.fixing_index = index;
        c.obs.tau_index = tau;
        c.obs.fixing_schedule.push_back(px::FixingDay{ordinal(fixing), tau, curve_time(vd, s), curve_time(vd, e), 1.0});
      }
      c.pay = curve_time(vd, pay);
      c.tau_pay = year_frac(dc, s, e, conv.calendar);
    } else {
      c = ois_coupon(vd, conv, s, e, dc);
    }
    c.spread = spread;
    c.scale = notional_at(notionals, i, periods.size());
    leg.coupons.push_back(c);
  }
  return leg;
}

inline cal::FixedLeg fixed_coupons_from(const Date& vd, const SwapConv& conv, const Date& effective, const Date& mat,
                                        int disc, const std::string& fixed_freq = "",
                                        const std::vector<double>& notionals = {}) {
  cal::FixedLeg leg;
  leg.discount = disc;
  const auto periods = swap_periods_between(effective, conv.calendar, mat,
                                            fixed_freq.empty() ? conv.fixed_freq_tok : fixed_freq, conv.bdc);
  for (std::size_t i = 0; i < periods.size(); ++i) {
    const auto& [s, e] = periods[i];
    const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
    if (!(pay > vd)) continue;  // settled
    px::FixedCoupon fc;
    fc.pay = curve_time(vd, pay);
    fc.tau = year_frac(conv.fixed_dc, s, e, conv.calendar);  // the FULL coupon is still owed (dirty value)
    fc.scale = notional_at(notionals, i, periods.size());
    leg.coupons.push_back(fc);
  }
  return leg;
}

// Par OIS/IRS swap (compile._par_swap): fixed leg on the PRODUCT's fixed frequency (conv.fixed_freq_tok,
// from the conventions DB; annual for USD/EUR/GBP/JPY OIS, semi/quarterly for SAR/AUD/CNY/ZAR ...) vs float
// forecasting `fc`, discounting `disc`. Optional: `float_spread` (additive float-leg spread, rate units),
// `notionals` (amortizing/step-up schedule applied to BOTH legs), `fixed_freq` (an explicit fixed-leg
// frequency override; "" = the product's). NOTE: a single `notionals` vector is fed to both legs, so it is
// one-per-period on EACH leg -- for a swap whose float and fixed period counts differ (e.g. quarterly float
// vs annual fixed) build the legs directly with float_leg/fixed_coupons instead.
inline cal::Instrument par_swap(const Date& vd, const SwapConv& conv, const Date& mat, int fc, int disc,
                                double market, double float_spread = 0.0,
                                const std::vector<double>& notionals = {},
                                const std::string& fixed_freq = "") {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = float_leg(vd, conv, mat, fc, disc, conv.float_freq_tok, conv.float_dc, -1, -1, 1.0, float_spread,
                      notionals);
  ins.fixed = fixed_coupons(vd, conv, mat, disc, fixed_freq.empty() ? conv.fixed_freq_tok : fixed_freq, notionals);
  ins.market = market;
  return ins;
}

// Basis swap (compile._basis_swap): quoted leg `fc` vs benchmark leg `bench`, ParSpread. Optional:
// `fwd_spread` (contractual spread on the quoted/fwd leg) and `notionals` (amortizing schedule applied to
// both float legs and the annuity). Defaults reproduce the old swap byte-for-byte.
inline cal::Instrument basis_swap(const Date& vd, const SwapConv& conv, const Date& mat, int fc, int bench,
                                  int disc, double market, double fwd_spread = 0.0,
                                  const std::vector<double>& notionals = {}) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParSpread;
  ins.fwd = float_leg(vd, conv, mat, fc, disc, conv.float_freq_tok, conv.float_dc, -1, -1, 1.0, fwd_spread,
                      notionals);
  ins.bench = float_leg(vd, conv, mat, bench, disc, conv.float_freq_tok, conv.float_dc, -1, -1, 1.0, 0.0,
                        notionals);
  ins.fixed = fixed_coupons(vd, conv, mat, disc, "1Y", notionals);
  ins.market = market;
  return ins;
}

// MtM cross-currency OIS basis swap (compile._xccy_mtm_basis).
inline cal::Instrument xccy_mtm_basis(const Date& vd, const XccyConv& x, const Date& mat, int ci, int foreign,
                                      int fund, double fx_spot, double market) {
  // The XccyConv shares the SwapConv shape for the leg builders (calendar/bdc/spot_lag/freq/dc).
  const SwapConv sc{x.calendar, x.bdc, x.dc, x.dc, x.freq_tok, x.spot_lag, x.pay_lag};
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::XccyMtmBasis;
  ins.fwd = float_leg(vd, sc, mat, ci, ci, x.freq_tok, x.dc);
  ins.bench = float_leg(vd, sc, mat, foreign, ci, x.freq_tok, x.dc);
  ins.mtm = float_leg(vd, sc, mat, fund, fund, x.freq_tok, x.dc, /*reset_num=*/ci, /*reset_den=*/fund, fx_spot);
  cal::FixedLeg fixed;
  fixed.discount = ci;
  for (const auto& [s, e] : swap_periods_to(vd, x.calendar, mat, x.freq_tok, x.bdc, x.spot_lag)) {
    const Date pay = advance_bd(x.calendar, e, x.pay_lag);
    px::FixedCoupon fc;
    fc.pay = curve_time(vd, pay);
    fc.tau = year_frac(x.dc, s, e, x.calendar);  // x.calendar needed only for BUS/252 (EUR/USD xccy never hits it)
    fixed.coupons.push_back(fc);
  }
  ins.fixed = fixed;
  ins.market = market;
  return ins;
}

// A futures/rate `Rate` quote (compile._rate_obs -> observation/scheduled_observation). `a`/`T` are the
// plain single-bracket span fallback (curve time) when no averaging/compounding convention is set.
inline cal::Instrument rate_instrument(int forecast, px::RateObservation obs, double market,
                                       double convexity = 0.0) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = forecast;
  ins.obs = std::move(obs);
  ins.convexity = convexity;
  ins.market = market;
  return ins;
}
inline px::RateObservation plain_rate_obs(double a, double T) {  // single bracket, no convention
  px::RateObservation o;
  o.sub_start = {a};
  o.sub_end = {T};
  o.tau_index = T - a;
  return o;
}

// FX forward (compile_spec FxForward): pins curve `ci`'s DF vs `den` via the outright FX level.
inline cal::Instrument fx_forward(int ci, int den, double fx_spot, double fx_time, double market) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::FxForward;
  ins.fx_num = ci;
  ins.fx_den = den;
  ins.fx_spot = fx_spot;
  ins.fx_time = fx_time;
  ins.market = market;
  return ins;
}

// Turn jump (compile_spec TurnJump): pins turn `turn_index` on curve `ci`.
inline cal::Instrument turn_jump(int ci, int turn_index, double market) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::TurnJump;
  ins.turn_curve = ci;
  ins.turn_index = turn_index;
  ins.market = market;
  return ins;
}

// ---- Convention-object overloads: let a caller pass a build::Convention (from Index::par_convention())
// straight to the builders, so the ground-up flow reads Index -> Convention -> Instrument object-to-object
// instead of threading a raw SwapConv. Each just flattens via conv.resolve(). ------------------------------
inline cal::Instrument par_swap(const Date& vd, const Convention& conv, const Date& mat, int fc, int disc,
                                double market, double float_spread = 0.0,
                                const std::vector<double>& notionals = {},
                                const std::string& fixed_freq = "1Y") {
  return par_swap(vd, conv.resolve(), mat, fc, disc, market, float_spread, notionals, fixed_freq);
}
inline cal::Instrument basis_swap(const Date& vd, const Convention& conv, const Date& mat, int fc, int bench,
                                  int disc, double market, double fwd_spread = 0.0,
                                  const std::vector<double>& notionals = {}) {
  return basis_swap(vd, conv.resolve(), mat, fc, bench, disc, market, fwd_spread, notionals);
}
inline cal::FixedLeg fixed_coupons(const Date& vd, const Convention& conv, const Date& mat, int disc,
                                   const std::string& fixed_freq = "1Y",
                                   const std::vector<double>& notionals = {}) {
  return fixed_coupons(vd, conv.resolve(), mat, disc, fixed_freq, notionals);
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_INSTRUMENTS_HPP
