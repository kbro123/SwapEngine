// swaps::build — rate-helper-style instrument builders. Each assembles a cal::Instrument (resolved coupon
// schedules + observation windows) from a value date + conventions. Faithful transcription of
// server/compile.py's _ois_coupon / _fixed_coupons / _float_leg / _par_swap / _basis_swap / _xccy_mtm_basis /
// _rate_obs and the FxForward / TurnJump / Portfolio quote assembly. Emits the frozen calc-engine structs.
#ifndef SWAPS_BUILD_INSTRUMENTS_HPP
#define SWAPS_BUILD_INSTRUMENTS_HPP

#include <string>
#include <utility>
#include <vector>

#include "swaps/build/conventions.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::build {

namespace cal = swaps::calibration;

// One compounded-overnight / float coupon over [s, e] (compile._ois_coupon). DF lookups in curve time
// (ACT/365F); accrual + pay on the instrument day count / calendar.
inline px::FloatCoupon ois_coupon(const Date& vd, const SwapConv& conv, const Date& s, const Date& e,
                                  const std::string& dc) {
  const double tau = year_frac(dc, s, e);
  const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
  px::FloatCoupon c;
  c.obs.sub_start = {curve_time(vd, s)};
  c.obs.sub_end = {curve_time(vd, e)};
  c.obs.tau_index = tau;
  c.pay = curve_time(vd, pay);
  c.tau_pay = tau;
  return c;
}

inline cal::FixedLeg fixed_coupons(const Date& vd, const SwapConv& conv, const Date& mat, int disc) {
  cal::FixedLeg leg;
  leg.discount = disc;
  for (const auto& [s, e] : swap_periods_to(vd, conv.calendar, mat, "1Y", conv.bdc, conv.spot_lag)) {
    const Date pay = advance_bd(conv.calendar, e, conv.pay_lag);
    px::FixedCoupon fc;
    fc.pay = curve_time(vd, pay);
    fc.tau = year_frac(conv.fixed_dc, s, e);
    leg.coupons.push_back(fc);
  }
  return leg;
}

inline cal::FloatLeg float_leg(const Date& vd, const SwapConv& conv, const Date& mat, int forecast, int disc,
                               const std::string& freq_tok, const std::string& dc,
                               int reset_num = -1, int reset_den = -1, double fx_spot = 1.0) {
  cal::FloatLeg leg;
  leg.forecast = forecast;
  leg.discount = disc;
  leg.reset_num = reset_num;
  leg.reset_den = reset_den;
  leg.fx_spot = fx_spot;
  for (const auto& [s, e] : swap_periods_to(vd, conv.calendar, mat, freq_tok, conv.bdc, conv.spot_lag))
    leg.coupons.push_back(ois_coupon(vd, conv, s, e, dc));
  return leg;
}

// Par OIS/IRS swap (compile._par_swap): annual fixed vs float forecasting `fc`, discounting `disc`.
inline cal::Instrument par_swap(const Date& vd, const SwapConv& conv, const Date& mat, int fc, int disc,
                                double market) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = float_leg(vd, conv, mat, fc, disc, conv.float_freq_tok, conv.float_dc);
  ins.fixed = fixed_coupons(vd, conv, mat, disc);
  ins.market = market;
  return ins;
}

// Basis swap (compile._basis_swap): quoted leg `fc` vs benchmark leg `bench`, ParSpread.
inline cal::Instrument basis_swap(const Date& vd, const SwapConv& conv, const Date& mat, int fc, int bench,
                                  int disc, double market) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParSpread;
  ins.fwd = float_leg(vd, conv, mat, fc, disc, conv.float_freq_tok, conv.float_dc);
  ins.bench = float_leg(vd, conv, mat, bench, disc, conv.float_freq_tok, conv.float_dc);
  ins.fixed = fixed_coupons(vd, conv, mat, disc);
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
    fc.tau = year_frac(x.dc, s, e);
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

}  // namespace swaps::build

#endif  // SWAPS_BUILD_INSTRUMENTS_HPP
