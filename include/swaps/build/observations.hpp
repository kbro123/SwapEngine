// swaps::build — the observation sub-period windows an instrument's Rate quote looks at (the subtlest part
// of the port). Faithful transcription of server/conventions.py observation()/scheduled_observation()/
// business_days()/daily_periods(). Emits the engine's pricing::RateObservation directly.
//
// rate = ( Σ_k w_k·[DF(s_k)/DF(e_k) − 1] + realized ) / tau_index. AVERAGED (1M): one bracket per fixing day,
// weight = accrual-earned / index-year-fraction(observation window) (obs_weight). COMPOUNDED (3M): one
// telescoped bracket; a partial-fix prefix rides
// in as weight=1+r·τ_past, realized=f-1 (avoids RateObservation::compounded, which the W-cache rejects).
#ifndef SWAPS_BUILD_OBSERVATIONS_HPP
#define SWAPS_BUILD_OBSERVATIONS_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::build {

namespace px = swaps::pricing;

// Fixing dates are Unix-day serials (Date::serial(), days since 1970-01-01) — the engine's ONE integer date
// convention (pricing/fixings.hpp, market/fixing_series.hpp, cb meetings). Until 2026-09-09 this file emitted
// Python proleptic ordinals (+719163) while FixingSeries used serials, so the two fixing stores disagreed.

// Fixing dates in [start, end) on the index calendar (conventions.business_days). The id is REQUIRED; the DB
// calendar "NONE" is the explicit weekends-only calendar (an empty id used to mean that silently).
inline std::vector<Date> business_days(const Date& start, const Date& end, const std::string& cal) {
  if (cal.empty()) throw std::invalid_argument("business_days: calendar id required (\'NONE\' = weekends only)");
  std::vector<Date> days;
  for (Date d = start; d < end; d = d.plus_days(1)) {
    const bool bd = is_business_day(cal, d);
    if (bd) days.push_back(d);
  }
  return days;
}
// ONE fixing day of an overnight window: the FIXING period whose rate applies, and the ACCRUAL the window
// actually earns at that rate. The two differ only at the ends of the window, and that difference is the
// whole point of this type (E3, 2026-09-10).
//
// A window that STARTS ON A NON-BUSINESS DAY is the case that matters, and it is not exotic: a CME 30-Day
// Fed Funds future references the CALENDAR month, so its window starts on the 1st whether or not that is a
// business day -- roughly a third of contract months. The rate applying on Saturday 1 August is the fixing
// published for Friday 31 July, which runs 31 Jul -> 3 Aug and earns the window 2 of its 3 days.
//
// Until 2026-09-10 this enumerated the business days INSIDE [start, end), so those leading days simply
// vanished from the sum while tau_index still spanned the whole window: a flat 4 % curve priced the August
// 2026 FF future at 3.6910 % instead of 3.9456 %, exactly 29/31 of it (-25 bp), and a Sunday start lost
// 29/30. See ASSUMPTIONS.md E3.
struct FixingPeriod {
  Date fix_start, fix_end;  // the fixing's own overnight window -- the rate is DF(fix_start)/DF(fix_end) - 1
  Date acc_start, acc_end;  // the part of it this observation window earns
};

// The fixing periods covering [start, end) on `cal` (conventions.daily_periods). The first fixing is the
// business day ON OR BEFORE `start`, so no accrual is lost when the window opens on a weekend or holiday;
// when `start` IS a business day this is byte-identical to enumerating the business days within it.
inline std::vector<FixingPeriod> daily_periods(const Date& start, const Date& end, const std::string& cal) {
  std::vector<FixingPeriod> out;
  if (!(start < end)) return out;
  Date first = start;
  while (!is_business_day(cal, first)) first = first.plus_days(-1);  // the fixing that applies on `start`
  const auto rest = business_days(start, end, cal);
  std::vector<Date> fix;
  fix.reserve(rest.size() + 1);
  fix.push_back(first);
  for (const Date& d : rest)
    if (first < d) fix.push_back(d);
  out.reserve(fix.size());
  for (std::size_t i = 0; i < fix.size(); ++i) {
    const Date fs = fix[i];
    // A fixing's window ALWAYS runs to the next business day -- for the last fixing that may sit beyond
    // `end`, and it must, because the rate published on Friday is the three-day rate whether or not the
    // window closes on the Sunday. Truncating it to `end` would price a two-day rate instead, which is the
    // same error as dropping a leading day, mirrored: the window's END clips the ACCRUAL, never the FIXING.
    Date fe = fs.plus_days(1);
    while (!is_business_day(cal, fe)) fe = fe.plus_days(1);
    const Date as = (fs < start) ? start : fs;  // the leading fixing accrues only from the window's start
    const Date ae = (fe < end) ? fe : end;      // the trailing fixing accrues only to the window's end
    out.push_back(FixingPeriod{fs, fe, as, ae});
  }
  return out;
}

// The obs for a futures-style Rate quote (conventions.observation). `realized_pct` in PERCENT.
// THE observation weight (fixed 2026-09-10, item 17): a bracket contributes DF(s)/DF(e) − 1, which is the
// window's GROWTH, not a rate. It becomes "this day's fixing times the accrual it earns" only when divided by
// the window's length IN THE INDEX'S OWN DAY COUNT:
//     w = accrual_earned / index_year_fraction(observation window),   rate = Σ w·(DF ratio − 1) / tau_index.
// Until now these builders divided by the window's CURVE-TIME length instead, so every averaged overnight leg
// (and every lookback/lockout coupon) came out (index dc)/(curve dc) too high — exactly 365/360 = +1.389 % for
// an ACT/360 index on the ACT/365F curve clock: a flat 4 % curve produced a 400.04 bp average where the daily
// EFFR fixings it implies average 394.56 bp. The QuantLib oracle for averaged futures never saw it because its
// fixture (tests/reference_multicurrency.hpp avg_future_obs_idx) computes this weight correctly, and
// BuildInstruments.ObservationWindowsMatchPython pinned the wrong number as "C++/Python parity" — parity with
// the web compiler, which carries the same error (TASKS-API §A0.5).
// Where the observation window IS the accrual window (a plain averaged leg) this is exactly 1.
inline double obs_weight(const std::string& dc, const std::string& cal, const Date& acc_start,
                         const Date& acc_end, const Date& obs_start, const Date& obs_end) {
  const double obs = year_frac(dc, obs_start, obs_end, cal);
  return obs > 0.0 ? year_frac(dc, acc_start, acc_end, cal) / obs : 1.0;
}

inline px::RateObservation observation(const Date& vd, const Date& start, const Date& end,
                                       const std::string& accrual, double realized_pct,
                                       const std::string& dc, const std::string& cal) {  // both REQUIRED (P2)
  const double tau = year_frac(dc, start, end, cal);  // `cal` is ignored unless dc==BUS/252
  const Date fwd_from = (start > vd) ? start : vd;
  const double tau_past = (start < vd) ? year_frac(dc, start, vd, cal) : 0.0;
  const double r = realized_pct / 100.0;
  px::RateObservation o;
  if (accrual == "averaged") {
    bool all_one = true;
    for (const auto& p : daily_periods(fwd_from, end, cal)) {
      const double ts = curve_time(vd, p.fix_start), te = curve_time(vd, p.fix_end);
      // The fixing's rate, earning the accrual THIS window takes from it: 1 for every interior day, and a
      // fraction only where the window opens or closes mid-fixing (a non-business start -- see FixingPeriod).
      const double w = obs_weight(dc, cal, p.acc_start, p.acc_end, p.fix_start, p.fix_end);
      o.sub_start.push_back(ts);
      o.sub_end.push_back(te);
      o.weight.push_back(w);
      if (std::abs(w - 1.0) > 1e-12) all_one = false;
    }
    o.realized = r * tau_past;
    o.tau_index = tau;
    if (all_one) o.weight.clear();  // empty == all-ones (bit-exact fast path)
    return o;
  }
  // compounded: one telescoped bracket
  o.sub_start.push_back(curve_time(vd, fwd_from));
  o.sub_end.push_back(curve_time(vd, end));
  o.tau_index = tau;
  if (tau_past > 0.0) {
    const double f = 1.0 + r * tau_past;
    o.weight.push_back(f);
    o.realized = f - 1.0;
  }
  return o;
}

// ---- RFR observation-timing conventions (Priority-2: fixing lag / lookback / lockout) --------------------
// Advance `n` observation business days (n may be negative) using the SAME weekend-only-when-empty rule as

// MOMENT-path observation of a daily ARITHMETIC-AVERAGE window [start, end) (docs/bezier-and-moments.md Part B):
// one bracket in curve time plus the calendar-derived day-count moments fixing_step = Σ_d dt_d²/(b−a) and
// fixing_step3 = Σ_d dt_d³/(b−a) over the index calendar's business days (weekends give 3-day accruals). The engine
// then prices the average from curve moments instead of ~250 daily sub-periods (a documented ~5e-9 relative
// approximation of the daily sum; `observation(..., "averaged", ...)` remains the exact daily path). Spot-start only.
inline px::RateObservation moment_observation(const Date& vd, const Date& start, const Date& end,
                                              const std::string& dc, const std::string& cal) {
  if (start < vd) throw std::invalid_argument("moment_observation: a partially-fixed window needs the daily path (fixings)");
  px::RateObservation o;
  const double a = curve_time(vd, start), b = curve_time(vd, end);
  // The average of the daily fixings is Σ(e^{δ_d} − 1)/tau_index (each day's growth over the period's index
  // accrual — obs_weight above is 1 here, the window being the accrual). The moment expansion replaces that
  // daily sum by ∫f plus the day-count moments. It is valid only while each day's index accrual is the SAME
  // multiple of its curve-time step (true for any ACT-based day count, false for 30/360, which is refused
  // rather than approximated twice) — that is what makes ∫f / tau_index the correctly annualised average.
  // The expansion assumes every day is a WHOLE fixing period, so a window opening mid-fixing (a non-business
  // start, e.g. the CME FF contract month) is refused rather than approximated twice -- the exact daily path
  // handles it. This is the same refusal the day-count check below makes, for the same reason.
  if (!is_business_day(cal, start))
    throw std::invalid_argument(
        "moment_observation: the window starts on a non-business day, so its first fixing is only partly "
        "earned; use the exact daily path (observation(..., \"averaged\", ...))");
  double s2 = 0.0, s3 = 0.0, rmin = 1e300, rmax = -1e300;
  for (const auto& p : daily_periods(start, end, cal)) {
    const double dt = curve_time(vd, p.fix_end) - curve_time(vd, p.fix_start);
    const double r = year_frac(dc, p.fix_start, p.fix_end, cal) / dt;
    s2 += dt * dt; s3 += dt * dt * dt;
    rmin = std::min(rmin, r); rmax = std::max(rmax, r);
  }
  if (rmax - rmin > 1e-10 * std::max(1.0, rmax))
    throw std::invalid_argument("moment_observation: the index day count '" + dc + "' is not a constant multiple of curve time; use the daily path");
  o.sub_start = {a}; o.sub_end = {b};
  o.tau_index = year_frac(dc, start, end, cal);
  o.fixing_step = s2 / (b - a);
  o.fixing_step3 = s3 / (b - a);
  return o;
}

inline Date advance_obs_bd(const std::string& cal, Date d, int n) {
  if (cal.empty()) throw std::invalid_argument("advance_obs_bd: calendar id required (\'NONE\' = weekends only)");
  const int step = n >= 0 ? 1 : -1;
  for (int left = std::abs(n); left; ) {
    d = d.plus_days(step);
    const bool bd = is_business_day(cal, d);
    if (bd) --left;
  }
  return d;
}

// RFR observation-timing style for a compounded overnight coupon. Default None reproduces the plain single
// telescoped bracket (byte-identical to today's ois_coupon). Mechanics (pricing/cashflows.hpp RateObservation):
//   Shift    : OBSERVATION SHIFT -- the whole [s,e] window shifts back `days` business days for BOTH the rate
//              observation AND the accrual. Both the DF-ratio endpoints and the accrual move together, so
//              daily compounding still telescopes to ONE arithmetic bracket over the shifted window
//              (compounded stays false) -- it remains W-cacheable and calibration-safe.
//   Lookback : the rate is observed `days` business days earlier but applied over the ACTUAL day's accrual;
//              this does NOT telescope -> compounded product, one daily bracket per business day, the forecast
//              window looked back and the weight = actual_accrual / looked_back_curve_time.
//   Lockout  : the last `days` business days all re-use the rate observed on the lockout-start day; does NOT
//              telescope -> compounded product, daily brackets with the tail frozen to the lockout day's window.
enum class RfrStyle { None, Shift, Lookback, Lockout };
struct RfrLag {
  RfrStyle style = RfrStyle::None;
  int days = 0;          // business-day lag / lockout length (<=0 or style==None => plain single bracket)
  std::string cal = "";  // observation calendar ("" => weekends only), for the daily/shift stepping
  bool active() const { return style != RfrStyle::None && days > 0; }
};

// Build the RateObservation for a compounded overnight coupon over [s, e] on index day count `dc`, applying
// an optional RFR observation-timing convention. `lag` inactive (the default) yields EXACTLY the plain single
// telescoped bracket ois_coupon has always built (sub_start={ct(s)}, sub_end={ct(e)}, tau_index=tau(s,e)).
// `accr_cal` is the ACCRUAL calendar (the product/index calendar), needed ONLY by BUS/252 to count business
// days; every other day count ignores it, so `accr_cal=""` keeps every existing call byte-identical. It is
// distinct from `lag.cal`, the observation-timing calendar used to STEP the fixing/shift windows.
inline px::RateObservation rfr_observation(const Date& vd, const Date& s, const Date& e,
                                           const std::string& dc, const RfrLag& lag = {},
                                           const std::string& accr_cal = "") {
  px::RateObservation o;
  if (!lag.active()) {  // plain telescoped bracket -- byte-identical to ois_coupon's obs
    o.sub_start = {curve_time(vd, s)};
    o.sub_end = {curve_time(vd, e)};
    o.tau_index = year_frac(dc, s, e, accr_cal);
    return o;
  }
  if (lag.style == RfrStyle::Shift) {  // shifts the WHOLE window -> still one telescoped bracket
    const Date ss = advance_obs_bd(lag.cal, s, -lag.days);
    const Date ee = advance_obs_bd(lag.cal, e, -lag.days);
    o.sub_start = {curve_time(vd, ss)};
    o.sub_end = {curve_time(vd, ee)};
    o.tau_index = year_frac(dc, ss, ee, accr_cal);  // obs-shift accrues on the shifted window
    return o;
  }
  // Lookback / Lockout: a genuine daily compounded product (does NOT telescope).
  o.compounded = true;
  o.tau_index = year_frac(dc, s, e, accr_cal);
  const auto per = daily_periods(s, e, lag.cal);  // keeps a non-business start's leading fixing (FixingPeriod)
  const std::size_t lock_from =
      (lag.style == RfrStyle::Lockout && per.size() > std::size_t(lag.days))
          ? per.size() - std::size_t(lag.days)
          : 0;
  for (std::size_t i = 0; i < per.size(); ++i) {
    Date obs0 = per[i].fix_start, obs1 = per[i].fix_end;  // the window whose forward rate this day earns
    if (lag.style == RfrStyle::Lookback) {
      obs0 = advance_obs_bd(lag.cal, obs0, -lag.days);
      obs1 = advance_obs_bd(lag.cal, obs1, -lag.days);
    } else if (i >= lock_from) {  // Lockout tail: freeze to the lockout-start day's window
      obs0 = per[lock_from].fix_start;
      obs1 = per[lock_from].fix_end;
    }
    const double ts = curve_time(vd, obs0), te = curve_time(vd, obs1);
    // the looked-back / locked-out window's rate, earning THIS day's accrual (obs_weight above)
    const double w = obs_weight(dc, accr_cal, per[i].acc_start, per[i].acc_end, obs0, obs1);
    o.sub_start.push_back(ts);
    o.sub_end.push_back(te);
    o.weight.push_back(w);
  }
  return o;
}

// A fixings-RESOLVABLE observation for a partially-started RFR contract (conventions.scheduled_observation):
// the engine resolves realized/forecast from its fixing table via the per-day schedule.
inline px::RateObservation scheduled_observation(const Date& vd, const Date& start, const Date& end,
                                                 const std::string& accrual, const std::string& dc,
                                                 const std::string& cal, const std::string& index) {
  const bool compounded = (accrual == "compounded");
  px::RateObservation o;
  o.fixing_index = index;
  o.compounded = compounded;
  o.tau_index = year_frac(dc, start, end, cal);  // `cal` is ignored unless dc==BUS/252
  // FixingDay carries the ACCRUAL THIS WINDOW EARNS (used directly once the day is realized) and the
  // FIXING window to forecast over while it is not, with `weight` converting one to the other. They differ
  // only where the window opens or closes mid-fixing -- a non-business start (FixingPeriod) -- and the
  // weight is that fraction for a COMPOUNDED product too: the factor is (1 + weight*growth), so hard-wiring
  // 1.0 there would have earned a partial leading day in full.
  for (const auto& p : daily_periods(start, end, cal)) {
    const double acc = year_frac(dc, p.acc_start, p.acc_end, cal);
    const double ts = curve_time(vd, p.fix_start), te = curve_time(vd, p.fix_end);
    const double w = obs_weight(dc, cal, p.acc_start, p.acc_end, p.fix_start, p.fix_end);
    o.fixing_schedule.push_back(px::FixingDay{int(p.fix_start.serial()), acc, ts, te, w});
  }
  return o;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_OBSERVATIONS_HPP
