// swaps::build — the observation sub-period windows an instrument's Rate quote looks at (the subtlest part
// of the port). Faithful transcription of server/conventions.py observation()/scheduled_observation()/
// business_days()/daily_periods(). Emits the engine's pricing::RateObservation directly.
//
// rate = ( Σ_k w_k·[DF(s_k)/DF(e_k) − 1] + realized ) / tau_index. AVERAGED (1M): one bracket per fixing day,
// weight = index-accrual/curve-accrual. COMPOUNDED (3M): one telescoped bracket; a partial-fix prefix rides
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
// (fixing_date, accrual_end) per fixing day covering [start, end) (conventions.daily_periods).
inline std::vector<std::pair<Date, Date>> daily_periods(const Date& start, const Date& end,
                                                        const std::string& cal) {
  const auto days = business_days(start, end, cal);
  std::vector<std::pair<Date, Date>> out;
  out.reserve(days.size());
  for (std::size_t i = 0; i < days.size(); ++i)
    out.emplace_back(days[i], i + 1 < days.size() ? days[i + 1] : end);
  return out;
}

// The obs for a futures-style Rate quote (conventions.observation). `realized_pct` in PERCENT.
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
    for (const auto& [s, e] : daily_periods(fwd_from, end, cal)) {
      const double ts = curve_time(vd, s), te = curve_time(vd, e), crv = te - ts;
      const double w = crv > 0 ? year_frac(dc, s, e, cal) / crv : 1.0;
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
  // Each day's term is (index accrual τ_d / curve-time step dt_d)·(e^{δ_d} − 1). For an ACT-based index day
  // count the ratio is the SAME every day (365/360 for ACT/360 on the ACT/365F curve clock), so it factors out
  // of the whole sum and rides as the bracket's single weight; a day count for which it varies (30/360) has no
  // moment form and is refused rather than approximated twice.
  double s2 = 0.0, s3 = 0.0, w = 0.0, wmin = 1e300, wmax = -1e300;
  for (const auto& [s, e] : daily_periods(start, end, cal)) {
    const double dt = curve_time(vd, e) - curve_time(vd, s);
    const double r = year_frac(dc, s, e, cal) / dt;
    s2 += dt * dt; s3 += dt * dt * dt;
    w = r; wmin = std::min(wmin, r); wmax = std::max(wmax, r);
  }
  if (wmax - wmin > 1e-10 * std::max(1.0, wmax))
    throw std::invalid_argument("moment_observation: the index day count '" + dc + "' is not a constant multiple of curve time; use the daily path");
  o.sub_start = {a}; o.sub_end = {b};
  if (std::abs(w - 1.0) > 1e-15) o.weight = {w};
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
  const auto days = business_days(s, e, lag.cal);
  const std::size_t lock_from =
      (lag.style == RfrStyle::Lockout && days.size() > std::size_t(lag.days))
          ? days.size() - std::size_t(lag.days)
          : 0;
  for (std::size_t i = 0; i < days.size(); ++i) {
    const Date d0 = days[i];
    const Date d1 = (i + 1 < days.size()) ? days[i + 1] : e;  // ACTUAL accrual span of this fixing day
    const double acc = year_frac(dc, d0, d1, accr_cal);
    Date obs0 = d0, obs1 = d1;  // the observation window whose forward rate this day earns
    if (lag.style == RfrStyle::Lookback) {
      obs0 = advance_obs_bd(lag.cal, d0, -lag.days);
      obs1 = advance_obs_bd(lag.cal, d1, -lag.days);
    } else if (i >= lock_from) {  // Lockout tail: freeze to the lockout-start day's window
      obs0 = days[lock_from];
      obs1 = (lock_from + 1 < days.size()) ? days[lock_from + 1] : e;
    }
    const double ts = curve_time(vd, obs0), te = curve_time(vd, obs1), crv = te - ts;
    const double w = crv > 0 ? acc / crv : 1.0;  // rate over the obs window reweighted to the actual accrual
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
  const auto days = business_days(start, end, cal);
  px::RateObservation o;
  o.fixing_index = index;
  o.compounded = compounded;
  o.tau_index = year_frac(dc, start, end, cal);  // `cal` is ignored unless dc==BUS/252
  for (std::size_t i = 0; i < days.size(); ++i) {
    const Date nxt = i + 1 < days.size() ? days[i + 1] : end;
    const double acc = year_frac(dc, days[i], nxt, cal);
    const double ts = curve_time(vd, days[i]), te = curve_time(vd, nxt), crv = te - ts;
    const double w = compounded ? 1.0 : (crv > 0 ? acc / crv : 1.0);
    o.fixing_schedule.push_back(px::FixingDay{int(days[i].serial()), acc, ts, te, w});
  }
  return o;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_OBSERVATIONS_HPP
