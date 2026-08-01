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
#include <string>
#include <utility>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::build {

namespace px = swaps::pricing;

// Proleptic Gregorian ordinal (Python date.toordinal(): 0001-01-01 == 1). 1970-01-01 == 719163.
inline int ordinal(const Date& d) { return int(d.serial()) + 719163; }

// Fixing dates in [start, end) on the index calendar ("" -> weekends only) (conventions.business_days).
inline std::vector<Date> business_days(const Date& start, const Date& end, const std::string& cal) {
  std::vector<Date> days;
  for (Date d = start; d < end; d = d.plus_days(1)) {
    const bool bd = cal.empty() ? (d.weekday() < 5) : is_business_day(cal, d);
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
                                       const std::string& dc = "ACT/360", const std::string& cal = "") {
  const double tau = year_frac(dc, start, end);
  const Date fwd_from = (start > vd) ? start : vd;
  const double tau_past = (start < vd) ? year_frac(dc, start, vd) : 0.0;
  const double r = realized_pct / 100.0;
  px::RateObservation o;
  if (accrual == "averaged") {
    bool all_one = true;
    for (const auto& [s, e] : daily_periods(fwd_from, end, cal)) {
      const double ts = curve_time(vd, s), te = curve_time(vd, e), crv = te - ts;
      const double w = crv > 0 ? year_frac(dc, s, e) / crv : 1.0;
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
  o.tau_index = year_frac(dc, start, end);
  for (std::size_t i = 0; i < days.size(); ++i) {
    const Date nxt = i + 1 < days.size() ? days[i + 1] : end;
    const double acc = year_frac(dc, days[i], nxt);
    const double ts = curve_time(vd, days[i]), te = curve_time(vd, nxt), crv = te - ts;
    const double w = compounded ? 1.0 : (crv > 0 ? acc / crv : 1.0);
    o.fixing_schedule.push_back(px::FixingDay{ordinal(days[i]), acc, ts, te, w});
  }
  return o;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_OBSERVATIONS_HPP
