// swaps::build — day-count accrual factors (calendars.py year_frac / _thirty_e / _thirty_us). Distinct from
// curve time (ACT/365F from the value date, schedule.hpp curve_time): these weight each coupon in the residual.
#ifndef SWAPS_BUILD_DAY_COUNT_HPP
#define SWAPS_BUILD_DAY_COUNT_HPP

#include <stdexcept>
#include <string>

#include "swaps/build/date.hpp"

namespace swaps::build {

// 30E/360 (Eurobond / 30/360 ICMA) — EUR IRS fixed leg. Both day-31 start and end truncated to 30.
inline double thirty_e(const Date& d1, const Date& d2) {
  const int dd1 = int(d1.day()) < 30 ? int(d1.day()) : 30;
  const int dd2 = int(d2.day()) < 30 ? int(d2.day()) : 30;
  return ((d2.year() - d1.year()) * 360 + (int(d2.month()) - int(d1.month())) * 30 + (dd2 - dd1)) / 360.0;
}
// 30/360 US (bond basis) — conditional end-truncation.
inline double thirty_us(const Date& d1, const Date& d2) {
  int dd1 = int(d1.day()), dd2 = int(d2.day());
  if (dd1 == 31) dd1 = 30;
  if (dd2 == 31 && dd1 == 30) dd2 = 30;
  return ((d2.year() - d1.year()) * 360 + (int(d2.month()) - int(d1.month())) * 30 + (dd2 - dd1)) / 360.0;
}

// ACT/ACT (ISDA) — split the span at each year boundary and weight each part by that year's length
// (365 or 366). Matches QuantLib::ActualActual(ActualActual::ISDA). Reference-period-free, so it fits
// the (d1,d2) year_frac signature; the ICMA variant (which NEEDS the coupon reference period) is a
// separate free function below.
inline double act_act_isda(const Date& d1, const Date& d2) {
  if (d2 <= d1) return -act_act_isda(d2, d1);
  const int y1 = d1.year(), y2 = d2.year();
  auto is_leap = [](int y) { return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0); };
  if (y1 == y2) return (d2 - d1) / (is_leap(y1) ? 366.0 : 365.0);
  double sum = 0.0;
  const Date start_y1_next = Date::ymd(y1 + 1, 1, 1);
  sum += (start_y1_next - d1) / (is_leap(y1) ? 366.0 : 365.0);
  const Date start_y2 = Date::ymd(y2, 1, 1);
  sum += (d2 - start_y2) / (is_leap(y2) ? 366.0 : 365.0);
  sum += double(y2 - y1 - 1);  // whole intervening years each contribute exactly 1.0
  return sum;
}

// Accrual factor between two dates under `dc` (calendars.year_frac). Reference-period-free day counts
// only; ACT/ACT(ICMA) needs the coupon period and is `act_act_icma` below.
inline double year_frac(const std::string& dc, const Date& d1, const Date& d2) {
  if (dc == "ACT/360") return (d2 - d1) / 360.0;
  if (dc == "ACT/365F") return (d2 - d1) / 365.0;
  if (dc == "ACT/ACT" || dc == "ACT/ACT.ISDA") return act_act_isda(d1, d2);
  if (dc == "30E/360") return thirty_e(d1, d2);
  if (dc == "30U/360") return thirty_us(d1, d2);
  throw std::invalid_argument("unknown day count: " + dc);
}

// ACT/ACT (ICMA / ISMA) year fraction over [d1,d2] measured against a coupon reference period
// [ref_start, ref_end] paid `freq` times a year (bond basis). For a sub-span of a regular period this
// is (d2−d1) / (freq·(ref_end−ref_start)); a full period is exactly 1/freq. This is the treasury/bond
// accrual and street-yield day count — QuantLib::ActualActual(ActualActual::ISMA) with that reference.
inline double act_act_icma(const Date& d1, const Date& d2, const Date& ref_start, const Date& ref_end,
                           int freq) {
  const double period_days = double(ref_end - ref_start);
  return (d2 - d1) / (double(freq) * period_days);
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_DAY_COUNT_HPP
