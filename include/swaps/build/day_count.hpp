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

// Accrual factor between two dates under `dc` (calendars.year_frac).
inline double year_frac(const std::string& dc, const Date& d1, const Date& d2) {
  if (dc == "ACT/360") return (d2 - d1) / 360.0;
  if (dc == "ACT/365F") return (d2 - d1) / 365.0;
  if (dc == "30E/360") return thirty_e(d1, d2);
  if (dc == "30U/360") return thirty_us(d1, d2);
  throw std::invalid_argument("unknown day count: " + dc);
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_DAY_COUNT_HPP
