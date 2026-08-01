// swaps::build — QuantLib-free CONSTRUCTION object model. This layer reproduces QuantLib's construction
// abstractions (Date/Calendar/DayCounter/Schedule/Index/instrument helpers) but its OUTPUT is the engine's
// own flat structs (RateObservation/FloatCoupon/Instrument -> BundleProblem, MultiCurveBook). It NEVER
// touches the calculation engine (curve/pricing/calibration/portfolio/ad) — it only builds what feeds it.
//
// date.hpp: a thin calendar Date over C++20 <chrono> (year_month_day / sys_days) + the pure date arithmetic
// the calendars/schedules need (weekday, nth/last weekday, month-end, Easter, IMM 3rd-Wednesday). Faithful
// transcription of server/calendars.py + server/dates.py (both already QuantLib-free).
#ifndef SWAPS_BUILD_DATE_HPP
#define SWAPS_BUILD_DATE_HPP

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace swaps::build {

namespace chr = std::chrono;

// A calendar date backed by chrono::sys_days. `serial()` is days since the Unix epoch — exactly the
// "integer serial the caller defines" the engine's FixingTable / eval date want (pricing/fixings.hpp).
struct Date {
  chr::sys_days d{};

  static Date ymd(int y, unsigned m, unsigned day) {
    return {chr::sys_days{chr::year{y} / chr::month{m} / chr::day{day}}};
  }
  // Clamp `day` to the month's length (Feb-30 -> Feb-28/29), matching calendars._add_period's clamp loop.
  static Date ymd_clamped(int y, unsigned m, unsigned day) {
    const unsigned last = unsigned(chr::year_month_day_last{chr::year{y} / chr::month{m} / chr::last}.day());
    return ymd(y, m, day < last ? day : last);
  }
  static Date from_iso(const std::string& s) {  // "YYYY-MM-DD"
    if (s.size() != 10 || s[4] != '-' || s[7] != '-')
      throw std::invalid_argument("date must be YYYY-MM-DD, got " + s);
    return ymd(std::stoi(s.substr(0, 4)), unsigned(std::stoi(s.substr(5, 2))),
               unsigned(std::stoi(s.substr(8, 2))));
  }

  chr::year_month_day ymd_() const { return chr::year_month_day{d}; }
  int year() const { return int(ymd_().year()); }
  unsigned month() const { return unsigned(ymd_().month()); }
  unsigned day() const { return unsigned(ymd_().day()); }
  long serial() const { return d.time_since_epoch().count(); }
  // Python date.weekday(): Mon=0 .. Sun=6.
  int weekday() const { return int(chr::weekday{d}.iso_encoding()) - 1; }

  Date plus_days(int n) const { return {d + chr::days{n}}; }
  Date plus_weeks(int n) const { return {d + chr::weeks{n}}; }
  // Add n whole months with day-clamp (calendars._add_period 'M' / dates._add_months).
  Date plus_months(int n) const {
    const long total = long(year()) * 12 + (long(month()) - 1) + n;
    const int y = int(total >= 0 ? total / 12 : -((-total + 11) / 12));
    const unsigned m = unsigned(total - long(y) * 12) + 1;
    return ymd_clamped(y, m, day());
  }

  bool operator<(const Date& o) const { return d < o.d; }
  bool operator<=(const Date& o) const { return d <= o.d; }
  bool operator>(const Date& o) const { return d > o.d; }
  bool operator>=(const Date& o) const { return d >= o.d; }
  bool operator==(const Date& o) const { return d == o.d; }
  bool operator!=(const Date& o) const { return d != o.d; }
};

// Whole-day difference a - b (calendars uses (d2 - d1).days).
inline int operator-(const Date& a, const Date& b) { return int((a.d - b.d).count()); }

inline std::string iso(const Date& x) {
  char buf[11];
  std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", x.year(), x.month(), x.day());
  return std::string(buf);
}

// Add an ISO period token ('3M','6M','1Y','2W','5D') unadjusted (calendars._add_period).
inline Date add_period(const Date& d, const std::string& tok) {
  if (tok.empty()) throw std::invalid_argument("bad period token");
  const char unit = std::toupper(tok.back());
  const int n = std::stoi(tok.substr(0, tok.size() - 1));
  switch (unit) {
    case 'D': return d.plus_days(n);
    case 'W': return d.plus_weeks(n);
    case 'M': return d.plus_months(n);
    case 'Y': return d.plus_months(12 * n);
    default: throw std::invalid_argument("bad period token: " + tok);
  }
}

// Whole months in a frequency token (calendars._tok_months): '3M'->3, '1Y'->12.
inline int tok_months(const std::string& tok) {
  const char unit = std::toupper(tok.back());
  const int n = std::stoi(tok.substr(0, tok.size() - 1));
  if (unit == 'M') return n;
  if (unit == 'Y') return 12 * n;
  throw std::invalid_argument("frequency must be whole months/years, got " + tok);
}

// The n-th `weekday` (Mon=0) of a month, n>=1 (calendars._nth_weekday).
inline Date nth_weekday(int y, unsigned month, int weekday, int n) {
  const Date first = Date::ymd(y, month, 1);
  const int shift = ((weekday - first.weekday()) % 7 + 7) % 7 + (n - 1) * 7;
  return first.plus_days(shift);
}
// The last `weekday` (Mon=0) of a month (calendars._last_weekday).
inline Date last_weekday(int y, unsigned month, int weekday) {
  const Date nxt = (month == 12) ? Date::ymd(y + 1, 1, 1) : Date::ymd(y, month + 1, 1);
  const Date d = nxt.plus_days(-1);
  return d.plus_days(-(((d.weekday() - weekday) % 7 + 7) % 7));
}
// Last calendar day of a month.
inline Date end_of_month(int y, unsigned m) {
  return {chr::sys_days{chr::year_month_day_last{chr::year{y} / chr::month{m} / chr::last}}};
}
// IMM date: 3rd Wednesday of the month (dates._third_wednesday). weekday(): Wed=2.
inline Date third_wednesday(int y, unsigned month) {
  const Date first = Date::ymd(y, month, 1);
  const int first_wed = 1 + ((2 - first.weekday()) % 7 + 7) % 7;
  return Date::ymd(y, month, unsigned(first_wed + 14));
}
// Easter Sunday, Anonymous Gregorian computus (calendars._easter).
inline Date easter(int year) {
  const int a = year % 19, b = year / 100, c = year % 100, d = b / 4, e = b % 4;
  const int f = (b + 8) / 25, g = (b - f + 1) / 3;
  const int h = (19 * a + b - d - g + 15) % 30, i = c / 4, k = c % 4;
  const int l = (32 + 2 * e + 2 * i - h - k) % 7, m = (a + 11 * h + 22 * l) / 451;
  const int month = (h + l - 7 * m + 114) / 31, day = ((h + l - 7 * m + 114) % 31) + 1;
  return Date::ymd(year, unsigned(month), unsigned(day));
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_DATE_HPP
