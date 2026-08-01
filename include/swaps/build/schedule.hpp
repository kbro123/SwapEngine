// swaps::build — token resolution + curve time + rolled schedule generation. Transcribes server/dates.py
// (token -> date, weekend-only "following"; IMM) and server/calendars.py (curve_time, spot_date,
// swap_periods_to, the full-calendar schedule roller). Two roll behaviours are reproduced FAITHFULLY: token
// resolution uses the POC weekend-only roll (dates._following); schedule interior boundaries use the full
// business-day adjust (calendars.adjust) — matching the Python compiler exactly.
#ifndef SWAPS_BUILD_SCHEDULE_HPP
#define SWAPS_BUILD_SCHEDULE_HPP

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"

namespace swaps::build {

// Weekend-only "following" roll (dates._following). NOT holiday-aware — used only for token resolution.
inline Date following_weekend(Date d) {
  while (d.weekday() >= 5) d = d.plus_days(1);
  return d;
}

inline int imm_month(char c) {
  static const std::string L = "FGHJKMNQUVXZ";  // Jan..Dec
  const auto p = L.find(char(std::toupper(c)));
  if (p == std::string::npos) throw std::invalid_argument("bad IMM month letter");
  return int(p) + 1;
}

// Resolve a token to a calendar date (dates.resolve). `roll` applies the weekend "following" adjustment.
inline Date resolve(const std::string& token, const Date& value_date, bool roll = true) {
  std::string t = token;
  // strip surrounding whitespace
  while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
  while (!t.empty() && std::isspace((unsigned char)t.back())) t.pop_back();
  if (t.empty()) throw std::invalid_argument("empty date");
  if (t == "?\?\?" || t == "?") throw std::invalid_argument("meeting date not yet known");

  std::string low = t;
  for (char& ch : low) ch = char(std::tolower(ch));
  if (low == "sp" || low == "spot" || low == "0d" || low == "b" || low == "t+0")
    return roll ? following_weekend(value_date) : value_date;
  if (low == "on") return following_weekend(value_date.plus_days(1));
  if (low == "tn") return following_weekend(following_weekend(value_date.plus_days(1)).plus_days(1));

  // ISO YYYY-MM-DD
  if (t.size() == 10 && t[4] == '-' && t[7] == '-' && std::isdigit((unsigned char)t[0])) {
    const Date d = Date::from_iso(t);
    return roll ? following_weekend(d) : d;
  }
  // IMM code: letter + 2 digits (e.g. U27)
  if (t.size() == 3 && std::isalpha((unsigned char)t[0]) && std::isdigit((unsigned char)t[1]) &&
      std::isdigit((unsigned char)t[2])) {
    const int month = imm_month(t[0]);
    const int year = 2000 + std::stoi(t.substr(1, 2));
    return third_wednesday(year, unsigned(month));  // already a Wednesday
  }
  // Tenor: number + unit d/w/m/y
  {
    const char unit = char(std::tolower(t.back()));
    if (unit == 'd' || unit == 'w' || unit == 'm' || unit == 'y') {
      const double n = std::stod(t.substr(0, t.size() - 1));
      Date d = value_date;
      if (unit == 'd') d = value_date.plus_days(int(std::lround(n)));
      else if (unit == 'w') d = value_date.plus_days(int(std::lround(n * 7)));
      else if (unit == 'm') d = value_date.plus_months(int(std::lround(n)));
      else d = value_date.plus_months(int(std::lround(n * 12)));
      return roll ? following_weekend(d) : d;
    }
  }
  throw std::invalid_argument("unrecognized date/tenor: " + token);
}

// Curve time (ACT/365F from the value date) — the engine's ModularCurve axis (calendars.curve_time).
inline double curve_time(const Date& value_date, const Date& d) { return (d - value_date) / 365.0; }

// Spot/settlement date: `spot_lag` business days after the value date on `cal_id` (calendars.spot_date).
inline Date spot_date(const Date& value_date, const std::string& cal_id, int spot_lag) {
  return advance_bd(cal_id, value_date, spot_lag);
}

using Period = std::pair<Date, Date>;

// Rolled accrual periods from spot to a GIVEN maturity date (calendars.swap_periods_to). Steps `freq_tok`
// from spot, business-day-adjusting each interior boundary, ending the final period EXACTLY at maturity_date
// (a short final stub is allowed and correct — the engine weights each coupon by its own accrual factor).
inline std::vector<Period> swap_periods_to(const Date& value_date, const std::string& cal_id,
                                           const Date& maturity_date, const std::string& freq_tok,
                                           const std::string& bdc = "ModifiedFollowing", int spot_lag = 2) {
  const Date spot = spot_date(value_date, cal_id, spot_lag);
  if (maturity_date <= spot) return {{spot, maturity_date}};
  const int step_m = tok_months(freq_tok);
  std::vector<Date> bounds{spot};
  for (int m = step_m;; m += step_m) {
    const Date d = adjust(cal_id, add_period(spot, std::to_string(m) + "M"), bdc);
    if (d >= maturity_date) break;
    bounds.push_back(d);
  }
  bounds.push_back(maturity_date);
  std::vector<Period> out;
  out.reserve(bounds.size() - 1);
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.emplace_back(bounds[i], bounds[i + 1]);
  return out;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_SCHEDULE_HPP
