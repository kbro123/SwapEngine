// swaps::build — market calendars: rule-based holidays + business-day adjustment. Faithful transcription of
// server/calendars.py (TARGET / US-SIFMA bond market / US Federal Reserve / EURUSD join). QuantLib-free.
//
// Calendar ids are the conventions-DB keys. USD / USD-SOFR = SIFMA US-government-securities (bond market),
// which CLOSES Good Friday. USD-FED = Federal Reserve (Fedwire): Good Friday OPEN, and a Saturday holiday is
// NOT observed on the preceding Friday. EURUSD = union (closed if either leg is closed).
#ifndef SWAPS_BUILD_CALENDAR_HPP
#define SWAPS_BUILD_CALENDAR_HPP

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "swaps/build/date.hpp"

namespace swaps::build {

// US bond-market observance: Saturday holiday -> preceding Friday, Sunday -> following Monday.
inline Date obs_sat_fri_sun_mon(const Date& d) {
  if (d.weekday() == 5) return d.plus_days(-1);
  if (d.weekday() == 6) return d.plus_days(1);
  return d;
}
// Federal Reserve observance: Sunday -> Monday; Saturday holidays NOT taken on the Friday.
inline Date obs_sun_mon(const Date& d) { return d.weekday() == 6 ? d.plus_days(1) : d; }

// Holiday set (excluding plain weekends) for a base (non-joint) calendar in `year` (calendars._holidays).
inline std::set<long> holidays_serial(const std::string& cal_id, int year) {
  std::set<long> hs;
  const auto add = [&](const Date& x) { hs.insert(x.serial()); };
  if (cal_id == "EUR") {  // TARGET — fixed set, no weekend-observance shifting.
    const Date e = easter(year);
    add(Date::ymd(year, 1, 1));
    add(e.plus_days(-2));  // Good Friday
    add(e.plus_days(1));   // Easter Monday
    add(Date::ymd(year, 5, 1));
    add(Date::ymd(year, 12, 25));
    add(Date::ymd(year, 12, 26));
    return hs;
  }
  const bool fed = (cal_id == "USD-FED");
  const auto obs = [&](const Date& x) { return fed ? obs_sun_mon(x) : obs_sat_fri_sun_mon(x); };
  add(obs(Date::ymd(year, 1, 1)));    // New Year
  add(obs(Date::ymd(year, 7, 4)));    // Independence
  add(obs(Date::ymd(year, 11, 11)));  // Veterans
  add(obs(Date::ymd(year, 12, 25)));  // Christmas
  if (year >= 2021) add(obs(Date::ymd(year, 6, 19)));  // Juneteenth (US federal holiday since 2021)
  add(nth_weekday(year, 1, 0, 3));    // MLK — 3rd Monday of January
  add(nth_weekday(year, 2, 0, 3));    // Washington's Birthday — 3rd Monday of February
  add(last_weekday(year, 5, 0));      // Memorial Day — last Monday of May
  add(nth_weekday(year, 9, 0, 1));    // Labor Day — 1st Monday of September
  add(nth_weekday(year, 10, 0, 2));   // Columbus Day — 2nd Monday of October
  add(nth_weekday(year, 11, 3, 4));   // Thanksgiving — 4th Thursday of November
  if (!fed) add(easter(year).plus_days(-2));  // bond market (SIFMA/SOFR) closes Good Friday; the Fed does not
  return hs;
}

inline bool is_business_day(const std::string& cal_id, const Date& d) {
  if (d.weekday() >= 5) return false;
  if (cal_id == "EURUSD")  // union: a business day only if BOTH legs are open
    return is_business_day("USD", d) && is_business_day("EUR", d);
  return holidays_serial(cal_id, d.year()).count(d.serial()) == 0;
}

inline Date roll(const std::string& cal_id, Date d, int step) {
  while (!is_business_day(cal_id, d)) d = d.plus_days(step);
  return d;
}

// Business-day-adjust under `bdc` (Following / ModifiedFollowing / Preceding / ModifiedPreceding).
inline Date adjust(const std::string& cal_id, const Date& d, const std::string& bdc = "ModifiedFollowing") {
  if (is_business_day(cal_id, d)) return d;
  if (bdc == "Following") return roll(cal_id, d, 1);
  if (bdc == "Preceding") return roll(cal_id, d, -1);
  if (bdc == "ModifiedFollowing") {
    const Date r = roll(cal_id, d, 1);
    return r.month() == d.month() ? r : roll(cal_id, d, -1);
  }
  if (bdc == "ModifiedPreceding") {
    const Date r = roll(cal_id, d, -1);
    return r.month() == d.month() ? r : roll(cal_id, d, 1);
  }
  throw std::invalid_argument("unknown business-day convention: " + bdc);
}

// Advance n business days (n may be negative; n==0 -> adjust Following).
inline Date advance_bd(const std::string& cal_id, Date d, int n) {
  if (n == 0) return roll(cal_id, d, 1);
  const int step = n > 0 ? 1 : -1;
  int left = std::abs(n);
  while (left) {
    d = d.plus_days(step);
    if (is_business_day(cal_id, d)) --left;
  }
  return d;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_CALENDAR_HPP
