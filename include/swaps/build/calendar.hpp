// swaps::build — market calendars: rule-based holidays + business-day adjustment. QuantLib-free.
//
// The holiday RULES are DATA: conventions/conventions.json `calendars[].holidays` (codegen'd into
// swaps/conventions_data.hpp as kCalendars/kHolidayRules by tools/gen_conventions_hpp.py). This header only
// INTERPRETS them — fixed dates, nth/last-weekday, Easter offsets (the computus itself stays in date.hpp:
// it is an algorithm, not data), weekend-observance shifting, and joint calendars (closed if any leg is
// closed). Adding a market calendar is a JSON entry, not a code branch. Behavior is bit-for-bit the old
// hard-coded transcription of server/calendars.py (build_calendar_test.cpp + calendar_data_test.cpp gate it):
//   - USD / USD-SOFR = SIFMA US-government-securities (bond market): CLOSES Good Friday; a Saturday holiday
//     is observed the preceding Friday, a Sunday holiday the following Monday.
//   - USD-FED = Federal Reserve (Fedwire): Good Friday OPEN; Sunday -> Monday, Saturday NOT observed.
//   - EUR = TARGET: fixed set, no observance shifting. EURUSD = join(USD, EUR).
//   - Legacy quirk kept: a year's set is generated FROM that year's rules, so an observance can spill into
//     the previous year's dates (Jan-1 on a Saturday -> Dec-31) yet is_business_day buckets by the queried
//     date's own year, leaving that spilled Friday a business day (e.g. 2021-12-31 on USD).
//   - Legacy fallback kept: an id with no DB entry gets the "USD" (SIFMA bond-market) rules, exactly as the
//     old code's default branch did (Calendar{""} relied on it).
#ifndef SWAPS_BUILD_CALENDAR_HPP
#define SWAPS_BUILD_CALENDAR_HPP

#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "swaps/build/date.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::build {

// US bond-market observance: Saturday holiday -> preceding Friday, Sunday -> following Monday.
inline Date obs_sat_fri_sun_mon(const Date& d) {
  if (d.weekday() == 5) return d.plus_days(-1);
  if (d.weekday() == 6) return d.plus_days(1);
  return d;
}
// Federal Reserve observance: Sunday -> Monday; Saturday holidays NOT taken on the Friday.
inline Date obs_sun_mon(const Date& d) { return d.weekday() == 6 ? d.plus_days(1) : d; }

namespace calendar_detail {

// The calendar's DB row; an unknown (or empty) id falls back to the "USD" bond-market rules — the exact
// behavior of the pre-data code, whose default branch was the US bond calendar.
inline const conventions::CalendarConv& db_row(std::string_view cal_id) {
  for (const auto& c : conventions::kCalendars)
    if (c.id == cal_id) return c;
  for (const auto& c : conventions::kCalendars)
    if (c.id == "USD") return c;
  throw std::logic_error("conventions DB has no USD calendar to fall back to");
}

inline Date observe(std::string_view policy, const Date& d) {
  if (policy == "sat_to_fri_sun_to_mon") return obs_sat_fri_sun_mon(d);
  if (policy == "sun_to_mon") return obs_sun_mon(d);
  if (policy == "none" || policy.empty()) return d;
  throw std::invalid_argument("unknown observance policy: " + std::string(policy));
}

inline Date rule_date(const conventions::HolidayRule& r, int year) {
  const std::string_view k = r.kind;
  if (k == "fixed") return Date::ymd(year, unsigned(r.month), unsigned(r.day));
  if (k == "nth_weekday") return nth_weekday(year, unsigned(r.month), r.weekday, r.n);
  if (k == "last_weekday") return last_weekday(year, unsigned(r.month), r.weekday);
  if (k == "easter_offset") return easter(year).plus_days(r.days);
  throw std::invalid_argument("unknown holiday rule kind: " + std::string(k));
}

}  // namespace calendar_detail

// Holiday set (excluding plain weekends) for a base (non-joint) calendar in `year`: `year`'s rules
// evaluated and observance-shifted (calendars._holidays). A joint calendar returns the union of its legs.
inline std::set<long> holidays_serial(const std::string& cal_id, int year) {
  const auto& cal = calendar_detail::db_row(cal_id);
  std::set<long> hs;
  if (cal.join_count) {  // joint: the union of the legs' holiday sets
    for (std::size_t j = 0; j < cal.join_count; ++j) {
      const auto leg = holidays_serial(std::string(conventions::kCalendarJoins[cal.join_begin + j]), year);
      hs.insert(leg.begin(), leg.end());
    }
    return hs;
  }
  for (std::size_t i = 0; i < cal.rule_count; ++i) {
    const auto& r = conventions::kHolidayRules[cal.rule_begin + i];
    if (r.from_year && year < r.from_year) continue;
    const auto policy = r.observance.empty() ? cal.observance : r.observance;
    hs.insert(calendar_detail::observe(policy, calendar_detail::rule_date(r, year)).serial());
  }
  return hs;
}

inline bool is_business_day(const std::string& cal_id, const Date& d) {
  const auto& cal = calendar_detail::db_row(cal_id);
  if ((cal.weekend_mask >> d.weekday()) & 1) return false;
  if (cal.join_count) {  // joint: a business day only if EVERY leg is open
    for (std::size_t j = 0; j < cal.join_count; ++j)
      if (!is_business_day(std::string(conventions::kCalendarJoins[cal.join_begin + j]), d)) return false;
    return true;
  }
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
