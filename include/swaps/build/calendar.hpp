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
//   - An id with no DB entry (or an empty id) THROWS (since 2026-09-08; the old code fell back to the "USD"
//     SIFMA rules). "NONE" is the explicit weekends-only calendar for synthetic problems.
#ifndef SWAPS_BUILD_CALENDAR_HPP
#define SWAPS_BUILD_CALENDAR_HPP

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

// The calendar's DB row (baked or runtime-overlay) with its rule/join storage. An unknown or EMPTY id
// THROWS (PRINCIPLES.md P2) — until 2026-09-08 it silently fell back to the "USD" SIFMA rules. Weekend-only
// synthetic problems name the DB calendar "NONE" explicitly.
inline conventions::CalendarView db_row(std::string_view cal_id) { return conventions::require_calendar(cal_id); }

// Observance policies (data: calendars[].observance / holidays[].observance):
//   none                        the date as it falls (no substitute)
//   sat_to_fri_sun_to_mon       US bond market (SIFMA): Saturday -> Friday, Sunday -> Monday
//   sun_to_mon                  Federal Reserve style: Sunday -> Monday only
//   weekend_to_next_weekday     UK / Australia / Canada: Saturday or Sunday -> the next weekday that is not
//                               already a holiday of this calendar (Christmas Sat + Boxing Sun -> Mon + Tue)
//   sun_to_next_weekday         Japan: a Sunday holiday -> the next weekday that is not already a holiday
//                               (Saturday holidays are NOT substituted)
// The chained policies need the holiday set built so far (`taken`) and the calendar's own weekend mask
// (`weekend_mask`, bit w = Mon..Sun: the "next weekday" of a Fri/Sat market is not Sunday's).
inline bool on_weekend(int weekend_mask, const Date& d) { return (weekend_mask >> d.weekday()) & 1; }
inline Date observe(std::string_view policy, const Date& d, int weekend_mask, const std::set<long>* taken = nullptr) {
  if (policy == "sat_to_fri_sun_to_mon") return obs_sat_fri_sun_mon(d);
  if (policy == "sun_to_mon") return obs_sun_mon(d);
  if (policy == "none" || policy.empty()) return d;
  if (policy == "weekend_to_next_weekday" || policy == "sun_to_next_weekday") {
    const bool shift = (d.weekday() == 6) || (d.weekday() == 5 && policy == "weekend_to_next_weekday");
    if (!shift) return d;
    Date x = d;
    while (on_weekend(weekend_mask, x) || (taken && taken->count(x.serial()))) x = x.plus_days(1);
    return x;
  }
  throw std::invalid_argument("unknown observance policy: " + std::string(policy));
}

// Japan's equinox days (the Japan calendar's own astronomical approximation, valid 2000-2099; the same
// formula QuantLib's Japan calendar uses).
inline Date vernal_equinox(int y) {
  return Date::ymd(y, 3, unsigned(int(20.69115 + 0.242194 * (y - 2000) - (y - 2000) / 4)));
}
inline Date autumnal_equinox(int y) {
  return Date::ymd(y, 9, unsigned(int(23.09 + 0.242194 * (y - 2000) - (y - 2000) / 4)));
}

// Rule kinds (data: holidays[].rule): fixed | nth_weekday | last_weekday | easter_offset |
//   weekday_before (the last `weekday` strictly before month/day — Canada's Victoria Day) |
//   vernal_equinox | autumnal_equinox (Japan) | working_day (a WEEKEND date that IS a business day — China's
//   adjusted working days; interpreted in is_business_day, never a holiday).
inline Date rule_date(const conventions::HolidayRule& r, int year) {
  const std::string_view k = r.kind;
  if (k == "fixed" || k == "working_day") return Date::ymd(year, unsigned(r.month), unsigned(r.day));
  if (k == "nth_weekday") return nth_weekday(year, unsigned(r.month), r.weekday, r.n);
  if (k == "last_weekday") return last_weekday(year, unsigned(r.month), r.weekday);
  if (k == "easter_offset") return easter(year).plus_days(r.days);
  if (k == "weekday_before") {
    Date d = Date::ymd(year, unsigned(r.month), unsigned(r.day)).plus_days(-1);
    while (d.weekday() != r.weekday) d = d.plus_days(-1);
    return d;
  }
  if (k == "vernal_equinox") return vernal_equinox(year);
  if (k == "autumnal_equinox") return autumnal_equinox(year);
  throw std::invalid_argument("unknown holiday rule kind: " + std::string(k));
}

}  // namespace calendar_detail

// Holiday set (excluding plain weekends) for a base (non-joint) calendar in `year`: `year`'s rules
// evaluated and observance-shifted (calendars._holidays). A joint calendar returns the union of its legs.
inline std::set<long> holidays_serial(const std::string& cal_id, int year) {
  const auto view = calendar_detail::db_row(cal_id);
  const auto& cal = view.row;
  std::set<long> hs;
  if (cal.join_count) {  // joint: the union of the legs' holiday sets
    for (std::size_t j = 0; j < cal.join_count; ++j) {
      const auto leg = holidays_serial(std::string(view.joins[j]), year);
      hs.insert(leg.begin(), leg.end());
    }
    return hs;
  }
  // Rules apply in DATE order so chained substitution sees the earlier holiday (Christmas before Boxing Day).
  std::vector<std::pair<Date, const conventions::HolidayRule*>> dated;
  for (std::size_t i = 0; i < cal.rule_count; ++i) {
    const auto& r = view.rules[i];
    if (r.kind == "working_day") continue;  // a working weekend day is not a holiday (see is_business_day)
    if (r.from_year && year < r.from_year) continue;
    if (r.to_year && year > r.to_year) continue;   // a per-year dated holiday: from_year==to_year==Y
    const Date d = calendar_detail::rule_date(r, year);
    // SIFMA: Good Friday is an early close, not a closure, when it is the first Friday of the month (the
    // jobs report). Data: holidays[].except_first_friday (QuantLib GovernmentBond: `d > 7`).
    if (r.except_first_friday && d.day() <= 7) continue;
    dated.emplace_back(d, &r);
  }
  std::sort(dated.begin(), dated.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  // Two passes: holidays that fall on a weekday are placed FIRST; weekend holidays are then substituted
  // into the next free weekday (Christmas on a Sunday -> Tuesday, because Boxing Day already holds the
  // Monday; Japan's Greenery Day on a Sunday -> the Tuesday after Children's Day).
  for (const auto& [d, r] : dated)
    if (!calendar_detail::on_weekend(cal.weekend_mask, d)) hs.insert(d.serial());
  for (const auto& [d, r] : dated) {
    if (!calendar_detail::on_weekend(cal.weekend_mask, d)) continue;
    const auto policy = r->observance.empty() ? cal.observance : r->observance;
    hs.insert(calendar_detail::observe(policy, d, cal.weekend_mask, &hs).serial());
  }
  // Japan's "citizen's holiday": a weekday sandwiched between two holidays is itself a holiday.
  if (cal.sandwich) {
    std::vector<long> add;
    for (long sv : hs) {
      const Date mid = Date{chr::sys_days{chr::days{sv + 1}}};
      if (!calendar_detail::on_weekend(cal.weekend_mask, mid) && !hs.count(sv + 1) && hs.count(sv + 2)) add.push_back(sv + 1);
    }
    hs.insert(add.begin(), add.end());
  }
  return hs;
}

// Weekend dates that ARE business days (China's adjusted working days): rules of kind "working_day".
inline bool is_working_weekend_day(const conventions::CalendarView& view, const Date& d) {
  for (std::size_t i = 0; i < view.row.rule_count; ++i) {
    const auto& r = view.rules[i];
    if (r.kind != "working_day") continue;
    if (r.from_year && d.year() < r.from_year) continue;
    if (r.to_year && d.year() > r.to_year) continue;
    if (int(r.month) == int(d.month()) && int(r.day) == int(d.day())) return true;
  }
  return false;
}

namespace calendar_detail {
// The closed weekdays of ONE (calendar, year) as a bitmap over the year's days, built once from
// holidays_serial(cal_id, year) — exactly the set is_business_day consulted per call before 2026-09-09, when
// every business-day query re-evaluated the whole year's rules (schedule building runs thousands of queries;
// the swaption-cube cold build doubled). Keyed by the registry generation, so a runtime `conventions` add or
// clear_overlay invalidates it. Weekends, working weekend days and joins are handled by the caller.
struct YearClosed {
  long jan1 = 0;
  std::array<unsigned char, 366> closed{};
};
inline const YearClosed& year_closed(const std::string& cal_id, int year) {
  static std::shared_mutex mu;
  static unsigned long gen = 0;
  static std::map<std::pair<std::string, int>, YearClosed> cache;
  const unsigned long g = conventions::Registry::instance().generation();
  const auto key = std::make_pair(cal_id, year);
  {
    std::shared_lock lk(mu);
    if (gen == g)
      if (auto it = cache.find(key); it != cache.end()) return it->second;
  }
  std::unique_lock lk(mu);
  if (gen != g) { cache.clear(); gen = g; }
  if (auto it = cache.find(key); it != cache.end()) return it->second;
  YearClosed y;
  y.jan1 = Date::ymd(year, 1, 1).serial();
  for (long sv : holidays_serial(cal_id, year)) {
    const long k = sv - y.jan1;
    if (k >= 0 && k < 366) y.closed[static_cast<std::size_t>(k)] = 1;  // the year's own days only (as before)
  }
  return cache.emplace(key, y).first->second;
}
}  // namespace calendar_detail

inline bool is_business_day(const std::string& cal_id, const Date& d) {
  const auto view = calendar_detail::db_row(cal_id);
  const auto& cal = view.row;
  if ((cal.weekend_mask >> d.weekday()) & 1) return is_working_weekend_day(view, d);
  if (cal.join_count) {  // joint: a business day only if EVERY leg is open
    for (std::size_t j = 0; j < cal.join_count; ++j)
      if (!is_business_day(std::string(view.joins[j]), d)) return false;
    return true;
  }
  const auto& y = calendar_detail::year_closed(cal_id, d.year());
  return !y.closed[static_cast<std::size_t>(d.serial() - y.jan1)];
}

inline Date roll(const std::string& cal_id, Date d, int step) {
  while (!is_business_day(cal_id, d)) d = d.plus_days(step);
  return d;
}

// Business-day-adjust under `bdc` (Following / ModifiedFollowing / Preceding / ModifiedPreceding).
inline Date adjust(const std::string& cal_id, const Date& d, const std::string& bdc) {  // bdc REQUIRED (no default)
  if (bdc == "Unadjusted") return d;  // ISDA: no business-day adjustment at all
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
