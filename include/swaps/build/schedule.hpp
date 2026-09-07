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

// ---- ISDA schedule-generation options (Priority-1 roller richness) ----------------------------------
// EVERY field defaults to TODAY'S behaviour, so a `swap_periods_to(...)` call with no `ScheduleRule`
// (or a default-constructed one) reproduces the legacy short-back-stub forward roll BYTE-IDENTICALLY
// (the legacy fast path below is taken verbatim). The new richness is OPT-IN only.
//
// The odd (stub) period is the one whose length differs from the regular frequency step. Semantics:
//   * Back  (default): the SCHEDULE is anchored at spot and rolled FORWARD; the odd period is LAST
//                      (spot .. b1 .. bN .. maturity, with [bN, maturity] the stub).
//   * Front:           the schedule is anchored at maturity and rolled BACKWARD; the odd period is FIRST
//                      ([spot, b1] the stub, then regular steps to maturity).
//   * Short (default): the odd period is a partial step (shorter than the frequency).
//   * Long:            the odd period is MERGED into its adjacent regular period (so it is longer than a
//                      full step); the boundary between them is dropped.
enum class StubSide { Back, Front };
enum class StubLen { Short, Long };
struct ScheduleRule {
  StubSide side = StubSide::Back;
  StubLen length = StubLen::Short;
  bool eom = false;        // force every rolled boundary to month-end.
  bool eom_auto = false;   // ISDA EOM: infer `eom` when the roll anchor (spot for Back, maturity for
                           // Front) is itself a month-end date.
  int roll_dom = 0;        // day-of-month the regular boundaries land on; 0 = derive from the anchor
                           // (== today's behaviour: spot's own day of month rolled forward).
  bool is_default() const {
    return side == StubSide::Back && length == StubLen::Short && !eom && !eom_auto && roll_dom == 0;
  }
};

// Rolled accrual periods from spot to a GIVEN maturity date (calendars.swap_periods_to). With the default
// `ScheduleRule` this steps `freq_tok` FORWARD from spot, business-day-adjusting each interior boundary and
// ending the final period EXACTLY at maturity_date (a short final stub — the legacy behaviour, byte-for-byte
// unchanged; the engine weights each coupon by its own accrual factor). A non-default `rule` selects ISDA
// EOM / stub-location / roll-day-anchor generation (see ScheduleRule above).
inline std::vector<Period> swap_periods_between(const Date& spot, const std::string& cal_id,
                                                const Date& maturity_date, const std::string& freq_tok,
                                                const std::string& bdc = "ModifiedFollowing",
                                                const ScheduleRule& rule = {});

inline std::vector<Period> swap_periods_to(const Date& value_date, const std::string& cal_id,
                                           const Date& maturity_date, const std::string& freq_tok,
                                           const std::string& bdc = "ModifiedFollowing", int spot_lag = 2,
                                           const ScheduleRule& rule = {}) {
  return swap_periods_between(spot_date(value_date, cal_id, spot_lag), cal_id, maturity_date, freq_tok, bdc, rule);
}

// The same roll from an ARBITRARY start date (`spot` here is just the first accrual start): what a BOOKED
// trade needs -- its periods are anchored at its own effective date, which for a seasoned or forward-
// starting deal is not today's spot. swap_periods_to is exactly this with start = spot(value_date).
inline std::vector<Period> swap_periods_between(const Date& spot, const std::string& cal_id,
                                                const Date& maturity_date, const std::string& freq_tok,
                                                const std::string& bdc, const ScheduleRule& rule) {
  if (maturity_date <= spot) return {{spot, maturity_date}};
  const int step_m = tok_months(freq_tok);

  // Legacy fast path — taken verbatim so the default output can never drift by even one ulp of date.
  if (rule.is_default()) {
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

  // Extended ISDA path. The roll anchor and direction depend on the stub side. `dom`/`eom` describe the
  // day-of-month the regular (interior) boundaries land on before business-day adjustment.
  const Date anchor = (rule.side == StubSide::Back) ? spot : maturity_date;
  const bool eom = rule.eom || (rule.eom_auto && is_month_end(anchor));
  const int dom = rule.roll_dom > 0 ? rule.roll_dom : int(anchor.day());

  // Build the UNADJUSTED interior regular boundaries, strictly between spot and maturity, ascending.
  std::vector<Date> interior;
  if (rule.side == StubSide::Back) {
    for (int k = 1;; ++k) {
      const Date base = anchor.plus_months(k * step_m);
      const Date d = roll_in_month(base.year(), base.month(), dom, eom);
      if (d >= maturity_date) break;
      if (d > spot) interior.push_back(d);
    }
  } else {  // Front: step BACKWARD from maturity, collect descending then reverse.
    for (int k = 1;; ++k) {
      const Date base = anchor.plus_months(-k * step_m);
      const Date d = roll_in_month(base.year(), base.month(), dom, eom);
      if (d <= spot) break;
      if (d < maturity_date) interior.push_back(d);
    }
    for (std::size_t i = 0, j = interior.size(); i + 1 < j; ++i, --j)
      std::swap(interior[i], interior[j - 1]);
  }

  // LONG stub: drop the interior boundary ADJACENT to the stub end, merging the odd period into its
  // neighbour (Back -> drop the last interior boundary; Front -> drop the first). A no-op when there is
  // no stub to merge (no interior boundary, or maturity already coincides with the last regular date).
  if (rule.length == StubLen::Long && !interior.empty()) {
    if (rule.side == StubSide::Back)
      interior.pop_back();
    else
      interior.erase(interior.begin());
  }

  std::vector<Date> bounds{spot};
  for (const Date& d : interior) bounds.push_back(adjust(cal_id, d, bdc));
  bounds.push_back(maturity_date);
  std::vector<Period> out;
  out.reserve(bounds.size() - 1);
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.emplace_back(bounds[i], bounds[i + 1]);
  return out;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_SCHEDULE_HPP
