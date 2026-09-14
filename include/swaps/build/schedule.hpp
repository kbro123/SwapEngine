// swaps::build — token resolution + curve time + rolled schedule generation (server/dates.py + calendars.py
// parity). Since 2026-09-09 token resolution is CALENDAR-AWARE and anchored at SPOT (PRINCIPLES.md P2): a
// tenor "5Y" is spot(value_date, calendar, spot_lag) + 5Y, business-day-adjusted under `bdc` on the product's
// calendar; ON/TN are 1/2 business days; ISO dates are adjusted; IMM codes are contract dates (never rolled).
// The old weekend-only "following" roll from the value date is expressible as data (calendar "NONE", bdc
// "Following", spot_lag 0) and is what the synthetic tests ask for explicitly.
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

// Spot/settlement date: `spot_lag` business days after the value date on `cal_id` (calendars.spot_date).
inline Date spot_date(const Date& value_date, const std::string& cal_id, int spot_lag) {
  return advance_bd(cal_id, value_date, spot_lag);
}

inline int imm_month(char c) {
  static const std::string L = "FGHJKMNQUVXZ";  // Jan..Dec
  const auto p = L.find(char(std::toupper(c)));
  if (p == std::string::npos) throw std::invalid_argument("bad IMM month letter");
  return int(p) + 1;
}

// Resolve a token to a calendar date on a product's conventions (dates.resolve):
//   sp / spot / t+0 / 0d  -> spot = value_date + spot_lag business days on `cal_id`
//   on / tn               -> value_date + 1 / + 2 business days
//   YYYY-MM-DD            -> that date, adjusted under `bdc` on `cal_id` ("Unadjusted" = as entered)
//   IMM code (U27)        -> the contract's 3rd Wednesday (a contract date; never rolled)
//   tenor (3M, 2Y, 1Y6M, 28D, 2W) -> spot + tenor, adjusted under `bdc`
inline Date resolve(const std::string& token, const Date& value_date, const std::string& cal_id,
                    const std::string& bdc, int spot_lag) {
  std::string t = token;
  while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
  while (!t.empty() && std::isspace((unsigned char)t.back())) t.pop_back();
  if (t.empty()) throw std::invalid_argument("empty date");
  if (t == "?\?\?" || t == "?") throw std::invalid_argument("meeting date not yet known");
  if (spot_lag < 0) throw std::invalid_argument("resolve: spot_lag must be >= 0");

  std::string low = t;
  for (char& ch : low) ch = char(std::tolower(ch));
  const Date spot = spot_date(value_date, cal_id, spot_lag);
  if (low == "sp" || low == "spot" || low == "0d" || low == "b" || low == "t+0") return spot;
  if (low == "on") return advance_bd(cal_id, value_date, 1);
  if (low == "tn") return advance_bd(cal_id, value_date, 2);

  // ISO YYYY-MM-DD
  if (t.size() == 10 && t[4] == '-' && t[7] == '-' && std::isdigit((unsigned char)t[0]))
    return adjust(cal_id, Date::from_iso(t), bdc);
  // IMM code: letter + 2 digits (e.g. U27)
  if (t.size() == 3 && std::isalpha((unsigned char)t[0]) && std::isdigit((unsigned char)t[1]) &&
      std::isdigit((unsigned char)t[2])) {
    const int month = imm_month(t[0]);
    const int year = 2000 + std::stoi(t.substr(1, 2));
    return third_wednesday(year, unsigned(month));  // already a Wednesday
  }
  // Tenor (strict; compound allowed): anchored at SPOT, adjusted under bdc.
  const Step st = parse_step(t);
  return adjust(cal_id, plus_step(spot, st), bdc);
}

// Curve time (ACT/365F from the value date) — the engine's ModularCurve axis (calendars.curve_time).
inline double curve_time(const Date& value_date, const Date& d) { return (d - value_date) / 365.0; }

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
                                                const std::string& bdc, const ScheduleRule& rule = {});

inline std::vector<Period> swap_periods_to(const Date& value_date, const std::string& cal_id,
                                           const Date& maturity_date, const std::string& freq_tok,
                                           const std::string& bdc, int spot_lag, const ScheduleRule& rule = {}) {
  return swap_periods_between(spot_date(value_date, cal_id, spot_lag), cal_id, maturity_date, freq_tok, bdc, rule);
}

// The same roll from an ARBITRARY start date (`spot` here is just the first accrual start): what a BOOKED
// trade needs -- its periods are anchored at its own effective date, which for a seasoned or forward-
// starting deal is not today's spot. swap_periods_to is exactly this with start = spot(value_date).
inline std::vector<Period> swap_periods_between(const Date& spot, const std::string& cal_id,
                                                const Date& maturity_date, const std::string& freq_tok,
                                                const std::string& bdc, const ScheduleRule& rule) {
  if (maturity_date <= spot)
    throw std::invalid_argument("schedule: maturity " + iso(maturity_date) + " is on/before the start " + iso(spot) +
                                " (a swap needs a positive accrual span)");
  const Step step = tok_step(freq_tok);  // months (3M/1Y) or days (28D/1W) — never both
  // The schedule ENDS on the termination date rolled by the convention: a maturity booked on a non-business day
  // accrues to the business day it rolls to (QuantLib's Schedule does the same), and adjusting an already-adjusted
  // date changes nothing. Before 2026-09-14 the raw date closed the last period, so a trade booked to a Sunday
  // anniversary accrued to the Sunday (tests/trade_weekend_maturity_repro_test.cpp). The regular grid and the roll
  // day below still come from the UNADJUSTED dates -- roll conventions anchor on unadjusted dates.
  const Date end = adjust(cal_id, maturity_date, bdc);

  // Regular grid, forward from `spot` (the legacy path, taken verbatim for monthly steps so the default
  // output can never drift by even one ulp of date; day-based steps, e.g. MXN 28D, step by days).
  if (rule.is_default()) {
    std::vector<Date> bounds{spot};
    for (int k = 1;; ++k) {
      const Date d = adjust(cal_id, plus_step(spot, step, k), bdc);
      if (d >= end) break;
      bounds.push_back(d);
    }
    bounds.push_back(end);
    std::vector<Period> out;
    out.reserve(bounds.size() - 1);
    for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.emplace_back(bounds[i], bounds[i + 1]);
    return out;
  }

  // Extended ISDA path. The roll anchor and direction depend on the stub side. `dom`/`eom` describe the
  // day-of-month the regular (interior) boundaries land on before business-day adjustment.
  if (step.days) throw std::invalid_argument("schedule: ISDA EOM/stub/roll-day rules need a monthly frequency, got " + freq_tok);
  const int step_m = step.months;
  const Date anchor = (rule.side == StubSide::Back) ? spot : maturity_date;
  const bool eom = rule.eom || (rule.eom_auto && is_month_end(anchor));
  const int dom = rule.roll_dom > 0 ? rule.roll_dom : int(anchor.day());

  // Build the UNADJUSTED interior regular boundaries, strictly between spot and maturity, ascending.
  // `on_grid` records whether the far end sits EXACTLY on a regular boundary (then there is no stub).
  std::vector<Date> interior;
  bool on_grid = false;
  if (rule.side == StubSide::Back) {
    for (int k = 1;; ++k) {
      const Date base = anchor.plus_months(k * step_m);
      const Date d = roll_in_month(base.year(), base.month(), dom, eom);
      if (d >= maturity_date) { on_grid = (d == maturity_date); break; }
      if (d > spot) interior.push_back(d);
    }
  } else {  // Front: step BACKWARD from maturity, collect descending then reverse.
    for (int k = 1;; ++k) {
      const Date base = anchor.plus_months(-k * step_m);
      const Date d = roll_in_month(base.year(), base.month(), dom, eom);
      if (d <= spot) { on_grid = (d == spot); break; }
      if (d < maturity_date) interior.push_back(d);
    }
    for (std::size_t i = 0, j = interior.size(); i + 1 < j; ++i, --j)
      std::swap(interior[i], interior[j - 1]);
  }

  // LONG stub: merge the odd period into its neighbour by dropping the interior boundary adjacent to the
  // stub end — ONLY when a stub exists (the far end is off the regular grid). Until 2026-09-09 this also
  // merged two REGULAR periods when the maturity sat exactly on the grid (audit E28).
  if (rule.length == StubLen::Long && !interior.empty() && !on_grid) {
    if (rule.side == StubSide::Back)
      interior.pop_back();
    else
      interior.erase(interior.begin());
  }

  std::vector<Date> bounds{spot};
  for (const Date& d : interior) {
    const Date a = adjust(cal_id, d, bdc);
    // An unadjusted boundary just before the maturity can roll onto the end, and one just after spot can roll back ONTO
    // spot (Modified Following at a weekend month-end): neither is a period boundary (FS1, 2026-09-14; QuantLib's
    // Schedule drops both).
    if (a < end && a > bounds.back()) bounds.push_back(a);
  }
  bounds.push_back(end);
  std::vector<Period> out;
  out.reserve(bounds.size() - 1);
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) out.emplace_back(bounds[i], bounds[i + 1]);
  return out;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_SCHEDULE_HPP
