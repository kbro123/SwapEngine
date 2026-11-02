#pragma once
// A European swaption's underlying forward-starting swap, in curve time: the option expiry, the swap start (spot
// after expiry on the product's calendar) and the FIXED leg's pay times + accruals on the product's own schedule
// (frequency / bdc / pay lag / day count -- it used to be hard-wired annual with no pay lag, wrong for every
// non-annual-fixed market: SAR/AUD/CNY/ZAR/CAD...). THE one walk (2026-11-01, review item 2.2): the swaption verb,
// the vol cube, the compiled VolSurface and the vega ladder each carried a copy.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::build {

// One cell's CURVE-INDEPENDENT schedule in curve time (ACT/365F). Building it walks the calendar (business-day
// adjustment / holidays) for every pay date, which is the same for every reprice -- so a VolSurface resolves it once.
struct SwaptionSchedule {
  double t_start = 0.0;
  double t_expiry = 0.0;
  std::vector<double> pay_time;
  std::vector<double> tau;
};

// `what` prefixes the refusals ("vol_cube", "swaption", "vega").
inline SwaptionSchedule swaption_schedule(const Date& vd, const SwapConv& conv, const std::string& expiry,
                                          const std::string& tenor, const char* what) {
  if (expiry.empty() || tenor.empty())
    throw std::invalid_argument(std::string(what) + ": each cell needs a non-empty 'expiry' and 'tenor'");
  const Date expiry_date = resolve(expiry, vd, conv.calendar, conv.bdc, 0);  // throws on an unrecognized expiry token
  const Date swap_start = spot_date(expiry_date, conv.calendar, conv.spot_lag);
  const double tenor_years = conventions::period_years(tenor);  // 0 for an unrecognized token
  if (!(tenor_years > 0.0))
    throw std::invalid_argument(std::string(what) + ": unrecognized or non-positive tenor '" + tenor +
                                "' (use e.g. 2Y, 5Y, 10Y)");
  const int months = std::max(1, static_cast<int>(std::lround(tenor_years * 12.0)));
  SwaptionSchedule s;
  s.t_start = curve_time(vd, swap_start);
  s.t_expiry = curve_time(vd, expiry_date);
  const Date und_mat = swap_start.plus_months(months);
  for (const auto& [ps, pe] : swap_periods_between(swap_start, conv.calendar, und_mat, conv.fixed_freq_tok, conv.bdc)) {
    s.tau.push_back(year_frac(conv.fixed_dc, ps, pe));
    s.pay_time.push_back(curve_time(vd, advance_bd(conv.calendar, pe, conv.pay_lag)));
  }
  return s;
}

}  // namespace swaps::build
