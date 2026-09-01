#pragma once
// FixingSeries — the time series of ONE index's realized fixings, as a first-class value object.
//
// This is the richer replacement for what pricing/fixings.hpp's FixingTable stores per index: that class
// is a flat map<index, map<date_serial, rate>> of scalars with exact-only lookup (has/at/get) and an
// observer protocol — it has NO as-of / time-series semantics. A FixingSeries is the SERIES for a single
// index, kept in ascending date order, and adds the query the flat map lacks: `as_of(d)` returns the most
// recent fixing ON OR BEFORE `d` (spanning gaps in the series), plus latest()/latest_date()/count().
//
// Clean standalone value type: value semantics, header-only, QuantLib-free. Depends ONLY on build/date.hpp
// (for the Date overloads / serials) and the standard library — it deliberately does NOT depend on
// FixingTable. Rates are DECIMALS (0.043), not percent, matching the engine convention.

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "swaps/build/date.hpp"

namespace swaps::market {

// The realized-fixing time series for one index. Dates are integer serials (build::Date::serial()).
class FixingSeries {
 public:
  std::string index;  // which index this series is for (e.g. "USD-SOFR")

  FixingSeries() = default;
  explicit FixingSeries(std::string idx) : index(std::move(idx)) {}

  // ---- mutation -----------------------------------------------------------------------------------
  // Upsert a single fixing (overwrites any existing value on that date). Rate is a DECIMAL (0.043).
  void add(int date_serial, double rate) { data_[date_serial] = rate; }
  void add(const build::Date& d, double rate) { add(int(d.serial()), rate); }

  // ---- exact lookup -------------------------------------------------------------------------------
  bool has(int date_serial) const { return data_.find(date_serial) != data_.end(); }
  bool has(const build::Date& d) const { return has(int(d.serial())); }

  // Exact fixing on `date_serial`; throws if none is present on that exact date.
  double at(int date_serial) const {
    auto it = data_.find(date_serial);
    if (it == data_.end())
      throw std::out_of_range("FixingSeries('" + index + "'): no fixing on date-serial " +
                              std::to_string(date_serial));
    return it->second;
  }
  double at(const build::Date& d) const { return at(int(d.serial())); }

  // ---- as-of query (the key semantic the flat FixingTable lacks) ----------------------------------
  // True iff at least one fixing lies ON OR BEFORE `date_serial`.
  bool has_as_of(int date_serial) const {
    // upper_bound = first strictly AFTER date_serial; anything before it is on-or-before.
    return data_.upper_bound(date_serial) != data_.begin();
  }
  bool has_as_of(const build::Date& d) const { return has_as_of(int(d.serial())); }

  // The most recent fixing ON OR BEFORE `date_serial` (spanning any gaps in the series). Throws if the
  // series starts strictly after `date_serial` (i.e. no fixing is yet available as of that date).
  double as_of(int date_serial) const {
    auto it = data_.upper_bound(date_serial);  // first entry strictly AFTER date_serial
    if (it == data_.begin())
      throw std::out_of_range("FixingSeries('" + index + "'): no fixing as of date-serial " +
                              std::to_string(date_serial) + " (series starts later)");
    --it;  // step back to the last entry on-or-before date_serial
    return it->second;
  }
  double as_of(const build::Date& d) const { return as_of(int(d.serial())); }

  // ---- latest / size ------------------------------------------------------------------------------
  // Most recent fixing in the series; throws if empty.
  double latest() const {
    if (data_.empty()) throw std::out_of_range("FixingSeries('" + index + "'): empty, no latest fixing");
    return data_.rbegin()->second;
  }
  // Date-serial of the most recent fixing; throws if empty.
  int latest_date() const {
    if (data_.empty()) throw std::out_of_range("FixingSeries('" + index + "'): empty, no latest date");
    return data_.rbegin()->first;
  }

  int count() const { return int(data_.size()); }
  bool empty() const { return data_.empty(); }

 private:
  std::map<int, double> data_;  // date-serial -> rate, kept ascending by std::map
};

}  // namespace swaps::market
