#pragma once
// Fixing tables as part of the PRICING CONTEXT.
//
// An instrument's observation splits at the evaluation date: fixing dates in the PAST are already fixed and
// must come from a fixing table; dates in the FUTURE are forecast from the curve. A coupon built by the
// instrument builders carries that split as a FIXING SCHEDULE (RateObservation::fixing_schedule, one FixingDay
// per fixing); resolve_into() splits it against a PricingContext {evaluation date, table} IN PLACE, writing the
// constant `realized` / `realized_factor` and the forecast sub-periods -- bit-identical to a hand-baked
// observation, so the W-cache / AAD hot paths never see a schedule. BundleSession calls it on build and after
// every fixing-table update (resolve_fixings). STRICT: a past day that is in neither the curve nor the table
// throws MissingFixing; pricing that instrument cannot silently proceed.
//
// (Until 2026-11-01 this header also carried an attach-and-observe design -- Observer / Observable, a standalone
// FixingSchedule, a BoundObservation that re-resolved itself on table updates. No session or verb used it: the
// session resolves in place and re-resolves itself. Deleted, review item 1.5.)
//
// QuantLib-free (the shipped core links no QuantLib): dates are Unix-day serials (days since 1970-01-01 ==
// build::Date::serial(); 2026-01-01 == 20454), index is a string. This is the engine's ONE integer date
// convention -- the CB meeting schedule and the builders' fixing schedules all use it; set() REJECTS a value
// outside the plausible range so a Python date.toordinal() (739xxx) can never be stored as a date.

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/pricing/cashflows.hpp"

namespace swaps::pricing {

// The plausible range of a Unix-day serial for a market date: 1900-01-01 (-25567) .. 2299-12-31 (120528).
// A Python proleptic ordinal (1970-01-01 == 719163) or a YYYYMMDD integer falls far outside it.
inline void check_date_serial(int serial, const char* what) {
  if (serial < -25567 || serial > 120528)
    throw std::invalid_argument(std::string(what) + ": date " + std::to_string(serial) +
                                " is not a Unix-day serial (days since 1970-01-01, build::Date::serial()); a date.toordinal() value is not accepted");
}

// Thrown when a required PAST fixing is absent from the table (and cannot be forecast from the curve).
struct MissingFixing : std::runtime_error {
  std::string index;
  int date;  // Unix-day serial
  MissingFixing(std::string idx, int d)
      : std::runtime_error("missing fixing for index '" + idx + "' on date-serial " + std::to_string(d)),
        index(std::move(idx)), date(d) {}
};

// ---- Fixing table (index -> date-serial -> rate) -------------------------------------------------
class FixingTable {
 public:
  // Upsert a single fixing. Rate is a DECIMAL (0.043), not percent.
  void set(const std::string& index, int date, double rate) {
    check_date_serial(date, "FixingTable::set");
    data_[index][date] = rate;
  }
  // Bulk load (a table refresh).
  void bulk_set(const std::string& index, const std::vector<std::pair<int, double>>& rows) {
    for (const auto& [d, r] : rows) check_date_serial(d, "FixingTable::bulk_set");
    auto& m = data_[index];
    for (const auto& [d, r] : rows) m[d] = r;
  }
  std::optional<double> get(const std::string& index, int date) const {
    auto i = data_.find(index);
    if (i == data_.end()) return std::nullopt;
    auto j = i->second.find(date);
    if (j == i->second.end()) return std::nullopt;
    return j->second;
  }
  bool has(const std::string& index, int date) const { return get(index, date).has_value(); }
  double at(const std::string& index, int date) const {
    if (auto v = get(index, date)) return *v;
    throw MissingFixing(index, date);
  }
  std::size_t size(const std::string& index) const {
    auto i = data_.find(index);
    return i == data_.end() ? 0 : i->second.size();
  }

 private:
  std::map<std::string, std::map<int, double>> data_;
};

// ---- Pricing context: evaluation date + a (possibly null) fixing table ----------------------------
struct PricingContext {
  int evaluation_date = 0;              // Unix-day serial; fixings strictly before this are "already fixed"
  const FixingTable* fixings = nullptr;  // may be null until a table is attached
};

// ---- Resolution --------------------------------------------------------------------------------------
// Resolve an observation that carries its own schedule (obs.fixing_index + obs.fixing_schedule) against a context,
// IN PLACE: past days (fixing_date < eval, or == eval with a fixing present) come from the table -- throwing
// MissingFixing if absent -- into realized (averaged) / realized_factor (compounded); future days become the
// forecast sub-periods. The schedule is preserved so the observation can re-resolve on a later table or date.
// No-op if the observation carries no schedule (a baked observation). Bit-identical to the hand-baked
// observation the market layer builds.
inline void resolve_into(RateObservation& obs, const PricingContext& ctx) {
  if (obs.fixing_schedule.empty()) return;  // already-baked observation -- leave untouched
  double realized = 0.0, rf = 1.0;
  bool all_one = true;
  obs.sub_start.clear();
  obs.sub_end.clear();
  obs.weight.clear();
  for (const FixingDay& d : obs.fixing_schedule) {
    // A day that STARTS before curve time 0 (the value date the schedule was built on) is in the past by construction,
    // whatever the context's evaluation date says: before 2026-09-14 a session with no evaluation date (0) called every
    // such day the future and FORECAST a seasoned coupon's realized part off a curve that starts at the value date
    // (tests/seasoned_eval_date_repro_test.cpp). It needs a fixing like any other past day.
    const bool is_past =
        d.t_start < 0.0 || d.fixing_date < ctx.evaluation_date ||
        (d.fixing_date == ctx.evaluation_date && ctx.fixings && ctx.fixings->has(obs.fixing_index, d.fixing_date));
    if (is_past) {
      if (!ctx.fixings) throw MissingFixing(obs.fixing_index, d.fixing_date);  // past day, no table attached
      const double fx = ctx.fixings->at(obs.fixing_index, d.fixing_date);       // throws if absent
      if (obs.compounded) rf *= (1.0 + fx * d.accrual);
      else realized += fx * d.accrual;
    } else {
      obs.sub_start.push_back(d.t_start);
      obs.sub_end.push_back(d.t_end);
      obs.weight.push_back(d.weight);
      if (d.weight != 1.0) all_one = false;
    }
  }
  if (all_one) obs.weight.clear();  // empty == all-ones (keeps the standard reduction bit-exact)
  obs.realized = realized;
  obs.realized_factor = rf;
  obs.resolved = true;  // the kernels may now price it (see RateObservation::resolved)
}

}  // namespace swaps::pricing
