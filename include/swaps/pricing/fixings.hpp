#pragma once
// Fixing tables as part of the PRICING CONTEXT (design: attach-and-observe).
//
// An instrument's observation splits at the evaluation date: fixing dates in the PAST are already fixed
// and must come from a fixing table; dates in the FUTURE are forecast from the curve. Today this split is
// done in the market-construction layer (tests/reference_curve.hpp reads QuantLib's fixing history and
// bakes a constant `realized` into the RateObservation). This header moves the fixing table INTO the
// engine so it is:
//   * AMBIENT   — attached to the pricing context, not requested per-instrument;
//   * OBSERVED  — a `FixingTable` is Observable; a resolvable observation registers as an Observer, so
//                 when the table gains/《changes》fixings the observation re-resolves (you can build the
//                 context WITHOUT a table, attach one later, and updates propagate);
//   * STRICT    — resolving a past fixing that is in neither the curve (unforecastable) nor the table
//                 THROWS MissingFixing; pricing that instrument cannot silently proceed.
//
// The RESOLVED result is a plain RateObservation (constant `realized`/`realized_factor` + forecast
// sub-periods) — bit-identical to today's baked observation — so the W-cache / AAD hot paths are
// unchanged. Resolution runs on attach and on each table update, and is cached in between. QuantLib-free
// (the shipped core links no QuantLib): dates are Unix-day serials (days since 1970-01-01 == build::Date::serial();
// 2026-01-01 == 20454), index is a string. This is the engine's ONE integer date convention — FixingSeries, the
// CB meeting schedule and the builders' fixing schedules all use it; set() REJECTS a value outside the plausible
// range so a Python date.toordinal() (739xxx) can never be stored as a date.

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

// ---- Observable / Observer (lightweight, single-threaded; QuantLib-free) --------------------------
class Observer {
 public:
  virtual ~Observer() = default;
  virtual void on_fixings_changed() = 0;
};

class Observable {
 public:
  void add_observer(Observer* o) {
    if (o) observers_.push_back(o);
  }
  void remove_observer(Observer* o) {
    for (auto it = observers_.begin(); it != observers_.end(); ++it)
      if (*it == o) { observers_.erase(it); return; }
  }
 protected:
  void notify() {
    // copy so an observer that (de)registers during notification can't invalidate the loop
    const std::vector<Observer*> snapshot = observers_;
    for (Observer* o : snapshot) o->on_fixings_changed();
  }
 private:
  std::vector<Observer*> observers_;
};

// ---- Fixing table (index -> date-serial -> rate), Observable --------------------------------------
class FixingTable : public Observable {
 public:
  // Upsert a single fixing and notify observers. Rate is a DECIMAL (0.043), not percent.
  void set(const std::string& index, int date, double rate) {
    check_date_serial(date, "FixingTable::set");
    data_[index][date] = rate;
    notify();
  }
  // Bulk load without notifying per row; notifies ONCE at the end (use for a table refresh).
  void bulk_set(const std::string& index, const std::vector<std::pair<int, double>>& rows) {
    for (const auto& [d, r] : rows) check_date_serial(d, "FixingTable::bulk_set");
    auto& m = data_[index];
    for (const auto& [d, r] : rows) m[d] = r;
    notify();
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

// ---- A resolvable observation's per-day schedule --------------------------------------------------
// FixingDay lives in cashflows.hpp (RateObservation carries a std::vector<FixingDay>). The standalone
// FixingSchedule below is the same data as {index, RateObservation.fixing_schedule, tau_index, compounded}
// and is used by resolve()/tests; resolve_into() operates directly on an observation that carries one.

// The unresolved schedule for one observation: enough to split past/future against a context and either
// look up (past) or forecast (future) each day.
struct FixingSchedule {
  std::string index;
  std::vector<FixingDay> days;
  double tau_index = 0.0;
  bool compounded = false;
};

// Resolve a schedule against a context into a plain RateObservation. Past days (fixing_date < eval, or ==
// eval with a fixing present) come from the table — throwing MissingFixing if absent; future days become
// forecast sub-periods. Bit-identical to the hand-baked observation the market layer builds today.
inline RateObservation resolve(const FixingSchedule& sch, const PricingContext& ctx) {
  RateObservation obs;
  obs.tau_index = sch.tau_index;
  obs.compounded = sch.compounded;
  double realized = 0.0, rf = 1.0;
  bool all_one = true;
  for (const FixingDay& d : sch.days) {
    const bool is_past =
        d.fixing_date < ctx.evaluation_date ||
        (d.fixing_date == ctx.evaluation_date && ctx.fixings && ctx.fixings->has(sch.index, d.fixing_date));
    if (is_past) {
      if (!ctx.fixings) throw MissingFixing(sch.index, d.fixing_date);  // past day, no table attached
      const double fx = ctx.fixings->at(sch.index, d.fixing_date);       // throws if absent
      if (sch.compounded) rf *= (1.0 + fx * d.accrual);
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
  return obs;
}

// Resolve an observation that carries its own schedule (obs.fixing_index + obs.fixing_schedule), IN PLACE:
// writes realized / realized_factor and the forecast sub-periods from the context, preserving the schedule
// (so it can re-resolve later). No-op if the observation carries no schedule (legacy baked path). Throws
// MissingFixing if a required past fixing is absent. This is what the BundleSession calls on build + on
// each fixing-table update.
inline void resolve_into(RateObservation& obs, const PricingContext& ctx) {
  if (obs.fixing_schedule.empty()) return;  // already-baked observation — leave untouched
  FixingSchedule sch;
  sch.index = obs.fixing_index;
  sch.days = obs.fixing_schedule;
  sch.tau_index = obs.tau_index;
  sch.compounded = obs.compounded;
  const RateObservation r = resolve(sch, ctx);
  obs.realized = r.realized;
  obs.realized_factor = r.realized_factor;
  obs.sub_start = r.sub_start;
  obs.sub_end = r.sub_end;
  obs.weight = r.weight;
  obs.resolved = true;  // the kernels may now price it (see RateObservation::resolved)
}

// ---- A context-bound observation: resolves on attach + re-resolves whenever the table updates -------
// Holds the schedule + a cache of the resolved RateObservation. Register it as an observer of the table
// so `on_fixings_changed()` re-resolves. `error()` is set (instead of throwing) when a required fixing is
// missing, so a bundle can report which instrument is un-priceable rather than aborting the whole update.
class BoundObservation : public Observer {
 public:
  BoundObservation(FixingSchedule sch, PricingContext ctx) : sch_(std::move(sch)), ctx_(ctx) {
    if (ctx_.fixings) const_cast<FixingTable*>(ctx_.fixings)->add_observer(this);
    reresolve();
  }
  ~BoundObservation() override {
    if (ctx_.fixings) const_cast<FixingTable*>(ctx_.fixings)->remove_observer(this);
  }
  BoundObservation(const BoundObservation&) = delete;
  BoundObservation& operator=(const BoundObservation&) = delete;

  void on_fixings_changed() override { reresolve(); }

  // Re-point at a new context/table (e.g. build with no table, then attach one): unregister from the old
  // table, register with the new, and re-resolve. This is the "add the fixing table later" path.
  void attach(PricingContext ctx) {
    if (ctx_.fixings) const_cast<FixingTable*>(ctx_.fixings)->remove_observer(this);
    ctx_ = ctx;
    if (ctx_.fixings) const_cast<FixingTable*>(ctx_.fixings)->add_observer(this);
    reresolve();
  }

  bool ok() const { return !error_.has_value(); }
  const std::optional<MissingFixing>& error() const { return error_; }
  // Valid only when ok(); the resolved observation the pricer consumes.
  const RateObservation& observation() const { return obs_; }

 private:
  void reresolve() {
    try {
      obs_ = resolve(sch_, ctx_);
      error_.reset();
    } catch (const MissingFixing& e) {
      error_ = e;  // stay alive but un-priceable until the table gains the fixing
    }
  }
  FixingSchedule sch_;
  PricingContext ctx_;
  RateObservation obs_;
  std::optional<MissingFixing> error_;
};

}  // namespace swaps::pricing
