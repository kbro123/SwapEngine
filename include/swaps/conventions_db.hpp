// swaps::conventions — the RUNTIME conventions registry (PRINCIPLES.md P2).
//
// The codegen'd arrays in conventions_data.hpp (kCurrencies / kCalendars / kIndices / kProducts / kBonds) are the
// DEFAULT set: the JSON baked at build time, zero parsing cost. This header adds the one thing the baked arrays
// cannot do — let ANY API add or override an entry at runtime (the `conventions` run_json verb, the C ABI, pybind)
// without a rebuild — and defines the lookups every consumer uses:
//
//   product(id) / index(id) / bond(id) / currency(id) / calendar(id)   -> std::optional (overlay first, then baked)
//   require_product(id) / ...                                          -> the row, or throws std::invalid_argument
//
// A lookup miss is never silently defaulted (that was the 2026-09 audit's class-B finding: unknown calendar ⇒ US
// SIFMA, unknown currency ⇒ USD, empty day count ⇒ ACT/360). Callers that need a default ask the DB for it
// (currencies[ccy].default_swap_product, .settlement_calendar, .discount_index) — the default is DATA.
//
// This file is #included at the END of conventions_data.hpp (it needs the struct types and the arrays); include
// "swaps/conventions_data.hpp", never this file directly.
#ifndef SWAPS_CONVENTIONS_DB_HPP
#define SWAPS_CONVENTIONS_DB_HPP
#ifndef SWAPS_CONVENTIONS_DATA_INCLUDED
#error "include swaps/conventions_data.hpp (conventions_db.hpp is its tail)"
#endif

#include <atomic>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace swaps::conventions {

// A calendar row together with the storage its rule/join slices point into (baked arrays or the overlay).
struct CalendarView {
  CalendarConv row;
  const HolidayRule* rules = nullptr;       // rules[0 .. row.rule_count)
  const std::string_view* joins = nullptr;  // joins[0 .. row.join_count)
};

// ---- row rules (E7 stage 4.2) ------------------------------------------------------------------------------------
// conventions/conventions.schema.json's per-row constraints, enforced on EVERY runtime row whoever adds it (the
// `conventions` verb, pybind, a C++ caller). Before 2026-09-14 the registry checked nothing and the verb only a few
// fields: a bond row without stub_discount priced as Compound, weekend day 7 set a bit on_weekend never reads, a
// recovery of 1.5 was accepted (tests/conventions_repro_test.cpp). check_schema.py holds the baked DB to the same
// schema and tests/conventions_rules_test.cpp holds every baked row to these rules, so they are never stricter than the
// data, and the enum vocabularies are GENERATED from the schema (kSchema* in conventions_data.hpp), so it stays their
// one source. On a struct a lag of -1 means unset: a required lag must be >= its minimum, an optional one >= -1 (the JSON
// decoder rejects a negative lag before it gets here). What a struct cannot see stays with the decoder: JSON-only
// fields (description), an absent key versus "", an empty `{}` leg, a leg spelt two ways, a real calendar date.
namespace rules {

[[noreturn]] inline void reject(std::string_view family, std::string_view id, const std::string& why) {
  throw std::invalid_argument("conventions: " + std::string(family) + " '" + std::string(id) + "' " + why);
}

inline bool leg_present(const LegConv& l) {
  return !l.index.empty() || !l.day_count.empty() || !l.frequency.empty() || !l.compounding.empty() ||
         l.carries_spread || l.notional_resets || l.flat;
}

// The row under check: its family and id name every message.
struct Row {
  std::string_view family, id;

  [[noreturn]] void fail(const std::string& why) const { reject(family, id, why); }
  void need(std::string_view v, std::string_view field) const {
    if (v.empty()) fail("needs '" + std::string(field) + "'");
  }
  void at_least(int v, std::string_view field, int min) const {
    if (v < min) fail("needs '" + std::string(field) + "' >= " + std::to_string(min));
  }
  void unset_or_non_negative(int v, std::string_view field) const {
    if (v < -1) fail("'" + std::string(field) + "' must be >= 0");
  }
  template <class Vocabulary>  // one of the generated kSchema* arrays (conventions_data.hpp)
  void one_of(std::string_view v, std::string_view field, const Vocabulary& allowed) const {
    for (std::string_view a : allowed)
      if (v == a) return;
    std::string list;
    for (std::string_view a : allowed) list += (list.empty() ? "" : " | ") + std::string(a);
    fail("'" + std::string(field) + "' must be " + list + ", got '" + std::string(v) + "'");
  }
  void tenor(std::string_view v, std::string_view field) const {  // ^[0-9]+[DWMY]$, when set
    if (v.empty()) return;
    bool ok = v.size() >= 2 && (v.back() == 'D' || v.back() == 'W' || v.back() == 'M' || v.back() == 'Y');
    for (std::size_t k = 0; ok && k + 1 < v.size(); ++k) ok = v[k] >= '0' && v[k] <= '9';
    if (!ok) fail("'" + std::string(field) + "' must be a tenor like 3M, got '" + std::string(v) + "'");
  }
  void iso_date(std::string_view v, std::string_view field) const {  // ^[0-9]{4}-[0-9]{2}-[0-9]{2}$, when set
    if (v.empty()) return;
    bool ok = v.size() == 10;
    for (std::size_t k = 0; ok && k < v.size(); ++k) ok = (k == 4 || k == 7) ? v[k] == '-' : (v[k] >= '0' && v[k] <= '9');
    if (!ok) fail("'" + std::string(field) + "' must be YYYY-MM-DD, got '" + std::string(v) + "'");
  }
  void leg(const LegConv& l, std::string_view name) const {  // the shape of whatever a leg sets
    tenor(l.frequency, std::string(name) + ".frequency");
    if (!l.compounding.empty()) one_of(l.compounding, std::string(name) + ".compounding", kSchemaLegCompounding);
  }
  void need_leg(const LegConv& l, std::string_view name) const {
    if (!leg_present(l)) fail("needs '" + std::string(name) + "'");
  }
};

}  // namespace rules

inline void validate(const ProductConv& p) {
  const rules::Row r{"product", p.id};
  r.need(p.id, "id");
  r.one_of(p.type, "type", kSchemaProductTypes);
  r.need(p.calendar, "calendar");
  if (!p.bdc.empty()) r.one_of(p.bdc, "bdc", kSchemaBusinessDayConventions);
  r.tenor(p.frequency, "frequency");
  r.unset_or_non_negative(p.spot_lag, "spot_lag");
  r.unset_or_non_negative(p.payment_lag, "payment_lag");
  r.unset_or_non_negative(p.exchange_lag_initial, "exchange_lag_initial");
  r.unset_or_non_negative(p.exchange_lag_intermediate, "exchange_lag_intermediate");
  r.unset_or_non_negative(p.exchange_lag_final, "exchange_lag_final");
  r.unset_or_non_negative(p.fx_reset_fixing_lag, "fx_reset_fixing_lag");
  const bool swap = p.type == "ois" || p.type == "irs", basis = p.type == "basis", xccy = p.type == "xccy_mtm";
  const std::string_view floating = basis ? "spread_leg" : xccy ? "usd_leg" : "float_leg";
  const std::string_view other = xccy ? "eur_leg" : "flat_leg";
  r.leg(p.fixed, "fixed_leg");
  r.leg(p.floating, floating);
  r.leg(p.other, other);
  if (p.zero_coupon && !swap) r.fail("is zero_coupon, which needs type ois | irs");
  if (!swap && !basis && !xccy) return;  // fx_forward / future / administered-basis: description, type, calendar
  r.need(p.bdc, "bdc");
  r.at_least(p.spot_lag, "spot_lag", 0);
  r.at_least(p.payment_lag, "payment_lag", 0);
  if (xccy) {
    r.need(p.pair, "pair");
    r.need(p.base_currency, "base_currency");
    r.need(p.frequency, "frequency");
    r.at_least(p.exchange_lag_initial, "exchange_lag_initial", 0);
    r.at_least(p.exchange_lag_intermediate, "exchange_lag_intermediate", 0);
    r.at_least(p.exchange_lag_final, "exchange_lag_final", 0);
    r.at_least(p.fx_reset_fixing_lag, "fx_reset_fixing_lag", 0);
    r.need(p.fx_reset_calendar, "fx_reset_calendar");
    r.need_leg(p.floating, floating);
    r.need_leg(p.other, other);
    return;
  }
  r.need(p.currency, "currency");
  const auto full_leg = [&r](const LegConv& l, std::string_view name) {
    r.need_leg(l, name);
    r.need(l.index, std::string(name) + ".index");
    r.need(l.day_count, std::string(name) + ".day_count");
    r.need(l.frequency, std::string(name) + ".frequency");
  };
  if (basis) {
    r.need(p.discount_index, "discount_index");
    full_leg(p.floating, floating);
    full_leg(p.other, other);
    return;
  }
  if (!p.zero_coupon) {
    r.need_leg(p.fixed, "fixed_leg");
    r.need(p.fixed.day_count, "fixed_leg.day_count");
    r.need(p.fixed.frequency, "fixed_leg.frequency");
    full_leg(p.floating, floating);
    return;
  }
  // Zero-coupon: one period spot -> maturity, so neither leg carries a frequency.
  r.need_leg(p.fixed, "fixed_leg");
  r.need(p.fixed.day_count, "fixed_leg.day_count");
  r.need_leg(p.floating, floating);
  r.need(p.floating.index, "float_leg.index");
  r.need(p.floating.day_count, "float_leg.day_count");
  if (!p.fixed.frequency.empty() || !p.floating.frequency.empty())
    r.fail("is zero_coupon: its legs carry no frequency");
}

inline void validate(const IndexConv& i) {
  const rules::Row r{"index", i.id};
  r.need(i.id, "id");
  r.need(i.currency, "currency");
  r.one_of(i.type, "type", kSchemaIndexTypes);
  r.need(i.day_count, "day_count");
  r.need(i.calendar, "calendar");
  r.tenor(i.tenor, "tenor");
  r.unset_or_non_negative(i.fixing_lag, "fixing_lag");
  r.unset_or_non_negative(i.publication_lag, "publication_lag");
}

inline void validate(const BondConv& b) {
  const rules::Row r{"bond", b.id};
  r.need(b.id, "id");
  r.need(b.currency, "currency");
  r.need(b.calendar, "calendar");
  r.need(b.day_count, "day_count");
  r.need(b.frequency, "frequency");
  r.at_least(b.settle_lag, "settle_lag", 0);
  r.one_of(b.stub_discount, "stub_discount", kSchemaStubDiscounts);
}

inline void validate(const CurrencyConv& c) {
  const rules::Row r{"currency", c.code};
  r.need(c.code, "code");
  r.need(c.name, "name");
  r.at_least(c.minor_units, "minor_units", 0);
  if (c.minor_units > 4) r.fail("'minor_units' must be <= 4");
  r.need(c.settlement_calendar, "settlement_calendar");
  r.need(c.discount_index, "discount_index");
  r.need(c.default_swap_product, "default_swap_product");
  r.need(c.repo_day_count, "repo_day_count");
}

inline void validate(const CreditConv& c) {
  const rules::Row r{"cds product", c.id};
  r.need(c.id, "id");
  r.need(c.currency, "currency");
  r.need(c.calendar, "calendar");
  r.need(c.day_count, "day_count");
  r.need(c.frequency, "frequency");
  r.tenor(c.frequency, "frequency");
  if (!(c.recovery_default >= 0.0 && c.recovery_default <= 1.0)) r.fail("needs 'recovery_default' in [0, 1]");
  r.at_least(c.settlement_lag, "settlement_lag", 0);
  r.at_least(c.protection_steps, "protection_steps", 1);
}

inline void validate(const BondFutureConv& f) {
  const rules::Row r{"bond future", f.id};
  r.need(f.id, "id");
  r.need(f.currency, "currency");
  r.need(f.exchange_calendar, "exchange_calendar");
  r.need(f.deliverable_convention, "deliverable_convention");
  r.need(f.repo_day_count, "repo_day_count");
  if (!(std::isfinite(f.notional_coupon) && f.notional_coupon >= 0.0)) r.fail("needs 'notional_coupon' >= 0");
  if (!std::isfinite(f.basket_min_years) || !std::isfinite(f.basket_max_years)) r.fail("basket years must be finite");
  r.at_least(f.maturity_rounding_months, "maturity_rounding_months", 1);
  r.at_least(f.conversion_factor_decimals, "conversion_factor_decimals", 0);
}

inline void validate(const FxPairConv& f) {
  const rules::Row r{"fx pair", f.id};
  r.need(f.base, "base");
  r.need(f.quote, "quote");
  if (std::string(f.base) + std::string(f.quote) != f.id) r.fail("id must equal base + quote");
  r.need(f.calendar, "calendar");
  r.need(f.premium_currency, "premium_currency");
  r.at_least(f.spot_lag, "spot_lag", 0);
  r.one_of(f.delta_convention, "delta_convention", kSchemaDeltaConventions);
  r.one_of(f.atm_convention, "atm_convention", kSchemaAtmConventions);
  if (!std::isfinite(f.smile_pillar_lo) || !std::isfinite(f.smile_pillar_hi)) r.fail("smile_pillars must be finite");
}

inline void validate(const FixingSourceConv& s) {  // the index it names is checked by Registry::apply
  const rules::Row r{"fixing source", s.id};
  r.need(s.id, "index id");
  r.one_of(s.provider, "provider", kSchemaFixingProviders);
  r.need(s.series, "series");
  r.one_of(s.granularity, "granularity", kSchemaFixingGranularities);
  r.iso_date(s.start, "start");
}

inline void validate(const InflationIndexConv& x) {
  const rules::Row r{"inflation index", x.id};
  r.need(x.id, "id");
  r.need(x.currency, "currency");
  r.need(x.calendar, "calendar");
  r.need(x.frequency, "frequency");
  r.tenor(x.frequency, "frequency");
  r.at_least(x.observation_lag_months, "observation_lag_months", 0);
  r.one_of(x.interpolation, "interpolation", kSchemaInflationInterpolations);
}

inline void validate(const CbScheduleConv& c, const std::vector<long>& meetings) {
  const rules::Row r{"cb schedule", c.currency};
  r.need(c.currency, "currency");
  r.need(c.bank, "bank");
  r.need(c.source, "source");
  r.need(c.as_of, "as_of");
  r.iso_date(c.as_of, "as_of");
  for (std::size_t k = 1; k < meetings.size(); ++k)
    if (meetings[k] <= meetings[k - 1]) r.fail("meetings must be strictly ascending");
}

inline void validate(const CalendarConv& c, const std::vector<HolidayRule>& rules,
                     const std::vector<std::string_view>& joins) {
  const rules::Row r{"calendar", c.id};
  r.need(c.id, "id");
  r.need(c.name, "name");
  if (c.weekend_mask & ~0x7F) r.fail("weekend days are 0 (Mon) .. 6 (Sun)");
  r.one_of(c.observance, "observance", kSchemaCalendarObservances);
  for (const HolidayRule& h : rules) {
    r.one_of(h.kind, "holiday rule", kSchemaHolidayRules);
    if (!h.observance.empty())
      r.one_of(h.observance, "holiday observance", kSchemaHolidayObservances);
  }
  for (std::string_view j : joins) r.need(j, "join entry");
}

// One runtime update, applied ALL-OR-NOTHING by Registry::apply. Families are committed in this order, rows in the
// order given (a later row with the same id wins a lookup, as before). The string_views need only outlive apply():
// the registry interns them.
struct OverlayBatch {
  struct Calendar {
    CalendarConv row;
    std::vector<HolidayRule> rules;
    std::vector<std::string_view> joins;
  };
  struct CbSchedule {
    CbScheduleConv row;
    std::vector<long> meetings;  // Unix-day serials
  };
  bool clear_first = false;  // drop the existing overlay in the same commit
  std::vector<CurrencyConv> currencies;
  std::vector<Calendar> calendars;
  std::vector<IndexConv> indices;
  std::vector<ProductConv> products;
  std::vector<BondConv> bonds;
  std::vector<CreditConv> credit_products;
  std::vector<BondFutureConv> bond_futures;
  std::vector<FxPairConv> fx_pairs;
  std::vector<CbSchedule> cb_schedules;
  std::vector<FixingSourceConv> fixing_sources;
  std::vector<InflationIndexConv> inflation_indices;

  std::size_t rows() const {
    return currencies.size() + calendars.size() + indices.size() + products.size() + bonds.size() +
           credit_products.size() + bond_futures.size() + fx_pairs.size() + cb_schedules.size() +
           fixing_sources.size() + inflation_indices.size();
  }
};

inline void validate(const OverlayBatch& b) {
  for (const auto& x : b.currencies) validate(x);
  for (const auto& x : b.calendars) validate(x.row, x.rules, x.joins);
  for (const auto& x : b.indices) validate(x);
  for (const auto& x : b.products) validate(x);
  for (const auto& x : b.bonds) validate(x);
  for (const auto& x : b.credit_products) validate(x);
  for (const auto& x : b.bond_futures) validate(x);
  for (const auto& x : b.fx_pairs) validate(x);
  for (const auto& x : b.cb_schedules) validate(x.row, x.meetings);
  for (const auto& x : b.fixing_sources) validate(x);
  for (const auto& x : b.inflation_indices) validate(x);
}

class Registry {
 public:
  static Registry& instance() {
    static Registry r;
    return r;
  }

  // ---- lookups: overlay (most recent wins) first, then the baked defaults ----------------------------
  std::optional<ProductConv> product(std::string_view id) const {
    if (auto o = find_overlay(products_, id)) return o;
    for (const auto& p : kProducts) if (p.id == id) return p;
    return std::nullopt;
  }
  std::optional<IndexConv> index(std::string_view id) const {
    if (auto o = find_overlay(indices_, id)) return o;
    for (const auto& i : kIndices) if (i.id == id) return i;
    return std::nullopt;
  }
  std::optional<BondConv> bond(std::string_view id) const {
    if (auto o = find_overlay(bonds_, id)) return o;
    for (const auto& b : kBonds) if (b.id == id) return b;
    return std::nullopt;
  }
  std::optional<CurrencyConv> currency(std::string_view code) const {
    if (overlay_n_.load(std::memory_order_acquire)) {
      std::shared_lock lk(mu_);
      for (auto it = currencies_.rbegin(); it != currencies_.rend(); ++it) if (it->code == code) return *it;
    }
    for (const auto& c : kCurrencies) if (c.code == code) return c;
    return std::nullopt;
  }
  std::optional<CreditConv> credit_product(std::string_view id) const {
    if (auto o = find_overlay(credit_, id)) return o;
    for (const auto& c : kCredit) if (c.id == id) return c;
    return std::nullopt;
  }
  std::optional<BondFutureConv> bond_future(std::string_view id) const {
    if (auto o = find_overlay(bond_futures_, id)) return o;
    for (const auto& f : kBondFutures) if (f.id == id) return f;
    return std::nullopt;
  }
  std::optional<FxPairConv> fx_pair(std::string_view id) const {
    if (auto o = find_overlay(fx_pairs_, id)) return o;
    for (const auto& f : kFxPairs) if (f.id == id) return f;
    return std::nullopt;
  }
  std::optional<FixingSourceConv> fixing_source(std::string_view index_id) const {
    if (auto o = find_overlay(fixing_sources_, index_id)) return o;
    for (const auto& s : kFixingSources) if (s.id == index_id) return s;
    return std::nullopt;
  }
  std::optional<InflationIndexConv> inflation_index(std::string_view id) const {
    if (auto o = find_overlay(inflation_, id)) return o;
    for (const auto& x : kInflationIndices) if (x.id == id) return x;
    return std::nullopt;
  }
  // Central-bank meeting dates for a currency (Unix-day serials, ascending); overlay REPLACES the baked list.
  std::optional<std::vector<long>> cb_meetings(std::string_view currency) const {
    if (overlay_n_.load(std::memory_order_acquire)) {
      std::shared_lock lk(mu_);
      for (std::size_t k = cb_schedules_.size(); k-- > 0;)
        if (cb_schedules_[k].currency == currency) return cb_meetings_[k];
    }
    for (const auto& c : kCbSchedules)
      if (c.currency == currency)
        return std::vector<long>(kCbMeetings.begin() + c.begin, kCbMeetings.begin() + c.begin + c.count);
    return std::nullopt;
  }
  std::optional<CalendarView> calendar(std::string_view id) const {
    if (overlay_n_.load(std::memory_order_acquire)) {
      std::shared_lock lk(mu_);
      for (std::size_t k = calendars_.size(); k-- > 0;)
        if (calendars_[k].id == id)
          return CalendarView{calendars_[k], cal_rules_[k].data(), cal_joins_[k].data()};
    }
    for (const auto& c : kCalendars)
      if (c.id == id) return CalendarView{c, kHolidayRules.data() + c.rule_begin, kCalendarJoins.data() + c.join_begin};
    return std::nullopt;
  }

  // ---- runtime overlay (add or override) ------------------------------------------------------------------
  // apply() is the ONE way in. Every row is checked (validate) before anything is touched; then, under ONE exclusive
  // lock, every string is interned and every vector reserved (the steps that can throw, invisible to readers), and
  // only then is the batch committed with no-throw moves and ONE generation bump. So a bad row -- or bad_alloc --
  // leaves the registry exactly as it was, and no reader sees half a batch. The fixing-source index check runs under
  // the same lock against a lock-free scan: calling index() here would re-lock the shared_mutex and deadlock.
  int apply(const OverlayBatch& b) {  // -> the overlay size this commit left
    validate(b);
    std::unique_lock lk(mu_);
    for (const FixingSourceConv& s : b.fixing_sources)
      if (!index_known_after(b, s.id))
        rules::reject("fixing source", s.id, "names an index the registry does not know (add the index first)");

    std::vector<CurrencyConv> currencies;
    for (const auto& x : b.currencies) currencies.push_back(owned(x));
    std::vector<CalendarConv> calendars;
    std::vector<std::vector<HolidayRule>> cal_rules;
    std::vector<std::vector<std::string_view>> cal_joins;
    for (const auto& x : b.calendars) {
      calendars.push_back(owned(x.row, x.rules.size(), x.joins.size()));
      std::vector<HolidayRule> rs;
      for (const auto& h : x.rules) rs.push_back(owned(h));
      cal_rules.push_back(std::move(rs));
      std::vector<std::string_view> js;
      for (const auto& j : x.joins) js.push_back(intern(j));
      cal_joins.push_back(std::move(js));
    }
    std::vector<IndexConv> indices;
    for (const auto& x : b.indices) indices.push_back(owned(x));
    std::vector<ProductConv> products;
    for (const auto& x : b.products) products.push_back(owned(x));
    std::vector<BondConv> bonds;
    for (const auto& x : b.bonds) bonds.push_back(owned(x));
    std::vector<CreditConv> credit;
    for (const auto& x : b.credit_products) credit.push_back(owned(x));
    std::vector<BondFutureConv> bond_futures;
    for (const auto& x : b.bond_futures) bond_futures.push_back(owned(x));
    std::vector<FxPairConv> fx_pairs;
    for (const auto& x : b.fx_pairs) fx_pairs.push_back(owned(x));
    std::vector<CbScheduleConv> cb_schedules;
    std::vector<std::vector<long>> cb_meetings;
    for (const auto& x : b.cb_schedules) {
      cb_schedules.push_back(owned(x.row, x.meetings.size()));
      cb_meetings.push_back(x.meetings);
    }
    std::vector<FixingSourceConv> fixing_sources;
    for (const auto& x : b.fixing_sources) fixing_sources.push_back(owned(x));
    std::vector<InflationIndexConv> inflation;
    for (const auto& x : b.inflation_indices) inflation.push_back(owned(x));
    const auto reserve = [](auto& dst, const auto& src) { dst.reserve(dst.size() + src.size()); };
    reserve(currencies_, currencies); reserve(calendars_, calendars); reserve(cal_rules_, cal_rules);
    reserve(cal_joins_, cal_joins); reserve(indices_, indices); reserve(products_, products); reserve(bonds_, bonds);
    reserve(credit_, credit); reserve(bond_futures_, bond_futures); reserve(fx_pairs_, fx_pairs);
    reserve(cb_schedules_, cb_schedules); reserve(cb_meetings_, cb_meetings);
    reserve(fixing_sources_, fixing_sources); reserve(inflation_, inflation);

    // Commit: nothing below throws (clear keeps capacity; every append fits the reservation).
    if (b.clear_first) clear_rows();
    const auto append = [](auto& dst, auto& src) {
      for (auto& x : src) dst.push_back(std::move(x));
    };
    append(currencies_, currencies); append(calendars_, calendars); append(cal_rules_, cal_rules);
    append(cal_joins_, cal_joins); append(indices_, indices); append(products_, products); append(bonds_, bonds);
    append(credit_, credit); append(bond_futures_, bond_futures); append(fx_pairs_, fx_pairs);
    append(cb_schedules_, cb_schedules); append(cb_meetings_, cb_meetings);
    append(fixing_sources_, fixing_sources); append(inflation_, inflation);
    const int after = (b.clear_first ? 0 : overlay_n_.load(std::memory_order_acquire)) + static_cast<int>(b.rows());
    overlay_n_.store(after, std::memory_order_release);
    gen_.fetch_add(1, std::memory_order_release);
    return after;
  }

  // One-row conveniences over apply(): the same rules, the same single commit.
  void add_product(const ProductConv& p) { OverlayBatch b; b.products.push_back(p); apply(b); }
  void add_index(const IndexConv& i) { OverlayBatch b; b.indices.push_back(i); apply(b); }
  void add_bond(const BondConv& x) { OverlayBatch b; b.bonds.push_back(x); apply(b); }
  void add_currency(const CurrencyConv& c) { OverlayBatch b; b.currencies.push_back(c); apply(b); }
  void add_credit_product(const CreditConv& c) { OverlayBatch b; b.credit_products.push_back(c); apply(b); }
  void add_bond_future(const BondFutureConv& f) { OverlayBatch b; b.bond_futures.push_back(f); apply(b); }
  void add_fx_pair(const FxPairConv& f) { OverlayBatch b; b.fx_pairs.push_back(f); apply(b); }
  void add_fixing_source(const FixingSourceConv& s) { OverlayBatch b; b.fixing_sources.push_back(s); apply(b); }
  void add_inflation_index(const InflationIndexConv& x) { OverlayBatch b; b.inflation_indices.push_back(x); apply(b); }
  void add_cb_schedule(const CbScheduleConv& c, std::vector<long> meetings) {
    OverlayBatch b;
    b.cb_schedules.push_back({c, std::move(meetings)});
    apply(b);
  }
  void add_calendar(const CalendarConv& c, const std::vector<HolidayRule>& rules,
                    const std::vector<std::string_view>& joins) {
    OverlayBatch b;
    b.calendars.push_back({c, rules, joins});
    apply(b);
  }
  // Drop every runtime entry (tests; a session reset). The baked defaults are untouched.
  void clear_overlay() {
    OverlayBatch b;
    b.clear_first = true;
    apply(b);
  }

  // ---- listings (the `list_conventions` verb): baked ids and overlay ids per family ----------------------
  struct Listing { std::vector<std::string> baked, overlay; };
  // Every family's ids and the overlay size read under ONE shared lock: a consistent snapshot even while another
  // thread applies a batch (the per-family list_* below each lock on their own).
  struct Listings {
    Listing currencies, calendars, indices, products, bonds, credit_products, bond_futures, fx_pairs, cb_schedules,
        fixing_sources, inflation_indices;
    int overlay_size = 0;
  };
  Listings listing() const {
    const auto fill = [](Listing& L, const auto& baked, const auto& overlay, auto id) {
      for (const auto& r : baked) L.baked.emplace_back(id(r));
      for (const auto& r : overlay) L.overlay.emplace_back(id(r));
    };
    const auto by_id = [](const auto& r) { return r.id; };
    Listings out;
    std::shared_lock lk(mu_);
    fill(out.currencies, kCurrencies, currencies_, [](const CurrencyConv& c) { return c.code; });
    fill(out.calendars, kCalendars, calendars_, by_id);
    fill(out.indices, kIndices, indices_, by_id);
    fill(out.products, kProducts, products_, by_id);
    fill(out.bonds, kBonds, bonds_, by_id);
    fill(out.credit_products, kCredit, credit_, by_id);
    fill(out.bond_futures, kBondFutures, bond_futures_, by_id);
    fill(out.fx_pairs, kFxPairs, fx_pairs_, by_id);
    fill(out.cb_schedules, kCbSchedules, cb_schedules_, [](const CbScheduleConv& c) { return c.currency; });
    fill(out.fixing_sources, kFixingSources, fixing_sources_, by_id);
    fill(out.inflation_indices, kInflationIndices, inflation_, by_id);
    out.overlay_size = overlay_n_.load(std::memory_order_acquire);
    return out;
  }
  Listing list_products() const { return listing().products; }
  Listing list_indices() const { return listing().indices; }
  Listing list_bonds() const { return listing().bonds; }
  Listing list_currencies() const { return listing().currencies; }
  Listing list_calendars() const { return listing().calendars; }
  Listing list_credit_products() const { return listing().credit_products; }
  Listing list_bond_futures() const { return listing().bond_futures; }
  Listing list_fx_pairs() const { return listing().fx_pairs; }
  Listing list_cb_schedules() const { return listing().cb_schedules; }
  Listing list_fixing_sources() const { return listing().fixing_sources; }
  Listing list_inflation_indices() const { return listing().inflation_indices; }
  int overlay_size() const { return overlay_n_.load(std::memory_order_acquire); }
  // Monotone count of registry mutations (add_* and clear_overlay): the key that invalidates derived caches
  // (build/calendar.hpp's per-year holiday bitmaps).
  unsigned long generation() const { return gen_.load(std::memory_order_acquire); }

 private:
  Registry() = default;
  template <class Row>
  std::optional<Row> find_overlay(const std::vector<Row>& rows, std::string_view id) const {
    if (!overlay_n_.load(std::memory_order_acquire)) return std::nullopt;
    std::shared_lock lk(mu_);
    for (auto it = rows.rbegin(); it != rows.rend(); ++it) if (it->id == id) return *it;
    return std::nullopt;
  }
  std::string_view intern(std::string_view s) {
    if (s.empty()) return {};
    pool_.emplace_back(s);           // std::deque: stable element addresses
    return pool_.back();
  }
  // Interned copies (caller holds the unique lock).
  LegConv owned(const LegConv& l) {
    LegConv o = l;
    o.index = intern(l.index); o.day_count = intern(l.day_count); o.frequency = intern(l.frequency); o.compounding = intern(l.compounding);
    return o;
  }
  ProductConv owned(const ProductConv& p) {
    ProductConv o = p;
    o.id = intern(p.id); o.type = intern(p.type); o.currency = intern(p.currency); o.calendar = intern(p.calendar);
    o.bdc = intern(p.bdc); o.frequency = intern(p.frequency); o.discount_index = intern(p.discount_index);
    o.pair = intern(p.pair); o.base_currency = intern(p.base_currency); o.fx_reset_calendar = intern(p.fx_reset_calendar);
    o.fixed = owned(p.fixed); o.floating = owned(p.floating); o.other = owned(p.other);
    return o;
  }
  IndexConv owned(const IndexConv& i) {
    IndexConv o = i;
    o.id = intern(i.id); o.currency = intern(i.currency); o.type = intern(i.type); o.day_count = intern(i.day_count);
    o.calendar = intern(i.calendar); o.par_product = intern(i.par_product); o.tenor = intern(i.tenor);
    return o;
  }
  BondConv owned(const BondConv& b) {
    BondConv o = b;
    o.id = intern(b.id); o.currency = intern(b.currency); o.calendar = intern(b.calendar); o.day_count = intern(b.day_count);
    o.frequency = intern(b.frequency); o.stub_discount = intern(b.stub_discount);
    return o;
  }
  CurrencyConv owned(const CurrencyConv& c) {
    CurrencyConv o = c;
    o.code = intern(c.code); o.name = intern(c.name); o.settlement_calendar = intern(c.settlement_calendar);
    o.discount_index = intern(c.discount_index); o.default_swap_product = intern(c.default_swap_product);
    o.repo_day_count = intern(c.repo_day_count);
    return o;
  }
  CreditConv owned(const CreditConv& c) {
    CreditConv o = c;
    o.id = intern(c.id); o.currency = intern(c.currency); o.calendar = intern(c.calendar); o.day_count = intern(c.day_count);
    o.frequency = intern(c.frequency); o.roll = intern(c.roll);
    return o;
  }
  BondFutureConv owned(const BondFutureConv& f) {
    BondFutureConv o = f;
    o.id = intern(f.id); o.currency = intern(f.currency); o.exchange_calendar = intern(f.exchange_calendar);
    o.deliverable_convention = intern(f.deliverable_convention); o.repo_day_count = intern(f.repo_day_count); o.delivery = intern(f.delivery);
    return o;
  }
  FxPairConv owned(const FxPairConv& f) {
    FxPairConv o = f;
    o.id = intern(f.id); o.base = intern(f.base); o.quote = intern(f.quote); o.calendar = intern(f.calendar);
    o.premium_currency = intern(f.premium_currency); o.delta_convention = intern(f.delta_convention); o.atm_convention = intern(f.atm_convention);
    o.xccy_product = intern(f.xccy_product); o.forward_product = intern(f.forward_product);
    return o;
  }
  FixingSourceConv owned(const FixingSourceConv& s) {
    FixingSourceConv o = s;
    o.id = intern(s.id); o.provider = intern(s.provider); o.series = intern(s.series); o.start = intern(s.start); o.granularity = intern(s.granularity);
    return o;
  }
  InflationIndexConv owned(const InflationIndexConv& x) {
    InflationIndexConv o = x;
    o.id = intern(x.id); o.label = intern(x.label); o.currency = intern(x.currency); o.calendar = intern(x.calendar);
    o.interpolation = intern(x.interpolation); o.frequency = intern(x.frequency);
    return o;
  }
  CbScheduleConv owned(const CbScheduleConv& c, std::size_t n_meetings) {
    CbScheduleConv o = c;
    o.currency = intern(c.currency); o.bank = intern(c.bank); o.source = intern(c.source); o.as_of = intern(c.as_of);
    o.begin = 0; o.count = n_meetings;
    return o;
  }
  CalendarConv owned(const CalendarConv& c, std::size_t n_rules, std::size_t n_joins) {
    CalendarConv o = c;
    o.id = intern(c.id); o.name = intern(c.name); o.observance = intern(c.observance);
    o.rule_begin = 0; o.rule_count = n_rules; o.join_begin = 0; o.join_count = n_joins;
    return o;
  }
  HolidayRule owned(const HolidayRule& h) {
    HolidayRule o = h;
    o.kind = intern(h.kind); o.observance = intern(h.observance);
    return o;
  }
  void clear_rows() noexcept {  // caller holds the unique lock
    products_.clear(); indices_.clear(); bonds_.clear(); currencies_.clear();
    calendars_.clear(); cal_rules_.clear(); cal_joins_.clear();
    credit_.clear(); bond_futures_.clear(); fx_pairs_.clear(); cb_schedules_.clear(); cb_meetings_.clear();
    fixing_sources_.clear(); inflation_.clear();
  }
  // Will `id` resolve as an index once `b` is committed? (Caller holds the unique lock: scans without locking.)
  bool index_known_after(const OverlayBatch& b, std::string_view id) const {
    for (const auto& i : b.indices) if (i.id == id) return true;
    if (!b.clear_first)
      for (const auto& i : indices_) if (i.id == id) return true;
    for (const auto& i : kIndices) if (i.id == id) return true;
    return false;
  }

  mutable std::shared_mutex mu_;
  std::deque<std::string> pool_;
  std::vector<ProductConv> products_;
  std::vector<IndexConv> indices_;
  std::vector<BondConv> bonds_;
  std::vector<CurrencyConv> currencies_;
  std::vector<CalendarConv> calendars_;
  std::vector<std::vector<HolidayRule>> cal_rules_;
  std::vector<std::vector<std::string_view>> cal_joins_;
  std::vector<CreditConv> credit_;
  std::vector<BondFutureConv> bond_futures_;
  std::vector<FxPairConv> fx_pairs_;
  std::vector<CbScheduleConv> cb_schedules_;
  std::vector<std::vector<long>> cb_meetings_;
  std::vector<FixingSourceConv> fixing_sources_;
  std::vector<InflationIndexConv> inflation_;
  std::atomic<int> overlay_n_{0};
  std::atomic<unsigned long> gen_{0};
};

// ---- the lookup API every consumer uses -------------------------------------------------------------
inline std::optional<ProductConv>  product(std::string_view id)   { return Registry::instance().product(id); }
inline std::optional<IndexConv>    index(std::string_view id)     { return Registry::instance().index(id); }
inline std::optional<BondConv>     bond(std::string_view id)      { return Registry::instance().bond(id); }
inline std::optional<CurrencyConv> currency(std::string_view id)  { return Registry::instance().currency(id); }
inline std::optional<CalendarView> calendar(std::string_view id)  { return Registry::instance().calendar(id); }
inline std::optional<CreditConv>     credit_product(std::string_view id) { return Registry::instance().credit_product(id); }
inline std::optional<BondFutureConv> bond_future(std::string_view id)    { return Registry::instance().bond_future(id); }
inline std::optional<FxPairConv>     fx_pair(std::string_view id)        { return Registry::instance().fx_pair(id); }
inline std::optional<std::vector<long>> cb_meetings(std::string_view ccy) { return Registry::instance().cb_meetings(ccy); }
inline std::optional<FixingSourceConv>   fixing_source(std::string_view index_id) { return Registry::instance().fixing_source(index_id); }
inline std::optional<InflationIndexConv> inflation_index(std::string_view id)     { return Registry::instance().inflation_index(id); }

[[noreturn]] inline void unknown(const char* kind, std::string_view id) {
  throw std::invalid_argument(std::string("conventions DB: unknown ") + kind + " '" + std::string(id) +
                              "' (add it to conventions/conventions.json or via the `conventions` verb)");
}
inline ProductConv  require_product(std::string_view id)  { if (id.empty()) unknown("product (empty id)", id);  if (auto r = product(id))  return *r; unknown("product", id); }
inline IndexConv    require_index(std::string_view id)    { if (id.empty()) unknown("index (empty id)", id);    if (auto r = index(id))    return *r; unknown("index", id); }
inline BondConv     require_bond(std::string_view id)     { if (id.empty()) unknown("bond (empty id)", id);     if (auto r = bond(id))     return *r; unknown("bond convention", id); }
inline CurrencyConv require_currency(std::string_view id) { if (id.empty()) unknown("currency (empty code)", id); if (auto r = currency(id)) return *r; unknown("currency", id); }
inline CreditConv     require_credit_product(std::string_view id) { if (id.empty()) unknown("cds product (empty id)", id); if (auto r = credit_product(id)) return *r; unknown("cds product", id); }
inline BondFutureConv require_bond_future(std::string_view id)    { if (id.empty()) unknown("bond-future contract (empty id)", id); if (auto r = bond_future(id)) return *r; unknown("bond-future contract", id); }
inline FxPairConv     require_fx_pair(std::string_view id)        { if (id.empty()) unknown("fx pair (empty id)", id); if (auto r = fx_pair(id)) return *r; unknown("fx pair", id); }
inline std::vector<long> require_cb_meetings(std::string_view ccy) { if (ccy.empty()) unknown("cb schedule (empty currency)", ccy); if (auto r = cb_meetings(ccy)) return *r; unknown("cb schedule for currency", ccy); }
inline FixingSourceConv   require_fixing_source(std::string_view index_id) { if (index_id.empty()) unknown("fixing source (empty index id)", index_id); if (auto r = fixing_source(index_id)) return *r; unknown("fixing source for index", index_id); }
inline InflationIndexConv require_inflation_index(std::string_view id)     { if (id.empty()) unknown("inflation index (empty id)", id); if (auto r = inflation_index(id)) return *r; unknown("inflation index", id); }
inline CalendarView require_calendar(std::string_view id) { if (id.empty()) unknown("calendar (empty id; use 'NONE' for weekends-only)", id); if (auto r = calendar(id)) return *r; unknown("calendar", id); }

// A DB field that a builder cannot proceed without. Never defaulted (P2).
inline std::string_view require_field(std::string_view v, const char* what, std::string_view row_id) {
  if (v.empty())
    throw std::invalid_argument(std::string("conventions DB: '") + std::string(row_id) + "' has no " + what +
                                " — add it to conventions/conventions.json");
  return v;
}
inline int require_lag(int v, const char* what, std::string_view row_id) {
  if (v < 0)
    throw std::invalid_argument(std::string("conventions DB: '") + std::string(row_id) + "' has no " + what +
                                " — add it to conventions/conventions.json");
  return v;
}

}  // namespace swaps::conventions

#endif  // SWAPS_CONVENTIONS_DB_HPP
