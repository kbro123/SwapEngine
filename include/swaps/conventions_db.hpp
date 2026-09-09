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

  // ---- runtime overlay (add or override). Strings are interned so the string_views outlive the caller. ----
  void add_product(const ProductConv& p) {
    std::unique_lock lk(mu_);
    ProductConv o = p;
    o.id = intern(p.id); o.type = intern(p.type); o.currency = intern(p.currency); o.calendar = intern(p.calendar);
    o.bdc = intern(p.bdc); o.frequency = intern(p.frequency); o.discount_index = intern(p.discount_index);
    o.pair = intern(p.pair); o.base_currency = intern(p.base_currency);
    o.fixed = own(p.fixed); o.floating = own(p.floating); o.other = own(p.other);
    products_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_index(const IndexConv& i) {
    std::unique_lock lk(mu_);
    IndexConv o = i;
    o.id = intern(i.id); o.currency = intern(i.currency); o.type = intern(i.type); o.day_count = intern(i.day_count);
    o.calendar = intern(i.calendar); o.par_product = intern(i.par_product); o.tenor = intern(i.tenor);
    indices_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_bond(const BondConv& b) {
    std::unique_lock lk(mu_);
    BondConv o = b;
    o.id = intern(b.id); o.currency = intern(b.currency); o.calendar = intern(b.calendar); o.day_count = intern(b.day_count);
    o.frequency = intern(b.frequency); o.stub_discount = intern(b.stub_discount);
    bonds_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_currency(const CurrencyConv& c) {
    std::unique_lock lk(mu_);
    CurrencyConv o = c;
    o.code = intern(c.code); o.name = intern(c.name); o.settlement_calendar = intern(c.settlement_calendar);
    o.discount_index = intern(c.discount_index); o.default_swap_product = intern(c.default_swap_product);
    currencies_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_calendar(const CalendarConv& c, const std::vector<HolidayRule>& rules,
                    const std::vector<std::string_view>& joins) {
    std::unique_lock lk(mu_);
    CalendarConv o = c;
    o.id = intern(c.id); o.name = intern(c.name); o.observance = intern(c.observance);
    o.rule_begin = 0; o.rule_count = rules.size(); o.join_begin = 0; o.join_count = joins.size();
    std::vector<HolidayRule> rs;
    for (const auto& r : rules) { HolidayRule x = r; x.kind = intern(r.kind); x.observance = intern(r.observance); rs.push_back(x); }
    std::vector<std::string_view> js;
    for (const auto& j : joins) js.push_back(intern(j));
    calendars_.push_back(o);
    cal_rules_.push_back(std::move(rs));
    cal_joins_.push_back(std::move(js));
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  // Drop every runtime entry (tests; a session reset). The baked defaults are untouched.
  void clear_overlay() {
    std::unique_lock lk(mu_);
    products_.clear(); indices_.clear(); bonds_.clear(); currencies_.clear();
    calendars_.clear(); cal_rules_.clear(); cal_joins_.clear();
    overlay_n_.store(0, std::memory_order_release);
  }

  // ---- listings (the `list_conventions` verb): baked ids and overlay ids per family ----------------------
  struct Listing { std::vector<std::string> baked, overlay; };
  Listing list_products() const  { Listing L; for (const auto& p : kProducts) L.baked.emplace_back(p.id);  { std::shared_lock lk(mu_); for (const auto& p : products_)  L.overlay.emplace_back(p.id); } return L; }
  Listing list_indices() const   { Listing L; for (const auto& i : kIndices)  L.baked.emplace_back(i.id);  { std::shared_lock lk(mu_); for (const auto& i : indices_)   L.overlay.emplace_back(i.id); } return L; }
  Listing list_bonds() const     { Listing L; for (const auto& b : kBonds)    L.baked.emplace_back(b.id);  { std::shared_lock lk(mu_); for (const auto& b : bonds_)     L.overlay.emplace_back(b.id); } return L; }
  Listing list_currencies() const{ Listing L; for (const auto& c : kCurrencies) L.baked.emplace_back(c.code); { std::shared_lock lk(mu_); for (const auto& c : currencies_) L.overlay.emplace_back(c.code); } return L; }
  Listing list_calendars() const { Listing L; for (const auto& c : kCalendars) L.baked.emplace_back(c.id);  { std::shared_lock lk(mu_); for (const auto& c : calendars_) L.overlay.emplace_back(c.id); } return L; }
  int overlay_size() const { return overlay_n_.load(std::memory_order_acquire); }

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
  LegConv own(const LegConv& l) {
    LegConv o = l;
    o.index = intern(l.index); o.day_count = intern(l.day_count); o.frequency = intern(l.frequency); o.compounding = intern(l.compounding);
    return o;
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
  std::atomic<int> overlay_n_{0};
};

// ---- the lookup API every consumer uses -------------------------------------------------------------
inline std::optional<ProductConv>  product(std::string_view id)   { return Registry::instance().product(id); }
inline std::optional<IndexConv>    index(std::string_view id)     { return Registry::instance().index(id); }
inline std::optional<BondConv>     bond(std::string_view id)      { return Registry::instance().bond(id); }
inline std::optional<CurrencyConv> currency(std::string_view id)  { return Registry::instance().currency(id); }
inline std::optional<CalendarView> calendar(std::string_view id)  { return Registry::instance().calendar(id); }

[[noreturn]] inline void unknown(const char* kind, std::string_view id) {
  throw std::invalid_argument(std::string("conventions DB: unknown ") + kind + " '" + std::string(id) +
                              "' (add it to conventions/conventions.json or via the `conventions` verb)");
}
inline ProductConv  require_product(std::string_view id)  { if (id.empty()) unknown("product (empty id)", id);  if (auto r = product(id))  return *r; unknown("product", id); }
inline IndexConv    require_index(std::string_view id)    { if (id.empty()) unknown("index (empty id)", id);    if (auto r = index(id))    return *r; unknown("index", id); }
inline BondConv     require_bond(std::string_view id)     { if (id.empty()) unknown("bond (empty id)", id);     if (auto r = bond(id))     return *r; unknown("bond convention", id); }
inline CurrencyConv require_currency(std::string_view id) { if (id.empty()) unknown("currency (empty code)", id); if (auto r = currency(id)) return *r; unknown("currency", id); }
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
