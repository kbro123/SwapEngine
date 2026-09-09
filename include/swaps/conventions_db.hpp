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
    o.repo_day_count = intern(c.repo_day_count);
    currencies_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_credit_product(const CreditConv& c) {
    std::unique_lock lk(mu_);
    CreditConv o = c;
    o.id = intern(c.id); o.currency = intern(c.currency); o.calendar = intern(c.calendar); o.day_count = intern(c.day_count);
    o.frequency = intern(c.frequency); o.roll = intern(c.roll);
    credit_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_bond_future(const BondFutureConv& f) {
    std::unique_lock lk(mu_);
    BondFutureConv o = f;
    o.id = intern(f.id); o.currency = intern(f.currency); o.exchange_calendar = intern(f.exchange_calendar);
    o.deliverable_convention = intern(f.deliverable_convention); o.repo_day_count = intern(f.repo_day_count); o.delivery = intern(f.delivery);
    bond_futures_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_fx_pair(const FxPairConv& f) {
    std::unique_lock lk(mu_);
    FxPairConv o = f;
    o.id = intern(f.id); o.base = intern(f.base); o.quote = intern(f.quote); o.calendar = intern(f.calendar);
    o.premium_currency = intern(f.premium_currency); o.delta_convention = intern(f.delta_convention); o.atm_convention = intern(f.atm_convention);
    o.xccy_product = intern(f.xccy_product); o.forward_product = intern(f.forward_product);
    fx_pairs_.push_back(o);
    overlay_n_.fetch_add(1, std::memory_order_release);
  }
  void add_cb_schedule(const CbScheduleConv& c, std::vector<long> meetings) {
    std::unique_lock lk(mu_);
    CbScheduleConv o = c;
    o.currency = intern(c.currency); o.bank = intern(c.bank); o.source = intern(c.source); o.as_of = intern(c.as_of);
    o.begin = 0; o.count = meetings.size();
    cb_schedules_.push_back(o); cb_meetings_.push_back(std::move(meetings));
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
    credit_.clear(); bond_futures_.clear(); fx_pairs_.clear(); cb_schedules_.clear(); cb_meetings_.clear();
    overlay_n_.store(0, std::memory_order_release);
  }

  // ---- listings (the `list_conventions` verb): baked ids and overlay ids per family ----------------------
  struct Listing { std::vector<std::string> baked, overlay; };
  Listing list_products() const  { Listing L; for (const auto& p : kProducts) L.baked.emplace_back(p.id);  { std::shared_lock lk(mu_); for (const auto& p : products_)  L.overlay.emplace_back(p.id); } return L; }
  Listing list_indices() const   { Listing L; for (const auto& i : kIndices)  L.baked.emplace_back(i.id);  { std::shared_lock lk(mu_); for (const auto& i : indices_)   L.overlay.emplace_back(i.id); } return L; }
  Listing list_bonds() const     { Listing L; for (const auto& b : kBonds)    L.baked.emplace_back(b.id);  { std::shared_lock lk(mu_); for (const auto& b : bonds_)     L.overlay.emplace_back(b.id); } return L; }
  Listing list_currencies() const{ Listing L; for (const auto& c : kCurrencies) L.baked.emplace_back(c.code); { std::shared_lock lk(mu_); for (const auto& c : currencies_) L.overlay.emplace_back(c.code); } return L; }
  Listing list_calendars() const { Listing L; for (const auto& c : kCalendars) L.baked.emplace_back(c.id);  { std::shared_lock lk(mu_); for (const auto& c : calendars_) L.overlay.emplace_back(c.id); } return L; }
  Listing list_credit_products() const { Listing L; for (const auto& c : kCredit) L.baked.emplace_back(c.id); { std::shared_lock lk(mu_); for (const auto& c : credit_) L.overlay.emplace_back(c.id); } return L; }
  Listing list_bond_futures() const { Listing L; for (const auto& f : kBondFutures) L.baked.emplace_back(f.id); { std::shared_lock lk(mu_); for (const auto& f : bond_futures_) L.overlay.emplace_back(f.id); } return L; }
  Listing list_fx_pairs() const { Listing L; for (const auto& f : kFxPairs) L.baked.emplace_back(f.id); { std::shared_lock lk(mu_); for (const auto& f : fx_pairs_) L.overlay.emplace_back(f.id); } return L; }
  Listing list_cb_schedules() const { Listing L; for (const auto& c : kCbSchedules) L.baked.emplace_back(c.currency); { std::shared_lock lk(mu_); for (const auto& c : cb_schedules_) L.overlay.emplace_back(c.currency); } return L; }
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
  std::vector<CreditConv> credit_;
  std::vector<BondFutureConv> bond_futures_;
  std::vector<FxPairConv> fx_pairs_;
  std::vector<CbScheduleConv> cb_schedules_;
  std::vector<std::vector<long>> cb_meetings_;
  std::atomic<int> overlay_n_{0};
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
