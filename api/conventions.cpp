// Conventions-registry verbs (see include/swaps/api/conventions.hpp). Parses conventions.json-shaped rows into
// the registry's structs (the same field mapping tools/gen_conventions_hpp.py uses for the baked arrays) and
// hands them to swaps::conventions::Registry, which interns the strings. QuantLib-free.
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/conventions.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cvd = swaps::conventions;

namespace {

std::string_view js(const json::object& o, const char* k) {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return {};
  const auto& s = o.at(k).as_string();
  return std::string_view(s.data(), s.size());
}
int ji(const json::object& o, const char* k, int d) {
  return o.contains(k) && !o.at(k).is_null() ? static_cast<int>(o.at(k).to_number<double>()) : d;
}
bool jb(const json::object& o, const char* k) { return o.contains(k) && o.at(k).is_bool() && o.at(k).as_bool(); }
const json::object* jobj(const json::object& o, const char* k) {
  return o.contains(k) && o.at(k).is_object() ? &o.at(k).as_object() : nullptr;
}

// Mirrors gen_conventions_hpp.py::leg — fixing_lag -1 = unset (the builders require it where it matters).
cvd::LegConv leg_of(const json::object* l) {
  cvd::LegConv L{};
  if (!l) return L;
  L.index = js(*l, "index"); L.day_count = js(*l, "day_count"); L.frequency = js(*l, "frequency");
  L.compounding = js(*l, "compounding"); L.fixing_lag = ji(*l, "fixing_lag", -1);
  L.carries_spread = jb(*l, "carries_spread"); L.notional_resets = jb(*l, "notional_resets"); L.flat = jb(*l, "flat");
  return L;
}

void add_product(cvd::Registry& R, std::string_view id, const json::object& p) {
  cvd::ProductConv c{};
  c.id = id; c.type = js(p, "type"); c.currency = js(p, "currency"); c.calendar = js(p, "calendar"); c.bdc = js(p, "bdc");
  c.frequency = js(p, "frequency"); c.discount_index = js(p, "discount_index"); c.pair = js(p, "pair");
  c.base_currency = js(p, "base_currency");
  c.spot_lag = ji(p, "spot_lag", -1); c.payment_lag = ji(p, "payment_lag", -1);
  c.fixed = leg_of(jobj(p, "fixed_leg"));
  const json::object* fl = jobj(p, "float_leg"); if (!fl) fl = jobj(p, "spread_leg"); if (!fl) fl = jobj(p, "usd_leg");
  const json::object* ol = jobj(p, "flat_leg"); if (!ol) ol = jobj(p, "eur_leg");
  c.floating = leg_of(fl); c.other = leg_of(ol);
  if (c.type.empty()) throw std::invalid_argument("conventions: product '" + std::string(id) + "' needs a 'type'");
  R.add_product(c);
}
void add_index(cvd::Registry& R, std::string_view id, const json::object& i) {
  cvd::IndexConv c{};
  c.id = id; c.currency = js(i, "currency"); c.type = js(i, "type"); c.day_count = js(i, "day_count");
  c.calendar = js(i, "calendar"); c.par_product = js(i, "par_product"); c.tenor = js(i, "tenor");
  c.fixing_lag = ji(i, "fixing_lag", -1); c.publication_lag = ji(i, "publication_lag", -1);
  if (c.currency.empty() || c.type.empty() || c.day_count.empty() || c.calendar.empty())
    throw std::invalid_argument("conventions: index '" + std::string(id) + "' needs currency/type/day_count/calendar");
  R.add_index(c);
}
void add_bond(cvd::Registry& R, std::string_view id, const json::object& b) {
  cvd::BondConv c{};
  c.id = id; c.currency = js(b, "currency"); c.calendar = js(b, "calendar"); c.day_count = js(b, "day_count");
  c.frequency = js(b, "frequency"); c.stub_discount = js(b, "stub_discount"); c.settle_lag = ji(b, "settle_lag", -1);
  c.final_period_simple = jb(b, "final_period_simple");
  R.add_bond(c);
}
void add_currency(cvd::Registry& R, std::string_view code, const json::object& c) {
  cvd::CurrencyConv x{};
  x.code = code; x.name = js(c, "name"); x.settlement_calendar = js(c, "settlement_calendar");
  x.discount_index = js(c, "discount_index"); x.default_swap_product = js(c, "default_swap_product");
  x.repo_day_count = js(c, "repo_day_count"); x.minor_units = ji(c, "minor_units", -1);
  if (x.settlement_calendar.empty() || x.minor_units < 0)
    throw std::invalid_argument("conventions: currency '" + std::string(code) + "' needs settlement_calendar + minor_units");
  R.add_currency(x);
}
void add_calendar(cvd::Registry& R, std::string_view id, const json::object& c) {
  cvd::CalendarConv row{};
  row.id = id; row.name = js(c, "name"); row.observance = js(c, "observance");
  int mask = 0;
  if (c.contains("weekend") && c.at("weekend").is_array())
    for (const auto& w : c.at("weekend").as_array()) mask |= 1 << static_cast<int>(w.to_number<double>());
  else
    throw std::invalid_argument("conventions: calendar '" + std::string(id) + "' needs 'weekend' (e.g. [5,6])");
  row.weekend_mask = mask; row.sandwich = jb(c, "sandwich");
  std::vector<cvd::HolidayRule> rules;
  if (c.contains("holidays") && c.at("holidays").is_array())
    for (const auto& hv : c.at("holidays").as_array()) {
      const auto& h = hv.as_object();
      cvd::HolidayRule r{};
      r.kind = js(h, "rule"); r.month = ji(h, "month", 0); r.day = ji(h, "day", 0); r.weekday = ji(h, "weekday", -1);
      r.n = ji(h, "n", 0); r.days = ji(h, "days", 0); r.from_year = ji(h, "from_year", 0); r.to_year = ji(h, "to_year", 0);
      r.observance = js(h, "observance"); r.except_first_friday = jb(h, "except_first_friday");
      if (r.kind.empty()) throw std::invalid_argument("conventions: calendar '" + std::string(id) + "' holiday needs 'rule'");
      rules.push_back(r);
    }
  std::vector<std::string_view> joins;
  if (c.contains("join") && c.at("join").is_array())
    for (const auto& j : c.at("join").as_array()) joins.emplace_back(j.as_string().data(), j.as_string().size());
  if (rules.empty() && joins.empty() && !c.contains("holidays"))
    throw std::invalid_argument("conventions: calendar '" + std::string(id) + "' needs 'holidays' ([] = none) or 'join'");
  R.add_calendar(row, rules, joins);
}

void add_credit(cvd::Registry& R, std::string_view id, const json::object& c) {
  cvd::CreditConv x{};
  x.id = id; x.currency = js(c, "currency"); x.calendar = js(c, "calendar"); x.day_count = js(c, "day_count");
  x.frequency = js(c, "frequency"); x.roll = js(c, "roll");
  x.recovery_default = c.contains("recovery_default") ? c.at("recovery_default").to_number<double>() : -1.0;
  x.settlement_lag = ji(c, "settlement_lag", -1); x.protection_steps = ji(c, "protection_steps", -1);
  if (x.currency.empty() || x.day_count.empty() || x.frequency.empty() || x.recovery_default < 0 || x.protection_steps < 1)
    throw std::invalid_argument("conventions: cds product '" + std::string(id) + "' needs currency/day_count/frequency/recovery_default/protection_steps");
  R.add_credit_product(x);
}
void add_bond_future(cvd::Registry& R, std::string_view id, const json::object& f) {
  cvd::BondFutureConv x{};
  x.id = id; x.currency = js(f, "currency"); x.exchange_calendar = js(f, "exchange_calendar");
  x.deliverable_convention = js(f, "deliverable_convention"); x.repo_day_count = js(f, "repo_day_count"); x.delivery = js(f, "delivery");
  x.notional_coupon = f.contains("notional_coupon") ? f.at("notional_coupon").to_number<double>() : -1.0;
  x.basket_min_years = f.contains("basket_min_years") ? f.at("basket_min_years").to_number<double>() : 0.0;
  x.basket_max_years = f.contains("basket_max_years") ? f.at("basket_max_years").to_number<double>() : 0.0;
  x.maturity_rounding_months = ji(f, "maturity_rounding_months", -1);
  if (x.deliverable_convention.empty() || x.notional_coupon < 0 || x.maturity_rounding_months < 1 || x.repo_day_count.empty())
    throw std::invalid_argument("conventions: bond future '" + std::string(id) + "' needs deliverable_convention/notional_coupon/maturity_rounding_months/repo_day_count");
  R.add_bond_future(x);
}
void add_fx_pair(cvd::Registry& R, std::string_view id, const json::object& f) {
  cvd::FxPairConv x{};
  x.id = id; x.base = js(f, "base"); x.quote = js(f, "quote"); x.calendar = js(f, "calendar");
  x.premium_currency = js(f, "premium_currency"); x.delta_convention = js(f, "delta_convention"); x.atm_convention = js(f, "atm_convention");
  x.xccy_product = js(f, "xccy_product"); x.forward_product = js(f, "forward_product"); x.spot_lag = ji(f, "spot_lag", -1);
  if (f.contains("smile_pillars") && f.at("smile_pillars").is_array() && !f.at("smile_pillars").as_array().empty()) {
    const auto& a = f.at("smile_pillars").as_array();
    x.smile_pillar_lo = a.front().to_number<double>(); x.smile_pillar_hi = a.back().to_number<double>();
  }
  if (x.base.empty() || x.quote.empty() || x.calendar.empty() || x.spot_lag < 0 || x.delta_convention.empty() || x.atm_convention.empty())
    throw std::invalid_argument("conventions: fx pair '" + std::string(id) + "' needs base/quote/calendar/spot_lag/delta_convention/atm_convention");
  if (std::string(x.base) + std::string(x.quote) != std::string(id))
    throw std::invalid_argument("conventions: fx pair id '" + std::string(id) + "' must equal base+quote");
  R.add_fx_pair(x);
}
long serial_of_iso(std::string_view iso) {  // YYYY-MM-DD -> Unix-day serial (days since 1970-01-01)
  if (iso.size() != 10) throw std::invalid_argument("conventions: bad ISO date '" + std::string(iso) + "'");
  const int y = std::stoi(std::string(iso.substr(0, 4))), m = std::stoi(std::string(iso.substr(5, 2))), d = std::stoi(std::string(iso.substr(8, 2)));
  // days-from-civil (Howard Hinnant)
  const int yy = y - (m <= 2); const int era = (yy >= 0 ? yy : yy - 399) / 400; const unsigned yoe = static_cast<unsigned>(yy - era * 400);
  const unsigned doy = (153u * static_cast<unsigned>(m + (m > 2 ? -3 : 9)) + 2u) / 5u + static_cast<unsigned>(d) - 1u;
  const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
  return static_cast<long>(era) * 146097L + static_cast<long>(doe) - 719468L;
}
void add_cb_schedule(cvd::Registry& R, std::string_view ccy, const json::object& c) {
  cvd::CbScheduleConv x{};
  x.currency = ccy; x.bank = js(c, "bank"); x.source = js(c, "source"); x.as_of = js(c, "as_of");
  std::vector<long> m;
  if (!c.contains("meetings") || !c.at("meetings").is_array())
    throw std::invalid_argument("conventions: cb schedule '" + std::string(ccy) + "' needs a 'meetings' array of ISO dates");
  for (const auto& v : c.at("meetings").as_array()) m.push_back(serial_of_iso(std::string_view(v.as_string().data(), v.as_string().size())));
  for (std::size_t i = 1; i < m.size(); ++i)
    if (m[i] <= m[i - 1]) throw std::invalid_argument("conventions: cb schedule '" + std::string(ccy) + "' meetings must be strictly ascending");
  R.add_cb_schedule(x, std::move(m));
}

json::array names_of(const std::vector<std::string>& v) {
  json::array a;
  for (const auto& s : v) a.emplace_back(s);
  return a;
}
json::object listing(const cvd::Registry::Listing& L) {
  return json::object{{"baked", names_of(L.baked)}, {"overlay", names_of(L.overlay)}};
}

}  // namespace

std::string conventions_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("conventions") ? top.at("conventions").as_object() : top;
  auto& R = cvd::Registry::instance();
  json::object added;
  if (jb(o, "clear_overlay")) R.clear_overlay();
  struct Fam { const char* key; void (*add)(cvd::Registry&, std::string_view, const json::object&); };
  // Order matters only for the caller's readability; lookups are by id at build time, not at add time.
  const Fam fams[] = {{"currencies", add_currency}, {"calendars", add_calendar}, {"indices", add_index},
                      {"products", add_product}, {"bonds", add_bond}, {"bond_futures", add_bond_future},
                      {"fx_pairs", add_fx_pair}, {"cb_schedules", add_cb_schedule}};
  // credit.cds_products nests one level deeper than the other families (mirrors the JSON file).
  if (const json::object* cr = jobj(o, "credit"))
    if (const json::object* cp = jobj(*cr, "cds_products")) {
      json::array ids;
      for (const auto& kv : *cp) {
        if (!kv.value().is_object()) throw std::invalid_argument("conventions: credit.cds_products entries must be objects");
        add_credit(R, std::string_view(kv.key().data(), kv.key().size()), kv.value().as_object());
        ids.emplace_back(kv.key());
      }
      added["cds_products"] = ids;
    }
  for (const auto& f : fams) {
    const json::object* fo = jobj(o, f.key);
    if (!fo) continue;
    json::array ids;
    for (const auto& kv : *fo) {
      if (!kv.value().is_object()) throw std::invalid_argument(std::string("conventions: ") + f.key + " entries must be objects");
      f.add(R, std::string_view(kv.key().data(), kv.key().size()), kv.value().as_object());
      ids.emplace_back(kv.key());
    }
    added[f.key] = ids;
  }
  json::object out;
  out["conventions"] = json::object{{"added", added}, {"overlay_size", R.overlay_size()}};
  return json::serialize(out);
}

std::string list_conventions_json(const std::string& /*request*/) {
  const auto& R = cvd::Registry::instance();
  json::object out;
  out["conventions"] = json::object{
      {"currencies", listing(R.list_currencies())}, {"calendars", listing(R.list_calendars())},
      {"indices", listing(R.list_indices())},       {"products", listing(R.list_products())},
      {"bonds", listing(R.list_bonds())},           {"cds_products", listing(R.list_credit_products())},
      {"bond_futures", listing(R.list_bond_futures())}, {"fx_pairs", listing(R.list_fx_pairs())},
      {"cb_schedules", listing(R.list_cb_schedules())}, {"overlay_size", R.overlay_size()}};
  return json::serialize(out);
}

}  // namespace swaps::api
