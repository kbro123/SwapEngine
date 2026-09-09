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
  x.minor_units = ji(c, "minor_units", -1);
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
  row.weekend_mask = mask;
  std::vector<cvd::HolidayRule> rules;
  if (c.contains("holidays") && c.at("holidays").is_array())
    for (const auto& hv : c.at("holidays").as_array()) {
      const auto& h = hv.as_object();
      cvd::HolidayRule r{};
      r.kind = js(h, "rule"); r.month = ji(h, "month", 0); r.day = ji(h, "day", 0); r.weekday = ji(h, "weekday", -1);
      r.n = ji(h, "n", 0); r.days = ji(h, "days", 0); r.from_year = ji(h, "from_year", 0); r.to_year = ji(h, "to_year", 0);
      r.observance = js(h, "observance");
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
                      {"products", add_product}, {"bonds", add_bond}};
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
      {"bonds", listing(R.list_bonds())},           {"overlay_size", R.overlay_size()}};
  return json::serialize(out);
}

}  // namespace swaps::api
