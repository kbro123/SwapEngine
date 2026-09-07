// Implementation of the bundle compiler (include/swaps/api/compile.hpp) — the C++ analog of
// server/compile.py's compile_spec. Faithful, generic translation of a composer spec (curves + generic
// instrument rows + interpolation regions) into a cal::BundleProblem + streaming config, built on the
// QuantLib-free construction object model (include/swaps/build/*). Boost.JSON's implementation is compiled
// in bundle_api.cpp (this TU includes only the declarations).
#include "swaps/api/compile.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"        // bundle_to_json (reused for the "bundle" field)
#include "swaps/build/instruments.hpp"     // the whole build/ object model (schedule/conventions/observations)
#include "swaps/market/quote.hpp"          // CalibrationTarget -- the ONE quote->instrument hand-off POD

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace cvd = swaps::conventions;
namespace curve = swaps::curve;
namespace px = swaps::pricing;

namespace {

// ---- Boost.JSON field accessors (mirror bundle_api.cpp) ------------------------------------------
double get_d(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
bool get_b(const json::object& o, const char* k, bool d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).as_bool() : d;
}
std::string get_s(const json::object& o, const char* k, const char* d = "") {
  if (!o.contains(k) || o.at(k).is_null()) return d;
  const auto& v = o.at(k);
  if (v.is_string()) return std::string(v.as_string().c_str());
  return d;
}
// Present and non-null and (if a string) non-empty — Python's `x not in (None, "")`.
bool present(const json::object& o, const char* k) {
  if (!o.contains(k) || o.at(k).is_null()) return false;
  const auto& v = o.at(k);
  return !(v.is_string() && v.as_string().empty());
}
// A numeric field that may arrive as a number OR a numeric string (composer values). nullopt if absent.
std::optional<double> opt_num(const json::object& o, const char* k) {
  if (!o.contains(k) || o.at(k).is_null()) return std::nullopt;
  const auto& v = o.at(k);
  if (v.is_string()) {
    const std::string s(v.as_string().c_str());
    if (s.empty()) return std::nullopt;
    try { return std::stod(s); } catch (...) { return std::nullopt; }
  }
  return v.to_number<double>();
}
std::optional<std::string> opt_str(const json::object& o, const char* k) {
  if (!o.contains(k) || o.at(k).is_null()) return std::nullopt;
  const auto& v = o.at(k);
  if (!v.is_string()) return std::nullopt;
  std::string s(v.as_string().c_str());
  if (s.empty()) return std::nullopt;
  return s;
}
const json::object* obj_ptr(const json::object& o, const char* k) {
  if (o.contains(k) && o.at(k).is_object()) return &o.at(k).as_object();
  return nullptr;
}

std::string upper(std::string s) {
  for (char& c : s) c = char(std::toupper((unsigned char)c));
  return s;
}

// A stable string key for an instrument (Python `ins.get("id", j)`): its explicit id (string or int) or
// its positional index.
std::string ins_key(const json::object& o, std::size_t j) {
  if (o.contains("id") && !o.at("id").is_null()) {
    const auto& v = o.at("id");
    if (v.is_string()) return std::string(v.as_string().c_str());
    if (v.is_int64()) return std::to_string(v.as_int64());
    if (v.is_uint64()) return std::to_string(v.as_uint64());
    if (v.is_double()) return std::to_string((long long)v.as_double());
  }
  return "#" + std::to_string(j);
}

curve::Scheme scheme_from_policy(const std::string& p) {
  if (p == "Flat") return curve::Scheme::Flat;
  if (p == "Linear") return curve::Scheme::Linear;
  if (p == "NaturalCubic") return curve::Scheme::NaturalCubic;
  if (p == "Hermite") return curve::Scheme::Hermite;
  if (p == "MonotoneCubic") return curve::Scheme::MonotoneCubic;
  if (p == "BSpline") return curve::Scheme::BSpline;
  if (p == "Tension") return curve::Scheme::Tension;
  throw CompileError("unknown interpolation policy '" + p + "'.");
}
const char* policy_name(curve::Scheme s) {
  switch (s) {
    case curve::Scheme::Flat: return "Flat";
    case curve::Scheme::Linear: return "Linear";
    case curve::Scheme::NaturalCubic: return "NaturalCubic";
    case curve::Scheme::Hermite: return "Hermite";
    case curve::Scheme::MonotoneCubic: return "MonotoneCubic";
    case curve::Scheme::BSpline: return "BSpline";
    case curve::Scheme::Tension: return "Tension";
  }
  return "Hermite";
}

double to_decimal(double value, const std::string& unit) {
  if (unit == "pct") return value / 100.0;
  if (unit == "bp") return value / 1e4;
  return value;  // fx/px/raw: an outright level
}

// (target, upper, lower, decay) in the instrument's display unit (compile._quote_band). Falls back to the
// legacy `start_quote` target + nested `band` shape.
struct QuoteBand {
  double target = 0.0;
  std::optional<double> upper, lower, decay;
};
QuoteBand quote_band(const json::object& ins) {
  QuoteBand q;
  auto t = opt_num(ins, "target");
  q.target = t ? *t : opt_num(ins, "start_quote").value_or(0.0);
  q.upper = opt_num(ins, "upper");
  q.lower = opt_num(ins, "lower");
  q.decay = opt_num(ins, "decay");
  if (!q.upper && !q.lower) {
    if (const json::object* band = obj_ptr(ins, "band")) {
      q.upper = opt_num(*band, "upper");
      q.lower = opt_num(*band, "lower");
      if (!q.decay) q.decay = opt_num(*band, "decay");
    }
  }
  return q;
}
// Attach a bid/offer band to the instrument when it declares a REAL band (upper > lower), in rate units
// (compile._apply_band). A hard pin emits no band, so it stays exact. The wire shape is target + band —
// the target is NOT necessarily the band mid and the default decay is 1 (parity with server/compile.py) —
// so this maps onto the ONE hand-off POD (market::CalibrationTarget) and lands via Instrument::set_target,
// the single definition of how a target + band reaches an instrument (no third band semantics here).
void apply_band(cal::Instrument& obj, const json::object& ins, const std::string& unit) {
  const QuoteBand q = quote_band(ins);
  swaps::market::CalibrationTarget t{obj.market, 0.0, 0.0, 1.0};  // the already-set target, hard pin
  if (q.lower && q.upper) {
    const double lo = to_decimal(*q.lower, unit), hi = to_decimal(*q.upper, unit);
    if (hi > lo) {
      t.band_lower = lo;
      t.band_upper = hi;
      t.band_decay = q.decay ? *q.decay : 1.0;
    }
  }
  obj.set_target(t);
}

// resolve a token to (date, curve-time), throwing CompileError with context on a bad token.
struct RT { b::Date date; double t; };
RT resolve_time(const std::string& tok, const b::Date& vd, bool roll, const std::string& ctx) {
  try {
    const b::Date d = b::resolve(tok, vd, roll);
    return {d, b::curve_time(vd, d)};
  } catch (const std::exception& e) {
    throw CompileError(ctx + ": " + e.what());
  }
}

// ---- resolved instrument window (compile._resolve_instrument) ------------------------------------
struct Resolved {
  double t_start = 0.0, t_end = 0.0;
  bool has_sd = false, has_ed = false;
  b::Date sd, ed;
};
Resolved resolve_instrument(const json::object& ins, const b::Date& vd,
                            const std::optional<std::string>& start_override, const std::string& ctx) {
  const std::string type = get_s(ins, "type");
  const bool roll = !(type == "fut1m" || type == "fut3m");
  Resolved r;
  if (present(ins, "end")) {
    const RT e = resolve_time(get_s(ins, "end"), vd, roll, ctx);
    r.ed = e.date; r.t_end = e.t; r.has_ed = true;
  } else {
    r.t_end = get_d(ins, "time", 0.0);
  }
  if (get_s(ins, "quote_kind") == "Rate") {
    std::optional<std::string> start_tok = start_override ? start_override : opt_str(ins, "start");
    if (start_tok) {
      const RT s = resolve_time(*start_tok, vd, roll, ctx);
      r.sd = s.date; r.t_start = s.t; r.has_sd = true;
    } else {
      r.t_start = get_d(ins, "span_start", 0.0);
    }
  }
  return r;
}

// Chain a curve's QUOTED meetings into consecutive inter-meeting OIS spans (compile._chained_meeting_starts):
// {ins_key -> start token} where the token is the previous quoted meeting's ISO date ("0d" for the first).
std::unordered_map<std::string, std::string> chained_meeting_starts(const json::array& insts,
                                                                    const b::Date& vd) {
  struct Q { double te; std::string ed_iso, iid; };
  std::vector<Q> quoted;
  for (std::size_t j = 0; j < insts.size(); ++j) {
    const auto& ins = insts[j].as_object();
    if (get_s(ins, "type") == "meeting" && get_b(ins, "has_quote", false) && present(ins, "end")) {
      try {
        const b::Date ed = b::resolve(get_s(ins, "end"), vd, true);  // meeting roll = spot 'following'
        quoted.push_back({b::curve_time(vd, ed), b::iso(ed), ins_key(ins, j)});
      } catch (const std::exception&) { /* skip unresolvable, like Python */ }
    }
  }
  std::sort(quoted.begin(), quoted.end(), [](const Q& a, const Q& c) { return a.te < c.te; });
  std::unordered_map<std::string, std::string> out;
  std::optional<std::string> prev;
  for (const auto& q : quoted) {
    out[q.iid] = prev ? *prev : std::string("0d");
    prev = q.ed_iso;
  }
  return out;
}

// The sub-period brackets a Rate quote observes (compile._rate_obs).
px::RateObservation rate_obs(const json::object& ins, const Resolved& r, double a, double T,
                             const b::Date& vd, const std::string& dc, const std::string& cal,
                             const std::string& index) {
  const std::string accrual = get_s(ins, "accrual");
  if ((accrual == "averaged" || accrual == "compounded") && r.has_sd && r.has_ed) {
    if (!index.empty() && r.sd < vd)
      return b::scheduled_observation(vd, r.sd, r.ed, accrual, dc, cal, index);
    return b::observation(vd, r.sd, r.ed, accrual, get_d(ins, "realized", 0.0), dc, cal);
  }
  return b::plain_rate_obs(a, T);
}

// ---- curve layout (compile._curve_layout) --------------------------------------------------------
struct KnotItem {
  double t = 0.0;
  std::optional<std::string> region, knot_region;
};
struct Layout {
  bool classic = true;
  std::vector<double> front, back;            // classic
  std::vector<curve::CurveModule> modules;    // modular
  std::string policies;                        // display
};
Layout curve_layout(const json::object& c, const std::vector<KnotItem>& knot_items) {
  const std::string name = get_s(c, "name", "?");
  const json::array* regions = nullptr;
  if (c.contains("regions") && c.at("regions").is_array() && !c.at("regions").as_array().empty())
    regions = &c.at("regions").as_array();

  Layout L;
  if (!regions) {  // legacy: front(Flat) + back(Hermite) by knot_region (default "back")
    for (const auto& k : knot_items)
      (k.knot_region.value_or("back") == "front" ? L.front : L.back).push_back(k.t);
    std::sort(L.front.begin(), L.front.end());
    std::sort(L.back.begin(), L.back.end());
    L.classic = true;
    L.policies = "Flat+Hermite";
    return L;
  }

  std::vector<std::string> order_ids;
  std::unordered_map<std::string, curve::Scheme> policy;
  std::unordered_map<std::string, double> sigma;
  for (const auto& e : *regions) {
    const auto& ro = e.as_object();
    const std::string rid = get_s(ro, "id");
    order_ids.push_back(rid);
    policy[rid] = scheme_from_policy(get_s(ro, "policy", "Hermite"));
    sigma[rid] = std::max(0.0, opt_num(ro, "sigma").value_or(0.0));
  }
  std::unordered_map<std::string, std::vector<double>> buckets;
  for (const auto& rid : order_ids) buckets[rid];
  for (const auto& k : knot_items) {
    std::string rid = k.region.value_or("");
    if (!buckets.count(rid))  // unassigned -> legacy front/back to first/last region (default "front")
      rid = (k.knot_region.value_or("front") == "front") ? order_ids.front() : order_ids.back();
    buckets[rid].push_back(k.t);
  }
  std::vector<curve::CurveModule> modules;
  for (const auto& rid : order_ids) {
    auto& ks = buckets[rid];
    if (ks.empty()) continue;
    std::sort(ks.begin(), ks.end());
    curve::CurveModule m;
    m.scheme = policy[rid];
    m.knots = ks;
    if (m.scheme == curve::Scheme::Tension) m.sigma = sigma[rid];
    modules.push_back(std::move(m));
  }
  if (modules.empty())
    throw CompileError("Curve '" + name + "' has no knots — add at least one knot instrument.");
  // Fast path: a plain Flat->Hermite layout is exactly the classic shipped curve.
  if (modules.size() == 2 && modules[0].scheme == curve::Scheme::Flat &&
      modules[1].scheme == curve::Scheme::Hermite) {
    L.classic = true;
    L.front = modules[0].knots;
    L.back = modules[1].knots;
    L.policies = "Flat+Hermite";
    return L;
  }
  std::vector<double> flat;
  for (const auto& m : modules)
    for (double t : m.knots) flat.push_back(t);
  for (std::size_t i = 1; i < flat.size(); ++i)
    if (!(flat[i] > flat[i - 1]))
      throw CompileError("Curve '" + name + "': region knots must strictly increase across regions "
                         "(assign earlier-dated knots to earlier regions).");
  L.classic = false;
  L.policies.clear();
  for (std::size_t i = 0; i < modules.size(); ++i)
    L.policies += (i ? " \xE2\x86\x92 " : "") + std::string(policy_name(modules[i].scheme));
  L.modules = std::move(modules);
  return L;
}

// A turn's single date -> its overnight-accrual window (compile._turn_window). Returns (start, end) year
// fractions + the resolved dates.
struct TurnWindow { double start, end; b::Date sd, ed; };
TurnWindow turn_window(const json::object& ins, const b::Date& vd, const std::string& index,
                       const std::string& currency, const std::string& ctx) {
  const std::string tok = present(ins, "date") ? get_s(ins, "date") : get_s(ins, "end");
  b::Date raw;
  try {
    raw = b::resolve(tok, vd, false);  // the entered calendar date, unadjusted
  } catch (const std::exception& e) {
    throw CompileError(ctx + ": " + e.what());
  }
  std::string cal, dc = "ACT/360";
  if (!index.empty()) {
    if (auto ix = cvd::index(index)) {
      if (ix->type == "overnight") {
        cal = std::string(ix->calendar);
        if (!ix->day_count.empty()) dc = std::string(ix->day_count);
      }
    }
  }
  if (cal.empty()) cal = (upper(currency.empty() ? "USD" : currency) == "EUR") ? "EUR" : "USD";
  const b::Date start_date = b::adjust(cal, raw, "Preceding");  // last business day on/before
  const b::Date end_date = b::advance_bd(cal, start_date, 1);    // accrues to next business day
  const double start = b::curve_time(vd, start_date);
  const double tau = b::year_frac(dc, start_date, end_date);
  return {start, start + tau, start_date, end_date};
}

// ---- curve ordering (compile._order_curves) ------------------------------------------------------
std::vector<const json::object*> order_curves(const json::array& curves) {
  std::unordered_map<std::string, const json::object*> by_id;
  for (const auto& e : curves) by_id[get_s(e.as_object(), "id")] = &e.as_object();
  std::vector<const json::object*> order;
  std::unordered_set<std::string> seen, stack;
  std::function<void(const std::string&)> visit = [&](const std::string& cid) {
    if (seen.count(cid)) return;
    if (stack.count(cid)) throw CompileError("Spread curves form a cycle.");
    auto it = by_id.find(cid);
    if (it == by_id.end()) throw CompileError("A spread curve references a base that does not exist.");
    stack.insert(cid);
    const json::object& c = *it->second;
    if (get_s(c, "kind") == "spread" && present(c, "base")) visit(get_s(c, "base"));
    stack.erase(cid);
    seen.insert(cid);
    order.push_back(&c);
  };
  for (const auto& e : curves) visit(get_s(e.as_object(), "id"));
  return order;
}

// ---- display grid (compile._display_grid) --------------------------------------------------------
constexpr double ONE_DAY = 1.0 / 365.0;
std::vector<double> display_grid(double max_t, const std::vector<double>& knots) {
  std::set<double> g;
  auto r6 = [](double x) { return std::round(x * 1e6) / 1e6; };
  for (double t = 1.0 / 12.0; t <= std::min(2.0, max_t) + 1e-9; t += 1.0 / 12.0) g.insert(r6(t));
  for (double t = 2.25; t <= std::min(10.0, max_t) + 1e-9; t += 0.25) g.insert(r6(t));
  for (double t = 11.0; t <= max_t + 1e-9; t += 1.0) g.insert(r6(t));
  g.insert(r6(1.0 / 12.0));
  g.insert(r6(max_t));
  for (double k : knots) {
    g.insert(r6(k));
    g.insert(r6(k - ONE_DAY));
    g.insert(r6(k + ONE_DAY));
  }
  std::vector<double> out;
  for (double x : g)
    if (x > 0 && x <= max_t + ONE_DAY + 1e-9) out.push_back(x);
  return out;  // std::set already sorted ascending
}

}  // namespace

// ==================================================================================================
CompileResult compile_spec(const json::value& spec_v, const std::string& today_iso) {
  if (!spec_v.is_object()) throw CompileError("spec must be a JSON object.");
  const json::object& spec = spec_v.as_object();
  if (!spec.contains("curves") || !spec.at("curves").is_array() || spec.at("curves").as_array().empty())
    throw CompileError("Add at least one curve.");
  const json::array& curves = spec.at("curves").as_array();

  std::string vd_iso = get_s(spec, "value_date");
  if (vd_iso.empty()) vd_iso = today_iso;
  if (vd_iso.empty()) throw CompileError("value date must be YYYY-MM-DD");
  b::Date value_date;
  try {
    value_date = b::Date::from_iso(vd_iso);
  } catch (const std::exception&) {
    throw CompileError("value date must be YYYY-MM-DD, got '" + vd_iso + "'");
  }

  const std::vector<const json::object*> order = order_curves(curves);
  std::unordered_map<std::string, int> idx;
  for (std::size_t i = 0; i < order.size(); ++i) idx[get_s(*order[i], "id")] = int(i);

  int root_outright = -1;
  for (std::size_t i = 0; i < order.size(); ++i)
    if (get_s(*order[i], "kind") != "spread") { root_outright = int(i); break; }
  if (root_outright < 0) throw CompileError("At least one curve must be outright (a discount/base curve).");

  // Currency -> engine-blind integer tag, by first use.
  std::vector<std::string> ccy_order;
  auto ccy_index = [&](const std::string& name) -> int {
    const std::string n = upper(name.empty() ? "USD" : name);
    auto it = std::find(ccy_order.begin(), ccy_order.end(), n);
    if (it != ccy_order.end()) return int(it - ccy_order.begin());
    ccy_order.push_back(n);
    return int(ccy_order.size() - 1);
  };
  // Discount root per currency: first OUTRIGHT curve in that currency.
  std::unordered_map<int, int> ccy_root;
  for (std::size_t i = 0; i < order.size(); ++i)
    if (get_s(*order[i], "kind") != "spread")
      ccy_root.emplace(ccy_index(get_s(*order[i], "currency")), int(i));

  CompileResult R;
  R.value_date = b::iso(value_date);

  // ---- pass 0: resolve every instrument's tokens once, keyed by (curve id, ins key) --------------
  std::map<std::pair<std::string, std::string>, Resolved> resolved;
  for (const json::object* cp : order) {
    const std::string cid = get_s(*cp, "id");
    const json::array empty_arr;
    const json::array& insts = (cp->contains("instruments") && cp->at("instruments").is_array())
                                   ? cp->at("instruments").as_array() : empty_arr;
    const auto starts_map = chained_meeting_starts(insts, value_date);
    for (std::size_t j = 0; j < insts.size(); ++j) {
      const auto& ins = insts[j].as_object();
      const std::string iid = ins_key(ins, j);
      std::optional<std::string> ov;
      if (auto it = starts_map.find(iid); it != starts_map.end()) ov = it->second;
      const std::string ctx =
          get_s(*cp, "name", "?") + " \xC2\xB7 " + get_s(ins, "label", get_s(ins, "type", "?").c_str());
      resolved[{cid, iid}] = resolve_instrument(ins, value_date, ov, ctx);
    }
  }

  // ---- pass 1: curve topology (turns, knot layout, per-curve summary) ----------------------------
  std::map<std::pair<std::string, std::string>, int> turn_index_of;
  double max_t = 1.0;
  std::vector<double> all_knots;
  for (const json::object* cp : order) {
    const json::object& c = *cp;
    const std::string cid = get_s(c, "id");
    const std::string cname = get_s(c, "name", "?");
    const json::array empty_arr;
    const json::array& insts = (c.contains("instruments") && c.at("instruments").is_array())
                                   ? c.at("instruments").as_array() : empty_arr;

    // Calibration turns: single-date banded overnight jumps, one free δ AFTER the interp knots.
    std::vector<px::Turn> curve_turns;
    for (std::size_t j = 0; j < insts.size(); ++j) {
      const auto& i = insts[j].as_object();
      if (get_s(i, "type") != "turn") continue;
      const std::string iid = ins_key(i, j);
      const TurnWindow w = turn_window(i, value_date, get_s(c, "index"), get_s(c, "currency"),
                                       cname + " \xC2\xB7 turn");
      turn_index_of[{cid, iid}] = int(curve_turns.size());
      curve_turns.push_back(px::Turn{w.start, w.end});
      Resolved& rr = resolved[{cid, iid}];
      rr.t_start = w.start; rr.t_end = w.end; rr.has_sd = true; rr.has_ed = true; rr.sd = w.sd; rr.ed = w.ed;
    }

    // Knot-adding instruments: each contributes ONE knot (t_end); a portfolio contributes its `knots`.
    std::vector<KnotItem> knot_items;
    for (std::size_t j = 0; j < insts.size(); ++j) {
      const auto& i = insts[j].as_object();
      if (get_s(i, "type") == "portfolio") {
        if (i.contains("knots") && i.at("knots").is_array())
          for (const auto& ke : i.at("knots").as_array()) {
            const auto& k = ke.as_object();
            const RT kt = resolve_time(get_s(k, "end"), value_date, true, cname + " \xC2\xB7 portfolio knot");
            KnotItem it;
            it.t = std::round(kt.t * 1e10) / 1e10;
            it.region = opt_str(k, "region");
            it.knot_region = get_s(k, "knot_region", "back");
            knot_items.push_back(std::move(it));
          }
      } else if (get_b(i, "adds_knot", false)) {
        KnotItem it;
        it.t = std::round(resolved[{cid, ins_key(i, j)}].t_end * 1e10) / 1e10;
        it.region = opt_str(i, "region");
        it.knot_region = opt_str(i, "knot_region");
        knot_items.push_back(std::move(it));
      }
    }
    if (knot_items.empty())
      throw CompileError("Curve '" + cname + "' has no knots — add at least one knot instrument.");
    for (const auto& k : knot_items)
      if (k.t <= 0)
        throw CompileError("Curve '" + cname + "' has a knot on/before the value date — use a future date.");

    int base = -1;
    if (get_s(c, "kind") == "spread" && present(c, "base")) {
      base = idx.at(get_s(c, "base"));
      if (base >= idx.at(cid))
        throw CompileError("Curve '" + cname + "' is spread off a curve that comes after it.");
    }

    const Layout layout = curve_layout(c, knot_items);

    bool is_xccy = false;
    for (const auto& e : insts) {
      const std::string qk = get_s(e.as_object(), "quote_kind");
      if (qk == "FxForward" || qk == "XccyMtmBasis") { is_xccy = true; break; }
    }
    const std::string ctype = present(c, "type") ? get_s(c, "type") : (is_xccy ? "xccy" : "rate");
    double fwd_tenor = 0.0;
    if (present(c, "index")) {
      if (auto ix = cvd::index(get_s(c, "index")))
        if (ix->type == "ibor" && !ix->tenor.empty()) fwd_tenor = cvd::period_years(ix->tenor);
    }

    PerCurveInfo pc;
    pc.name = cname;
    pc.currency = upper(get_s(c, "currency", "USD"));
    pc.type = ctype;
    pc.fwd_tenor = fwd_tenor;
    pc.policies = layout.policies;
    if (base >= 0) { pc.has_base = true; pc.base_name = get_s(*order[base], "name", "?"); }
    for (const auto& e : insts) if (get_b(e.as_object(), "has_quote", false)) pc.n_quotes++;

    cal::BundleCurveSpec cs;
    cs.base = base;
    cs.currency = ccy_index(get_s(c, "currency"));
    if (layout.classic) {
      cs.regions = curve::flat_hermite(layout.front, layout.back);
      pc.n_front = int(layout.front.size());
      pc.n_back = int(layout.back.size());
      pc.n_knots = pc.n_front + pc.n_back;
    } else {
      cs.regions = layout.modules;
      pc.n_front = 0; pc.n_back = 0;
      for (const auto& m : layout.modules) pc.n_knots += int(m.knots.size());
    }
    if (!curve_turns.empty()) { cs.turns = curve_turns; pc.n_turns = int(curve_turns.size()); }

    R.bundle.curves.push_back(std::move(cs));
    R.per_curve.push_back(std::move(pc));
    for (const auto& k : knot_items) { all_knots.push_back(k.t); max_t = std::max(max_t, k.t); }
  }

  // ---- pass 2: instruments + starts + drifts -----------------------------------------------------
  for (const json::object* cp : order) {
    const json::object& c = *cp;
    const std::string cid = get_s(c, "id");
    const std::string cname = get_s(c, "name", "?");
    const int ci = idx.at(cid);
    const int my_ccy = ccy_index(get_s(c, "currency"));
    const int disc = ccy_root.count(my_ccy) ? ccy_root.at(my_ccy) : root_outright;
    const int spread_base = (get_s(c, "kind") == "spread" && present(c, "base")) ? idx.at(get_s(c, "base")) : disc;

    auto bench_of = [&](const json::object& ins) -> int {
      const std::string bnm = get_s(ins, "bench");
      if (bnm.empty()) return spread_base;
      auto it = idx.find(bnm);
      if (it == idx.end())
        throw CompileError("Basis instrument on '" + cname + "' is quoted against a curve that no longer exists.");
      if (it->second == ci)
        throw CompileError("Basis instrument on '" + cname + "' cannot be quoted against itself.");
      return it->second;
    };

    const json::array empty_arr;
    const json::array& insts = (c.contains("instruments") && c.at("instruments").is_array())
                                   ? c.at("instruments").as_array() : empty_arr;
    for (std::size_t j = 0; j < insts.size(); ++j) {
      const auto& ins = insts[j].as_object();
      const std::string iid = ins_key(ins, j);

      if (get_s(ins, "type") == "portfolio") {
        const b::SwapConv pconv = b::swap_conv(get_s(c, "currency"), 1.0, get_s(c, "index"));
        cal::Instrument obj;
        obj.quote = cal::QuoteKind::Portfolio;
        if (ins.contains("components") && ins.at("components").is_array())
          for (const auto& ce : ins.at("components").as_array()) {
            const auto& comp = ce.as_object();
            const RT ct = resolve_time(get_s(comp, "end"), value_date, true, cname + " \xC2\xB7 portfolio component");
            if (ct.t <= 0)
              throw CompileError("Portfolio component on '" + cname + "' needs a future tenor.");
            cal::WeightedInstrument wi;
            wi.weight = get_d(comp, "weight", 1.0);
            wi.instrument = b::par_swap(value_date, pconv, ct.date, ci, disc, 0.0);
            obj.combination.push_back(std::move(wi));
          }
        if (obj.combination.empty())
          throw CompileError("Portfolio on '" + cname + "' has no component legs.");
        const double mkt = to_decimal(quote_band(ins).target, "bp");
        obj.market = mkt;
        apply_band(obj, ins, "bp");
        R.bundle.instruments.push_back(std::move(obj));
        R.starts.push_back(mkt);
        R.drifts.push_back(get_d(ins, "drift", 0.0));
        continue;
      }
      if (!get_b(ins, "has_quote", false)) continue;

      const std::string kind = get_s(ins, "quote_kind", "ParRate");
      const std::string unit = get_s(ins, "unit", "pct");
      const double mkt = to_decimal(quote_band(ins).target, unit);
      const Resolved& r = resolved.at({cid, iid});
      const double T = r.t_end;
      const b::Date mat_date = r.ed;  // resolved maturity (valid when r.has_ed)

      cal::Instrument obj;
      if (kind == "Rate") {
        const double a = r.t_start;
        if (T <= a)
          throw CompileError("Rate instrument on '" + cname + "' needs end after start.");
        obj = b::rate_instrument(ci,
                                 rate_obs(ins, r, a, T, value_date, b::index_day_count(get_s(c, "index")),
                                          b::index_calendar(get_s(c, "index")), get_s(c, "index")),
                                 mkt);
      } else if (kind == "ParSpread") {
        obj = b::basis_swap(value_date, b::swap_conv(get_s(c, "currency"), 1.0, get_s(c, "index")), mat_date,
                            ci, bench_of(ins), disc, mkt);
      } else if (kind == "FxForward") {
        const std::string den = get_s(ins, "fx_den");
        if (den.empty() || !idx.count(den))
          throw CompileError("FX forward on '" + cname + "' needs a valid 'fx_den' curve.");
        if (idx.at(den) == ci)
          throw CompileError("FX forward on '" + cname + "' cannot use its own curve as fx_den.");
        obj = b::fx_forward(ci, idx.at(den), get_d(ins, "fx_spot", 1.0), T, mkt);
      } else if (kind == "XccyMtmBasis") {
        const std::string fund = get_s(ins, "fx_den");
        if (fund.empty() || !idx.count(fund))
          throw CompileError("Xccy basis on '" + cname + "' needs a valid 'fx_den' curve.");
        if (idx.at(fund) == ci)
          throw CompileError("Xccy basis on '" + cname + "' cannot use its own curve as the fund leg.");
        obj = b::xccy_mtm_basis(value_date, b::xccy_conv(), mat_date, ci, bench_of(ins), idx.at(fund),
                               get_d(ins, "fx_spot", 1.0), mkt);
      } else if (kind == "TurnJump") {
        auto it = turn_index_of.find({cid, iid});
        if (it == turn_index_of.end())
          throw CompileError("Turn on '" + cname + "' has no resolved window (internal).");
        obj = b::turn_jump(ci, it->second, mkt);
      } else {  // ParRate
        obj = b::par_swap(value_date,
                          b::swap_conv(get_s(c, "currency"), get_d(ins, "float_freq", 1.0), get_s(c, "index")),
                          mat_date, ci, disc, mkt);
      }
      apply_band(obj, ins, unit);
      R.bundle.instruments.push_back(std::move(obj));
      R.starts.push_back(mkt);
      R.drifts.push_back(get_d(ins, "drift", 0.0));
    }
  }

  if (R.bundle.instruments.empty())
    throw CompileError("Add at least one quote instrument (a curve needs something to calibrate to).");

  // ---- accounting ---------------------------------------------------------------------------------
  R.n_knots = 0;
  for (const auto& cs : R.bundle.curves) R.n_knots += cs.n_knots();  // interp knots + one δ per turn
  R.n_residuals = int(R.bundle.instruments.size());
  R.under_determined = R.n_residuals < R.n_knots;
  for (const auto& ins : R.bundle.instruments)
    if (ins.band_upper > ins.band_lower) R.has_bands = true;  // mirrors compile.py's has_bands
  if (R.under_determined)
    R.warnings.push_back("Under-determined: " + std::to_string(R.n_residuals) + " quotes for " +
                         std::to_string(R.n_knots) + " knots — smoothness regularization keeps it well-posed.");

  R.grid = display_grid(max_t, all_knots);
  for (const json::object* cp : order) R.curve_names.push_back(get_s(*cp, "name", "?"));
  for (const json::object* cp : order) R.order_ids.push_back(get_s(*cp, "id"));
  R.currency_codes = ccy_order;
  R.smoothness = get_s(spec, "smoothness", "light");
  if (present(spec, "reg_op")) { R.reg_op = get_s(spec, "reg_op"); R.has_reg_op = true; }
  R.tension_sigma = std::max(0.0, opt_num(spec, "tension_sigma").value_or(0.0));

  // Resolved windows, for the UI's read-only date columns.
  for (const json::object* cp : order) {
    const std::string cid = get_s(*cp, "id");
    const json::array empty_arr;
    const json::array& insts = (cp->contains("instruments") && cp->at("instruments").is_array())
                                   ? cp->at("instruments").as_array() : empty_arr;
    for (std::size_t j = 0; j < insts.size(); ++j) {
      const std::string iid = ins_key(insts[j].as_object(), j);
      auto it = resolved.find({cid, iid});
      if (it == resolved.end()) continue;
      ResolvedRow row;
      row.curve_id = cid; row.ins_id = iid;
      row.t_start = it->second.t_start; row.t_end = it->second.t_end;
      if (it->second.has_sd) row.start_date = b::iso(it->second.sd);
      if (it->second.has_ed) row.end_date = b::iso(it->second.ed);
      R.resolved.push_back(std::move(row));
    }
  }
  return R;
}

// ==================================================================================================
json::value compile_to_json(const CompileResult& r) {
  json::object out;
  out["bundle"] = bundle_to_json(r.bundle);
  out["starts"] = json::array(r.starts.begin(), r.starts.end());
  out["drifts"] = json::array(r.drifts.begin(), r.drifts.end());
  out["grid"] = json::array(r.grid.begin(), r.grid.end());
  out["curve_names"] = json::array(r.curve_names.begin(), r.curve_names.end());
  out["order_ids"] = json::array(r.order_ids.begin(), r.order_ids.end());
  out["n_knots"] = r.n_knots;
  out["n_residuals"] = r.n_residuals;

  json::array per_curve;
  for (const auto& pc : r.per_curve) {
    json::object o;
    o["name"] = pc.name;
    o["currency"] = pc.currency;
    o["type"] = pc.type;
    o["base_name"] = pc.has_base ? json::value(pc.base_name) : json::value(nullptr);
    o["fwd_tenor"] = pc.fwd_tenor;
    o["n_quotes"] = pc.n_quotes;
    o["n_front"] = pc.n_front;
    o["n_back"] = pc.n_back;
    o["n_knots"] = pc.n_knots;
    o["policies"] = pc.policies;
    if (pc.n_turns > 0) o["n_turns"] = pc.n_turns;
    per_curve.push_back(std::move(o));
  }
  out["per_curve"] = std::move(per_curve);
  out["warnings"] = json::array(r.warnings.begin(), r.warnings.end());
  out["smoothness"] = r.smoothness;
  out["reg_op"] = r.has_reg_op ? json::value(r.reg_op) : json::value(nullptr);
  out["tension_sigma"] = r.tension_sigma;
  out["under_determined"] = r.under_determined;
  out["has_bands"] = r.has_bands;
  out["value_date"] = r.value_date;
  out["currency_codes"] = json::array(r.currency_codes.begin(), r.currency_codes.end());

  // resolved: {curve_id: {ins_id: {t_start, t_end, start_date|null, end_date|null}}}
  json::object resolved;
  for (const auto& row : r.resolved) {
    if (!resolved.contains(row.curve_id)) resolved[row.curve_id] = json::object();
    json::object cell;
    cell["t_start"] = row.t_start;
    cell["t_end"] = row.t_end;
    cell["start_date"] = row.start_date.empty() ? json::value(nullptr) : json::value(row.start_date);
    cell["end_date"] = row.end_date.empty() ? json::value(nullptr) : json::value(row.end_date);
    resolved[row.curve_id].as_object()[row.ins_id] = std::move(cell);
  }
  out["resolved"] = std::move(resolved);
  return out;
}

std::string compile_json(const std::string& spec_json, const std::string& today_iso) {
  const json::value spec = json::parse(spec_json);
  return json::serialize(compile_to_json(compile_spec(spec, today_iso)));
}

}  // namespace swaps::api
