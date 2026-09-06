// NDF/NDS seam: the stateless "ndf" run_json verb. Prices a book of non-deliverable FX forwards with the
// LINEAR forward/discount kernel (pricing/ndf.hpp) off a supplied market {spot, r_settle, r_nd} — no bundle,
// no calibrated curve, no vol (NDFs are linear). Optionally solves a non-deliverable swap (NDS) fixed strip
// for its fair rate. QuantLib-free. Structured exactly like api/fx_option.cpp (same jd/js/jb helpers, same
// SoA response shape). See api/ndf.hpp for the request/response schema.
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/ndf.hpp"
#include "swaps/pricing/ndf.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace p = swaps::pricing;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const char* d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
bool has_num(const json::object& o, const char* k) {
  return o.contains(k) && (o.at(k).is_number() || o.at(k).is_double() || o.at(k).is_int64() ||
                           o.at(k).is_uint64());
}
// direction: "sell"/"short" -> −1, else (incl. "buy"/"long"/missing) -> +1.
double dir_of(const json::object& o) {
  const std::string s = js(o, "direction", "buy");
  return (s == "sell" || s == "Sell" || s == "SELL" || s == "short" || s == "Short") ? -1.0 : 1.0;
}
}  // namespace

std::string ndf_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("ndf") ? top.at("ndf").as_object() : top;

  const double spot = jd(o, "spot", 0.0);
  if (!(spot > 0.0)) throw std::invalid_argument("ndf: 'spot' must be positive");
  const double r_settle = jd(o, "r_settle", 0.0);
  const double r_nd = jd(o, "r_nd", 0.0);
  const std::string pair = js(o, "pair", "");

  json::array out_trades;
  if (o.contains("trades") && o.at("trades").is_array()) {
    for (const auto& te : o.at("trades").as_array()) {
      const json::object& t = te.as_object();
      const double T = jd(t, "maturity", 0.0);
      if (!(T > 0.0)) throw std::invalid_argument("ndf: each trade needs a positive 'maturity' (years)");
      const double notional = jd(t, "notional", 1.0);
      const double direction = dir_of(t);
      // A trade with no strike prices at the fair forward (PV ≈ 0).
      const double fwd = p::ndf_fair_forward<double>(spot, T, r_settle, r_nd);
      const double strike = has_num(t, "strike") ? jd(t, "strike", fwd) : fwd;

      const p::NdfGreeks<double> g =
          p::ndf_greeks<double>(spot, strike, T, r_settle, r_nd, notional, direction);
      json::object r;
      r["fair_forward"] = g.fair_forward;
      r["strike"] = strike;
      r["maturity"] = T;
      r["pv"] = g.pv;
      r["delta_spot"] = g.delta_spot;
      r["dpv_dr_settle"] = g.dpv_dr_settle;
      r["dpv_dr_nd"] = g.dpv_dr_nd;
      r["notional_settlement"] = g.pv;  // settlement-ccy PV for this trade (already scaled by notional)
      out_trades.push_back(std::move(r));
    }
  }

  json::object out;
  if (!pair.empty()) out["pair"] = pair;
  out["forward"] = p::ndf_fair_forward<double>(spot, 1.0, r_settle, r_nd);  // reference 1y outright
  out["trades"] = std::move(out_trades);

  // Optional NDS: a strip of NDF fixings against one fixed rate -> its fair rate and (optional) strip PV.
  if (o.contains("nds") && o.at("nds").is_object()) {
    const json::object& n = o.at("nds").as_object();
    std::vector<double> maturities;
    std::vector<double> notionals;
    if (n.contains("maturities") && n.at("maturities").is_array())
      for (const auto& m : n.at("maturities").as_array()) maturities.push_back(m.to_number<double>());
    if (n.contains("notionals") && n.at("notionals").is_array())
      for (const auto& v : n.at("notionals").as_array()) notionals.push_back(v.to_number<double>());
    const double fair = p::nds_fair_rate<double>(spot, r_settle, r_nd, maturities, notionals);
    json::object nds_out;
    nds_out["fair_rate"] = fair;
    const double fixed = has_num(n, "fixed_rate") ? jd(n, "fixed_rate", fair) : fair;
    const double direction = dir_of(n);
    nds_out["fixed_rate"] = fixed;
    nds_out["pv"] = p::nds_pv<double>(spot, fixed, r_settle, r_nd, maturities, notionals, direction);
    out["nds"] = std::move(nds_out);
  }

  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
