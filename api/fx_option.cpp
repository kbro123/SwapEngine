// FX-options seam: the stateless "fx_option" (alias "fx_vol") run_json verb. Prices a book of vanilla FX
// options with the Garman-Kohlhagen lognormal model (vol/fx_black.hpp) off a supplied market and a vol
// source — a flat vol, an interbank delta-quoted smile {ATM, RR25, BF25[, RR10, BF10]}, or explicit
// (strike, vol) knots (vol/fx_vol_surface.hpp). No bundle, no calibrated curve: the market is given
// directly. QuantLib-free. Mirrors the shape of api/options.cpp's stateless verbs.
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/fx_option.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/vol/fx_black.hpp"
#include "swaps/vol/fx_vol_surface.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace v = swaps::vol;

namespace {
bool has_num(const json::object& o, const char* k) {
  return o.contains(k) && (o.at(k).is_number() || o.at(k).is_double() || o.at(k).is_int64() ||
                           o.at(k).is_uint64());
}

// Parse a delta-convention string. Default (and every unrecognised value) = spot-unadjusted, so existing
// requests are byte-identical. Accepts the common spellings for {spot,forward}×{unadjusted,premium-adj}.
v::DeltaConv parse_delta_conv(const std::string& s) {
  if (s == "forward" || s == "fwd" || s == "forward_unadj" || s == "fwd_unadj") return v::DeltaConv::FwdUnadj;
  if (s == "spot_pa" || s == "spot_premium_adjusted" || s == "spot_premium_adj") return v::DeltaConv::SpotPA;
  if (s == "forward_pa" || s == "fwd_pa" || s == "forward_premium_adjusted") return v::DeltaConv::FwdPA;
  return v::DeltaConv::SpotUnadj;  // "spot", "spot_unadj", "", or anything else
}

// Parse an ATM-convention string. Default (and unrecognised) = delta-neutral straddle.
v::AtmConv parse_atm_conv(const std::string& s) {
  if (s == "forward" || s == "fwd" || s == "atm_forward" || s == "atmf") return v::AtmConv::Forward;
  return v::AtmConv::DeltaNeutral;  // "dns", "delta_neutral", "straddle", "", or anything else
}
}  // namespace

std::string fx_option_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o = top.contains("fx_option")  ? top.at("fx_option").as_object()
                          : top.contains("fx_vol")    ? top.at("fx_vol").as_object()
                                                      : top;

  const double spot = jd(o, "spot", 0.0);
  const double expiry = jd(o, "expiry", 0.0);
  if (!(spot > 0.0)) throw std::invalid_argument("fx_option: 'spot' must be positive");
  if (!(expiry > 0.0)) throw std::invalid_argument("fx_option: 'expiry' (years) must be positive");
  const double r_dom = jd(o, "r_dom", 0.0);
  const double r_for = jd(o, "r_for", 0.0);
  const std::string pair = js(o, "pair", "");
  // Convention switches (default = today's behaviour: spot-unadjusted delta, delta-neutral-straddle ATM).
  const v::DeltaConv delta_conv = parse_delta_conv(js(o, "delta_convention", ""));
  const v::AtmConv atm_conv = parse_atm_conv(js(o, "atm_convention", ""));

  const v::FxSurfaceMarket mkt{spot, r_dom, r_for};
  const double fwd = v::gk_forward<double>(spot, expiry, r_dom, r_for);
  const double df_for = std::exp(-r_for * expiry);

  // ---- resolve the vol source into an FxVolSurface (delta-quoted smile, strike knots, or flat vol).
  v::FxVolSurface surf = v::FxVolSurface::flat(fwd, expiry, 0.0, df_for);
  bool have_surface = false;
  if (o.contains("surface") && o.at("surface").is_object()) {
    const json::object& s = o.at("surface").as_object();
    if (s.contains("strikes") && s.at("strikes").is_array()) {
      std::vector<double> strikes, vols;
      for (const auto& k : s.at("strikes").as_array()) strikes.push_back(k.to_number<double>());
      if (s.contains("vols") && s.at("vols").is_array())
        for (const auto& vv : s.at("vols").as_array()) vols.push_back(vv.to_number<double>());
      surf = v::FxVolSurface::from_strike_vols(std::move(strikes), std::move(vols), fwd, expiry, df_for);
    } else {
      v::FxDeltaQuotes q;
      q.atm = jd(s, "atm", 0.0);
      q.rr25 = jd(s, "rr25", 0.0);
      q.bf25 = jd(s, "bf25", 0.0);
      if (has_num(s, "rr10") || has_num(s, "bf10")) {
        q.has10 = true;
        q.rr10 = jd(s, "rr10", 0.0);
        q.bf10 = jd(s, "bf10", 0.0);
      }
      surf = v::FxVolSurface::from_delta_quotes(q, fwd, expiry, df_for, delta_conv, atm_conv);
    }
    have_surface = true;
  } else if (has_num(o, "vol")) {
    surf = v::FxVolSurface::flat(fwd, expiry, jd(o, "vol", 0.0), df_for);
    have_surface = true;
  }

  json::array out_opts;
  if (o.contains("options") && o.at("options").is_array()) {
    for (const auto& oe : o.at("options").as_array()) {
      const json::object& t = oe.as_object();
      const v::CallPut cp = jb(t, "put", false) ? v::CallPut::Put : v::CallPut::Call;
      const bool by_delta = has_num(t, "delta") && !has_num(t, "strike");
      const double notional = jd(t, "notional", 1.0);

      double strike;
      if (by_delta) {
        if (!have_surface)
          throw std::invalid_argument("fx_option: a by-delta option needs a 'surface' or 'vol' source");
        strike = surf.strike_for_delta(jd(t, "delta", 0.25), cp, delta_conv);
      } else {
        strike = jd(t, "strike", fwd);
      }

      json::object r;
      r["strike"] = strike;
      r["forward"] = fwd;
      r["call"] = (cp == v::CallPut::Call);

      // Implied-vol mode: an option carrying a 'price' (and no external vol source) inverts to a vol.
      if (has_num(t, "price") && !have_surface) {
        const double px = jd(t, "price", 0.0);
        const double iv = v::gk_implied_vol(px, spot, strike, expiry, r_dom, r_for, cp);
        r["price"] = px;
        r["implied_vol"] = iv;
        const v::GkGreeks<double> g = v::gk_greeks<double>(spot, strike, iv, expiry, r_dom, r_for, cp);
        r["vol"] = iv;
        r["delta"] = g.delta;
        r["gamma"] = g.gamma;
        r["vega"] = g.vega;
        r["theta"] = g.theta;
        r["rho_dom"] = g.rho_dom;
        r["rho_for"] = g.rho_for;
        r["notional_price"] = px * notional;
        out_opts.push_back(std::move(r));
        continue;
      }

      if (!have_surface)
        throw std::invalid_argument("fx_option: no vol source (need 'vol', 'surface', or a per-option 'price')");
      const double vol = surf.vol_at_strike(strike);
      const v::GkGreeks<double> g = v::gk_greeks<double>(spot, strike, vol, expiry, r_dom, r_for, cp);
      r["vol"] = vol;
      r["price"] = g.price;
      r["delta"] = g.delta;
      r["gamma"] = g.gamma;
      r["vega"] = g.vega;
      r["theta"] = g.theta;
      r["rho_dom"] = g.rho_dom;
      r["rho_for"] = g.rho_for;
      r["notional_price"] = g.price * notional;
      out_opts.push_back(std::move(r));
    }
  }

  json::object out;
  if (!pair.empty()) out["pair"] = pair;
  out["forward"] = fwd;
  out["expiry"] = expiry;
  if (have_surface) {
    json::array kx, kv;
    for (double x : surf.logm_knots()) kx.push_back(x);
    for (double vv : surf.vol_knots()) kv.push_back(vv);
    out["smile_logm"] = std::move(kx);
    out["smile_vol"] = std::move(kv);
  }
  out["options"] = std::move(out_opts);
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string fx_option_json(const std::string& request) { return fx_option_json(json::parse(request).as_object()); }
}  // namespace swaps::api
