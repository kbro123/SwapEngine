// RV seam: the stateless bond-RV / swap-spread verbs — the production wiring of the derive/ + market/
// preview layers (the audit's "unreachable capabilities" 1-5).
//
//   * bond_universe — BATCHED street/yield-space analytics over a whole universe on the vectorized
//     portfolio::BondUniverse (Horner/FMA sweep + batched Newton), not a per-bond loop: cleans<->yields,
//     modified duration, convexity, accrued, in one call.
//   * govvie_fit    — MINIMUM-PRICING-ERROR govvie curve fit over a universe of bond clean prices
//     (spline / Nelson-Siegel / Svensson), via derive::make_govvie_fit / make_parametric_fit over a real
//     market::Market built from the request. Returns the fitted state, per-bond fair-value residuals (the
//     RV richness/cheapness ladder), and — for the spline — per-bond z-spreads off the fitted curve
//     (portfolio::CompiledBondBook, the W-cache bond repricer).
//   * swap_spread   — the HEADLINE swap-spread derivation (derive::derive_asset_swap): benchmark bond
//     clean price -> street yield (WI-aware via build::BondId), then the {pin, asw} calibration rows of
//     the asset-swap BASIS (build/swap_spread.hpp) serialized as instrument JSON, ready to compose into a
//     compile/bundle spec. This is the basis-row mode the shipped par-par `asset_swap` verb lacked.
//
// Bonds are identified by build::BondId rows ({id, issue, maturity, coupon, first_coupon?}) with a shared
// conventions-DB `convention` id; quotes are decimals (prices per unit notional, rates absolute).
//
// Each verb is decode -> one library call (derive/bond_rv.hpp bond_universe / govvie_fit, derive/asset_swap.hpp
// swap_spread) -> emit (E7 stage 3.4).
#include <string>

#include <boost/json.hpp>

#include "swaps/api/codec.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/api/rv.hpp"
#include "swaps/derive/bond_rv.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace der = swaps::derive;

std::string bond_universe_json(const json::object& request) {
  return json::serialize(json::object{
      {"bond_universe", bond_universe_to_json(der::bond_universe(bond_universe_request_from_json(sub(request, "bond_universe"))))}});
}

std::string govvie_fit_json(const json::object& request) {
  return json::serialize(json::object{
      {"govvie_fit", govvie_fit_to_json(der::govvie_fit(govvie_fit_request_from_json(sub(request, "govvie_fit"))))}});
}

std::string swap_spread_json(const json::object& request) {
  return json::serialize(json::object{
      {"swap_spread", swap_spread_to_json(der::swap_spread(swap_spread_request_from_json(sub(request, "swap_spread"))))}});
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bond_universe_json(const std::string& request) { return bond_universe_json(json::parse(request).as_object()); }
std::string govvie_fit_json(const std::string& request) { return govvie_fit_json(json::parse(request).as_object()); }
std::string swap_spread_json(const std::string& request) { return swap_spread_json(json::parse(request).as_object()); }
}  // namespace swaps::api
