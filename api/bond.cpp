// Bond seams: the stateless "bonds" and "asset_swap" run_json verbs (schemas in include/swaps/api/bond.hpp).
//
//   * bonds      -- street / yield-space analytics (curve-free): decode -> build::street_bonds -> emit (E7 3.2).
//   * asset_swap -- par asset-swap spreads off a CALIBRATED bundle: decode -> calibrate the bundle -> build::asset_swaps
//                   over that curve -> emit (E7 3.5). The float leg, the purchase-price rule and the spread live in
//                   build/par_asset_swap.hpp, pinned against QuantLib::AssetSwap in tests/asset_swap_oracle_test.cpp.
//
// Units are the model's decimals: coupon/yield/spread are absolute rates (0.045 = 4.5%), prices per unit notional
// (1.0 = par). The thin Python/Excel client converts to %/100-basis.
#include <cstddef>
#include <string>
#include <utility>

#include <boost/json.hpp>

#include "swaps/api/bond.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/par_asset_swap.hpp"
#include "swaps/calibration/regularize.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;

std::string bonds_json(const json::object& request) {
  return json::serialize(
      street_analytics_to_json(b::street_bonds(street_bond_request_from_json(sub(request, "bonds")))));
}

namespace {
// The calibrated bundle curve as the discount(t) the asset-swap library prices off.
struct SessionDiscount {
  const BundleSession& session;
  std::size_t curve;
  double discount(double t) const { return session.sample({t}).at(curve).discount.at(0); }
};
}  // namespace

std::string asset_swap_json(const json::object& request) {
  AssetSwapRequest r = asset_swap_request_from_json(sub(request, "asset_swap"));
  BundleSession sess(std::move(r.bundle));
  sess.calibrate(flat_x0(sess.problem()),
                 cal::smoothing_preset(cal::Smoothing::Light, static_cast<int>(sess.problem().curves.size())));
  return json::serialize(asset_swap_to_json(
      b::asset_swaps(r.bonds, r.value_date, SessionDiscount{sess, static_cast<std::size_t>(r.curve)})));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bonds_json(const std::string& request) { return bonds_json(json::parse(request).as_object()); }
std::string asset_swap_json(const std::string& request) { return asset_swap_json(json::parse(request).as_object()); }
}  // namespace swaps::api
