// Bond-future seam: the stateless "bond_future" run_json verb — deliverable-basket / cheapest-to-deliver
// analytics (curve-free, no bundle). For each bond in a delivery basket it returns the exchange CONVERSION
// FACTOR (CME/Eurex 6% notional closed form), the GROSS and NET basis, the IMPLIED REPO rate and the
// INVOICE price, and it selects the CTD (max implied repo). Decode -> build::analyze_delivery_basket
// (build/bond_future.hpp) -> emit (E7 stage 3.3). QuantLib-free.
//
// Units are the model's decimals (like the bonds verb): coupon/repo are absolute rates (0.06 = 6%), prices
// and basis per unit face (1.0 = par). The thin Python/Excel client converts to the 32nds / bp desk basis.
#include <string>

#include <boost/json.hpp>

#include "swaps/api/bond_future.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/build/bond_future.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;

std::string bond_future_json(const json::object& request) {
  return json::serialize(
      delivery_basket_to_json(b::analyze_delivery_basket(delivery_basket_request_from_json(sub(request, "bond_future")))));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bond_future_json(const std::string& request) { return bond_future_json(json::parse(request).as_object()); }
}  // namespace swaps::api
