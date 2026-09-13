// GenerateRisk: the stateless "generate_risk" run_json verb (schema: include/swaps/api/generate_risk.hpp). One book, N
// bundles, all risk implied off bundle[0]'s curve, under-determined bundles rank-completed by self-quoted null pillars.
// decode -> cal::consistent_risk<BundleSession> (calibration/consistent_risk.hpp) -> emit (E7 stage 3.7).
#include <string>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/generate_risk.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/calibration/consistent_risk.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string generate_risk_json(const json::object& request) {
  return json::serialize(consistent_risk_to_json(
      cal::consistent_risk<BundleSession>(consistent_risk_request_from_json(sub(request, "generate_risk")))));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string generate_risk_json(const std::string& request) { return generate_risk_json(json::parse(request).as_object()); }
}  // namespace swaps::api
