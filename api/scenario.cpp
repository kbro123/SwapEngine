// Scenario / stress verb (declared in include/swaps/api/scenario.hpp): decode -> derive::scenarios -> emit
// (PRINCIPLES.md P14). The whole computation -- calibrate the base once, fork the fitted state per move, sample and
// value the book there -- is derive/scenario.hpp; the JSON mapping is api/codec.cpp.

#include <string>

#include <boost/json.hpp>

#include "swaps/api/scenario.hpp"

#include "swaps/api/bundle_api.hpp"  // BundleSession
#include "swaps/api/codec.hpp"
#include "swaps/derive/scenario.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string scenario_json(const json::object& request) {
  return json::serialize(
      scenario_result_to_json(derive::scenarios<BundleSession>(scenario_request_from_json(request))));
}

// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string scenario_json(const std::string& request) { return scenario_json(json::parse(request).as_object()); }
}  // namespace swaps::api
