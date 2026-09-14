// Scenario-GRID verb (declared in include/swaps/api/scenario_grid.hpp): decode -> derive::scenario_grid -> emit
// (PRINCIPLES.md P14). The whole computation -- calibrate the base once, fork it per cell of one or two shock axes,
// reprice through the cached compiled book -- is derive/scenario_grid.hpp; the JSON mapping is api/codec.cpp.

#include <string>

#include <boost/json.hpp>

#include "swaps/api/scenario_grid.hpp"

#include "swaps/api/bundle_api.hpp"  // BundleSession
#include "swaps/api/codec.hpp"
#include "swaps/derive/scenario_grid.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string scenario_grid_json(const json::object& request) {
  return json::serialize(
      scenario_grid_result_to_json(derive::scenario_grid<BundleSession>(scenario_grid_request_from_json(request))));
}

// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string scenario_grid_json(const std::string& request) { return scenario_grid_json(json::parse(request).as_object()); }
}  // namespace swaps::api
