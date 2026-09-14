// Full-revaluation VaR / Expected-Shortfall verb (declared in include/swaps/api/var.hpp): decode -> derive::var -> emit
// (PRINCIPLES.md P14). The whole computation -- a supplied P&L series, or calibrate the base once, fork it per market
// move and reprice the book through the cached compiled twin; then reduce the distribution to VaR / ES -- is
// derive/var.hpp; the JSON mapping is api/codec.cpp.

#include <string>

#include <boost/json.hpp>

#include "swaps/api/var.hpp"

#include "swaps/api/bundle_api.hpp"  // BundleSession
#include "swaps/api/codec.hpp"
#include "swaps/derive/var.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string var_json(const json::object& request) {
  return json::serialize(var_result_to_json(derive::var<BundleSession>(var_request_from_json(request))));
}

// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string var_json(const std::string& request) { return var_json(json::parse(request).as_object()); }
}  // namespace swaps::api
