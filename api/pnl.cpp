// P&L EXPLAIN verb (declared in include/swaps/api/pnl.hpp): decode -> calibration::pnl_report -> emit (PRINCIPLES.md
// P14). The whole computation -- calibrate bundle0 (and bundle1), take the analytic ladder at x0, and attribute the
// NPV change into carry / roll / market / residual -- is calibration/pnl_explain.hpp; the JSON mapping is api/codec.cpp.
#include <string>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"  // BundleSession
#include "swaps/api/codec.hpp"
#include "swaps/api/pnl.hpp"
#include "swaps/calibration/pnl_explain.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string pnl_json(const json::object& request) {
  return json::serialize(pnl_report_to_json(cal::pnl_report<BundleSession>(pnl_request_from_json(request))));
}

// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string pnl_json(const std::string& request) { return pnl_json(json::parse(request).as_object()); }
}  // namespace swaps::api
