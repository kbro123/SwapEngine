// Calibration diagnostics: the stateless "calib_report" run_json verb (schema: include/swaps/api/calib_report.hpp).
// decode -> cal::calibration_report<BundleSession> (calibration/diagnostics.hpp: condition number, singular
// spectrum, rank deficiency, per-quote identifiability and in-band fit) -> emit (E7 stage 3.6). QuantLib-free.
#include <string>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/calib_report.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/calibration/diagnostics.hpp"

namespace swaps::api {

namespace json = boost::json;

std::string calib_report_json(const json::object& request) {
  return json::serialize(calibration_report_to_json(
      cal::calibration_report<BundleSession>(calib_report_request_from_json(sub(request, "calib_report")))));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string calib_report_json(const std::string& request) { return calib_report_json(json::parse(request).as_object()); }
}  // namespace swaps::api
