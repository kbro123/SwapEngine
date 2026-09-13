// Conventions-registry verbs (see include/swaps/api/conventions.hpp): decode -> Registry -> emit (PRINCIPLES.md P14).
// The request decodes to ONE OverlayBatch (api/codec.cpp) and Registry::apply checks every row against the schema's
// rules and commits all of it or none (conventions_db.hpp). QuantLib-free.
#include <string>

#include <boost/json.hpp>

#include "swaps/api/codec.hpp"
#include "swaps/api/conventions.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cvd = swaps::conventions;

std::string conventions_json(const json::object& request) {
  const cvd::OverlayBatch batch = overlay_batch_from_json(request);  // views into `request`
  return json::serialize(
      json::object{{"conventions", overlay_added_to_json(batch, cvd::Registry::instance().apply(batch))}});
}

std::string list_conventions_json(const json::object& /*request*/) {
  return json::serialize(json::object{{"conventions", conventions_listing_to_json(cvd::Registry::instance().listing())}});
}

// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string conventions_json(const std::string& request) { return conventions_json(json::parse(request).as_object()); }
std::string list_conventions_json(const std::string& request) { return list_conventions_json(json::parse(request).as_object()); }
}  // namespace swaps::api
