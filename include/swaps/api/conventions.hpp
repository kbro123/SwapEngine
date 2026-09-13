#pragma once
#include <boost/json.hpp>
// Conventions-registry seam (PRINCIPLES.md P2): the two stateless run_json verbs that let ANY API (pybind,
// the C ABI, Excel, a web server) add or override market conventions at runtime and list what the engine
// knows. The baked conventions.json is the DEFAULT set; entries added here overlay it for the life of the
// process (last write wins) and are consumed by the very next build/compile/calibrate.
//
//   {"conventions": {"currencies": {...}, "calendars": {...}, "indices": {...}, "products": {...}, "bonds": {...},
//                    "bond_futures": {...}, "fx_pairs": {...}, "cb_schedules": {...}, "credit": {"cds_products": {...}},
//                    "fixing_sources": {...}, "inflation": {...}}}
//       entries use EXACTLY the conventions.json row shapes (the schema in conventions/conventions.schema.json).
//       The request is ONE batch (E7 4.2): every row is checked against the schema's rules (Registry::apply in
//       conventions_db.hpp) and the batch is committed whole or not at all, a clear_overlay in it included. A wrong
//       JSON type, a fractional or negative integer, a date that is not a calendar date, one leg under two names or an
//       unknown family throws std::invalid_argument (api/codec.cpp, overlay_batch_from_json).
//       -> {"conventions": {"added": {family: [ids], ...only the families with rows}, "overlay_size": n}}
//   {"conventions": {"clear_overlay": true}}   -> drops every runtime entry (tests / session reset)
//   {"list_conventions": true}
//       -> {"conventions": {"currencies": {"baked": [...], "overlay": [...]}, ...every family..., "overlay_size": n}},
//          read as one snapshot (Registry::listing)
#include <string>

namespace swaps::api {

std::string conventions_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string conventions_json(const std::string& request);
std::string list_conventions_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string list_conventions_json(const std::string& request);

}  // namespace swaps::api
