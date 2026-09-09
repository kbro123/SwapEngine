#pragma once
// Conventions-registry seam (PRINCIPLES.md P2): the two stateless run_json verbs that let ANY API (pybind,
// the C ABI, Excel, a web server) add or override market conventions at runtime and list what the engine
// knows. The baked conventions.json is the DEFAULT set; entries added here overlay it for the life of the
// process (last write wins) and are consumed by the very next build/compile/calibrate.
//
//   {"conventions": {"currencies": {...}, "calendars": {...}, "indices": {...}, "products": {...}, "bonds": {...}}}
//       entries use EXACTLY the conventions.json row shapes (the schema in conventions/conventions.schema.json)
//       -> {"conventions": {"added": {"currencies": [...], "calendars": [...], ...}, "overlay_size": n}}
//   {"conventions": {"clear_overlay": true}}   -> drops every runtime entry (tests / session reset)
//   {"list_conventions": true}
//       -> {"conventions": {"currencies": {"baked": [...], "overlay": [...]}, "calendars": {...}, "indices": {...},
//                           "products": {...}, "bonds": {...}}}
#include <string>

namespace swaps::api {

std::string conventions_json(const std::string& request);
std::string list_conventions_json(const std::string& request);

}  // namespace swaps::api
