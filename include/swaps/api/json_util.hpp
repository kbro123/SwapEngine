#pragma once
// ONE set of tolerant JSON accessors + emitters for the api/ verbs (E6.3b, 2026-09-10). Until then eleven verb
// files carried their own copies of jd / js / jb / vecf / darr — the same six lines written eleven times, and
// two of them (`js`'s default, `exposure`'s required default) had already drifted apart. Tolerant by design: a
// missing key returns the supplied default, so a minimal document is valid and every field falls back to the
// engine struct's own default.
//
// NB the string seam these read is not bit-exact: Boost.JSON's serialize→parse loses 1 ULP on ~9.5 % of
// doubles (ASSUMPTIONS.md E1), which is why run_json hands verbs the PARSED object rather than re-parsing text.
#include <Eigen/Core>
#include <boost/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace swaps::api {

namespace json = boost::json;

// ---- required readers (missing / null / wrong type => throw: a field the library has NO default for) -------------
// The codec's side of P14 (2026-09-14, O-X3): a verb that needs a value no struct can default names it here instead of
// branching or inventing a fallback itself.
inline double required_number(const json::object& o, const char* k, const std::string& why) {
  if (!o.contains(k) || !o.at(k).is_number()) throw std::invalid_argument(why);
  return o.at(k).to_number<double>();
}
inline std::string required_text(const json::object& o, const char* k, const std::string& why) {
  if (!o.contains(k) || !o.at(k).is_string() || o.at(k).as_string().empty()) throw std::invalid_argument(why);
  return std::string(o.at(k).as_string().c_str());
}

// ---- readers (missing / null => the supplied default) -----------------------------------------------
inline double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
inline int ji(const json::object& o, const char* k, int d) {
  return o.contains(k) && !o.at(k).is_null() ? static_cast<int>(o.at(k).to_number<double>()) : d;
}
inline bool jb(const json::object& o, const char* k, bool d = false) {
  return o.contains(k) && o.at(k).is_bool() ? o.at(k).as_bool() : d;
}
inline std::string js(const json::object& o, const char* k, const std::string& d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
inline std::vector<double> darr(const json::object& o, const char* k) {
  std::vector<double> out;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) out.push_back(e.to_number<double>());
  return out;
}
inline std::vector<int> iarr(const json::object& o, const char* k) {
  std::vector<int> out;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) out.push_back(static_cast<int>(e.to_number<double>()));
  return out;
}
inline std::vector<int> jia(const json::object& o, const char* k) { return iarr(o, k); }
// A bare JSON array value -> doubles (the C ABI's market / times payloads arrive unwrapped).
inline std::vector<double> to_vec(const json::value& v) {
  std::vector<double> out;
  if (v.is_array())
    for (const auto& e : v.as_array()) out.push_back(e.to_number<double>());
  return out;
}

// ---- emitters --------------------------------------------------------------------------------------
inline json::array vecf(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
inline json::array vecf(const Eigen::VectorXd& v) {
  json::array a;
  a.reserve(static_cast<std::size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}

// A verb's payload: the request's `key` sub-object when present, else the request itself — every verb accepts
// both the {"key": {...}} run_json envelope and the bare payload.
inline const json::object& sub(const json::object& top, const char* key) {
  return top.contains(key) && top.at(key).is_object() ? top.at(key).as_object() : top;
}

}  // namespace swaps::api
