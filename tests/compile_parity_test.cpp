// C++/Python compiler-parity gate. For each golden spec (tools/compile_parity.py in the web repo), compile
// it with the C++ api::compile_spec (the QuantLib-free build/ construction object model) and assert the
// resolved bundle — every curve knot/region/turn and every instrument coupon pay/tau/observation bracket —
// plus the per-instrument start markets and drifts match server/compile.py's compile_spec to ~1e-9.
//
// Both bundles are normalized through bundle_from_json/bundle_to_json so the two sides share one emitter and
// only NUMBERS can differ; a recursive tolerant compare then pins them. QuantLib-free (swaps_api_tests).
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/compile.hpp"

namespace json = boost::json;
using swaps::api::bundle_from_json;
using swaps::api::bundle_to_json;
using swaps::api::compile_spec;

namespace {

std::string golden_dir() { return std::string(SWAPS_GOLDEN_DIR) + "/compile"; }

json::value read_json(const std::string& path) {
  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << "cannot open golden file: " << path;
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}

bool is_num(const json::value& v) { return v.is_double() || v.is_int64() || v.is_uint64(); }

// Recursive tolerant JSON compare: numbers within `tol`, everything else exact. `path` for diagnostics.
void expect_json_eq(const json::value& a, const json::value& b, double tol, const std::string& path) {
  if (is_num(a) && is_num(b)) {
    EXPECT_NEAR(a.to_number<double>(), b.to_number<double>(), tol) << "at " << path;
    return;
  }
  ASSERT_EQ(a.kind(), b.kind()) << "type mismatch at " << path;
  switch (a.kind()) {
    case json::kind::string:
      EXPECT_EQ(a.as_string(), b.as_string()) << "at " << path;
      break;
    case json::kind::bool_:
      EXPECT_EQ(a.as_bool(), b.as_bool()) << "at " << path;
      break;
    case json::kind::null:
      break;
    case json::kind::array: {
      const auto& aa = a.as_array();
      const auto& ba = b.as_array();
      ASSERT_EQ(aa.size(), ba.size()) << "array size mismatch at " << path;
      for (std::size_t i = 0; i < aa.size(); ++i)
        expect_json_eq(aa[i], ba[i], tol, path + "[" + std::to_string(i) + "]");
      break;
    }
    case json::kind::object: {
      const auto& ao = a.as_object();
      const auto& bo = b.as_object();
      ASSERT_EQ(ao.size(), bo.size()) << "object key count mismatch at " << path;
      for (const auto& kv : ao) {
        const std::string key(kv.key());
        ASSERT_TRUE(bo.contains(key)) << "missing key '" << key << "' at " << path;
        expect_json_eq(kv.value(), bo.at(key), tol, path + "." + key);
      }
      break;
    }
    default:
      break;
  }
}

void compare_num_array(const std::vector<double>& got, const json::value& exp, double tol,
                       const std::string& what) {
  ASSERT_TRUE(exp.is_array()) << what << " not an array in golden";
  const auto& ea = exp.as_array();
  ASSERT_EQ(got.size(), ea.size()) << what << " size mismatch";
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], ea[i].to_number<double>(), tol) << what << "[" << i << "]";
}

void run_case(const std::string& name) {
  SCOPED_TRACE(name);
  const json::value spec = read_json(golden_dir() + "/" + name + ".spec.json");
  const json::value expected = read_json(golden_dir() + "/" + name + ".expected.json");
  const auto& exp = expected.as_object();

  const swaps::api::CompileResult r = compile_spec(spec);

  // Bundle: normalize BOTH through the same emitter so only numbers can differ, then compare to 1e-9.
  const json::value cpp_bundle = bundle_to_json(r.bundle);
  const json::value py_bundle = bundle_to_json(bundle_from_json(exp.at("bundle")));
  expect_json_eq(cpp_bundle, py_bundle, 1e-9, name + ".bundle");

  // Streaming config + accounting.
  compare_num_array(r.starts, exp.at("starts"), 1e-12, "starts");
  compare_num_array(r.drifts, exp.at("drifts"), 1e-12, "drifts");
  EXPECT_EQ(r.n_knots, static_cast<int>(exp.at("n_knots").to_number<long long>()));
  EXPECT_EQ(r.n_residuals, static_cast<int>(exp.at("n_residuals").to_number<long long>()));
  EXPECT_EQ(r.value_date, std::string(exp.at("value_date").as_string().c_str()));
}

}  // namespace

TEST(CompileParity, AllGoldenSpecsMatchPython) {
  const json::value idx = read_json(golden_dir() + "/index.json");
  ASSERT_TRUE(idx.is_array());
  ASSERT_FALSE(idx.as_array().empty()) << "no golden specs — run tools/compile_parity.py";
  for (const auto& n : idx.as_array()) run_case(std::string(n.as_string().c_str()));
}
