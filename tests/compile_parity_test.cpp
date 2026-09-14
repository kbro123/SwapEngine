// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// C++/Python compiler-parity gate. For each golden spec (tools/compile_parity.py in the web repo), compile
// it with the C++ api::compile_spec (the QuantLib-free build/ construction object model) and assert the
// resolved bundle — every curve knot/region/turn and every instrument coupon pay/tau/observation bracket —
// plus the per-instrument start markets and drifts match server/compile.py's compile_spec to ~1e-9.
//
// Both bundles are normalized through bundle_from_json/bundle_to_json so the two sides share one emitter and
// only NUMBERS can differ; a recursive tolerant compare then pins them. QuantLib-free (swaps_api_tests).
#include <gtest/gtest.h>

#include <cstring>
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
      // THE TWO KNOWN, DELIBERATE C++/PYTHON DIVERGENCES, both in an averaged overnight observation, both
      // fixed in C++ and still present in server/conventions.py (web work deferred, TASKS-API §A0.5). They
      // land in the SAME arrays, so they are pinned together, tightly enough that a third one still fails:
      //
      //   item 17  the per-day WEIGHT is `accrual earned / index year-fraction of the observation window`.
      //            Python still divides by the window's CURVE-TIME length, so every Python weight is exactly
      //            365/360 for an ACT/360 index where ours is 1 (or a fraction at a partly-earned end).
      //   E3       a window opening or closing on a NON-BUSINESS DAY keeps the fixing's own overnight window
      //            and clips only the ACCRUAL. Python enumerates the business days strictly inside the
      //            window, so it drops a leading fixing entirely and truncates the trailing one. Ours can
      //            therefore have ONE extra leading entry, and a later final sub_end.
      const auto ends_with = [&path](const char* suf) {
        const std::size_t n = std::strlen(suf);
        return path.size() >= n && path.compare(path.size() - n, n, suf) == 0;
      };
      if (ends_with(".sub_start") || ends_with(".sub_end") || ends_with(".weight")) {
        if (ends_with(".weight")) {
          // Ours is EMPTY whenever every weight is 1 (the all-ones fast path), so no size relation holds here.

          for (std::size_t i = 0; i < ba.size(); ++i)
            EXPECT_NEAR(ba[i].to_number<double>(), 365.0 / 360.0, 1e-12)
                << "at " << path << "[" << i << "]: Python's averaged weight is the pre-item-17 365/360";
          for (std::size_t i = 0; i < aa.size(); ++i) {
            const double w = aa[i].to_number<double>();
            EXPECT_GT(w, 0.0) << "at " << path << "[" << i << "]";
            EXPECT_LE(w, 1.0) << "at " << path << "[" << i << "]";
            if (i > 0 && i + 1 < aa.size())
              EXPECT_NEAR(w, 1.0, 1e-12) << "at " << path << "[" << i << "]: an interior day earns its whole fixing";
          }
          return;
        }
        // sub_start / sub_end: beyond the recovered leading fixing the windows must match exactly, except
        // the final sub_end, which E3 may extend to the trailing fixing's own end.
        ASSERT_TRUE(aa.size() == ba.size() || aa.size() == ba.size() + 1)
            << "at " << path << ": E3 recovers at most ONE leading fixing";
        const std::size_t off = aa.size() - ba.size();
        for (std::size_t i = 0; i + 1 < ba.size(); ++i)
          EXPECT_NEAR(aa[i + off].to_number<double>(), ba[i].to_number<double>(), tol)
              << "at " << path << "[" << i << "]";
        if (!ba.empty()) {
          const double ours = aa[aa.size() - 1].to_number<double>(), theirs = ba[ba.size() - 1].to_number<double>();
          if (ends_with(".sub_end")) EXPECT_GE(ours, theirs - tol) << "at " << path << " (last)";
          else EXPECT_NEAR(ours, theirs, tol) << "at " << path << " (last)";
        }
        return;
      }
      ASSERT_EQ(aa.size(), ba.size()) << "array size mismatch at " << path;
      for (std::size_t i = 0; i < aa.size(); ++i)
        expect_json_eq(aa[i], ba[i], tol, path + "[" + std::to_string(i) + "]");
      break;
    }
    case json::kind::object: {
      const auto& ao = a.as_object();
      const auto& bo = b.as_object();
      // THE FOURTH DELIBERATE DIVERGENCE (O-X3 piece 2, 2026-09-14; TASKS-API §A0.5): C++ quotes fx_spot for the pair's SPOT
      // date and emits fx_spot_time (FX forwards, the MtM leg) plus each MtM coupon's fx_fixing_time; compile.py emits neither.
      // Ours must be a spot time in (0, 7/365] and fixing times in [0, the coupon's start] within a week of it; the keys are
      // then dropped and everything else compares at 1e-9.
      if (ao.contains("quote") && ao.at("quote").is_string() && ao.at("quote").as_string() == "FxForward" &&
          ao.contains("fx_spot_time")) {
        json::object a2 = ao;
        const double ts = a2.at("fx_spot_time").to_number<double>();
        EXPECT_GT(ts, 0.0) << "at " << path << ".fx_spot_time";
        EXPECT_LE(ts, 7.0 / 365.0) << "at " << path << ".fx_spot_time";
        EXPECT_FALSE(bo.contains("fx_spot_time")) << "compile.py now emits fx_spot_time: delete this divergence";
        a2.erase("fx_spot_time");
        ASSERT_EQ(a2.size(), bo.size()) << "object key count mismatch at " << path;
        for (const auto& kv : a2) {
          const std::string key(kv.key());
          ASSERT_TRUE(bo.contains(key)) << "missing key '" << key << "' at " << path;
          expect_json_eq(kv.value(), bo.at(key), tol, path + "." + key);
        }
        break;
      }
      // THE THIRD DELIBERATE DIVERGENCE (XB1, 2026-09-14; TASKS-API §A0.5): an XccyMtmBasis self leg (fwd) is the
      // constant leg's notional-exchange pair, so C++ pays each self coupon on its ACCRUAL END; compile.py still
      // pays it with the lagged coupons. Each self coupon's pay must be our accrual end (the OIS bracket's last
      // sub_end) and strictly before Python's; that one field is then neutralised and the rest compares normally.
      if (ao.contains("quote") && ao.at("quote").is_string() && ao.at("quote").as_string() == "XccyMtmBasis" &&
          ao.contains("fwd") && bo.contains("fwd")) {
        json::object a2 = ao, b2 = bo;
        auto& ac = a2["fwd"].as_object()["coupons"].as_array();
        const auto& bc = b2["fwd"].as_object()["coupons"].as_array();
        ASSERT_EQ(ac.size(), bc.size()) << "at " << path << ".fwd.coupons";
        for (std::size_t i = 0; i < ac.size(); ++i) {
          auto& x = ac[i].as_object();
          const double ours = x.at("pay").to_number<double>();
          const double theirs = bc[i].as_object().at("pay").to_number<double>();
          const auto& ends = x.at("obs").as_object().at("sub_end").as_array();
          ASSERT_FALSE(ends.empty()) << "at " << path << ".fwd.coupons[" << i << "]";
          EXPECT_NEAR(ours, ends.back().to_number<double>(), 1e-15)
              << "at " << path << ".fwd.coupons[" << i << "]: XB1 pays the self coupon on its accrual end";
          EXPECT_GT(theirs, ours) << "at " << path << ".fwd.coupons[" << i << "]: compile.py lags the self leg";
          x["pay"] = theirs;
        }
        if (a2.contains("mtm")) {  // the 4th divergence on the MtM leg (see above)
          auto& am = a2["mtm"].as_object();
          if (am.contains("fx_spot_time")) {
            const double ts = am.at("fx_spot_time").to_number<double>();
            EXPECT_GT(ts, 0.0) << "at " << path << ".mtm.fx_spot_time";
            EXPECT_LE(ts, 7.0 / 365.0) << "at " << path << ".mtm.fx_spot_time";
            am.erase("fx_spot_time");
          }
          for (auto& cv : am["coupons"].as_array()) {
            auto& co = cv.as_object();
            if (!co.contains("fx_fixing_time")) continue;
            const double ft = co.at("fx_fixing_time").to_number<double>();
            const double start = co.at("obs").as_object().at("sub_start").as_array().front().to_number<double>();
            EXPECT_GE(ft, 0.0) << "at " << path << ".mtm fixing";
            EXPECT_LE(ft, start + 1e-12) << "at " << path << ".mtm fixing: the FX fixes on or before the period start";
            EXPECT_LE(start - ft, 7.0 / 365.0) << "at " << path << ".mtm fixing";
            co.erase("fx_fixing_time");
          }
        }
        ASSERT_EQ(a2.size(), b2.size()) << "object key count mismatch at " << path;
        for (const auto& kv : a2) {
          const std::string key(kv.key());
          ASSERT_TRUE(b2.contains(key)) << "missing key '" << key << "' at " << path;
          expect_json_eq(kv.value(), b2.at(key), tol, path + "." + key);
        }
        break;
      }
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

// compile_reg_spec reads the ONE smoothing table (calibration/regularize.hpp smoothing_preset, E7 stage 3). The
// expectations restate, by hand, the table compile_reg_spec carried inline until 2026-09-13: tension light 0.02 /
// strong 0.2, second-difference light 0.5 / strong 5.0, "off" promoted to light when the bundle is under-determined
// or banded, the penalty spanning the named curves (else every bundle curve), sigma only for the tension operator.
TEST(CompileRegSpec, ReadsTheOneSmoothingTable) {
  swaps::api::CompileResult r;
  r.smoothness = "light";
  r.curve_names = {"A", "B"};
  r.tension_sigma = 0.3;
  swaps::api::RegSpec g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 0.02);
  EXPECT_TRUE(g.tension);
  EXPECT_EQ(g.sigma, 0.3);
  EXPECT_EQ(g.curves, (std::vector<int>{0, 1}));

  r.smoothness = "strong";
  g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 0.2);

  r.has_reg_op = true;
  r.reg_op = "second_difference";
  g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 5.0);
  EXPECT_FALSE(g.tension);
  EXPECT_EQ(g.sigma, 0.0);

  r.smoothness = "off";
  g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 0.0);
  EXPECT_FALSE(g.on());

  r.under_determined = true;
  g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 0.5) << "off is promoted to light (second-difference) when under-determined";

  r.under_determined = false;
  r.has_bands = true;
  r.has_reg_op = false;
  r.curve_names.clear();
  r.bundle.curves.resize(3);
  g = swaps::api::compile_reg_spec(r);
  EXPECT_EQ(g.lambda, 0.02) << "off is promoted to light when banded";
  EXPECT_EQ(g.curves, (std::vector<int>{0, 1, 2})) << "no named curves => every bundle curve";
}
