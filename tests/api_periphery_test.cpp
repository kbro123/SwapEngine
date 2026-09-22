// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T6 regression (fails on the reverted bug)
// Periphery seams fixed 2026-09-10 (E4.E): F1 SABR beta carried by every vol verb; F4 the inflation seasonal
// anchored to the base reference month (value date minus the index's observation lag); D4 the C-ABI session
// and the run_json compile+sample rewrite calibrate under the spec's smoothing (the web's table, now engine data).
#include <gtest/gtest.h>

#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/api/capi.h"
#include "swaps/api/compile.hpp"
#include "swaps/calibration/risk.hpp"  // null_completed_ladder (moved from api, E7 3.7)

namespace api = swaps::api;
namespace json = boost::json;

namespace {
const json::object* find_with_key(const json::value& v, const char* key) {
  if (v.is_object()) {
    if (v.as_object().contains(key)) return &v.as_object();
    for (const auto& kv : v.as_object()) if (const auto* f = find_with_key(kv.value(), key)) return f;
  } else if (v.is_array()) {
    for (const auto& e : v.as_array()) if (const auto* f = find_with_key(e, key)) return f;
  }
  return nullptr;
}
double num(const json::object& o, const char* k) { return o.at(k).to_number<double>(); }
}  // namespace

// F1: beta reaches the vol cube, the swaption verb and the SABR strip fit (they used to be bit-identical
// across beta; {"alpha":0.30,"beta":1.0} priced as a 3039 bp NORMAL vol).
TEST(ApiPeriphery, SabrBetaReachesEveryVolVerb) {
  const auto s = swaps::shapes::ois_lag();
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  const auto cube = [&](double beta, double alpha) {
    return sess.price_vol_cube_json(R"({"value_date":"2026-07-08","index":"USD-SOFR","cells":[{"expiry":"1Y","tenor":"5Y",)"
                                    R"("sabr":{"alpha":)" + std::to_string(alpha) + R"(,"rho":-0.2,"nu":0.3,"beta":)" + std::to_string(beta) +
                                    R"(},"moneyness_bp":[-100,0,100]}]})");
  };
  const api::VolCube c0 = cube(0.0, 0.009), c1 = cube(1.0, 0.25), c5 = cube(0.5, 0.05);
  ASSERT_EQ(c0.n_points, 3);
  EXPECT_NEAR(c0.normal_vol[1], 0.009, 2e-3);  // beta 0: alpha IS the ATM normal vol (to the nu correction)
  // beta 1 with a 25 % Black-level alpha: the ATM NORMAL vol is ~alpha*F (~1 %), not 25 %
  const double F = c1.cell_forward[0];
  EXPECT_LT(c1.normal_vol[1], 0.05);
  EXPECT_NEAR(c1.normal_vol[1], 0.25 * F, 0.3 * 0.25 * F);
  EXPECT_NE(c5.normal_vol[0], c0.normal_vol[0]);
  // the swaption verb
  json::object req;
  json::object sw;
  sw["value_date"] = "2026-07-08";
  sw["bundle"] = api::bundle_to_json(s.prob);
  sw["index"] = "USD-SOFR";
  json::object sabr;
  sabr["alpha"] = 0.30; sabr["rho"] = 0.0; sabr["nu"] = 0.0; sabr["beta"] = 1.0;
  json::object t;
  t["expiry"] = "1Y"; t["tenor"] = "5Y"; t["strike"] = 0.04; t["sabr"] = sabr;
  sw["trades"] = json::array{t};
  req["swaption"] = sw;
  const json::value out = json::parse(api::run_json(json::serialize(req)));
  const json::object* tr = find_with_key(out, "normal_vol");
  ASSERT_NE(tr, nullptr) << json::serialize(out);
  EXPECT_LT(num(*tr, "normal_vol"), 0.05) << "a 30 % lognormal alpha must not price as a 30 % normal vol";
  EXPECT_GT(num(*tr, "normal_vol"), 0.003);
  // beta out of range is refused
  sabr["beta"] = 1.5; t["sabr"] = sabr; sw["trades"] = json::array{t}; req["swaption"] = sw;
  EXPECT_NE(api::run_json(json::serialize(req)).find("\"error\""), std::string::npos);
  // the strip fit echoes beta and fits a different smile at beta 0.5 than at 0
  const auto fit = [&](double beta) {
    json::object r, sc, g;
    sc["forward"] = 0.04; sc["expiry"] = 1.0;
    sc["strikes"] = json::array{0.03, 0.035, 0.04, 0.045, 0.05};
    sc["market_vols"] = json::array{0.0105, 0.0098, 0.0094, 0.0096, 0.0102};
    g["alpha"] = 0.009; g["rho"] = 0.0; g["nu"] = 0.3; g["beta"] = beta;
    sc["guess"] = g;
    r["sabr_calibrate"] = sc;
    return json::parse(api::run_json(json::serialize(r))).as_object();
  };
  const json::object f0 = fit(0.0), f5 = fit(0.5);
  ASSERT_TRUE(f0.contains("beta") && f5.contains("beta")) << json::serialize(f5);
  EXPECT_DOUBLE_EQ(num(f0, "beta"), 0.0);
  EXPECT_DOUBLE_EQ(num(f5, "beta"), 0.5);
  EXPECT_NE(num(f0, "alpha"), num(f5, "alpha"));
}

// F4: the inflation verb anchors the seasonal to the base reference month (value_date - lag) and reports it;
// a whole-year ZCIS strip is invariant to the seasonal for any anchor.
TEST(ApiPeriphery, InflationSeasonalIsAnchoredToTheReferenceMonth) {
  const auto run = [](const char* extra) {
    const std::string req = std::string(R"({"inflation":{"index":"US-CPI-U","base":300.0,"instruments":[)"
        R"({"type":"zcis","maturity":1.0,"rate":0.025},{"type":"zcis","maturity":2.0,"rate":0.026},{"type":"zcis","maturity":5.0,"rate":0.027}],)"
        R"("output_times":[1.0,2.5,5.0])") + extra + "}}";
    return json::parse(api::run_json(req)).as_object();
  };
  const char* seas = R"(,"seasonality":[0.006,0.004,0.003,0.002,0.001,-0.001,-0.002,-0.003,-0.004,-0.003,-0.002,-0.001])";
  const json::object plain = run(""), anchored = run((std::string(seas) + R"(,"value_date":"2026-07-15")").c_str()),
                     legacy = run(seas);
  ASSERT_FALSE(plain.contains("error")) << json::serialize(plain);
  ASSERT_FALSE(anchored.contains("error")) << json::serialize(anchored);
  EXPECT_FALSE(legacy.at("seasonality_anchored").as_bool());
  EXPECT_TRUE(anchored.at("seasonality_anchored").as_bool());
  EXPECT_EQ(std::string(anchored.at("base_reference_month").as_string()), "2026-04");  // July minus the 3-month lag
  EXPECT_NEAR(num(anchored, "seasonality_phase_months"), 3.0 + 14.0 / 30.0, 1e-12);   // April 15th, daily-interpolated
  // whole-year strip: the calibrated forwards do not depend on the seasonal, anchored or not
  const auto fw = [](const json::object& o) { std::vector<double> v; for (const auto& e : o.at("forwards").as_array()) v.push_back(e.to_number<double>()); return v; };
  const auto f_plain = fw(plain), f_anch = fw(anchored), f_leg = fw(legacy);
  for (std::size_t i = 0; i < f_plain.size(); ++i) {
    EXPECT_NEAR(f_anch[i], f_plain[i], 1e-10) << i;
    EXPECT_NEAR(f_leg[i], f_plain[i], 1e-10) << i;
  }
  // the mid-year index (2.5y) differs between the January anchor and the April anchor: a different calendar month
  const double i_leg = legacy.at("index").as_array()[1].to_number<double>(), i_anch = anchored.at("index").as_array()[1].to_number<double>();
  EXPECT_GT(std::abs(i_leg - i_anch), 1e-6);
  // UK-RPI (flat interpolation, 2-month lag): whole-month phase
  const json::object ukr = json::parse(api::run_json(std::string(R"({"inflation":{"index":"UK-RPI","base":300.0,"value_date":"2026-07-15","instruments":[{"type":"zcis","maturity":1.0,"rate":0.03},{"type":"zcis","maturity":2.0,"rate":0.031}])") + seas + "}}")).as_object();
  ASSERT_FALSE(ukr.contains("error")) << json::serialize(ukr);
  EXPECT_EQ(std::string(ukr.at("base_reference_month").as_string()), "2026-05");
  EXPECT_DOUBLE_EQ(num(ukr, "seasonality_phase_months"), 4.0);
}

// D4: the C-ABI session calibrates under the spec's smoothing and agrees with run_json compile+sample_times,
// which now defaults to the same smoothing (the probe: rank_deficiency=1 and 25 bp from the web at 10y).
TEST(ApiPeriphery, CapiSessionAndCompileRewriteHonourTheSpecsSmoothing) {
  const char* spec = R"({"value_date":"2026-07-08","smoothness":"light","curves":[{"id":"SOFR","currency":"USD","index":"USD-SOFR","kind":"outright",
    "regions":[{"id":"r0","name":"Flat","policy":"Flat","sigma":0.0},{"id":"r1","name":"Hermite","policy":"Hermite","sigma":0.0}],
    "instruments":[
      {"type":"rate","end":"1w","start":"0d","adds_knot":true,"knot_region":"front","quote_kind":"Rate","has_quote":true,"unit":"dec","target":0.0432},
      {"type":"swap","end":"1Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0415},
      {"type":"swap","end":"2Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0398},
      {"type":"swap","end":"3Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":false,"unit":"dec","target":0.0},
      {"type":"swap","end":"5Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0389},
      {"type":"swap","end":"10Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0396}]}]})";
  const auto cr = api::compile_spec(json::parse(spec), "2026-07-08");
  ASSERT_TRUE(cr.under_determined);
  const api::RegSpec reg = api::compile_reg_spec(cr);
  EXPECT_TRUE(reg.on());
  EXPECT_DOUBLE_EQ(reg.lambda, 0.5);  // the curvature operator's Light (0.02 under the tension table until 2026-09-21)
  // "off" on an under-determined spec floors to light; "strong" is 5.0; reg_op "tension" is REFUSED (retired 2026-09-22)
  api::CompileResult r2 = cr; r2.smoothness = "off";
  EXPECT_DOUBLE_EQ(api::compile_reg_spec(r2).lambda, 0.5);
  r2.smoothness = "strong";
  EXPECT_DOUBLE_EQ(api::compile_reg_spec(r2).lambda, 5.0);
  r2.has_reg_op = true; r2.reg_op = "tension";
  EXPECT_THROW(api::compile_reg_spec(r2), std::invalid_argument);
  r2.reg_op = "second_difference";  // naming the operator is fine
  EXPECT_DOUBLE_EQ(api::compile_reg_spec(r2).lambda, 5.0);
  r2.under_determined = false; r2.has_bands = false; r2.smoothness = "off";
  EXPECT_FALSE(api::compile_reg_spec(r2).on());

  void* h = swaps_session_create(spec, "2026-07-08");
  ASSERT_NE(h, nullptr);
  const char* cal_s = swaps_session_calibrate(h);
  std::cout << "  [capi] calibrate: " << std::string(cal_s).substr(0, 300) << "\n";
  const json::object cal = json::parse(cal_s).as_object();
  swaps_string_free(cal_s);
  ASSERT_FALSE(cal.contains("error")) << json::serialize(cal);
  EXPECT_TRUE(cal.at("regularize_applied").as_bool());
  EXPECT_DOUBLE_EQ(num(cal, "regularize_lambda"), 0.5);
  EXPECT_EQ(cal.at("rank_deficiency").as_int64(), 0) << "regularised: the un-quoted 3Y knot is pinned by the penalty";
  const char* smp_s = swaps_session_sample(h, "[1.0,2.0,3.0,4.0,5.0,10.0]");
  std::cout << "  [capi] sample: " << std::string(smp_s).substr(0, 200) << "\n";
  const json::value smp = json::parse(smp_s);
  swaps_string_free(smp_s);
  swaps_session_free(h);
  // both shapes: {"curves":[{forward:[...]}]} or a bare array of curves
  const auto fwd = [](const json::value& v) {
    const json::value& c0 = v.is_object() ? v.as_object().at("curves").as_array()[0] : v.as_array()[0];
    std::vector<double> out;
    for (const auto& e : c0.as_object().at("forward").as_array()) out.push_back(e.to_number<double>());
    return out;
  };
  const std::vector<double> f_capi = fwd(smp);
  // run_json compile + sample_times, no explicit regularize: the spec's smoothing is the default now
  json::object rq; rq["compile"] = json::parse(spec); rq["sample_times"] = json::array{1.0, 2.0, 3.0, 4.0, 5.0, 10.0};
  const json::object rw = json::parse(api::run_json(json::serialize(rq))).as_object();
  ASSERT_FALSE(rw.contains("error")) << json::serialize(rw);
  EXPECT_TRUE(rw.at("calibration").as_object().at("regularize_applied").as_bool());
  const std::vector<double> f_rw = fwd(json::value(rw));
  // ... and equals an explicit light curvature request
  json::object rg; rg["lambda"] = 0.5; rg["curves"] = json::array{0};
  rq["regularize"] = rg;
  const std::vector<double> f_ex = fwd(json::parse(api::run_json(json::serialize(rq))));
  ASSERT_EQ(f_capi.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(f_capi[i], f_ex[i], 1e-10) << i;
    EXPECT_NEAR(f_rw[i], f_ex[i], 1e-10) << i;
  }
}

// G5: generate_risk's null completion decides "unseen by the quotes" at the engine's ONE rank threshold on the
// singular values (was: normal-equation eigenvalues at 1e-9 of the largest, i.e. singular values at 3e-5 --
// a stiff-but-constrained direction was self-quoted as a synthetic pillar).
TEST(ApiPeriphery, GenerateRiskNullCompletionUsesTheSharedRankThreshold) {
  std::vector<int> syn;
  Eigen::MatrixXd J(2, 2);
  J << 1.0, 0.0, 0.0, 1e-6;  // a stiff (sigma ratio 1e-6) but CONSTRAINED direction: not null
  Eigen::VectorXd g(2); g << 1.0, 2.0;
  const swaps::calibration::NullCompletedLadder L = swaps::calibration::null_completed_ladder(J, g, Eigen::VectorXd::Ones(J.rows()));
  syn = L.synthetic_knot;
  const Eigen::VectorXd lad = L.full;
  EXPECT_EQ(syn.size(), 0u) << "sigma 1e-6 is constrained at kRankThreshold 1e-10 (the old 3e-5 cut self-quoted it)";
  ASSERT_EQ(lad.size(), 2);
  EXPECT_NEAR(lad[0], 1.0, 1e-12);
  EXPECT_NEAR(lad[1], 2.0 / 1e-6, 1e-3);
  Eigen::MatrixXd J1(1, 2);
  J1 << 1.0, 0.0;  // knot 1 genuinely unseen
  const swaps::calibration::NullCompletedLadder L1 = swaps::calibration::null_completed_ladder(J1, g, Eigen::VectorXd::Ones(J1.rows()));
  syn = L1.synthetic_knot;
  const Eigen::VectorXd lad1 = L1.full;
  ASSERT_EQ(syn.size(), 1u);
  EXPECT_EQ(syn[0], 1);
  ASSERT_EQ(lad1.size(), 2);
  EXPECT_NEAR(lad1[0], 1.0, 1e-12);
  EXPECT_NEAR(std::abs(lad1[1]), 2.0, 1e-12);  // the synthetic pillar carries the unseen knot's gradient
}

// REVIEW REGRESSION (2026-09-21, findings 1 & 4): the C ABI's error path must return JSON, and a failed
// session_create must leave a diagnostic behind. Exception messages quote CALLER strings back, so the test
// deliberately sends a position kind CONTAINING A DOUBLE QUOTE -- the exact byte that used to escape the
// hand-built error literal and hand the host un-parseable bytes.
TEST(ApiPeriphery, TheCAbiErrorPathReturnsParseableJsonEvenWhenTheMessageQuotesTheCaller) {
  const char* spec = R"({"value_date":"2026-07-08","curves":[{"id":"SOFR","currency":"USD","index":"USD-SOFR","kind":"outright",
    "regions":[{"id":"r0","name":"Hermite","policy":"Hermite","sigma":0.0}],
    "instruments":[
      {"type":"swap","end":"1Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0415},
      {"type":"swap","end":"5Y","start":null,"adds_knot":true,"knot_region":"back","quote_kind":"ParRate","has_quote":true,"unit":"dec","target":0.0389}]}]})";
  void* h = swaps_session_create(spec, "2026-07-08");
  ASSERT_NE(h, nullptr) << "diagnostic: " << swaps_last_error();
  swaps_string_free(swaps_session_calibrate(h));

  const char* bad = swaps_session_price(h, R"({"positions":[{"kind":"sw\"ap","notional":1.0}]})");
  ASSERT_NE(bad, nullptr);
  const std::string raw(bad);
  swaps_string_free(bad);
  swaps_session_free(h);
  json::error_code ec;
  const json::value v = json::parse(raw, ec);
  ASSERT_FALSE(ec) << "the error is not JSON: " << raw;   // the bug: parse failed at the embedded quote
  ASSERT_TRUE(v.is_object());
  ASSERT_TRUE(v.as_object().contains("error"));
  const std::string msg(v.as_object().at("error").as_string());
  EXPECT_NE(msg.find("sw\"ap"), std::string::npos) << "the message must still carry the offending kind: " << msg;
  EXPECT_NE(raw.find("sw\\\"ap"), std::string::npos) << "and it must be ESCAPED on the wire: " << raw;
}

// FINDING 4's other half: a spec that cannot compile answers NULL *and* records why.
TEST(ApiPeriphery, AFailedSessionCreateRecordsItsDiagnostic) {
  EXPECT_EQ(swaps_session_create(R"({"curves":[)", "2026-07-08"), nullptr);
  const std::string diag = swaps_last_error();
  EXPECT_FALSE(diag.empty()) << "a NULL handle used to be the whole story";
  EXPECT_EQ(swaps_session_create(nullptr, "2026-07-08"), nullptr);
}
