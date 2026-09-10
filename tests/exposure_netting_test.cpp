// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// Netting-set gate for the "exposure" run_json verb (api/exposure.cpp). NO QuantLib: a small single-curve
// bundle is hand-built as generic Instruments (api_test.cpp style), made self-consistent from a known
// x_true, and driven through the JSON contract. Proves the aggregation boundary moved to trade::NettingSet:
//   (a) a two-set request returns two per-set profiles;
//   (b) the netting boundary MATTERS — two offsetting trades in the SAME set net to ~zero exposure, the
//       same two trades in DIFFERENT sets do not;
//   (c) the CSA decides the discount role — a set whose CSA maps to role 0 produces the profile of the
//       equivalent hand-built whole book discounted on role 0 (the legacy path), bit-for-bit;
//   (d) the legacy whole-book request is byte-identical to before: rerunning it reproduces the exact
//       response (modulo wall_us), and adding netting_sets alongside leaves every legacy field untouched.

#include <gtest/gtest.h>

#include <algorithm>

#include <boost/json.hpp>

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/exposure.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

// Single classic Flat+Hermite curve (the exposure verb's demo scope).
const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
constexpr int kNk = 7;

// One OIS-style float coupon over [a,b]: a single telescoped sub-period (tau_pay == tau_index).
px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}

// A par-rate OIS swap to maturity T (annual schedule), forecast/discount on curve 0.
cal::Instrument make_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = 0;
  ins.fwd.discount = 0;
  ins.fixed.discount = 0;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    ins.fwd.coupons.push_back(ois_coupon(prev, t));
    px::FixedCoupon x;
    x.pay = t;
    x.tau = t - prev;
    ins.fixed.coupons.push_back(x);
    prev = t;
  }
  return ins;
}

// A front "Rate" pin over [a,b] on curve 0 (pins the flat meeting segments).
cal::Instrument make_rate(double a, double b) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// Build the single-curve bundle (instruments with market = model quote at x_true — self-consistent).
cal::BundleProblem build_bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25));
  p.instruments.push_back(make_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T));

  Eigen::VectorXd x_true(kNk);
  for (int i = 0; i < kNk; ++i) x_true[i] = 0.030 + 0.001 * i;
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return x_true[i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

constexpr int kPaths = 256;
constexpr int kNodes = 6;

// The shared request scaffold: bundle + MC controls + the trade binding (value_date / curve_roles).
json::object base_request(const cal::BundleProblem& p) {
  json::object o;
  o["bundle"] = api::bundle_to_json(p);
  o["n_paths"] = kPaths;
  o["n_nodes"] = kNodes;
  o["horizon_years"] = 5.0;
  o["value_date"] = "2026-09-04";
  json::object roles;
  roles["USD-SOFR"] = 0;
  o["curve_roles"] = roles;
  return o;
}

json::object trade(const char* id, const char* pay, double rate = 0.03,
                   const char* maturity = "2031-09-08") {
  json::object t;
  t["id"] = id;
  t["notional"] = 1e6;
  t["pay"] = pay;
  t["fixed_rate"] = rate;
  t["index"] = "USD-SOFR";
  t["effective"] = "2026-09-08";
  t["maturity"] = maturity;
  return t;
}

json::object usd_csa() {
  json::object csa;
  csa["collateral_currency"] = "USD";
  return csa;
}

json::object nset(const char* id, json::array trades) {
  json::object s;
  s["id"] = id;
  s["csa"] = usd_csa();
  s["trades"] = std::move(trades);
  return s;
}

json::object run(const json::object& req) {
  return json::parse(api::exposure_json(json::serialize(json::value(req)))).as_object();
}

std::vector<double> arr(const json::object& o, const char* k) {
  std::vector<double> v;
  for (const auto& e : o.at(k).as_array()) v.push_back(e.to_number<double>());
  return v;
}

}  // namespace

// (a) Two netting sets -> two per-set profiles (and no whole-book fields when no "book" is given).
TEST(ExposureNetting, TwoSetRequestReturnsTwoPerSetProfiles) {
  const cal::BundleProblem p = build_bundle();
  json::object req = base_request(p);
  json::array sets;
  sets.push_back(nset("CP-1", json::array{trade("T1", "fixed")}));
  sets.push_back(nset("CP-2", json::array{trade("T2", "fixed", 0.035, "2029-09-08")}));
  req["netting_sets"] = std::move(sets);

  const json::object out = run(req);
  EXPECT_FALSE(out.contains("epe")) << "no legacy whole-book fields without a 'book'";
  ASSERT_TRUE(out.contains("netting_sets"));
  const json::array& ns = out.at("netting_sets").as_array();
  ASSERT_EQ(ns.size(), 2u);
  EXPECT_EQ(json::value_to<std::string>(ns[0].as_object().at("id")), "CP-1");
  EXPECT_EQ(json::value_to<std::string>(ns[1].as_object().at("id")), "CP-2");
  for (const auto& e : ns) {
    const json::object& so = e.as_object();
    EXPECT_EQ(so.at("epe").as_array().size(), static_cast<std::size_t>(kNodes));
    EXPECT_EQ(so.at("ene").as_array().size(), static_cast<std::size_t>(kNodes));
    EXPECT_EQ(so.at("pfe").as_array().size(), static_cast<std::size_t>(kNodes));
    EXPECT_EQ(so.at("node_time").as_array().size(), static_cast<std::size_t>(kNodes));
    EXPECT_EQ(so.at("n").as_int64(), 1);
  }
}

// (b) The netting boundary matters: payer + receiver on identical terms net to ~zero exposure INSIDE one
// set; split across two sets each side carries real (positive-EPE) exposure.
TEST(ExposureNetting, OffsettingTradesNetWithinASetButNotAcrossSets) {
  const cal::BundleProblem p = build_bundle();

  // Same set: the two trades' state-by-state NPVs cancel exactly -> flat-zero profile.
  json::object same = base_request(p);
  same["netting_sets"] =
      json::array{nset("CP-1", json::array{trade("T1", "fixed"), trade("T2", "float")})};
  const json::object out_same = run(same);
  const json::object& s0 = out_same.at("netting_sets").as_array()[0].as_object();
  for (double v : arr(s0, "epe")) EXPECT_NEAR(v, 0.0, 1e-6);
  for (double v : arr(s0, "ene")) EXPECT_NEAR(v, 0.0, 1e-6);
  for (double v : arr(s0, "pfe")) EXPECT_NEAR(v, 0.0, 1e-6);

  // Different sets: no cross-set netting — each side shows genuine future exposure.
  json::object split = base_request(p);
  split["netting_sets"] = json::array{nset("CP-A", json::array{trade("T1", "fixed")}),
                                      nset("CP-B", json::array{trade("T2", "float")})};
  const json::object out_split = run(split);
  const json::array& ns = out_split.at("netting_sets").as_array();
  ASSERT_EQ(ns.size(), 2u);
  for (const auto& e : ns) {
    const std::vector<double> epe = arr(e.as_object(), "epe");
    // The book is aged per node (2026-09-10): at the 5y horizon a 5y swap has days of accrual left, so the
    // exposure that matters is the interior maximum, not the last node.
    EXPECT_GT(*std::max_element(epe.begin(), epe.end()), 100.0) << "an un-netted 1mm 5y swap must carry real exposure before it matures";
  }
}

// (c) The CSA decides the discount role: a set whose CSA (USD -> USD-SOFR) maps to role 0 reproduces the
// equivalent hand-built whole book discounted on role 0 — the legacy path — bit-for-bit (same trades, same
// simulated state grid).
TEST(ExposureNetting, CsaDiscountRoleMatchesTheEquivalentWholeBook) {
  const cal::BundleProblem p = build_bundle();

  json::object via_set = base_request(p);
  via_set["netting_sets"] = json::array{nset("CP-1", json::array{trade("T1", "fixed")})};
  const json::object out_set = run(via_set);  // hold the result: a reference into the temporary dangles
  const json::object& so = out_set.at("netting_sets").as_array()[0].as_object();

  json::object legacy = base_request(p);
  json::object t = trade("T1", "fixed");
  t["csa"] = usd_csa();  // book_from_json: the CSA -> collateral OIS -> curve_roles["USD-SOFR"] = 0
  json::object book;
  book["value_date"] = "2026-09-04";
  json::object roles;
  roles["USD-SOFR"] = 0;
  book["curve_roles"] = roles;
  book["trades"] = json::array{t};
  legacy["book"] = std::move(book);
  const json::object out_legacy = run(legacy);

  for (const char* k : {"epe", "ene", "pfe", "node_time"}) {
    const std::vector<double> a = arr(so, k);
    const std::vector<double> b = arr(out_legacy, k);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_DOUBLE_EQ(a[i], b[i]) << k << "[" << i << "]";
  }
  EXPECT_DOUBLE_EQ(so.at("mtm").to_number<double>(), out_legacy.at("mtm").to_number<double>());
}

// (d) The legacy whole-book request is byte-identical to before: the same request twice gives the same
// response (modulo the wall_us telemetry), and adding netting_sets alongside leaves every legacy field
// byte-identical — the new code path never perturbs the old one.
TEST(ExposureNetting, LegacyWholeBookRequestIsByteIdentical) {
  const cal::BundleProblem p = build_bundle();
  json::object legacy = base_request(p);
  json::object book;
  book["value_date"] = "2026-09-04";
  json::object roles;
  roles["USD-SOFR"] = 0;
  book["curve_roles"] = roles;
  json::object t1 = trade("T1", "fixed");
  t1["csa"] = usd_csa();
  json::object t2 = trade("T2", "float", 0.028);
  t2["csa"] = usd_csa();
  book["trades"] = json::array{t1, t2};
  legacy["book"] = book;

  json::object a = run(legacy);
  json::object b = run(legacy);  // rerun: deterministic modulo the wall-clock stamp
  a.erase("wall_us");
  b.erase("wall_us");
  EXPECT_EQ(json::serialize(json::value(a)), json::serialize(json::value(b)));

  // Same legacy book + a netting set alongside: the legacy fields must be untouched, byte for byte.
  json::object combined = legacy;
  combined["netting_sets"] = json::array{nset("CP-1", json::array{trade("T3", "fixed")})};
  json::object c = run(combined);
  ASSERT_TRUE(c.contains("netting_sets"));
  c.erase("netting_sets");
  c.erase("wall_us");
  EXPECT_EQ(json::serialize(json::value(c)), json::serialize(json::value(a)));
}

// Role/CSA resolution fails loudly, mirroring book_from_json's semantics.
TEST(ExposureNetting, UnmappedIndexAndUnknownCollateralCurrencyFailLoudly) {
  const cal::BundleProblem p = build_bundle();

  json::object no_role = base_request(p);
  no_role["curve_roles"] = json::object{};  // empty binding: USD-SOFR unmapped
  no_role["netting_sets"] = json::array{nset("CP-1", json::array{trade("T1", "fixed")})};
  EXPECT_THROW(run(no_role), std::invalid_argument);

  json::object bad_ccy = base_request(p);
  json::object s;
  s["id"] = "CP-1";
  json::object csa;
  csa["collateral_currency"] = "XXX";  // unknown -> empty discount index id
  s["csa"] = csa;
  s["trades"] = json::array{trade("T1", "fixed")};
  bad_ccy["netting_sets"] = json::array{s};
  EXPECT_THROW(run(bad_ccy), std::invalid_argument);
}

// (E3-F2, 2026-09-10) The book is AGED to every node: a 2y trade has zero exposure at every node at or past its
// maturity (before the fix every node repriced today's cashflow set: EPE 35k at 4..10y for a matured swap),
// PFE is a quantile of the POSITIVE exposure (>= 0 everywhere), and node 0 is today's MtM.
TEST(ExposureNetting, TheBookIsAgedToEveryNode) {
  const cal::BundleProblem p = build_bundle();
  json::object req = base_request(p);
  req["n_nodes"] = 6;  // t = 0, 1, 2, 3, 4, 5
  req["horizon_years"] = 5.0;
  req["netting_sets"] = json::array{nset("CP", json::array{trade("t2y", "fixed", 0.03, "2028-09-08")})};
  const json::object r = run(req);
  const auto& s = r.at("netting_sets").as_array()[0].as_object();
  const auto tv = s.at("node_time").as_array();
  const auto epe = s.at("epe").as_array(), ene = s.at("ene").as_array(), pfe = s.at("pfe").as_array();
  const double mtm = s.at("mtm").to_number<double>();
  ASSERT_EQ(epe.size(), 6u);
  EXPECT_NEAR(epe[0].to_number<double>(), std::max(mtm, 0.0), 1e-9 * std::max(1.0, std::abs(mtm)));
  EXPECT_NEAR(pfe[0].to_number<double>(), std::max(mtm, 0.0), 1e-9 * std::max(1.0, std::abs(mtm)));
  EXPECT_GT(epe[1].to_number<double>() + std::abs(ene[1].to_number<double>()), 0.0) << "alive at 1y";
  for (std::size_t j = 0; j < 6; ++j) {
    const double t = tv[j].to_number<double>();
    EXPECT_GE(pfe[j].to_number<double>(), 0.0) << "node " << t;
    if (t >= 2.5) {  // the trade matures 2028-09-08 = 2.01y; at the 2y node a 4-day stub is still alive
      EXPECT_EQ(epe[j].to_number<double>(), 0.0) << "matured at node t=" << t;
      EXPECT_EQ(ene[j].to_number<double>(), 0.0) << "matured at node t=" << t;
      EXPECT_EQ(pfe[j].to_number<double>(), 0.0) << "matured at node t=" << t;
    }
  }
}
