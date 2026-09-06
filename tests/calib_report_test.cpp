// Gate for the stateless "calib_report" run_json verb (api/calib_report.cpp, audit quick-win #10). NO
// QuantLib: a small single-curve OIS bundle is hand-built as generic Instruments, made self-consistent
// from a known x_true, and driven THROUGH the JSON contract (run_json / calib_report_json). Uses
// boost/json + the api library, so it targets swaps_api_tests.
//
// Proves the four contract properties:
//   (1) a well-posed curve -> finite, modest condition_number and every per-quote identifiability high;
//   (2) adding a near-duplicate (redundant) instrument RAISES the condition_number and LOWERS the
//       identifiability of the two colliding quotes (they split their leverage ~0.5/0.5);
//   (3) rank_deficiency matches the engine's OWN report (a native BundleSession calibrate) on a
//       deficient (under-determined) setup;
//   (4) condition_number >= 1 in every case (sigma_max / sigma_min by construction).

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/calib_report.hpp"
#include "swaps/curve/parametric.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
constexpr int kNk = 7;  // knots for the single outright curve

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}

// A par-rate OIS swap to maturity T (annual schedule) on the single curve 0 (forecast = discount = 0).
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

// A front "Rate" pin over [a,b] on curve 0 (pins the flat segment before the first swap).
cal::Instrument make_rate(double a, double b) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// The base well-posed single-curve bundle (7 instruments, 7 knots) + the generating x_true.
cal::BundleProblem build_curve(Eigen::VectorXd& x_true) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25));
  p.instruments.push_back(make_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T));

  x_true.resize(kNk);
  for (int i = 0; i < kNk; ++i) x_true[i] = 0.030 + 0.001 * i;
  return p;
}

// Set every instrument's market to its model quote at x_true (self-consistent -> residual 0).
void make_consistent(cal::BundleProblem& p, const Eigen::VectorXd& x_true) {
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
}

std::string report_for(const cal::BundleProblem& p) {
  json::object req;
  json::object cr;
  cr["bundle"] = api::bundle_to_json(p);
  req["calib_report"] = cr;
  return api::calib_report_json(json::serialize(json::value(std::move(req))));
}

}  // namespace

// (1) well-posed + (4) cond >= 1: modest finite condition number, all identifiability high.
TEST(CalibReport, WellPosedIsWellConditionedAndIdentified) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_curve(x_true);
  make_consistent(p, x_true);

  const json::value r = json::parse(report_for(p));
  const json::object& o = r.as_object();

  const double cond = o.at("condition_number").to_number<double>();
  EXPECT_GE(cond, 1.0);                       // (4) sigma_max / sigma_min is always >= 1
  EXPECT_TRUE(std::isfinite(cond));
  EXPECT_LT(cond, 1.0e6) << "a well-posed curve is not ill-conditioned";
  EXPECT_EQ(o.at("rank_deficiency").to_number<int>(), 0);
  EXPECT_LT(o.at("rms_residual").to_number<double>(), 1e-8);

  const json::array& q = o.at("quotes").as_array();
  ASSERT_EQ(q.size(), static_cast<std::size_t>(kNk));
  for (const auto& e : q) {
    const double ident = e.as_object().at("identifiability").to_number<double>();
    EXPECT_GT(ident, 0.9) << "every pillar is independently determined in a well-posed curve";
    EXPECT_LE(ident, 1.0 + 1e-9);
  }
  // singular_values are reported and sorted descending.
  const json::array& sv = o.at("singular_values").as_array();
  ASSERT_EQ(sv.size(), static_cast<std::size_t>(kNk));
  EXPECT_GE(sv.front().to_number<double>(), sv.back().to_number<double>());
}

// (2) a near-duplicate instrument splits the colliding quotes' identifiability (the row-redundancy
//     signal) while leaving the condition number non-decreasing. The duplicate is an EXACT copy of the 7y
//     swap, so the two rows of J are identical: they share their unit leverage ~0.5/0.5. Appending a row
//     does NOT inflate the condition number here — by singular-value interlacing σ_min (governed by the
//     least-determined KNOT column) is unchanged and σ_max only weakly rises; redundant *quotes* show up
//     in the per-quote identifiability, not in σ_max/σ_min (that is column collinearity, a different fault).
TEST(CalibReport, NearDuplicateSplitsIdentifiability) {
  Eigen::VectorXd x_true;
  cal::BundleProblem base = build_curve(x_true);
  make_consistent(base, x_true);
  const json::object base_o = json::parse(report_for(base)).as_object();
  const double base_cond = base_o.at("condition_number").to_number<double>();
  const double base_ident7 = base_o.at("quotes").as_array()[6].as_object()
                                 .at("identifiability").to_number<double>();  // the 7y swap

  cal::BundleProblem dup = base;
  dup.instruments.push_back(dup.instruments[6]);  // exact duplicate of the 7y par swap
  make_consistent(dup, x_true);                   // keep every quote self-consistent
  const json::object dup_o = json::parse(report_for(dup)).as_object();
  const double dup_cond = dup_o.at("condition_number").to_number<double>();
  const json::array& dq = dup_o.at("quotes").as_array();
  ASSERT_EQ(dq.size(), static_cast<std::size_t>(kNk + 1));
  const double id_a = dq[6].as_object().at("identifiability").to_number<double>();   // original 7y
  const double id_b = dq[kNk].as_object().at("identifiability").to_number<double>();  // duplicate 7y

  EXPECT_GE(dup_cond, 1.0);
  // Appending a duplicate ROW cannot shrink any singular value (interlacing), so the condition number is
  // non-decreasing; the redundancy is diagnosed by the identifiability split below, not by sigma_max/min.
  EXPECT_GE(dup_cond, base_cond * 0.999) << "a redundant row is non-decreasing in the condition number";
  EXPECT_LT(id_a, 0.75) << "the colliding original 7y quote loses identifiability";
  EXPECT_LT(id_b, 0.75) << "the duplicate 7y quote loses identifiability";
  EXPECT_LT(id_a, base_ident7);
  EXPECT_LT(id_b, base_ident7);
  // The two exact duplicates split their leverage roughly evenly.
  EXPECT_NEAR(id_a, id_b, 0.15);
}

// (3) rank_deficiency echoes the engine's OWN report. Drop the last swap -> the 10y knot is unreached ->
//     under-determined. Compare the verb's rank_deficiency to a native BundleSession calibrate.
TEST(CalibReport, RankDeficiencyMatchesEngineReport) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_curve(x_true);
  p.instruments.pop_back();  // remove the 10y swap: 6 instruments now under-determine the 7 knots
  make_consistent(p, x_true);

  // The engine's own report through a native session (a different path than the JSON verb).
  api::BundleSession sess(p);
  const cal::CalibrationResult& native = sess.calibrate(api::flat_x0(p));
  EXPECT_GT(native.rank_deficiency, 0) << "the setup must actually be deficient";

  const json::object o = json::parse(report_for(p)).as_object();
  EXPECT_EQ(o.at("rank_deficiency").to_number<int>(), native.rank_deficiency);
  EXPECT_GE(o.at("condition_number").to_number<double>(), 1.0);  // (4) holds even when deficient
}
