// E5 taxonomy: T6 regression (fails on the reverted bug) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// P12 REPRODUCTIONS of the two scale bugs found in the E7 stage-3 design review (2026-09-13). Every test here FAILS on
// today's code and passes once the bug is fixed (then it is the T6 pin: it fails on the reverted fix).
//
//   BUG 1  generate_risk (api/generate_risk.cpp null_completed_ladder) returns pinv(J_f)ᵀ·g where J = sess.jacobian()
//          is the RESIDUAL Jacobian dr/dx, and never applies the residual market scale D = diag(−∂r/∂q) that
//          BundleSession::risk_operator() applies (M = pinv(J)·D). For every real row, today_i = correct_i / D_i:
//          a banded quote is overstated by 1/band_decay, an FX forward by q·T.
//   BUG 2  calib_report's identifiability is diag(J·M) = diag(P·D) with P = J·pinv(J), not the projector diag(P) its
//          contract documents (h_ii in [0,1], trace = rank): a banded row reports decay·P_ii, an FX forward
//          P_ii/(q·T) -- which the [0,1] clamp HIDES for short forwards (q·T < 1) and exposes for long ones.
//
// The BUG 1 reference is the market delta itself: a central finite difference of the verb's OWN npv under a
// re-calibration with that one quote bumped. price_portfolio_risk(book).ladder matches it today (asserted as a
// passing precondition; BundleApi.RiskOperatorMatchesBumpAndRecalibrateOnBandedRows pins M the same way).
// The numbers in comments come from an independent numpy replica of these fixtures (derive.py in the review notes).

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/curve/curve_module.hpp"
#include "tolerances.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;
namespace cv = swaps::curve;
namespace api = swaps::api;
namespace json = boost::json;
namespace tol = swaps::tol;

namespace {

const std::vector<double> kFront{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
const std::vector<double> kKnots{0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0};
constexpr int kNk = 7;
constexpr int kBand = 4;        // the 4y par swap (row 4 of the domestic block)
constexpr int kFx4y = 7 + 4;    // the 4y FX forward (row 11 of the FX bundle)
constexpr double kDecay = 0.25;
constexpr double kSpot = 1.1;
// A central FD of a re-calibrated NPV: O(h²) truncation (derive.py: 5.6e-10 relative for the band row at h = 1e-5,
// 1.3e-10 for the FX row at h = 1e-6) plus the LM stop's NPV noise / 2h (~1e-10). Far inside the bugs' factors 4x.
constexpr double kFdRel = 1e-6;

double dom_line(double t) { return 0.03 + 0.002 * t; }   // domestic forward, linear in time
double for_line(double t) { return 0.02 + 0.001 * t; }   // foreign forward, linear in time

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}

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

cal::Instrument make_rate(double a, double b) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// F = spot · DF_foreign(T) / DF_domestic(T): pins the foreign curve (1) against the domestic one (0).
cal::Instrument make_fx_forward(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::FxForward;
  ins.fx_num = 1;
  ins.fx_den = 0;
  ins.fx_spot = kSpot;
  ins.fx_time = T;
  return ins;
}

void make_consistent(cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
}

Eigen::VectorXd dom_x() {
  Eigen::VectorXd x(kNk);
  for (int i = 0; i < kNk; ++i) x[i] = dom_line(kKnots[i]);
  return x;
}

// Square (7x7), full rank (cond ~41), self-consistent at the domestic line. Same fixture as consistent_risk_test.
cal::BundleProblem ois_bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite(kFront, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25));
  p.instruments.push_back(make_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T));
  make_consistent(p, dom_x());
  return p;
}

// The same bundle with the 4y swap quoted as a soft band: ±20 bp around its (self-consistent) mid, decay 0.25. The
// zero-residual fit sits at the mid, INSIDE the band, where the residual slope is the decay.
cal::BundleProblem banded_bundle() {
  cal::BundleProblem p = ois_bundle();
  cal::Instrument& b = p.instruments[kBand];
  b.band_lower = b.market - 20e-4;
  b.band_upper = b.market + 20e-4;
  b.band_decay = kDecay;
  return p;
}

Eigen::VectorXd fx_x() {
  Eigen::VectorXd x(2 * kNk);
  for (int i = 0; i < kNk; ++i) {
    x[i] = dom_line(kKnots[i]);
    x[kNk + i] = for_line(kKnots[i]);
  }
  return x;
}

// Two curves (domestic 0, foreign 1, 7 knots each), 14 instruments: the 7 domestic quotes + 7 FX forwards at the knot
// times. Square (14x14), full rank (cond ~579), self-consistent at both lines.
cal::BundleProblem fx_bundle() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite(kFront, kBack)});
  p.curves.push_back({.base = -1, .currency = 1, .regions = cv::flat_hermite(kFront, kBack)});
  p.instruments.push_back(make_rate(0.0, 0.25));
  p.instruments.push_back(make_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T));
  for (double T : kKnots) p.instruments.push_back(make_fx_forward(T));
  make_consistent(p, fx_x());
  return p;
}

json::object swap_position(double T, double K, double N, int curve = 0) {
  json::array fl, xl;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    json::object obs;
    obs["sub_start"] = json::array{prev};
    obs["sub_end"] = json::array{t};
    obs["tau_index"] = t - prev;
    json::object c;
    c["obs"] = std::move(obs);
    c["pay"] = t;
    c["tau_pay"] = t - prev;
    fl.push_back(std::move(c));
    json::object x;
    x["pay"] = t;
    x["tau"] = t - prev;
    xl.push_back(std::move(x));
    prev = t;
  }
  json::object po;
  po["notional"] = N;
  po["fixed_rate"] = K;
  po["fwd_curve"] = curve;
  po["disc_curve"] = curve;
  po["fixed_curve"] = curve;
  po["float_coupons"] = std::move(fl);
  po["fixed_coupons"] = std::move(xl);
  return po;
}

// 5y payer at 4 % (N = 1) + 9y payer at 4.5 % (N = -0.5) on the domestic curve; with_foreign adds a 5y payer at 2.5 %
// on the foreign curve, so the book loads on the FX-pinned knots.
json::object book_json(bool with_foreign = false) {
  json::array pos;
  pos.push_back(swap_position(5.0, 0.040, 1.0));
  pos.push_back(swap_position(9.0, 0.045, -0.5));
  if (with_foreign) pos.push_back(swap_position(5.0, 0.025, 1.0, /*curve=*/1));
  json::object b;
  b["positions"] = std::move(pos);
  return b;
}

pf::MultiCurveBook book_struct(const json::object& b) { return api::book_from_json(json::value(b)); }

// A NEGLIGIBLE regulariser (row weight 1e-8, mu = 1e-16) so generate_risk does not apply its LIGHT tension floor: a
// bumped quote moves the curve off the line, where the floor would add its own dx/dq response to the finite
// difference. mu is ~12 orders below the smallest JᵀJ eigenvalue of either fixture (derive.py: sigma_min² ~ 1e-3 / 1e-5).
json::object negligible_reg(int n_curves) {
  json::object r;
  r["lambda"] = 1e-8;
  json::array cs;
  for (int c = 0; c < n_curves; ++c) cs.push_back(c);
  r["curves"] = std::move(cs);
  r["tension"] = true;
  r["sigma"] = 0.0;
  return r;
}

json::object generate_risk_request(const json::object& book, const std::vector<cal::BundleProblem>& bundles,
                                   const json::object* reg = nullptr) {
  json::object g;
  g["book"] = book;
  json::array arr;
  for (const auto& b : bundles) arr.push_back(api::bundle_to_json(b));
  g["bundles"] = std::move(arr);
  if (reg) g["regularize"] = *reg;
  json::object req;
  req["generate_risk"] = std::move(g);
  return req;
}

json::object calib_report_request(const cal::BundleProblem& p, const Eigen::VectorXd& x0) {
  json::object cr;
  cr["bundle"] = api::bundle_to_json(p);
  json::array xa;
  for (int i = 0; i < x0.size(); ++i) xa.push_back(x0[i]);
  cr["x0"] = std::move(xa);
  json::object req;
  req["calib_report"] = std::move(cr);
  return req;
}

json::object run(const json::object& req) { return json::parse(api::run_json(req)).as_object(); }

double num(const json::object& o, const char* k) { return o.at(k).to_number<double>(); }

Eigen::VectorXd nums(const json::value& v) {
  const json::array& a = v.as_array();
  Eigen::VectorXd e(static_cast<Eigen::Index>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i) e[static_cast<Eigen::Index>(i)] = a[i].to_number<double>();
  return e;
}

void expect_close(const Eigen::VectorXd& got, const Eigen::VectorXd& want, double rel, const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (Eigen::Index i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], rel * std::max(1.0, std::abs(want[i]))) << what << " row " << i;
}

// generate_risk's own top-level npv for a one-bundle request.
double verb_npv(const json::object& book, const cal::BundleProblem& p, const json::object& reg) {
  const json::object r = run(generate_risk_request(book, {p}, &reg));
  if (r.contains("error")) {
    ADD_FAILURE() << json::serialize(r);
    return std::numeric_limits<double>::quiet_NaN();
  }
  return num(r, "npv");
}

// dP/dm_row by a central difference of the verb's npv, re-calibrating to the bumped market each side.
double verb_market_delta(const json::object& book, const cal::BundleProblem& p, int row, double h,
                         const json::object& reg) {
  cal::BundleProblem up = p, dn = p;
  up.instruments[static_cast<std::size_t>(row)].market += h;
  dn.instruments[static_cast<std::size_t>(row)].market -= h;
  return (verb_npv(book, up, reg) - verb_npv(book, dn, reg)) / (2.0 * h);
}

}  // namespace

// BUG 1, banded leg. The 4y swap is a soft band (decay 0.25) that still pins its knot: the zero-residual fit reprices
// it exactly, so its market delta is the SAME as a hard pin's (8.15077625620616 per unit rate, replica).
// Today generate_risk reports 32.60310502482464 = 8.15077625620616 / 0.25 on that row; every other row agrees; and
// ladder_dv01 is ~2.5048e-3 instead of ~5.952e-5.
TEST(RiskScaleRepro, GenerateRiskBandedLadderEntryIsTheMarketDelta) {
  const cal::BundleProblem hard = ois_bundle();
  const cal::BundleProblem soft = banded_bundle();
  const json::object book = book_json();
  const pf::MultiCurveBook bk = book_struct(book);
  const json::object reg = negligible_reg(1);

  // Preconditions (pass today): the native session lands on the line with the banded row inside its band.
  api::BundleSession ref(soft);
  ref.calibrate(dom_x());
  ASSERT_EQ(ref.result().rank_deficiency, 0);
  ASSERT_LT((ref.x() - dom_x()).cwiseAbs().maxCoeff(), tol::step_tol);
  ASSERT_TRUE(ref.quote_diagnostics()[kBand].as_object().at("in_band").as_bool());
  const api::PortfolioRisk rk = ref.price_portfolio_risk(bk);

  // The market delta by finite difference through the verb (h = 0.1 bp stays inside the ±20 bp band).
  const double fd = verb_market_delta(book, soft, kBand, 1e-5, reg);
  EXPECT_NEAR(fd, 8.15077625620616, kFdRel * 8.15) << "replica value (derive.py FD: 8.150776260738844)";
  EXPECT_NEAR(rk.ladder[kBand], fd, kFdRel * std::abs(fd)) << "price_portfolio_risk IS the market delta (passes today)";

  // T5 identity (passes today): an in-band, zero-residual band has the hard pin's market delta on every row.
  api::BundleSession ref_hard(hard);
  ref_hard.calibrate(dom_x());
  // tol::parity_jacobian: two pinv(J)·D ladders at the same x, J differing by a row scale (cond ~41 -> ~1e-14).
  expect_close(rk.ladder, ref_hard.price_portfolio_risk(bk).ladder, tol::parity_jacobian, "banded vs hard delta");

  // THE BUG (fails today on row 4 and on ladder_dv01).
  const json::object r = run(generate_risk_request(book, {soft}, &reg));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::object& b = r.at("bundles").as_array()[0].as_object();
  ASSERT_EQ(b.at("n_synthetic").to_number<int>(), 0);
  const Eigen::VectorXd lad = nums(b.at("ladder"));
  ASSERT_EQ(lad.size(), kNk);
  EXPECT_NEAR(lad[kBand], fd, kFdRel * std::abs(fd))
      << "generate_risk's banded row is the market delta / band_decay (today 32.60310502482464 vs 8.15077625620616)";
  // tol::parity_jacobian: the verb and the session evaluate the same pinv(J)-based operator at x agreeing to LM precision.
  expect_close(lad, rk.ladder, tol::parity_jacobian, "generate_risk vs price_portfolio_risk (banded)");
  EXPECT_NEAR(num(b, "ladder_dv01"), 1e-4 * rk.ladder.sum(), 1e-4 * tol::parity_jacobian * rk.ladder.cwiseAbs().sum())
      << "today ~2.5048e-3 (inflated by 3 x 8.1508 on the banded row) instead of ~5.952e-5";
}

// BUG 1, FX leg. The 4y FX forward's market delta is -1.3380751025225255 (replica; FD -1.3380751023445159). Today
// generate_risk reports -6.177410257698535 = that x q·T = x 1.1541598535935957 x 4. Every FX row k is scaled by q_k·T_k
// = {0.2757, 0.5529, 1.1117, 2.2491, 4.6166, 8.4637, 12.781} (0.25y..10y); the 7 domestic rows agree.
TEST(RiskScaleRepro, GenerateRiskFxForwardLadderEntryIsTheMarketDelta) {
  const cal::BundleProblem p = fx_bundle();
  ASSERT_EQ(p.n_knots(), 2 * kNk);
  ASSERT_EQ(p.n_residuals(), 2 * kNk);
  const json::object book = book_json(/*with_foreign=*/true);
  const pf::MultiCurveBook bk = book_struct(book);
  const json::object reg = negligible_reg(2);

  // Preconditions (pass today).
  api::BundleSession ref(p);
  ref.calibrate(fx_x());
  ASSERT_EQ(ref.result().rank_deficiency, 0);
  ASSERT_LT((ref.x() - fx_x()).cwiseAbs().maxCoeff(), tol::step_tol);
  const api::PortfolioRisk rk = ref.price_portfolio_risk(bk);
  const double fd = verb_market_delta(book, p, kFx4y, 1e-6, reg);
  EXPECT_NEAR(fd, -1.3380751025225255, kFdRel * 1.34) << "replica value";
  EXPECT_NEAR(rk.ladder[kFx4y], fd, kFdRel * std::abs(fd)) << "price_portfolio_risk IS the market delta (passes today)";

  // THE BUG (fails today on every FX row, 7..13).
  const json::object r = run(generate_risk_request(book, {p}, &reg));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::object& b = r.at("bundles").as_array()[0].as_object();
  ASSERT_EQ(b.at("n_synthetic").to_number<int>(), 0);
  const Eigen::VectorXd lad = nums(b.at("ladder"));
  ASSERT_EQ(lad.size(), 2 * kNk);
  EXPECT_NEAR(lad[kFx4y], fd, kFdRel * std::abs(fd))
      << "generate_risk's FX row is the market delta x q·T (today -6.177410257698535 vs -1.3380751025225255)";
  // tol::parity_jacobian: same operator, same x to LM precision; cond(J) ~579 still leaves ~1e-13 relative.
  expect_close(lad, rk.ladder, tol::parity_jacobian, "generate_risk vs price_portfolio_risk (FX)");
}

// BUG 2, banded leg. A square full-rank J has P = J·pinv(J) = I, so every quote independently pins its pillar and the
// documented hat diagonal is 1 -- a band changes the residual's SCALE, not what the quote identifies.
// Today calib_report reports 0.25 (= decay) for the banded row, and Σ identifiability = 6.25 instead of rank(J) = 7.
TEST(RiskScaleRepro, CalibReportIdentifiabilityOfAnIndependentBandedPinIsOne) {
  const cal::BundleProblem hard = ois_bundle();
  const cal::BundleProblem soft = banded_bundle();
  const json::object rh = run(calib_report_request(hard, dom_x()));
  const json::object rs = run(calib_report_request(soft, dom_x()));
  ASSERT_FALSE(rh.contains("error")) << json::serialize(rh);
  ASSERT_FALSE(rs.contains("error")) << json::serialize(rs);
  ASSERT_EQ(rh.at("rank_deficiency").to_number<int>(), 0);
  ASSERT_EQ(rs.at("rank_deficiency").to_number<int>(), 0);
  const json::array& qh = rh.at("quotes").as_array();
  const json::array& qs = rs.at("quotes").as_array();
  ASSERT_EQ(qh.size(), static_cast<std::size_t>(kNk));
  ASSERT_EQ(qs.size(), static_cast<std::size_t>(kNk));
  ASSERT_TRUE(qs[kBand].as_object().at("soft").as_bool());
  ASSERT_TRUE(qs[kBand].as_object().at("in_band").as_bool());

  // tol::parity_jacobian: diag(J·pinv(J)) of a cond-41 square matrix is 1 to ~1e-14 (and the clamp caps it at 1).
  // Control (passes today): the hard bundle.
  for (std::size_t i = 0; i < qh.size(); ++i)
    EXPECT_NEAR(num(qh[i].as_object(), "identifiability"), 1.0, tol::parity_jacobian) << "hard row " << i;
  // THE BUG (fails today: row 4 = 0.25, sum = 6.25).
  double sum = 0.0;
  for (std::size_t i = 0; i < qs.size(); ++i) {
    const double h = num(qs[i].as_object(), "identifiability");
    sum += h;
    EXPECT_NEAR(h, 1.0, tol::parity_jacobian) << "banded bundle row " << i << (i == kBand ? " (today: 0.25 = decay)" : "");
  }
  EXPECT_NEAR(sum, static_cast<double>(kNk), kNk * tol::parity_jacobian) << "trace(H) = rank(J) = 7 (today 6.25)";
}

// BUG 2, FX leg. Same square full-rank argument: every FX forward's hat diagonal is 1. Today the unclamped diagonal is
// P_ii/(q·T) = {3.627, 1.809, 0.8995, 0.4446, 0.2166, 0.1182, 0.0782} for 0.25y..10y; the clamp turns the first two
// into 1 (hiding the bug), the 1y..10y rows report < 1.
TEST(RiskScaleRepro, CalibReportIdentifiabilityOfFxForwardPinsIsOne) {
  const cal::BundleProblem p = fx_bundle();
  const json::object r = run(calib_report_request(p, fx_x()));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  ASSERT_EQ(r.at("rank_deficiency").to_number<int>(), 0);
  const json::array& q = r.at("quotes").as_array();
  ASSERT_EQ(q.size(), static_cast<std::size_t>(2 * kNk));
  // tol::parity_jacobian: diag(J·pinv(J)) at cond ~579 is 1 to ~1e-13.
  for (std::size_t i = 0; i < q.size(); ++i)
    EXPECT_NEAR(num(q[i].as_object(), "identifiability"), 1.0, tol::parity_jacobian)
        << "row " << i << (i >= static_cast<std::size_t>(kNk) ? " (FX forward)" : " (domestic)");
}
