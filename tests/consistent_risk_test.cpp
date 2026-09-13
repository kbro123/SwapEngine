// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// PIN-BEFORE-REFACTOR gate for the stateless `generate_risk` and `calib_report` verbs (E7 stage 3 lifts both into
// calibration-layer templates). Everything is driven through the run_json OBJECT seam -- the one entry the
// refactor keeps -- and pins behaviour that is CORRECT today and must not move.
//
// NOT pinned here: the two scale bugs from the stage-3 design review (generate_risk's ladder ignores the residual
// market scale D; calib_report's identifiability is diag(P·D), not diag(P)). They are reproduced, FAILING, in
// risk_scale_repro_test.cpp. Every generate_risk fixture below is UNBANDED (D = I, where both bugs vanish), and the
// calib_report banded case asserts only the echoed diagnostics, never `identifiability`.
//
// Fixture: one outright OIS curve, flat front {0.25, 0.5} + Hermite back {1, 2, 4, 7, 10} (7 knots); 2 front Rate
// pins + 1/2/4/7/10y annual par swaps (7 instruments: square, full rank, cond(J) ~ 41), self-consistent at a
// forward LINEAR IN TIME, f(t) = 3% + 0.2%·t. A linear forward carries no tension energy (Flat pieces have f'' = 0
// and a Bessel-tangent Hermite reproduces a line exactly), so the light tension floor generate_risk puts on its
// calibrations leaves the optimum AT x_true. Every path below therefore lands on the same curve whatever its seed
// or smoothing, so a refactor may change those without breaking a pin.
// The literals come from an INDEPENDENT numpy replica of this curve / quote / book (derive.py beside this file in the
// review notes: the same region maths re-implemented, complex-step derivatives), not from engine output.

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/calib_report.hpp"
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
constexpr int kNk = 7;  // knots of the base fixture
constexpr int kNr = 7;  // instruments of the base fixture

// ASSUMPTIONS E1: a Boost.JSON serialize->parse returns ~9.5 % of doubles 1 ULP off (worst 2.2e-16 relative). A
// value that crossed the text seam at most twice (calib_report re-parses quote_diagnostics_json(); the test parses
// the response) is compared at 4 relative ULPs -- never at zero tolerance.
constexpr double kUlps = 4.0;

double fwd_line(double t) { return 0.03 + 0.002 * t; }

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}

// A par-rate OIS swap to maturity T (annual schedule), forecast = discount = curve 0.
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

// x_true: the linear forward at every region knot, in state order.
Eigen::VectorXd line_x(const cal::BundleProblem& p) {
  Eigen::VectorXd x(p.n_knots());
  int i = 0;
  for (const auto& r : p.curves[0].regions)
    for (double t : r.knots) x[i++] = fwd_line(t);
  return x;
}

void make_consistent(cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
}

// The base fixture. tail_knot appends a single-knot Flat region {15y}: a Flat region ignores the incoming boundary,
// so its forward lives only on (10y, 15y] and NO instrument (all <= 10y) sees it -- column 7 of J is exactly zero.
cal::BundleProblem ois_bundle(bool tail_knot = false) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite(kFront, kBack)});
  if (tail_knot) p.curves[0].regions.push_back(cv::CurveModule{{15.0}, cv::Scheme::Flat});
  p.instruments.push_back(make_rate(0.0, 0.25));
  p.instruments.push_back(make_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T));
  make_consistent(p, line_x(p));
  return p;
}

// A payer-of-fixed annual swap position on `curve` (forecast = discount = fixed-leg discount), as book JSON.
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

// Book: 5y payer at 4 % (N = 1) and 9y payer at 4.5 % (N = -0.5). with_12y adds a 12y payer at 5 % whose 10y..12y
// cashflows load on the tail knot.
json::object book_json(bool with_12y = false) {
  json::array pos;
  pos.push_back(swap_position(5.0, 0.040, 1.0));
  pos.push_back(swap_position(9.0, 0.045, -0.5));
  if (with_12y) pos.push_back(swap_position(12.0, 0.050, 1.0));
  json::object b;
  b["positions"] = std::move(pos);
  return b;
}

// The SAME decoder the verb uses, so the native reference prices the identical book.
pf::MultiCurveBook book_struct(const json::object& b) { return api::book_from_json(json::value(b)); }

json::object generate_risk_request(const json::object& book, const std::vector<cal::BundleProblem>& bundles) {
  json::object g;
  g["book"] = book;
  json::array arr;
  for (const auto& b : bundles) arr.push_back(api::bundle_to_json(b));
  g["bundles"] = std::move(arr);
  json::object req;
  req["generate_risk"] = std::move(g);
  return req;
}

json::object calib_report_request(const cal::BundleProblem& p, const Eigen::VectorXd* x0) {
  json::object cr;
  cr["bundle"] = api::bundle_to_json(p);
  if (x0) {
    json::array xa;
    for (int i = 0; i < x0->size(); ++i) xa.push_back((*x0)[i]);
    cr["x0"] = std::move(xa);
  }
  json::object req;
  req["calib_report"] = std::move(cr);
  return req;
}

json::object run(const json::object& req) { return json::parse(api::run_json(req)).as_object(); }

double num(const json::object& o, const char* k) { return o.at(k).to_number<double>(); }

std::vector<double> nums(const json::value& v) {
  std::vector<double> out;
  for (const auto& e : v.as_array()) out.push_back(e.to_number<double>());
  return out;
}

::testing::AssertionResult within_ulps(double a, double b) {
  const double bound = kUlps * std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b));
  if (std::abs(a - b) <= bound) return ::testing::AssertionSuccess();
  std::ostringstream s;
  s << std::setprecision(17) << a << " vs " << b << " differ by " << std::abs(a - b) << " > " << bound;
  return ::testing::AssertionFailure() << s.str();
}

// Element-wise |got - want| <= rel * max(1, |want|).
void expect_close(const std::vector<double>& got, const Eigen::VectorXd& want, double rel, const std::string& what) {
  ASSERT_EQ(got.size(), static_cast<std::size_t>(want.size())) << what;
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[static_cast<Eigen::Index>(i)], rel * std::max(1.0, std::abs(want[static_cast<Eigen::Index>(i)])))
        << what << " row " << i;
}

Eigen::VectorXd to_eigen(const std::vector<double>& v) {
  Eigen::VectorXd e(static_cast<Eigen::Index>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) e[static_cast<Eigen::Index>(i)] = v[i];
  return e;
}

}  // namespace

// ================================================================================================================
// Fixture precondition. If this fails, every "lands on x_true" argument below is void -- fix the fixture first.
// ================================================================================================================

// The anchor recipe generate_risk uses today (flat market-implied seed + the LIGHT tension floor) lands on x_true:
// the linear forward is in the tension operator's null space and the square bundle reprices exactly there.
TEST(ConsistentRisk, FixtureLinearForwardSurvivesTheLightTensionFloor) {
  const cal::BundleProblem p = ois_bundle();
  ASSERT_EQ(p.n_knots(), kNk);
  ASSERT_EQ(p.n_residuals(), kNr);
  api::BundleSession s(p);
  const cal::CalibrationResult& res =
      s.calibrate(api::flat_x0(p), cal::smoothing_preset(cal::Smoothing::Light, p.n_curves()));
  EXPECT_TRUE(res.converged) << res.status;
  EXPECT_EQ(res.rank_deficiency, 0);
  // tol::step_tol: the committed-x contract -- two solves of the same zero-residual problem agree on x to it.
  EXPECT_LT((s.x() - line_x(p)).cwiseAbs().maxCoeff(), tol::step_tol) << "the tension floor pulled x off the line";
}

// ================================================================================================================
// generate_risk
// ================================================================================================================

// Identical bundles -> identical ladders, no synthetic pillars, and every per-bundle npv / pv01 equal to the
// top-level invariants (the re-level of bundles 1 and 2 onto the anchor is a no-op up to rounding).
TEST(ConsistentRisk, IdenticalBundlesGiveIdenticalLaddersAndTheTopLevelInvariants) {
  const cal::BundleProblem p = ois_bundle();
  const json::object book = book_json();
  const json::object r = run(generate_risk_request(book, {p, p, p}));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  EXPECT_EQ(r.at("n").to_number<int>(), 2);
  const json::array& bs = r.at("bundles").as_array();
  ASSERT_EQ(bs.size(), 3u);

  api::BundleSession ref(p);
  ref.calibrate(line_x(p));
  const pf::MultiCurveBook bk = book_struct(book);
  const api::PortfolioRisk rk = ref.price_portfolio_risk(bk);
  // Two calibrations of bitwise-near quotes agree on x to tol::step_tol (the committed-x contract), so an NPV-like
  // number moves by at most step_tol·||dP/dx||_1. Expected agreement is ~1e-15; this is the principled ceiling.
  const double inv_tol = tol::step_tol * rk.curve_grad.lpNorm<1>();
  const double npv = num(r, "npv"), pv01 = num(r, "pv01");
  EXPECT_NEAR(npv, ref.price_portfolio(bk).npv, inv_tol);
  EXPECT_NEAR(pv01, ref.price_portfolio(bk).pv01, inv_tol);

  const std::vector<double> l0 = nums(bs[0].as_object().at("ladder"));
  ASSERT_EQ(l0.size(), static_cast<std::size_t>(kNr));
  for (std::size_t k = 0; k < bs.size(); ++k) {
    const json::object& b = bs[k].as_object();
    EXPECT_EQ(b.at("n_residuals").to_number<int>(), kNr) << k;
    EXPECT_EQ(b.at("n_synthetic").to_number<int>(), 0) << k;
    EXPECT_TRUE(b.at("synthetic").as_array().empty()) << k;
    EXPECT_TRUE(b.at("synthetic_knot").as_array().empty()) << k;
    EXPECT_NEAR(num(b, "npv"), npv, inv_tol) << k;
    EXPECT_NEAR(num(b, "pv01"), pv01, inv_tol) << k;
    const std::vector<double> lk = nums(b.at("ladder"));
    // tol::parity_jacobian: the ladder is pinv(J)ᵀ·g at two x agreeing to LM precision; cond(J) ~ 41 amplifies the
    // ~1e-15 x-difference to ~1e-13 relative, well inside the Jacobian-parity bound.
    expect_close(lk, to_eigen(l0), tol::parity_jacobian, "bundle " + std::to_string(k) + " vs bundle 0");
    // ladder_dv01 == 1bp · Σ(ladder + synthetic): an identity; the slack is the summation's rounding plus E1 ULPs.
    double s = 0.0, sabs = 0.0;
    for (double v : lk) { s += v; sabs += std::abs(v); }
    EXPECT_NEAR(num(b, "ladder_dv01"), 1e-4 * s, 1e-4 * 8.0 * kUlps * std::numeric_limits<double>::epsilon() * sabs) << k;
  }
}

// T3: a full-rank UNBANDED bundle's ladder equals the session's native price_portfolio_risk(book).ladder. (With D = I
// the risk-operator scale bug is invisible; the banded version FAILS today -- risk_scale_repro_test.cpp.)
TEST(ConsistentRisk, FullRankUnbandedLadderEqualsPricePortfolioRisk) {
  const cal::BundleProblem p = ois_bundle();
  const json::object book = book_json();
  const json::object r = run(generate_risk_request(book, {p}));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::object& b = r.at("bundles").as_array()[0].as_object();

  api::BundleSession ref(p);  // a different recipe on purpose (x_true seed, no penalty): same curve, other path
  ref.calibrate(line_x(p));
  const api::PortfolioRisk rk = ref.price_portfolio_risk(book_struct(book));
  // tol::parity_jacobian: see IdenticalBundles (pinv-derived quantity at two x agreeing to LM precision).
  expect_close(nums(b.at("ladder")), rk.ladder, tol::parity_jacobian, "generate_risk vs price_portfolio_risk");
  EXPECT_EQ(b.at("n_synthetic").to_number<int>(), 0);
}

// T5: the same ladder, npv and pv01 against the independent numpy replica (derive.py). ladder = J⁻ᵀ·g with J = dq/dx
// and g = dP/dx by complex step at x_true; pv01 = 1bp · Σ g (the parallel-knot directional derivative).
TEST(ConsistentRisk, UnbandedLadderMatchesTheIndependentReplica) {
  const cal::BundleProblem p = ois_bundle();
  const json::object book = book_json();
  const json::object r = run(generate_risk_request(book, {p}));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::object& b = r.at("bundles").as_array()[0].as_object();

  Eigen::VectorXd want(kNr);
  want << -1.6444709677043923, -3.4004722474388043, 9.100545051123838, -7.349208818217408, 8.15077625620616,
          -2.390229250708069, -1.8717429389932179;
  // tol::parity_jacobian: the replica solves J⁻ᵀg in numpy (cond ~41) against an LM-converged x; both limits are
  // ~1e-13 relative. (A replica/engine MODEL mismatch shows up at >= 1e-6, far outside.)
  expect_close(nums(b.at("ladder")), want, tol::parity_jacobian, "ladder vs replica");
  // |ΔP| <= step_tol·||g||_1 with ||g||_1 = 3.44 (replica) -> 3.4e-9: the committed-x ceiling (expected ~1e-15).
  EXPECT_NEAR(num(r, "npv"), 0.0013371078405715053, tol::step_tol * 3.44);
  EXPECT_NEAR(num(r, "pv01"), 7.554546790346137e-05, tol::step_tol * 3.44 * 1e-4);
}

// An unreached knot becomes EXACTLY ONE synthetic pillar, labelled with that knot, carrying the book's sensitivity to
// it; the real-instrument head equals the rank-safe (min-norm COD) price_portfolio_risk ladder.
TEST(ConsistentRisk, UnreachedKnotBecomesExactlyOneSyntheticPillar) {
  const cal::BundleProblem p = ois_bundle(/*tail_knot=*/true);
  ASSERT_EQ(p.n_knots(), kNk + 1);
  ASSERT_EQ(p.n_residuals(), kNr);
  const json::object book = book_json(/*with_12y=*/true);
  const json::object r = run(generate_risk_request(book, {p}));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::object& b = r.at("bundles").as_array()[0].as_object();

  EXPECT_EQ(b.at("n_residuals").to_number<int>(), kNr);
  ASSERT_EQ(b.at("n_synthetic").to_number<int>(), 1);
  const json::array& sk = b.at("synthetic_knot").as_array();
  ASSERT_EQ(sk.size(), 1u);
  EXPECT_EQ(sk[0].to_number<int>(), 7) << "the tail knot is state index 7";
  const std::vector<double> syn = nums(b.at("synthetic"));
  ASSERT_EQ(syn.size(), 1u);
  const std::vector<double> head = nums(b.at("ladder"));
  ASSERT_EQ(head.size(), static_cast<std::size_t>(kNr));

  // Native mirror of the anchor calibration: the unseen knot sits at the SEED (calibrate_with's seed-anchored
  // completion), so the reference must use the same seed. The light floor cannot see it either (Flat region: no
  // tension energy), so both x agree bitwise.
  api::BundleSession ref(p);
  const cal::CalibrationResult& res =
      ref.calibrate(api::flat_x0(p), cal::smoothing_preset(cal::Smoothing::Light, p.n_curves()));
  ASSERT_EQ(res.rank_deficiency, 1);
  ASSERT_NEAR(ref.x()[7], api::flat_x0(p)[7], tol::step_tol) << "the unseen knot is anchored at the seed";
  ASSERT_LT((ref.x().head(kNk) - line_x(p).head(kNk)).cwiseAbs().maxCoeff(), tol::step_tol);
  const api::PortfolioRisk rk = ref.price_portfolio_risk(book_struct(book));

  // J's column 7 is exactly zero -> the null direction is ±e_7 -> the synthetic pillar is ±dP/dx_7 (SVD sign is free).
  // tol::parity_jacobian: an SVD null vector of a matrix with an exactly-zero column (rounding-level mixing only).
  EXPECT_NEAR(std::abs(syn[0]), std::abs(rk.curve_grad[7]), tol::parity_jacobian * std::max(1.0, std::abs(rk.curve_grad[7])));
  // The head is J_red⁻ᵀ·g_red, which is also what the COD min-norm operator gives (row 7 of pinv(J) is zero).
  expect_close(head, rk.ladder, tol::parity_jacobian, "synthetic-completed head vs price_portfolio_risk");

  // T5 literals (derive.py: seed level = mean outright quote = 0.034000257476194555 at knot 7).
  EXPECT_NEAR(std::abs(syn[0]), 1.3473588964948986, tol::parity_jacobian * 1.35);
  Eigen::VectorXd want(kNr);
  want << -1.6379644771004642, -3.387007621594446, 9.070360622793322, -7.304712455466383, 8.189698944258165,
          -2.210684920947077, 6.689004854548187;
  expect_close(head, want, tol::parity_jacobian, "head vs replica");
}

// RE-LEVEL: bundle[1] = bundle[0] with every market shifted +25 bp. generate_risk overwrites bundle[1]'s quotes with
// the anchor curve's model quotes, so bundle[1] describes the anchor curve: its npv / pv01 / ladder equal bundle[0]'s.
TEST(ConsistentRisk, ReLevelingOntoTheAnchorReproducesTheAnchorNpv) {
  const cal::BundleProblem p = ois_bundle();
  cal::BundleProblem shifted = p;
  for (auto& ins : shifted.instruments) ins.market += 25e-4;
  const json::object book = book_json();
  const json::object r = run(generate_risk_request(book, {p, shifted}));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::array& bs = r.at("bundles").as_array();
  ASSERT_EQ(bs.size(), 2u);

  api::BundleSession ref(p);
  ref.calibrate(line_x(p));
  const pf::MultiCurveBook bk = book_struct(book);
  const double inv_tol = tol::step_tol * ref.price_portfolio_risk(bk).curve_grad.lpNorm<1>();  // see IdenticalBundles
  const json::object& b0 = bs[0].as_object();
  const json::object& b1 = bs[1].as_object();
  EXPECT_NEAR(num(b1, "npv"), num(r, "npv"), inv_tol);
  EXPECT_NEAR(num(b1, "pv01"), num(r, "pv01"), inv_tol);
  EXPECT_NEAR(num(b0, "npv"), num(r, "npv"), inv_tol);
  expect_close(nums(b1.at("ladder")), to_eigen(nums(b0.at("ladder"))), tol::parity_jacobian, "re-leveled vs anchor");

  // Control: the shift is MATERIAL -- a session calibrated to the shifted market itself prices the book ~1.5e-3 away
  // (Σ ladder ≈ 0.596 per unit rate × 25 bp), so the equalities above cannot pass by the re-level being skipped.
  api::BundleSession own(shifted);
  own.calibrate(line_x(p));
  EXPECT_GT(std::abs(own.price_portfolio(bk).npv - num(r, "npv")), 1e3 * inv_tol);
}

// A bundle whose curve set differs from the anchor's (here: currency) is refused; the same request with a matching
// set -- and the mismatched bundle on its own -- succeed, so the error is the curve-set check, not the fixture.
TEST(ConsistentRisk, MismatchedCurveSetIsRefused) {
  const cal::BundleProblem p = ois_bundle();
  cal::BundleProblem other = p;
  other.curves[0].currency = 1;
  const json::object book = book_json();
  EXPECT_FALSE(run(generate_risk_request(book, {p, p})).contains("error"));
  EXPECT_FALSE(run(generate_risk_request(book, {other})).contains("error"));
  const json::object bad = run(generate_risk_request(book, {p, other}));
  EXPECT_TRUE(bad.contains("error")) << json::serialize(bad);
  EXPECT_TRUE(run(generate_risk_request(book, {})).contains("error")) << "an empty bundle list is refused";
}

// ================================================================================================================
// calib_report
// ================================================================================================================

// An x0 whose length is not the bundle's knot count is refused (both shorter and longer); the right length is not.
TEST(ConsistentRiskCalibReport, X0LengthMismatchIsRefused) {
  const cal::BundleProblem p = ois_bundle();
  const Eigen::VectorXd x_ok = line_x(p);
  const Eigen::VectorXd x_short = x_ok.head(kNk - 1);
  Eigen::VectorXd x_long(kNk + 1);
  x_long << x_ok, 0.03;

  const json::object ok = run(calib_report_request(p, &x_ok));
  ASSERT_FALSE(ok.contains("error")) << json::serialize(ok);
  EXPECT_LT(num(ok, "rms_residual"), tol::reprice);
  EXPECT_TRUE(run(calib_report_request(p, &x_short)).contains("error"));
  EXPECT_TRUE(run(calib_report_request(p, &x_long)).contains("error"));
  EXPECT_THROW((void)api::calib_report_json(calib_report_request(p, &x_short)), std::invalid_argument);
}

// A BANDED row's echoed diagnostics equal the session's own quote_diagnostics() to a few ULPs (E1: the verb re-parses
// quote_diagnostics_json(), and the response is parsed again here). Identifiability is deliberately NOT read.
TEST(ConsistentRiskCalibReport, BandedRowDiagnosticsEchoTheSession) {
  cal::BundleProblem p = ois_bundle();
  constexpr int kBand = 4;  // the 4y swap
  cal::Instrument& bi = p.instruments[kBand];
  bi.band_lower = bi.market - 20e-4;
  bi.band_upper = bi.market + 20e-4;
  bi.band_decay = 0.25;
  const Eigen::VectorXd x0 = line_x(p);

  // Same inputs, same recipe (explicit x0, no regulariser) -> the verb's session and this one reach the same x.
  api::BundleSession s(p);
  s.calibrate(x0);
  const json::array qd = s.quote_diagnostics();

  const json::object r = run(calib_report_request(p, &x0));
  ASSERT_FALSE(r.contains("error")) << json::serialize(r);
  const json::array& q = r.at("quotes").as_array();
  ASSERT_EQ(q.size(), qd.size());
  ASSERT_EQ(q.size(), static_cast<std::size_t>(kNr));
  EXPECT_EQ(r.at("n").to_number<int>(), kNr);

  for (std::size_t i = 0; i < q.size(); ++i) {
    const json::object& got = q[i].as_object();
    const json::object& want = qd[i].as_object();
    for (const char* k : {"model", "target", "residual", "weight"})
      EXPECT_TRUE(within_ulps(num(got, k), num(want, k))) << "row " << i << " field " << k;
    EXPECT_EQ(got.at("in_band").as_bool(), want.at("in_band").as_bool()) << "row " << i;
    EXPECT_EQ(got.at("soft").as_bool(), want.at("soft").as_bool()) << "row " << i;
  }
  const json::object& band = q[kBand].as_object();
  EXPECT_TRUE(band.at("soft").as_bool());
  EXPECT_TRUE(band.at("in_band").as_bool()) << "zero-residual fit sits at the mid, inside ±20 bp";
  EXPECT_TRUE(within_ulps(num(band, "weight"), 0.25)) << "in-band slope is the decay";
  EXPECT_FALSE(q[0].as_object().at("soft").as_bool());
  EXPECT_TRUE(within_ulps(num(q[0].as_object(), "weight"), 1.0));
}
