// Self-contained gate for the public API (include/swaps/api/bundle_api.hpp). NO QuantLib: a small
// 2-curve bundle (outright base + a spread curve) is hand-built as generic Instruments, made
// self-consistent from a known x_true, and driven THROUGH the JSON contract + BundleSession.
//
// Proves: (1) JSON (de)serialization is lossless -- parse(dump(p)) reprices bit-for-bit at x_true;
// (2) BundleSession.calibrate off the JSON-reconstructed problem recovers x_true and matches a native
// calibrate; (3) sample()/model_quote()/streaming behave; (4) run_json end-to-end returns sane JSON.

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <vector>

#include "swaps/api/bundle_api.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

// Curve topology shared by both curves (curve 1 is a spread over curve 0 on the same knots).
const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 4.0, 7.0, 10.0};
constexpr int kNk = 7;  // per curve

// One OIS-style float coupon over [a,b]: a single telescoped sub-period (tau_pay == tau_index).
px::FloatCoupon ois_coupon(double a, double b, int forecast, int discount) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  (void)forecast;
  (void)discount;  // roles live on the leg, set by the caller
  return c;
}

// A par-rate OIS swap to maturity T (annual schedule): float leg forecast=fc/discount=dc, annual fixed.
cal::Instrument make_swap(double T, int fc, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.fixed.discount = dc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    ins.fwd.coupons.push_back(ois_coupon(prev, t, fc, dc));
    px::FixedCoupon x;
    x.pay = t;
    x.tau = t - prev;
    ins.fixed.coupons.push_back(x);
    prev = t;
  }
  return ins;
}

// A basis (par-spread) swap: fwd leg on `fc` vs benchmark leg on `bench`, both discounted on `dc`.
cal::Instrument make_basis(double T, int fc, int bench, int dc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParSpread;
  ins.fwd.forecast = fc;
  ins.fwd.discount = dc;
  ins.bench.forecast = bench;
  ins.bench.discount = dc;
  ins.fixed.discount = dc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    ins.fwd.coupons.push_back(ois_coupon(prev, t, fc, dc));
    ins.bench.coupons.push_back(ois_coupon(prev, t, bench, dc));
    px::FixedCoupon x;
    x.pay = t;
    x.tau = t - prev;
    ins.fixed.coupons.push_back(x);
    prev = t;
  }
  return ins;
}

// A front "Rate" pin over [a,b] on curve `fc` (like a futures rate; pins the flat segment).
cal::Instrument make_rate(double a, double b, int fc) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = fc;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}

// Build the 2-curve bundle (instruments with market=0) + the true parameter vector.
cal::BundleProblem build_bundle(Eigen::VectorXd& x_true) {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});  // curve 0: outright
  p.curves.push_back({.base = 0, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});   // curve 1: spread over 0

  // Curve 0 (forecast=discount=0): front pins + par swaps.
  p.instruments.push_back(make_rate(0.0, 0.25, 0));
  p.instruments.push_back(make_rate(0.25, 0.5, 0));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_swap(T, 0, 0));
  // Curve 1 (forecast=1, discount=0): front spread pins + basis vs curve 0.
  p.instruments.push_back(make_rate(0.0, 0.25, 1));
  p.instruments.push_back(make_rate(0.25, 0.5, 1));
  for (double T : {1.0, 2.0, 4.0, 7.0, 10.0}) p.instruments.push_back(make_basis(T, 1, 0, 0));

  // x_true: curve 0 forwards ~3% with a little shape; curve 1 spreads ~ +12bp.
  x_true.resize(2 * kNk);
  for (int i = 0; i < kNk; ++i) x_true[i] = 0.030 + 0.001 * i;
  for (int i = 0; i < kNk; ++i) x_true[kNk + i] = 0.0012 + 0.00005 * i;

  // Self-consistent markets: set each instrument's quote to its model quote at x_true.
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

}  // namespace

TEST(BundleApi, JsonRoundTripIsLossless) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);

  // dump -> parse -> the reconstructed problem must reprice IDENTICALLY at x_true.
  const cal::BundleProblem q = api::bundle_from_json(api::bundle_to_json(p));
  ASSERT_EQ(q.n_curves(), p.n_curves());
  ASSERT_EQ(q.n_residuals(), p.n_residuals());
  ASSERT_EQ(q.n_knots(), p.n_knots());
  const Eigen::VectorXd rp = p.residuals<double>(x_true);
  const Eigen::VectorXd rq = q.residuals<double>(x_true);
  EXPECT_EQ((rp - rq).cwiseAbs().maxCoeff(), 0.0) << "serialization must be bit-for-bit lossless";
  // And self-consistent by construction.
  EXPECT_LT(rp.cwiseAbs().maxCoeff(), 1e-12);
}

TEST(BundleApi, SessionCalibrateRecoversAndMatchesNative) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  const cal::BundleProblem q = api::bundle_from_json(api::bundle_to_json(p));

  // Start away from the answer.
  Eigen::VectorXd x0 = x_true;
  for (int i = 0; i < x0.size(); ++i) x0[i] += (i % 2 ? 1.0 : -1.0) * 0.0015 + 0.0008;

  api::BundleSession sess(q);
  const cal::CalibrationResult& r = sess.calibrate(x0);
  const double err = (sess.x() - x_true).cwiseAbs().maxCoeff();
  std::cout << "  [api] iters=" << r.iterations << " rms=" << r.rms_residual << " ||x-xtrue||=" << err
            << " stationarity=" << r.stationarity << "\n";
  EXPECT_LT(err, 1e-7) << "the API session must recover the generating curves";
  EXPECT_LT(r.stationarity, 1e-8);

  // Matches a native calibrate of the same problem.
  const cal::CalibrationResult native = cal::calibrate(p, x0);
  EXPECT_LT((sess.x() - native.x).cwiseAbs().maxCoeff(), 1e-10);
}

// The OO/hot-path switch at the session level: a re-quote keeps the compiled W-cache valid (warm tick),
// a topology edit invalidates it (recompile).
TEST(BundleApi, SameStructureIsTheWarmVsRecompileSwitch) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  const api::BundleSession sess(p);

  cal::BundleProblem requoted = p;  // market-only move -> same structure -> warm-tickable
  for (auto& ins : requoted.instruments) ins.market += 5e-4;
  EXPECT_TRUE(sess.same_structure(requoted));

  cal::BundleProblem moved = p;  // move a knot -> different structure -> must recompile
  moved.curves[0].regions.back().knots.back() += 0.25;
  EXPECT_FALSE(sess.same_structure(moved));
}

TEST(BundleApi, SampleAndPriceOffSolvedCurves) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  api::BundleSession sess(p);
  Eigen::VectorXd x0 = x_true;
  x0.array() += 0.001;
  sess.calibrate(x0);

  const auto s = sess.sample({0.0, 1.0, 5.0, 10.0});
  ASSERT_EQ(s.size(), 2u);
  EXPECT_NEAR(s[0].discount[0], 1.0, 1e-12);           // DF(0) == 1
  EXPECT_LT(s[0].discount[3], s[0].discount[1]);       // DF decreasing in t
  EXPECT_GT(s[1].forward[2], s[0].forward[2]);         // spread curve forward > base (positive spread)

  // Pricing an instrument that IS in the calibration set reprices to ~its market (self-consistent).
  const cal::Instrument swap10 = make_swap(10.0, 0, 0);
  cal::Instrument s10 = swap10;
  s10.market = 0.0;  // market unused by model_quote
  const double mq = sess.model_quote(s10);
  EXPECT_NEAR(mq, sess.model_quote(p.instruments[6]), 1e-12);  // instrument 6 is the 10y curve-0 swap
  EXPECT_GT(mq, 0.02);
  EXPECT_LT(mq, 0.05);
}

TEST(BundleApi, StreamingTracksAMarketMove) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  api::BundleSession sess(p);
  sess.calibrate(x_true);  // start at the exact solution
  sess.start_streaming();

  // Bump every quote +2bp and stream: the curve must reprice the new market exactly.
  Eigen::VectorXd q = p.market().array() + 2e-4;
  const Eigen::VectorXd& xnew = sess.stream_update(q);

  // Reprice the instruments off the streamed curve: residual vs the new market ~ 0.
  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return xnew[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  double maxr = 0;
  for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i) {
    cal::Instrument ins = p.instruments[i];
    ins.market = q[i];
    maxr = std::max(maxr, std::abs(cal::instrument_residual<double>(ins, curve_of)));
  }
  EXPECT_LT(maxr, 1e-9) << "streamed curve must reprice the moved market exactly";
}

namespace {

// A synthetic single-coupon OIS par-rate swap to T (float leg telescopes to 1 − DF(T)); pins back knots.
cal::Instrument turn_ois(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  px::FloatCoupon fc;
  fc.obs.sub_start = {0.0};
  fc.obs.sub_end = {T};
  fc.obs.tau_index = T;
  fc.pay = T;
  fc.tau_pay = T;
  ins.fwd.coupons = {fc};
  ins.fixed.coupons = {px::FixedCoupon{T, T, 1.0}};
  return ins;
}
// A synthetic overnight future over [s,e] as a single-sub Rate instrument.
cal::Instrument turn_fut(double s, double e) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {s};
  ins.obs.sub_end = {e};
  ins.obs.tau_index = e - s;
  return ins;
}

// One outright curve carrying a single year-end turn + a banded TurnJump instrument, market generated
// from a known x_true (incl. δ) so x_true is a stationary point.
cal::BundleProblem build_turn_bundle(Eigen::VectorXd& x_true, int& delta_index) {
  const double a = 0.98, b = 1.02, delta_true = 0.0030;
  cal::BundleProblem p;
  px::CurveStructure s;
  s.regions = swaps::curve::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0});
  s.turns = {{a, b}};
  p.curves = {s};
  delta_index = s.n_interp_knots();  // δ sits right after the 6 interp knots

  // Spanning + non-spanning futures reconcile only via the turn; pillar OIS pins the smooth back.
  p.instruments = {turn_fut(0.80, 0.95), turn_fut(0.95, 1.05), turn_fut(0.90, 1.10),
                   turn_fut(1.05, 1.20), turn_ois(0.5),        turn_ois(1.0),
                   turn_ois(2.0),        turn_ois(3.0),        turn_ois(5.0)};
  cal::Instrument pin;
  pin.quote = cal::QuoteKind::TurnJump;
  pin.turn_curve = 0;
  pin.turn_index = 0;
  pin.market = delta_true;
  pin.band_lower = delta_true - 0.001;
  pin.band_upper = delta_true + 0.001;
  pin.band_decay = 0.05;
  p.instruments.push_back(pin);

  x_true.resize(p.n_knots());
  x_true << 0.030, 0.032, 0.035, 0.037, 0.039, 0.041, delta_true;

  const auto C = cal::build_bundle_curves<double>(
      p.curves, [&](int c, int i) { return x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

}  // namespace

TEST(BundleApi, TurnsJsonRoundTripPreservesStructureAndCalibrates) {
  Eigen::VectorXd x_true;
  int delta_index = 0;
  const cal::BundleProblem p = build_turn_bundle(x_true, delta_index);

  // Round-trip through the JSON contract.
  const json::value doc = api::bundle_to_json(p);
  const cal::BundleProblem q = api::bundle_from_json(doc);

  // (1) The turn overlay survives: same knot count (6 interp + 1 δ) and the turn window round-trips.
  ASSERT_EQ(q.n_curves(), 1);
  ASSERT_EQ(q.n_knots(), p.n_knots());
  ASSERT_EQ(q.n_knots(), 7);
  ASSERT_EQ(q.curves[0].turns.size(), 1u);
  EXPECT_EQ(q.curves[0].turns[0].start, p.curves[0].turns[0].start);
  EXPECT_EQ(q.curves[0].turns[0].end, p.curves[0].turns[0].end);

  // (2) The TurnJump instrument survives with all its fields.
  const cal::Instrument& qp = q.instruments.back();
  EXPECT_EQ(qp.quote, cal::QuoteKind::TurnJump);
  EXPECT_EQ(qp.turn_curve, 0);
  EXPECT_EQ(qp.turn_index, 0);
  EXPECT_EQ(qp.market, p.instruments.back().market);
  EXPECT_EQ(qp.band_lower, p.instruments.back().band_lower);
  EXPECT_EQ(qp.band_upper, p.instruments.back().band_upper);
  EXPECT_EQ(qp.band_decay, p.instruments.back().band_decay);

  // (3) Bit-for-bit lossless: reprices identically at x_true.
  const Eigen::VectorXd rp = p.residuals<double>(x_true);
  const Eigen::VectorXd rq = q.residuals<double>(x_true);
  EXPECT_EQ((rp - rq).cwiseAbs().maxCoeff(), 0.0) << "turn serialization must be bit-for-bit lossless";

  // (4) flat_x0 seeds the δ overlay at 0 (no jump), not at the rate level.
  const Eigen::VectorXd x0seed = api::flat_x0(q);
  EXPECT_EQ(x0seed[delta_index], 0.0) << "turn δ must seed at 0, not the rate level";

  // (5) Calibrates to first-order optimality off the JSON-reconstructed problem.
  Eigen::VectorXd x0 = x_true;
  x0.array() += 0.002;
  x0[delta_index] = 0.0;  // start from no jump
  api::BundleSession sess(q);
  const cal::CalibrationResult& r = sess.calibrate(x0);
  EXPECT_LT(r.stationarity, 1e-7) << "banded-turn bundle must reach ‖Jᵀr‖∞ ≈ 0";
  EXPECT_NEAR(sess.x()[delta_index], x_true[delta_index], 5e-4) << "turn δ recovered near its target";
}

TEST(BundleApi, RunJsonEndToEnd) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  json::object req;
  req["bundle"] = api::bundle_to_json(p);
  req["sample_times"] = json::array{1.0, 5.0, 10.0};
  req["risk"] = true;
  const std::string resp = api::run_json(json::serialize(json::value(req)));

  const json::value v = json::parse(resp);
  const json::object& o = v.as_object();
  ASSERT_FALSE(o.contains("error")) << resp;
  ASSERT_TRUE(o.contains("calibration"));
  EXPECT_LT(o.at("calibration").as_object().at("stationarity").to_number<double>(), 1e-7);
  ASSERT_TRUE(o.contains("x"));
  EXPECT_EQ(o.at("x").as_array().size(), static_cast<std::size_t>(p.n_knots()));
  ASSERT_TRUE(o.contains("curves"));
  EXPECT_EQ(o.at("curves").as_array().size(), 2u);
  // risk operator is n_knots x n_residuals
  ASSERT_TRUE(o.contains("risk_operator"));
  EXPECT_EQ(o.at("risk_operator").as_array().size(), static_cast<std::size_t>(p.n_knots()));
  EXPECT_EQ(o.at("risk_operator").as_array()[0].as_array().size(),
            static_cast<std::size_t>(p.n_residuals()));
}

// The stateless `bonds` verb: convention selection and the when-issued path across the JSON seam. This is
// the contract the web/Excel layer consumes, so it is pinned here rather than only in the C++ builders.
TEST(BundleApi, BondsVerbSelectsConventionAndHandlesWhenIssued) {
  auto run = [](const std::string& req) {
    return boost::json::parse(api::run_json(req)).as_object();
  };
  // Same bond, same yield, two conventions -> two different prices (street vs 31 CFR App B).
  const auto o = run(R"({"bonds":{"bonds":[
      {"convention":"US-TREASURY","issue":"2019-08-15","settle":"2024-01-16",
       "maturity":"2029-08-15","coupon":0.025,"yield":0.04},
      {"convention":"US-TREASURY-TSY","issue":"2019-08-15","settle":"2024-01-16",
       "maturity":"2029-08-15","coupon":0.025,"yield":0.04}]}})");
  ASSERT_TRUE(o.contains("dirty")) << api::run_json("");
  const auto& d = o.at("dirty").as_array();
  ASSERT_EQ(d.size(), 2u);
  const double street = d[0].as_double(), tsy = d[1].as_double();
  // Rateslib us_gb / ust_31bii on this bond, per 100 face.
  EXPECT_NEAR(street * 100.0, 93.607147151563, 1e-9);
  EXPECT_NEAR(tsy * 100.0, 93.604631472068, 1e-9);
  EXPECT_GT(std::abs(street - tsy), 1e-6);
  // Omitting `convention` defaults to US-TREASURY (street).
  const auto def = run(R"({"bonds":{"bonds":[{"issue":"2019-08-15","settle":"2024-01-16",
      "maturity":"2029-08-15","coupon":0.025,"yield":0.04}]}})");
  EXPECT_EQ(def.at("dirty").as_array()[0].as_double(), street);

  // WHEN-ISSUED: `dated` + `first_coupon` instead of `issue`. A new issue settles on the dated date, so
  // accrued is exactly zero and clean == dirty.
  const auto wi = run(R"({"bonds":{"bonds":[{"convention":"US-TREASURY-TSY","dated":"2024-06-15",
      "first_coupon":"2024-11-15","settle":"2024-06-15","maturity":"2034-11-15",
      "coupon":0.045,"yield":0.047}]}})");
  EXPECT_NEAR(wi.at("accrued").as_array()[0].as_double(), 0.0, 1e-15);
  EXPECT_EQ(wi.at("clean").as_array()[0].as_double(), wi.at("dirty").as_array()[0].as_double());

  // Bad input is rejected, not silently defaulted.
  EXPECT_TRUE(run(R"({"bonds":{"bonds":[{"convention":"NO-SUCH","issue":"2019-08-15",
      "settle":"2024-01-16","maturity":"2029-08-15","coupon":0.025,"yield":0.04}]}})").contains("error"));
  EXPECT_TRUE(run(R"({"bonds":{"bonds":[{"dated":"2024-06-15","settle":"2024-06-15",
      "maturity":"2034-11-15","coupon":0.045,"yield":0.047}]}})").contains("error"));
}
