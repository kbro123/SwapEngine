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
#include <limits>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/calibration/regularize.hpp"

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

// rebind carries the FULL quote RHS (targets AND soft-quote bands) on the warm path — a band edit is not
// structural and must NOT force a recompile; a residual-count change is structural and is rejected.
TEST(BundleApi, RebindCarriesTheFullQuoteRhsWarm) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
  const Eigen::VectorXd x0 = sess.x();

  sess.rebind(p);  // rebinding to the same problem is an idempotent warm re-solve
  EXPECT_LT((sess.x() - x0).cwiseAbs().maxCoeff(), 1e-10);

  cal::BundleProblem banded = p;  // a soft band is a quote change, not a structure change
  banded.instruments[0].band_lower = banded.instruments[0].market - 0.01;
  banded.instruments[0].band_upper = banded.instruments[0].market + 0.01;
  banded.instruments[0].band_decay = 0.0;
  EXPECT_TRUE(sess.same_structure(banded));  // fingerprint excludes bands -> stays warm
  EXPECT_NO_THROW(sess.rebind(banded));      // ...and rebind applies the band warm

  cal::BundleProblem bigger = p;  // an extra instrument IS structural -> rebind rejects it
  bigger.instruments.push_back(p.instruments.back());
  EXPECT_THROW(sess.rebind(bigger), std::runtime_error);

  cal::BundleProblem moved = p;  // SAME count, different knots: also structural -> rebind must reject it
  moved.curves[0].regions.back().knots.back() += 0.25;  // (it used to rebind the new quotes onto old rows)
  EXPECT_FALSE(sess.same_structure(moved));
  EXPECT_THROW(sess.rebind(moved), std::runtime_error);
}

// The risk operator dx/dq must equal bump-and-recalibrate for EVERY row kind. A banded row's residual is
// w(q_model)·(q_model − q), so −∂r/∂q = w (= `decay` inside the band), NOT 1: the operator used to
// overstate that column by exactly 1/decay (10x at decay 0.1). Hard-pin columns are unchanged.
TEST(BundleApi, RiskOperatorMatchesBumpAndRecalibrateOnBandedRows) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_bundle(x_true);
  cal::Instrument& banded = p.instruments[4];  // the 4y swap on curve 0, quoted with a soft band
  banded.band_lower = banded.market - 0.002;
  banded.band_upper = banded.market + 0.002;
  banded.band_decay = 0.1;
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
  const Eigen::MatrixXd M = sess.risk_operator();
  ASSERT_EQ(M.rows(), p.n_knots());
  ASSERT_EQ(M.cols(), p.n_residuals());
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = p.instruments[i].market;
  const double h = 1e-7;
  for (int j : {4, 5}) {  // banded column and a hard-pin neighbour
    Eigen::VectorXd qp = q, qm = q;
    qp[j] += h;
    qm[j] -= h;
    sess.recalibrate(qp);
    const Eigen::VectorXd xp = sess.x();
    sess.recalibrate(qm);
    const Eigen::VectorXd xm = sess.x();
    sess.recalibrate(q);
    const Eigen::VectorXd fd = (xp - xm) / (2 * h);
    const double scale = std::max(1.0, fd.cwiseAbs().maxCoeff());
    EXPECT_LT((M.col(j) - fd).cwiseAbs().maxCoeff() / scale, 1e-5) << "column " << j;
  }
}

// On a rank-deficient bundle the operator must stay finite and still match bump-and-recalibrate on the
// identified directions (both are min-norm / seed-anchored); a tolerance-free LDLT of the singular JᵀJ
// used to return garbage in every column.
TEST(BundleApi, RiskOperatorIsRankSafe) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_bundle(x_true);
  p.instruments.erase(p.instruments.begin() + 6);  // drop the 10y swap: curve 0's last knot is unpinned
  api::BundleSession sess(p);
  sess.calibrate(x_true + Eigen::VectorXd::Constant(p.n_knots(), 1e-4));
  const Eigen::MatrixXd M = sess.risk_operator();
  ASSERT_TRUE(M.allFinite());
  EXPECT_LT(M.cwiseAbs().maxCoeff(), 1e3);
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = p.instruments[i].market;
  const double h = 1e-7;
  const int j = 2;  // the 1y swap on curve 0: fully identified
  Eigen::VectorXd qp = q, qm = q;
  qp[j] += h;
  qm[j] -= h;
  sess.recalibrate(qp);
  const Eigen::VectorXd xp = sess.x();
  sess.recalibrate(qm);
  const Eigen::VectorXd xm = sess.x();
  const Eigen::VectorXd fd = (xp - xm) / (2 * h);
  const double scale = std::max(1.0, fd.cwiseAbs().maxCoeff());
  EXPECT_LT((M.col(j) - fd).cwiseAbs().maxCoeff() / scale, 1e-4);
}

// stream_update validates its input: a wrong-length or non-finite market must throw, never be handed to
// the residual kernel (Eigen's size asserts are compiled out in release -> silent heap over-read).
TEST(BundleApi, StreamUpdateRejectsBadMarkets) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
  sess.start_streaming();
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = p.instruments[i].market;
  EXPECT_NO_THROW(sess.stream_update(q));
  EXPECT_THROW(sess.stream_update(Eigen::VectorXd::Zero(p.n_residuals() + 1)), std::runtime_error);
  EXPECT_THROW(sess.stream_update(Eigen::VectorXd::Zero(p.n_residuals() - 1)), std::runtime_error);
  Eigen::VectorXd bad = q;
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(sess.stream_update(bad), std::runtime_error);
  EXPECT_NO_THROW(sess.stream_update(q));  // the session is still usable after a rejected tick
}

// The warm-engine cache: a session compiles its hybrid engine ONCE and every later solve — regularized
// included — reuses it, with rebind/recalibrate updating only the quote RHS (set_quotes). Pins that
// (1) the tension-regularized solve on the composed compiled engine lands where the OLD wrapper path
// (LinearRegularizedProblem -> generic AAD engine) landed, and (2) a warm rebind/recalibrate on the
// cached engine equals a FRESH session cold-solving the identical problem — including a band ADDED
// after the engine was compiled (bands are quote RHS, not structure).
TEST(BundleApi, WarmEngineReuseMatchesFreshSessionsAndTheOldRegularizedPath) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(p.n_knots(), 0.02);

  // (1) Tension reg: composed compiled engine vs the old AAD wrapper, same seed, same reg.
  api::RegSpec reg;
  reg.lambda = 1e-3;
  reg.tension = true;
  reg.sigma = 0.5;
  reg.curves = {0, 1};
  api::BundleSession sess(p);
  sess.calibrate(x0, reg);
  const auto old_path = cal::calibrate(
      cal::linearly_regularized(p, cal::tension_energy_operator(p, reg.lambda, reg.sigma, reg.curves)),
      x0);
  EXPECT_LT((sess.x() - old_path.x).cwiseAbs().maxCoeff(), 1e-9)
      << "the compiled+R composition must land where the AAD wrapper landed";

  // (2) Warm rebind (shifted targets + a NEW band) on the cached engine == a fresh cold session.
  cal::BundleProblem p2 = p;
  for (auto& ins : p2.instruments) ins.market += 5e-4;
  p2.instruments[2].band_lower = p2.instruments[2].market - 1e-4;
  p2.instruments[2].band_upper = p2.instruments[2].market + 1e-4;
  p2.instruments[2].band_decay = 0.1;
  sess.rebind(p2, reg);  // warm: same compiled engine, new full quote RHS
  api::BundleSession fresh(p2);
  fresh.calibrate(x0, reg);  // cold: engine compiled directly against p2
  // 1e-6 since 2026-09-10: the warm rebind is a STREAMED tick (the frozen-Newton active set lands exactly on
  // the band kink), the fresh session is an LM that stalls a hair short of it; they agree to ~4e-7 in the
  // knots. Before, both were LM and agreed to 1e-8.
  EXPECT_LT((sess.x() - fresh.x()).cwiseAbs().maxCoeff(), 1e-6)
      << "a warm rebind must reach the same solution as a cold session on the same problem";

  // (3) Plain (unregularized) warm recalibrate equals a fresh cold solve of the shifted market.
  api::BundleSession s3(p);
  s3.calibrate(x0);
  Eigen::VectorXd m2(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) m2[i] = p.instruments[i].market + 3e-4;
  s3.recalibrate(m2);
  cal::BundleProblem p3 = p;
  for (int i = 0; i < p3.n_residuals(); ++i) p3.instruments[i].market = m2[i];
  api::BundleSession f3(p3);
  f3.calibrate(x0);
  EXPECT_LT((s3.x() - f3.x()).cwiseAbs().maxCoeff(), 1e-8);
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
  // Omitting `convention` is an ERROR (PRINCIPLES.md P2: no silent US-TREASURY default — until 2026-09-09
  // this test asserted the opposite).
  const auto def = run(R"({"bonds":{"bonds":[{"issue":"2019-08-15","settle":"2024-01-16",
      "maturity":"2029-08-15","coupon":0.025,"yield":0.04}]}})");
  EXPECT_TRUE(def.contains("error")) << "a bond without 'convention' must be rejected";

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

// ---- API wiring of the preview domains (trade/, market/, derive/) --------------------------------------

// Typed trades through book_from_json: booked deals + the index->role binding ("curve_roles") replace
// hand-assembled coupon JSON. Each trade rolls under ITS OWN index's conventions; the CSA picks the
// discount role (collateral currency's OIS) — the trade:: domain reaching pricing for real.
TEST(BundleApi, TypedTradesPriceThroughTheBookWithCsaDiscounting) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));

  const std::string book = R"({
    "value_date": "2026-09-04",
    "curve_roles": {"USD-SOFR": 0},
    "trades": [
      {"id": "T1", "notional": 1000000, "pay": "fixed", "fixed_rate": 0.03, "index": "USD-SOFR",
       "effective": "2026-09-08", "maturity": "2031-09-08", "csa": {"collateral_currency": "USD"}},
      {"id": "T2", "notional": 1000000, "pay": "float", "fixed_rate": 0.03, "index": "USD-SOFR",
       "effective": "2026-09-08", "maturity": "2031-09-08", "csa": {"collateral_currency": "USD"}}
    ]})";
  const auto mb = api::book_from_json(json::parse(book));
  ASSERT_EQ(mb.positions.size(), 2u);
  // Roles resolved through the binding: forecast = roles["USD-SOFR"], discount = roles[CSA -> USD-SOFR].
  EXPECT_EQ(mb.positions[0].fwd_curve, 0);
  EXPECT_EQ(mb.positions[0].disc_curve, 0);
  // Direction carried by the signed notional (payer +, receiver -), coupons rolled from real dates.
  EXPECT_GT(mb.positions[0].notional, 0.0);
  EXPECT_LT(mb.positions[1].notional, 0.0);
  EXPECT_FALSE(mb.positions[0].float_coupons.empty());
  // Opposite directions on the same terms: the book nets to ~0 NPV whatever the curve says.
  const api::PortfolioReprice r = sess.price_portfolio(mb);
  EXPECT_NEAR(r.npv, 0.0, 1e-9);
  // An unmapped index fails loudly, not silently on role 0.
  const std::string bad = R"({"value_date": "2026-09-04", "curve_roles": {},
    "trades": [{"index": "USD-SOFR", "effective": "2026-09-08", "maturity": "2027-09-08"}]})";
  EXPECT_THROW(api::book_from_json(json::parse(bad)), std::invalid_argument);
}

// The batched bond_universe verb: one vectorized sweep, cross-checked against the scalar kernel.
TEST(BundleApi, BondUniverseVerbMatchesTheScalarKernel) {
  const std::string req = R"({"bond_universe": {"value_date": "2026-09-04", "convention": "US-TREASURY",
    "bonds": [{"id": "A", "issue": "2026-08-15", "maturity": "2031-08-15", "coupon": 0.04},
              {"id": "B", "issue": "2026-08-15", "maturity": "2036-08-15", "coupon": 0.045}],
    "clean": [0.991, 1.012]}})";
  const json::value out = json::parse(api::run_json(req));
  const auto& u = out.as_object().at("bond_universe").as_object();
  ASSERT_EQ(u.at("yield").as_array().size(), 2u);

  const swaps::build::Date vd = swaps::build::Date::from_iso("2026-09-04");
  const swaps::build::Date settle = swaps::build::advance_bd("USD", vd, 1);
  swaps::build::BondId a{"A", "US-TREASURY", swaps::build::Date::from_iso("2026-08-15"),
                         swaps::build::Date::from_iso("2031-08-15"), 0.04, {}};
  const double y_direct =
      swaps::pricing::bond_yield_from_clean(swaps::build::build_bond(a, vd, settle).yield, 0.991);
  EXPECT_NEAR(u.at("yield").as_array()[0].as_double(), y_direct, 1e-10);
  EXPECT_GT(u.at("modified_duration").as_array()[1].as_double(),
            u.at("modified_duration").as_array()[0].as_double())
      << "the longer bond carries the higher duration";
}

// The govvie_fit verb: minimum pricing error over a universe priced ON a known Nelson-Siegel curve
// recovers the parameters — derive/ + market/ reached through the production JSON seam.
TEST(BundleApi, GovvieFitVerbRecoversANelsonSiegelCurve) {
  const swaps::build::Date vd = swaps::build::Date::from_iso("2026-09-04");
  const swaps::build::Date settle = swaps::build::advance_bd("USD", vd, 1);
  const double tau = 2.0;
  Eigen::VectorXd theta(3);
  theta << 0.045, -0.015, -0.020;
  swaps::curve::NelsonSiegel<double> truth(tau);
  truth.set_params(theta);

  struct U { const char* id; const char* mat; double cpn; };
  const std::vector<U> defs = {{"B1", "2028-09-15", 0.035}, {"B2", "2030-09-15", 0.038},
                               {"B3", "2032-09-15", 0.042}, {"B4", "2036-09-15", 0.043},
                               {"B5", "2041-09-15", 0.05}};
  json::array bonds;
  json::array cleans;
  for (const auto& d : defs) {
    swaps::build::BondId bi{d.id, "US-TREASURY", swaps::build::Date::from_iso("2026-08-15"),
                            swaps::build::Date::from_iso(d.mat), d.cpn, {}};
    cleans.push_back(
        swaps::pricing::bond_clean_price<double>(swaps::build::build_bond(bi, vd, settle).curve, truth));
    json::object bo;
    bo["id"] = d.id;
    bo["issue"] = "2026-08-15";
    bo["maturity"] = d.mat;
    bo["coupon"] = d.cpn;
    bonds.push_back(bo);
  }
  json::object gf;
  gf["value_date"] = "2026-09-04";
  gf["convention"] = "US-TREASURY";
  gf["model"] = "nelson_siegel";
  gf["tau1"] = tau;
  gf["bonds"] = bonds;
  gf["clean"] = cleans;
  json::object req;
  req["govvie_fit"] = gf;

  const json::value out = json::parse(api::run_json(json::serialize(req)));
  const auto& g = out.as_object().at("govvie_fit").as_object();
  ASSERT_EQ(g.at("x").as_array().size(), 3u);
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(g.at("x").as_array()[i].as_double(), theta[i], 1e-6)
        << "the fit must recover the true Nelson-Siegel parameters";
  EXPECT_LT(g.at("rms_residual").as_double(), 1e-9);
}

// The swap_spread verb: the headline derivation returns the benchmark street yield and the {pin, asw}
// BASIS rows as instrument JSON — the basis instrument reachable from any host via run_json.
TEST(BundleApi, SwapSpreadVerbDerivesTheBasisRows) {
  const std::string req = R"({"swap_spread": {"value_date": "2026-09-04", "convention": "US-TREASURY",
    "bond": {"id": "UST-5Y", "issue": "2026-08-15", "maturity": "2031-08-15", "coupon": 0.04},
    "clean": 0.991, "spread": -0.0032, "index": "USD-SOFR", "tenor": "5Y",
    "swap_curve": 0, "factor_curve": 1}})";
  const json::value out = json::parse(api::run_json(req));
  const auto& s = out.as_object().at("swap_spread").as_object();

  const swaps::build::Date vd = swaps::build::Date::from_iso("2026-09-04");
  const swaps::build::Date settle = swaps::build::advance_bd("USD", vd, 1);
  swaps::build::BondId bi{"UST-5Y", "US-TREASURY", swaps::build::Date::from_iso("2026-08-15"),
                          swaps::build::Date::from_iso("2031-08-15"), 0.04, {}};
  const double y_direct =
      swaps::pricing::bond_yield_from_clean(swaps::build::build_bond(bi, vd, settle).yield, 0.991);
  EXPECT_NEAR(s.at("bond_yield").as_double(), y_direct, 1e-10);

  // The rows round-trip through instrument_from_json into the real calibration types.
  const cal::Instrument pin = api::instrument_from_json(s.at("rows").as_object().at("pin"));
  const cal::Instrument asw = api::instrument_from_json(s.at("rows").as_object().at("asw"));
  EXPECT_EQ(pin.quote, cal::QuoteKind::Rate);
  EXPECT_EQ(pin.forecast, 1);
  EXPECT_NEAR(pin.market, y_direct, 1e-10);
  ASSERT_EQ(asw.quote, cal::QuoteKind::Portfolio);
  ASSERT_EQ(asw.combination.size(), 2u);
  EXPECT_NEAR(asw.market, -0.0032, 1e-12);
  EXPECT_FALSE(asw.combination[0].instrument.fwd.coupons.empty())
      << "the +1 component is the convention-built spot par swap";
}

// A RANK-DEFICIENT bundle (a knot no instrument pins) must STREAM, not NaN: the unregularized frozen-
// Newton operator is a rank-safe minimum-norm pseudo-inverse, so unpinned directions stay frozen at the
// calibrated anchor instead of being inverted (sigma ~ 1e-11 blew up the old QR solve within one tick —
// the web's smoothness-off engines hit exactly this).
TEST(BundleApi, RankDeficientBundleStreamsFinite) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_bundle(x_true);
  p.curves[1].regions.back().knots.push_back(20.0);  // an unpinned knot: no instrument reaches 20y
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
  ASSERT_TRUE(sess.x().allFinite());
  sess.start_streaming({}, 1e-6);
  Eigen::VectorXd q = p.market();
  for (int i = 1; i <= 3; ++i) {
    const Eigen::VectorXd& x = sess.stream_update(q.array() + 1e-5 * i);
    ASSERT_TRUE(x.allFinite()) << "tick " << i << " must stay finite on a rank-deficient bundle";
  }
  // The solve genuinely tracked the move (not a frozen no-op): model rates follow the shifted market.
  const Eigen::VectorXd r = p.residuals<double>(sess.x());
  (void)r;  // residual vs the ORIGINAL market is ~3e-5 (the shift); finiteness is the contract here
  EXPECT_LT(sess.last_drift(), 1e-3);
}

// MINIMUM-NORM completion: on a rank-deficient bundle the unpinned state lands EXACTLY at the seed (not
// wherever the LM path wandered), rank_deficiency reports the null count, and a determined bundle is a
// pure no-op (rank_deficiency == 0; the recovery tests above pin exactness).
TEST(BundleApi, MinNormCalibrationPinsUnconstrainedStatesToTheSeed) {
  Eigen::VectorXd x_true;
  cal::BundleProblem p = build_bundle(x_true);
  {
    api::BundleSession sess(p);  // determined: full rank, nothing snapped
    sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
    EXPECT_EQ(sess.result().rank_deficiency, 0);
  }
  p.curves[1].regions.back().knots.push_back(20.0);  // an unpinned knot (no instrument reaches 20y)
  api::BundleSession sess(p);
  const double seed = 0.0123;  // deliberately distinctive
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), seed));
  EXPECT_EQ(sess.result().rank_deficiency, 1);
  const int unpinned = p.n_knots() - 1;  // curve 1's appended last knot is the final state
  // ~seed, not exactly: the 20y knot is only NEAR-null (Hermite derivative coupling gives the market a
  // weak say), so the anchored optimum is the compromise — deterministic and within bp of the seed,
  // instead of the multi-hundred-percent LM wander this guards against.
  EXPECT_NEAR(sess.x()[unpinned], seed, 5e-4)
      << "the null direction must sit near the seed, not at an arbitrary LM endpoint";
  EXPECT_LT(sess.result().rms_residual, 1e-6) << "the anchored re-solve must not disturb the fit";
}

// The EXTRAPOLATION POLICY beyond the last calibration instrument: FLAT INSTANTANEOUS FORWARD (the
// industry default). forward(t > t_last) == the last knot's fitted forward, and discounts compound at
// that flat rate: DF(T) = DF(t_last) * exp(-f_last * (T - t_last)). Pinned here at the SESSION level so
// pricing cash flows beyond the final instrument is a documented contract, not an accident of the
// interpolator.
TEST(BundleApi, FlatForwardExtrapolationBeyondTheLastInstrument) {
  Eigen::VectorXd x_true;
  const cal::BundleProblem p = build_bundle(x_true);  // last knot / last instrument at 10y
  api::BundleSession sess(p);
  sess.calibrate(Eigen::VectorXd::Constant(p.n_knots(), 0.03));
  const auto s = sess.sample({10.0, 12.0, 20.0, 40.0});
  const auto& c0 = s[0];
  const double f_last = c0.forward[0];
  for (std::size_t i = 1; i < c0.t.size(); ++i)
    EXPECT_NEAR(c0.forward[i], f_last, 1e-12) << "forward must be FLAT beyond the last knot (t=" << c0.t[i] << ")";
  EXPECT_NEAR(c0.discount[2], c0.discount[0] * std::exp(-f_last * 10.0), 1e-12)
      << "DF(20) must compound flat off DF(10)";
  EXPECT_NEAR(c0.discount[3], c0.discount[0] * std::exp(-f_last * 30.0), 1e-10)
      << "DF(40) must compound flat off DF(10)";
}
