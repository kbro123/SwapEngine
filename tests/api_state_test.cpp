// E5 taxonomy: T4 hot-path invariant (allocation / determinism / structure) | T3 cross-path parity (two engine paths, same inputs)
// The object model at the session seam (E4.A step A, 2026-09-10): ONE compiled engine per session, quotes and
// bands as scalar row updates that reach EVERY consumer, a warm re-solve that is a streamed tick, fixings and
// bound books that are never stale, and structure checked by an O(n) equality rather than a hash.
// Each test is one of the E3-D findings reproduced from probe D-stale / D-fingerprint / D-rebind:
//   D1  a rebind (band edit) after start_streaming was invisible to the streamer (it owned an engine copy);
//   D2  set_fixings after start_streaming was invisible to the streamer;
//   D3  reprice_bound priced stale fixings (the book was resolved once at bind);
//   D5  rebind was a cold LM with 15.7k allocations;
//   D16 same_structure(problem()) was FALSE from construction on any schedule-carrying bundle.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <boost/json.hpp>
#include <cmath>
#include <random>
#include <string>

#include "malloc_count.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace js = boost::json;

namespace {
cal::Instrument par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}
// 6 Hermite knots, 10 annual par swaps; odd rows carry a band of `half_bp` around a market perturbed by
// `noise_bp` (the same seed => the same markets for every half-width, so two problems differ ONLY in bands).
cal::BundleProblem banded_bundle(double half_bp, double noise_bp, double decay) {
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, noise_bp * 1e-4);
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3, 5, 7, 10})});
  Eigen::VectorXd xt(6); xt << 0.040, 0.041, 0.042, 0.044, 0.046, 0.047;
  for (int T = 1; T <= 10; ++T) p.instruments.push_back(par_swap(T));
  const Eigen::VectorXd r = p.residuals<double>(xt);
  for (int i = 0; i < 10; ++i) {
    auto& ins = p.instruments[i];
    ins.market = r[i] + ((i % 2) ? noise(rng) : 0.0);
    if (half_bp > 0.0 && i % 2 == 1) {
      ins.band_lower = ins.market - half_bp * 1e-4;
      ins.band_upper = ins.market + half_bp * 1e-4;
      ins.band_decay = decay;
    }
  }
  return p;
}
const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.03);

// api_fixings_test's fixture: one curve, one Rate obs with a 5-day fixing schedule (serials 97..101), and a
// one-position book whose float coupon carries the SAME schedule (a seasoned coupon).
const char* kBundle = R"({
  "curves":[{"meeting":[],"back":[1.0,2.0],"base":-1,"currency":0}],
  "instruments":[{"quote":"Rate","forecast":0,"market":0.04,
    "obs":{"tau_index":0.0138889,"fixing_index":"USD-SOFR","fixing_schedule":[
      {"fixing_date":97, "accrual":0.0027778,"t_start":0.000,"t_end":0.001,"weight":1.0},
      {"fixing_date":98, "accrual":0.0027778,"t_start":0.001,"t_end":0.002,"weight":1.0},
      {"fixing_date":99, "accrual":0.0027778,"t_start":0.002,"t_end":0.003,"weight":1.0},
      {"fixing_date":100,"accrual":0.0027778,"t_start":0.003,"t_end":0.004,"weight":1.0},
      {"fixing_date":101,"accrual":0.0027778,"t_start":0.004,"t_end":0.005,"weight":1.0}]}}]})";
const char* kBook = R"({"positions":[{"kind":"swap","notional":1.0,"fixed_rate":0.0,"fwd_curve":0,"disc_curve":0,
  "float_coupons":[{"pay":0.005,"tau_pay":0.0138889,"obs":{"tau_index":0.0138889,"fixing_index":"USD-SOFR","fixing_schedule":[
      {"fixing_date":97, "accrual":0.0027778,"t_start":0.000,"t_end":0.001,"weight":1.0},
      {"fixing_date":98, "accrual":0.0027778,"t_start":0.001,"t_end":0.002,"weight":1.0},
      {"fixing_date":99, "accrual":0.0027778,"t_start":0.002,"t_end":0.003,"weight":1.0},
      {"fixing_date":100,"accrual":0.0027778,"t_start":0.003,"t_end":0.004,"weight":1.0},
      {"fixing_date":101,"accrual":0.0027778,"t_start":0.004,"t_end":0.005,"weight":1.0}]}}],
  "fixed_coupons":[]}]})";
double inf(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).cwiseAbs().maxCoeff(); }
}  // namespace

// D1: a band edit via rebind AFTER start_streaming is what the next tick prices.
TEST(ApiState, RebindAfterStartStreamingIsSeenByTheNextTick) {
  const cal::BundleProblem wide = banded_bundle(5.0, 2.0, 0.1), narrow = banded_bundle(0.5, 2.0, 0.1);
  const Eigen::VectorXd q = wide.market();
  api::BundleSession s(wide);
  s.calibrate(x0);
  s.start_streaming();
  s.stream_update(q);
  const Eigen::VectorXd x_wide = s.x();
  s.rebind(narrow);  // the band edit
  const Eigen::VectorXd x_rebind = s.x();
  const Eigen::VectorXd x_tick = s.stream_update(q);  // next tick, same market
  api::BundleSession fresh(narrow);
  fresh.calibrate(x0);
  fresh.start_streaming();
  const Eigen::VectorXd x_fresh = fresh.stream_update(q);
  std::cout << "  [state] rebind->tick vs fresh(narrow) " << inf(x_tick, x_fresh) << ", vs the wide solution " << inf(x_tick, x_wide) << "\n";
  // 1e-6 (0.01 bp): both are streamed kink optima of the same problem, reached from different seeds (the
  // wide-band solution vs an LM that stalls short at the kinks); they agree to ~1e-7, the multiplier
  // hysteresis of the active set. The stale-state discrepancy this test guards was 4e-4.
  EXPECT_LT(inf(x_tick, x_fresh), 1e-6) << "the streamer must price the rebound bands";
  EXPECT_LT(inf(x_rebind, x_tick), 1e-6);
  EXPECT_GT(inf(x_tick, x_wide), 1e-5) << "sanity: the narrow bands bind (the two solutions differ)";
}

// D2: a fixing that arrives AFTER start_streaming is priced by the next tick (the engine + streamer recompile).
TEST(ApiState, FixingsAfterStartStreamingAreSeenByTheNextTick) {
  const double r = 0.043;
  api::BundleSession s(api::bundle_from_json(js::parse(kBundle)));
  s.set_evaluation_date(101);
  s.set_fixings("USD-SOFR", {{97, r}, {98, r}, {99, r}, {100, r}});
  s.calibrate(api::flat_x0(s.problem()));
  s.start_streaming();
  const Eigen::VectorXd q = s.problem().market();
  const Eigen::VectorXd x_before = s.stream_update(q);
  s.set_fixings("USD-SOFR", {{100, r + 0.02}});  // yesterday's fixing corrected, 200 bp off
  EXPECT_TRUE(s.streaming());                     // armed across the recompile
  const Eigen::VectorXd x_tick = s.stream_update(q);
  EXPECT_TRUE(s.last_converged()) << s.last_reason();
  const cal::Instrument& ins = s.problem().instruments[0];
  EXPECT_NEAR(ins.obs.realized, 3 * r * 0.0027778 + (r + 0.02) * 0.0027778, 1e-12);
  EXPECT_LT(std::abs(s.model_quote(ins) - ins.market), 1e-9) << "the tick must reprice the NEW realized part (step_tol contract)";
  EXPECT_GT(inf(x_tick, x_before), 1e-6) << "the fixing moved the curve";
  // and a later tick does not overwrite it with a stale answer
  const Eigen::VectorXd x_tick2 = s.stream_update(q);
  EXPECT_LT(inf(x_tick2, x_tick), 1e-9);  // a converged tick at the same market stays put to the step_tol contract
}

// D3: the bound book reprices with a fixing that arrives after bind_portfolio (== the one-shot price).
TEST(ApiState, BoundBookRepricesWithNewFixings) {
  const double r = 0.043;
  api::BundleSession s(api::bundle_from_json(js::parse(kBundle)));
  s.set_evaluation_date(101);
  s.set_fixings("USD-SOFR", {{97, r}, {98, r}, {99, r}, {100, r}});
  s.calibrate(api::flat_x0(s.problem()));
  const auto book = api::book_from_json(js::parse(kBook));
  s.bind_portfolio(book);
  const double npv1 = s.reprice_bound().npv;
  s.set_fixings("USD-SOFR", {{100, r + 0.02}});
  const double npv2 = s.reprice_bound().npv, oneshot = s.price_portfolio(book).npv;
  std::cout << "  [state] bound npv before " << npv1 << " after " << npv2 << " one-shot " << oneshot << "\n";
  EXPECT_NEAR(npv2, oneshot, 1e-12 * std::max(1.0, std::abs(oneshot)));
  EXPECT_GT(std::abs(npv2 - npv1), 1e-9) << "the fixing changed the seasoned coupon";
}

// D5: on a streaming session a rebind/recalibrate IS a tick -- no LM, a handful of Newton steps, and (after
// the first, which starts streaming) no allocation; the answer equals a cold solve of the new market.
TEST(ApiState, RebindOnAStreamingSessionIsATick) {
  const cal::BundleProblem p = banded_bundle(0.0, 0.0, 1.0);  // hard quotes: the cold solve is exact
  api::BundleSession s(p);
  s.calibrate(x0);
  cal::BundleProblem p2 = p;
  for (auto& ins : p2.instruments) ins.market += 1e-4;
  s.rebind(p2);  // starts streaming (one Jacobian) + tick
  EXPECT_TRUE(s.streaming());
  EXPECT_TRUE(s.result().converged);
  EXPECT_NE(std::string(s.result().status).find("streamed"), std::string::npos) << s.result().status;
  EXPECT_LE(s.last_newton_steps(), 8);
  EXPECT_LT(inf(s.x(), cal::calibrate(p2, x0).x), 1e-10);
  cal::BundleProblem p3 = p;
  for (auto& ins : p3.instruments) ins.market -= 1e-4;
  unsigned long allocs = 0;
  {
    swaps::testing::AllocScope scope;
    s.rebind(p3);
    allocs = scope.allocs();
  }
  std::cout << "  [state] second rebind: " << s.last_newton_steps() << " steps, " << allocs << " allocs\n";
  EXPECT_LT(inf(s.x(), cal::calibrate(p3, x0).x), 1e-10);
  if (swaps::testing::alloc_counting_available()) EXPECT_LE(allocs, 8u) << "rebind must be a tick, not a copy + LM";
  // recalibrate rides the same path
  Eigen::VectorXd q4 = p.market().array() + 2e-4;
  s.recalibrate(q4);
  EXPECT_NE(std::string(s.result().status).find("streamed"), std::string::npos);
  cal::BundleProblem p4 = p;
  for (int i = 0; i < 10; ++i) p4.instruments[i].market = q4[i];
  EXPECT_LT(inf(s.x(), cal::calibrate(p4, x0).x), 1e-10);
}

// result() reports the LAST solve on EVERY entry point. Until 2026-09-21 a converged stream_update moved x() but
// left result() at the previous LM solve (converged == true, the old curve): a streaming client reading result().x
// priced a curve hundreds of ticks stale. Found by a soak probe that did exactly that.
TEST(ApiState, StreamUpdateReportsItselfInResultLikeResolveDoes) {
  const cal::BundleProblem p = banded_bundle(0.0, 0.0, 1.0);
  api::BundleSession s(p);
  s.calibrate(x0);
  const Eigen::VectorXd x_cold = s.result().x;
  s.start_streaming();
  Eigen::VectorXd q = p.market();
  for (int k = 1; k <= 5; ++k) {  // several converged ticks, none of which runs the LM
    q.array() += 1e-4;
    const Eigen::VectorXd& x = s.stream_update(q);
    ASSERT_TRUE(s.last_converged()) << "tick " << k << ": " << s.last_reason();
    EXPECT_EQ(inf(s.result().x, x), 0.0) << "tick " << k << ": result().x must be the curve stream_update returned";
    EXPECT_EQ(inf(s.result().x, s.x()), 0.0);
  }
  EXPECT_GT(inf(s.result().x, x_cold), 1e-6) << "the market moved 5 bp; a stale report would still show the cold solve";
  EXPECT_TRUE(s.result().converged);
  EXPECT_NE(std::string(s.result().status).find("streamed"), std::string::npos) << s.result().status;
  EXPECT_EQ(s.result().iterations, s.last_newton_steps());
  // The streamed rms is the streamer's final corrector residual (the state one sub-step_tol step before x()),
  // NOT a re-evaluation at x() -- that would put a residual pass on the gated tick. It is a report at the
  // O(||J||·step_tol) level: measured 4.3e-8 here against 2.7e-9 re-evaluated (predictive convergence stops the
  // corrector one step early), both ~1e-3 bp and below anything a client acts on. Pinned at that level.
  cal::BundleProblem pq = p;
  for (int i = 0; i < pq.n_residuals(); ++i) pq.instruments[i].market = q[i];
  const Eigen::VectorXd r = pq.residuals<double>(s.x());
  const double rms_exact = std::sqrt(r.squaredNorm() / r.size());
  std::cout << "  [state] streamed rms " << s.result().rms_residual << " vs exact " << rms_exact << "\n";
  EXPECT_LT(s.result().rms_residual, 1e-6);
  EXPECT_GE(s.result().rms_residual, rms_exact) << "the corrector's residual can only be larger than the committed one";
  // resolve() on the same market re-evaluates (its callers read rms to 1e-10): same curve, the exact number.
  s.resolve();
  EXPECT_EQ(inf(s.result().x, s.x()), 0.0);
  EXPECT_LT(s.result().rms_residual, 1e-8);
}

// set_band + resolve (the scalar API) == rebind with the same band == a fresh session on that problem.
TEST(ApiState, ScalarBandUpdateMatchesRebindAndAFreshSession) {
  const cal::BundleProblem p = banded_bundle(0.0, 2.0, 1.0);  // inconsistent hard quotes, no bands yet
  cal::BundleProblem pb = p;
  auto& b = pb.instruments[3];
  b.band_lower = b.market - 0.5e-4; b.band_upper = b.market + 0.5e-4; b.band_decay = 0.1;
  api::BundleSession s1(p), s2(p), fresh(pb);
  s1.calibrate(x0); s2.calibrate(x0); fresh.calibrate(x0);
  s1.set_band(3, b.band_lower, b.band_upper, b.band_decay);
  s1.resolve();
  s2.rebind(pb);
  fresh.start_streaming();
  fresh.stream_update(pb.market());
  EXPECT_LT(inf(s1.x(), s2.x()), 1e-10);
  EXPECT_LT(inf(s1.x(), fresh.x()), 1e-6);  // two streamed kink optima from different seeds (see above)
  EXPECT_TRUE(s1.has_band());
  EXPECT_TRUE(s1.same_structure(pb));
}

// D16 + the structural contract: same_structure is an O(n) equality that ignores quotes/bands and the
// session's own fixings resolution; every structural edit is rejected by rebind (compile a new session).
TEST(ApiState, StructuralEditsAreRejectedAndSameStructureIgnoresResolution) {
  api::BundleSession s(api::bundle_from_json(js::parse(kBundle)));
  s.set_evaluation_date(101);
  s.set_fixings("USD-SOFR", {{97, 0.043}, {98, 0.043}, {99, 0.043}});
  EXPECT_TRUE(s.same_structure(s.problem()));                             // was FALSE from construction (D16)
  EXPECT_TRUE(s.same_structure(api::bundle_from_json(js::parse(kBundle))));  // the client's UNRESOLVED copy
  const cal::BundleProblem p = banded_bundle(0.0, 0.0, 1.0);
  api::BundleSession t(p);
  t.calibrate(x0);
  cal::BundleProblem requoted = p;
  for (auto& ins : requoted.instruments) { ins.market += 3e-4; ins.band_lower = ins.market - 1e-4; ins.band_upper = ins.market + 1e-4; ins.band_decay = 0.5; }
  EXPECT_TRUE(t.same_structure(requoted));
  cal::BundleProblem moved = p;
  moved.curves[0].regions.back().knots.back() += 0.25;
  EXPECT_FALSE(t.same_structure(moved));
  EXPECT_THROW(t.rebind(moved), std::runtime_error);
  cal::BundleProblem scheme = p;
  scheme.curves[0].regions.back().scheme = cv::Scheme::Linear;
  EXPECT_FALSE(t.same_structure(scheme));
  EXPECT_THROW(t.rebind(scheme), std::runtime_error);
  cal::BundleProblem bigger = p;
  bigger.instruments.push_back(p.instruments.back());
  EXPECT_THROW(t.rebind(bigger), std::runtime_error);
  cal::BundleProblem tenor = p;
  tenor.instruments[4] = par_swap(6.0);  // same count, a different instrument on the same row
  tenor.instruments[4].market = p.instruments[4].market;
  EXPECT_FALSE(t.same_structure(tenor));
  EXPECT_THROW(t.rebind(tenor), std::runtime_error);
  // the session after the rejections is intact
  EXPECT_NO_THROW(t.rebind(requoted));
}
